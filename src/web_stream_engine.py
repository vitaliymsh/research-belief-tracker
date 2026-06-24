import os
import ctypes
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import threading

# ── NvBufUtils zero-copy frame access ────────────────────────────────────────
# On Jetson, NVMM buffers (GPU DRAM) can be CPU-mapped via NvBufferMemMap.
# This gives a virtual address into the same physical DRAM the GPU uses,
# eliminating the VIC GPU→System-RAM DMA that nvvidconv(system) performs.
# API: libnvbufsurface.so (part of JetPack)

_NVBUF_LIB_PATH = "/usr/lib/aarch64-linux-gnu/tegra/libnvbuf_utils.so"
_NvBufferMem_Read = 0   # NvBufferMemFlags::NvBufferMem_Read

_nvbuf_lib = None
_nvbuf_available = False

def _init_nvbuf():
    global _nvbuf_lib, _nvbuf_available
    try:
        lib = ctypes.CDLL(_NVBUF_LIB_PATH)
        # int ExtractFdFromNvBuffer(void *nvbuf, int *dmabuf_fd)
        lib.ExtractFdFromNvBuffer.restype  = ctypes.c_int
        lib.ExtractFdFromNvBuffer.argtypes = [ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_int)]
        # int NvBufferMemMap(int dmabuf_fd, unsigned int plane,
        #                    NvBufferMemFlags flag, void **pVirtAddr)
        lib.NvBufferMemMap.restype  = ctypes.c_int
        lib.NvBufferMemMap.argtypes = [ctypes.c_int, ctypes.c_uint,
                                        ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_void_p)]
        # int NvBufferMemSyncForCpu(int dmabuf_fd, unsigned int plane,
        #                           void **pVirtAddr)
        lib.NvBufferMemSyncForCpu.restype  = ctypes.c_int
        lib.NvBufferMemSyncForCpu.argtypes = [ctypes.c_int, ctypes.c_uint,
                                               ctypes.POINTER(ctypes.c_void_p)]
        # int NvBufferMemUnMap(int dmabuf_fd, unsigned int plane,
        #                      void **pVirtAddr)
        lib.NvBufferMemUnMap.restype  = ctypes.c_int
        lib.NvBufferMemUnMap.argtypes = [ctypes.c_int, ctypes.c_uint,
                                          ctypes.POINTER(ctypes.c_void_p)]
        _nvbuf_lib = lib
        _nvbuf_available = True
        print("[WebStreamEngine] NvBufUtils loaded — NVMM zero-copy path enabled.")
    except Exception as e:
        print(f"[WebStreamEngine] NvBufUtils unavailable ({e}); "
              "falling back to system-memory path.")

_init_nvbuf()

# Frame geometry constants (must match GStreamer caps below)
_FRAME_W   = 640
_FRAME_H   = 640
_FRAME_CHN = 4  # RGBA
_FRAME_BYTES = _FRAME_W * _FRAME_H * _FRAME_CHN


class WebStreamEngine:
    """Low-latency web streaming pipeline with zero-copy NVMM frame path."""

    def __init__(self, callback=None):
        self.callback = callback
        self.pipeline = None
        self.loop = GObject.MainLoop()
        self.running = False
        self.hw_fps = 30.0
        self.frame_count = 0

        self.inference_enabled = False
        self.inference_busy = False

        # Pre-allocated ctypes objects reused every callback to avoid GC pressure
        self._dmabuf_fd  = ctypes.c_int(0)
        self._virt_addr  = ctypes.c_void_p(0)

        Gst.init(None)

    def _bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("WebStream Pipeline: End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"WebStream GStreamer Error: {err}")
            if debug:
                print(f"Debug info: {debug}")
            loop.quit()
        return True

    def _infer_block_probe(self, pad, info, u_data):
        if not self.inference_enabled or self.inference_busy:
            return Gst.PadProbeReturn.DROP
        self.inference_busy = True
        return Gst.PadProbeReturn.OK

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_pipeline)
        self.thread.daemon = True
        self.thread.start()

    def _run_pipeline(self):
        self.pipeline = Gst.Pipeline()

        # Elements
        source = Gst.ElementFactory.make("nvarguscamerasrc", "src")
        source.set_property('do-timestamp', True)
        source.set_property('bufapi-version', True)
        source.set_property('sensor-mode', 4)  # 60 FPS Mode

        # OPTIMIZATION: Disable ISP post-processing to save power and cycles
        source.set_property('tnr-mode', 0)
        source.set_property('ee-mode', 0)

        caps_src = Gst.ElementFactory.make("capsfilter", "caps_src")
        caps_src.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), width=1280, height=720, framerate=60/1"))

        # Leaky queue drops frames to throttle 60fps → ~30fps input
        queue_throttle = Gst.ElementFactory.make("queue", "queue_throttle")
        queue_throttle.set_property("leaky", 2)
        queue_throttle.set_property("max-size-buffers", 1)

        muxer = Gst.ElementFactory.make("nvstreammux", "muxer")
        muxer.set_property('width', 1280)
        muxer.set_property('height', 720)
        muxer.set_property('batch-size', 1)
        muxer.set_property('batched-push-timeout', 4000)
        muxer.set_property('live-source', True)
        muxer.set_property('nvbuf-memory-type', 4)
        muxer.set_property('enable-padding', False)
        muxer.set_property('interpolation-method', 0)
        muxer.set_property('num-surfaces-per-frame', 1)

        # Inference branch
        queue_infer = Gst.ElementFactory.make("queue", "queue_infer")
        queue_infer.set_property("leaky", 2)
        queue_infer.set_property("max-size-buffers", 1)

        # nvvidconv: scale to 640×640 RGBA.
        # ── ZERO-COPY PATH (preferred): keep frame in NVMM (GPU DRAM).
        #    The callback uses NvBufferMemMap to get a CPU virtual address
        #    pointing to the same physical DRAM — no VIC GPU→CPU DMA.
        # ── FALLBACK PATH: output to system memory if NvBufUtils unavailable.
        nvvidconv_infer = Gst.ElementFactory.make("nvvidconv", "nvvidconv_infer")

        if _nvbuf_available:
            # Stay in NVMM — CPU will map via NvBufferMemMap (zero VIC DMA)
            caps_out = Gst.Caps.from_string(
                "video/x-raw(memory:NVMM), format=RGBA, "
                f"width={_FRAME_W}, height={_FRAME_H}")
        else:
            # Fallback: system RGBA (original path)
            caps_out = Gst.Caps.from_string(
                f"video/x-raw, format=RGBA, width={_FRAME_W}, height={_FRAME_H}")

        caps_rgba = Gst.ElementFactory.make("capsfilter", "caps_rgba")
        caps_rgba.set_property("caps", caps_out)

        appsink_infer = Gst.ElementFactory.make("appsink", "appsink_infer")
        appsink_infer.set_property("emit-signals", True)
        appsink_infer.set_property("drop", True)
        appsink_infer.set_property("max-buffers", 1)
        appsink_infer.set_property("sync", False)
        appsink_infer.set_property("async", False)
        appsink_infer.connect("new-sample", self._on_new_sample_infer)

        # Add to pipeline
        for el in [source, caps_src, queue_throttle, muxer,
                   queue_infer, nvvidconv_infer, caps_rgba, appsink_infer]:
            self.pipeline.add(el)

        # Link
        source.link(caps_src)
        caps_src.link(queue_throttle)
        sinkpad = muxer.get_request_pad("sink_0")
        srcpad  = queue_throttle.get_static_pad("src")
        srcpad.link(sinkpad)
        muxer.link(queue_infer)

        queue_infer.get_static_pad("sink").add_probe(
            Gst.PadProbeType.BUFFER, self._infer_block_probe, 0)
        queue_infer.link(nvvidconv_infer)
        nvvidconv_infer.link(caps_rgba)
        caps_rgba.link(appsink_infer)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call, self.loop)

        path_name = "NVMM zero-copy" if _nvbuf_available else "system-memory"
        print(f"WebStream Pipeline: Starting ({path_name} path)...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        print(f"WebStream Pipeline State Change Return: {ret}")
        try:
            self.loop.run()
        except Exception:
            pass
        self.pipeline.set_state(Gst.State.NULL)

    # ── Frame delivery ────────────────────────────────────────────────────────

    def _on_new_sample_infer(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        self.frame_count += 1

        try:
            if _nvbuf_available:
                self._deliver_nvmm(buf)
            else:
                self._deliver_system(buf)
        except Exception as e:
            print(f"Error in _on_new_sample_infer: {e}")

        return Gst.FlowReturn.OK

    def _deliver_nvmm(self, buf):
        """Zero-copy NVMM path: map GPU buffer to CPU virtual addr, pass raw ptr.

        Avoids the VIC GPU→system-RAM DMA that the system-memory path requires.
        On Jetson unified DRAM, NvBufferMemMap returns a CPU virtual address
        that points directly into GPU physical memory — no data movement.
        """
        mem = buf.peek_memory(0)
        # Get the underlying nv_buffer pointer from GstMemory.
        # GstNvmmMem stores nv_buffer as the first field after the GstMemory header.
        # We use GstMemory.data (via map) to reach the raw nv_buffer handle.
        success, map_info = mem.map(Gst.MapFlags.READ)
        if not success:
            return

        try:
            # For NVMM memory, map_info.data IS the NvBuffer virtual address
            # when the GStreamer element wrote it via NvBufSurface.
            # However, the portable approach: use nv_buffer embedded in GstMemory.
            # GstNvmmMemoryType stores nv_buffer at offset after GstMemory base.
            # We extract it via ExtractFdFromNvBuffer using the raw pointer.
            nv_buf_ptr = ctypes.cast(map_info.data, ctypes.c_void_p).value

            ret = _nvbuf_lib.ExtractFdFromNvBuffer(
                ctypes.c_void_p(nv_buf_ptr),
                ctypes.byref(self._dmabuf_fd)
            )
            if ret != 0:
                # ExtractFd failed — fall back to treating map_info.data as frame
                if self.callback:
                    self.callback(map_info.data, None, self.frame_count,
                                  True, self.hw_fps)
                return

            fd = self._dmabuf_fd.value

            # Map NVMM plane 0 to CPU virtual address (zero-copy on unified DRAM)
            ret = _nvbuf_lib.NvBufferMemMap(
                fd, 0, _NvBufferMem_Read, ctypes.byref(self._virt_addr))
            if ret != 0:
                if self.callback:
                    self.callback(map_info.data, None, self.frame_count,
                                  True, self.hw_fps)
                return

            # Sync NVMM cache → CPU visibility (required before CPU read)
            _nvbuf_lib.NvBufferMemSyncForCpu(
                fd, 0, ctypes.byref(self._virt_addr))

            # Deliver the raw CPU virtual address (points into GPU DRAM)
            if self.callback:
                self.callback(self._virt_addr.value, None, self.frame_count,
                              True, self.hw_fps, nvmm=True)

            # Unmap
            _nvbuf_lib.NvBufferMemUnMap(
                fd, 0, ctypes.byref(self._virt_addr))

        finally:
            mem.unmap(map_info)

    def _deliver_system(self, buf):
        """Fallback system-memory path (original behaviour)."""
        mem = buf.peek_memory(0)
        success, map_info = mem.map(Gst.MapFlags.READ)
        if success:
            try:
                if self.callback:
                    self.callback(map_info.data, None, self.frame_count,
                                  True, self.hw_fps)
            finally:
                mem.unmap(map_info)

    def set_inference_enabled(self, enabled: bool):
        self.inference_enabled = enabled
        print(f"WebStreamEngine: Set inference enabled to {enabled}")

    def release(self):
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        self.loop.quit()
