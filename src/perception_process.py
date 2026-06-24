"""
perception_process.py — Isolated CUDA perception subprocess.

Runs YOLO + GPU-crop + OSNet in a fully separate Python process so that
TensorRT's Python binding GIL holds (~47ms YOLO + ~75ms OSNet) are
completely invisible to the main process UI, tracking, and GStreamer threads.

Communication protocol (all via multiprocessing primitives):
  Main → Subprocess  : cmd_conn.send((frame_idx, capture_time))
  Subprocess → Main  : result_conn.send(result_dict)

Frame data is exchanged through a shared mp.Array (POSIX shared memory),
which means zero IPC copy for the 1.6 MB frame buffer.
"""

import os
import time
import traceback
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


# ── CUDA Kernel: RGBA → YOLO input (NCHW float, RGB normalised) ──────────────

cuda_preprocess_src = """
__global__ void preprocess_yolo_kernel(
    const uchar4* input_frame,
    float* output_tensor,
    int total_pixels
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total_pixels) return;
    uchar4 pixel = input_frame[idx];
    int cs = total_pixels;
    output_tensor[0*cs+idx] = (float)pixel.x / 255.0f;
    output_tensor[1*cs+idx] = (float)pixel.y / 255.0f;
    output_tensor[2*cs+idx] = (float)pixel.z / 255.0f;
}
"""

# ── CUDA Kernel: GPU crop + resize + normalize for ReID ──────────────────────

cuda_crop_src = """
__global__ void crop_and_normalize_kernel(
    const unsigned char* input_frame,
    const float* bboxes,
    float* output_tensor,
    int img_w, int img_h
) {
    int w_out = threadIdx.x + blockIdx.x * blockDim.x;
    int h_out = threadIdx.y + blockIdx.y * blockDim.y;
    int n_out = blockIdx.z;
    if (w_out >= 128 || h_out >= 256 || n_out >= 4) return;

    float x1 = bboxes[n_out*4+0];
    float y1 = bboxes[n_out*4+1];
    float w  = bboxes[n_out*4+2];
    float h  = bboxes[n_out*4+3];

    float src_x = x1 + (w_out / 127.0f) * w;
    float src_y = y1 + (h_out / 255.0f) * h;

    int ix = max(0, min((int)roundf(src_x), 639));
    int iy = max(0, min((int)roundf(src_y), 639));
    int in_idx = (iy * 640 + ix) * 4;

    float r = (float)input_frame[in_idx+0] / 255.0f;
    float g = (float)input_frame[in_idx+1] / 255.0f;
    float b = (float)input_frame[in_idx+2] / 255.0f;

    float r_n = (r - 0.485f) / 0.229f;
    float g_n = (g - 0.456f) / 0.224f;
    float b_n = (b - 0.406f) / 0.225f;

    int cs = 256 * 128;
    int bs = 3 * cs;
    int pi = h_out * 128 + w_out;
    output_tensor[n_out*bs + 0*cs + pi] = r_n;
    output_tensor[n_out*bs + 1*cs + pi] = g_n;
    output_tensor[n_out*bs + 2*cs + pi] = b_n;
}
"""


# ── TRTEngine ─────────────────────────────────────────────────────────────────

class TRTEngine:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.allocations = [], [], []
        for i in range(self.engine.num_bindings):
            is_in  = self.engine.binding_is_input(i)
            dtype  = trt.nptype(self.engine.get_binding_dtype(i))
            shape  = self.engine.get_binding_shape(i)
            size   = int(np.prod(shape))
            alloc  = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            b = {'index': i, 'dtype': dtype, 'shape': shape,
                 'allocation': alloc, 'size': size}
            self.allocations.append(int(alloc))
            (self.inputs if is_in else self.outputs).append(b)

    def run_async(self, in_bufs, out_bufs, stream):
        bindings = [None] * self.engine.num_bindings
        for b in self.inputs:
            bindings[b['index']] = int(in_bufs[b['index']])
        for b in self.outputs:
            bindings[b['index']] = int(out_bufs[b['index'] - len(self.inputs)])
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)


# ── Subprocess entry point ────────────────────────────────────────────────────

def run_perception(frame_shm, cmd_conn, result_conn, stop_event,
                   yolo_path, osnet_path):
    """
    Perception subprocess entry point. Called by multiprocessing.Process.

    Args:
        frame_shm:   mp.Array('B', 640*640*4) — shared frame buffer (written by main)
        cmd_conn:    Pipe recv-end — receives (frame_idx, capture_time) per frame
        result_conn: Pipe send-end — sends result dicts back to main
        stop_event:  mp.Event — set by main to trigger clean shutdown
        yolo_path:   absolute path to YOLO TRT engine file
        osnet_path:  absolute path to OSNet TRT engine file
    """
    # Ensure nvcc is on PATH for PyCUDA kernel compilation
    os.environ['PATH'] = os.environ.get('PATH', '') + ':/usr/local/cuda/bin'

    _log("[PercProc] Initialising CUDA context...")
    cuda.init()
    ctx = cuda.Device(0).retain_primary_context()
    ctx.push()

    try:
        _log("[PercProc] Compiling CUDA kernels...")
        pre_kern = SourceModule(cuda_preprocess_src).get_function("preprocess_yolo_kernel")
        crp_kern = SourceModule(cuda_crop_src).get_function("crop_and_normalize_kernel")

        _log("[PercProc] Loading TRT engines...")
        yolo_engine  = TRTEngine(yolo_path)
        osnet_engine = TRTEngine(osnet_path)
        _log("[PercProc] Engines ready.")

        # ── GPU buffers ───────────────────────────────────────────────────────
        img_h = img_w = 640
        frame_gpu  = cuda.mem_alloc(img_h * img_w * 4)
        bboxes_gpu = cuda.mem_alloc(4 * 4 * 4)   # 4 boxes × 4 floats × 4 bytes

        # Pinned host buffers (page-locked for fast async DMA)
        yolo_out_host  = cuda.pagelocked_empty(
            tuple(yolo_engine.outputs[0]['shape']),  dtype=np.float32)
        osnet_out_host = cuda.pagelocked_empty(
            tuple(osnet_engine.outputs[0]['shape']), dtype=np.float32)
        bboxes_cpu     = cuda.pagelocked_empty((4, 4), dtype=np.float32)

        # Zero-copy view of the shared memory frame buffer.
        # Used directly for HtoD — eliminates the intermediate 1.6 MB pinned copy.
        frame_np = np.frombuffer(frame_shm, dtype=np.uint8)

        # ── Try to register shared memory as CUDA pinned memory ───────────────
        # On Jetson unified DRAM, cuda.register_host_memory makes the POSIX shm
        # page-locked, turning cuda.memcpy_htod_async into a true DMA transfer.
        # This avoids the CUDA staging-buffer bounce and reduces DRAM bus stalls
        # that slow YOLO when the HtoD and inference compete for bandwidth.
        _shm_pinned = False
        try:
            cuda.register_host_memory(
                frame_np,
                cuda.mem_host_register_flags.PORTABLE |
                cuda.mem_host_register_flags.DEVICE_MAP
            )
            _shm_pinned = True
            _log("[PercProc] Shared memory registered as CUDA pinned (optimal DMA).")
        except Exception as _pin_err:
            _log(f"[PercProc] Note: shared memory pinning skipped ({_pin_err}). "
                 "Using standard HtoD — OK on unified DRAM.")

        stream = cuda.Stream()
        last_perc_time = time.time()

        # Signal main process that initialisation is complete
        result_conn.send({'type': 'ready'})
        _log("[PercProc] Waiting for frames...")


        while not stop_event.is_set():
            # Block until main process sends (frame_idx, capture_time)
            if not cmd_conn.poll(timeout=0.1):
                continue

            msg = cmd_conn.recv()
            if msg is None:   # explicit shutdown signal
                break

            # Drain any backlogged commands so we always process the freshest frame.
            # Main now releases inference_busy immediately after sending, so commands
            # can pile up at ~30fps while we run at ~5fps.
            while cmd_conn.poll():
                drained = cmd_conn.recv()
                if drained is None:
                    msg = None
                    break
                msg = drained
            if msg is None:
                break

            frame_idx, capture_time = msg

            try:
                t0 = time.perf_counter()

                # ── A. Shared-memory frame → GPU (direct, no intermediate copy) ──
                # frame_np is a view of mp.Array (POSIX shm). On Jetson unified
                # memory, htod_async works directly — saves one 1.6 MB CPU copy.
                cuda.memcpy_htod_async(frame_gpu, frame_np, stream)

                # ── B. Preprocess (RGBA → NCHW float) ────────────────────────
                total_px = img_h * img_w
                pre_kern(
                    frame_gpu, yolo_engine.inputs[0]['allocation'],
                    np.int32(total_px),
                    block=(256, 1, 1),
                    grid=((total_px + 255) // 256, 1, 1),
                    stream=stream
                )

                # ── C. YOLO ───────────────────────────────────────────────────
                yolo_engine.run_async(
                    [yolo_engine.inputs[0]['allocation']],
                    [yolo_engine.outputs[0]['allocation']],
                    stream
                )
                cuda.memcpy_dtoh_async(
                    yolo_out_host, yolo_engine.outputs[0]['allocation'], stream)
                stream.synchronize()

                yolo_ms = (time.perf_counter() - t0) * 1000

                # ── D. NMS (vectorised) ───────────────────────────────────────
                preds      = yolo_out_host[0]
                conf_mask  = preds[:, 4] > 0.25
                class_mask = preds[:, 5].astype(np.int32) == 0
                valid      = preds[conf_mask & class_mask]
                if len(valid):
                    valid = valid[np.argsort(-valid[:, 4])[:4]]
                num_det = len(valid)

                if num_det > 0:
                    x1r  = valid[:, 0];  y1r = valid[:, 1]
                    x2r  = valid[:, 2];  y2r = valid[:, 3]
                    conf = valid[:, 4]

                    # ── E. GPU crop + OSNet ───────────────────────────────────
                    bboxes_cpu.fill(0)
                    bboxes_cpu[:num_det, 0] = x1r
                    bboxes_cpu[:num_det, 1] = y1r
                    bboxes_cpu[:num_det, 2] = x2r - x1r
                    bboxes_cpu[:num_det, 3] = y2r - y1r
                    if num_det < 4:
                        bboxes_cpu[num_det:] = [0.0, 0.0, 1.0, 1.0]

                    t_reid = time.perf_counter()
                    cuda.memcpy_htod_async(bboxes_gpu, bboxes_cpu, stream)
                    crp_kern(
                        frame_gpu, bboxes_gpu, osnet_engine.inputs[0]['allocation'],
                        np.int32(img_w), np.int32(img_h),
                        block=(16, 16, 1), grid=(8, 16, 4), stream=stream
                    )
                    osnet_engine.run_async(
                        [osnet_engine.inputs[0]['allocation']],
                        [osnet_engine.outputs[0]['allocation']],
                        stream
                    )
                    cuda.memcpy_dtoh_async(
                        osnet_out_host, osnet_engine.outputs[0]['allocation'], stream)
                    stream.synchronize()
                    reid_ms = (time.perf_counter() - t_reid) * 1000

                    # ── F. Scale to 1280×720 + L2-normalise features ──────────
                    x1 = x1r * (1280.0 / 640.0);  x2 = x2r * (1280.0 / 640.0)
                    y1 = y1r * (720.0  / 640.0);  y2 = y2r * (720.0  / 640.0)
                    w_s = x2 - x1;  h_s = y2 - y1

                    feats      = osnet_out_host[:num_det]
                    norms      = np.linalg.norm(feats, axis=1, keepdims=True)
                    feats_norm = feats / (norms + 1e-6)

                    dets_arr = np.hstack([
                        x1[:, np.newaxis],  y1[:, np.newaxis],
                        w_s[:, np.newaxis], h_s[:, np.newaxis],
                        conf[:, np.newaxis], feats_norm
                    ]).astype(np.float32)

                else:
                    reid_ms = 0.0
                    dets_arr = np.empty((0, 5 + 512), dtype=np.float32)

                # ── G. Update per-cycle telemetry ─────────────────────────────
                now_p = time.time()
                dp    = now_p - last_perc_time
                perc_fps = (1.0 / dp) if dp > 0 else 0.0
                last_perc_time = now_p

                # ── H. Return results to main process ─────────────────────────
                # ui_predictions removed from IPC dict — main process reconstructs
                # them from dets_arr, reducing pickle payload size.
                result_conn.send({
                    'type':         'dets',
                    'dets_arr':     dets_arr,
                    'capture_time': capture_time,
                    'frame_idx':    frame_idx,
                    'yolo_ms':      yolo_ms,
                    'reid_ms':      reid_ms,
                    'perc_fps':     perc_fps,
                })

            except Exception as exc:
                _log(f"[PercProc] ERROR frame {frame_idx}: {exc}")
                traceback.print_exc()
                # Still unblock the main process so GStreamer can advance
                result_conn.send({
                    'type':      'error',
                    'frame_idx': frame_idx,
                    'error':     str(exc),
                })

    except Exception as fatal:
        _log(f"[PercProc] Fatal: {fatal}")
        traceback.print_exc()
    finally:
        _log("[PercProc] Shutting down.")
        try:
            if _shm_pinned:
                cuda.unregister_host_memory(frame_np)
        except Exception:
            pass
        ctx.pop()



def _log(msg):
    print(msg, flush=True)
