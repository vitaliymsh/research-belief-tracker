import os
import sys
import time
import ctypes
import threading
import queue
import traceback
import multiprocessing as mp
import cv2
import orjson
import numpy as np

from perception_process import run_perception

from telemetry import TelemetryTracker
from lib.belief_tracking.belief_tracker import BeliefTracker
from web_stream_engine import WebStreamEngine

import config
from config import get_logger
logger = get_logger("Pipeline")

# Configuration
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS
YOLO_ENGINE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolo26n.engine")
OSNET_ENGINE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "osnet_x0_25_b4_opt_fp16.engine")

def log(msg):
    logger.info(msg)


class TrackingPipeline:
    def __init__(self):
        # Component Instances
        self.ds_engine = WebStreamEngine(callback=self._on_ds_frame)
        self.camera = self.ds_engine # Alias for app.py compatibility
        self.tracker = BeliefTracker(camera_fps=30.0)
        
        # Subprocess IPC — shared memory frame buffer (POSIX shm, no IPC copy)
        self._frame_shm = mp.Array(ctypes.c_uint8, 640 * 640 * 4, lock=False)
        self._frame_np  = np.frombuffer(self._frame_shm, dtype=np.uint8)  # zero-copy view

        # Pipes: cmd = main→subprocess, result = subprocess→main
        self._cmd_send,    self._cmd_recv    = mp.Pipe()
        self._result_send, self._result_recv = mp.Pipe()

        # Subprocess lifecycle
        self._perc_stop = mp.Event()
        self._perc_proc = None

        # Communication Buffers & Queues
        self.enriched_detection_queue = queue.Queue(maxsize=10)

        # Tracker performance metrics
        self.tracker_update_ms = 0.0
        self.tracker_predict_ms = 0.0
        self.tracker_latency_ms = 0.0

        # ctypes pointer to shared memory for zero-copy memmove
        self._frame_shm_base = ctypes.addressof(self._frame_shm)
        self._frame_bytes = 640 * 640 * 4


        # Payload cache: JSON is only re-serialized when predictions change, not every 30fps tick
        self._cached_payload_bytes = None

        # Telemetry & State
        self.status = "BOOTING..."
        self.fps_val = 0.0
        self.perception_fps = 0.0
        self.tracker_fps = 0.0
        self.predict_fps = 0.0
        self._yolo_time_ms = 0.0
        self._reid_time_ms = 0.0
        self.system_stats = {}
        self._cached_stats = {
            "cpu": 0, "temp": 0, "ram_used": 0, "ram_total": 0, "gpu": 0
        }
        self.tracking_enabled = True
        self.pipeline_enabled = False
        self.last_detections = []
        self.telemetry_tracker = TelemetryTracker()

        # WebSocket / Frame publishing hooks
        self.latest_payload = None
        self.on_payload_ready = None
        self.lock = threading.Lock()
        self.last_frame_idx = 0

    def start(self):
        """Starts the perception subprocess (before GStreamer) then all background threads."""
        # Fork perception subprocess BEFORE GStreamer initialises its CUDA context
        # so the child process gets a clean fork without any conflicting CUDA state.
        self._start_perception_subprocess()

        threading.Thread(target=self._stats_worker,              daemon=True).start()
        threading.Thread(target=self._detection_receiver_worker, daemon=True).start()
        threading.Thread(target=self._tracking_worker,           daemon=True).start()
        threading.Thread(target=self._ui_stream_worker,          daemon=True).start()

        self.ds_engine.start()   # GStreamer starts after subprocess is ready

    def _start_perception_subprocess(self):
        """Fork the perception subprocess and block until it signals 'ready'."""
        self._perc_proc = mp.Process(
            target=run_perception,
            args=(
                self._frame_shm,
                self._cmd_recv,
                self._result_send,
                self._perc_stop,
                YOLO_ENGINE_PATH,
                OSNET_ENGINE_PATH,
            ),
            daemon=True,
            name="PerceptionProcess",
        )
        self._perc_proc.start()
        log("Waiting for perception subprocess to load TRT engines (up to 30s)...")
        while True:
            if self._result_recv.poll(timeout=30.0):
                msg = self._result_recv.recv()
                if msg.get('type') == 'ready':
                    log("Perception subprocess ready.")
                    break
                log(f"Unexpected startup message from subprocess: {msg}")
            else:
                raise RuntimeError(
                    "Perception subprocess did not become ready within 30s. "
                    "Check engine paths and CUDA installation."
                )

    def _on_ds_frame(self, frame, detections, frame_idx, b_infer_done=True,
                     hw_fps=30.0, nvmm=False):

        """GStreamer callback — copies frame to shared memory and notifies the subprocess."""
        t0 = time.time()
        self.fps_val = hw_fps
        self.last_frame_idx = frame_idx

        if not self.pipeline_enabled:
            self.ds_engine.inference_busy = False
            return

        # ── Copy frame into shared memory ─────────────────────────────────────
        # NVMM path: `frame` is a raw int (CPU virtual addr from NvBufferMemMap).
        #   ctypes.memmove avoids numpy object creation overhead.
        # Fallback path: `frame` is a GStreamer memoryview — use np.frombuffer.
        if nvmm:
            ctypes.memmove(self._frame_shm_base, frame, self._frame_bytes)
        else:
            np.copyto(self._frame_np, np.frombuffer(frame, dtype=np.uint8))

        # Send frame metadata to subprocess (tiny send, no GIL-heavy serialisation)
        try:
            self._cmd_send.send((frame_idx, t0))
        except Exception as exc:
            log(f"[Pipeline] Failed to send frame cmd to subprocess: {exc}")
            self.ds_engine.inference_busy = False
            return

        # Release the GStreamer gate immediately after writing shm + sending cmd.
        # The subprocess drains stale pipe commands and always picks up the newest
        # frame, so there is no correctness problem with early release.
        # Previously the gate stayed closed for the full 197ms GPU round-trip,
        # costing ~30-60ms of dead time per cycle.
        self.ds_engine.inference_busy = False

        self.status = "GPU ACTIVE"
        return self.tracking_enabled

    def toggle_pipeline(self) -> bool:
        """Toggles the overall pipeline (inference + tracking) and returns the new state."""
        self.pipeline_enabled = not self.pipeline_enabled
        if hasattr(self, 'ds_engine') and hasattr(self.ds_engine, 'set_inference_enabled'):
            self.ds_engine.set_inference_enabled(self.pipeline_enabled)
        log(f"Overall pipeline toggled: {self.pipeline_enabled}")
        return self.pipeline_enabled

    def toggle_tracking(self) -> bool:
        """Toggles tracking and returns the new state."""
        self.tracking_enabled = not self.tracking_enabled
        log(f"Tracking toggled: {self.tracking_enabled}")
        return self.tracking_enabled

    def _ui_stream_worker(self):
        log("UI Stream loop thread starting...")
        last_ui_time = time.time()
        last_predict_time = time.time()
        while True:
            # Enforce target FPS (30hz) to match belief-tracking and web display rate
            now = time.time()
            elapsed = now - last_ui_time
            if elapsed < FRAME_TIME:
                time.sleep(max(0, FRAME_TIME - elapsed))
            last_ui_time = time.time()

            try:
                frame_idx = getattr(self, 'last_frame_idx', 0)

                # ── Predict at 30 FPS ─────────────────────────────────────────
                t_predict_start = time.time()
                if self.tracking_enabled and self.pipeline_enabled:
                    raw_beliefs = self.tracker.predict(frame_idx, bypass_filters=True)
                else:
                    raw_beliefs = np.empty((0, 6))
                self.tracker_predict_ms = (time.time() - t_predict_start) * 1000

                # Calculate Tracker Predict FPS
                now_t = time.time()
                delta_p = now_t - last_predict_time
                if delta_p > 0:
                    current_predict_fps = 1.0 / delta_p
                    self.predict_fps = (self.predict_fps * 0.9) + (current_predict_fps * 0.1)
                last_predict_time = now_t

                # Format visual predictions for UI with actual tracked IDs
                ui_predictions = []
                human_only = (config.CLASSES == [0])
                for b in raw_beliefs:
                    # raw_beliefs has elements: [top_left_x, top_left_y, width, height, belief_id, confidence_score]
                    cls_id = 0
                    if not human_only and len(b) > 6:
                        cls_id = int(b[6])
                    ui_predictions.append({
                        "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "id": int(b[4]),
                        "conf": float(b[5]),
                        "class_id": cls_id
                    })

                with self.lock:
                    self._ui_predictions = ui_predictions
                    cached_telemetry = self._cached_stats.copy()

                metadata = {
                    "frame_id": frame_idx,
                    "detections": ui_predictions,
                    "visualizations": [],
                    "metrics": {
                        "status": self.status if self.pipeline_enabled else "PIPELINE DISABLED",
                        "cam_fps": round(self.fps_val, 1),

                        # Perception metrics (Sequential YOLO + ReID)
                        "perception_fps": round(self.perception_fps, 1) if self.pipeline_enabled else 0.0,
                        "perception_time_ms": round(self._yolo_time_ms + self._reid_time_ms, 1) if self.pipeline_enabled else 0.0,
                        "yolo_time_ms": round(self._yolo_time_ms, 1) if self.pipeline_enabled else 0.0,
                        "reid_time_ms": round(self._reid_time_ms, 1) if self.pipeline_enabled else 0.0,

                        # Tracker (Update) metrics
                        "track_update_fps": round(self.tracker_fps, 1) if (self.tracking_enabled and self.pipeline_enabled) else 0.0,
                        "track_update_ms": round(self.tracker_update_ms, 1) if (self.tracking_enabled and self.pipeline_enabled) else 0.0,

                        # Tracker (Predict) metrics
                        "track_predict_fps": round(self.predict_fps, 1) if (self.tracking_enabled and self.pipeline_enabled) else 0.0,
                        "track_predict_ms": round(self.tracker_predict_ms, 1) if (self.tracking_enabled and self.pipeline_enabled) else 0.0,

                        "latency_time_ms": round(self.tracker_latency_ms, 1) if (self.tracking_enabled and self.pipeline_enabled) else 0.0,
                        **cached_telemetry
                    }
                }

                payload_bytes = orjson.dumps(metadata)
                with self.lock:
                    self._cached_payload_bytes = payload_bytes
                    self.latest_payload = payload_bytes.decode('utf-8')

                if payload_bytes is not None and self.on_payload_ready:
                    try:
                        self.on_payload_ready(payload_bytes.decode('utf-8'))
                    except Exception as ee:
                        log(f"Callback error: {ee}")
            except Exception as e:
                log(f"UI stream worker error: {e}")

    def _detection_receiver_worker(self):
        """Receives perception results from the subprocess and feeds the tracking queue.
        Also releases the GStreamer inference gate (inference_busy = False) so the next
        frame can be delivered."""
        log("Detection receiver thread starting...")
        last_perception_time = time.time()

        while True:
            try:
                # Block until subprocess sends a result (or errors)
                if not self._result_recv.poll(timeout=1.0):
                    # Check subprocess is still alive
                    if self._perc_proc and not self._perc_proc.is_alive():
                        log("WARNING: perception subprocess died unexpectedly!")
                    continue

                result = self._result_recv.recv()

                if result.get('type') == 'error':
                    log(f"Perception subprocess error: {result.get('error')}")
                    self.ds_engine.inference_busy = False
                    continue

                if result.get('type') != 'dets':
                    continue

                # ── Unpack result ──────────────────────────────────────────────
                dets_arr     = result['dets_arr']
                capture_time = result['capture_time']
                frame_idx    = result['frame_idx']

                # ── Update telemetry ───────────────────────────────────────────
                self._yolo_time_ms = result['yolo_ms']
                self._reid_time_ms = result['reid_ms']

                now_p  = time.time()
                delta_p = now_p - last_perception_time
                if delta_p > 0:
                    self.perception_fps = (
                        self.perception_fps * 0.9 + (1.0 / delta_p) * 0.1
                    )
                last_perception_time = now_p

                # ── Update UI prediction cache ──────────────────────────────────
                # Reconstruct ui_predictions from dets_arr locally —
                # ui_predictions is no longer sent over IPC to keep pickle smaller.
                num_det = len(dets_arr)
                if num_det > 0:
                    ui_predictions = [
                        {'bbox': [float(dets_arr[i, 0]), float(dets_arr[i, 1]),
                                  float(dets_arr[i, 2]), float(dets_arr[i, 3])],
                         'id': i + 1, 'conf': float(dets_arr[i, 4])}
                        for i in range(num_det)
                    ]
                else:
                    ui_predictions = []
                with self.lock:
                    self._ui_predictions = ui_predictions

                # ── Push to tracking queue ──────────────────────────────────
                if not self.enriched_detection_queue.full():
                    self.enriched_detection_queue.put(
                        (dets_arr, capture_time, frame_idx)
                    )

                # inference_busy is now released in _on_ds_frame immediately
                # after writing shm — no longer released here.

            except EOFError:
                log("Perception subprocess pipe closed — subprocess has exited.")
                break
            except Exception as exc:
                log(f"Detection receiver error: {exc}")
                log(traceback.format_exc())
                self.ds_engine.inference_busy = False

    def _stats_worker(self):
        log("Stats thread starting...")
        self.telemetry_tracker.start()
        try:
            while True:
                stats = self.telemetry_tracker.get_stats()
                self.system_stats = stats
                with self.lock:
                    self._cached_stats.update({
                        "cpu": stats.get("cpu", 0),
                        "temp": stats.get("temp", 0),
                        "ram_used": stats.get("ram_used", 0),
                        "ram_total": stats.get("ram_total", 0),
                        "gpu": stats.get("gpu", 0)
                    })
                time.sleep(1)
        finally:
            self.telemetry_tracker.stop()

    def _tracking_worker(self):
        log("Tracking logic thread starting...")
        last_update_time = time.time()
        while True:
            try:
                # Pull enriched detections from the queue
                dets_arr, capture_time, frame_idx = self.enriched_detection_queue.get(timeout=0.1)
                
                # Pass detections to tracker
                t_update_start = time.time()
                self.tracker.update(frame_idx, dets_arr)
                self.tracker_update_ms = (time.time() - t_update_start) * 1000
                
                self.tracker_latency_ms = (time.time() - capture_time) * 1000
                
                # Calculate Tracker Update FPS
                now = time.time()
                delta = now - last_update_time
                if delta > 0:
                    current_track_fps = 1.0 / delta
                    self.tracker_fps = (self.tracker_fps * 0.9) + (current_track_fps * 0.1)
                last_update_time = now
            except queue.Empty:
                pass
            except Exception as e:
                log(f"Tracking worker error: {e}")
                log(traceback.format_exc())
