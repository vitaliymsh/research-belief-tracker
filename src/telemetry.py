import psutil
import os
from config import get_logger

logger = get_logger("Telemetry")

try:
    from jtop import jtop
    HAS_JTOP = True
except ImportError:
    HAS_JTOP = False

class TelemetryTracker:
    def __init__(self):
        self.total_ram = psutil.virtual_memory().total / (1024**3)
        self.has_jtop = HAS_JTOP
        self._jetson = None
        self._gpu_path = "/sys/devices/gpu.0/load"
        self._temp_path = "/sys/class/thermal/thermal_zone0/temp"

    def start(self):
        if self.has_jtop:
            try:
                self._jetson = jtop()
                self._jetson.start()
            except Exception as e:
                logger.debug(f"jtop service not available: {e}")
                self.has_jtop = False

    def _read_sysfs(self, path: str, divisor: float = 1.0) -> float:
        """Helper to read values from sysfs."""
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return float(f.read().strip()) / divisor
        except Exception:
            pass
        return 0.0

    def get_stats(self) -> dict:
        mem = psutil.virtual_memory()
        stats = {
            "gpu": 0.0,
            "gpu_clock": 0,
            "cpu": psutil.cpu_percent(),
            "temp": 0.0,
            "ram_used": mem.used / (1024**3),
            "ram_total": self.total_ram,
            "ram_free": mem.available / (1024**3)
        }
        
        # Priority 1: jtop (comprehensive stats)
        if self.has_jtop and self._jetson and self._jetson.ok():
            try:
                s = self._jetson.stats
                stats["gpu"] = s.get('GPU', s.get('gpu', s.get('GR3D', 0)))
                stats["gpu_clock"] = s.get('curr_freq_gpu', s.get('GR3D_FREQ', 0))
                stats["temp"] = s.get('AO', s.get('thermal', 0))
                return stats
            except Exception:
                pass
        
        # Priority 2: Direct Sysfs (robust fallback for Jetson)
        # GPU load is in units of 0.1%, so divide by 10 to get percentage
        stats["gpu"] = self._read_sysfs(self._gpu_path, divisor=10.0)
        # Temperature is in millidegrees Celsius
        stats["temp"] = self._read_sysfs(self._temp_path, divisor=1000.0)
                
        return stats

    def stop(self):
        if self._jetson:
            try:
                self._jetson.close()
            except Exception:
                pass
            self._jetson = None
