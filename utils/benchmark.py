import time
from utils.logging import get_logger

logger = get_logger("benchmark")

def benchmark(func, *args, **kwargs):
    """Returns (success, duration_seconds)."""
    t0 = time.perf_counter()
    try:
        func(*args, **kwargs)
        success = True
    except Exception as e:
        logger.error(f"Benchmark target failed: {e}")
        success = False
    t1 = time.perf_counter()
    return success, t1 - t0
