import contextlib
import time


@contextlib.contextmanager
def print_time():
    t1 = time.perf_counter()
    yield
    t2 = time.perf_counter()
    print(f"takes: {t2-t1:.2f}s")
