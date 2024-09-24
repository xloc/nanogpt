from functools import wraps
from pathlib import Path
import pickle


def serialize_args(fn_name, args, kwargs):
    args = [str(a) for a in args]
    kwargs = [f'{k}={v}' for k, v in kwargs.items()]
    args_str = ', '.join(args + kwargs)
    return f"{fn_name}({args_str})"


class CacheFile:
    def __init__(self, fn_name, args, kwargs):
        folder = Path('cache')
        folder.mkdir(exist_ok=True)
        self.path = folder / serialize_args(fn_name, args, kwargs)

    def exists(self):
        return self.path.exists()

    def load(self):
        with self.path.open('rb') as f:
            return pickle.load(f)

    def save(self, data):
        with self.path.open('wb') as f:
            pickle.dump(data, f)


def cache_iter(n_iter: int):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cf = CacheFile(func.__name__, args, kwargs)
            if cf.exists():
                cache = cf.load()
                if len(cache) >= n_iter:
                    # use cached data first
                    yield from cache
                    # init real iter, fastforward, and yield the rest
                    iterator = iter(func(*args, **kwargs))
                    for _ in range(n_iter):
                        next(iterator)
                    yield from iterator
                    return
            else:
                # iterate and append first n_iter
                cache = []
                iterator = iter(func(*args, **kwargs))
                for _, i in zip(range(n_iter), iterator):
                    cache.append(i)
                    yield i
                # save cache to file
                cf.save(cache)
                # yield the rest
                yield from iterator

        return wrapper
    return decorator


if __name__ == "__main__":
    import time

    @cache_iter(5)
    def numbers(n):
        for i in range(n):
            time.sleep(0.1)
            yield i

    for i in numbers(10):
        print(i)
