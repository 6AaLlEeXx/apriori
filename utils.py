from time import perf_counter

def time_manager(operation_name, verbose=False, time_tracker=None, time_ref=0.):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = perf_counter()
            result = func(*args, **kwargs)
            total_seconds = perf_counter() - start

            if verbose:
                print(f"The operation {operation_name} took {total_seconds*1000}ms to complete.")
            if time_tracker is not None:
                time_tracker[operation_name] = (start*1000-time_ref, start*1000-time_ref+total_seconds*1000)
            return result
        return wrapper
    return decorator