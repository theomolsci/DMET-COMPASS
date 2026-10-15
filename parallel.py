"""Process-based parallel helpers (loky, shipped with joblib).

Statevector simulation of a 12-qubit circuit is fast, but the Python side
(circuit construction, transpilation, estimator overhead) holds the GIL, so
processes rather than threads are used. ``n_jobs == 1`` runs everything
serially in the current process, without starting any workers.
"""

import math
import os

from joblib.externals.loky import get_reusable_executor


def resolve_n_jobs(n_jobs):
    """``-1`` (or any value < 1) means "all available cores"."""
    if n_jobs is None or n_jobs < 1:
        return os.cpu_count() or 1
    return int(n_jobs)


def _executor(n_jobs):
    return get_reusable_executor(max_workers=n_jobs)


def pmap(fn, items, n_jobs=-1):
    """Ordered parallel map: ``[fn(x) for x in items]``."""
    items = list(items)
    n_jobs = resolve_n_jobs(n_jobs)
    if n_jobs == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    return list(_executor(n_jobs).map(fn, items))


class CountingMap:
    """Map-like callable for SciPy's ``workers=`` option.

    SciPy evaluates finite-difference gradients as ``workers(fun, xs)``. The
    gradient components are independent, so they are spread over processes;
    the arithmetic SciPy performs on the returned values is unchanged, so the
    optimisation trajectory is bit-for-bit the serial one. ``count`` records how
    many function evaluations were sent to workers.
    """

    def __init__(self, n_jobs=-1):
        self.n_jobs = resolve_n_jobs(n_jobs)
        self.count = 0

    def __call__(self, fn, iterable):
        items = list(iterable)
        self.count += len(items)
        if self.n_jobs == 1 or len(items) <= 1:
            return [fn(x) for x in items]
        # One chunk per worker: the (large) circuit inside ``fn`` is pickled
        # once per worker instead of once per gradient component.
        chunksize = max(1, math.ceil(len(items) / self.n_jobs))
        return list(_executor(self.n_jobs).map(fn, items, chunksize=chunksize))
