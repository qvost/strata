"""Linux performance-counter wrapper for the benchmark.

``perf(1)`` is not installed in every environment, and hardware counters are
often disabled for unprivileged users.  This module opens the counters
directly with the ``perf_event_open(2)`` syscall and counts only user-space
events (``exclude_kernel=1``), which works with
``/proc/sys/kernel/perf_event_paranoid=2``.

If a counter is unavailable the wrapper degrades gracefully: the corresponding
value is omitted rather than failing the benchmark.
"""

from __future__ import annotations

import ctypes
import os
import platform
from typing import Callable, Iterable, Optional, TypeVar

T = TypeVar("T")

# perf_event_attr flags
_PERF_ATTR_DISABLED = 1 << 0
_PERF_ATTR_EXCLUDE_KERNEL = 1 << 5
_PERF_ATTR_EXCLUDE_HV = 1 << 6

# perf_event_open syscall numbers
_SYSCALLS = {"x86_64": 298, "amd64": 298, "aarch64": 241, "arm64": 241}

# perf_type_id
PERF_TYPE_HARDWARE = 0
PERF_TYPE_HW_CACHE = 3

# hardware event ids
PERF_COUNT_HW_CPU_CYCLES = 0
PERF_COUNT_HW_INSTRUCTIONS = 1
PERF_COUNT_HW_CACHE_MISSES = 3
PERF_COUNT_HW_CACHE_REFERENCES = 4
PERF_COUNT_HW_BRANCH_MISSES = 5

# cache event encoding
PERF_COUNT_HW_CACHE_L1D = 0
PERF_COUNT_HW_CACHE_LL = 2
PERF_COUNT_HW_CACHE_OP_READ = 0
PERF_COUNT_HW_CACHE_OP_WRITE = 1
PERF_COUNT_HW_CACHE_OP_PREFETCH = 2
PERF_COUNT_HW_CACHE_RESULT_ACCESS = 0
PERF_COUNT_HW_CACHE_RESULT_MISS = 1

_IOCTL_ENABLE = 0x2400
_IOCTL_DISABLE = 0x2401
_IOCTL_RESET = 0x2403

_LIBC = ctypes.CDLL(None, use_errno=True)


class _PerfEventAttr(ctypes.Structure):
    """Mirror of ``struct perf_event_attr`` (Linux UAPI), 128 bytes."""

    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32),
        ("bp_addr", ctypes.c_uint64),
        ("bp_len", ctypes.c_uint64),
        ("branch_sample_type", ctypes.c_uint64),
        ("sample_regs_user", ctypes.c_uint64),
        ("sample_stack_user", ctypes.c_uint32),
        ("clockid", ctypes.c_int32),
        ("sample_regs_intr", ctypes.c_uint64),
        ("aux_watermark", ctypes.c_uint32),
        ("sample_max_stack", ctypes.c_uint16),
        ("__reserved_2", ctypes.c_uint16),
        ("aux_sample_size", ctypes.c_uint32),
        ("__reserved_3", ctypes.c_uint32),
        ("sig_data", ctypes.c_uint64),
    ]


def _cache_config(cache: int, op: int, result: int) -> int:
    return int(cache) | (int(op) << 8) | (int(result) << 16)


# Friendly event name -> (perf_type, config)
EVENTS: dict[str, tuple[int, int]] = {
    "llc_loads": (PERF_TYPE_HW_CACHE, _cache_config(
        PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS
    )),
    "llc_load_misses": (PERF_TYPE_HW_CACHE, _cache_config(
        PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS
    )),
    "l1d_load_misses": (PERF_TYPE_HW_CACHE, _cache_config(
        PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS
    )),
    "cache_misses": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES),
    "cache_references": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES),
    "cycles": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES),
    "instructions": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS),
    "branch_misses": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES),
}

DEFAULT_EVENTS = ("llc_loads", "llc_load_misses", "cycles", "instructions")


def perf_available() -> bool:
    """Return True if at least one user-space hardware counter can be opened."""

    counter = PerfCounter(events=("cycles",))
    opened = bool(counter.opened)
    counter.close()
    return opened


class PerfCounter:
    """A small set of process-wide user-space hardware counters."""

    def __init__(self, events: Iterable[str] = DEFAULT_EVENTS) -> None:
        self.events = tuple(dict.fromkeys(events))
        self.fds: dict[str, int] = {}
        self.errors: dict[str, str] = {}
        self._running = False
        self._open()

    def _open(self) -> None:
        machine = platform.machine().lower()
        syscall = _SYSCALLS.get(machine)
        if syscall is None:
            self.errors["_all"] = f"unsupported architecture: {machine}"
            return
        for name in self.events:
            if name not in EVENTS:
                self.errors[name] = "unknown event"
                continue
            perf_type, config = EVENTS[name]
            attr = _PerfEventAttr()
            attr.type = perf_type
            attr.size = ctypes.sizeof(_PerfEventAttr)
            attr.config = config
            attr.flags = (
                _PERF_ATTR_DISABLED
                | _PERF_ATTR_EXCLUDE_KERNEL
                | _PERF_ATTR_EXCLUDE_HV
            )
            # pid=0 (self), cpu=-1 (follow task), group_fd=-1, flags=0
            fd = _LIBC.syscall(
                syscall, ctypes.byref(attr), 0, -1, -1, 0
            )
            if fd < 0:
                errno = ctypes.get_errno()
                self.errors[name] = os.strerror(errno)
            else:
                self.fds[name] = int(fd)

    @property
    def opened(self) -> tuple[str, ...]:
        return tuple(self.fds)

    def _ioctl_all(self, request: int) -> None:
        for fd in self.fds.values():
            _LIBC.ioctl(fd, request, 0)

    def reset(self) -> None:
        if self.fds:
            self._ioctl_all(_IOCTL_RESET)

    def start(self) -> "PerfCounter":
        if self.fds:
            self._ioctl_all(_IOCTL_RESET)
            self._ioctl_all(_IOCTL_ENABLE)
            self._running = True
        return self

    def stop(self) -> "PerfCounter":
        if self.fds and self._running:
            self._ioctl_all(_IOCTL_DISABLE)
            self._running = False
        return self

    def read(self) -> dict[str, int]:
        values: dict[str, int] = {}
        for name, fd in self.fds.items():
            buf = ctypes.c_uint64(0)
            result = _LIBC.read(fd, ctypes.byref(buf), ctypes.sizeof(buf))
            if result == ctypes.sizeof(buf):
                values[name] = int(buf.value)
        return values

    def close(self) -> None:
        if self._running:
            self.stop()
        for fd in self.fds.values():
            try:
                os.close(fd)
            except OSError:  # pragma: no cover
                pass
        self.fds.clear()

    def __enter__(self) -> "PerfCounter":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


def measure(
    fn: Callable[[], T],
    *,
    events: Iterable[str] = DEFAULT_EVENTS,
) -> tuple[T, dict[str, int], dict[str, str]]:
    """Run ``fn`` while counting hardware events.

    Returns ``(result, counters, errors)``.  If no counter can be opened,
    ``counters`` is empty and ``errors`` explains why.
    """

    counter = PerfCounter(events=events)
    try:
        counter.start()
        result = fn()
        counter.stop()
        return result, counter.read(), dict(counter.errors)
    finally:
        counter.close()


def summarize(counters: dict[str, int]) -> dict[str, float]:
    """Derive miss rates from raw counter values."""

    summary: dict[str, float] = {}
    loads = counters.get("llc_loads")
    misses = counters.get("llc_load_misses")
    if loads is not None and misses is not None:
        summary["llc_miss_rate"] = misses / loads if loads else 0.0
    cycles = counters.get("cycles")
    instructions = counters.get("instructions")
    if cycles:
        summary["ipc"] = instructions / cycles if instructions is not None else 0.0
    return summary


if __name__ == "__main__":  # pragma: no cover
    import time

    def work() -> int:
        total = 0
        for i in range(3_000_000):
            total += i * i
        return total

    start = time.perf_counter()
    _, counters, errors = measure(work)
    elapsed = time.perf_counter() - start
    print(f"elapsed: {elapsed:.3f}s")
    print("counters:", counters)
    print("derived :", summarize(counters))
    if errors:
        print("unavailable:", errors)
