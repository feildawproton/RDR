"""Context-manager benchmark timer writing per-run JSON metrics."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import TracebackType
from typing import Any

import torch

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


class Benchmark:
    """Measure total and per-stage wall times plus peak CUDA memory, saving JSON on exit."""

    def __init__(
        self,
        name: str,
        results_dir: str | Path | None = None,
    ) -> None:
        if not name:
            raise ValueError("Benchmark name must be non-empty")
        self.name = name
        self.results_dir = (
            Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
        )
        self.stages: dict[str, float] = {}
        self.output_path: Path | None = None
        self.result: dict[str, Any] | None = None
        self._started_utc: datetime | None = None
        self._start_perf: float | None = None
        self._cuda_available: bool = False

    def __enter__(self) -> Benchmark:
        self._cuda_available = torch.cuda.is_available()
        if self._cuda_available:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._started_utc = datetime.now(timezone.utc)
        self._start_perf = time.perf_counter()
        return self

    @contextmanager
    def stage(self, stage_name: str) -> Iterator[None]:
        """Time a named stage within the benchmark, accumulating repeated entries."""
        if self._start_perf is None:
            raise RuntimeError("Benchmark.stage() must be used inside a 'with Benchmark(...)' block")
        if self._cuda_available:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if self._cuda_available:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        self.stages[stage_name] = self.stages.get(stage_name, 0.0) + elapsed

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if exc_type is not None:
            return False

        assert self._start_perf is not None
        assert self._started_utc is not None

        if self._cuda_available:
            torch.cuda.synchronize()
        total_seconds = time.perf_counter() - self._start_perf

        peak_cuda_memory_bytes: int | None
        device_name: str
        if self._cuda_available:
            peak_cuda_memory_bytes = int(torch.cuda.max_memory_allocated())
            device_name = torch.cuda.get_device_name(0)
        else:
            peak_cuda_memory_bytes = None
            device_name = "cpu"

        io_over_compute: float | None = None
        if "io" in self.stages and "compute" in self.stages:
            io_over_compute = self.stages["io"] / self.stages["compute"]

        record: dict[str, Any] = {
            "name": self.name,
            "timestamp": self._started_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "stages": dict(self.stages),
            "total_seconds": total_seconds,
            "io_over_compute": io_over_compute,
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
            "torch_version": torch.__version__,
            "device": device_name,
        }

        self.results_dir.mkdir(parents=True, exist_ok=True)
        ts_slug = self._started_utc.strftime("%Y%m%dT%H%M%S_%fZ")
        out_path = self.results_dir / f"{self.name}_{ts_slug}.json"
        out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

        self.output_path = out_path
        self.result = record
        return False
