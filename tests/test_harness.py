"""Tests for ``benchmarks/harness.py``."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from benchmarks.harness import DEFAULT_RESULTS_DIR, Benchmark


def test_default_results_dir_location() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert DEFAULT_RESULTS_DIR == repo_root / "benchmarks" / "results"


def test_benchmark_writes_json_with_expected_fields(tmp_path: Path) -> None:
    with Benchmark("trivial_two_stage", results_dir=tmp_path) as b:
        with b.stage("io"):
            host = torch.ones(1024, dtype=torch.float32)
        with b.stage("compute"):
            if torch.cuda.is_available():
                dev = host.cuda()
                _ = (dev * 2.0).sum().item()
            else:
                _ = (host * 2.0).sum().item()

    assert b.output_path is not None
    assert b.output_path.exists()
    assert b.output_path.parent == tmp_path
    assert b.output_path.name.startswith("trivial_two_stage_")
    assert b.output_path.suffix == ".json"

    data = json.loads(b.output_path.read_text(encoding="utf-8"))
    assert data["name"] == "trivial_two_stage"
    assert isinstance(data["timestamp"], str) and data["timestamp"].endswith("Z")
    assert set(data["stages"].keys()) == {"io", "compute"}
    assert data["stages"]["io"] > 0.0
    assert data["stages"]["compute"] > 0.0
    assert data["total_seconds"] >= data["stages"]["io"] + data["stages"]["compute"]
    assert isinstance(data["io_over_compute"], float)
    assert data["io_over_compute"] > 0.0
    assert data["torch_version"] == torch.__version__

    if torch.cuda.is_available():
        assert isinstance(data["peak_cuda_memory_bytes"], int)
        assert data["peak_cuda_memory_bytes"] > 0
        assert data["device"] == torch.cuda.get_device_name(0)
    else:
        assert data["peak_cuda_memory_bytes"] is None
        assert data["device"] == "cpu"
