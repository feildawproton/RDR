"""Benchmark CPHD reading stages against raw disk/page-cache file reads."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from benchmarks.harness import Benchmark
from rdr.io.cphd import CphdFile, parse_file_header_bytes


def _read_signal_block_extent(cphd_path: Path) -> tuple[int, int]:
    """Read only the File Header to find SIGNAL_BLOCK_BYTE_OFFSET and SIGNAL_BLOCK_SIZE."""
    with cphd_path.open("rb") as f:
        buf = bytearray()
        while b"\f\n" not in buf:
            chunk = f.read(4096)
            if not chunk:
                raise EOFError("Unexpected EOF reading CPHD header")
            buf.extend(chunk)
    term = buf.index(b"\f\n")
    _, kvps = parse_file_header_bytes(bytes(buf[:term]))
    return int(kvps["SIGNAL_BLOCK_BYTE_OFFSET"]), int(kvps["SIGNAL_BLOCK_SIZE"])


def _sanitize_identifier(stem: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", stem).strip("_")
    return cleaned or "file"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cphd_path", type=Path, help="Path to CPHD 1.1.0 file")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Target torch device for signal() stage (e.g. 'cpu' or 'cuda')",
    )
    args = parser.parse_args(argv)

    cphd_path = args.cphd_path.resolve()
    target_device = torch.device(args.device)
    if target_device.type == "cuda":
        # Initialize CUDA context outside the benchmark timer and peak-memory reset.
        _ = torch.empty(1, device=target_device)
        torch.cuda.synchronize()

    sig_offset, sig_size = _read_signal_block_extent(cphd_path)
    bench_name = f"cphd_read_{_sanitize_identifier(cphd_path.stem)}"

    # Preallocate the raw_read destination buffer before entering Benchmark so
    # host buffer allocation is excluded from raw_read timing, then release it
    # immediately after raw_read so host RAM stays well within 15 GB.
    raw_buf = bytearray(sig_size)

    with Benchmark(bench_name) as b:
        with b.stage("raw_read"):
            mv = memoryview(raw_buf)
            pos = 0
            with cphd_path.open("rb") as f:
                f.seek(sig_offset)
                while pos < sig_size:
                    n = f.readinto(mv[pos:])
                    if not n:
                        raise EOFError(
                            f"Truncated signal block during raw_read ({pos}/{sig_size})"
                        )
                    pos += n
        print(f"stage raw_read:   {b.stages['raw_read']:.4f} s ({sig_size / 1e9 / b.stages['raw_read']:.2f} GB/s)", flush=True)
        del mv, raw_buf

        with b.stage("header_xml"):
            cf = CphdFile(cphd_path)
        print(f"stage header_xml: {b.stages['header_xml']:.4f} s", flush=True)

        channel = cf.channels[0]
        with b.stage("pvp"):
            pvp = cf.pvp(channel)
        print(f"stage pvp:        {b.stages['pvp']:.4f} s ({len(pvp)} fields)", flush=True)

        with b.stage("io"):
            sig = cf.signal(channel, device=target_device)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
        print(
            f"stage io:         {b.stages['io']:.4f} s "
            f"(shape={tuple(sig.shape)}, dtype={sig.dtype}, device={sig.device})",
            flush=True,
        )

    assert b.result is not None
    io_to_raw = b.stages["io"] / b.stages["raw_read"]
    print(f"io / raw_read:    {io_to_raw:.3f}x", flush=True)
    if b.result["peak_cuda_memory_bytes"] is not None:
        peak_b = b.result["peak_cuda_memory_bytes"]
        print(
            f"peak CUDA mem:    {peak_b} bytes ({peak_b / sig_size:.3f}x signal bytes)",
            flush=True,
        )
    print(f"json output:      {b.output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
