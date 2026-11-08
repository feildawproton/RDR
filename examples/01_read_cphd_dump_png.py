"""Smoke-test example: read the 139 MB Umbra CPHD and dump raw & range-profile PNGs.

Physics & DFT shift notes (for inspecting range profiles or plotting a single pulse)
------------------------------------------------------------------------------------
1. FX-domain frequency grid (CPHD 1.1.0 DIDD §4.2 & §4.5, pp. 46-47, 55):
   Each row ``v`` is one pulse (``NumVectors = 4346``) and each column
   ``s in {0, ..., N - 1}`` (``N = NumSamples = 3999``) is a compensated/dechirped
   RF frequency sample at:

       f(v, s) = SC0[v] + s * SCSS[v]    [Hz]

   spanning roughly ``[f_c - B/2, f_c + B/2)``. On this 2023-09-11 Umbra-05 file
   (at ``v = 2000``, ``N = 3999``):
     - ``SC0 = 9.552617 GHz``, ``SCSS = 23.7031 kHz``, ``N * SCSS = 94.79 MHz``.
     - Saved echo band ``[FX1, FX2] = [9.553174 GHz, 9.646824 GHz]`` (``93.65 MHz``),
       leaving ~0.57 MHz of roll-off/margin at ``s = 0`` and ``s = N - 1`` (visible as
       thin dark left/right borders in ``01_raw_signal_amplitude.png``).
     - Center sample ``s_c = N // 2 = 1999`` sits at ``9.5999993 GHz`` (``~FxC``).

2. CPHD FX phase model (CPHD 1.1.0 DIDD §4.5, p. 55):
   After compensation to the Stabilization Reference Point (SRP), a scatterer with
   two-way Time-of-Arrival difference ``DeltaTOA(v) = t_tgt(v) - t_SRP(v)`` has
   simple phase (in cycles):

       DeltaPhi_0(v, s) = SGN * f(v, s) * DeltaTOA(v)

   (ignoring the small Doppler time-dilation micro-parameters ``aFDOP``, ``aFRR1``,
   ``aFRR2`` in §4.5, which shift range peaks by < 0.25 sample). Writing
   ``f(v, s) = f_ref + (s - s_c) * SCSS`` with ``s_c = N // 2`` and
   ``f_ref = SC0 + s_c * SCSS``, a point target produces:

       S[v, s] ~ exp(j * 2*pi * SGN * f_ref * DeltaTOA)
                 * exp(j * 2*pi * SGN * (s - s_c) * SCSS * DeltaTOA)

   In all three Umbra files ``SGN = -1``, so positive ``DeltaTOA`` (far range,
   arriving later than the SRP) has *negative* phase slope vs. frequency, and
   negative ``DeltaTOA`` (near range, arriving earlier than the SRP) has *positive*
   phase slope vs. frequency.

3. Why bare ``torch.fft.ifft(sig, dim=-1)`` looks "inverted / inside-out":
   At the SRP (``DeltaTOA = 0``), phase across ``s`` is flat (DC, DFT bin ``k = 0``).
   An unshifted ``ifft`` therefore puts ``DeltaTOA = 0`` at **column 0 (left edge)**:
     - Columns ``0 .. N//2`` hold **far range** (``DeltaTOA >= 0``, out to
       ``+1 / (2 * SCSS) = +21.09 us``).
     - Column ``N//2`` (the middle of the PNG!) is the **Nyquist wraparound / TOA
       guard band** (``|DeltaTOA| ~ 21.09 us``, outside the saved extended swath
       ``[TOAE1, TOAE2] = [-16.52 us, +16.53 us]``), creating a dark vertical band
       right down the center of an unshifted image.
     - Columns ``N//2 + 1 .. N - 1`` (right half) hold **near range**
       (``DeltaTOA < 0``) wrapped to negative DFT frequencies.

4. Why ``ifftshift`` *before* ``ifft`` alone does NOT change the amplitude image:
   By the DFT shift theorem, circularly shifting the input in frequency
   (``ifftshift(sig, dim=-1)``, moving sample ``s_c = N // 2`` to index ``0``) only
   multiplies each ``ifft`` output bin ``k`` by the unit-magnitude phase ramp
   ``exp(+j * 2*pi * k * (N // 2) / N)``. Because ``|exp(j * theta)| == 1``:

       |ifft(ifftshift(sig))| == |ifft(sig)|    (identically!)

   Each shift serves a distinct physical role:
     - ``ifftshift`` **before** the transform (frequency shift) moves ``f_ref``
       (``~FxC``) to index ``0``, referencing the compressed peak's **phase** to
       ``f_ref`` instead of band-edge ``SC0``. Invisible in a single-pulse magnitude
       plot, but essential for coherent cross-pulse/azimuth processing and phase plots.
     - ``fftshift`` **after** the transform (delay/range shift) moves ``DeltaTOA = 0``
       (the SRP) from bin ``0`` to the center column ``N // 2``, placing **near range
       on the left**, **SRP in the middle**, **far range on the right**, and the dark
       TOA guard bands on both outer edges.

5. Role of ``SGN`` (``cf.meta.sgn``):
   ``torch.fft.ifft`` uses the ``+j`` kernel ``(1/N) * sum_s x[s] exp(+j 2*pi s k / N)``.
   For ``SGN == -1``, ``+j`` cancels ``exp(-j 2*pi s SCSS DeltaTOA)`` at positive ``k``
   when ``DeltaTOA > 0``, so column index increases from near range to far range.
   For a ``SGN == +1`` CPHD, ``torch.fft.fft(..., dim=-1) / N`` (the ``-j`` kernel
   with the same ``1/N`` scaling, equivalent to ``norm="forward"``) is needed instead;
   using ``ifft`` on ``SGN == +1`` (or ``fft`` on ``SGN == -1``) mirrors near and far
   range left-to-right.

6. Coordinate axes for pulling apart a single pulse ``v`` in plots:
     - Frequency axis of ``sig[v]``:
         ``f_hz = pvp["SC0"][v] + np.arange(N) * pvp["SCSS"][v]``
     - Two-way relative delay axis of ``range_profiles[v]`` (SRP at ``N // 2``):
         ``delta_toa_s = (np.arange(N) - (N // 2)) / (N * pvp["SCSS"][v])``
     - Approximate monostatic slant-range offset relative to the SRP:
         ``delta_r_m = 0.5 * 299_792_458.0 * delta_toa_s``
       (full-resolution echoes lie within ``[pvp["TOA1"][v], pvp["TOA2"][v]]``,
       extended/partially-eclipsed echoes within ``[pvp["TOAE1"][v], pvp["TOAE2"][v]]``,
       and samples outside ``[TOAE1, TOAE2]`` are the dark TOA guard band).

Note: project-wide reusable FFT/shift helpers belong to Milestone M3 (``rdr.signal``);
the centered range-profile transform below is written inline here for clarity.
"""

from __future__ import annotations

import os
from pathlib import Path
from pprint import pprint

import torch

from rdr.io.cphd import CphdFile
from rdr.io.png import density_remap, write_png

DEFAULT_FILENAME = "2023-09-11-10-37-05_UMBRA-05_CPHD.cphd"


def main() -> int:
    data_dir = Path(os.environ.get("RDR_DATA_DIR", "/home/feildaw/data"))
    cphd_path = data_dir / DEFAULT_FILENAME
    out_dir = Path(__file__).resolve().parent / "output"

    cf = CphdFile(cphd_path)
    pprint(cf.meta)

    ch_id = cf.channels[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sig = cf.signal(ch_id, device=device)

    raw_u8, raw_params = density_remap(sig)
    raw_png = out_dir / "01_raw_signal_amplitude.png"
    write_png(raw_png, raw_u8)
    print(f"Wrote raw signal PNG:     {raw_png} ({raw_params})")

    # Form centered complex range profiles along fast-time frequency samples (dim=-1):
    #   1) ifftshift moves center frequency sample s_c = N // 2 (~FxC) to DFT index 0
    #      so output phase is referenced to center frequency (does not alter |amplitude|).
    #   2) Transform sign follows CPHD Global/SGN (DIDD §4.5): SGN == -1 -> ifft (+j
    #      kernel); SGN == +1 -> fft / N (-j kernel, matching 1/N scale) so increasing
    #      column index always runs from near range (-DeltaTOA) to far range (+DeltaTOA).
    #   3) fftshift moves DeltaTOA = 0 (the SRP) from bin 0 to center column N // 2,
    #      putting near range on the left, far range on the right, and TOA guard bands
    #      on both outer edges.
    sig_centered = torch.fft.ifftshift(sig, dim=-1)
    if cf.meta.sgn == -1:
        rp_unshifted = torch.fft.ifft(sig_centered, dim=-1)
    elif cf.meta.sgn == 1:
        rp_unshifted = torch.fft.fft(sig_centered, dim=-1) / sig.shape[-1]
    else:
        raise ValueError(f"Unexpected CPHD Global/SGN={cf.meta.sgn!r}; expected -1 or +1")
    range_profiles = torch.fft.fftshift(rp_unshifted, dim=-1)

    rp_u8, rp_params = density_remap(range_profiles)
    rp_png = out_dir / "01_range_profiles_amplitude.png"
    write_png(rp_png, rp_u8)
    print(f"Wrote range profiles PNG: {rp_png} ({rp_params})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
