"""Project-wide physical, array-shape, and numerical dtype conventions for RDR.

All higher layers import physical constants from this module rather than restating them.

Conventions pinned here:
- Speed of light: ``SPEED_OF_LIGHT = 299792458.0`` m/s (exact SI definition).
- Spatial frequency ``k`` is in cycles per meter, with ``k = 2 * f / c`` for monostatic
  collections (where ``f`` is RF frequency in Hz and ``c`` is ``SPEED_OF_LIGHT``).
- Phase history arrays are shaped ``(vectors, samples)``, with vectors (pulses) first
  and frequency/fast-time samples second.
- Images are shaped ``(rows, cols)``, with ``rows`` along the first image-plane basis
  vector handed to the image former and ``cols`` along the second. Which physical axis
  each corresponds to (range or cross-range) is chosen by the caller and is not assumed
  by lower layers.
- Signal data is stored and processed as ``complex64``. Phases, geometry, and k-space
  coordinates are computed in ``float64`` and cast at the last step.
- FFT sign and shift conventions are NOT decided here; they are pinned in Milestone M3
  in ``rdr.signal``.
"""

SPEED_OF_LIGHT = 299792458.0
