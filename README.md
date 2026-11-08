# RDR

RDR is an exploratory Python and PyTorch project for differentiable radar signal processing, synthetic phase-history simulation, and SAR image formation (targeting backprojection, Polar Format Algorithm, Range Migration Algorithm, stepped-chirp simulation, fast CPHD reading, and CLEAN). It is organized as an installable `src/`-layout package (`rdr`) with an enforced layered import hierarchy, currently at the M0 package skeleton milestone.

## Installation

Install in editable mode into the project virtual environment (`/home/feildaw/mypyenv`):

```bash
cd /home/feildaw/projects/RDR
/home/feildaw/mypyenv/bin/pip install -e .
```

To include test dependencies (`pytest`, `scipy`, `sarkit`) or visualization dependencies (`matplotlib`, `pillow`):

```bash
/home/feildaw/mypyenv/bin/pip install -e ".[test,viz]"
```

## Running Tests

```bash
cd /home/feildaw/projects/RDR
/home/feildaw/mypyenv/bin/python -m pytest -q
```

## Project Plan

See [`GOALS.md`](GOALS.md) for overall goals, [`PLAN.md`](PLAN.md) for architecture and milestone definitions, and [`briefs/`](briefs/) for per-milestone implementation briefs.
