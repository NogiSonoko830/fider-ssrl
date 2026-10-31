# FiDeR and SSRL research cores

Audited core code accompanying *FiDeR prefiltering for Richardson–Lucy deconvolution of astronomical images*, revised draft of 31 October 2026.

This package contains the FiDeR prefilter, classical Richardson–Lucy (RL), signed Snyder RL (SSRL), and the explicitly distinguished local masked implementation SSRL*. It includes fixed F1000W/F1800W simulation and observation examples. This is a core-method release, not the complete 311-realization study, an automatic JWST reduction pipeline, or the multi-exposure UPDC engine.

## Install and check

Use Python 3.11 or later. In a terminal opened in this directory:

```text
python -m venv .venv
```

On Windows, the following commands work without activating the environment:

```text
.venv\Scripts\python -m pip install -r requirements-tested.txt
.venv\Scripts\python -m pip install --no-deps -e .
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python examples/replay_simulation.py
.venv\Scripts\python examples/replay_real.py
```

On macOS/Linux, replace `.venv\Scripts\python` with `.venv/bin/python`. The demonstration data are already included; neither an account nor a MAST download is needed to run them. The scripts compare against archived arrays and stop with an error if a relative maximum difference exceeds `1e-10`. Successful runs print their checks.

The retained environment uses NumPy 2.3.5 and SciPy 1.17.0. Validation of this release passed 28 fixed-array comparisons and three convention tests; details are in [docs/validation.json](docs/validation.json). These are numerical regression checks, not estimates of general scientific performance.

## Where to start

| Module | Role |
| --- | --- |
| `simulation_fider.py` | Original total-intensity RL1000, calibration at `beta=b/Fc`, Fourier matching with `eta=3e-14` and unit DC |
| `simulation_rl.py` | Original complete-data RL with inherited `1e-12` iterate/prediction safeguards |
| `simulation_ssrl.py` | Signed, background-aware same-grid SSRL; persistent neutral ratios and absorbing zeros |
| `real_fider.py` | Native-mask FiDeR on the fixed 225-pixel canvas, including subpixel PSF calibration |
| `masked_backend.py` | Final common-mask RL/SSRL*, harmonic initialization, and masked sensitivity |
| `masked_rl.py`, `fider_core.py`, `psf_sampling.py` | Retained local numerical primitives |

The original numerical function bodies are retained, not approximated through a new library call. `PROVENANCE.json` records source paths, hashes, function locations and AST equality. Imports are reduced to NumPy/SciPy; old machine paths and batch drivers are not executed. The local independent `FFTModel` has a different zero-input floor convention from the original simulation implementation. The two are intentionally separate: do not replace one by the other solely because ordinary high-count examples agree.

## Scientific conventions

- FiDeR's 1000 internal iterations and each final backend's 40 iterations are separate stages. No output renormalization or parameter selection from scores is performed.
- Simulation sources and backgrounds are in effective counts. `source` is signed, background-subtracted; `source+b` must satisfy the retained positive-total checks. Supply a finite, nonnegative, normalized PSF. Odd square stamps embed centrally in an odd periodic FFT canvas.
- SSRL retains negative source values; zero is absorbing. A nonpositive predicted total permanently disables that observed ratio by assigning it one. This does not freeze a latent pixel.
- Missing observations contribute zero ratios and sensitivity is `H^T M`. Harmonic interpolation initializes latent/filter inputs; it does not add observations.
- For real examples, source/coadd/output arrays are MJy/sr. `C` converts these to effective counts, while `b` and `Fc` are already effective counts. The native validity mask is used in the FiDeR front end; the stricter common mask is used in final evaluation, including the coadd reference.
- The real reference is an independent-visit direct Lanczos-3 coadd passed through the same unfiltered 40-step backend. It is not Drizzle, intrinsic ground truth, or one common reference across RL and SSRL*.
- Real `C`, `b`, `Fc`, stellar centres, PSFs and coadd membership are frozen supplied inputs. Their extraction, nuisance fitting, complete coadd construction, other six prefilters, full parameter grids and paper plotting are outside this core release. See [docs/real_inputs.md](docs/real_inputs.md).

## Attribution and scope of release

The original simulation functions were supplied with the FiDeR mother project; the local observation modules are the separately audited masked extensions. SSRL follows the documented same-grid signed reduction of UPDC 1.2.0, not its geometry frontend. Provenance retains that distinction. No third-party BM3D source, binaries, proprietary data, full raw CAL files, environment caches or unpublished manuscript PDF is included here.

The companion description is the author-supplied manuscript by Lei Wang, Bin Li, Lin Nie, Zhaojun Yan, Xuechun Chen, Yachong Diao and Huanyuan Shan, *FiDeR: Denoising Astronomical Images at the Observational Resolution*, version 8 October 2026. A formal publication identifier and the final software citation remain author items; no DOI is invented.

An explicit project licence was not present in the supplied core sources. This prepared package does not invent or change a licence; the authors can add their chosen licence and final citation before a formal software release. Dependency licences remain with their respective projects.

中文使用说明见 [README_zh.md](README_zh.md)。
