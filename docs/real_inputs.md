# Prepared observational inputs

The two small derived fixtures correspond to Figure 14's fixed examples:

| Band | Evaluated CAL product stem | Independent reference |
| --- | --- | --- |
| F1000W | `jw01027007001_02105_00003_mirimage` | Six exposures, three other visits |
| F1800W | `jw01027004001_02103_00003_mirimage` | Four exposures, two other visits |

The JSON companion of each NPZ lists the frozen parameters and exact reference members. `REAL_FIXTURE_PROVENANCE.json` provides the archived inputs' hashes. Evaluation visits and PSF-training exposures are excluded from their reference pools. Arrays are on the evaluation's 225 by 225 detector grid; `[57:168,57:168]` is the 111 by 111 displayed stamp.

Upstream calibrated products were retrieved from MAST. SCI is MJy/sr; SCI, ERR, VAR_POISSON, DQ and AREA belong to the same CAL product. DQ bits 1, 2 and 512, nonfinite data and invalid ERR values are excluded. The count scale is the median positive SCI/VAR_POISSON on valid pixels in the fixed 60–100 pixel annulus. After the retained sky-plane subtraction, `b=C**2 * variance` uses clipped background variance. `Fc` is the scaled valid r50 source sum divided by fitted PSF mass; `beta=b/Fc`. The archived shared PSF was fitted on a separate training pair and is not fitted by these examples.

Primary coadds use detector translations, separable Lanczos-3 interpolation and scalar inverse-background-variance weights `C**2/b`. Every nonzero interpolation tap must be valid. The final common likelihood mask intersects native evaluation validity with coverage in both disjoint reference halves. It is intentionally stricter than the full coadd's coverage. Displaying the full coadd through this common analysis mask can produce apparent holes despite valid coadd data.

To apply the numerical core to a new target, prepare and validate the corresponding units, masks, PSF and nuisance parameters first. Passing arbitrary MJy/sr images as effective counts is not the published workflow. The examples take frozen inputs to make numerical regression reproducible; they do not claim an end-to-end raw-data pipeline or intrinsic-sky reconstruction accuracy.

The supplied `expected_outputs` use axes `[RL, SSRL*]`, then `[Raw, FiDeR]`, then `[y,x]`. `expected_references` uses `[RL, SSRL*]`, then `[y,x]`. All contain signed MJy/sr. Stored expected arrays are used only by the comparison checks, never as inputs to the algorithm under test.
