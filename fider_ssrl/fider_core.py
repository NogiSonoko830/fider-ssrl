"""Independent equation-based implementation for the supplied MIRI mocks.

This is not the author's original code and not a general JWST pipeline.
Validated domain: constant background, odd same-grid PSF, periodic padded
FFT canvas, unmasked pixels. Do not silently fill real DQ masks with zero.
"""
from dataclasses import dataclass
import numpy as np
from scipy.fft import rfft2, irfft2


@dataclass
class FFTModel:
    psf: np.ndarray
    canvas_size: int = 225

    def __post_init__(self):
        self.psf = np.asarray(self.psf, dtype=float)
        if self.psf.ndim != 2 or self.psf.shape[0] != self.psf.shape[1]:
            raise ValueError("A square PSF is required.")
        size = self.psf.shape[0]
        if size % 2 != 1 or self.canvas_size % 2 != 1 or size > self.canvas_size:
            raise ValueError("Odd stamp/canvas sizes with canvas >= stamp are required.")
        if not np.all(np.isfinite(self.psf)) or np.any(self.psf < 0) or self.psf.sum() <= 0:
            raise ValueError("PSF must be finite, nonnegative, and have positive sum.")
        self.shape = (self.canvas_size, self.canvas_size)
        lo = (self.canvas_size - size) // 2
        self.crop = np.s_[lo:lo + size, lo:lo + size]
        self.psf_canvas = np.zeros(self.shape)
        self.psf_canvas[self.crop] = self.psf / self.psf.sum()
        self.otf = rfft2(np.fft.ifftshift(self.psf_canvas))

    def embed(self, source, background=0.):
        source = np.asarray(source, dtype=float)
        if source.shape != self.psf.shape or not np.all(np.isfinite(source)):
            raise ValueError("Source shape/finite-value check failed.")
        result = np.full(self.shape, float(background))
        result[self.crop] += source
        return result

    def forward(self, image):
        return irfft2(rfft2(image) * self.otf, s=self.shape)

    def adjoint(self, image):
        return irfft2(rfft2(image) * self.otf.conj(), s=self.shape)

    def iterate(self, total, iterations=40, backend="RL", background=0., floor=1e-12):
        total = np.asarray(total, dtype=float)
        if total.shape != self.shape or not np.all(np.isfinite(total)) or np.any(total < 0):
            raise ValueError("Total input must be finite/nonnegative on the complete canvas.")
        if backend not in ("RL", "SSRL"):
            raise ValueError("backend must be RL or SSRL")
        if iterations < 0 or int(iterations) != iterations:
            raise ValueError("iterations must be a nonnegative integer")
        if background < 0 or floor <= 0:
            raise ValueError("background must be nonnegative and floor positive")
        x = total.copy() if backend == "RL" else total - background
        disabled = np.zeros(self.shape, dtype=bool)
        for _ in range(iterations):
            prediction = self.forward(x)
            if backend == "RL":
                ratio = total / np.maximum(prediction, floor)
            else:
                prediction += background
                disabled |= prediction <= 0
                ratio = np.ones(self.shape)
                np.divide(total, prediction, out=ratio, where=~disabled)
            # Only protect the otherwise nonnegative multiplicative correction.
            # Signed source pixels and absorbing zeros are retained in SSRL.
            x *= np.maximum(self.adjoint(ratio), 0.)
        return x, {"disabled_ratio_pixels": int(disabled.sum())}

    def reconstruct(self, source, background, iterations=40, backend="RL"):
        x, diagnostics = self.iterate(self.embed(source, background), iterations,
                                      backend, background)
        if backend == "RL":
            x = x - background
        return x[self.crop].copy(), diagnostics

    def fider(self, source, background, calibration_flux, iterations=1000, eta=3e-14):
        if calibration_flux <= 0 or eta <= 0:
            raise ValueError("Calibration flux and eta must be positive.")
        beta = background / calibration_flux
        calibration, _ = self.iterate(self.psf_canvas + beta, iterations)
        response = calibration - beta
        response_ft = rfft2(np.fft.ifftshift(response))
        matching_ft = response_ft.conj() * self.otf / (abs(response_ft)**2 + eta)
        matching_ft[0, 0] = 1.
        total = self.embed(source, background)
        reconstruction, _ = self.iterate(total, iterations)
        output = irfft2(rfft2(reconstruction) * matching_ft, s=self.shape) - background
        residual = matching_ft * response_ft - self.otf
        diagnostics = {
            "beta": float(beta), "eta": eta, "iterations": iterations,
            "calibrated_response_sum": float(response.sum()),
            "maximum_matching_spectral_gain": float(abs(matching_ft).max()),
            "maximum_matching_residual": float(abs(residual).max()),
            "full_canvas_net_flux_change": float(output.sum() - source.sum()),
            "cropped_net_flux_change": float(output[self.crop].sum() - source.sum()),
        }
        return output[self.crop].copy(), diagnostics
