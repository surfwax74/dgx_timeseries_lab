"""L4 — multi-distribution noise (the headline complexity layer).

Seven distinct noise components, each driven by a different distribution,
designed to be stacked on a single channel to simulate the rich noise
budget seen on real spacecraft telemetry:

    GaussianNoise            white Gaussian sensor noise
    PinkNoise                1/f noise via FFT — thermal/instrument drift
    StudentTNoise            heavy-tailed — cosmic-ray impulses, ESD events
    QuantizationNoise        ADC bit-depth quantization
    CorrelatedGaussianNoise  multivariate Gaussian across a channel group
    PoissonBurstNoise        clustered impulses — radiation event bursts
    MultiplicativeGainNoise  slowly-varying AR(1) gain instability

Per the plan, the L4 components are the user-emphasized "multiple layers
of varying noise distributions" — they exist to give detectors a hard,
realistic noise floor to learn through.
"""

from __future__ import annotations

import numpy as np

from .component import Component, GenState


class GaussianNoise(Component):
    """White Gaussian noise — the classical sensor noise model."""

    kind = "gaussian_noise"

    def __init__(self, channel: str, std: float) -> None:
        self.channel = channel
        self.std = float(std)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        state.data[:, idx] += rng.normal(0.0, self.std, size=state.n_steps).astype(np.float32)


class PinkNoise(Component):
    """1/f noise generated via FFT scaling of white noise.

    Useful for thermal drift, instrument 1/f noise, and other low-frequency-
    biased disturbances. The resulting series has unit-variance-like behavior
    rescaled to the requested ``std``.
    """

    kind = "pink_noise"

    def __init__(self, channel: str, std: float) -> None:
        self.channel = channel
        self.std = float(std)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        n = state.n_steps
        white = rng.normal(0.0, 1.0, size=n)
        # Scale FFT by 1/sqrt(f) for pink. Avoid DC singularity.
        freqs = np.fft.rfftfreq(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(freqs > 0, 1.0 / np.sqrt(freqs), 0.0)
        pink = np.fft.irfft(np.fft.rfft(white) * scale, n=n)
        # Rescale to requested std.
        cur_std = float(pink.std())
        if cur_std > 1e-12:
            pink = pink * (self.std / cur_std)
        state.data[:, idx] += pink.astype(np.float32)


class StudentTNoise(Component):
    """Heavy-tailed Student-t noise. Captures cosmic-ray impulses, ESD
    transients, and other distributions where Gaussian underweights tails.

    ``df`` is the degrees of freedom; smaller → heavier tails. df=4 is a
    reasonable default for impulsive disturbances.
    """

    kind = "student_t_noise"

    def __init__(self, channel: str, scale: float, df: float = 4.0) -> None:
        self.channel = channel
        self.scale = float(scale)
        self.df = float(df)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        n = state.n_steps
        # t = Z / sqrt(W/df) where Z~N(0,1), W~Chi2(df)
        z = rng.normal(0.0, 1.0, size=n)
        chi2 = rng.chisquare(self.df, size=n)
        t_samples = z / np.sqrt(np.maximum(chi2 / self.df, 1e-12))
        state.data[:, idx] += (self.scale * t_samples).astype(np.float32)


class QuantizationNoise(Component):
    """N-bit ADC quantization. Replaces the channel value with its quantized
    representation (round-to-nearest), then clips to [-full_scale, full_scale]."""

    kind = "quantization_noise"

    def __init__(self, channel: str, n_bits: int = 12, full_scale: float = 5.0) -> None:
        self.channel = channel
        self.n_bits = int(n_bits)
        self.full_scale = float(full_scale)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        lsb = (2.0 * self.full_scale) / float(2**self.n_bits)
        x = state.data[:, idx]
        x_clipped = np.clip(x, -self.full_scale, self.full_scale)
        state.data[:, idx] = (np.round(x_clipped / lsb) * lsb).astype(np.float32)


class CorrelatedGaussianNoise(Component):
    """Multivariate Gaussian noise across a group of channels.

    ``covariance`` must be a square PSD matrix of side len(channels).
    Used to model shared sensor noise (e.g., a common voltage reference
    drifting across all power channels).
    """

    kind = "correlated_gaussian_noise"

    def __init__(self, channels: list[str], covariance: list[list[float]]) -> None:
        self.channels = list(channels)
        self.cov = np.asarray(covariance, dtype=np.float64)
        if self.cov.shape != (len(self.channels), len(self.channels)):
            raise ValueError(
                f"covariance must be {len(self.channels)}x{len(self.channels)}, "
                f"got {self.cov.shape}"
            )

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        indices = [state.channel_idx(c) for c in self.channels]
        samples = rng.multivariate_normal(
            mean=np.zeros(len(self.channels)),
            cov=self.cov,
            size=state.n_steps,
        ).astype(np.float32)
        for i, ch_idx in enumerate(indices):
            state.data[:, ch_idx] += samples[:, i]


class PoissonBurstNoise(Component):
    """Clustered impulse bursts modeling radiation events.

    A Poisson process generates burst onsets; each burst is a Gaussian
    sample of ``burst_size`` impulses around the onset.
    """

    kind = "poisson_burst_noise"

    def __init__(
        self,
        channel: str,
        event_rate_per_hour: float = 0.5,
        burst_size: int = 5,
        magnitude: float = 1.0,
    ) -> None:
        self.channel = channel
        self.event_rate = float(event_rate_per_hour)
        self.burst_size = int(burst_size)
        self.magnitude = float(magnitude)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        n = state.n_steps
        hours = n / (state.sample_rate_hz * 3600.0)
        n_events = int(rng.poisson(self.event_rate * hours))
        if n_events == 0 or n <= self.burst_size:
            return
        starts = rng.integers(0, n - self.burst_size, size=n_events)
        for s in starts:
            burst = rng.normal(0.0, self.magnitude, size=self.burst_size)
            state.data[s : s + self.burst_size, idx] += burst.astype(np.float32)


class MultiplicativeGainNoise(Component):
    """Slowly-varying multiplicative gain instability modeled as AR(1).

    Multiplies the channel value by (1 + g[t]) where g[t] is an AR(1) noise
    process with given std and time constant. Captures effects like
    amplifier gain drift that scale rather than add.
    """

    kind = "multiplicative_gain_noise"

    def __init__(
        self,
        channel: str,
        std: float = 0.01,
        time_constant_s: float = 60.0,
    ) -> None:
        self.channel = channel
        self.std = float(std)
        self.tau = float(time_constant_s)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        n = state.n_steps
        dt = 1.0 / state.sample_rate_hz
        alpha = float(np.exp(-dt / max(self.tau, dt)))
        innov_std = self.std * float(np.sqrt(max(1.0 - alpha * alpha, 1e-12)))
        innovations = rng.normal(0.0, innov_std, size=n)
        gain = np.empty(n, dtype=np.float32)
        gain[0] = float(rng.normal(0.0, self.std))
        for i in range(1, n):
            gain[i] = alpha * gain[i - 1] + innovations[i]
        state.data[:, idx] *= (1.0 + gain).astype(np.float32)
