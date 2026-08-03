"""Feature extraction for the MetroPT-3 sensor stream."""

import numpy as np

# The seven analog channels. The eight digital channels are binary control
# signals; they are carried through for data-quality checks but kept out of the
# density feature vector, where binary flags would dominate the geometry.
ANALOG_COLUMNS: tuple[str, ...] = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
)

DIGITAL_COLUMNS: tuple[str, ...] = (
    "COMP",
    "DV_eletric",  # the dataset's own spelling
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
)

# Plausible operating envelopes, used only for data-quality range checks -- not
# for filtering or clipping the values fed to the sketch.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "TP2": (-1.0, 12.0),
    "TP3": (-1.0, 12.0),
    "H1": (-1.0, 12.0),
    "DV_pressure": (-1.0, 12.0),
    "Reservoirs": (-1.0, 12.0),
    "Oil_temperature": (-20.0, 120.0),
    "Motor_current": (-1.0, 15.0),
}


class WarmupStandardizer:
    """Standardizes features using statistics frozen after a warmup period.

    The Euclidean kernel is scale-sensitive, and the raw channels differ by
    orders of magnitude (bar vs degrees C vs amps), so without scaling the
    largest-magnitude channel would dictate the geometry.

    Statistics are frozen after warmup rather than updated continuously on
    purpose: a rolling scaler would slowly absorb a developing fault into its
    own notion of "normal" and mask the very drift we are trying to detect.
    """

    def __init__(self, n_features: int, warmup: int = 2000):
        self.warmup = warmup
        self.n_features = n_features
        self._buffer: list[np.ndarray] = []
        self.mean = np.zeros(n_features)
        self.scale = np.ones(n_features)
        self.fitted = False

    def observe(self, x: np.ndarray) -> None:
        if self.fitted:
            return
        self._buffer.append(x)
        if len(self._buffer) >= self.warmup:
            stacked = np.vstack(self._buffer)
            self.mean = stacked.mean(axis=0)
            scale = stacked.std(axis=0)
            # A channel that never moved during warmup would otherwise divide by 0.
            self.scale = np.where(scale < 1e-9, 1.0, scale)
            self.fitted = True
            self._buffer.clear()

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


def extract(record: dict, columns: tuple[str, ...] = ANALOG_COLUMNS) -> np.ndarray:
    """Pull the feature vector out of a decoded Kafka record.

    Missing or non-numeric fields become NaN; the quality checks report on them
    and the consumer drops those records rather than feeding NaN to the sketch.
    """
    values = np.empty(len(columns), dtype=float)
    for i, column in enumerate(columns):
        try:
            values[i] = float(record[column])
        except (KeyError, TypeError, ValueError):
            values[i] = np.nan
    return values
