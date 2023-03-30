from dataclasses import dataclass, field

import numpy as np

from gmphd_fusion.data import Track
from gmphd_fusion.gm import GaussianMixture
from gmphd_fusion.measurement_model import MeasurementModel
from gmphd_fusion.motion_models import MotionModel


@dataclass
class TestUseCase2D:
    surveillance_region: tuple[tuple[float, float], tuple[float, float]]
    clutter_rate: int
    detection_prob: float
    survival_prob: float
    prune_threshold: float
    merge_threshold: float
    max_components: int
    init_gm: GaussianMixture
    birth_gm: GaussianMixture
    motion_model: MotionModel
    measurement_model: MeasurementModel
    cpep_radius: float
    tracks_true: list[Track]
    target_weight_threshold: float = 0.5
    track_min_len: int = 10
    random_seed: int | None = None
    samples_per_test: int = 100
    # fuse gaussians (values) at time k (keys)
    fuse: dict[int, GaussianMixture] = field(default_factory=dict)

    @property
    def xmin(self) -> float:
        return self.surveillance_region[0][0]

    @property
    def xmax(self) -> float:
        return self.surveillance_region[1][0]

    @property
    def xlim(self) -> tuple[int, int]:
        return int(self.xmin * 1.1), int(self.xmax * 1.7)

    @property
    def ymin(self) -> float:
        return self.surveillance_region[0][1]

    @property
    def ymax(self) -> float:
        return self.surveillance_region[1][1]

    @property
    def ylim(self) -> tuple[int, int]:
        return int(self.ymin * 1.1), int(self.ymax * 1.1)

    @property
    def unif_min(self) -> np.ndarray:
        return np.array(self.surveillance_region).T[:, 0:1]

    @property
    def unif_max(self) -> np.ndarray:
        return np.array(self.surveillance_region).T[:, 1:2]

    @property
    def surveillance_area(self) -> float:
        return float(np.prod(np.diff(np.array(self.surveillance_region), axis=0)))

    @property
    def clutter_spatial_density(self) -> float:
        return self.clutter_rate / self.surveillance_area
