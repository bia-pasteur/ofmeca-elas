"""Config"""

from dataclasses import dataclass
from typing import List

@dataclass
class ElasticSimuParams:
    """Parameters for the elastic cell simulation
    """
    img_paths: List[str]
    masks_paths: List[str]
    t_end: int
    num_time_steps: int
    eta: float
    ym_for_t_nu: float
    t_for_ym_nu: float
    nu_for_ym_t: float
    traction_zone: List[float]
    youngs_modulus: List[float]
    nu: List[float]
    
@dataclass
class NoiseSimuParams:
    """Parameters for the noisy images
    """
    traction_zone: float
    ym: float
    nu: float
    eta: float
    im: str
    noise_stds: List[float]