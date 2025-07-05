"""SeisJAX: A JAX-powered library for high-performance seismic attribute computation."""

__version__ = "0.1.0"

# Import from attributes module (complex trace attributes)
from .attributes import (
    analytic_signal,
    envelope,
    instantaneous_phase,
    cosine_instantaneous_phase,
    instantaneous_frequency,
    instantaneous_bandwidth
)

# Import from coherence module
from .coherence import (
    semblance,
    eigenstructure_coherence,
    energy_ratio_coherence,
    c3_coherence,
    variance_coherence
)

# Import from curvature module
from .curvature import (
    mean_curvature,
    gaussian_curvature,
    principal_curvatures,
    maximum_curvature,
    minimum_curvature,
    curvedness,
    shape_index,
    dip_curvature,
    strike_curvature,
    contour_curvature,
    profile_curvature,
    planform_curvature,
    most_positive_curvature,
    most_negative_curvature,
    total_curvature,
    tangential_curvature
)

# Import from spectral module
from .spectral import (
    power_spectrum,
    dominant_frequency,
    spectral_centroid,
    spectral_bandwidth,
    spectral_slope,
    spectral_rolloff,
    spectral_flux,
    spectral_flatness,
    spectral_contrast,
    zero_crossing_rate,
    spectral_energy,
    peak_frequency,
    spectral_decomposition,
    instantaneous_amplitude_spectrum,
    time_frequency_decomposition,
    rgb_frequency_blend
)

# Import from geometric module
from .geometric import (
    dip_magnitude,
    dip_azimuth,
    strike_angle,
    true_dip,
    apparent_dip,
    reflection_intensity,
    relative_acoustic_impedance,
    convergence,
    parallelism,
    continuity,
    fault_likelihood,
    edge_detection,
    texture_energy,
    local_structural_entropy
)

# Import utility functions
from . import utils

__all__ = [
    # Complex trace attributes
    "analytic_signal",
    "envelope",
    "instantaneous_phase",
    "cosine_instantaneous_phase",
    "instantaneous_frequency",
    "instantaneous_bandwidth",
    
    # Coherence attributes
    "semblance",
    "eigenstructure_coherence",
    "energy_ratio_coherence",
    "c3_coherence",
    "variance_coherence",
    
    # Curvature attributes
    "mean_curvature",
    "gaussian_curvature",
    "principal_curvatures",
    "maximum_curvature",
    "minimum_curvature",
    "curvedness",
    "shape_index",
    "dip_curvature",
    "strike_curvature",
    "contour_curvature",
    "profile_curvature",
    "planform_curvature",
    "most_positive_curvature",
    "most_negative_curvature",
    "total_curvature",
    "tangential_curvature",
    
    # Spectral attributes
    "power_spectrum",
    "dominant_frequency",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_slope",
    "spectral_rolloff",
    "spectral_flux",
    "spectral_flatness",
    "spectral_contrast",
    "zero_crossing_rate",
    "spectral_energy",
    "peak_frequency",
    "spectral_decomposition",
    "instantaneous_amplitude_spectrum",
    "time_frequency_decomposition",
    "rgb_frequency_blend",
    
    # Geometric attributes
    "dip_magnitude",
    "dip_azimuth",
    "strike_angle",
    "true_dip",
    "apparent_dip",
    "reflection_intensity",
    "relative_acoustic_impedance",
    "convergence",
    "parallelism",
    "continuity",
    "fault_likelihood",
    "edge_detection",
    "texture_energy",
    "local_structural_entropy",
] 