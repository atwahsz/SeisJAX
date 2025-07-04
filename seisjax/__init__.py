"""SeisJAX: A JAX-powered library for high-performance seismic attribute computation."""

__version__ = "0.1.0"

from .attributes import (
    analytic_signal,
    envelope,
    instantaneous_phase,
    cosine_instantaneous_phase,
    instantaneous_frequency,
    instantaneous_bandwidth
)

__all__ = [
    "analytic_signal",
    "envelope",
    "instantaneous_phase",
    "cosine_instantaneous_phase",
    "instantaneous_frequency",
    "instantaneous_bandwidth",
] 