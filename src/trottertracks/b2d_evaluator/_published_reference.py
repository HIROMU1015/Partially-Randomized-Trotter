"""Fixed upstream sequence constructors, preserving mathematical list order.

SOURCE 20d64c2fea1ed2d2188777df4f3d962ee94df486, product_formula.py blob
70d68691a1dcdedea5447aac4b8bbfa51141d4cb. Bodies unchanged; no science imports.
The old validator appends this list directly; our independent reference treats
it as the documented mathematical P K^n P^-1 product. No legacy equivalence
is assumed. Runtime comparison is mandatory before any performance comparison.
"""
from typing import List
from trottertracks.b2d_adapter._published import (
    symmetric_s2_sequence, inverse_s2_sequence,
    morales_2025_yp8m8_kernel_list, morales_2025_yp8m8_processor_sequence,
)
PFLabel = str


def _get_w_list(label):
    raise ValueError("This reference only supports published YP8m8.")


def _get_kernel_s2_sequence(num_w: PFLabel) -> List[float]:
    """Return the repeated kernel as an explicit sequence of S2 blocks."""
    if num_w == "8th(Morales-YP8m8)":
        return symmetric_s2_sequence(morales_2025_yp8m8_kernel_list())
    return symmetric_s2_sequence(_get_w_list(num_w))


def _get_processor_s2_sequence(num_w: PFLabel) -> List[float]:
    """Return the one-sided processor P, or an empty sequence."""
    if num_w == "8th(Morales-YP8m8)":
        return morales_2025_yp8m8_processor_sequence()
    return []


def _get_s2_sequence(num_w: PFLabel, *, kernel_steps: int = 1) -> List[float]:
    """Return the complete S2 sequence, including processing when required.

    A processed formula is returned as ``P K**kernel_steps P^-1`` so the
    processor is paid only once around all repeated kernel steps.
    """
    if kernel_steps < 1:
        raise ValueError("kernel_steps must be at least 1")
    kernel = _get_kernel_s2_sequence(num_w)
    processor = _get_processor_s2_sequence(num_w)
    return (
        processor
        + kernel * int(kernel_steps)
        + inverse_s2_sequence(processor)
    )
