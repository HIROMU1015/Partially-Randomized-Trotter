"""Interpreter compatibility only; never changes a scientific tolerance.

65536-bit bounded rationals need fewer than 20000 decimal digits. New Python
releases expose a configurable integer-string limit; early 3.11 builds do not.
The latter use the existing Bounds bit cap, output cap and time caps directly.
No result-dependent threshold or arithmetic fallback is used.
"""
import sys


def configure_integer_serialization():
    setter = getattr(sys, 'set_int_max_str_digits', None)
    if setter is not None:
        setter(25000)
