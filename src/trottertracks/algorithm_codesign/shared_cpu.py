"""Reuse shared source under a private CPU-only import namespace.

The normal trotterlib __init__ imports df_gpu_statevector, whose top-level
code scans CUDA paths and preloads CUDA libraries. BF-1 must not execute it.
This boundary leaves the shared package/source unchanged; no library copy
is made. Every unavailable GPU entry point raises rather than falling back.
"""
from __future__ import annotations
import importlib
from pathlib import Path
import sys
import types

PACKAGE = "_bf1_shared_cpu"


def forbidden_gpu(*args, **kwargs):
    raise RuntimeError("GPU operations are forbidden by the BF-1 contract")


def module(name):
    if PACKAGE not in sys.modules:
        source = Path(__file__).parents[2]/"trotterlib"
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(source)]
        package.__package__ = PACKAGE
        sys.modules[PACKAGE] = package
        stub = types.ModuleType(PACKAGE+".df_gpu_statevector")
        for attr in ("AerSimulator", "DFGPUParameterizedTemplate", "build_parameterized_gpu_template",
                     "run_parameterized_gpu_template", "simulate_statevector_gpu"):
            setattr(stub, attr, forbidden_gpu)
        sys.modules[stub.__name__] = stub
    return importlib.import_module(PACKAGE+"."+name)
