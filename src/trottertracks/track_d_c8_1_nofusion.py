"""Disable Aer simulation fusion for saved C8-1 postchecks, keeping P0 intact.

Reuses official double backends and complete-column propagation. No local gate
matrix, compiler, tolerance, input state, phase or output normalization changes.
"""
from .track_d_state_action_acceleration import BatchedStateActions


class UnfusedStateActions(BatchedStateActions):
    def __init__(self, threads=1):
        super().__init__(threads)
        self.backend.set_options(fusion_enable=False)


def install(threads, *, existing_install):
    from . import c1_h6_state_action as primitive
    engine = existing_install(threads)
    engine.backend.set_options(fusion_enable=False)
    primitive.StateActions = UnfusedStateActions
    return engine
