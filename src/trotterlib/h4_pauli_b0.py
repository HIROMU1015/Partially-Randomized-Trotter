"""Matched deterministic Pauli B0/B0I bodies with the existing D2 control.

No input loading, signal evaluation, sampling or compilation lives here. Calling
the builders with scientific inputs requires separate execution permission.
"""
from dataclasses import replace


def matched_config(pr_config, *, alias="B0"):
    """Keep the saved PR partition explicitly; B0I is the same one candidate.

    r/K/seed remain comparison metadata and are inactive for this deterministic
    method. In particular, do not use h4_pilot_science.case_config, whose B0
    branch does not forward the PR prefix.
    """
    from .fixed_signal_evaluation import FixedSignalConfig

    if not isinstance(pr_config, FixedSignalConfig):
        raise TypeError("FixedSignalConfig required")
    if pr_config.hamiltonian_type != "pauli_jw" or pr_config.method != "PR":
        raise ValueError("A Pauli PR comparison config is required")
    if pr_config.L_D is None or pr_config.L_D < 1:
        raise ValueError("Explicit nonempty retained prefix required")
    if alias not in ("B0", "B0I"):
        raise ValueError("Only B0/B0I aliases are supported")
    return replace(pr_config, method="B0", cost_samples=1)


def build_body(hamiltonian, b0_config):
    """Reuse PauliSignalAdapter's S2 retained halves and constant phase once.

    No event generation occurs. Return the adapter as well as the body so its
    partition, identity policy and mathematical log_B_step=0 remain visible.
    """
    from .fixed_signal_evaluation import prepare_signal_adapter

    if (b0_config.hamiltonian_type != "pauli_jw" or b0_config.method != "B0"
            or b0_config.L_D is None or b0_config.L_D < 1
            or b0_config.cost_samples != 1):
        raise ValueError("Use the explicit matched B0 config")
    adapter = prepare_signal_adapter(hamiltonian, b0_config)
    # The seed argument is required by the shared API but is never consumed in
    # the nonrandomized trajectory branch. Do not call the pilot's seed_for.
    body = adapter.trajectory(q=b0_config.q, seed=b0_config.seed)
    return body, adapter


def central_wrappers(retained_body):
    """Lift one retained body once and build the two measured Hadamard axes.

    The caller establishes the saved body's partition/time/SOURCE identity.
    This function also accepts a validated D1 retained-only body. It neither
    substitutes D1's generic wrapper nor infers its compiled price.
    """
    from .fixed_signal_evaluation import hadamard_wrapper
    from .h4_d2_pauli_control import central_control_body

    controlled = central_control_body(retained_body)
    wrappers = {axis: hadamard_wrapper(controlled.body, axis,
                                      already_controlled=True)
                for axis in ("cosine", "sine")}
    return wrappers, controlled
