"""Q3-C/D: one past-history angle, one fresh canonical m17 trajectory/shot.

No event is drawn on import. The probability theorem uses the preserved ideal
native / IEEE / directed-interval / independent-bit model, not backend noise.
"""
from dataclasses import dataclass
from fractions import Fraction as F

from . import pr_reference_feedback_q1 as q1
from . import pr_reference_feedback_q3b as branch
from . import pr_reference_v6q_consumer as consumer
from . import pr_reference_v6q_producer as producer_module
from . import pr_reference_v6q_schedule as schedule


def _bprime_core(inputs, batch):
    from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
    from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
    from . import pr_reference_bprime as bp
    rp = inputs.prep.rte_preparation
    builder = QiskitDFRTEEventCircuitBuilder(basis_registry=rp.basis_registry)
    meta = batch.metadata
    tails = []
    for i in range(meta.q):
        request = DFRTEEventSequenceCircuitRequest(
            events=tuple(batch.events[i*meta.r:(i+1)*meta.r]),
            component_specs=rp.component_specs, controlled=True, ancilla_qubit=8,
            tail_id=rp.symbolic_tail.tail_id, tail_hash=rp.symbolic_tail.tail_hash,
            occurrence_rte_steps=meta.r)
        tail, _ = bp.rewrite_rte(builder.build_sequence(request).circuit)
        tails.append(tail)
    return bp.pilot.symmetric_fused(tails)


@dataclass(frozen=True)
class AdaptiveReadout:
    arm: str
    prefix: q1.Prefix
    feedback: q1.FeedbackAngle

    def __post_init__(self):
        if (self.arm not in ('Bprime', 'Reference_B1')
                or not isinstance(self.prefix, q1.Prefix)
                or self.prefix.saved_arm != self.arm
                or self.feedback.bound_prefix is not self.prefix
                or self.feedback.arithmetic_error_upper > branch.ARITHMETIC_ANGLE_LIMIT):
            raise ValueError('A matching fixed-prefix/feedback binding is required')

    @classmethod
    def from_counts(cls, arm, history):
        prefix = branch.prefix_from_counts(history, saved_arm=arm)
        return cls(arm, prefix, branch.feedback_from_prefix(prefix))

    def physical_shot(self, inputs, *, producer):
        """Fresh producer call; compile Q itself; never relabel an X/Y circuit.

        Calling this on scientific inputs builds/compiles a full m17 circuit.
        Q3-C/D establishes its SOURCE semantics without executing that huge call.
        The caller executes exactly 340 fresh requests and supplies actual counts.
        """
        if not isinstance(producer, producer_module.CanonicalFiniteBitProducer):
            raise TypeError('Use the fixed canonical producer with a continuing bit source')
        if (not isinstance(inputs, consumer.FixedInputs)
                or inputs.cache_sha256 != consumer.CACHE_SHA256
                or inputs.common_data['preps'][3] is not inputs.prep):
            raise ValueError('Use the bound fixed-cache context')
        runtime = consumer._runtime()
        if runtime.common() is not inputs.common_data:
            raise ValueError('Assembly no longer uses the bound preparation')
        meta, _ = schedule.bind_round(17, q=131072, r=32, bits=88,
                                      mode='exact_binary_m17')
        batch = producer(meta, inputs.prep)  # One fresh call for this physical shot.
        consumer._validate_batch(inputs, batch, meta)
        if len(batch.events) != meta.q*meta.r:
            raise ValueError('Event count does not match the saved m17')
        if self.arm == 'Reference_B1':
            result = schedule.assemble_reference_for_saved_round(17,
                q=meta.q, r=meta.r, bits=meta.bits, cfg=batch.cfg,
                dist=batch.dist, events=batch.events, mode='exact_binary_m17')
            core = result['core']
        else:
            core = _bprime_core(inputs, batch)
        raw = q1.feedback_readout(core, self.feedback, family=self.arm)
        native = (consumer._compile_and_reset(raw, runtime) if self.arm == 'Reference_B1'
                  else runtime.compile_reference(raw))
        return dict(circuit=native, axis='Q', metadata=meta,
            feedback_parameter_hex=self.feedback.parameter.hex(),
            producer_source=batch.producer_source,
            model='fixed ideal-native gates/reset/measurement and fresh IID bits')

    def decode(self, plus):
        # P(-binary_parameter) -> RZ(-binary_parameter) preserves the angle
        # exactly in Compiler A. General H/core errors remain in the saved raw
        # disk, not in this additional angular charge. See NATIVE_GUARANTEE.md.
        result = q1.decode_counts(self.prefix, self.feedback, plus,
            q1.saved_q0_model(self.arm), additional_feedback_error=F(0), n=340)
        result['Q3CD_evidence_scope'] = 'fixed model; see Q3-C/D evidence, no backend-success claim'
        return result
