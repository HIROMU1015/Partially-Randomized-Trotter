"""Artificial inputs only. No saved scientific data or molecule loading."""
from copy import deepcopy
from fractions import Fraction as F
from . import c_direct_round_signal as m
from . import c_fresh_iid_calibration as fresh
from . import c_iid_calibration_integration as integration
from . import c_pr_aware_df_fixed_certificate as fixed
from . import c_pr_aware_df_zsxx_enclosure as z


def fixture(m_round=0, n=2, scale=1, angle=.25, index=0):
    from flint import ctx,acb,acb_mat
    from trottertracks.c_pr_aware_df_fixed_car import pack
    from trottertracks.c_b1_b2_scope_integration import identity
    from trottertracks.c_pr_aware_df_stage_c2 import choice_grid
    context = fresh.make_context(source_id='d'*64,identity=identity(),atom_ids=['INVENTED'],
        table=choice_grid([1.]),event_angle_bits={'INVENTED':angle.hex()})
    sample = fresh.sample_from_integers(context,dataset_id='INVENTED-direct',sample_index=index,
        integers=[[0]*64 for _ in range(1 << m_round)],unit='round',m=m_round,axis='paired_XY')
    d = 1 << (n-1)
    target = dict(target=integration.population.TARGET,candidate=context['identity']['candidate'],
        invented=True,system_dimension=d,scale=str(scale),angle=angle.hex())
    state = m.bind_state([['1','0']]+[['0','0'] for _ in range(d-1)],
        source_id=context['source_id'],target_id=fresh.digest(target),provenance='INVENTED exact |0>')
    parts = []
    with z._BALL_LOCK,ctx.workprec(192):
        phase = scale*(acb(0,1)*z._real(angle)).exp()
        for i,k in enumerate(sample['kernels']):
            record = fixed.identity(n)
            record['gates'] = [dict(name='rz',qubits=[n-1],params_hex=[angle.hex()])]
            raw = integration.bind_operator(record,sample,context,i,role='raw')
            compiled = integration.bind_operator(record,sample,context,i,role='compiled',raw_id=raw['record_id'])
            ref = dict(status='VALIDATED_LITERAL_REFERENCE_BLOCKS',target=integration.population.TARGET,
                binding=dict(target=target,target_id=fresh.digest(target),word=k['word'],word_id=k['word_id']),
                system_dimension=d,precision_bits=192,
                blocks=[dict(indices=[j],entries=pack(acb_mat([[phase]]))) for j in range(d)])
            parts.append(dict(raw=raw,compiled=compiled,reference=ref,compile_status='SUCCESS',
                actual_inverse=m.bind_inverse(fixed.adjoint(record),compiled,sample,context,i)))
    return dict(sample=sample,context=context,target=target,state=state,occurrences=parts,shells=m.ideal_shells(n))


def long_fixture(m_round=12):
    """Identical invented words, with independent-model occurrence records.

    Use a once-bound prototype, avoiding the old per-word binding adapter's
    repeated replay of the complete long-round sample during fixture creation.
    This constructs no RNG draws, molecule, Qiskit circuit or compiler output.
    """
    args=fixture();context=args['context'];template=args['occurrences'][0]
    sample=fresh.sample_from_integers(context,dataset_id='INVENTED-long-direct',sample_index=0,
        integers=[[0]*64 for _ in range(1 << m_round)],unit='round',m=m_round,axis='paired_XY')
    parts=[]
    for i in range(1 << m_round):
        p=deepcopy(template)
        for role in ('raw','compiled'):
            p[role].update(m=m_round,calibration_binding=m._binding(sample,i))
        p['actual_inverse']['calibration_binding']=m._binding(sample,i)
        parts.append(p)
    return dict(args,sample=sample,occurrences=parts)
