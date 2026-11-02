"""Artificial PoC for tail-run relative Gaussian and central diagonal folds.

No molecular loader, sampler, science runner or default circuit execution.
Reuses the frozen typed-event convention, append_central, precision Gaussian
and StrongBuilder.transition. Existing scientific emitters remain untouched.
"""
import hashlib,math

VERSION='tail_relative_gaussian_artificial_PoC_v1'


def _records(events,atoms):
    """Same order/sign/phase convention as existing build_event_sequence.

Its flattening is private local code, not an exported reusable function. Keep
the narrow validation/flatten here; do not patch the frozen emitter globally.
"""
    by_id={a.atom_id:a for a in atoms};flat=[];phase=0.
    if len(by_id)!=len(atoms):raise ValueError('DUPLICATE_ATOM')
    for event in events:
        if tuple(x.component_id for x in event.application_sequence)!=event.selected_component_ids:
            raise ValueError('Event order metadata mismatch')
        phase+=math.atan2(event.phase.imag,event.phase.real)
        for app in event.application_sequence:
            atom=by_id[app.component_id]
            if app.coefficient_sign!=(1 if atom.coefficient>0 else -1):raise ValueError('Stale event coefficient sign')
            if app.is_identity!=(atom.kind=='IDENTITY') or app.basis_id!=atom.basis_key:
                raise ValueError('Stale typed event metadata')
            angle=event.unsigned_rotation_angle if app.role=='rotation' else None
            if app.role=='product' and app.coefficient_sign<0:phase+=math.pi
            if atom.kind in ('IDENTITY','FRAME_Z'):flat.append((atom,angle))
            else:raise ValueError('FRAME_Z_ONLY_PoC')
    # Preserve the old emitter's accumulation order: event/product phases
    # first, then identity rotations in flattened application order.
    for i,(atom,angle) in enumerate(flat):
        if atom.kind=='IDENTITY':
            if angle is not None:phase-=angle
            flat[i]=None
    return flat,phase


def _runs(flat):
    groups=[];i=0
    while i<len(flat):
        if flat[i] is None:groups.append(None);i+=1;continue
        j=i+1
        while j<len(flat) and flat[j] is not None and flat[j][0].basis_key==flat[i][0].basis_key:j+=1
        groups.append(flat[i:j]);i=j
    return groups


def _block(atom,plan):
    from .c1_stage2_native import NativeBlock
    rotation=plan.frames[atom.frame_id]
    if hashlib.sha256(repr(rotation).encode()).hexdigest()!=atom.basis_key:
        raise ValueError('EXACT_DECLARED_BASIS_KEY')
    return NativeBlock(atom.frame_id,rotation,None)


def central_fold(qc,run,ancilla):
    """Candidate C: commute only diagonal factors INSIDE one exact-basis run.

Products of the same unsigned Z support cancel in pairs. Rotation angles use
fsum/exact-zero; no tolerance, modulo-angle or cross-support simplification.
Negative-product and event phases are retained by the caller, as before.
"""
    from .pr_dictionary_validation.circuits import append_central
    if not run:raise ValueError('NONEMPTY_DIAGONAL_RUN')
    key=run[0][0].basis_key;angles={};parity={}
    for atom,angle in run:
        if atom.kind!='FRAME_Z' or atom.basis_key!=key:raise ValueError('ONE_EXACT_DIAGONAL_RUN')
        support=atom.support
        if angle is None:parity[support]=1-parity.get(support,0)
        else:angles.setdefault(support,[]).append(angle)
    for support in sorted(set(angles)|set(parity)):
        value=math.fsum(angles.get(support,()))
        if value!=0.:append_central(qc,support,ancilla,value)
        if parity.get(support):append_central(qc,support,ancilla,None)


def tail_sequence(events,plan,builder,*,relative=True,fold_central=False):
    """Tail-only PoC; leave both external Gaussian endpoints in place.

Candidate A replaces ONLY adjacent different-basis run boundaries. Identity
records remain barriers. No head/tail, head-internal, law or event-order change.
fold_central is an artificial comparison of candidate C, not a second science
candidate. No real-H4 runner imports this module.
"""
    from qiskit import QuantumCircuit
    from .c1_h4_strong_baseline import StrongBuilder
    from .pr_dictionary_validation.circuits import append_central
    if not isinstance(builder,StrongBuilder):raise TypeError('EXISTING_PRECISE_STRONG_BUILDER_REQUIRED')
    flat,phase=_records(tuple(events),tuple(plan.atoms));groups=_runs(flat)
    qc=QuantumCircuit(plan.model.modes+1);anc=plan.model.modes;opened=False
    for i,run in enumerate(groups):
        if run is None:continue
        block=_block(run[0][0],plan)
        if any(_block(a,plan).rotation!=block.rotation for a,_ in run):raise ValueError('BASIS_KEY_COLLISION')
        basis=builder._basis(plan.model,block)
        if not opened:qc.compose(basis.inverse(),range(anc),inplace=True)
        if fold_central:central_fold(qc,run,anc)
        else:
            for atom,angle in run:append_central(qc,atom.support,anc,angle)
        following=groups[i+1] if i+1<len(groups) else None
        if relative and following is not None:
            qc.compose(builder.transition(plan.model,block,_block(following[0][0],plan)),range(anc),inplace=True)
            opened=True
        else:
            qc.compose(basis,range(anc),inplace=True);opened=False
    if phase:qc.p(phase,anc)
    qc.metadata=dict(PoC=VERSION,relative_tail=relative,central_fold=fold_central,
        head_tail_fusion=False,event_order_preserved=True,global_phase_fit=False)
    return qc
