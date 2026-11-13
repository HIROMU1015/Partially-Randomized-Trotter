"""Supplied unitary -> explicitly branched Hermitian generator diagnostics.

No PF/H builder, molecular loader, branch oracle, RPE/N/RMSE solver or search.
Numerical leaves on real inputs require a separately approved science scope.
"""
from copy import deepcopy
from fractions import Fraction
import json
import math

KINDS=('FULL_PF','TRUNCATED_PF','PR_HEAD','PR_EXACT_TAIL')


def _number(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('Finite numeric scalar required')
    return float(value)


def _exact(value):
    return Fraction(value['numerator'],value['denominator']) if isinstance(value,dict) else Fraction(value)


def _finite_payload(value):
    if isinstance(value,dict):return {k:_finite_payload(v) for k,v in value.items()}
    if isinstance(value,list):return [_finite_payload(v) for v in value]
    return None if isinstance(value,float) and not math.isfinite(value) else value


def common_step_schedule(rows):
    """Metadata only. step_key is producer supplied, not a new candidate ID.

    Include actual delta/H/sector/split/formula/order/scalar/recipe/SOURCE in
    every key. R/K may be omitted ONLY for an explicitly PR_EXACT_TAIL key.
    """
    if not rows or [r['m'] for r in rows]!=list(range(len(rows))):
        raise ValueError('Complete consecutive round metadata required')
    required=('kind','binding','formula','prefix','scalar','source','actual_delta')
    key=rows[0]['step_key']
    if any(any(k not in r['step_key'] or r['step_key'][k] is None for k in required)
            or r['step_key']['kind'] not in KINDS or not r['step_key']['source']
            or _exact(r['step_key']['actual_delta'])<=0 for r in rows):
        raise ValueError('Explicit deterministic unitary step descriptor required')
    same=lambda a,b:json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
    equal=all(same(r['step_key'],key) for r in rows)
    grid=all(type(r['q_PF']) is int and r['q_PF']>0 and _exact(r['T'])==r['q_PF']*_exact(r['step_key']['actual_delta']) for r in rows)
    return dict(status='COMMON_STEP_DECLARED' if equal and grid else 'UNKNOWN_NO_COMMON_STEP',
        same_declared_step=equal,integer_power_times=grid,operator_equality_numerically_checked=False,
        common_step_qualified=False,global_RMSE_bound=None)


def _groups(energies,weights,tolerance):
    groups=[]
    for i,e in enumerate(energies):
        if not groups or e-energies[groups[-1][0]]>tolerance:groups.append([i])
        else:groups[-1].append(i)
    return [dict(indices=g,minimum_energy=float(energies[g[0]]),maximum_energy=float(energies[g[-1]]),
        weight=float(sum(weights[i] for i in g)),multiplicity=len(g)) for g in groups]


def evaluate(unitary,delta,*,state,operator,reference,branch,tolerances,reference_hamiltonian=None):
    """Explicit numerical leaf, with UNKNOWN and observed diagnostics on failure.

    reference.energy is supplied, never inferred/replaced by a ground solver.
    H_eff minimum shift and spectral weights are diagnostics, not RPE bias.
    Method adoption is workflow metadata, never an individual qualification.
    """
    result=dict(status='UNKNOWN',operator=deepcopy(operator),reference=deepcopy(reference),
        branch=deepcopy(branch),tolerances=deepcopy(tolerances),diagnostics={},H_eff=None,spectrum=None,
        reference_comparison=None,spectral_shift_diagnostics=None,rpe_bias_ha=None,
        sufficient_N=None,global_RMSE_bound=None,G=None,
        scientific_method_status=('ADOPTED_MAIN_DETERMINISTIC_PF_ERROR_ROUTE'
            if operator.get('kind') in ('FULL_PF','TRUNCATED_PF') else
            'NOT_ADOPTED_FOR_TOTAL_PR_ERROR' if operator.get('kind') in ('PR_HEAD','PR_EXACT_TAIL') else 'NOT_APPLICABLE'),
        physical_alias_qualified=False,ground_identification_qualified=False,numerical_error_certified=False,
        renormalized=False,unitary_repaired=False)
    if operator.get('kind') not in KINDS:
        result.update(status='NOT_APPLICABLE',reason='A deterministic unitary is required; finite-K corrected/raw are excluded')
        return result
    import numpy as np
    from scipy.linalg import expm,schur,eigh
    def require(condition,reason):
        if not condition:raise ValueError(reason)
    diag=result['diagnostics']
    try:
        require(bool(operator.get('source')) and bool(operator.get('identity')),'Operator SOURCE/identity missing')
        require(bool(reference.get('source')) and bool(reference.get('definition')),'Declared reference energy SOURCE/definition missing')
        require(operator.get('target_hamiltonian')==reference.get('hamiltonian') and reference.get('hamiltonian') is not None,
            'Original target Hamiltonian differs from reference')
        require(operator.get('retained_hamiltonian') is not None,'Retained/operator Hamiltonian descriptor missing')
        require(operator.get('scalar_policy')=='INCLUDED_ONCE_AT_STEP_TIME',
            'Supply the unitary including its declared scalar once at actual step time')
        if operator['kind']=='FULL_PF':
            require(operator['retained_hamiltonian']==operator['target_hamiltonian'],'Full PF cannot declare a truncated H')
        require(bool(branch.get('source')) and bool(tolerances.get('source')),'Branch/numerical policy SOURCE missing')
        delta=_number(delta);require(delta>0,'Positive actual step width required')
        energy_ref=_number(reference['energy'])
        tol={k:_number(tolerances[k]) for k in ('unitarity','reconstruction','generator_hermiticity_ha','phase_guard_rad','energy_cluster_ha')}
        require(all(v>0 for v in tol.values()),'Positive numerical tolerances required')
        u=np.asarray(unitary,dtype=np.complex128);psi=np.asarray(state,dtype=np.complex128)
        require(u.ndim==2 and u.shape[0]==u.shape[1] and u.shape[0]>0,'Finite square step required')
        require(psi.shape==(len(u),) and np.isfinite(u).all() and np.isfinite(psi).all(),'Finite supplied unitary/state required')
        diag['input_state_norm_error']=abs(float(np.linalg.norm(psi))-1.)
        require(diag['input_state_norm_error']<=1e-12,'Input state norm gate; no renormalization')
        eye=np.eye(len(u),dtype=complex)
        diag['unitarity_defect_fro']=float(np.linalg.norm(u.conj().T@u-eye,'fro'))
        require(diag['unitarity_defect_fro']<=tol['unitarity'],'Nonunitary or numerically unresolved step')
        triangular,vectors=schur(u,output='complex',check_finite=True)
        values=np.diag(triangular)
        diag.update(schur_offdiagonal_fro=float(np.linalg.norm(triangular-np.diag(values),'fro')),
            eigenvalue_modulus_error=float(np.max(abs(abs(values)-1.))),
            eigenvector_gram_defect_fro=float(np.linalg.norm(vectors.conj().T@vectors-eye,'fro')),
            unitary_eigen_residual_fro=float(np.linalg.norm(u@vectors-vectors*values,'fro')))
        require(max(diag[k] for k in ('schur_offdiagonal_fro','eigenvalue_modulus_error',
            'eigenvector_gram_defect_fro','unitary_eigen_residual_fro'))<=tol['reconstruction'],
            'Schur eigenvector/numerical normality gate')
        principal=np.angle(values)
        diag['principal_cut_clearance_rad']=float(np.min(math.pi-abs(principal)))
        mode=branch.get('mode')
        if mode=='PRINCIPAL':
            center=0.;lower,upper=-math.pi/delta,math.pi/delta
            result['branch_status']='CONVENTION_ONLY_ALIAS_EVIDENCE_NOT_SUPPLIED'
        elif mode=='ENERGY_WINDOW':
            require(branch.get('purpose')=='EFFECTIVE_SPECTRUM_WINDOW','A spectral branch window is not the RPE decoder prior')
            lower,upper=map(_number,branch['window'])
            require(lower<upper and (upper-lower)*delta<2*math.pi,'Branch window must be narrower than one alias period')
            center=lower+(upper-lower)/2
            result['branch_status']='UNIQUE_IN_SUPPLIED_WINDOW_ASSUMPTION_NOT_PROVED'
        else:raise ValueError('Explicit PRINCIPAL or ENERGY_WINDOW branch required')
        result['branch_center_ha']=center
        # A window is caller supplied; reference energy never selects the log.
        phase=np.angle(np.exp(1j*center*delta)*values)
        diag['selected_cut_clearance_rad']=float(np.min(math.pi-abs(phase)))
        require(diag['selected_cut_clearance_rad']>tol['phase_guard_rad'],'Branch cut too close; unresolved logarithm')
        energies=center-phase/delta
        require(np.isfinite(energies).all(),'Nonfinite branched energies')
        if mode=='ENERGY_WINDOW':
            diag['window_edge_clearance_rad']=float(np.min(np.minimum(energies-lower,upper-energies))*delta)
            require(diag['window_edge_clearance_rad']>tol['phase_guard_rad'],'Spectrum outside or too near supplied window boundary')
        order=np.argsort(energies,kind='stable');energies=energies[order];vectors=vectors[:,order]
        values=values[order];principal=principal[order]
        heff=(vectors*energies)@vectors.conj().T
        diag['H_eff_hermiticity_defect_fro']=float(np.linalg.norm(heff-heff.conj().T,'fro'))
        diag['exponential_reconstruction_fro']=float(np.linalg.norm(expm(-1j*delta*heff)-u,'fro'))
        require(diag['H_eff_hermiticity_defect_fro']<=tol['generator_hermiticity_ha']
            and diag['exponential_reconstruction_fro']<=tol['reconstruction'],
            'Hermitian-generator exponential reconstruction gate')
        weights=abs(vectors.conj().T@psi)**2
        diag['weight_sum_error']=abs(float(np.sum(weights))-1.)
        require(diag['weight_sum_error']<=tol['reconstruction'],'Spectral weight sum unresolved; no renormalization')
        groups=_groups(energies,weights,tol['energy_cluster_ha'])
        encode=lambda a:dict(real=a.real.tolist(),imag=a.imag.tolist())
        result.update(status='EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE',H_eff=encode(heff),
            spectrum=dict(energies_ha=energies.tolist(),eigenvectors=encode(vectors),unitary_eigenvalues=encode(values),
                branch_integers=[int(round(float((e+p/delta)*delta/(2*math.pi)))) for e,p in zip(energies,principal)],
                state_weights=weights.tolist(),energy_groups=groups,
                minimum_energy_group_weight=groups[0]['weight'],maximum_group_weight=max(g['weight'] for g in groups),
                degenerate_individual_weights_basis_dependent=True),
            spectral_shift_diagnostics=dict(minimum_branched_energy_minus_declared_reference_ha=float(energies[0]-energy_ref),
                scope='BRANCH_DEPENDENT_SPECTRAL_DIAGNOSTIC_NOT_RPE_BIAS',
                reference_energy_not_used_to_choose_branch=True))
        if reference_hamiltonian is not None:
            h=np.asarray(reference_hamiltonian,dtype=complex)
            require(h.shape==u.shape and np.isfinite(h).all(),'Reference H shape/finite mismatch')
            diag['reference_H_hermiticity_defect_fro']=float(np.linalg.norm(h-h.conj().T,'fro'))
            require(diag['reference_H_hermiticity_defect_fro']<=tol['generator_hermiticity_ha'],'Reference H is not Hermitian')
            evals,basis=eigh(h,check_finite=True)
            reference_weights=abs(basis.conj().T@psi)**2
            reference_groups=_groups(evals,reference_weights,tol['energy_cluster_ha'])
            overlaps=abs(basis.conj().T@vectors)**2
            diag.update(reference_eigen_residual_fro=float(np.linalg.norm(h@basis-basis*evals,'fro')),
                reference_eigenvector_gram_defect_fro=float(np.linalg.norm(basis.conj().T@basis-eye,'fro')),
                reference_weight_sum_error=abs(float(np.sum(reference_weights))-1.))
            require(diag['reference_eigen_residual_fro']<=tol['generator_hermiticity_ha']
                and max(diag['reference_eigenvector_gram_defect_fro'],diag['reference_weight_sum_error'])<=tol['reconstruction'],
                'Reference eigensystem/weight numerical gate')
            diag['delta_reference_H_norm']=float(delta*np.linalg.norm(h,2))
            diag['reference_norm_less_than_pi']=diag['delta_reference_H_norm']<math.pi
            diag['delta_centered_reference_H_norm']=float(delta*np.linalg.norm(h-center*eye,2))
            diag['centered_reference_norm_less_than_pi']=diag['delta_centered_reference_H_norm']<math.pi
            # This is not a proof that the PF log follows the original H branch.
            result['reference_comparison']=dict(eigenvalues_ha=evals.tolist(),
                eigenvectors=encode(basis),state_weights=reference_weights.tolist(),energy_groups=reference_groups,
                squared_eigenbasis_overlaps=overlaps.tolist(),
                reference_lowest_group_to_effective_groups_trace_overlap=[
                    float(np.sum(overlaps[np.ix_(reference_groups[0]['indices'],g['indices'])])) for g in groups],
                trace_overlap_is_not_a_normalized_state_probability=True,
                matched_effective_group=None,
                degenerate_basis_correspondence_not_unique=True,
                reference_ground_energy_replaces_declared_reference=False,
                H_eff_minus_original_H_operator_norm=float(np.linalg.norm(heff-h,2)))
            result['spectral_shift_diagnostics'].update(
                minimum_branched_energy_minus_reference_matrix_minimum_ha=float(energies[0]-evals[0]),
                declared_reference_minus_matrix_minimum_ha=float(energy_ref-evals[0]))
        return _finite_payload(result)
    except (ValueError,TypeError,KeyError,OverflowError,FloatingPointError,np.linalg.LinAlgError) as error:
        # Invalid inputs or numerical ambiguity never become a bias/RMSE PASS.
        result.update(status='UNKNOWN',reason=str(error),error_type=type(error).__name__,H_eff=None,
            spectrum=None,reference_comparison=None,spectral_shift_diagnostics=None)
        return _finite_payload(result)
