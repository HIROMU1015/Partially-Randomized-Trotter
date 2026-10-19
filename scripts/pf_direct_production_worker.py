"""One owned PF pair, using saved inputs and four approved control probes."""
import argparse
from dataclasses import asdict
import gc
import importlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import pf_direct_compile_contract as c
import pf_direct_compile_adapter as a
import pf_direct_execution as e
from pf_direct_retry_support import array_record, scalar_value


def load_context(system, task, oracle, journal):
    cfg=system['config'];n,rank=c.SYSTEM_LAYOUTS[task['system']]
    sys.path.insert(0,str(Path(cfg['provider_root'])/'src'))
    m=importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')
    native=importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
    paths=cfg['mapping'];journal.take('saved_input_load')
    for name,h in system['input_hashes'].items():c.require(e.sha(name)==h,'INPUT_CHANGED:'+name)
    if task['system'] in ('H2_DF3','H4_DF10'):
        import numpy as np
        from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector
        from trotterlib.df_partial_s2 import df_hamiltonian_hash
        from trotterlib.pr2_s0_s1_validation import _state_hash
        receipt=c.read_json(paths['tol_receipt']);c.require(e.sha(paths['tol_snapshot'])==receipt['sha256'],'SAVED_SNAPSHOT_SHA')
        with np.load(paths['tol_snapshot'],allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
        observed={k:array_record(v,np) for k,v in arrays.items()}
        c.require(observed==receipt['arrays'],'SAVED_ARRAYS')
        ham=DFHamiltonian(float(arrays['constant']),arrays['one_body'],arrays['lambdas'],tuple(arrays['g_matrices']),receipt['hamiltonian_metadata'])
        sector=PhysicalSector.spin_sector(n_qubits=n,nelec_alpha=n//4,nelec_beta=n//4)
        full,state=arrays['state_vector'],arrays['sector_state_vector']
        c.require(np.array_equal(arrays['sector_basis_indices'],sector.basis_indices)
            and np.array_equal(full[sector.basis_indices],state),'SAVED_STATE_LIFT')
        md=dict(hamiltonian_hash=df_hamiltonian_hash(ham),state_hash=_state_hash(full,state))
    else:
        if task['system']=='H6_DF19':loader=m.load_h6_snapshot;key='h6_input_snapshot.npz'
        elif task['system']=='H8_DF28':
            loader=importlib.import_module('trottertracks.resource_applicability.h8_signal_preflight_port_v1').load_h8_snapshot;key='h8_input_snapshot.npz'
        else:
            loader=importlib.import_module('trottertracks.resource_applicability.h4_common_panel_port_v1').h4_loader;key='snapshot'
        rkey='snapshot_receipt' if key=='snapshot' else 'snapshot_receipt.json'
        dkey='df_receipt' if key=='snapshot' else 'df_receipt.json'
        ham,sector,full,state,md,_=loader(Path(paths[key]),c.read_json(paths[rkey]),c.read_json(paths[dkey]))
    binding=cfg['binding']
    c.require((ham.n_qubits,len(ham.lambdas))==(n,rank)
        and md['hamiltonian_hash']==binding['hamiltonian'] and md['state_hash']==binding['state'],'INPUT_SERIES_AND_STATE')
    _,qstate=m.checked_basis_bridge(ham,sector,full,state,1e-12)
    journal.take('saved_preparation_restore')
    p=int(task['prefix']);method=task['candidate_id'].split('_')[0]
    if task['system']=='H6_DF19':
        codec=importlib.import_module('trottertracks.resource_applicability.h6_roundoff_preparation_v1')
        preps,archive=codec.read_preparations(cfg['archive']['path'],cfg['archive']['sha256'])
        c.require(archive['binding']==cfg['archive']['binding'],'PREPARATION_ARCHIVE_BINDING')
        whole=preps['faithful:p19'];c.require(whole.hamiltonian_hash==binding['hamiltonian'],'FULL_PREPARATION_INPUT')
        blocks=whole.deterministic_blocks[:p+1]
        identity=dict(policy='SAVED_FULL_PREPARATION_PREFIX_VIEW',full_preparation_hash=whole.preparation_hash,
                      prefix=p,archive_sha256=cfg['archive']['sha256'])
    elif task['system']=='H8_DF28':
        import h8_strengthened_export as exported
        from qiskit.circuit.library import RZGate,XXPlusYYGate
        from trotterlib.df_rte_tail import describe_basis_change_operations
        from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec,DFDeterministicFragmentSpec
        export=c.read_json(paths['h8_preparation_export'])
        exported.validate_export(export,expected_hashes=cfg['expected_preparation_hashes'])
        class Permit:
            def require(self,scope):c.require(scope=='signal' and journal.used['saved_preparation_restore']==1,'SCOPED_RESTORE')
            def take(self,call):c.require(call=='full_preparation_decode' and journal.used['saved_preparation_restore']==1,'RESERVED_RESTORE_ONLY')
        def inspect(ops):
            described=describe_basis_change_operations(ops)
            return described,json.loads(json.dumps([asdict(v) for v in described]))
        whole=exported.restore_blocks(export,Permit(),gate_factory=lambda name,params:{'rz':RZGate,'xx_plus_yy':XXPlusYYGate}[name](*params),
            inspect_operations=inspect,one_type=DFDeterministicOneBodySpec,fragment_type=DFDeterministicFragmentSpec)
        c.require(all(float(ham.lambdas[i]).hex()==block.lam.hex() for i,block in enumerate(whole[1:])),'RESTORED_BLOCK_LAMBDAS')
        blocks=whole[:p+1];identity=dict(policy='SAVED_FULL_PREPARATION_PREFIX_VIEW',prefix=p,
            full_preparation_hash=export['faithful:p28']['preparation_hash'],export_sha256=e.sha(paths['h8_preparation_export']))
    else:
        # Same original native preparation from the accepted saved arrays; never SCF/DF/state solve.
        prep=(m._prepare if method=='B1' else m._prepare_discard)(ham,p)
        c.require(prep.num_system_qubits==n and prep.deterministic_fragment_indices==tuple(range(p))
            and prep.randomized_block_indices==() and prep.exact_rte_lambda_r==0.,'RESTORED_NATIVE_PREFIX')
        c.require(prep.constant_coefficient==ham.constant,'ORIGINAL_SCALAR')
        blocks=prep.deterministic_blocks
        identity=dict(policy='ORIGINAL_NATIVE_PREPARATION_FROM_SAVED_ARRAYS',prefix=p,
            preparation_hash=prep.preparation_hash,hamiltonian_hash=prep.hamiltonian_hash)
    c.require(len(blocks)==p+1 and tuple(b.original_fragment_index for b in blocks[1:])==tuple(range(p)), 'ORDERED_PREFIX_BLOCKS')
    row=oracle['record'];c.require(row['cell']['id']==task['candidate_id'] and row['cell']['q']==int(task['q'])
        and row['cell']['prefix']==p and row['cell']['formula']==task['PF'],'SAVED_ORACLE_IDENTITY')
    scalar=float(ham.constant);receipt=row.get('scalar_receipt') or oracle.get('saved_outer_scalar')
    if method=='B0I':
        scalar=scalar_value(scalar,receipt)
    import qiskit
    from trotterlib.rte_compiled_cost import compiler_settings_hash
    cp=cfg['compiler_options'];c.require(qiskit.__version__=='1.3.0','QISKIT_VERSION')
    compiler=m.CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,cp['seed_transpiler'],qiskit.__version__)
    c.require(compiler_settings_hash(compiler)==c.COMPILER,'LIVE_COMPILER_PROFILE')
    def control(evolution):
        np=m.np;low=np.concatenate((qstate,np.zeros_like(qstate)));high=np.concatenate((np.zeros_like(qstate),qstate))
        t=time.monotonic();left=m.simulate_statevector(evolution.circuit,low);right=m.simulate_statevector(evolution.circuit,high)
        z=complex(np.vdot(qstate,right[len(qstate):]));errors=[float(np.linalg.norm(left-low)),float(np.linalg.norm(right[:len(qstate)])),abs(float(np.linalg.norm(right))-1)]
        expected=[row['signals']['corrected']]
        if row.get('independent_forward_signals'):expected.append(row['independent_forward_signals']['corrected'])
        for value in expected:errors.append(abs(z-complex(value['real'],value['imag'])))
        for axis,value in (('cosine',z.real),('sine',z.imag)):
            probe=m.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=False,max_instructions=len(evolution.circuit.data)+4)
            errors.append(abs(m.hadamard_expectation(m.simulate_statevector(probe,low),n)-value))
        tolerance=1e-9 if n<=8 else 1e-8
        c.require(all(math.isfinite(x) and x>=0 for x in errors) and max(errors)<tolerance,'CONTROL_MISMATCH')
        return dict(status='PASS',probes=4,max_error=max(errors),errors=errors,signal=dict(real=z.real,imag=z.imag),
            oracle_source=oracle['source'],oracle_source_sha256=oracle['source_sha256'],new_classical_oracle=False,tolerance=tolerance,wall_seconds=time.monotonic()-t)
    journal.write('preparation_binding.json',dict(identity,n_qubits=n,DF_rank=rank,scalar_hex=scalar.hex(),
        scalar_receipt=receipt,hamiltonian_hash=md['hamiltonian_hash'],state_hash=md['state_hash'],
        block_order=[b.block_id for b in blocks],control='symmetric_directional',reference_rotated=False))
    return dict(system=task['system'],candidate_id=task['candidate_id'],q=int(task['q']),PF=task['PF'],prefix=p,
        system_qubits=n,DF_rank=rank,identity_gate_passed=True,scalar_policy='original_plus_own_tail_once',
        control_policy='symmetric_directional',independent_control_oracle_bound=True,native_instruction_limit=None,
        provider=m,native=native,compiler=compiler,deterministic_blocks=blocks,saved_scalar=scalar,control_oracle=control)


class ProductionKernel(a.NativeKernel):
    def build(self,task):
        x=self.context;bound=e.instruction_bound(self.native,x['deterministic_blocks'],int(task['q']),task['PF'])
        return self.native.build_deterministic_native(x['deterministic_blocks'],num_system_qubits=x['system_qubits'],
            T=.8,q=int(task['q']),formula=task['PF'],scalar=x['saved_scalar'],control_policy='symmetric_directional',max_instructions=bound)

    def compile_axis(self,evolution,axis,journal):
        journal.take('wrapper_build')
        wrapper=self.m.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=len(evolution.circuit.data)+4)
        before=self.m.canonical_qiskit_circuit_fingerprint(wrapper)
        journal.take('wrapper_compile');start=time.monotonic()
        result=self.m.transpile_and_measure_cost(wrapper,self.compiler,actual_circuit_fingerprint=before)
        return dict(axis=axis,RZ=result.rz_count,CX=result.cx_count,RZ_depth=result.rz_depth,CX_depth=result.cx_depth,
            depth=result.total_depth,size=result.circuit_size,pretranspile_gate_counts=dict(result.pretranspile_gate_counts),
            posttranspile_gate_counts=dict(result.posttranspile_gate_counts),compiler_hash=result.compiler_settings_hash,
            fingerprint_before=before,fingerprint_after=self.m.canonical_qiskit_circuit_fingerprint(wrapper),
            compiled_input_fingerprint=result.actual_circuit_fingerprint,compile_seconds=time.monotonic()-start,
            native_exit_code=None,reaped=False)


def run(spec_path,grant_path,system,cid,output,*,authorize=e.authorize):
    spec,grant=c.read_json(spec_path),c.read_json(grant_path)
    _,selected=authorize(spec,grant)
    pair=[r for r in selected if (r['system'],r['candidate_id'])==(system,cid)]
    c.require(len(pair)==2,'PAIR_NOT_APPROVED')
    c.require(Path(output)==Path(spec['output_root'])/(system+'__'+cid),'OWNED_OUTPUT_PATH')
    c.require(Path(sys.executable).absolute()==Path(spec['systems'][system]['python']).absolute(),'PINNED_PYTHON')
    for rel,h in spec['source_hashes'].items():c.require(e.sha(e.ROOT/rel)==h,'SOURCE_CHANGED')
    journal=a.Journal(output,c.call_caps(pair));start=time.monotonic()
    try:
        for name in ('tmp','numba','mpl','cache'):(Path(output)/name).mkdir()
        for key,name in (('TMPDIR','tmp'),('NUMBA_CACHE_DIR','numba'),('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','cache')):
            os.environ[key]=str(Path(output)/name)
        import tempfile
        tempfile.tempdir=os.environ['TMPDIR'];sys.dont_write_bytecode=True
        oracle=c.read_json(spec['oracles'])[system][cid]
        context=load_context(spec['systems'][system],pair[0],oracle,journal)
        kernel=ProductionKernel(context);kernel.restore(pair[0])
        from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
        with use_fingerprint_reuse():
            journal.take('native_evolution');t=time.monotonic();evolution=kernel.build(pair[0]);build_seconds=time.monotonic()-t
            journal.take('control_probe',4);journal.write('control.json',kernel.control(evolution))
            rows=[]
            for axis in c.AXES:
                row=kernel.compile_axis(evolution,axis,journal)
                row.update(task_id=system+':'+cid+':'+axis,system=system,candidate_id=cid,
                    SOURCE=spec['SOURCE'],observation_kind='DIRECT_COMPILED_FULL_WRAPPER',model_evaluated=False)
                journal.write('raw_'+axis+'.json',row);rows.append(row);gc.collect()
        c.require(journal.used==journal.caps,'EXACT_PAIR_CALLS')
        journal.write('result.json',dict(rows=rows,calls=journal.used,build_seconds=build_seconds,
            wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        return 0
    except BaseException as error:
        journal.write('STOP.json',dict(status='STOP',reason=type(error).__name__+':'+str(error),traceback=traceback.format_exc(),
            calls=journal.used,wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            automatic_retry=False,reservations_refunded=False))
        return 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','grant','system','candidate','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    raise SystemExit(run(args.spec,args.grant,args.system,args.candidate,args.output))
