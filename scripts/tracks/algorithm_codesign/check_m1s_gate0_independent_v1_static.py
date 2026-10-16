#!/usr/bin/env python3
"""Metadata/AST checks only. No reference kernels, NPZ decode or quantum test."""
import argparse
import ast
import copy
from contextlib import redirect_stdout
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
PACKAGE = 'src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1/'


class BlockNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','scipy','openfermion','qiskit'} or fullname.endswith(('.reference','.execution')) or '.m1s_gate0_v5' in fullname or '.m1s_gate0_v4' in fullname:
            raise RuntimeError('Numerical/helper import forbidden in static checks: '+fullname)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract')
    parser.add_argument('--write-report')
    args = parser.parse_args()
    sys.meta_path.insert(0,BlockNumericalImports())
    accesses = {'NPZ':0,'kernel_calls':0}
    def audit(event, arguments):
        if event=='open' and isinstance(arguments[0],(str,bytes)) and str(arguments[0]).lower().endswith(('.npz', ".npz'")):
            accesses['NPZ'] += 1
            raise RuntimeError('NPZ open forbidden in static checks')
    sys.addaudithook(audit)
    def profile(frame,event,arg):
        if event=='call' and frame.f_code.co_filename.endswith(('/m1s_gate0_independent_v1/reference.py','/m1s_gate0_independent_v1/execution.py')):
            accesses['kernel_calls'] += 1
            raise RuntimeError('No reference/execution calls in static preparation')
    sys.setprofile(profile)
    sys.path.insert(0,str(ROOT/'src'))
    from trottertracks.algorithm_codesign.m1s_gate0_independent_v1 import control as c
    checks=[]
    def check(name,ok):
        checks.append({'name':name,'status':'PASS' if ok else 'FAIL'})
        if not ok:
            raise AssertionError(name)
    def parse(p):
        tree=ast.parse((ROOT/p).read_bytes(),filename=p)
        check('syntax:'+p,True)
        return tree
    def functions(t):
        return {n.name:n for n in ast.walk(t) if isinstance(n,ast.FunctionDef)}
    def text(n):
        return ast.unparse(n)
    trees={p:parse(p) for p in c.SOURCE_PATHS}
    rf,cf,ef=(functions(trees[PACKAGE+p]) for p in ('reference.py','control.py','execution.py'))
    for p,t in trees.items():
        for node in ast.walk(t):
            if isinstance(node,ast.Import):
                check('stdlib_import:'+p+':'+','.join(a.name for a in node.names),all(a.name.split('.')[0] in {'bisect','decimal','fractions','ast','hashlib','importlib','io','json','math','struct','zipfile','argparse','os','pathlib','re','subprocess','sys','resource','signal','time'} for a in node.names))
            elif isinstance(node,ast.ImportFrom):
                module=node.module or ''
                check('no_old_helper_import:'+p+':'+module,not any(s in module for s in ('m1s_gate0_v5','m1s_gate0_v4','trotterlib')))
    original=ast.parse(subprocess.check_output(['git','-C',str(ROOT),'show','d6d35cfb79b97f9cb218c9169b427aacf6a4a334:'+PACKAGE+'reference.py']))
    old=functions(original)
    for name in ('decode','dyadic','one_body','square','contracted_square','scale','combine','scalar_matrix','hermitian','ladder_correspondence','trace_pauli'):
        check('draft_kernel_reused_AST:'+name,ast.dump(rf[name],include_attributes=False)==ast.dump(old[name],include_attributes=False))
    check('permit_before_reference_initialization',text(rf['__init__'].body[0])=='require_execution(authorization)')
    check('permit_before_execution_IO',text(ef['run_authorized'].body[0])=='require_execution(permit)')
    build=text(rf['build']);compare=text(rf['compare_saved']);run=text(ef['run_authorized'])
    check('build_accepts_only_raw_input',text(rf['build'].args)=='self, input_raw')
    check('build_no_comparator_or_saved_fields',all(s not in build for s in ('comparator_raw','canonical_table','lambda_R\']','saved[')))
    check('all_reference_splits_before_freeze',build.index('for count in (3, 6, 9)') < build.index("phase='INDEPENDENT_REFERENCE_FROZEN'"))
    check('independent_snapshot_hash_before_comparison','independent_reference_sha256' in build and 'hashlib.sha256(raw)' in build)
    check('comparator_parse_only_in_final_stage',"self.progress['phase'] != 'INDEPENDENT_REFERENCE_FROZEN'" in compare and compare.index('INDEPENDENT_REFERENCE_FROZEN') < compare.index('saved = read_json(comparator_raw)'))
    check('adapter_build_before_comparator_read',run.index('independent = reference.build(input_raw)') < run.index('comparator_raw =') < run.index('reference.compare_saved('))
    check('reference_hash_preserved_after_comparison','independent reference changed during comparison' in run)
    check('runtime_binary_and_stdlib_hash_guard','Fixed Python binary hash required' in text(cf['authorize']) and 'Fixed stdlib source hash required' in text(cf['authorize']))
    check('comparison_does_not_rebuild_references',all(s not in compare for s in ('self.build(', 'self.state_metrics(', 'self.prepare_coefficients(')))
    state=text(rf['state_metrics'])
    check('exact_state_moments_no_Decimal',all(s not in state for s in ('Decimal','float(','.sqrt(','.dc(')) and 'Fraction(' in state)
    check('exact_variance_second_moment','variance = self.q(Fraction(second, mass * norm * norm) - mean * mean)' in state)
    check('exact_W_normalized_by_original_norm_and_lambda','self.dot(hpsi, hpsi)' in state and 'lam * lam' in state and 'norm)' in state)
    check('exact_matrix_law_mean_crosscheck','exact independent matrix/law mean' in state)
    check('no_reference_metric_rounding_allowance',"'reference_error_allowance': Fraction(0)" in state)
    check('Decimal_display_only',all('Decimal' not in text(n) for k,n in rf.items() if k!='encoded'))
    prepare=text(rf['prepare_coefficients'])
    check('CAR_square_crosscheck','contracted = self.contracted_square(g)' in prepare and 'squared != contracted' in prepare)
    check('full_HD_HR_independent_matrices','hd_before = self.combine([base, *blocks[:count]])' in prepare and 'tail = self.combine(blocks[count:])' in prepare)
    check('identity_move_exact_not_float','hr = self.shifted_identity(tail, -moved)' in prepare and 'hd_matrix = self.shifted_identity(hd_before, moved)' in prepare)
    check('trace_and_tensor_reconstruction','self.trace_pauli(hd_matrix)' in prepare and 'self.reconstruct(law)' in prepare and 'self.reconstruct(hd)' in prepare)
    check('unconditional_complete_65536_trace_loop','for x in range(256)' in text(rf['trace_pauli']) and any(isinstance(n,ast.For) and text(n.iter)=='enumerate(diagonal)' for n in ast.walk(rf['trace_pauli'])))
    check('nearest_even_probability_cell_no_float_division','bits % 2' in text(rf['rounded_binary64']) and 'float(' not in text(rf['rounded_binary64']))
    check('witnesses_bounded_before_encoding',"witness_words = [w for w in union if exact.get(w, 0) != approx.get(w, 0)][:64]" in text(rf['compare_law']))
    check('missing_full_HD_never_full_scope_PASS',"'UNDETERMINED_NOT_SAVED_BY_V5'" in text(rf['compare_law']) and "'status': 'FAIL' if primary == 'FAIL' else 'UNDETERMINED'" in compare)
    check('first_completed_comparison_discrepancy_STOP',"if status != 'PASS':" in compare and any(isinstance(n,ast.Break) for n in ast.walk(rf['compare_saved'])))
    check('new_authority_namespace_and_order','New independent-verification authority namespace' in text(cf['authorize']) and 'sc in (cc, head)' in text(cf['authorize']) and 'cc == head' in text(cf['authorize']))
    check('scientific_dispatch_after_full_guard',text(cf['main']).index('permit = authorize(') < text(cf['main']).index('from .execution import run_authorized'))
    check('exclusive_output_reservation',"output.mkdir(parents=True, exist_ok=False)" in run)
    check('exclusive_fsync_output',"path.open('xb')" in text(ef['_write_exclusive']) and 'os.fsync(' in text(ef['_write_exclusive']))
    check('result_log_limits_equal','resource.RLIMIT_FSIZE' in run and c.POLICY['resources']['output_bytes']==c.POLICY['resources']['log_bytes'])
    check('no_wall_or_CPU_cap',c.POLICY['resources']['wall_time_seconds'] is None and c.POLICY['resources']['cpu_time_seconds'] is None)
    check('one_worker_no_GPU',c.POLICY['resources']['workers']==1 and c.POLICY['resources']['gpu'] is False)
    check('future_result_all_three_statuses',"for s in ('G3', 'G6', 'G9')" in run and 'NOT_EVALUATED_AFTER_STOP' in run)
    check('source_has_no_retry_loop',not any(isinstance(n,(ast.For,ast.While)) and 'run_authorized(' in text(n) for n in ast.walk(trees[PACKAGE+'control.py'])))
    config=c.read_json((ROOT/args.contract).read_bytes()) if args.contract else c.candidate()
    status=c.validate_contract(config)
    check('candidate_shape_valid',status['valid'] and not status['errors'])
    check('candidate_unadopted_no_permission',config['adopted'] is False and not status['actual_execution_permitted'] and not status['contract_complete'])
    for key in c.POLICY['comparison']:
        changed=copy.deepcopy(config);changed['comparison'][key]='RESULT_DEPENDENT_VALUE'
        check('reject_comparison_change:'+key,not c.validate_contract(changed)['valid'])
    for key in c.POLICY['resources']:
        changed=copy.deepcopy(config);changed['resources'][key]='UNFIXED'
        check('reject_resource_change:'+key,not c.validate_contract(changed)['valid'])
    changed=copy.deepcopy(config);changed['resources']['workers']=True
    check('reject_bool_as_integer',not c.validate_contract(changed)['valid'])
    for argv in ([],['run'],['run','--contract','not-read.json','--authorization-commit','not-a-sha','--authorization-path','old-v5-authority']):
        with redirect_stdout(io.StringIO()) as output:
            code=c.main(argv)
        response=json.loads(output.getvalue())
        check('safe_CLI:'+repr(argv),code==(2 if argv else 0) and not response.get('actual_execution_permitted',False))
    for path,ref in [('artifacts/track_b_m1s_gate0_v5_execution_authorization/2026-10-16/v1/execution_authorization.json','8da63d672f779c947dc295eb2c5d524d927ac66b'),(c.AUTH_PREFIX+'not_created.json','not-a-sha')]:
        try:
            c.authorize(ROOT,'not-read.json',ref,path)
        except c.ExecutionClosed:
            refused=True
        else:
            refused=False
        check('old_or_missing_authority_refused:'+path,refused)
    if args.contract:
        for path,digest in config['source_sha256'].items():
            raw=(ROOT/path).read_bytes()
            committed=subprocess.check_output(['git','-C',str(ROOT),'show',config['source_commit']+':'+path])
            check('frozen_source_hash:'+path,raw==committed and hashlib.sha256(raw).hexdigest()==digest)
        check('only_adoption_pending',status['pending']==['validation_contract_adoption','adoption_record'])
    check('no_new_authority_or_result_namespace',not (ROOT/c.AUTH_PREFIX).exists() and not (ROOT/c.OUTPUT_PREFIX).exists())
    check('no_npz_or_kernel_calls',accesses=={'NPZ':0,'kernel_calls':0})
    report={'schema':'m1s_gate0_independent_static_validation_v1','PASS':len(checks),'FAIL':0,'checks':checks,
            'contract_status':status,'NPZ_reads':0,'reference_kernel_calls':0,'synthetic_numerical_tests':0,
            'Pauli_generation':0,'new_W_v_rho_evaluations':0,'scientific_execution_authorizations_grants_markers':0,
            'runtime_or_numerical_correctness_verified':False,'new_scientific_calculations':0,'mandatory_STOP':True}
    if args.write_report:
        Path(args.write_report).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('checks','contract_status')}))


if __name__=='__main__':
    main()
