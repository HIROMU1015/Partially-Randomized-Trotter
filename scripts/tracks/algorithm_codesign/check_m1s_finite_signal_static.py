#!/usr/bin/env python3
"""Metadata/AST validation only. No numerical imports, NPZ reads or kernels."""
import argparse,ast,copy,hashlib,importlib.abc,io,json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
PKG='src/trottertracks/algorithm_codesign/m1s_finite_signal_v1/'

class BlockScience(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ['numpy','scipy','mpmath','openfermion','qiskit'] or fullname.endswith(('.exact_input','.numerics','.word_reference','.pilot','.reference','.execution')) or '.m1s_gate0_v5' in fullname:
            raise RuntimeError('science import forbidden in static preparation: '+fullname)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--contract');parser.add_argument('--write-report');args=parser.parse_args()
    sys.meta_path.insert(0,BlockScience());access={'NPZ_reads':0,'kernel_calls':0}
    def audit(event,arguments):
        if event=='open' and isinstance(arguments[0],(str,bytes)) and str(arguments[0]).lower().endswith(('.npz',".npz'")):
            access['NPZ_reads']+=1;raise RuntimeError('NPZ open forbidden')
    def profile(frame,event,arg):
        if event=='call' and '/m1s_finite_signal_v1/' in frame.f_code.co_filename and not frame.f_code.co_filename.endswith(('/control.py','/__init__.py')):
            access['kernel_calls']+=1;raise RuntimeError('numerical kernel call forbidden')
    sys.addaudithook(audit);sys.setprofile(profile);sys.path.insert(0,str(ROOT/'src'))
    from trottertracks.algorithm_codesign.m1s_finite_signal_v1 import control as c
    checks=[]
    def check(n,v):
        checks.append({'name':n,'status':'PASS' if v else 'FAIL'})
        if not v:raise AssertionError(n)
    trees={}
    for p in c.POLICY['source_sha256']:
        trees[p]=ast.parse((ROOT/p).read_bytes(),filename=p);check('syntax:'+p,True)
    functions={name:{n.name:n for n in ast.walk(trees[PKG+name+'.py']) if isinstance(n,ast.FunctionDef)} for name in ['control','exact_input','numerics','word_reference','pilot']}
    tx=lambda n:ast.unparse(n)
    cf,nf,pf=functions['control'],functions['numerics'],functions['pilot']
    old=ast.parse(subprocess.check_output(['git','-C',str(ROOT),'show','4787cf90b053d251bf6392988af35516bdbbe663:src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1/reference.py']))
    oldf={n.name:n for n in ast.walk(old) if isinstance(n,ast.FunctionDef)}
    for name in ['add','mul','put','decode','dyadic','action','one_body','square','scale','combine','scalar_matrix','hermitian','tensor_word','reconstruct','matrix_digest']:
        check('declared_kernel_reuse_AST:'+name,ast.dump(oldf[name],include_attributes=False)==ast.dump(functions['exact_input'][name],include_attributes=False))
    check('own_permit_before_Kernels_init',tx(nf['__init__'].body[0])=='require_permit(permit)')
    check('own_permit_before_worker_IO',tx(pf['run_authorized'].body[0])=='require_permit(permit)')
    check('own_permit_before_exact_input',tx(functions['exact_input']['__init__'].body[0])=='require_permit(permit)')
    check('own_permit_before_future_correctness',tx(functions['word_reference']['verify_future'].body[0])=='require_permit(permit)')
    check('dispatch_after_full_guard',tx(cf['main']).index('permit = authorize(')<tx(cf['main']).index('from .pilot import run_authorized'))
    check('strict_new_namespace',c.AUTH_PREFIX=='artifacts/track_b_m1s_finite_signal_execution_authorization/' and c.OUTPUT_PREFIX=='artifacts/track_b_m1s_finite_signal_result/')
    check('strict_adopted_source_authority_order','strict SOURCE order' in tx(cf['authorize']) and 'S in (C, authority_commit)' in tx(cf['authorize']))
    check('input_law_review_independent_result_hashes_bound',all(s in tx(cf['authorize']) for s in ["config['input']","config['saved_law']","config['review']","config['independent_result']"]))
    check('runtime_native_hashes_before_import','numpy_build_files_sha256' in tx(cf['authorize']) and 'np_root = Path(spec.origin).parent.parent' in tx(cf['authorize']))
    for p,t in trees.items():
        for n in ast.walk(t):
            if isinstance(n,ast.ImportFrom):check('no_old_authority_or_science_import:'+p+':'+str(n.module),not any(x in (n.module or '') for x in ['m1s_gate0','trotterlib']))
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
                check('no_forbidden_numerical_operation:'+p+':'+n.func.attr,n.func.attr not in ['eig','eigh','eigvals','svd','kron_superoperator','unwrap','sample','choice'])
    check('shared_map_same_D_sides_no_dagger','self.mul(self.mul(d, x), d)' in tx(nf['shared']) and 'conjugate' not in tx(nf['shared']))
    phi=tx(nf['phi'])
    check('coherent_phi_signs',all(s in phi for s in ['c * c','1j * c * s','-s * s','self.mul(h, x)','self.mul(x, h)','self.twirl(x, plan)']))
    check('ordinary_and_independent_budget_matched',all(s in tx(pf['run']) for s in ['R // 2','angle / 2','k.mul(vp, vp)','k.exp_i(hd, Fraction(delta) / 2)']))
    check('reference_general_state_full_H_exp','k.signal(k.exp_i(full, Fraction(T)), state, norm)' in tx(pf['run']))
    check('no_dense_superoperator',c.POLICY['resources']['max_dense_superoperator_cells']==0)
    check('Walsh_conjugation_diagonal_reindex',all(s in tx(nf['twirl']) for s in ['a.m[rows, cols]','self.fwht(transformed * spectrum)','m[rows, cols] = result']))
    check('literal_tensor_full_size_reference_not_Walsh','np.kron(p, local[letter])' in tx(nf['verify_twirl_probe']) and 'np.argmax' in tx(nf['verify_twirl_probe']))
    check('rational_reference_no_mean_or_numpy_in_reference_construction',tx(functions['word_reference']['verify_future']).index('references[method] =')<tx(functions['word_reference']['verify_future']).index('np = kernels.np'))
    check('mandatory_future_correctness_before_real_input',tx(pf['run']).index('verify_future(self.permit, k)')<tx(pf['run']).index('target = exact.targets('))
    check('Taylor_remainder_and_unitary_bound','Fraction(8, 7)' in tx(nf['exp_i']) and 'math.factorial(degree + 1)' in tx(nf['exp_i']) and 'value.b = 1.0' in tx(nf['exp_i']))
    check('roundoff_operand_errors_retained','a.b * b.e + b.b * a.e + a.e * b.e' in tx(nf['mul']))
    check('no_result_dependent_precision','eps_signal_abs' in tx(pf['store']) and "row['status'] = 'PRECISION_UNRESOLVED'" in tx(pf['store']))
    check('phase_no_unwrap_and_branch_floor_STOP',all(x in tx(pf['phase_metrics']) for x in ['visibility floor','principal-angle branch cut','UNDETERMINED_PHASE','math.atan2']))
    check('absolute_contrast_and_lambda_squared',"beta = data['lambda'] ** 2 / Fraction(4 * R)" in tx(pf['run']) and 'UNRESOLVED_ABSOLUTE_CONTRAST' in tx(pf['run']) and "'relative_visibility_improvement': None" in tx(pf['run']))
    check('legacy_HD_not_recreated_as_old_comparator','this is a new pilot construction' in tx(pf['run']))
    check('exclusive_one_shot_and_fsync',"output.mkdir(parents=True, exist_ok=False)" in tx(pf['run_authorized']) and "path.open('xb')" in tx(pf['_write']) and 'os.fsync' in tx(pf['_write']))
    check('remaining_rows_persisted_after_STOP','UNDETERMINED_NOT_EVALUATED_AFTER_STOP' in tx(pf['run_authorized']))
    cfg=c.read_json((ROOT/args.contract).read_bytes()) if args.contract else copy.deepcopy(c.POLICY)
    status=c.validate_contract(cfg)
    check('candidate_valid_unadopted_closed',status['valid'] and cfg['adopted'] is False and cfg['coverage_model_adopted'] is False and status['actual_execution_permitted'] is False and not status['contract_complete'])
    for group in ['model','grid','semantics','target','precision','resources','runtime']:
        for key in cfg[group]:
            changed=copy.deepcopy(cfg);changed[group][key]='UNAUTHORIZED_CHANGE'
            check('reject_frozen_change:'+group+'.'+key,not c.validate_contract(changed)['valid'])
    plan=c.planned_rows(cfg)
    check('108_rows_36_zero_72_nonzero',len(plan)==108 and sum(x['s']==0 for x in plan)==36 and sum(x['s']>0 for x in plan)==72)
    check('all_three_methods_splits',set(x['method'] for x in plan)==set(cfg['grid']['methods']) and set(x['split'] for x in plan)=={'G3','G6','G9'})
    check('all_conceptual_label_counts_matched',all(x['conceptual_independent_labels']==x['s']*x['R'] and (x['r_per_active_branch']*2==x['R'] if x['method']=='independent_pm' else x['r_per_active_branch']==x['R']) for x in plan))
    for argv in [[],['run'],['run','--contract','NOT_READ','--authorization-path','old_gate0_authority','--authorization-commit','not-a-sha']]:
        with redirect_stdout(io.StringIO()) as capture:code=c.main(argv)
        check('safe_refusal:'+repr(argv),code==(2 if argv else 0) and json.loads(capture.getvalue())['actual_execution_permitted'] is False)
    try:c.require_permit(None)
    except c.ExecutionClosed:refused=True
    else:refused=False
    check('no_bare_permit',refused)
    if args.contract:
        for p,h in cfg['source_sha256'].items():
            raw=(ROOT/p).read_bytes();committed=subprocess.check_output(['git','-C',str(ROOT),'show',cfg['source_commit']+':'+p])
            check('frozen_source:'+p,raw==committed and hashlib.sha256(raw).hexdigest()==h)
        check('only_adoption_and_coverage_pending',set(status['pending'])=={'adopted','coverage_model_adopted','adoption_record'})
    check('no_new_execution_namespace',not(ROOT/c.AUTH_PREFIX).exists() and not(ROOT/c.OUTPUT_PREFIX).exists())
    check('no_numerical_processing',access=={'NPZ_reads':0,'kernel_calls':0})
    report={'schema':'m1s_finite_signal_static_v1','PASS':len(checks),'FAIL':0,'checks':checks,'contract_status':status,
      'new_scientific_calculations':0,'NPZ_reads':access['NPZ_reads'],'numerical_kernel_calls':access['kernel_calls'],
      'synthetic_numerical_correctness_tests':0,'Pauli_generation':0,'finite_signal_evaluations':0,
      'new_execution_authorizations_grants_markers':0,'runtime_or_numerical_correctness_verified':False,'mandatory_STOP':True}
    if args.write_report:(ROOT/args.write_report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['checks','contract_status']}))
if __name__=='__main__':main()
