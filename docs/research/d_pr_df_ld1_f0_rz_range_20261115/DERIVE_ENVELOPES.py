"""Authorized SOURCE envelopes and saved-TV products; no circuits or price samples.

Run in a checkout of this artifact's commit. Outputs are written only if absent.
The compiler argument and its applicability conditions are in REPORT/COMPILER_ROUTE.
"""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import ast,hashlib,json,subprocess,csv
B=Path('docs/research/d_pr_df_ld1_f0_rz_range_20261115')
BASE='d749ef92e8ac3087747481a46aeb28ec4ca04c4b'
F0=dict(commit=BASE,path='docs/research/d_pr_df_ld1_f0_law_diagnostic_20261115/C_HANDOFF.json')
RAW='ec78fde57e117e9bbf79a26bfaa2e1575c1f5784'
refs=dict(
 Accounting=('8457c443554af5fadaa3ba36f97f6316ca3c5050','src/trottertracks/h4_df10_price_v1.py'),
 Gaussian=('155529e20a8fac39e00786c40b57830b71203c4f','scripts/h46_new2_native.py'),
 Compose=('155529e20a8fac39e00786c40b57830b71203c4f','scripts/h46_long_time_native.py'),
 Schedule=('155529e20a8fac39e00786c40b57830b71203c4f','src/trotterlib/h46_m5_plan.py'),
 EventLabels=('c31ef424175c02379eafebe8fc6c7548c1d9aabe','src/trotterlib/rte.py'),
 Diagonals=('c31ef424175c02379eafebe8fc6c7548c1d9aabe','src/trotterlib/df_trotter/ops.py'),
 LegacyBuilder=('c31ef424175c02379eafebe8fc6c7548c1d9aabe','src/trotterlib/df_rte_qiskit.py'),
 NativeBuilder=('c31ef424175c02379eafebe8fc6c7548c1d9aabe','src/trottertracks/resource_applicability/ax2b_native_df_v5.py'),
 CompilerCall=('c31ef424175c02379eafebe8fc6c7548c1d9aabe','src/trotterlib/rte_compiled_cost.py'),
 Context=('1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7','src/trottertracks/r_price_phase1_science.py'),
 FastPath=('39ddd59edf54a3a901e5302207b2765ddbd03fbb','src/trottertracks/compiled_price_fast.py'),
 NewZZ=('39ddd59edf54a3a901e5302207b2765ddbd03fbb','src/trottertracks/efficient_controlled_zz.py'),
 Closure=('a637ed1b813dc3952339cb05f36beb79168ace23','docs/research/h4_pr_physical_closure_20261113/run02/RESULT.json'),
 CIndex=('c99d511e2422b8f250490b11054f754a81e536c5','docs/research/pr_df_ld1_q3_price_admission_20261115/ADMISSION.json'),
 LawMath=('d72dc4e6228bb380f4d1d000e5e030794fce7164','docs/research/finite_k_sampler_law_alignment_20261113/MATH.md'),
 LawHelper=('d72dc4e6228bb380f4d1d000e5e030794fce7164','src/trottertracks/finite_k_sampler_law.py'),
 OldResidual=('af6f27d7a6467bb355686421acd067544edcbbd2','docs/research/d_compiled_residual_c_minus_s_20261113/HANDOFF_C.md'),
 OldPricePreparation=('d72dc4e6228bb380f4d1d000e5e030794fce7164','docs/research/mean_compiled_price_preparation_20261112/HANDOFF.md'),
 PFCompilerStudy=('dd435d07e3f0490165ff49889be9c71f573bd5db','docs/research/d_pf_unobserved_round_closure_20261115/REPORT.md'),
 PFNewAudit=('606881b042ed69f9a1e7f00311417f2b65855d7b','docs/research/d_pf_qpy_closure_audit_20261115/HANDOFF.md'),
 CF0Admission=('259a0b2d7357d740502d6837db2cf3c2a8b75c5e','docs/research/pr_df_ld1_f0_admission_20261115/HANDOFF.md'))
blob=lambda c,p:subprocess.check_output(['git','show',c+':'+p])
read=lambda c,p:json.loads(blob(c,p))
loc=lambda c,p,*s:dict(commit=c,path=p,selector=list(s))
sources={k:dict(loc(*v),git_blob=subprocess.check_output(['git','rev-parse',v[0]+':'+v[1]],text=True).strip()) for k,v in refs.items()}
closure=read(*refs['Closure']);head=closure['head_diagonal_effects'];tail=closure['preparation']['basis_registry']
assert sorted((v['block'],v['directional']) for v in head)==[(0,False),(0,True),(1,False),(1,True)]
assert closure['preparation']['hash']=='b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7'
allops=[v['basis'] for v in head]+[v['operations'] for v in tail]
for ops in allops:
 assert {o['name'] for o in ops}<={'p','xx_plus_yy'} and all(0<=q<8 for o in ops for q in o['qubits'])
 counts=Counter(o['name'] for o in ops)
 assert counts['xx_plus_yy']<=28 and counts['p']<=8+2*28
# Count the literal expansion of XX+YY in Accounting, without running Accounting or any gate builder.
tree=ast.parse(blob(*refs['Accounting']));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Accounting')
gate=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_gate')
branch=next(n for n in ast.walk(gate) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and any(isinstance(v,ast.Constant) and v.value=='xx_plus_yy' for v in n.test.comparators))
loop=next(n for n in branch.body if isinstance(n,ast.For))
names=[e.elts[0].value for e in loop.iter.elts]
cost=dict(rz=1,sx=0,x=0,cx=0,p=1,s=1,sdg=1,sxdg=0,ry=1)
xx_RZ=sum(cost[n] for n in names);assert xx_RZ==10
N=8;GIVENS=28;BASIS=8+28*(10+2);assert BASIS==344
HEAD=8*BASIS+(8+36)+(9+37);assert HEAD==2842
TAIL_PER_EVENT=6*BASIS+2*8+2;assert TAIL_PER_EVENT==2082
formula=dict(n_system=8,max_Givens=28,xx_plus_yy_literal_RZ=10,max_basis_RZ=344,head_RZ_per_outer=2842,
 tail_RZ_per_event=2082,tail_relative_phase_RZ_per_outer=1,wrapper_plus_scalar_X=5,wrapper_plus_scalar_Y=6,
 formula='U_S(m,a)=q*(2843+2082*R)+5+int(a==Y)',
 proof='Ignore only inverse-basis cancellations and Z folding; bound Gaussian operations by n+3*n*(n-1)/2; two head blocks each in two directions; K2 at most 2 products+1 rotation, and a basis pair for each application.')
# Uniform bound on total basis operations, hence also on compiled RZ.
# Registered one-qubit decomposition paths cost <=5; native basis operations cost 1.
# XX+YY: 6 RZ + 1 SX + 2 CX + S + Sdg + SXdg + 2 RY <= 34.
# A Gaussian basis: <=8 initial P and <=28 triples (XX+YY,P,P).
# Head: 4 block visits * 2 bases; uncontrolled 8+(8+28*3),
# directional (8*3+1)+(8*3+28*5+1). Tail: 3 basis pairs,
# <=2 products * <=2 CZ (each <=11), one new-ZZ rotation <=2 CX+CRZ(4).
C_BASIS=8*5+28*(34+2*5);assert C_BASIS==1272
C_HEAD=8*C_BASIS+(8+8+28*3)+(8*3+1)+(8*3+28*5+1);assert C_HEAD==10466
C_TAIL=6*C_BASIS+2*2*11+6;assert C_TAIL==7682
cformula=dict(one_qubit_registered_path_upper=5,xx_plus_yy_upper=34,basis_total_ops_upper=C_BASIS,
 head_total_ops_per_outer=C_HEAD,tail_total_ops_per_event=C_TAIL,tail_phase_per_outer=5,
 wrapper_scalar_measure_X=16,wrapper_scalar_measure_Y=21,
 formula='U_C(m,a)=q*(10471+7682*R)+16+5*int(a==Y)',
 quantity='TOTAL_POST_TRANSLATION_BASIS_OPERATIONS_INCLUDING_MEASUREMENT_UPPER_BOUNDS_COMPILED_RZ',
 compiler_proof='COMPILER_ROUTE.json and REPORT.md section 4; standard Qiskit 1.3.0 no-target-error opt1 route')
f0=read(F0['commit'],F0['path']);bounds=read(RAW,'docs/research/d_pr_df_ld1_f0_law_diagnostic_20261115/run01/ROUND_BOUNDS.json');rows=[]
exact=lambda v:dict(numerator=v.numerator,denominator=v.denominator)
for a in f0['rows']:
 m,axis=a['m'],a['axis'];p=read(a['input']['commit'],a['input']['path'])
 for k in a['input']['selector']:p=p[k]
 q,R=p['row']['q'],p['row']['R_outer'];upper=q*(HEAD+1+TAIL_PER_EVENT*R)+5+(axis=='Y')
 cupper=q*(C_HEAD+5+C_TAIL*R)+16+5*(axis=='Y')
 bound=bounds['rows'][m];assert bound['m']==m and bound['physical_id']==a['physical_id'] and bound['event_positions']==q*R
 delta=Fraction(**bound['path_TV_upper']);osc=upper+cupper;shift=delta*osc
 rows.append(dict(m=m,axis=axis,physical_id=a['physical_id'],q=q,R_outer=R,
 support='UNION_OF_IDEAL_SOURCE_AND_CONDITIONAL_IID_GRID_PATH_SUPPORTS_INCLUDING_IDEAL_POSITIVE_GRID_ZERO_LABELS',
 F0_path_bound=a['conditional_round_bound'],base_axis_index=loc('950577f416c17a20d309a8053a4bb016027b724d','docs/research/d_pr_df_ld1_ec_plan_20261115/AXIS_EVIDENCE.json','rows',len(rows)),
 C_lower=0,C_upper=cupper,S_lower=0,S_upper=upper,
 direct_residual_lower=None,direct_residual_upper=None,residual_oscillation_upper=osc,
 separate_envelope_residual_interval=dict(lower=-upper,upper=cupper),
 conditional_RZ_shift_upper=exact(shift),conditional_RZ_shift_display=float(shift),actual_PCG64_RZ_shift_upper=None,
 classification='COARSE_SEPARATE_C_S_SOURCE_ENVELOPES_CONDITIONAL_COMPILER_ROUTE',
 finite_compiled_evidence_role='CALIBRATION' if m in (0,3,6) else 'INDEPENDENT_HOLDOUT' if m==4 else 'NO_MATCHED_FINITE_PRICE_COHORT',
 scope='Fixed saved preparation and event grammar, successful fixed compiler algorithm, IID-grid law assumptions in saved F0; not actual PCG64 or empirical means.',
 missing='Actual PCG64 joint-law connection, analytic-real to saved-weight error, finite-cohort/model error, adopted price accuracy.',
 E_C=None,CI=None,N=None,G=None))
assert len(rows)==26
obj=dict(schema='LD1_F0_RESIDUAL_RANGE_EVIDENCE_V1',status='CONDITIONAL_COARSE_RZ_ENVELOPES_26_AXES_NOT_FORMAL_E_C',F0=F0,F0_raw_commit=RAW,sources=sources,
 structural_envelope_derivation=formula,compiled_envelope_derivation=cformula,
 structural_envelope_numerical_rows=26,direct_residual_range_rows=0,separate_envelope_residual_range_rows=26,conditional_RZ_conversion_rows=26,rows=rows,
 source_basis_metadata_check=True,observed_residual_minmax_used=False,m4_error_used_as_range=False,
 duplication_check=dict(reuse='Prior 950577f4 scope inventory plus d749ef92 F0; old residual/P2/mean-price source has requirements but no supplied全support residual/compiled envelope.',
 checked_remote_refs=dict(D2=BASE,C='259a0b2d7357d740502d6837db2cf3c2a8b75c5e',D='606881b042ed69f9a1e7f00311417f2b65855d7b',R='3a8b37f1a282c0c06ac1a363b070f4376e9ff41a'),
 C_delta='F0 admission published: fixed 14/13/26 accepted; no range/actual-PCG bound supplied. Consumer admission not prerequisite.',
 D_delta='PF audit completed; reuse common compiler source-route study only, no PF periodic closure or observed prices transferred.',
 matching_local_range_jobs=[],unresolved_external_storage='No claim of global absence outside scoped accessible refs/jobs'),
 new_paths=0,H4_builds=0,compiles=0,F0_reruns=0,rare_search=0,refit=0,E_C=None,CI=None,N=None,G=None,mandatory_stop=True)
with (B/'RANGE_EVIDENCE.json').open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
with (B/'AXES.csv').open('x',newline='') as f:
 keys=['m','axis','physical_id','q','R_outer','S_upper','C_upper','residual_oscillation_upper','conditional_RZ_shift_display','classification','finite_compiled_evidence_role']
 w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader()
 for r in rows:w.writerow({k:r[k] for k in keys})
print(json.dumps(dict(S_formula=formula,C_formula=cformula,axes=26,conditional_RZ_conversions=26,
 first_shift_RZ=rows[0]['conditional_RZ_shift_display'],last_shift_RZ=rows[-1]['conditional_RZ_shift_display']),indent=2))
