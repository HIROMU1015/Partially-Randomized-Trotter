"""Synthetic saved schemas and toy scalars only. No research numeric evaluation."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
import csv,json,subprocess,sys,tempfile,unittest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.b2d_evaluator import shot_weighted as s

def synthetic_payload():
    """IDs match metadata; all numeric inputs are invented toy observations."""
    inputs={k:'MOCK-'+k for k in s.IDENTITY}
    inputs.update(recipe=dict(ld=2,n_electrons=1,n_qubits=2,weight_rule='lambda_frobenius_squared'),
        occupied_modes=[0],sector_basis_indices=[1,2],deterministic_fragment_indices=[0,1],
        randomized_block_indices=[2],preparation_id='MOCK-PREP',fixture_definition=dict(name='NO_HAMILTONIAN'))
    data={study:dict(source_commit=s.SOURCES[study]) if study=='analytic' else dict(source=s.SOURCES[study]) for study in s.SOURCES}
    data.update(binding=dict(source=s.SOURCES['b1']),inputs={k:deepcopy(inputs) for k in ('analytic','a0','b1')},
        graph=dict(scope=dict(compiler=s.PROFILE),cases=[]),samples={},statistics={},trajectories={})
    data['analytic']['rows']=[];data['a0']['comparisons']=[]
    data['b1'].update(fixed_sample_set_complete=True,scientific_status='INCONCLUSIVE_RARE_EVENT_COVERAGE',statistics=dict(groups={}))
    data['rare'].update(scientific_status='CONDITIONAL_SENSITIVITY_ONLY',probabilities={})
    data['b0']['provenance']=dict(compiler=s.PROFILE,preparation_id=inputs['preparation_id'],tail_id=inputs['tail_id'])
    for c in s.IDS:
        method,arm,ref,r,K=s.EXPECTED[c]
        for t in s.TIMES:
            k=c+':'+t;sign=1 if t==s.TIMES[0] else -1
            count=4 if c=='B4' else 16 if c=='F1' else 6
            case=dict(method=method,inner_arm=arm,macro_steps=2,n_d=1,refinement=ref,total_time=t)
            specs=[];stages=[];details=[]
            p,norm=s.dec(1),s.dec(1)
            # Independent artificial finite law, never a real saved candidate.
            from trottertracks.b2d_evaluator.rare_sensitivity import finite_law
            probs,B=finite_law(.02,K)
            for j in range(count):
                signed=sign*(-1 if j%2 else 1);tau=signed*.02;duration=float(tau*r)
                spec=dict(duration=str(F.from_float(duration)),rte_steps=r,cutoff=K,seed=41+j)
                conf=dict(rte_steps=r,finite_taylor_order=K,seed=41+j,evolution_time=duration,
                    dimensionless_step_time=tau,lambda_r=1.,tail_hash=inputs['tail_id'],taylor_distribution='rte_even_taylor_paired')
                law=dict(dimensionless_step_time=tau,orders=list(probs),order_probabilities=[float(p) for p in probs.values()],exact_finite_distribution=float(B))
                specs.append(dict(payload=spec));stages.append(dict(append_occurrence=j,config=conf,event_distribution=law,normalization_product=float(B**r)))
                details.append(dict(occurrence=j,spec=spec,dimensionless_step_time=tau,orders=law['orders'],order_probabilities=law['order_probabilities'],exact_finite_distribution=law['exact_finite_distribution'],events=[dict(order=0) for _ in range(r)]))
            row=dict(candidate_id=c,case=case,rte_parameters=dict(r_rte=r,K=K),outputs=dict(
                rte_stages=stages,normalization=dict(normalization_product=float(B**(r*count))),
                signals=dict(corrected_physical_signal_error=.01),bias_qualification=dict(observed_bias=.01)))
            data['analytic']['rows'].append(row)
            data['graph']['cases'].append(dict(candidate_id=c,case=case,rte_parameters=row['rte_parameters'],plan=dict(body=specs)))
            if c in ('B3','F1','R4'):data['a0']['comparisons'].append(dict(key=k,bias=dict(real='0.01',imag='0'),certified=False,u_num=None))
            if c not in ('R2','F1'):data['rare']['probabilities'][k]=dict(p_rare='0.001')
            data['trajectories'][k]=[dict(id=k+':'+str(i),index=i,key='MOCK-KEY-'+k+':'+str(i),details=deepcopy(details)) for i in range(256)]
            sample_rows=[]
            for axis in s.AXES:
                group=k+':'+axis;data['b1']['statistics']['groups'][group]={}
                for metric in s.METRICS:
                    vals=[i+3 if axis=='cosine' else 258-i for i in range(256)]
                    mean=F(sum(vals),256);var=sum((F(x)-mean)**2 for x in vals)/255
                    saved=dict(mean=float(mean),unbiased_sample_variance=float(var),standard_error=float(var/256)**.5,
                        minimum=min(vals),maximum=max(vals),sample_count=256)
                    data['b1']['statistics']['groups'][group][metric]=saved
                    data['statistics'][(group,metric)]={k:str(v) for k,v in saved.items()}
                for i in range(256):sample_rows.append(dict(id=k+':'+str(i),candidate=c,T=t,sample=str(i),axis=axis,
                    physical_key='MOCK-KEY-'+k+':'+str(i),measurement_count='1',**{m:str(i+3 if axis=='cosine' else 258-i) for m in s.METRICS}))
            data['samples'][k]=sample_rows
    return data

def opaque_normalization(case):
    return dict(N_used=F(2),N_model_low=F(2),N_model_high=F(2),U=F(2),rescaling_bias_upper=F(0),
        occurrences=[dict(r=stage['config']['rte_steps']) for stage in case['stages']],mock=True)

def opaque_shots(norm,epsilon,axis):
    return dict(axis=axis,epsilon=epsilon,m_decimal_string=str((1<<80)+13),shot_status='MOCK_ONLY',
        N_used=F(2),N_model_low=F(2),N_model_high=F(2),U=F(2))

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.payload=synthetic_payload()
    def tearDown(self):
        for name in ('numpy','scipy','qiskit','pyscf','trotterlib'):self.assertNotIn(name,sys.modules)

    def test_fixed_scope_and_call_identity(self):
        x=s.scope();self.assertEqual(x['boundaries'],{'shot.bind':12,'shot.normalization':12,'shot.rare':4,
            'shot.joint':72,'shot.bias':24,'shot.count':48,'shot.resource':144,'shot.compare':24,'shot.qualification':24})
        ledger=s.FollowupLedger(x)
        with self.assertRaises(s.Stage1Failure):ledger.call('shot.count',lambda:None,identity='BAD')
        ledger.call('shot.bind',lambda:None,identity=s.case_keys()[0])
        with self.assertRaises(s.Stage1Failure):ledger.call('shot.bind',lambda:None,identity=s.case_keys()[0])
        with self.assertRaises(s.Stage1Failure):ledger.finish()

    def test_artificial_bind_and_rejected_source_input_condition_pairing(self):
        c=s.bind_case('R2:-9/50',self.payload);self.assertEqual(len(c['pairs']),256)
        bad=deepcopy(self.payload);bad['b1']['source']='bad'
        with self.assertRaisesRegex(ValueError,'SOURCE'):s.bind_case('B3:9/50',bad)
        for field in ('hamiltonian_id','sector_id','state_id','deterministic_fragment_indices'):
            with self.subTest(field=field):
                bad=deepcopy(self.payload);bad['inputs']['a0'][field]='BAD'
                with self.assertRaisesRegex(ValueError,'identity'):s.bind_case('B3:9/50',bad)
        bad=deepcopy(self.payload);bad['samples']['B3:9/50'][0]['sample']='1'
        with self.assertRaisesRegex(ValueError,'ID'):s.bind_case('B3:9/50',bad)
        bad=deepcopy(self.payload);bad['graph']['cases'][0]['case']['inner_arm']='BAD'
        with self.assertRaises(ValueError):s.bind_case('B3:9/50',bad)
        bad=deepcopy(self.payload);bad['trajectories']['B3:9/50'][0]['details'][1]['spec']['duration']='1/7'
        with self.assertRaisesRegex(ValueError,'occurrence'):s.bind_case('B3:9/50',bad)

    def test_toy_signed_normalization_enclosure_and_extra_mass(self):
        c=s.bind_case('R2:9/50',self.payload);n=s.normalization_enclosure(c)
        self.assertGreaterEqual(n['U'],n['N_model_high']);self.assertGreaterEqual(n['U'],n['N_used'])
        self.assertIsNone(n['u_num_total']);self.assertGreater(s.extra_rare_probability(c),0)
        self.assertEqual(sum(x['r'] for x in n['occurrences']),24)
        bad=deepcopy(c);bad['stages'][0]['config']['dimensionless_step_time']*=-1
        with self.assertRaisesRegex(ValueError,'Signed'):s.normalization_enclosure(bad)
        with self.assertRaises(ValueError):s.frac(float('nan'))

    def test_toy_shots_large_integer_and_unresolved_ceil(self):
        n=dict(U=F(2),N_used=F(2),N_model_low=F(2),N_model_high=F(2))
        shot=s.scalar_shots(n,'1e-12','cosine')
        self.assertGreater(int(shot['m_decimal_string']),2**53)
        self.assertIsInstance(shot['m_decimal_string'],str)
        with patch.object(s,'shot_count_from_enclosure',side_effect=ValueError('SHOT_COUNT_UNRESOLVED')):
            with self.assertRaisesRegex(s.Stage1Failure,'SHOT_COUNT_UNRESOLVED'):s.scalar_shots(n,'1/17','sine')
        with self.assertRaises(ValueError):s.scalar_shots(dict(n,U=float('inf')),'1/17','sine')

    def test_covariance_exact_huge_weight_and_unknown_rare(self):
        c=s.bind_case('B3:9/50',self.payload);j=s.joint_moments(c['pairs'],'rz_count',c['saved_statistics'])
        self.assertLess(j['sample_covariance_XY'],0)
        mx,my=(1<<90)+13,(1<<90)+17
        shots=dict(cosine=dict(m_decimal_string=str(mx)),sine=dict(m_decimal_string=str(my)))
        w=s.weighted_resource(j,shots,F(1,10))
        totals=[mx*int(p['cosine']['rz_count'])+my*int(p['sine']['rz_count']) for p in c['pairs']]
        mean=F(sum(totals),256);var=sum((F(x)-mean)**2 for x in totals)/255
        self.assertEqual(w['G_S0_fraction'],mean);self.assertEqual(w['variance_S0_sample'],var)
        self.assertIsNone(w['unconditional_G']);self.assertIsNone(w['unknown_mean_u_X'])
        self.assertEqual(w['coefficient_u_X'],F(mx,10))
        bad=deepcopy(c['pairs']);bad[0]['cosine']['rz_count']=1.5
        with self.assertRaisesRegex(ValueError,'Integer'):s.joint_moments(bad,'rz_count')

    def test_boundary_shot_weights_SE_sensitivity_and_null_qualifications(self):
        def resource(m,p,mean,se):return dict(m_X=str(m),m_Y=str(m),p_rare=F(p),known_contribution=F(mean),SE_known_upper=F(se),coefficient_u_X=F(p)*m,coefficient_u_Y=F(p)*m)
        p=resource(7,'1/10',100,2);n=resource(11,'1/5',50,3)
        b=s.compare_resources(p,n);self.assertNotEqual(b['mixture_boundary_slope'],F(1,2))
        for point in b['sensitivity']:
            up=F(13);un=point['negative_mixture_intercept']+b['mixture_boundary_slope']*up
            self.assertEqual(point['delta_known']+F(1,10)*14*up-F(1,5)*22*un,0)
        flag=s.bias_status(dict(observed_bias=F(1),A0_auxiliary=None),'1/10')
        q=s.qualifications(flag,dict(cosine={},sine={}))
        self.assertEqual(q['observed_bias_budget_status'],'OUTSIDE_POINT_BUDGET')
        self.assertIsNone(q['end_to_end_qualified']);self.assertIsNone(q['gamma']);self.assertIsNone(q['u_num'])

    def test_full_fixed_mock_output_scope_rows_partial_fresh_and_STOP(self):
        with tempfile.TemporaryDirectory() as d:
            run=dict(root=Path(d),output=Path(d)/'fresh',source='MOCK',scope=s.scope(),input_blobs={})
            with patch.object(s,'normalization_enclosure',opaque_normalization),patch.object(s,'scalar_shots',opaque_shots):
                result=s.execute(run,mock_only=True,payload_reader=lambda _:self.payload)
            self.assertEqual(result['status'],'COMPLETED_DESCRIPTIVE_MANDATORY_STOP',result)
            self.assertEqual(result['mode'],'MOCK_ONLY');self.assertEqual(result['calls']['completed'],s.scope()['boundaries'])
            events=[json.loads(line) for line in (run['output']/'events.jsonl').read_text().splitlines()]
            joins=[i for i,e in enumerate(events) if e.get('event')=='complete' and e.get('boundary')=='shot.bind']
            numerical=[i for i,e in enumerate(events) if e.get('event')=='start' and e.get('boundary')=='shot.normalization']
            self.assertLess(max(joins),min(numerical))
            for name,count in (('SHOTS.csv',48),('RESOURCES.csv',144),('COMPARISONS.csv',24)):
                with (run['output']/name).open() as f:self.assertEqual(len(list(csv.DictReader(f))),count)
            raw=json.loads((run['output']/'result.json').read_text())
            self.assertEqual(raw['scientific_status'],'INCONCLUSIVE_QUALIFICATIONS_AND_RARE_COSTS')
            self.assertTrue(all(v['end_to_end_qualified'] is None for v in raw['qualifications']))
            self.assertTrue((run['output']/'STOP.json').is_file())
            with self.assertRaises(FileExistsError):s.execute(run,mock_only=True,payload_reader=lambda _:self.payload)

    def test_failure_and_output_save_failure_preserve_STOP_and_partials(self):
        for failure in ('SHOT_COUNT_UNRESOLVED','save'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as d:
                run=dict(root=Path(d),output=Path(d)/'fail',source='MOCK',scope=s.scope(),input_blobs={})
                def broken(*args,**kw):
                    raise s.Stage1Failure('SHOT_COUNT_UNRESOLVED')
                def factory(*args,**kw):
                    e=s.Evidence(*args,**kw);original=e.write
                    def write(name,value):
                        if name=='result.json':raise OSError('synthetic storage failure')
                        return original(name,value)
                    e.write=write;return e
                with patch.object(s,'normalization_enclosure',opaque_normalization),patch.object(s,'scalar_shots',broken if failure!='save' else opaque_shots):
                    status=s.execute(run,mock_only=True,payload_reader=lambda _:self.payload,evidence_factory=factory if failure=='save' else s.Evidence)
                self.assertTrue((run['output']/'STOP.json').exists());self.assertTrue((run['output']/'partial_results.json').exists())
                if failure=='save':self.assertEqual(status['status'],'EVIDENCE_SAVE_FAILED_MANDATORY_STOP')
                else:
                    self.assertEqual(status['failure_category'],'INCONCLUSIVE_NUMERICAL_STOP')
                    self.assertTrue(json.loads((run['output']/'partial_results.json').read_text())['rows'])

    def test_preflight_rejects_source_input_blob_mismatch_without_numeric_calls(self):
        out=ROOT/'artifacts/research_b_outer_shot_weighted/UNCREATED_MOCK'
        with patch.object(s,'_check_source',side_effect=ValueError('SOURCE mismatch')):
            with self.assertRaisesRegex(ValueError,'SOURCE'):s.preflight(ROOT,'a'*40,out)
        with patch.object(s,'_check_source'),patch.object(s.subprocess,'check_output',side_effect=['a\n','b\n']):
            with self.assertRaisesRegex(ValueError,'identity'):s.preflight(ROOT,'a'*40,out)
        with patch.object(s,'_check_source'),patch.object(s.subprocess,'check_output',return_value='a\n'),\
             patch.object(s,'normalization_enclosure',side_effect=AssertionError('science forbidden')),\
             patch.object(s,'read_payload',side_effect=AssertionError('payload forbidden')):
            result=s.preflight(ROOT,'a'*40,out)
            self.assertEqual(result['scope'],s.scope());self.assertEqual(len(result['input_blobs']),len(s.INPUTS))
        self.assertFalse(out.exists())

    def test_inspect_import_barrier_and_explicit_execute_required(self):
        code="import sys;sys.path.insert(0,'src');from trottertracks.b2d_evaluator import shot_weighted as s;s.scope();assert not any(x.split('.')[0] in ('numpy','qiskit','scipy','mpmath','pyscf') for x in sys.modules)"
        p=subprocess.run(['/usr/bin/python3.11','-I','-B','-S','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        p=subprocess.run(['/usr/bin/python3.11','-I','-B','-S','scripts/run_b2d_outer_shot_weighted.py','--inspect'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout)['status'],'METADATA_ONLY_NO_SCIENCE')
        p=subprocess.run(['/usr/bin/python3.11','-I','-B','-S','scripts/run_b2d_outer_shot_weighted.py'],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)

if __name__=='__main__':unittest.main()
