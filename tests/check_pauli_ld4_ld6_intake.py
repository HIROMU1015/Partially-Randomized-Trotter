"""Real LD4/LD6 configurations/dictionaries; artificial carriers only; no Q3.

All synthetic carriers and assembly commits live in temporary Git repositories.
The published configurations remain unchanged and are read at their original SOURCE.
"""
import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from check_pauli_signal_entry_intake import A,C,T,Repository,metadata_only,packet_function,blob

B2='d7a976a629a6db68197b24f8f86734e109c8bf5d'
OLD_B='bd42dd1e95eb1d4152216d62792ad3421b803f7c'
PUBLIC_B='ff9d25d62ada75fc56e89540459c15d49157b90c'
ADAPTER='scripts/run_rpe_optimizer_q3_job.py'
EQUIVALENT_ENTRY='ebab4209798b6e2253ea227739673ab91ea92559'
CONTRACT=dict(commit='b6d4dea49c39f364048af422bc2928ee7c169261',
              path='docs/research/pauli_ld5_two_vectors_q3_results_20261116/INPUTS.json')
CONFIGS={4:'PREPARATION_P1.0.CONFIGURATION.json',6:'PREPARATION.0.CONFIGURATION.json'}


def configuration_source(prefix):
    return dict(commit=B2,path=f'docs/research/rpe_split_preparation_20261116/LD{prefix}/'+CONFIGS[prefix])


def b2_api(*, adapted=False):
    """B2's actual pure case/metadata functions; one version-check fix only."""
    raw=blob(B2,ADAPTER).decode()
    old="accepted.get('assembly_implementation_source', {}).get('commit') != A.ENTRY_SOURCE"
    new="accepted.get('assembly_implementation_source', {}).get('commit') not in A.ENTRY_SOURCES"
    if adapted:
        assert raw.count(old)==1
        raw=raw.replace(old,new)
    names={'common_case','common_metadata','git_json','count_bounds','in_scope'}
    tree=ast.parse(raw)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    import json
    scope=dict(deepcopy=deepcopy,json=json,subprocess=subprocess,COMMON={})
    scope['CONTRACT']=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='CONTRACT' for t in n.targets))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),B2+':'+ADAPTER,'exec'),scope)
    return NS(**{n:scope[n] for n in names},COMMON=scope['COMMON'])


def fixture(repo,prefix,source):
    """Use A's unchanged packet() and P1 job() on the real saved configuration."""
    store=A.p.SavedArtifacts(repo.root);cfg_src=configuration_source(prefix);config=store.read(cfg_src)
    template=store.read(A.LD3_HANDOFF)['packets'][0]['rounds'][0]['mean_record']
    helper=A._ld3_metadata_helpers(str(repo.root));api=NS(validate=A.B._validate,exact_scalar=A.B.a.exact_scalar)
    # A's actual numerical SOURCE description, with no evaluation of its provider.
    numerical=dict(commit=source,path='src/trottertracks/pr_rpe_signal_provider.py',
        adapter=A.ENTRY_MODULE,kernel_source='e8641b84fd8a3e98a49b65ecaa8fbd371a9d8ab7',
        recipe='DIRECT_EXPM_HEAD_PAIRED_TAYLOR_MATRIX_POWER_V1')
    records={}
    for row in config['rounds']:
        m=row['m'];job=helper.job(config,m,numerical,api);rec=deepcopy(template)
        for k in ('m','B_configuration_id','B_candidate_id','B_identity','B_law_binding','physical_context'):
            rec[k]=deepcopy(job[k])
        rec.update(source=deepcopy(numerical),candidate=deepcopy(job['physical_context']['physical_candidate']),
            artificial_fixture='SYNTHETIC_NOT_AN_ACQUIRED_SIGNAL')
        for k in ('signed_weights','allocation','event_count'):rec[k]=deepcopy(row['structure'][k])
        rec['signal'].update(raw=dict(real=.25,imag=-.5),corrected=dict(real=.25,imag=-.5),
                             reference=dict(real=.25,imag=-.5),exact_tail=dict(real=.25,imag=-.5),
                             log_B=0.,B=1.,attenuation=1.)
        rec['allowance']=dict(artificial_fixture=True,status='UNKNOWN_NOT_A_NUMERICAL_CERTIFICATE')
        rec['numerical_failure_reasons']=[]
        records[m]=rec;repo.write(f'fixtures/LD{prefix}/m{m}.json',dict(record=rec))
    raw_commit=repo.commit()
    rows={m:(r,dict(commit=raw_commit,path=f'fixtures/LD{prefix}/m{m}.json',selector=['record'],scientific_source=source),False)
          for m,r in records.items()}
    handoff=packet_function(source)(config,rows,configuration_source=cfg_src,
        raw_result_source=dict(commit=raw_commit,path=f'fixtures/LD{prefix}/m0.json'))
    handoff['artificial_fixture']='SYNTHETIC_ONLY_NOT_FOR_SAVED_SIGNAL_CATALOG'
    ref=repo.save(handoff,f'LD{prefix}_MEAN_HANDOFF.json')
    return config,handoff,ref


def exercise(prefix,source=A.DESCRIPTOR_ENTRY_SOURCE):
    repo=Repository(source)
    try:
        # ebab42097 changes preparation code, while all three entry files are
        # byte-identical to 6e4b6bd6a. Keep the output/numerical locators intact;
        # assembly_implementation_source names the matching fixed code version.
        implementation=A.DESCRIPTOR_ENTRY_SOURCE if source==EQUIVALENT_ENTRY else source
        if source==EQUIVALENT_ENTRY:
            dep='src/trottertracks/pauli_split_descriptor.py'
            path=repo.root/dep;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob(source,dep))
        config,handoff,ref=fixture(repo,prefix,source);before=deepcopy(handoff)
        with metadata_only():
            accepted=A.accept_saved(repo.root,integration=C.interfaces(str(repo.root)).pipeline,handoff_source=ref)
            item=accepted['results'][0];binding=T.create(repo.root,config,original_source=configuration_source(prefix))
            bound=T.bind_signal(repo.root,binding,item['signal_packet'])
            target=item['signal_packet']['target'];view=bound['signal_packet']
            adapter=b2_api(adapted=True)
            request=dict(request_id=f'ARTIFICIAL_LD{prefix}_INPUT_ONLY',configuration_id=config['id'],
                N_X=[2]*13,N_Y=[3]*13,
                route=dict(contract_template=CONTRACT,configurations={config['id']:dict(
                    configuration=configuration_source(prefix),signal_packet=ref,binding_id=binding['id'],
                    total_configuration_id=binding['total_configuration']['id'])}))
            case,inputs=adapter.common_case(repo.root,request)
            adapter.COMMON.update(view=repo.root,X=NS(A=A,T=T))
            actual_binding,actual_bound=adapter.common_metadata(case)
            assert actual_binding==binding and actual_bound==bound
            # Build exactly the kernel-side input that T.evaluate would pass to
            # Q.evaluate, without calling either numerical entry.
            allocation=deepcopy(case['allocation']);allocation['source']=dict(fixture='TECHNICAL_INPUT_ONLY')
            contract=deepcopy(inputs['statistical_contract']);contract['source']=allocation['source']
            contract['simulation_policy']['source']=allocation['source']
            contract['summary_options']['unresolved_loss']['source']=allocation['source']
            entries=T.K.prepare(config,item['signal_packet'],[allocation],
                evaluation_target=case['evaluation_target'],statistical_contract=contract,
                integration=C.interfaces(str(repo.root)).pipeline)
            total_entries=T.K.prepare(bound['configuration'],view,[allocation],
                evaluation_target=case['evaluation_target'],statistical_contract=contract,
                integration=C.interfaces(str(repo.root)).pipeline)
            assert len(entries)==len(total_entries)==1
            assert [r['shots'] for r in entries[0]['prepared']['rows']]==allocation['shots']
            bounds=adapter.count_bounds(allocation['shots'])
            assert adapter.in_scope(bounds,inputs['options'])
            for original,mean in zip(config['rounds'],view['means']):
                assert mean['canonical_origin']['identity']==original['identity']
                assert mean['qualification']=='UNQUALIFIED'
                assert mean['raw']==dict(real=.25,imag=-.5)
            assert item['signal_packet']['target']==view['target']==target
            assert view['assembly_implementation_source']['commit']==implementation
            assert handoff==before
            for k in ('sufficient_N','global_RMSE_bound','G','winner'):assert view[k] is None
        return dict(prefix=prefix,original_configuration_id=config['id'],configuration_source=configuration_source(prefix),
            total_configuration_id=binding['total_configuration']['id'],binding_id=binding['id'],
            assembly_implementation_source=dict(commit=implementation,path=A.ENTRY_MODULE),
            intake='PASS_WITH_ARTIFICIAL_CARRIERS',C_metadata_binding='PASS_WITH_ARTIFICIAL_CARRIERS',
            total_view='PASS_WITH_ARTIFICIAL_CARRIERS',Q3_input='PREPARED_ONLY_NOT_EVALUATED',
            B2_common_case='PASS_UNMODIFIED',B2_common_metadata='PASS_WITH_ONE_VERSION_CHECK_ADAPTATION',
            actual_saved_inputs=dict(configuration=True,dictionary=True,descriptor=True,raw_means=False),
            all_rounds=list(range(13)),physical_ids=[r['candidate']['id'] for r in config['rounds']],
            fixture_shots=allocation['shots'],fixture_only=True,fixture_registered=False,
            numeric_gate='NOT_EXECUTED',Q3=0,signal=0,compile=0,sufficient_N=None,global_RMSE_bound=None,G=None)
    finally:repo.close()


class DescriptorIntake(unittest.TestCase):
    def test_b2_intake_and_ff9_have_identical_implementations(self):
        for path in ('src/trottertracks/rpe_pauli_signal_intake.py','src/trottertracks/rpe_c_interfaces.py'):
            self.assertEqual(blob(OLD_B,path),blob(PUBLIC_B,path))

    def test_saved_LD4_and_LD6_normal_intake_total_and_Q3_input(self):
        for prefix in (4,6):
            with self.subTest(prefix=prefix):self.assertEqual(exercise(prefix)['Q3'],0)

    def test_latest_descriptor_reuse_entry_version_and_old_sources_retained(self):
        self.assertEqual(exercise(6,EQUIVALENT_ENTRY)['Q3'],0)
        for source in (A.ENTRY_SOURCE,A.CURRENT_ENTRY_SOURCE):
            repo=Repository(source)
            try:
                self.assertEqual(A._entry_assembly_source(repo.root,dict(commit=repo.git('rev-parse','HEAD'))),source)
            finally:repo.close()

    def test_old_public_version_rejects_new_entry_and_old_B2_guard_reproduces(self):
        repo=Repository(A.DESCRIPTOR_ENTRY_SOURCE)
        try:
            config,_,ref=fixture(repo,4,A.DESCRIPTOR_ENTRY_SOURCE)
            scope=dict(__name__='trottertracks._old_ff9_intake',__package__='trottertracks')
            path='src/trottertracks/rpe_pauli_signal_intake.py'
            exec(compile(blob(PUBLIC_B,path),PUBLIC_B+':'+path,'exec'),scope)
            with metadata_only(),self.assertRaisesRegex(ValueError,'entry code differs'):
                scope['accept_saved'](repo.root,integration=C.interfaces(str(repo.root)).pipeline,handoff_source=ref)
            api=b2_api();api.COMMON.update(view=repo.root,X=NS(A=A,T=T))
            with metadata_only(),self.assertRaisesRegex(ValueError,'Not a common-entry Pauli assembly'):
                api.common_metadata(dict(signal_packet=ref))
        finally:repo.close()

    def test_modified_descriptor_source_law_and_missing_round_rejected(self):
        repo=Repository(A.DESCRIPTOR_ENTRY_SOURCE)
        try:
            config,h,ref=fixture(repo,6,A.DESCRIPTOR_ENTRY_SOURCE)
            with metadata_only():
                for mutate in ('law','raw','configuration','round'):
                    bad=deepcopy(h)
                    if mutate=='law':bad['packets'][0]['rounds'][0]['law_binding']={}
                    elif mutate=='raw':bad['packets'][0]['rounds'][0]['mean_record']['signal']['raw']['real']=.75
                    elif mutate=='configuration':bad['configuration_source']=configuration_source(4)
                    else:bad['packets'][0]['rounds'].pop()
                    bad_ref=repo.save(bad,'bad.json')
                    with self.assertRaises(ValueError):A.accept_saved(repo.root,integration=C.interfaces(str(repo.root)).pipeline,handoff_source=bad_ref)
                dep=repo.root/'src/trottertracks/pauli_split_descriptor.py'
                dep.write_bytes(dep.read_bytes()+b'\n# incompatible dependency\n')
                bad_ref=repo.save(h,'changed_descriptor.json')
                with self.assertRaisesRegex(ValueError,'entry code differs'):
                    A.accept_saved(repo.root,integration=C.interfaces(str(repo.root)).pipeline,handoff_source=bad_ref)
        finally:repo.close()


if __name__=='__main__':unittest.main(verbosity=2)
