"""Artificial saved metadata and prices; no real H4 computation or QPY parsing."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

import reassess_full_h_m0_zero_zz as R
from check_full_h_reference_scenario import fixture, stopped_fixture


def packet():
    rule,_,_,_=stopped_fixture();rule['physical_candidate']['s']=1
    for unit in rule['units']:unit['admission']=dict(mock='original_C',axis=unit['axis'])
    policy=dict(rule_id=R.RULE_ID,original_rule=R.C.RULE,
        alternative_parity_evidence={'applies_only_to_independent_N_zero':True})
    prior=dict(status='HELD_NO_TOTAL',T=None,m0_partial=None)
    checks=[dict(phase_alignment=False,check=str(i)) for i in range(7)]
    evidence=dict(physical=rule['physical_candidate'],cohort='mock',path_index=0,seeds=[17],independent_paths=1,
        execution_source='mockproducer',result_commit='mockpublication',D1_handoff=R.D1['commit'],
        zero_target_evidence=dict(target_count_before_substitution=0,target_counter_used_as_basis=False,
            no_deterministic_blocks=0,full_wrapper_scope='artificial full scope',applications=[dict(role='rotation',
                is_identity=False,central_support=[1],basis_system_only=True,replacement_target=False,
                phase={'real':1.,'imag':0.})]),
        original_saved=dict(context_check='PASS',native_status='PASS',sampler_calls=1,composite_gate='STOP',
            parity_return='NOT_PERSISTED',scalar_event_return='NOT_PERSISTED',XY_return='NOT_PERSISTED',
            runtime_lowering_marker='NOT_PERSISTED',numeric_checks=checks),
        current_read_only_checks=dict(legacy_new_equivalence='FIXED_SOURCE_BRANCH_ARGUMENT_ONLY_NO_LEGACY_CIRCUIT_COMPARISON',
            phase_and_XY=dict(status='PASS',same_controlled_evolution=True,axes={}),axes=[]))
    handoff={k:deepcopy(evidence[k]) for k in ('physical','cohort','path_index','seeds','independent_paths','execution_source','result_commit')}
    handoff['axis_rows']=[];objects={};blobs={}
    handoff['rule_source']=R.C.RULE
    for axis,value in (('X',7),('Y',11)):
        receipt=dict(compiler='mockcompiler',fingerprint=axis,metrics={'rz_count':value})
        native=dict(status='PASS',checks=checks,preparation_hash='mockprep',scalar=-1)
        objects[axis+'receipt']=receipt;objects[axis+'native']=native
        blobs[axis+'receipt']=json.dumps(receipt).encode();blobs[axis+'wrapper']=b'artificial QPY bytes'+axis.encode()
        blobs[axis+'compiled']=b'artificial compiled QPY bytes'+axis.encode()
        refs={kind:dict(commit='mockpublication',path=axis+path,sha256=hashlib.sha256(blobs[axis+path]).hexdigest())
              for kind,path in (('compile_record','receipt'),('wrapper','wrapper'),('compiled_circuit','compiled'))}
        phase=dict(status='PASS',scalar=-1,expected_scalar=-1,ancilla_p=1,event_phases=[],legacy_scalar_once=True)
        row=dict(physical=rule['physical_candidate'],physical_id='mock',axis=axis,m=0,cohort='mock',path_index=0,
            seeds=[17],source={'commit':'mockproducer'},per_shot_RZ=None,observed_compiled_RZ=value,fingerprint=axis,
            measurement=dict(axis=axis,include_measurement=True),native_trial=dict(path=axis+'native'),**refs)
        row['admission']=dict(mock='original_C',axis=axis)
        handoff['axis_rows'].append(row)
        evidence['current_read_only_checks']['axes'].append(dict(axis=axis,compiler_profile=rule['profile'],
            receipt=refs['compile_record'],wrapper=refs['wrapper'],compiled_circuit=refs['compiled_circuit'],fingerprint=axis,
            observed_RZ_quoted_from_original_record=value,measurement=[[596 if axis=='X' else 597,[8],[0]]],
            all_operations_at_most_two_wires=True,
            unique_top_level_operation_names=['crz','h','measure','p','xx_plus_yy']+(['sdg'] if axis=='Y' else []),
            global_phase=0.,current_read_only_phase_check=phase,original_implementation_gate_return='NOT_PERSISTED'))
        evidence['current_read_only_checks']['phase_and_XY']['axes']['cosine' if axis=='X' else 'sine']=phase
    return [rule,policy,evidence,handoff,prior],objects,blobs


def read(args,objects,blobs):
    with patch.object(R.C.G,'evidence',side_effect=lambda root,ref:objects[ref['path']]), \
         patch.object(R.C.G.C,'blob',side_effect=lambda root,commit,path:blobs[path]):
        return R.read_and_reassess(None,*args)


class Tests(unittest.TestCase):
    def test_real_reader_route_then_existing_account_with_artificial_prices(self):
        args,objects,blobs=packet();before=deepcopy(args)
        rows,admissions=read(args,objects,blobs)
        self.assertEqual([r['accepted_per_shot_RZ'] for r in admissions],[7,11])
        self.assertEqual([r['original_parity_return'] for r in admissions],['NOT_PERSISTED']*2)
        self.assertTrue(all('ALTERNATIVE' in r['current_integrity_evidence'] for r in admissions))
        self.assertEqual(args,before)
        all_rows,benchmarks=fixture();all_rows[:2]=rows
        result=R.C.account(all_rows,benchmarks)
        self.assertEqual(result['m0_partial'],R.C.exact(96*(7+11)))
        self.assertEqual(result['T'],R.C.exact(96*(7+11)+24))
        self.assertTrue(all(r['expectation']=='FINITE_COHORT_ONLY' and not r['qualified_E_C'] for r in rows))

    def test_unknown_or_nonzero_or_counter_only_is_not_not_applicable(self):
        for mode in ('unknown','positive','counter','missing_scope','two_wire'):
            args,objects,blobs=packet();z=args[2]['zero_target_evidence']
            if mode=='unknown':z['target_count_before_substitution']=None
            if mode=='positive':z['target_count_before_substitution']=1
            if mode=='counter':z['target_counter_used_as_basis']=True
            if mode=='missing_scope':z.pop('full_wrapper_scope')
            if mode=='two_wire':z['applications'][0]['central_support']=[1,2]
            rows,admissions=read(args,objects,blobs)
            self.assertTrue(all(r['accepted_per_shot_RZ'] is None for r in admissions))
            all_rows,benchmarks=fixture();all_rows[:2]=rows
            self.assertIsNone(R.C.account(all_rows,benchmarks)['T'])

    def test_missing_other_conditions_holds_axis_and_nulls_total(self):
        for mode in ('receipt_bytes','scalar','axis','missing_phase','missing_numeric','wrong_source'):
            args,objects,blobs=packet();check=args[2]['current_read_only_checks']['axes'][0]
            if mode=='receipt_bytes':blobs['Xreceipt']=b'changed'
            if mode=='scalar':check['current_read_only_phase_check']['scalar']=0
            if mode=='axis':check['measurement']=[[596,[7],[0]]]
            if mode=='missing_phase':check.pop('current_read_only_phase_check')
            if mode=='missing_numeric':objects['Xnative']['checks']=[]
            if mode=='wrong_source':args[3]['axis_rows'][0]['source']['commit']='other'
            rows,admissions=read(args,objects,blobs)
            self.assertIsNone(admissions[0]['accepted_per_shot_RZ'])
            self.assertEqual(admissions[1]['accepted_per_shot_RZ'],11)
            all_rows,benchmarks=fixture();all_rows[:2]=rows
            with patch.object(R.C,'rational',side_effect=AssertionError('Do not sum a held scenario')):
                self.assertIsNone(R.C.account(all_rows,benchmarks)['T'])

    def test_coverage_adoption_does_not_replace_missing_alternative_evidence(self):
        args,objects,blobs=packet()
        args[2]['current_read_only_checks']['legacy_new_equivalence']='UNKNOWN'
        _,admissions=read(args,objects,blobs)
        self.assertTrue(all(r['accepted_per_shot_RZ'] is None for r in admissions))


if __name__=='__main__':unittest.main()
