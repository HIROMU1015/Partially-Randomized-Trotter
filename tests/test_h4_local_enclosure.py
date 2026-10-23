"""Artificial local-ball and authority tests; never evaluate real H4 matrices."""
from copy import deepcopy
from fractions import Fraction as Q
import cmath
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


n = load("local_numerical_test", "src/trotterlib/h4_local_enclosure.py")
c = load("local_authority_test", "src/trotterlib/h4_local_control.py")


def raw_gate(name="rz", theta=.5, beta=.25, inverse=False):
    theta *= -1 if inverse else 1
    cos, sin = math.cos(theta/2), math.sin(theta/2)
    a = ([[cmath.exp(-.5j*theta), 0], [0, cmath.exp(.5j*theta)]] if name == "rz" else
         [[1, 0, 0, 0], [0, cos, -1j*sin*cmath.exp(-1j*beta), 0],
          [0, -1j*sin*cmath.exp(1j*beta), cos, 0], [0, 0, 0, 1]])
    return [[[float(complex(x).real), float(complex(x).imag)] for x in row] for row in a]


def artificial_model():
    ops, registry = [], {}
    for name, theta, qubits in [("rz", .5, [0]), ("xx_plus_yy", .75, [0, 1])]:
        raw, inv = raw_gate(name, theta), raw_gate(name, theta, inverse=True)
        h = n.matrix_hash(raw)
        params = [str(theta)] + (["0.25"] if name != "rz" else [])
        d = dict(name=name, parameters=params, qubits=qubits, matrix_sha256=h)
        ops.append(dict(description=d, qubits=qubits, matrix=raw, inverse_matrix=inv))
        registry[h] = dict(name=name, parameters=params)
    basis = dict(basis_hash=n.C.canonical(dict(num_system_qubits=2, operations=[x["description"] for x in ops],
                                              hash_policy="ordered_local_basis_operations_v1")), operations=ops)
    return dict(test_fixture_domain="MANUALLY_SUPPLIED_ARTIFICIAL", representation="df_native", qubit_order="qiskit_little_endian", n_qubits=2,
                cases=dict(PF=dict(blocks=[dict(basis=basis)], components=[]),
                           PR=dict(blocks=[dict(basis=basis)], components=[dict(basis=basis, support=[0, 1], sign=-1)]))), dict(local_gates=registry)


class ArtificialBounds(unittest.TestCase):
    def setUp(self):
        self.context = n.C.ctx.workprec(128)
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)

    def test_zero_RZ_and_XX_exact_identity(self):
        for name, dim in [("rz", 2), ("xx_plus_yy", 4)]:
            d = dict(name=name, parameters=["0"] + (["0.25"] if dim == 4 else []))
            self.assertTrue(n.ideal_gate(d).contains(n.C.identity(dim)))
            self.assertEqual(n.discrepancy(d, raw_gate(name, 0)), 0)

    def test_forward_inverse_contain_exact_identity(self):
        for name in ["rz", "xx_plus_yy"]:
            d = dict(name=name, parameters=["0.5"] + (["0.25"] if name != "rz" else []))
            a, inv = n.ideal_gate(d), n.ideal_gate(d, inverse=True)
            self.assertTrue((a*inv).contains(n.C.identity(a.nrows())))

    def test_local_difference_covers_float_fixture(self):
        d = dict(name="rz", parameters=["0.5"])
        upper = n.discrepancy(d, raw_gate())
        self.assertGreater(upper, 0)
        self.assertLess(upper, Q(1, 10**12))

    def test_wrong_phase_sign_detected(self):
        d = dict(name="xx_plus_yy", parameters=["0.5", "0.25"])
        wrong = raw_gate("xx_plus_yy", .5, -.25)
        self.assertGreater(n.discrepancy(d, wrong), Q(1, 100))

    def test_wrong_inverse_detected(self):
        d = dict(name="rz", parameters=["0.5"])
        self.assertGreater(n.discrepancy(d, raw_gate(), inverse=True), Q(1, 10))

    def test_inverse_preserves_beta(self):
        d = dict(name="xx_plus_yy", parameters=["0.75", "0.25"])
        self.assertLess(n.discrepancy(d, raw_gate("xx_plus_yy", .75, .25, inverse=True), inverse=True), Q(1, 10**12))

    def test_fixed_precision_rejects64(self):
        with n.C.ctx.workprec(64):
            with self.assertRaisesRegex(ValueError, "128_BIT"):
                n.discrepancy(dict(name="rz", parameters=["0"]), raw_gate(theta=0))

    def test_unsupported_gate_rejected(self):
        with self.assertRaisesRegex(ValueError, "UNSUPPORTED"):
            n.ideal_gate(dict(name="general_unitary", parameters=["0"]))

    def test_nonfinite_parameter_rejected(self):
        for value in ["nan", "inf", "1e999", True]:
            with self.assertRaises(ValueError):
                n.finite_float(value)

    def test_nonfinite_matrix_rejected(self):
        a = raw_gate();a[0][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NONFINITE"):
            n.matrix_hash(a)

    def test_wrong_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHAPE"):
            n.matrix_hash([[[1., 0.]]])

    def test_matrix_SHA_changes_on_perturbation(self):
        a = raw_gate();old = n.matrix_hash(a);a[0][0][0] += .01
        self.assertNotEqual(n.matrix_hash(a), old)

    def test_exact_rational_telescoping_oracle(self):
        errors = [Q(1, 10), Q(1, 5), Q(1, 100)]
        exact=Q(11, 10)*Q(6, 5)*Q(101, 100)-1
        self.assertGreaterEqual(n.product_bound(errors),exact)
        self.assertLess(n.product_bound(errors)-exact,Q(1,2**120))

    def test_negative_bound_rejected(self):
        with self.assertRaisesRegex(ValueError, "NONNEGATIVE"):
            n.product_bound([Q(-1)])

    def test_noncommuting_composition_and_inverse_order(self):
        m, _ = artificial_model();ops = m["cases"]["PF"]["blocks"][0]["basis"]["operations"]
        a = [n.C.embed(n.ideal_gate(x["description"]), x["qubits"], 2) for x in ops]
        self.assertFalse((a[0]*a[1]-a[1]*a[0]).contains(n.C.acb_mat(4, 4)))
        self.assertTrue((n.C.dagger(a[0])*n.C.dagger(a[1])*a[1]*a[0]).contains(n.C.identity(4)))

    def test_compositional_bound_covers_each_basis_vector(self):
        m, registry = artificial_model();plan = n.plan_from_saved(m, registry, artificial=True)
        ops = next(iter(plan["bases"].values()));raw = n.C.identity(4);ideal = n.C.identity(4);errors = []
        for op in ops:
            gate = plan["gates"][op["gate_hash"]]
            raw = n.C.embed(n.C.matrix(gate["forward"]), op["qubits"], 2)*raw
            ideal = n.C.embed(n.ideal_gate(gate["descriptor"]), op["qubits"], 2)*ideal
            errors.append(n.discrepancy(gate["descriptor"], gate["forward"]))
        bound = n.C.real(n.product_bound(errors))
        for col in range(4):
            difference = n.C.acb_mat([[raw[i,col]-ideal[i,col]] for i in range(4)])
            self.assertTrue(n.C.frobenius(difference).upper() <= bound)

    def test_control_zero_one_same_telescoping_bound(self):
        self.assertGreaterEqual(n.conjugation_bound(Q(1,10), Q(1,20)), Q(31,200))
        self.assertLess(n.conjugation_bound(Q(1,10), Q(1,20))-Q(31,200),Q(1,2**120))
        self.assertEqual(n.signed_controlled_identity(-1), dict(zero_branch=1, one_branch=-1))

    def test_prototype_interval_oracle_reused(self):
        enclosed = n.P.exponential_prototype([[0,Q(-1,4)],[Q(1,4),0]],degree=12)
        expected = n.ideal_gate(dict(name="rz", parameters=["0.5"]))
        for fixture, ball in [(enclosed[0][0], expected[0,0].real), (enclosed[1][0], -expected[0,0].imag)]:
            interval = n.C.real(fixture.lo).union(n.C.real(fixture.hi))
            self.assertTrue(interval.contains(ball))


class PlanAndCoverage(unittest.TestCase):
    def test_artificial_plan_matches_descriptor_registry(self):
        m,r = artificial_model();p=n.plan_from_saved(m,r,artificial=True)
        self.assertEqual((len(p["gates"]),len(p["bases"]),len(p["supports"])),(2,1,1))

    def test_artificial_does_not_pass_real_counts(self):
        m,r=artificial_model()
        with self.assertRaisesRegex(ValueError,"FIXED_H4_SCOPE"):
            n.plan_from_saved(m,r)

    def test_bad_descriptor_SHA_rejected(self):
        m,r=artificial_model();m['cases']['PF']['blocks'][0]['basis']['operations'][0]['description']['matrix_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'FORWARD_HASH'):
            n.plan_from_saved(m,r,artificial=True)

    def test_reordered_basis_rejected(self):
        m,r=artificial_model();m['cases']['PF']['blocks'][0]['basis']['operations'].reverse()
        with self.assertRaisesRegex(ValueError,'ORDERED_BASIS_HASH'):
            n.plan_from_saved(m,r,artificial=True)

    def test_bad_support_rejected(self):
        m,r=artificial_model();m['cases']['PR']['components'][0]['support']=[0,0]
        with self.assertRaisesRegex(ValueError,'SYMBOLIC_COMPONENT'):
            n.plan_from_saved(m,r,artificial=True)

    def test_bad_component_sign_rejected(self):
        m,r=artificial_model();m['cases']['PR']['components'][0]['sign']=0
        with self.assertRaisesRegex(ValueError,'COMPONENT_SIGN'):
            n.plan_from_saved(m,r,artificial=True)

    def test_coverage_detects_semantic_overlap(self):
        records=[dict(id='C',covers=['bias_distance','basis_reuse','phase_accumulation','pi_rounding'],bound_ref='saved total'),
                 dict(id='L',covers=['local_forward','local_inverse','basis_reuse'],bound_ref='new uniform')]
        ledger=n.coverage_ledger(records)
        self.assertEqual(ledger['overlapping_primitives'],{'basis_reuse':['C','L']})
        self.assertFalse(ledger['simple_addition_permitted'])

    def test_coverage_does_not_cancel_missing_terms(self):
        ledger=n.coverage_ledger([dict(id='L',covers=['local_forward','local_inverse'],bound_ref='L')])
        self.assertIn('phase_accumulation',ledger['uncovered_primitives'])
        self.assertFalse(ledger['simple_addition_permitted'])

    def test_duplicate_coverage_proof_rejected(self):
        x=dict(id='same',covers=['local_forward'],bound_ref='x')
        with self.assertRaisesRegex(ValueError,'DUPLICATE_COVERAGE'):
            n.coverage_ledger([x,x])

    def test_artificial_diagnosis_and_no_automatic_acceptance(self):
        m,r=artificial_model();plan=n.plan_from_saved(m,r,artificial=True);outputs={};calls=[]
        class Budget:
            def take(self,name):calls.append(name)
        result=n.diagnose(plan,Budget(),emit=lambda name,rec:outputs.__setitem__(name,rec),authorize_before_call=lambda:None)
        self.assertEqual(calls.count('forward'),2);self.assertEqual(calls.count('inverse'),2)
        self.assertEqual(calls.count('basis'),1);self.assertEqual(calls.count('support'),1)
        self.assertFalse(result['formal_certificate_accepted'])
        self.assertEqual(result['coverage_ledger']['status'],'COMBINATION_BLOCKED')

    def test_authority_failure_before_call_reserves_nothing(self):
        m,r=artificial_model();plan=n.plan_from_saved(m,r,artificial=True);calls=[]
        class Budget:
            def take(self,name):calls.append(name)
        with self.assertRaisesRegex(ValueError,'LOST_AUTHORITY'):
            n.diagnose(plan,Budget(),emit=lambda *_:None,authorize_before_call=lambda:(_ for _ in ()).throw(ValueError('LOST_AUTHORITY')))
        self.assertEqual(calls,[])

    def test_failed_local_call_preserves_nonrefundable_reservation(self):
        m,r=artificial_model();plan=n.plan_from_saved(m,r,artificial=True)
        with tempfile.TemporaryDirectory() as tmp:
            budget=c.C.CallBudget(tmp,c.LIMITS)
            with patch.object(n,'discrepancy',side_effect=ValueError('BAD_BOUND')):
                with self.assertRaisesRegex(ValueError,'BAD_BOUND'):
                    n.diagnose(plan,budget,emit=lambda *_:None,authorize_before_call=lambda:None)
            self.assertEqual(c.read(Path(tmp)/'call_ledger.json')['used']['forward'],1)


def inactive_contract():
    return dict(kind='H4_L_PRODUCTION_CONTRACT_V1',domain='ARTIFICIAL',precision_bits=128,
                limits=dict(c.LIMITS),forbidden_calls={k:0 for k in c.ZERO},expected_counts=dict(gates=148,bases=11,supports=324),
                automatic_retry=False,mandatory_STOP=True,execution_authorized=False,grant_valid=False,dispatch_enabled=False,
                formal_certificate_accepted=False,N_G_authorized=False,historical_compiler_target='ORIGINAL_PROGRAM_UNCHANGED_MISSING_EVIDENCE')


class Authority(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def active_artificial(self):
        ns=self.root/'new_namespace';contract=inactive_contract();contract.update(execution_authorized=True,grant_valid=True,dispatch_enabled=True)
        model,registry=artificial_model();c.save(self.root/'model.json',model);c.save(self.root/'registry.json',registry)
        inputs={str(p):c.sha(p) for p in [self.root/'model.json',self.root/'registry.json']}
        seal=dict(SOURCE='a'*40,domain='ARTIFICIAL',input_files=inputs,model_path=str(self.root/'model.json'),registry_path=str(self.root/'registry.json'))
        c.save(self.root/'seal.json',seal)
        approval=dict(kind='L_EXPLICIT_NUMERICAL_EXECUTION_APPROVAL',domain='ARTIFICIAL',technical_seal_sha256=c.sha(self.root/'seal.json'),
            namespace=str(ns),input_sha256=inputs,scope='DF_LOCAL_ONLY_FIXED128_NO_COMPILE_ACCEPTANCE_OR_NG')
        c.save(self.root/'approval.json',approval)
        contract.update(approval_file=str(self.root/'approval.json'),approval_sha256=c.sha(self.root/'approval.json'),
            technical_SOURCE='a'*40,technical_seal_sha256=c.sha(self.root/'seal.json'),input_sha256=inputs,model_path=seal['model_path'],registry_path=seal['registry_path'],output_namespace=str(ns))
        c.save(self.root/'contract.json',contract)
        return ns,contract,seal

    def test_inactive_contract(self):c.validate_contract(inactive_contract())

    def test_inactive_grant_never_issued(self):
        c.save(self.root/'contract.json',inactive_contract())
        with self.assertRaisesRegex(ValueError,'NOT_AUTHORIZED'):
            c.issue_grant(self.root/'contract.json',self.root/'absent_seal',self.root/'not_created')
        self.assertFalse((self.root/'not_created').exists())

    def test_changed_precision_and_budget_rejected(self):
        contract=inactive_contract();contract['precision_bits']=256
        with self.assertRaisesRegex(ValueError,'FIXED_PRECISION'):c.validate_contract(contract)
        contract=inactive_contract();contract['limits']['forward']=149
        with self.assertRaisesRegex(ValueError,'CALL_BUDGET'):c.validate_contract(contract)

    def test_forbidden_science_and_retry_rejected(self):
        for key in c.ZERO:
            contract=inactive_contract();contract['forbidden_calls'][key]=1
            with self.assertRaisesRegex(ValueError,'FORBIDDEN'):c.validate_contract(contract)
        contract=inactive_contract();contract['automatic_retry']=True
        with self.assertRaisesRegex(ValueError,'RETRY'):c.validate_contract(contract)

    def test_acceptance_or_NG_cannot_be_enabled(self):
        for key in ['formal_certificate_accepted','N_G_authorized']:
            contract=inactive_contract();contract[key]=True
            with self.assertRaisesRegex(ValueError,'NO_ACCEPTANCE'):c.validate_contract(contract)

    def test_each_active_flag_required(self):
        _,contract,_=self.active_artificial()
        for key in ['execution_authorized','grant_valid','dispatch_enabled']:
            changed=deepcopy(contract);changed[key]=False
            with self.assertRaisesRegex(ValueError,'NOT_AUTHORIZED'):c.validate_contract(changed,active=True)

    def test_approval_scope_rejected(self):
        _,contract,_=self.active_artificial();c.save(self.root/'wrong_approval.json',{})
        contract.update(approval_file=str(self.root/'wrong_approval.json'),approval_sha256=c.sha(self.root/'wrong_approval.json'))
        with self.assertRaisesRegex(ValueError,'APPROVAL_SCOPE'):c.validate_contract(contract,active=True)

    def test_artificial_grant_consume_duplicate_and_STOP(self):
        ns,_,_=self.active_artificial()
        with patch.object(c,'verify_seal'):
            grant=c.issue_grant(self.root/'contract.json',self.root/'seal.json',ns)
            self.assertEqual(grant['kind'],'L_ARTIFICIAL_ONE_SHOT')
            c.consume(ns)
            with self.assertRaisesRegex(ValueError,'ALREADY_USED'):c.consume(ns)
            c.save(ns/'STOP.json',{'mandatory_STOP':True})
            with self.assertRaisesRegex(ValueError,'ALREADY_STOPPED'):c.authority(ns,worker=True)

    def test_duplicate_launch_namespace_rejected(self):
        ns,_,_=self.active_artificial()
        with patch.object(c,'verify_seal'):
            c.issue_grant(self.root/'contract.json',self.root/'seal.json',ns)
            with self.assertRaisesRegex(ValueError,'FRESH_NAMESPACE'):c.issue_grant(self.root/'contract.json',self.root/'seal.json',ns)

    def test_worker_requires_consumption_and_claim(self):
        ns,_,_=self.active_artificial()
        with patch.object(c,'verify_seal'):
            c.issue_grant(self.root/'contract.json',self.root/'seal.json',ns)
            with self.assertRaises((ValueError,FileNotFoundError)):c.authority(ns,worker=True)

    def test_worker_artificial_end_to_end(self):
        ns,_,_=self.active_artificial()
        with patch.object(c,'verify_seal'):
            c.issue_grant(self.root/'contract.json',self.root/'seal.json',ns);c.consume(ns);c.worker(ns)
        self.assertEqual(c.read(ns/'worker_result.json')['domain'],'ARTIFICIAL')
        self.assertTrue((ns/'worker_STOP.json').exists())
        self.assertFalse(c.read(ns/'worker_result.json')['formal_certificate_accepted'])

    def test_worker_rechecks_authority_after_numbers(self):
        ns,_,_=self.active_artificial()
        with patch.object(c,'verify_seal'):
            c.issue_grant(self.root/'contract.json',self.root/'seal.json',ns);c.consume(ns)
            original=c.authority;calls=[]
            def changed(*args,**kwargs):
                calls.append(1)
                if len(calls)>1:raise ValueError('AUTHORITY_LOST_AFTER_CALLS')
                return original(*args,**kwargs)
            with patch.object(c,'authority',side_effect=changed):
                with self.assertRaisesRegex(ValueError,'AUTHORITY_LOST_AFTER_CALLS'):c.worker(ns)
        self.assertFalse((ns/'worker_result.json').exists())
        self.assertEqual(c.read(ns/'worker_STOP.json')['status'],'FAILED_PARTIALS_PRESERVED_NO_RETRY')
        self.assertTrue(any(ns.glob('local_*.json')))

    def test_artificial_seal_cannot_bind_real_domain(self):
        _,contract,seal=self.active_artificial();contract['domain']='SAVED_H4'
        with self.assertRaisesRegex(ValueError,'DOMAIN_MISMATCH'):c.routing(contract,seal)

    def test_hash_change_after_consumption_rejected(self):
        ns,_,_=self.active_artificial()
        with patch.object(c,'verify_seal'):
            c.issue_grant(self.root/'contract.json',self.root/'seal.json',ns);c.consume(ns)
            changed=c.read(ns/'contract.json');changed['model_path']='/changed';c.save(ns/'contract.json',changed,exclusive=False)
            with self.assertRaisesRegex(ValueError,'GRANT_BINDING'):c.authority(ns,worker=True)

    def test_runtime_and_source_seal_reject_before_numbers(self):
        with self.assertRaisesRegex(ValueError,'SEAL_KIND'):c.verify_seal({'kind':'wrong'})
        with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):c.C.hashes({'/nonexistent/artificial-file':'a'*64})

    def test_budget_exhaustion_and_forbidden_call(self):
        budget=c.C.CallBudget(self.root,{'forward':1})
        budget.take('forward')
        with self.assertRaisesRegex(ValueError,'EXHAUSTED'):budget.take('forward')
        with self.assertRaisesRegex(ValueError,'FORBIDDEN'):budget.take('reference')

    def test_cli_inactive_before_numerical_import(self):
        c.save(self.root/'contract.json',inactive_contract())
        result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_h4_local_enclosure.py'),'--issue-grant',str(self.root/'contract.json'),
             '--seal',str(self.root/'absent'),'--namespace',str(self.root/'never_created')],capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('NOT_AUTHORIZED',result.stderr)
        self.assertFalse((self.root/'never_created').exists())

    def test_native_success_and_failure_receipts(self):
        for exit_code in [0,3]:
            directory=self.root/('native_'+str(exit_code))
            command=[sys.executable,'-B','-c','raise SystemExit('+str(exit_code)+')']
            result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_h4_local_enclosure.py'),'--witness',str(directory),'--command-json',json.dumps(command)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0 if exit_code==0 else 1)
            receipt=c.read(directory/'worker_native_exit.json')
            self.assertEqual(receipt['exit_code'],exit_code);self.assertTrue(receipt['reaped'])
            self.assertEqual(receipt['scope'],'NEW_L_WORKER_ONLY')


if __name__=='__main__':unittest.main(verbosity=2)
