"""Off-domain metadata/mock/2x2 controls; no saved scientific sequences."""
from fractions import Fraction as F
import copy
import math
import unittest

from trottertracks.algorithm_codesign.parent_allocation import rotation_key, check_cache_metadata
from trottertracks.algorithm_codesign.parent_allocation_native import (
    sha, canonical, bound_json, tool_binding, legacy_spec, bridge_legacy_row,
    interval_angle, strict_primitive_guard, recheck_entry,
    native_accounting, realized_time_gates, operator_target)


def fixture(sequence='HSTW', ratio=F(2,11)):
    raw=canonical({'python':'off-domain', 'packages':{'pygridsynth':'mock-only'},
                   'source_py_tree_sha256':{'pygridsynth':'1'*64}})
    binding=tool_binding(raw, expected_tool_sha256=sha(raw),runtime_executable_sha256='2'*64,
                         options={'seed':71,'up_to_phase':False,'dps':50}, interval_dps=60)
    event={'ratio':ratio,'event_kind':'rotation'}
    expected=rotation_key(event,precision=F(1,10**7),tool_identity=binding)
    angle='atan:'+str(ratio)+':scale:1'
    row=dict(key=angle+':epsilon:1/10000000',angle_key=angle,epsilon='1/10000000',
             sequence=sequence,sequence_sha256=sha(sequence.encode()),
             T_count=sequence.count('T')+sequence.count('t'),Tdagger_count=sequence.count('t'),
             global_W_count=sequence.count('W'),one_qubit_count=len(sequence)-sequence.count('W'),
             strict_operator_error_upper='1/1000000000',error_pass=True)
    spec=legacy_spec(expected)
    provenance=dict(source_commit='3'*40,result_commit='4'*40,cache_sha256='5'*64,
                    saved_row_sha256=sha(canonical(row)),contract_sha256='6'*64,
                    inventory_sha256='7'*64,guard_audit_sha256='8'*64,
                    strict_guard_source_sha256='9'*64,tool_identity_sha256=sha(raw),
                    runtime_executable_sha256='2'*64,inventory_spec=spec,inventory_key=sha(canonical(spec)),
                    guard_policy='STRICT_RTE_RZ_FROBENIUS_NO_PHASE_MINIMIZATION')
    return row,expected,provenance


def mock_checked(sequence='HSTW', ratio=F(2,11)):
    row,key,p=fixture(sequence,ratio)
    entry=bridge_legacy_row(row,key,provenance=p)
    return recheck_entry(entry,key,guard=lambda *args,**kwargs:F(1,10**9))


def mul(a,b):
    return [[sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def adjoint(a):
    return [[a[j][i].conjugate() for j in range(2)] for i in range(2)]


def matrix(letter):
    w=complex(math.cos(math.pi/4),math.sin(math.pi/4))
    h=1/math.sqrt(2)
    return {'H':[[h,h],[h,-h]],'X':[[0j,1+0j],[1+0j,0j]],
            'S':[[1+0j,0j],[0j,1j]],'s':[[1+0j,0j],[0j,-1j]],
            'T':[[1+0j,0j],[0j,w]],'t':[[1+0j,0j],[0j,w.conjugate()]],
            'W':[[w,0j],[0j,w]],'w':[[w.conjugate(),0j],[0j,w.conjugate()]]}[letter]


def time_product(gates):
    out=[[1+0j,0j],[0j,1+0j]]
    for g in gates:out=mul(matrix(g[0]),out)
    return out


class BindingControls(unittest.TestCase):
    def test_raw_json_identity(self):
        raw=b'{"a":1}'
        self.assertEqual(bound_json(raw,expected_bytes=len(raw),expected_sha256=sha(raw)),{'a':1})
        for n,h in [(len(raw)+1,sha(raw)),(len(raw),'0'*64)]:
            with self.assertRaises(ValueError):bound_json(raw,expected_bytes=n,expected_sha256=h)

    def test_bridge_not_operator_validation(self):
        row,key,p=fixture();e=bridge_legacy_row(row,key,provenance=p)
        self.assertTrue(e['metadata_only']);self.assertFalse(e['strict_guard_rechecked'])
        self.assertFalse(e['eligible_for_guarded_reuse'])
        with self.assertRaises(ValueError):native_accounting([('R',0,'2/11',1)],{'2/11':e})

    def test_options_and_interval_precision_bind_key(self):
        _,key,_=fixture();tool=key['spec']['tool']
        for field,value in [('seed',72),('interval_dps',70),('runtime_executable_sha256','f'*64)]:
            changed=copy.deepcopy(tool);changed[field]=value
            other=rotation_key({'ratio':F(2,11)},precision=F(1,10**7),tool_identity=changed)
            self.assertNotEqual(key['key'],other['key'])
        changed=copy.deepcopy(tool);changed['synthesizer_options']['dps']=51
        self.assertNotEqual(key['key'],rotation_key({'ratio':F(2,11)},precision=F(1,10**7),tool_identity=changed)['key'])

    def test_projective_guard_receipt_rejected(self):
        row,key,p=fixture();p['guard_policy']='PROJECTIVE'
        with self.assertRaises(ValueError):bridge_legacy_row(row,key,provenance=p)

    def test_missing_or_altered_provenance_rejected(self):
        row,key,p=fixture()
        for field in ('cache_sha256','source_commit','guard_audit_sha256'):
            other=copy.deepcopy(p);other[field]='bad'
            with self.assertRaises(ValueError):bridge_legacy_row(row,key,provenance=other)
        for field in ('tool_identity_sha256','runtime_executable_sha256','inventory_key','saved_row_sha256'):
            other=copy.deepcopy(p);other[field]='a'*64
            with self.assertRaises(ValueError):bridge_legacy_row(row,key,provenance=other)

    def test_old_counts_or_key_cannot_be_repriced_silently(self):
        row,key,p=fixture()
        for field,value in [('T_count',10),('Tdagger_count',1),('one_qubit_count',1),
                            ('global_W_count',0),('epsilon','1/1000000'),('error_pass',False)]:
            other=dict(row,**{field:value});prov=dict(p,saved_row_sha256=sha(canonical(other)))
            with self.assertRaises(ValueError):bridge_legacy_row(other,key,provenance=prov)

    def test_changed_raw_sequence_fails_hash(self):
        row,key,p=fixture();row['sequence']='T';p['saved_row_sha256']=sha(canonical(row))
        with self.assertRaises(ValueError):bridge_legacy_row(row,key,provenance=p)

    def test_missing_expanded_tool_identity_rejected(self):
        row,key,p=fixture();del key['spec']['tool']['package_lock_sha256']
        with self.assertRaises(ValueError):bridge_legacy_row(row,key,provenance=p)

    def test_options_seed_conflict_rejected(self):
        _,key,_=fixture();key['spec']['tool']['seed']=72
        with self.assertRaises(ValueError):legacy_spec(key)

    def test_guard_failure_or_exception_cannot_become_success(self):
        row,key,p=fixture();e=bridge_legacy_row(row,key,provenance=p)
        with self.assertRaises(ValueError):recheck_entry(e,key,guard=lambda *a,**k:F(1))
        def broken(*a,**k):raise ArithmeticError('off-domain guard failure')
        with self.assertRaises(ArithmeticError):recheck_entry(e,key,guard=broken)
        self.assertFalse(e['eligible_for_guarded_reuse'])

    def test_post_guard_error_tamper_detected(self):
        e=mock_checked();e['operator_error_upper']='1/2000000000'
        with self.assertRaises(ValueError):native_accounting([('R',0,'2/11',1)],{'2/11':e})

    def test_empty_identity_sequence_is_supported_with_guard(self):
        row,key,p=fixture('',F(1,10**20));e=bridge_legacy_row(row,key,provenance=p)
        check_cache_metadata(e,key)
        e=recheck_entry(e,key)
        price=native_accounting([('R',0,str(F(1,10**20)),1)],{str(F(1,10**20)):e})
        self.assertEqual((price['T'],price['1Q']),(0,0))
        self.assertLess(price['strict_native_error_upper'],F(1,10**7))


class NativeConnectionControls(unittest.TestCase):
    def test_operator_target_handles_signed_endpoints_and_noncommuting_word(self):
        from trottertracks.algorithm_codesign.parent_allocation import ParentAllocation,native_plan
        from test_parent_allocation import I,Q,multiply,scale,add,controlled,evaluate_plan,cq,cr,PRECISION,distance
        def rotation(i,r,s):
            phi=math.atan(float(r))
            return add(scale(I,math.cos(phi)),scale(Q[i],-1j*s*math.sin(phi)))
        for b,z in [((3,4),(4,3)),((3,4),(7,0)),((0,0),(7,0))]:
            for sign in (-1,1):
                a=ParentAllocation(7,b,z,word=(1,0),sigma=sign,**PRECISION)
                for i in range(len(a.atoms)):
                    e=a.event(i)
                    target=operator_target(e,identity=lambda:I,multiply=multiply,
                                           involution=lambda j:Q[j],rotation=rotation,scale=scale)
                    actual=evaluate_plan(native_plan(e,cq,cr,control=1))
                    self.assertLess(distance(actual,controlled(target)),1e-12)

    def test_operator_target_rejects_invalid_phase_before_provider_access(self):
        def fail(*a):self.fail('invalid event accessed operator provider')
        e={'phase_i_power':0,'event_kind':'pure_child','word':(),'child':0,'ratio':None}
        with self.assertRaises(ValueError):operator_target(e,identity=fail,multiply=fail,
                                                        involution=fail,rotation=fail,scale=fail)

    def test_signed_written_product_time_order_and_W(self):
        e=mock_checked();cache={'2/11':e};s=e['sequence']
        written=[[1+0j,0j],[0j,1+0j]]
        for c in s:written=mul(written,matrix(c))
        for sign,wanted in [(1,written),(-1,adjoint(written))]:
            gates=realized_time_gates([('R',0,'2/11',sign)],cache)
            actual=time_product(gates)
            self.assertLess(max(abs(actual[i][j]-wanted[i][j]) for i in range(2) for j in range(2)),1e-12)
            self.assertIn(('W' if sign==1 else 'w',0),gates)

    def test_error_sum_counts_every_occurrence_and_both_signs(self):
        e=mock_checked();ir=[('R',0,'2/11',1),('CX',1,0),('R',0,'2/11',-1),('S',1)]
        p=native_accounting(ir,{'2/11':e})
        self.assertEqual((p['T'],p['CX'],p['1Q']),(2,1,7))
        self.assertEqual(p['strict_native_error_upper'],F(2,10**9));self.assertFalse(p['forecast'])

    def test_endpoint_exact_gates_do_not_read_cache(self):
        p=native_accounting([('s',1),('CZ',1,0)],{})
        self.assertEqual((p['T'],p['CX'],p['1Q']),(0,1,3));self.assertEqual(p['strict_native_error_upper'],0)

    def test_missing_or_wrong_ratio_is_fatal(self):
        with self.assertRaises(KeyError):native_accounting([('R',0,'2/11',1)],{})
        e=mock_checked(ratio=F(3,13))
        with self.assertRaises(ValueError):native_accounting([('R',0,'2/11',1)],{'2/11':e})

    def test_bad_wire_sign_or_unknown_gate_rejected(self):
        for ir in [[('R',0,'2/11',True)],[('R',-1,'2/11',1)],[('CX',1,1)],[('UNKNOWN',1)]]:
            with self.assertRaises(ValueError):native_accounting(ir,{'2/11':mock_checked()})

    def test_interval_atan_is_certified_and_restores_context(self):
        import mpmath as mp
        old=mp.mp.dps,mp.iv.dps
        angle=interval_angle(F(1),dps=70)
        self.assertLess(angle['hi']-angle['lo'],F(1,10**65))
        self.assertEqual((mp.mp.dps,mp.iv.dps),old)

    def test_strict_guard_retains_relative_phase(self):
        import mpmath as mp
        from trottertracks.algorithm_codesign.rte_reallocation.native import Angle
        from trottertracks.algorithm_codesign.rte_reallocation.numeric import strict_guard,configure
        old=mp.mp.dps,mp.iv.dps
        try:
            configure(70)
            # Artificial exact RZ(pi/2), no acquired or saved scientific string.
            self.assertLess(strict_guard('WWWWWWWS',Angle('pi',F(1,2))),F(1,10**60))
            self.assertGreater(strict_guard('S',Angle('pi',F(1,2))),F(1,10))
        finally:mp.mp.dps,mp.iv.dps=old

    def test_strict_guard_restores_context_on_failure(self):
        import mpmath as mp
        old=mp.mp.dps,mp.iv.dps
        with self.assertRaises(ValueError):strict_primitive_guard('BAD',F(1,17),dps=70)
        self.assertEqual((mp.mp.dps,mp.iv.dps),old)

    def test_signed_native_bound_connects_to_parent_ledger(self):
        from trottertracks.algorithm_codesign.parent_allocation import error_accounting
        e=mock_checked();p=native_accounting([('R',0,'2/11',-1)],{'2/11':e})
        budget=error_accounting([{'coefficient':F(3,2)}],coefficient_error_upper=F(1,10**12),
                               angle_error_bounds=[F(0)],native_error_bounds=[p['strict_native_error_upper']])
        self.assertEqual(budget['weighted_native_error_upper'],F(3,2*10**9))
        self.assertFalse(budget['confidence_or_shots_computed'])


class FormalAllocationControls(unittest.TestCase):
    def test_positive_cost_KKT_interior_witness(self):
        # Artificial exact witness: a=7, b=(4,3), sqrt(C)=(1,3/4).
        z=(F(3),F(4));b=(F(4),F(3));c=(F(5),F(5));weights=(F(1),F(3,4));lam=F(3,5)
        self.assertEqual(sum(z),7)
        for zi,bi,ci,w in zip(z,b,c,weights):
            self.assertEqual(ci*ci,zi*zi+bi*bi)
            self.assertEqual(w*zi/ci,lam)
            self.assertGreater(w*bi*bi/ci**3,0)
            self.assertEqual(zi*zi*(w*w-lam*lam),bi*bi*lam*lam)

    def test_vertices_can_miss_interior_fixed_cost_minimum(self):
        optimum_squared=F(35,4)**2
        common_squared=2*F(25,4)**2
        self.assertLess(optimum_squared,common_squared)
        # Vertex proxies: sqrt(65)+9/4 and 4+3sqrt(58)/4.
        self.assertLess(F(35,4),8+F(9,4))
        self.assertLess(F(35,4),4+F(3,4)*F(15,2))
        self.assertLess(8**2,65);self.assertLess(F(15,2)**2,58)

    def test_conditional_IS_is_not_global_optimum_with_fixed_rest(self):
        # Root group mass=1/2, c=(1,1), C=(1,4), q_rest=1/2,c_rest=1,C_rest=100.
        # Conditional optimum p=(2/3,1/3). Differentiate global cost*moment.
        g=F(1,2);p=F(2,3)
        average=F(50)+g*(p+4*(1-p));moment=F(2)+1/(g*p)+1/(g*(1-p))
        daverage=g*(1-4);dmoment=-1/(g*p*p)+1/(g*(1-p)**2)
        self.assertNotEqual(daverage*moment+average*dmoment,0)


if __name__=='__main__':unittest.main(verbosity=2)
