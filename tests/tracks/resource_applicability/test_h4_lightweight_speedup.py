"""Artificial equivalence only. The bounded runner injects actual old Git blobs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from qiskit import QuantumCircuit
from trottertracks.resource_applicability.h4_geometry import circuits, signal, ledger, identity, gates, execution
from trottertracks.resource_applicability.h4_geometry.inputs import reverse_bits

OLD_SIGNAL=OLD_LEDGER=None  # Injected from the fixed old SOURCE, never reimplemented.
ROOT=Path(__file__).absolute().parents[3]
COUNTS={}


def artificial_basis(U):
    # Exercises nontrivial complex basis action without OpenFermion/Gaussian work.
    qc=QuantumCircuit(8);qc.ry(float(U[0,0].real)*.125,0);qc.cx(0,1);qc.p(.0625,1)
    return qc


def arrays_fixture():
    arrays={'one':np.diag([.125,-.0625,-0.,0.,0.,0.,0.,0.]),
            'G':np.asarray([np.diag([.125+i/128,-.0625,0.,0.,0.,0.,0.,0.]) for i in range(12)]),
            'lambdas':np.asarray([(-1.)**i/64 for i in range(12)]),
            'nuclear':np.asarray(-0.),'H':np.zeros((256,256),dtype=complex)}
    base={'method':'B0','L_D':12}
    with patch.object(circuits,'diagonalize_block',side_effect=synthetic_block):
        _prep,det,_tail=OLD_SIGNAL.prepare(arrays,base)
    perm=[reverse_bits(i) for i in range(256)]
    arrays['H']=(sum(det,np.zeros((256,256),dtype=complex)))[np.ix_(perm,perm)]
    arrays['qiskit_state']=np.eye(1,256,dtype=complex)[0]
    return arrays


ORIGINAL_BLOCK=circuits.diagonalize_block
def synthetic_block(matrix,lam=None):
    return ORIGINAL_BLOCK(matrix,lam,basis_builder=artificial_basis)


def encoded(circuit,axis):
    return b''.join(identity.canonical_chunks(circuits.serialize(circuit,axis)))


class PreparationTests(unittest.TestCase):
    def assert_array_bytes(self,a,b):
        self.assertEqual((a.dtype.str,a.shape,a.tobytes()),(b.dtype.str,b.shape,b.tobytes()))

    def test_all_registered_templates_old_bytes_signals_and_wrappers(self):
        templates=json.loads((ROOT/gates.CONTRACT/'zero_compute_plan_v2.json').read_text())['templates']
        arrays=arrays_fixture();reference={}
        with patch.object(circuits,'diagonalize_block',side_effect=synthetic_block) as block:
            for t in templates:
                key=(t['method'],t['L_D'])
                if key not in reference:reference[key]=OLD_SIGNAL.prepare(arrays,t)
            self.assertEqual(block.call_count,13*len(reference))
        with patch.object(circuits,'diagonalize_block',side_effect=synthetic_block) as block:
            common=signal.GeometryPreparation(arrays)
            for t in templates:
                p,det,tail=signal.prepare(arrays,t,common=common)
                old,old_det,old_tail=reference[(t['method'],t['L_D'])]
                self.assertEqual(p['constant'].hex(),old['constant'].hex())
                self.assertEqual(len(det),len(old_det))
                for a,b in zip(det,old_det,strict=True):self.assert_array_bytes(a,b)
                if tail is None:self.assertIsNone(old_tail)
                else:
                    self.assertEqual(tail[0].hex(),old_tail[0].hex());self.assert_array_bytes(tail[1],old_tail[1])
                def coefficients(prep):
                    return [(c['support'],c['abs_coefficient'].hex(),c['sign'],list(c['block']['coefficients'].items()))
                            for c in prep['components']]
                self.assertEqual(coefficients(p),coefficients(old))
            self.assertEqual(block.call_count,13)
            self.assertEqual(len(common._prepared),len(reference))
            COUNTS.update(templates_checked=len(templates),distinct_preparations=len(reference),
                          old_block_calls_static=13*len(templates),new_block_calls_observed=block.call_count)
            wrapper_pairs=0
            for method in ('B0','B1','B2','B3'):
                t={**next(t for t in templates if t['method']==method),'q':1,'T':.125,'delta':.125}
                if method in ('B2','B3'):t.update(r=2,K=2)
                p,det,tail=signal.prepare(arrays,t,common=common)
                old,old_det,old_tail=reference[(method,t['L_D'])]
                before=[encoded(b['basis'],'cosine') for b in [common.one,*common.df]]
                events=signal.sample_events(p['components'],t,123)[0] if tail is not None else [[]]
                old_events=OLD_SIGNAL.sample_events(old['components'],t,123)[0] if old_tail is not None else [[]]
                for axis in ('cosine','sine'):
                    a=circuits.wrapper(circuits.build_evolution(p,t,events),axis)
                    b=circuits.wrapper(circuits.build_evolution(old,t,old_events),axis)
                    self.assertEqual(a.num_qubits,9)
                    self.assertEqual(encoded(a,axis),encoded(b,axis))
                    self.assertEqual(circuits.numerical_fingerprint(a,axis),circuits.numerical_fingerprint(b,axis))
                    wrapper_pairs+=1
                self.assertEqual(before,[encoded(b['basis'],'cosine') for b in [common.one,*common.df]])
                self.assertEqual(signal.corrected_signal(det,tail,p['constant'],arrays['qiskit_state'],t),
                                 OLD_SIGNAL.corrected_signal(old_det,old_tail,old['constant'],arrays['qiskit_state'],t))
            COUNTS['wrapper_byte_digest_pairs']=wrapper_pairs
            COUNTS['common_cached_array_bytes']=sum(a.nbytes for a in common._operators.values())+sum(a.nbytes for a in common._dense.values())+sum(
                value[2][1].nbytes for value in common._prepared.values() if value[2] is not None)

    def test_geometry_scope_readonly_and_fresh_small_containers(self):
        arrays=arrays_fixture();t={'method':'B2','L_D':3}
        with patch.object(circuits,'diagonalize_block',side_effect=synthetic_block):
            common=signal.GeometryPreparation(arrays)
            result=signal.prepare(arrays,t,common=common)
            result[0]['deterministic'].clear();result[0]['components'][0]['sign']=999;result[1].clear()
            fresh=signal.prepare(arrays,t,common=common)
            self.assertEqual(len(fresh[1]),4);self.assertIn(fresh[0]['components'][0]['sign'],(-1,1))
            with self.assertRaises(ValueError):fresh[1][0][0,0]=0
            with self.assertRaises(TypeError):fresh[0]['deterministic'][0]['coefficients'][()]=3
            with self.assertRaises(identity.Stop):signal.prepare(dict(arrays),t,common=common)
            old=arrays['one'];arrays['one']=old.copy()
            with self.assertRaises(identity.Stop):signal.prepare(arrays,t,common=common)
            arrays['one']=old;old.setflags(write=True)
            with self.assertRaises(identity.Stop):signal.prepare(arrays,t,common=common)
            old.setflags(write=False)
            other=arrays_fixture();other_common=signal.GeometryPreparation(other)
            self.assertIsNot(common.one,other_common.one)
            with self.assertRaises(identity.Stop):signal.prepare(other,t,common=common)

    def test_dense_complex_unitary_parameter_stays_unchanged(self):
        from itertools import zip_longest
        from qiskit.circuit.library import UnitaryGate
        matrix=np.diag(np.exp(1j*np.arange(256)/1024)).astype(np.complex128)
        matrix.real[2,3]=-0.
        arrays={'one':np.diag([.125,-.0625,0.,0.,0.,0.,0.,0.]),
                'G':np.zeros((12,8,8)),'lambdas':np.ones(12),'nuclear':np.asarray(-0.),
                'H':np.zeros((256,256),dtype=complex)}
        def block(a,lam=None):
            result=ORIGINAL_BLOCK(a,lam,basis_builder=artificial_basis)
            basis=QuantumCircuit(8);basis.append(UnitaryGate(matrix.copy()),range(8))
            return {**result,'basis':basis}
        t={'method':'B0','L_D':0,'q':1,'T':.125,'delta':.125,'r':0,'K':0}
        with patch.object(circuits,'diagonalize_block',side_effect=block):
            old,old_det,_=OLD_SIGNAL.prepare(arrays,t)
            common=signal.GeometryPreparation(arrays)
            parameter=common.one['basis'].data[0].operation.params[0]
            before=(parameter.dtype.str,parameter.shape,parameter.tobytes(),parameter.flags.writeable)
            new,new_det,_=signal.prepare(arrays,t,common=common)
            self.assert_array_bytes(old_det[0],new_det[0])
            a=circuits.wrapper(circuits.build_evolution(old,t,[[]]),'cosine')
            b=circuits.wrapper(circuits.build_evolution(new,t,[[]]),'cosine')
            # Compare bounded chunks rather than materializing either exact tree.
            for x,y in zip_longest(identity.canonical_chunks(circuits.serialize(a,'cosine')),
                                  identity.canonical_chunks(circuits.serialize(b,'cosine'))):self.assertEqual(x,y)
            self.assertEqual(circuits.numerical_fingerprint(a,'cosine'),circuits.numerical_fingerprint(b,'cosine'))
            self.assertEqual(before,(parameter.dtype.str,parameter.shape,parameter.tobytes(),parameter.flags.writeable))
            COUNTS['dense_256_complex_parameter_wrapper_pairs']=1

    def test_failed_validation_is_not_cached_and_nonfinite_serialization_stops(self):
        arrays=arrays_fixture();arrays['H'][0,0]+=1
        with patch.object(circuits,'diagonalize_block',side_effect=synthetic_block):
            common=signal.GeometryPreparation(arrays)
            for _ in range(2):
                with self.assertRaises(identity.Stop):signal.prepare(arrays,{'method':'B2','L_D':3},common=common)
                self.assertEqual(common._prepared,{})
            with self.assertRaises(identity.Stop):signal.prepare(arrays,{'method':'B2','L_D':12},common=common)
        for value in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(identity.Stop):b''.join(identity.canonical_chunks(circuits.number(np.asarray([value]))))
        self.assertNotEqual(b''.join(identity.canonical_chunks(circuits.number(np.asarray([-0.])))),
                            b''.join(identity.canonical_chunks(circuits.number(np.asarray([0.])))))

    def test_driver_has_one_common_preparation_per_geometry(self):
        from types import SimpleNamespace
        from contextlib import ExitStack
        rows=[{'energy':np.asarray(0.),'qiskit_state':np.asarray([1.])} for _ in range(2)]
        templates=[{'method':'B0','L_D':0,'T':1.},{'method':'B1','L_D':12,'T':1.}]
        permit=SimpleNamespace(stage='signal_compile',plan={'schema_version':'h4-newhost-plan-v2',
            'inputs':{d:dict.fromkeys(('input','H','DF','state'),'a'*64) for d in gates.DISTANCES[:2]},
            'templates':templates,'source_commit':'b'*40,'compiler_fingerprint':'c'*64,
            'environment_fingerprint':'d'*64,'caps':{'actual_invocations':74804}})
        run=MagicMock();common_objects=[object(),object()];seen=[]
        def prepare(arrays,t,*,common):
            seen.append((arrays,common));return {'constant':0.},[],None
        def compile_candidates(_run,_ledger,candidates,*a,**kw):
            self.assertEqual(len(list(candidates)),4)
            raise identity.Stop('artificial iterator consumed; no compile')
        with ExitStack() as stack:
            for owner,name,value in [(execution,'DISTANCES',gates.DISTANCES[:2]),
                (execution,'input_boundary',MagicMock(return_value={'consumed_seconds':0.})),(execution,'OwnedRun',MagicMock(return_value=run)),
                (execution,'Ledger',MagicMock()),(execution,'load_new_input',MagicMock(side_effect=rows)),
                (execution,'candidate_wrapper_jobs',MagicMock(return_value=iter(()))),
                (signal,'GeometryPreparation',MagicMock(side_effect=common_objects)),(signal,'prepare',prepare),
                (signal,'corrected_signal',MagicMock(return_value={})),
                (__import__('trottertracks.resource_applicability.h4_geometry.inputs',fromlist=['inputs']),
                 'validate_frozen_state',MagicMock()),
                (__import__('trottertracks.resource_applicability.h4_geometry.parallel',fromlist=['parallel']),
                 'compile_candidates',compile_candidates)]:stack.enter_context(patch.object(owner,name,value))
            with self.assertRaises(identity.Stop):execution.signal_stage(permit,{}, {})
        self.assertEqual([id(common) for arrays,common in seen],[id(common_objects[0])]*2+[id(common_objects[1])]*2)
        run.close.assert_called_once()


class FileBudget:
    def __init__(self,root):self.root=root;root.mkdir()
    def write(self,name,data):
        with (self.root/name).open('xb') as f:f.write(data)


def record(index,axis='cosine'):
    ident={'geometry':'0.70',**dict.fromkeys(('H','DF','state','input','template','compiler','environment'),'a'*64),
           'source':'b'*40,'wrapper_semantics':'h4-full-gaussian-paired-wrapper-v1'}
    return ledger.wire(ident,axis,100+index,index,'d'*64)


class NoScan(dict):
    def items(self):raise AssertionError('whole-history scan')
    def __iter__(self):raise AssertionError('whole-history iteration')


class DeltaLedgerTests(unittest.TestCase):
    def test_old_new_file_bytes_chain_reuse_carry_out_of_order(self):
        with tempfile.TemporaryDirectory(prefix='delta-equivalence-') as t:
            roots=[Path(t)/name for name in ('old','new')]
            heads=[]
            for module,root in zip((OLD_LEDGER,ledger),roots,strict=True):
                journal=module.Ledger(FileBudget(root),cap=23,prior_invocations=20)
                try:
                    rows=[record(i) for i in range(4)];keys=[r['wrapper_key'] for r in rows]
                    for row in rows:journal.register(row)
                    for key in keys[:3]:journal.reserve(key)
                    metrics=dict.fromkeys(ledger.METRICS,7)
                    for key in reversed(keys[:3]):journal.complete(key,metrics)
                    journal.complete(keys[3],{},owner_key=keys[0])
                    self.assertEqual(journal.audit(),{'logical_wrappers':4,'actual_invocations':23})
                    heads.append(journal.chain)
                    with self.assertRaises(identity.Stop):journal.complete(keys[3],metrics)
                finally:journal.close()
            files=[{p.name:p.read_bytes() for p in root.iterdir()} for root in roots]
            self.assertEqual(files[0],files[1]);self.assertEqual(heads[0],heads[1])
            COUNTS['ledger_identical_files']=len(files[0])

    def test_changed_keys_only_even_with_large_mock_history(self):
        with tempfile.TemporaryDirectory(prefix='delta-no-scan-') as t:
            journal=ledger.Ledger(FileBudget(Path(t)/'new'))
            try:
                # The history is in-memory fixture metadata, never real workers.
                journal.entries=NoScan({str(i):{'status':'COMPLETE'} for i in range(10000)})
                journal.reservations=NoScan({str(i):{'status':'COMPLETE'} for i in range(10000)})
                row=record(0);key=row['wrapper_key'];journal.register(row);journal.reserve(key)
                reserve=json.loads((journal.budget.root/'ledger-000001.json').read_text())
                self.assertEqual(reserve['entries'],{});self.assertEqual(list(reserve['reservations']),[key])
                journal.complete(key,dict.fromkeys(ledger.METRICS,1))
                complete=json.loads((journal.budget.root/'ledger-000002.json').read_text())
                self.assertEqual(list(complete['entries']),[key]);self.assertEqual(list(complete['reservations']),[key])
                COUNTS['ledger_history_fixture_rows']=10000
            finally:journal.close()

    def test_old_new_crash_window_bytes_and_reserved_charge(self):
        with tempfile.TemporaryDirectory(prefix='delta-crash-') as t:
            roots=[Path(t)/name for name in ('old','new')]
            for module,root in zip((OLD_LEDGER,ledger),roots,strict=True):
                journal=module.Ledger(FileBudget(root),prior_invocations=20)
                try:
                    row=record(0);key=row['wrapper_key'];journal.register(row);journal.reserve(key)
                    with self.assertRaises(identity.Stop):journal.complete(key,dict.fromkeys(ledger.METRICS,1),interrupt_after_record=True)
                    self.assertEqual(journal.reservations[key]['status'],'RESERVED')
                    with self.assertRaises(identity.Stop):journal.audit()
                    with self.assertRaises(identity.Stop):journal.reserve(key)
                finally:journal.close()
            self.assertEqual({p.name:p.read_bytes() for p in roots[0].iterdir()},
                             {p.name:p.read_bytes() for p in roots[1].iterdir()})
