"""C1 source state and iterator tests on artificial metadata only."""
from copy import deepcopy
import itertools
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from trottertracks import mean_compiled_price as m
from trottertracks import r_price_phase1 as c1
from trottertracks.r_price_phase1_science import structural
from trottertracks.h4_df10_price_v1 import Accounting as H4Accounting


def accounting(system):
    if system=='H4':return H4Accounting
    # Only this pure counter is loaded from fixed Git; no molecular backend.
    return c1.fixed_module(ROOT,c1.C_HELPERS,'src/trottertracks/h6_df19_price_v1.py',
        'trottertracks._stream_artificial_h6_counter').Accounting


def event(angle=.125,*,basis='a',identity=False,phase=0.):
    ops=[dict(name='p',qubits=[0],parameters=['.25' if basis=='a' else '.75'])]
    app=dict(role='rotation',coefficient_sign=1,application_index=0,is_identity=identity,
             basis_id=basis,basis_hash='ARTIFICIAL_'+basis,basis_change_operations=ops,
             diagonal_pauli_support=[] if identity else [0])
    return dict(phase=dict(real=math.cos(phase),imag=math.sin(phase)),taylor_order=0,
                rotation_angle=angle,application_sequence=[app])


def fixture(system='H4',q=2):
    weights=[1.] if system=='H4' else [1.]*5+[-.5]+[1.]*5
    allocation=[3]*len(weights);blocks=[];events=[]
    for j,w in enumerate(weights):
        ops=[dict(name='p',qubits=[0],parameters=['.25' if j%2==0 else '.75'])]
        diagonal=SimpleNamespace(rzz=(),rz=((0,w*.0625),),global_phase=w*.03125)
        blocks.append([dict(basis=ops,diagonal=diagonal)])
    for outer in range(q):
        for j,w in enumerate(weights):
            events.append([event(w*.125,basis='a' if (k+outer+j)%2==0 else 'b',
                                 identity=(k==1),phase=0.) for k in range(allocation[j])])
    row=dict(system=system,q=q,T=q*.5,R_outer=sum(allocation))
    return blocks,row,allocation,events


class NoMaterialize:
    def __init__(self,values):self.values=values
    def __iter__(self):return iter(self.values)
    def __len__(self):raise AssertionError('No event/batch length or materialization')
    def __getitem__(self,index):raise AssertionError('Producer is forward-only')


class Tests(unittest.TestCase):
    def test_generator_matches_all_original_H4_counter_outputs(self):
        blocks,row,allocation,events=fixture()
        for axis in c1.AXES:
            expected=structural(H4Accounting,blocks,row,.3,axis,deepcopy(events))
            lazy=NoMaterialize(NoMaterialize(deepcopy(batch)) for batch in events)
            actual=m.stream_structure(H4Accounting,blocks,row,.3,axis,lazy,allocation=allocation)
            self.assertEqual(actual['accounting'],expected)
            self.assertIsNone(actual['mean_structural_price']);self.assertFalse(actual['qualified_for_candidate'])

    def test_H6_eleven_signed_piece_order_and_outer_continuation(self):
        A=accounting('H6');blocks,row,allocation,events=fixture('H6')
        for axis in c1.AXES:
            expected=structural(A,blocks,row,.3,axis,events)
            actual=m.stream_structure(A,blocks,row,.3,axis,(iter(e) for e in events),allocation=allocation)
            self.assertEqual(actual['accounting'],expected)
            self.assertEqual(actual['stream']['tail_batches'],22)
            self.assertEqual(actual['stream']['events'],66)

    def test_chunk_transport_preserves_tail_phase_but_regrouping_is_not_equivalent(self):
        # Bounded artificial angles expose ordinary binary64 nonassociativity.
        values=[event(1.,identity=True),event(-1.,identity=True),event(2**-54,identity=True)]
        a=H4Accounting();a.tail(values);original=a.finish()
        a=H4Accounting();a.tail(itertools.chain.from_iterable([values[:1],values[1:]]));same=a.finish()
        a=H4Accounting();a.tail(values[:1]);a.tail(values[1:]);regrouped=a.finish()
        self.assertEqual(original,same)
        self.assertEqual(original['structural_rz'],1)
        self.assertEqual(regrouped['structural_rz'],0)

    def test_terminal_flush_is_once_and_not_a_piece_checkpoint(self):
        class Spy(H4Accounting):
            calls=0
            def finish(self):type(self).calls+=1;return super().finish()
        blocks,row,allocation,events=fixture(q=3)
        result=m.stream_structure(Spy,blocks,row,.3,'sine',(iter(e) for e in events),allocation=allocation)
        self.assertEqual(Spy.calls,1)
        self.assertEqual(result['accounting']['native_rz_by_role']['scalar_once'],1)
        a=H4Accounting();a._native('rz',(0,),.25);a._native('rz',(0,),-.25)
        uninterrupted=a.finish()['structural_rz']
        a=H4Accounting();a._native('rz',(0,),.25);a.finish();a._native('rz',(0,),-.25)
        self.assertEqual(uninterrupted,0);self.assertEqual(a.finish()['structural_rz'],2)

    def test_wrapper_and_scalar_once_preserve_original_exact_call_word(self):
        traces=[]
        class Trace(H4Accounting):
            def gate(self,name,wires,params=(),role='other'):
                traces[-1].append((name,tuple(wires),tuple(params),role))
                return super().gate(name,wires,params,role)
        blocks,row,allocation,events=fixture(q=3)
        for scalar in (0.,.3):
            for axis in c1.AXES:
                traces.append([]);structural(Trace,blocks,row,scalar,axis,events)
                traces.append([]);m.stream_structure(Trace,blocks,row,scalar,axis,(iter(e) for e in events),allocation=allocation)
                self.assertEqual(traces[-1],traces[-2])
                self.assertEqual(sum(g[3]=='scalar_once' for g in traces[-1]),int(scalar!=0))
                self.assertEqual(sum(g[0]=='measure' for g in traces[-1]),1)

    def test_lazy_producer_retains_constant_number_of_event_objects(self):
        class Tracked(dict):
            live=0;peak=0
            def __init__(self):
                super().__init__(event());type(self).live+=1;type(self).peak=max(type(self).peak,type(self).live)
            def __del__(self):type(self).live-=1
        for R in (512,2048):
            Tracked.peak=0
            q=4;row=dict(system='H4',q=q,T=.5*q,R_outer=R)
            batches=NoMaterialize(NoMaterialize(Tracked() for _ in range(R)) for _ in range(q))
            result=m.stream_structure(H4Accounting,[[]],row,0.,'cosine',batches,allocation=[R])
            self.assertEqual(result['stream']['events'],q*R)
            self.assertLessEqual(Tracked.peak,3);self.assertEqual(Tracked.live,0)
            self.assertFalse(result['stream']['consumer_materializes_events'])

    def test_missing_extra_or_wrong_allocation_fails_without_a_price(self):
        blocks,row,allocation,events=fixture()
        variants=[events[:-1],events+[events[0]],[events[0][:-1],events[1]],[events[0]+[event()],events[1]]]
        for batches in variants:
            with self.assertRaises(ValueError):m.stream_structure(H4Accounting,blocks,row,.3,'cosine',iter(batches),allocation=allocation)
        with self.assertRaises(ValueError):m.stream_structure(H4Accounting,blocks,row,.3,'cosine',iter(events),allocation=[2])

    def test_same_basis_and_nonzero_flags_are_not_a_closed_state(self):
        prefixes=[]
        for n in (1,2):
            a=H4Accounting()
            for _ in range(n):a._native('rz',(0,),.25)
            prefixes.append((a.pending_basis,a.pending_z[0]!=0,a.pending_z[0]))
            a._native('rz',(0,),-.5)
            prefixes[-1]+= (a.finish()['structural_rz'],)
        self.assertEqual(prefixes[0][:2],prefixes[1][:2])
        self.assertEqual([p[2] for p in prefixes],[.25,.5])
        self.assertEqual([p[3] for p in prefixes],[1,0])

    def test_modulo_angle_equivalence_is_not_C1_literal_cost_equivalence(self):
        values=[]
        for angle in (0.,2*math.pi):
            a=H4Accounting();a._native('rz',(0,),angle);values.append(a.finish()['structural_rz'])
        self.assertEqual(values,[0,1])


if __name__=='__main__':unittest.main(verbosity=2)
