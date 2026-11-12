"""S-only C1 context. Imported only inside the explicitly executed S0a leaf.

Reuse the pinned context's load/preparation and block code. Do not instantiate
its Backend, import its compiler plumbing or call sample_requests/compose.
"""
import ast
from contextlib import contextmanager
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import math
import hashlib
import json
import sys
import types

from . import r_price_phase1 as m
from .r_price_phase1_runner import read
from .r_price_phase1_science import functions
from .s0a_saved_structure import C1_SOURCE


def context_body(root):
    """Two fixed, inspected source slices; keep numerical expression order."""
    tree=ast.parse(m.blob(root,C1_SOURCE,'src/trottertracks/r_price_phase1_science.py'))
    backend=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Backend')
    method=next(n for n in backend.body if isinstance(n,ast.FunctionDef) and n.name=='context')
    body=method.body
    compiler_start=next(i for i,n in enumerate(body) if isinstance(n,ast.ImportFrom)
        and n.module=='trottertracks.resource_applicability')
    counter_start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='counter_path_local' for t in n.targets))
    ctx_start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='ctx' for t in n.targets))
    if not compiler_start<counter_start<ctx_start:raise ValueError('PINNED_CONTEXT_LAYOUT')
    # Original prefix loads/checks saved inputs and constructs preparation; tail
    # selects the original Accounting and constructs the signed head metadata.
    method.body=body[:compiler_start]+body[counter_start:ctx_start]+ast.parse(
        'return SimpleNamespace(prep=prep, scalar=ham.constant+prep.extracted_identity_coefficient, blocks=blocks, Accounting=counter)').body
    return method


def load_context(root,plan,system):
    # The historical package __init__ eagerly imports unrelated examples which
    # can invoke Qiskit. This new S-only process uses explicit submodules only.
    provider=Path(plan['runtime']['provider_root'])
    if 'trotterlib' not in sys.modules:
        package=types.ModuleType('trotterlib');package.__package__='trotterlib'
        package.__file__=str(provider/'src/trotterlib/__init__.py')
        package.__path__=[str(provider/'src/trotterlib')]
        sys.modules['trotterlib']=package
    method=context_body(root)
    scope=dict(m=m,Path=Path,SimpleNamespace=SimpleNamespace,read=read,functions=functions,math=math,hashlib=hashlib,json=json,sys=sys)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),
        C1_SOURCE+':context_S_only','exec'),scope)
    return scope['context'](SimpleNamespace(root=Path(root),plan=plan,contexts={}),system)


@contextmanager
def no_circuits():
    """Fail on accidental QuantumCircuit creation/transpile in this S-only leaf."""
    import qiskit
    with patch.object(qiskit.QuantumCircuit,'__init__',side_effect=RuntimeError('S0A_CIRCUIT_BUILD_FORBIDDEN')),patch.object(qiskit,'transpile',side_effect=RuntimeError('S0A_TRANSPILE_FORBIDDEN')):
        yield


class Context:
    def __init__(self,root,plan,system):
        # All preparation work is deferred to scientific execution.
        with no_circuits():self.ctx=load_context(root,plan,system)
        self.Accounting=self.ctx.Accounting;self.blocks=self.ctx.blocks;self.scalar=self.ctx.scalar
        import trotterlib.rte as rte
        from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
        self.rte=rte;self.tolerance=_explicit_cutoff_tolerance
        for name in ('src/trotterlib/rte.py','src/trotterlib/df_partial_s2.py'):
            module=rte if name.endswith('/rte.py') else __import__('trotterlib.df_partial_s2',fromlist=['_'])
            if Path(module.__file__).read_bytes()!=m.blob(root,m.PROVIDER,name):raise ValueError('PINNED_LAW_SOURCE')

    def event_validator(self,cell):
        c=cell['request']['identity']['candidate'];s=cell['request']['identity']['structure']
        prep=self.ctx.prep;tail=prep.rte_preparation.symbolic_tail;components=tail.components
        lookup={v.component_id:i for i,v in enumerate(components)}
        distributions=[];laws=[]
        for j in range(c['q']*len(s['allocation'])):
            piece=j%len(s['allocation']);steps=s['allocation'][piece]
            time=c['T']*s['signed_weights'][piece]/c['q']
            tau=self.ctx.prep.rte_preparation.symbolic_tail.lambda_r*(float(time)/steps)
            config,distribution=self.rte.make_rte_config(tail,evolution_time=time,rte_steps=steps,
                truncation_tolerance=self.tolerance(tau,c['K']),finite_taylor_order=c['K'],seed=cell['seeds'][0][j])
            if (config.tail_id!=tail.tail_id or config.tail_hash!=tail.tail_hash
                or config.dimensionless_step_time!=tau or config.rte_steps!=steps
                or prep.num_system_qubits!=(8 if c['system']=='H4' else 12)):raise ValueError('SIGNED_TAIL_CONFIG')
            distributions.append(distribution)
            laws.append(dict(steps=steps,orders=list(distribution.orders),probabilities=list(distribution.order_probabilities)))
        def validate(j,event):
            distribution=distributions[j]
            # Reconstruct one legal metadata record from its selected components,
            # never sample. Exact canonical comparison also checks basis operations,
            # signs, relative phase, normalization, probabilities and rotation angle.
            try:
                order_index=distribution.orders.index(event['taylor_order'])
                ids=[event['rotation_component_id'],*event['product_component_ids']]
                expected=self.rte._make_event([lookup[i] for i in ids],components,distribution,order_index).to_dict()
            except (KeyError,ValueError,TypeError) as exc:raise ValueError('INVALID_SAVED_EVENT_LAW') from exc
            if m.digest(event)!=m.digest(expected):raise ValueError('EVENT_BASIS_PHASE_ANGLE_OR_LAW')
        return laws,validate
