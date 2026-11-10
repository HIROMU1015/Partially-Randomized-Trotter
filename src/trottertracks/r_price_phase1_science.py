"""Lazy, price-only Phase 1 backend. Never instantiated during preparation.

Pinned sampler/native compose/wrapper/compiler and Accounting kernels are reused.
No SCF/DF/state solve, signal/N/G, historical authorization or worker launcher.
"""
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from . import r_price_phase1 as m
from .r_price_phase1_runner import read, write

ACTIVE = None


def functions(root, commit, path, names, scope, strip=()):
    nodes=[n for n in ast.parse(m.blob(root,commit,path)).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names): raise ValueError('PINNED_SCIENCE_FUNCTION_MISSING')
    class Adapter(ast.NodeTransformer):
        def visit_ImportFrom(self,node): return None if node.module in strip else node
    nodes=[Adapter().visit(n) for n in nodes]
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[])),commit+':'+path,'exec'),scope)


def force_pair(requests, pair_index, system, make_event):
    """Legal diagnostic pair; retained rotations, fixed supported products.

    Source requests are independent dedicated diagnostic draws, not normal
    cohorts. Every base order is forced to 0. The variant has exactly one 2.
    No conditional probability/expectation is inferred from these paths.
    """
    weights = [r.step_time for r in requests]
    target = (next(i for i,t in enumerate(weights) if t<0) if system=='H6' and pair_index==1 else 0)
    pos = 1 if system=='H4' and pair_index==1 else 0
    base=[];variant=[]
    for i,req in enumerate(requests):
        components=req.preparation.rte_preparation.symbolic_tail.components
        lookup={c.component_id:j for j,c in enumerate(components)};events=[]
        for e in req.rte_occurrence.events:
            events.append(make_event((lookup[e.rotation_component_id],),components,req.rte_distribution,0))
        change=list(events)
        if i==target:
            if pos>=len(events):raise ValueError('DIAGNOSTIC_EVENT_POSITION')
            rotation=lookup[events[pos].rotation_component_id]
            options=[j for j,c in enumerate(components) if not c.is_identity and
                (pair_index==0 or c.basis_hash!=components[rotation].basis_hash)]
            if not options:raise ValueError('NO_SUPPORTED_DIAGNOSTIC_PRODUCTS')
            products=(options[0],options[-1])
            change[pos]=make_event((rotation,*products),components,req.rte_distribution,1)
        base.append(replace(req,rte_occurrence=replace(req.rte_occurrence,events=tuple(events)),rte_basis_plan=None))
        variant.append(replace(req,rte_occurrence=replace(req.rte_occurrence,events=tuple(change)),rte_basis_plan=None))
    return tuple(base),tuple(variant)


def structural(Accounting, blocks, row, scalar, axis, events):
    """Same original Accounting word; only the declared q/R loop scope changes."""
    a=Accounting();control=8 if row['system']=='H4' else 12;J=len(blocks)
    a.gate('h',(control,),role='wrapper')
    for outer in range(row['q']):
        for j,bs in enumerate(blocks):
            for b in bs:a.block(b,False)
            a.tail(events[J*outer+j])
            for b in reversed(bs):a.block(b,True)
    if scalar*row['T']!=0:a.gate('p',(control,),(-scalar*row['T'],),'scalar_once')
    if axis=='sine':a.gate('sdg',(control,),role='wrapper')
    a.gate('h',(control,),role='wrapper');a.gate('measure',(control,),role='measurement')
    return a.finish()


def _worker(job):return ACTIVE.acquire(*job)


class Backend:
    def __init__(self, root, plan):
        self.root=Path(root);self.plan=deepcopy(plan);self.contexts={}

    def context(self, system):
        if system in self.contexts:return self.contexts[system]
        # Scientific libraries/input arrays are first touched here, after --execute.
        provider=Path(self.plan['runtime']['provider_root']);sys.path.insert(0,str(provider/'src'))
        if any(name=='trotterlib' or name.startswith('trotterlib.') for name in sys.modules):
            module=sys.modules['trotterlib']
            if not Path(module.__file__).is_relative_to(provider):raise ValueError('FOREIGN_PROVIDER_ALREADY_IMPORTED')
        import trottertracks
        path=str(provider/'src/trottertracks')
        if path not in trottertracks.__path__:trottertracks.__path__.insert(0,path)
        import numpy as np
        import qiskit
        if (np.__version__,qiskit.__version__)!=('1.26.4','1.3.0'):raise ValueError('FIXED_SCIENTIFIC_RUNTIME')
        def require(v,reason):
            if not v:raise ValueError(reason)
        tree=ast.parse(m.blob(self.root,m.CARRIER,'src/trotterlib/h46_m5_plan.py'))
        tail=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TAIL' for t in n.targets))
        core=dict(math=math,TAIL=tail)
        functions(self.root,m.CARRIER,'scripts/h46_new2_core.py',['allocation'],core)
        functions(self.root,m.CARRIER,'src/trotterlib/h46_m5_plan.py',['weights','schedule'],core)
        h=dict(np=np,math=math,json=json,hashlib=hashlib,require=require,core=SimpleNamespace(**{k:core[k] for k in ('weights','allocation','schedule')}))
        functions(self.root,m.CARRIER,'scripts/h46_new2_native.py',['gaussian_operations','prepare'],h)
        functions(self.root,m.CARRIER,'scripts/h46_pr_pilot_repair.py',['canonical_tau'],h)
        functions(self.root,m.CARRIER,'scripts/h46_long_time_native.py',['trial_seed','sample_requests','compose'],h,('h46_m5_contract','h46_pr_pilot_repair'))
        cfg=read(self.plan['runtime']['input_config'])['systems'][system]
        binding=self.plan['menus'][system]['binding'];m._binding_matches(cfg['binding'],binding)
        store=m.p.SavedArtifacts(self.root)
        base=f'docs/research/{system.lower()}_df'+('10' if system=='H4' else '19')+'_price_v1_20261110/DEVELOPMENT.json'
        prior=store.read(m.ref(m.H4V1 if system=='H4' else m.H6V1,base))
        for name,expected in prior['input_sha256'].items():
            record=cfg['inputs'][name];data=Path(record['path']).read_bytes()
            require(hashlib.sha256(data).hexdigest()==expected==record['sha256'] and len(data)==record['bytes'],'SAVED_INPUT_BYTES')
        if system=='H4':
            shim=SimpleNamespace(read=read,sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest(),require=require)
            load=dict(Path=Path,c=shim)
            functions(self.root,m.CARRIER,'scripts/h46_m5_tol_saved_input.py',['array_record','load'],load)
            ham,sector,full,state,md=load['load'](dict(mapping=dict(tol_snapshot=cfg['inputs']['H4_tol_snapshot.npz']['path'],tol_receipt=cfg['inputs']['H4_tol_receipt.json']['path'])))
            prov=store.read(m.ref(m.H4,'docs/research/h4_df10_price_v2_independent_q_20261110/completion01/PROVENANCE.json'))
            expected_prep=next(v['preparation_hash'] for v in prov['trial_bindings'] if v['method']=='PR')
            counter_commit=m.BASE;counter_path='src/trottertracks/h4_df10_price_v1.py'
        else:
            from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
            ham,sector,full,state,md,*_=load_h6_snapshot(Path(cfg['inputs']['H6_snapshot.npz']['path']),read(cfg['inputs']['H6_snapshot_receipt.json']['path']),read(cfg['inputs']['H6_df_receipt.json']['path']))
            prior=store.read(m.ref(m.H6,'docs/research/h6_df19_price_v2_20261110/PILOT_INPUTS.json'))
            expected_prep=prior['preparation_hash']['7'];counter_commit=m.C_HELPERS;counter_path='src/trottertracks/h6_df19_price_v1.py'
        prep=h['prepare'](ham,1 if system=='H4' else 7)
        require(prep.preparation_hash==expected_prep and prep.hamiltonian_hash==binding['hamiltonian'],'SAVED_PREPARATION_BINDING')
        require(tuple(prep.deterministic_fragment_indices)==tuple(range(prep.ld)),'SAVED_FRAGMENT_ORDER')
        from trottertracks.resource_applicability import ax2b_native_df_v5 as native
        from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
        from trotterlib.rte import CompilerSettings, _make_event
        from trotterlib.rte_compiled_cost import transpile_and_measure_cost
        import trotterlib.rte_compiled_cost as costs
        from functools import partial
        costs.transpile=partial(qiskit.transpile,num_processes=1)  # parallelize whole paths, not nested compiler pools
        for name in ('trotterlib.rte','trotterlib.df_partial_s2','trotterlib.df_rte_qiskit','trotterlib.rte_compiled_cost','trottertracks.resource_applicability.ax2b_native_df_v5'):
            module=importlib.import_module(name);file=Path(module.__file__);rel=file.relative_to(provider)
            require(file.read_bytes()==m.blob(self.root,m.PROVIDER,str(rel)),'PROVIDER_SOURCE:'+name)
        counter_path_local='src/trottertracks/track_d_native_resources.py'
        require((self.root/counter_path_local).read_bytes()==m.blob(self.root,m.BASE,counter_path_local),'FROZEN_STRUCTURAL_COUNTER')
        counter=m.fixed_module(self.root,counter_commit,counter_path,'trottertracks._phase1_'+system+'_Accounting').Accounting
        from trotterlib.df_trotter.ops import one_body_diagonal_primitives,df_squared_diagonal_primitives
        blocks=[];delta=.1 if system=='H4' else 1.6
        for w in core['weights']('2nd' if system=='H4' else '4th(m5_best)'):
            bs=[]
            for b in prep.deterministic_blocks:
                diag=one_body_diagonal_primitives(np.array(b.diagonal_eigenvalues),delta*w/2) if b.block_kind=='one_body' else df_squared_diagonal_primitives(np.array(b.diagonal_eta),b.lam,delta*w/2)
                basis=[dict(name=o.name,qubits=list(q),parameters=[str(t) for t in o.params]) for o,q in b.runtime_basis_operations]
                bs.append(dict(basis=basis,diagonal=diag))
            blocks.append(bs)
        ctx=SimpleNamespace(prep=prep,h=h,scalar=ham.constant+prep.extracted_identity_coefficient,blocks=blocks,Accounting=counter,
            native=native,fingerprint_context=use_fingerprint_reuse,CompilerSettings=CompilerSettings,compile=transpile_and_measure_cost,make_event=_make_event)
        self.contexts[system]=ctx;return ctx

    def price_requests(self, cell, index, directory, requests, seeds, laws, *, conditional):
        c=m.row_for(self.plan,cell)['candidate'];ctx=self.context(cell['system']);directory.mkdir(parents=True,exist_ok=False)
        row=m.row_for(self.plan,cell);allocation=row['structure']['allocation'];weights=row['structure']['signed_weights']
        if len(requests)!=c['q']*len(allocation) or any(len(r.rte_occurrence.events)!=allocation[i%len(allocation)] or r.step_time!=c['T']*weights[i%len(weights)]/c['q'] for i,r in enumerate(requests)):
            raise ValueError('SIGNED_EVENT_ALLOCATION_OR_TIME')
        events=[[e.to_dict() for e in req.rte_occurrence.events] for req in requests]
        write(directory/'events_PRIVATE.json',dict(events=events,seeds=seeds,laws=laws,conditional=conditional))
        event_digest=m.digest(events)
        S=[structural(ctx.Accounting,ctx.blocks,c,ctx.scalar,a,events)['structural_rz'] for a in m.AXES]
        write(directory/'STRUCTURAL.json',dict(S=S,event_digest=event_digest,scientific_source=self.plan['source']))
        C=[];fingerprints=[];hits=0
        with ctx.fingerprint_context() as fp:
            evolution=ctx.h['compose'](c,ctx.prep,requests,ctx.scalar,fp.fingerprint)
            for axis in m.AXES:
                wrapper=ctx.native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=len(evolution.circuit.data)+4)
                fingerprint=fp.fingerprint(wrapper);old=Path(self.plan['runtime']['old_cache'])/(fingerprint+'.json')
                write(directory/(axis+'.started.json'),dict(fingerprint=fingerprint,compiler=m.p.PROFILE['compiler']))
                hit=old.exists()
                if hit:
                    cache=read(old)
                    if cache['fingerprint']!=fingerprint or cache['compiler']!=m.p.PROFILE['compiler']:raise ValueError('CACHE_BINDING_NO_FALLBACK')
                    metrics=cache['metrics'];hits+=1
                else:
                    settings=ctx.CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0')
                    cost=ctx.compile(wrapper,settings,actual_circuit_fingerprint=fingerprint)
                    if cost.compiler_settings_hash!=m.p.PROFILE['compiler']:raise ValueError('COMPILER_HASH')
                    metrics={k:getattr(cost,k) for k in ('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')}
                    del cost
                value=metrics['rz_count']
                if type(value) is not int or value<0 or fp.fingerprint(wrapper)!=fingerprint:raise ValueError('COMPILED_PRICE_OR_MUTATED_INPUT')
                write(directory/(axis+'.json'),dict(metrics=metrics,fingerprint=fingerprint,compiler=m.p.PROFILE['compiler'],cache_hit=hit))
                C.append(value);fingerprints.append(fingerprint);del wrapper
        result=dict(cell_id=cell['id'],candidate_id=c['id'],index=index,cohort=cell['cohort'],conditional=conditional,C=C,S=S,
            source=dict(execution_source=self.plan['source'],private_result=str(directory/'result.json')),event_digest=event_digest,
            compiler=m.p.PROFILE['compiler'],fingerprints=fingerprints,cache_hits=hits,new_compiles=2-hits,
            full_wrapper_statevector_validation=False,signal_N_G_computed=False)
        write(directory/'result.json',result);return result

    def acquire(self, cell, index, directory):
        ctx=self.context(cell['system']);c=m.row_for(self.plan,cell)['candidate']
        requests,seeds,laws=ctx.h['sample_requests'](c,ctx.prep,cell['cohort'],index)
        if seeds!=cell['seeds'][index]:raise ValueError('FROZEN_SEEDS')
        return self.price_requests(cell,index,directory,requests,seeds,laws,conditional=False)

    def acquire_many(self,cells,directory):
        global ACTIVE
        for system in {c['system'] for c in cells}:self.context(system)
        ACTIVE=self;jobs=[(c,i,Path(directory)/c['id']/str(i)) for c in cells for i in range(c['n'])]
        workers=min(len(jobs),len(os.sched_getaffinity(0)));data={c['id']:[] for c in cells}
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
            pending=[pool.submit(_worker,job) for job in jobs]
            for future in as_completed(pending):
                record=future.result();data[record['cell_id']].append(record)
        ACTIVE=None
        for rows in data.values():rows.sort(key=lambda r:r['index'])
        return data

    def rare_pairs(self,directory):
        diagnostics=[]
        for system in ('H4','H6'):
            ctx=self.context(system);cell=deepcopy(next(c for c in self.plan['cells'] if c['system']==system and not c['saved'] and c['role']=='CALIBRATION'))
            cell['cohort']='R_PRICE_P1_20261111_'+system+'_FORCED_DIAGNOSTIC'
            c=m.row_for(self.plan,cell)['candidate']
            for pair in (0,1):
                requests,seeds,laws=ctx.h['sample_requests'](c,ctx.prep,cell['cohort'],pair)
                expected=next(v for v in self.plan['rare_groups'] if v['system']==system and v['pair']==pair)
                if seeds!=expected['seeds'] or cell['cohort']!=expected['cohort'] or c['id']!=expected['candidate_id']:raise ValueError('FROZEN_RARE_SEEDS')
                base,rare=force_pair(requests,pair,system,ctx.make_event);results=[]
                for version,reqs in [('order0',base),('order2',rare)]:
                    results.append(self.price_requests(cell,pair,Path(directory)/system/str(pair)/version,reqs,seeds,laws,conditional=True))
                diagnostics.append(dict(system=system,pair=pair,unit='FORCED_LEGAL_PAIR_NOT_UNCONDITIONAL_COHORT',
                    paths=results,compiled_difference=[b-a for a,b in zip(results[0]['C'],results[1]['C'])],all_law_weighted_contribution=None))
        return diagnostics
