"""Lazy bridge to pinned original H8 APIs. Never imports science in C0.5 tests."""
import importlib
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from .accounting import check_permit,sha
from .sampling import PCG64Bits,require,digest


class OriginalProvider:
    mock=False
    def __init__(self,context,permit):
        check_permit(permit,False);self.context=context;self.permit=permit
    @classmethod
    def load_parent(cls,permit,adapter_config,cells,output,modules,byte_rows,budget):
        check_permit(permit,False)
        require(digest(byte_rows)==permit.manifest_digest,'PROVIDER_MANIFEST_CHANGED')
        require(digest(dict(adapter_config=adapter_config,cells=cells,modules=modules))==permit.bridge_digest,'PROVIDER_CONFIG_CHANGED')
        for row in byte_rows:require(sha(row['path'])==row['sha256'],'PRELOAD_BYTES_CHANGED')
        for path in reversed(modules['import_paths']):sys.path.insert(0,path)
        native=importlib.import_module('pr_cost_native')
        expected=modules['files']['pr_cost_native']
        require(sha(native.__file__)==expected,'SHADOWED_NATIVE_MODULE')
        budget.take('saved_preparation_decode',1,'saved_preparation_decode')
        budget.take('prefix_reconstruction',len({c['prefix'] for c in cells}),'prefix_reconstruction')
        context=native.load_context('H8',adapter_config,cells,output)
        # Validate all actually imported critical modules, not only the loader name.
        for name,wanted in modules['files'].items():
            module=importlib.import_module(name);require(sha(module.__file__)==wanted,'IMPORTED_PROVIDER_BYTES')
        result=cls(context,permit);result.loaded_byte_rows=byte_rows
        return result
    def check(self):check_permit(self.permit,False)
    def api(self):
        self.check()
        from trotterlib.rte import _make_event,_validate_components
        from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
        from trotterlib.df_partial_s2 import DFPartialS2StepRequest
        import numpy as np
        return SimpleNamespace(make_event=_make_event,validate_components=_validate_components,
            sequence=DFRTEEventSequenceCircuitRequest,step=DFPartialS2StepRequest,
            rng=lambda seed:PCG64Bits(np.random.PCG64(seed)))
    def law(self,prep,cell,seed):
        self.check()
        from trotterlib.rte import make_rte_config
        from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
        tau=prep.exact_rte_lambda_r*(.8/cell['q'])/cell['r']
        return make_rte_config(prep.rte_preparation.symbolic_tail,evolution_time=.8/cell['q'],rte_steps=cell['r'],
            finite_taylor_order=cell['K'],truncation_tolerance=_explicit_cutoff_tolerance(tau,cell['K']),seed=seed)
    def evolve(self,requests):
        self.check();return self.context.native.partial_native_from_step_requests(requests,control_policy='symmetric_directional',max_instructions=1000000)
    def wrapper(self,evolution,axis):
        self.check();return self.context.m.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=1000000)
    def fingerprint_session(self):
        # Same bounded canonical byte-reuse utility used by original P, no cost
        # cache and no extra circuit/compile. State is local to each axis child.
        self.check()
        from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
        return use_fingerprint_reuse()
    def fingerprint(self,wrapper):self.check();return self.context.m.canonical_qiskit_circuit_fingerprint(wrapper)
    def compile(self,wrapper,fingerprint):
        self.check();value=self.context.m.transpile_and_measure_cost(wrapper,self.context.compiler,actual_circuit_fingerprint=fingerprint)
        require(value.compiler_settings_hash==self.context.binding['compiler'] and value.actual_circuit_fingerprint==fingerprint,'COMPILER_FINGERPRINT_BINDING')
        out={k:getattr(value,v) for k,v in dict(RZ='rz_count',CX='cx_count',depth='total_depth',RZ_depth='rz_depth',CX_depth='cx_depth',size='circuit_size').items()}
        require(all(type(v) is int and v>=0 for v in out.values()) and out['RZ']>0,'INTEGER_COMPILED_METRICS')
        return out
    def validate(self,evolution,requests,directory):
        self.check();native=importlib.import_module('pr_cost_native')
        return native.control(evolution,requests,self.context,native.Writer(directory),1e-8)
    def event_values(self,requests):return [[e.to_dict() for e in r.rte_occurrence.events] for r in requests]
    def final_gate(self):
        for row in self.loaded_byte_rows:require(sha(row['path'])==row['sha256'],'FINAL_SOURCE_INPUT_RUNTIME_CHANGED')
