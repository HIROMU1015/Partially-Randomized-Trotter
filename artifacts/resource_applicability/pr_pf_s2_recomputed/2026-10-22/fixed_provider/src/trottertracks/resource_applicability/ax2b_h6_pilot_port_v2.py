"""H6 v2 setup only: explicit coverage binding and pre-gate evidence.

All signal, oracle, primitive, sampling and cost paths inherit frozen v1.
No real molecular work is permitted without the v2 runner's new pinned grant.
"""
import os
import numpy as np
from trotterlib.df_hamiltonian import df_linear_operator
from trotterlib.pr2_matched_accuracy_m1_execution import _prepare, _prepare_discard
from .ax2b_h6_pilot_port_v1 import H6PilotPort as FrozenH6PilotPort, reference_pair
from .ax2b_molecular_ports_v3 import actual_bounds, check_bounds, occupation_column, complex_record
from .ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
from .ax2b_h4_science_v5 import primitive_sector_certificate, checked_basis_bridge
from .ax2b_h6_controller import bounded_sector_matrix, finite_scale_guard
from .ax2b_h6_pilot_contract_v2 import verify_parent, safe_path, read_json, resources
from .ax2b_h6_pilot_coverage_v2 import assert_actual_coverage


class H6PilotPort(FrozenH6PilotPort):
    def setup(self):
        self.phase('input_reference');self.progress.update(point='verify_saved_snapshot')
        identity=verify_parent(self.root)
        if identity!=self.manifest['input_identity']:raise ValueError('PILOT_PARENT_CHANGED')
        self.ham,self.sector,full,state,metadata,_=load_h6_snapshot(
            safe_path(self.root,identity['snapshot_path']),read_json(self.root/identity['snapshot_receipt_path']),
            read_json(self.root/identity['df_receipt_path']))
        certificate=primitive_sector_certificate(self.ham,self.sector)
        self.qindices,self.qstate=checked_basis_bridge(self.ham,self.sector,full,state,1e-12)
        # Preserve exactly the designated saved amplitudes; no second normalization.
        self.state=state.copy();self.saved_state=state.copy()
        self.progress.update(point='prepare_native_representation')
        self.preps={c['id']:(_prepare_discard if c['method']=='B0' else _prepare)(self.ham,c['prefix']) for c in self.cells}
        bounds=actual_bounds(self.cells,self.preps,T=self.plan['T'])
        records=[]
        for c in self.cells:
            p=self.preps[c['id']]
            records.append({'cell_id':c['id'],'hamiltonian_hash':p.hamiltonian_hash,'preparation_hash':p.preparation_hash,
                'partition_hash':p.partition_hash,'deterministic_fragment_indices':p.deterministic_fragment_indices,
                'randomized_block_indices':p.randomized_block_indices,'exact_rte_lambda_r':p.exact_rte_lambda_r,
                'ranking_proxy_lambda_r':p.ranking_proxy_lambda_r,'extracted_identity':p.extracted_identity_coefficient,
                'constant':p.constant_coefficient,'coefficient_atol':p.coefficient_atol,
                'basis_hashes':[b.basis_hash for b in p.deterministic_blocks]})
        self.writer.write('actual_prepared_representation.json',{'preparations':records,'bounds':bounds,
                'created_before_reference_probes_sampling_or_full_wrapper_build':True,
                'basis_conversion_circuits_may_already_exist':True,
                'gates_pending':True,'compiled_cost_claim':False})
        # Preserve bounds/representation and exact differences before any rejection.
        assert_actual_coverage(self.plan['coverage'],bounds,self.writer)
        check_bounds(bounds,self.caps)
        for c in self.cells:
            p=self.preps[c['id']]
            if p.deterministic_fragment_indices!=tuple(range(c['prefix'])) or p.coefficient_atol!=0. or p.threshold_dropped_component_count!=0:
                raise ValueError('PREPARATION_ORDER_OR_CUTOFF')
        operator,_=df_linear_operator(self.ham,self.sector,**self.plan['matrix_free'])
        self.progress.update(point='bounded_sector_reference_columns')
        matrix,count=bounded_sector_matrix(lambda v:operator@v,self.sector.dimension,self.calls,
                         per_action_cap=self.caps['reference_matvec_per_action'])
        maximum=0.
        for i,index in enumerate(self.sector.basis_indices):
            maximum=max(maximum,float(np.linalg.norm(matrix[:,i]-occupation_column(self.ham,self.sector.basis_indices,int(index)))))
        finite_scale_guard(log_B=0.,intermediate_norm=float(np.linalg.norm(matrix)),absolute_discrepancy=maximum)
        self.reference,pair=reference_pair(matrix,self.state,self.plan['T']);del matrix,operator
        self.writer.write('input_reference.json',{'metadata':metadata,'sector_certificate':certificate,
            'independent_occupation_all_columns_error':maximum,'reference_matvecs':count,'reference':complex_record(self.reference),
            'expm_eigh_agreement':pair,'saved_state_norm_before':float(np.linalg.norm(state)),
            'state_renormalized':False,'reference_kind':'binary64 sector expm/eigh and independent occupation columns',
            'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED','ground_state_certified':False})
        import numba
        self.writer.write('parallel_resource_receipt.json',{'actual_affinity':sorted(os.sched_getaffinity(0)),
            'numba_actual_threads':numba.get_num_threads(),'assigned_resources':self.manifest['assigned_resources'],
            'matrix_free':self.plan['matrix_free'],'BLAS_threads':1,'gpu':False})
        if resources(sorted(os.sched_getaffinity(0)))!=self.manifest['assigned_resources'] or numba.get_num_threads()!=4:
            raise ValueError('PILOT_PARALLEL_RESOURCE')
