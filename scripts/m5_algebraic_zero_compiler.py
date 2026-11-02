"""H6 zero-rounding algebraic policy; native adapter performs authorization.

The pure planner and exact ledger are reused from the artificial SOURCE.
The run body is preserved except scope/measurement acceptance and zero guard.
No Euler synthesis, angle threshold, angle wrapping or scalar update is used.
"""
import json
import math
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.circuit import Barrier,Measure
from qiskit.circuit.library import CXGate
from qiskit.transpiler import PassManager,TransformationPass
from qiskit.passmanager import ConditionalController,DoWhileController,FlowControllerLinear
from m5_algebraic_oneq_prototype import AngleLedger,AlgebraicPlanner,STANDARD,rational_payload,require
from m5_preserve_oneq_compiler import PreserveOneQubitRuns,patched_pass_manager

POLICY='M5_OPT1_ALGEBRAIC_ONEQ_ZERO_ROUNDING_V1'


class AlgebraicZeroRounding(TransformationPass):
    def __init__(self,recorder):
        super().__init__()
        self.ledger=AngleLedger(0.0)
        self.planner=AlgebraicPlanner()
        self.recorder=recorder

    def run(self, dag):
        require(1<=len(dag.qubits)<=13 and len(dag.clbits)<=1 and
                set(dag.count_ops()) <= {'rz','sx','x','id','cx','barrier','measure'}, 'REGISTERED_ALGEBRAIC_BASIS_DAG')
        # A custom lookalike can be omitted by collect_1q_runs; validate every
        # operation before collection, including gates that will be preserved.
        classes=dict(STANDARD,cx=CXGate,barrier=Barrier,measure=Measure)
        require(all(node.op.base_class is classes[node.op.name] for node in dag.op_nodes()),
                'STANDARD_GATE_IDENTITY_NO_LOOKALIKE')
        initial_phase = float(dag.global_phase).hex()
        before = self.ledger.spent
        changed = total = removed = removed_rz = blocked = overflow = 0
        for run in dag.collect_1q_runs():
            gates=[]
            for node in run:
                op=node.op
                require(op.name in STANDARD and op.base_class is STANDARD[op.name], 'STANDARD_GATE_IDENTITY_NO_LOOKALIKE')
                require(getattr(op,'condition',None) is None, 'NO_CLASSICAL_GATE_CONDITION')
                require(all(type(x) is float and math.isfinite(x) for x in op.params), 'ONLY_BINARY64_GATE_ANGLES')
                gates.append((op.name,tuple(op.params)))
            proposal=self.planner.plan(gates,self.ledger.remaining)
            total += 1
            blocked += proposal['budget_blocked_groups']
            overflow += proposal['overflow_preserved_groups']
            if not proposal['changed']:
                continue
            if proposal['output']:
                q=QuantumCircuit(1)
                for name,params in proposal['output']:
                    getattr(q,name)(*params,0)
                replacement=circuit_to_dag(q)
                require(float(replacement.global_phase)==0, 'NO_GENERATED_SCALAR_PHASE')
                dag.substitute_node_with_dag(run[0],replacement)
                for node in run[1:]:dag.remove_op_node(node)
            else:
                for node in run:dag.remove_op_node(node)
            self.ledger.commit(proposal['charge'])
            changed += 1
            removed += proposal['input_gates']-proposal['output_gates']
            removed_rz += proposal['input_RZ']-proposal['output_RZ']
        require(float(dag.global_phase).hex()==initial_phase, 'GLOBAL_PHASE_BITS_UNCHANGED')
        require(self.ledger.spent==0, 'NO_ADDITIONAL_RZ_ANGLE_ROUNDING')
        self.recorder(dict(policy=POLICY, scope='ALGEBRAIC_ZERO_ROUNDING', runs=total,changed_runs=changed,
            removed_gates=removed,removed_RZ=removed_rz,budget_blocked_groups=blocked,
            overflow_preserved_groups=overflow,budget=rational_payload(self.ledger.budget),
            spent_before=rational_payload(before),spent_after=rational_payload(self.ledger.spent),
            charged_this_pass=rational_payload(self.ledger.spent-before),
            global_phase_binary64_hex=initial_phase,global_phase_bits_unchanged=True,
            Euler_resynthesis=False,angle_wrap=False,small_angle_threshold=False,
            postcompile_correction=False,cache_hits=self.planner.hits,cache_misses=self.planner.misses))
        return dag


def manager(recorder):
    pm=patched_pass_manager(recorder=lambda row:None,basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17)
    sites=[]
    def replace(task):
        if type(task) is PreserveOneQubitRuns:
            sites.append(task)
            return AlgebraicZeroRounding(recorder)
        if type(task) is DoWhileController:return DoWhileController([replace(t) for t in task.tasks],do_while=task.do_while)
        if type(task) is ConditionalController:return ConditionalController([replace(t) for t in task.tasks],condition=task.condition)
        if type(task) is FlowControllerLinear:return FlowControllerLinear([replace(t) for t in task.tasks])
        require(not hasattr(task,'tasks'),'UNRECOGNIZED_FLOW_CONTROLLER')
        return task
    pm.optimization=PassManager([replace(pm.optimization.to_flow_controller())])
    require(len(sites)==1,'EXACT_ONE_ALGEBRAIC_PASS_SITE')
    return pm


def compile_with_algebraic_zero(circuit,*,phase_journal,callback=None,basis_gates,optimization_level,seed_transpiler,num_processes):
    require(basis_gates==['rz','sx','x','cx'] and optimization_level==1 and seed_transpiler==17 and num_processes==1,'REGISTERED_ALGEBRAIC_ZERO_PROFILE')
    require(circuit.num_qubits==13 and circuit.num_clbits==1 and not circuit.parameters and
        set(circuit.count_ops())<={'rz','sx','x','id','cx','barrier','measure'},'REGISTERED_BOUND_COSINE_INPUT')
    ordinal=0
    with open(phase_journal,'x') as stream:
        def record(row):
            nonlocal ordinal
            stream.write(json.dumps(dict(row,pass_ordinal=ordinal),sort_keys=True)+'\n');stream.flush();ordinal+=1
        compiled=manager(record).run(circuit,callback=callback,num_processes=1)
    require(ordinal>0 and not compiled.parameters,'ALGEBRAIC_ZERO_PASS_EXECUTED')
    return compiled
