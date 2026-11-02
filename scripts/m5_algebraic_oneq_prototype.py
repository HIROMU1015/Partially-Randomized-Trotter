"""Synthetic-only algebraic 1q reducer. No Euler synthesis or angle wrapping.

RZ sums use exact rationals representing binary64 inputs. SX**2=X and
SX**4=I introduce no scalar phase. An explicit, exact angle-rounding ledger
decides which RZ sums may be emitted as one binary64 angle. This is not an
H6 adapter, adopted compiler policy, or floating-simulator error certificate.
"""
from fractions import Fraction
import math
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.passmanager import ConditionalController, DoWhileController, FlowControllerLinear
from qiskit.transpiler import PassManager, TransformationPass
from qiskit.circuit.library import RZGate, SXGate, XGate, IGate, CXGate
from qiskit.circuit import Barrier
from m5_preserve_oneq_compiler import PreserveOneQubitRuns, patched_pass_manager

POLICY = 'M5_ALGEBRAIC_ONEQ_ARTIFICIAL_PROTOTYPE_V1'
STANDARD = {'rz':RZGate, 'sx':SXGate, 'x':XGate, 'id':IGate}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def rational_payload(x):
    return dict(numerator=str(x.numerator), denominator=str(x.denominator))


class AngleLedger:
    def __init__(self, budget_radians):
        require(type(budget_radians) is float and math.isfinite(budget_radians) and budget_radians >= 0,
                'EXPLICIT_FINITE_NONNEGATIVE_BINARY64_ANGLE_BUDGET')
        self.budget = Fraction.from_float(budget_radians)
        self.spent = Fraction()

    @property
    def remaining(self):
        return self.budget-self.spent

    def commit(self, charge):
        require(type(charge) is Fraction and 0 <= charge <= self.remaining, 'EXACT_CUMULATIVE_ANGLE_BUDGET')
        self.spent += charge


def signature(gates):
    result = []
    for name, params in gates:
        require(name in STANDARD and len(params)==(1 if name=='rz' else 0), 'SUPPORTED_STANDARD_1Q_SIGNATURE')
        require(all(type(x) is float and math.isfinite(x) for x in params), 'FINITE_BINARY64_ANGLE')
        result.append((name,tuple(params)))
    return tuple(result)


def tokens(key):
    """Exact symbolic cancellation. Nonzero RZ sums are never reduced modulo pi."""
    stack = []
    for name, params in key:
        if name == 'id':
            continue
        if name == 'rz':
            value = Fraction.from_float(params[0])
            if value == 0:
                continue
            if stack and stack[-1][0]=='z':
                stack[-1][1] += value
                stack[-1][2].append(params[0])
                if stack[-1][1] == 0:
                    stack.pop()
            else:
                stack.append(['z',value,[params[0]]])
        else:
            units = 1 if name == 'sx' else 2
            if stack and stack[-1][0]=='x':
                stack[-1][1] = (stack[-1][1]+units)%4
                if stack[-1][1]==0:
                    stack.pop()
            else:
                stack.append(['x',units,[]])
    return tuple((kind,value,tuple(original)) for kind,value,original in stack)


class AlgebraicPlanner:
    def __init__(self):
        self.cache = {}
        self.hits = self.misses = 0

    def plan(self, gates, remaining):
        require(type(remaining) is Fraction and remaining >= 0, 'EXACT_REMAINING_BUDGET')
        key = signature(gates)
        if key in self.cache:
            self.hits += 1
        else:
            self.cache[key] = tokens(key)
            self.misses += 1
        out, charge = [], Fraction()
        fused, blocked, overflows = 0, 0, 0
        for kind, value, original in self.cache[key]:
            if kind=='x':
                if value in (1,3):out.append(('sx',()))
                if value in (2,3):out.append(('x',()))
                continue
            try:
                rounded = float(value)
            except OverflowError:
                rounded = float('inf')
            if not math.isfinite(rounded):
                out.extend(('rz',(a,)) for a in original)
                overflows += 1
                continue
            require(value==0 or rounded!=0, 'NO_NONZERO_ANGLE_UNDERFLOW')
            delta = abs(value-Fraction.from_float(rounded))
            if charge+delta <= remaining:
                out.append(('rz',(rounded,)))
                charge += delta
                fused += len(original)>1
            else:
                # Preserve the original binary64 angles; no clamp or retry.
                out.extend(('rz',(a,)) for a in original)
                blocked += 1
        rz_count = lambda seq:sum(name=='rz' for name,_ in seq)
        require(len(out)<=len(key) and rz_count(out)<=rz_count(key), 'NONINCREASING_GATE_AND_RZ_COUNTS')
        changed = len(out)<len(key)
        return dict(output=tuple(out) if changed else key,
            charge=charge if changed else Fraction(), changed=changed,
            input_gates=len(key),output_gates=len(out) if changed else len(key),
            input_RZ=rz_count(key),output_RZ=rz_count(out) if changed else rz_count(key),
            fused_RZ_groups=fused if changed else 0, budget_blocked_groups=blocked,
            overflow_preserved_groups=overflows)


class AlgebraicOneQubitPrototype(TransformationPass):
    def __init__(self, ledger, planner, recorder, *, artificial=False):
        super().__init__()
        require(artificial is True, 'ARTIFICIAL_SCOPE_ONLY_NO_H6_EXECUTION')
        self.ledger,self.planner,self.recorder = ledger,planner,recorder

    def run(self, dag):
        require(len(dag.qubits)<=3 and len(dag.clbits)==0 and
                set(dag.count_ops()) <= {'rz','sx','x','id','cx','barrier'}, 'SYNTHETIC_BASIS_SCOPE_MAX_3_QUBITS')
        # A custom lookalike can be omitted by collect_1q_runs; validate every
        # operation before collection, including gates that will be preserved.
        classes=dict(STANDARD,cx=CXGate,barrier=Barrier)
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
        self.recorder(dict(policy=POLICY, scope='ARTIFICIAL_ONLY', runs=total,changed_runs=changed,
            removed_gates=removed,removed_RZ=removed_rz,budget_blocked_groups=blocked,
            overflow_preserved_groups=overflow,budget=rational_payload(self.ledger.budget),
            spent_before=rational_payload(before),spent_after=rational_payload(self.ledger.spent),
            charged_this_pass=rational_payload(self.ledger.spent-before),
            global_phase_binary64_hex=initial_phase,global_phase_bits_unchanged=True,
            Euler_resynthesis=False,angle_wrap=False,small_angle_threshold=False,
            postcompile_correction=False,cache_hits=self.planner.hits,cache_misses=self.planner.misses))
        return dag


def compile_artificial(circuit, *, budget_radians, artificial=False):
    require(artificial is True and circuit.num_qubits<=3 and circuit.num_clbits==0 and
            not circuit.parameters, 'ARTIFICIAL_SCOPE_ONLY_NO_H6_EXECUTION')
    require(not getattr(circuit,'_calibrations',{}), 'NO_CALIBRATIONS_IN_PROTOTYPE')
    ledger,planner,records=AngleLedger(budget_radians),AlgebraicPlanner(),[]
    manager=patched_pass_manager(recorder=lambda row:None,basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17)
    sites=[]
    def replace(task):
        if type(task) is PreserveOneQubitRuns:
            sites.append(task)
            return AlgebraicOneQubitPrototype(ledger,planner,records.append,artificial=True)
        if type(task) is DoWhileController:
            return DoWhileController([replace(t) for t in task.tasks],do_while=task.do_while)
        if type(task) is ConditionalController:
            return ConditionalController([replace(t) for t in task.tasks],condition=task.condition)
        if type(task) is FlowControllerLinear:
            return FlowControllerLinear([replace(t) for t in task.tasks])
        require(not hasattr(task,'tasks'), 'UNRECOGNIZED_FLOW_CONTROLLER')
        return task
    manager.optimization=PassManager([replace(manager.optimization.to_flow_controller())])
    require(len(sites)==1, 'ONE_PROTOTYPE_PASS_SITE')
    out=manager.run(circuit,num_processes=1)
    require(records and ledger.spent<=ledger.budget, 'EXACT_LEDGER_WITHIN_BUDGET')
    return out,records,ledger
