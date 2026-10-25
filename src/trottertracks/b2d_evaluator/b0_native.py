"""Fixed B0 native pilot. Metadata planning imports no scientific packages."""
from dataclasses import asdict
from fractions import Fraction as F
from itertools import product
import json

from .outer_rte_execution import candidate_graph
from .outer import DStage, OuterCase, make_outer_plan, mathematical_stages, INNER
from .input_package import fingerprint
from .followup_io import registered_scope, SNAPSHOT, Stage1Failure
from trottertracks.b2d_adapter._published import symmetric_s2_sequence

IDS=("B3","D1","R2","R4","B4","F1")
PROFILE=dict(qiskit="1.3.0",basis_gates=("rz","sx","x","cx"),optimization_level=1,
    seed_transpiler=17,approximation_degree=1.0,qubits_initially_zero=False,
    lowering="official_definitions",backend=None,target=None,coupling_map=None,layout=None,routing=None)


def cases():
    return [row for row in candidate_graph("B0_METADATA").rows if row["candidate_id"] in IDS]


def case_key(row):return row["candidate_id"]+":"+str(row["case"]["total_time"])


def outer_plan(row,context=None):
    context=context or candidate_graph("B0_METADATA").settings.context
    return make_outer_plan(context,OuterCase(**row["case"]),r_rte=row["rte_parameters"]["r_rte"],
        cutoff=row["rte_parameters"]["K"],truncation_tolerance=1.,seed=41,independent_occurrences=True)


def selected_stages(row,plan=None):
    plan=plan or outer_plan(row)
    rs=[s for s in plan.body if not isinstance(s,DStage)]
    opposite=next((i for i,s in enumerate(rs) if s.evolution_time*row["case"]["total_time"]<0),len(rs)-1)
    return 0,opposite


def trajectory(row,variant,plan=None):
    if variant not in range(4):raise ValueError("Four fixed representatives only")
    plan=plan or outer_plan(row);rs=[s for s in plan.body if not isinstance(s,DStage)]
    selected=selected_stages(row,plan)[1];offset=0;result=[]
    for j,stage in enumerate(rs):
        spec=stage.payload;events=[]
        for step in range(spec.rte_steps):
            order=(spec.cutoff if variant==2 and j==0 and step==0 else
                   2 if variant==3 and j==selected and step==spec.rte_steps-1 else 0)
            indices=tuple(0 if variant==0 else (offset+k)%3 for k in range(order+1))
            events.append((order,indices));offset+=order+1
        result.append((spec,tuple(events)))
    return tuple(result)


def event_key(spec,event):return fingerprint((asdict(spec),event))
def rte_key(spec,events,controlled):return fingerprint((asdict(spec),events,controlled))
def block_key(label,time,controlled):return fingerprint((label,str(time),controlled))
def exp_key(label,time):return label+":"+str(time)


def inventory(*, diagnostic=False):
    configs={};events={};sequences={};blocks={};exps={};means=[];moments={}
    for row in cases()[:1] if diagnostic else cases():
        plan=outer_plan(row);rs=[s for s in plan.body if not isinstance(s,DStage)]
        for stage in rs:configs[str(stage.payload)]=stage.payload
        for time in (() if diagnostic else dict.fromkeys(s.evolution_time for s in rs)):
            spec=next(s.payload for s in rs if s.evolution_time==time)
            moments[str(spec)]=spec
        for stage in plan.body:
            if isinstance(stage,DStage):
                for op in stage.append_operations():
                    label="D_"+str(stage.context.block_ids.index(op.block_id))
                    exps[exp_key(label,op.evolution_time)]=(label,op.evolution_time)
                    for controlled in (False,True):blocks[block_key(label,op.evolution_time,controlled)]=(label,op.evolution_time,controlled)
        # Independent mathematical reference may use unfused D/R factors.
        for kind,w in (() if diagnostic else mathematical_stages(row["case"]["method"])):
            time=row["case"]["total_time"]*w/plan.case.step_count
            exps[exp_key(kind,time)]=(kind,time)
        if not diagnostic:exps[exp_key("physical",plan.case.total_time)]=("physical",plan.case.total_time)
        for variant in range(1 if diagnostic else 4):
            for spec,word in trajectory(row,variant,plan):
                for event in word:events[event_key(spec,event)]=(spec,event)
                for controlled in (False,True):sequences[rte_key(spec,word,controlled)]=(spec,word,controlled)
        for slot,index in (() if diagnostic else enumerate(selected_stages(row,plan))):
            spec=rs[index].payload;terms=[]
            for order in range(0,spec.cutoff+1,2):
                for indices in product(range(3),repeat=order+1):
                    event=(order,indices);key=event_key(spec,event)
                    events[key]=(spec,event);terms.append(key)
            means.append((case_key(row)+":"+str(slot),spec,terms))
    return dict(configs=configs,events=events,sequences=sequences,blocks=blocks,exps=exps,means=means,moments=moments)


def scope(*, diagnostic=False):
    inv=inventory(diagnostic=diagnostic)
    calls=[("b0."+kind,"input") for kind in ("fixture","split","prepare","full_inputs","input_gate")]
    for group,kind in (("configs","config"),("events","event"),("events","event_operator"),
                       ("sequences","rte_build"),("blocks","block_build"),("exps","exponential"),("moments","moments")):
        calls.extend(("b0."+kind,k) for k in inv[group])
    calls.extend(("b0.single_mean",key) for key,spec,terms in inv["means"])
    for row in cases()[:1] if diagnostic else cases():
        key=case_key(row)
        if not diagnostic:calls.append(("b0.analytic_mean",key))
        for variant in range(1 if diagnostic else 4):
            item=key+":"+str(variant)
            for kind in ("system_build","controlled_build","reference","system_gate","controlled_gate"):
                calls.append(("b0."+kind,item))
            for axis in (("cosine",) if diagnostic else ("cosine","sine")):
                aid=item+":"+axis
                for kind in ("wrapper_build","wrapper_reference","raw_wrapper_gate","raw_expectation", "compile",
                             "compiled_gate","compiled_expectation")+(() if diagnostic else ("measure",)):
                    calls.append(("b0."+kind,aid))
    return registered_scope("b0",calls,diagnostic_first=diagnostic,cases=1 if diagnostic else 12,
        trajectories=1 if diagnostic else 48,system_builds=1 if diagnostic else 48,controlled_builds=1 if diagnostic else 48,
        wrappers=1 if diagnostic else 96,compile_requests=1 if diagnostic else 96,
        operator_gates=4 if diagnostic else 288,expectations=2 if diagnostic else 192,single_means=0 if diagnostic else 24,
        event_matrix_terms=sum(len(terms) for key,spec,terms in inv["means"]),
        compiler=PROFILE,atol=1e-12,rtol=0,phase_fit=False,
        sampling="DETERMINISTIC_REPRESENTATIVES_NOT_IID_NOT_EXPECTED_COST")


class Native:
    """Dormant DF/native connection, separate from the independent references."""
    def __init__(self,run,ledger):
        import numpy as np
        from scipy.linalg import expm
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator,Statevector
        from trotterlib.df_hamiltonian import DFHamiltonian
        from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
        from trotterlib.df_partial_s2 import prepare_df_partial_s2,QiskitDFPartialS2CircuitBuilder
        from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
        from trotterlib.rte import make_rte_config,_make_event,event_unitary,finite_rte_operator_moments
        from trottertracks.b2d_adapter.adapter import context_from_preparation
        from .input_package import read_fixture_definition
        self.np,self.expm,self.QC,self.Operator,self.Statevector=np,expm,QuantumCircuit,Operator,Statevector
        self.run,self.ledger=run,ledger;self.cache={};self.compile_cache={}
        call=self.call
        definition=read_fixture_definition(run["root"])
        self.h=call("fixture","input",lambda:DFHamiltonian(constant=definition["constant"],
            one_body=np.asarray(definition["one_body"],dtype=np.complex128),lambdas=np.asarray(definition["lambdas"]),
            g_matrices=tuple(np.asarray(g,dtype=np.complex128) for g in definition["g_matrices"]),metadata=definition["metadata"]))
        split=call("split","input",lambda:split_df_hamiltonian_by_ld(self.h,2,weight_rule="lambda_frobenius_squared"))
        self.prep=call("prepare","input",lambda:prepare_df_partial_s2(self.h,split,identity_policy="extract_identity_phase",coefficient_atol=0.))
        self.context=context_from_preparation(self.prep)
        self.block_builder=QiskitDFPartialS2CircuitBuilder();self.rte_builder=QiskitDFRTEEventCircuitBuilder(basis_registry=self.prep.rte_preparation.basis_registry)
        self.make_config,self.make_event,self.event_unitary,self.moments=make_rte_config,_make_event,event_unitary,finite_rte_operator_moments
        call("full_inputs","input",self.full_inputs)
        self.input_diagnostics=call("input_gate","input",self.input_gate)

    def call(self,kind,key,fn):
        numeric=kind.endswith("_gate") or kind in ("input_gate","single_mean","analytic_mean","raw_expectation","compiled_expectation")
        try:
            return self.ledger.call("b0."+kind,fn,identity=key,summary=(lambda value:value) if numeric else None)
        except BaseException as error:
            if hasattr(error,"diagnostics") and not getattr(error,"recorded",False):
                self.ledger.rows.append(dict(failed_boundary="b0."+kind,identity=key,
                    correctness="FAIL",diagnostics=error.diagnostics,metrics=None,gamma=None))
                error.recorded=True
            raise

    def cached(self,kind,key,fn):
        pair=(kind,key)
        if pair not in self.cache:self.cache[pair]=self.call(kind,key,fn)
        return self.cache[pair]

    def full_inputs(self):
        np=self.np;eye=np.eye(4,dtype=complex);ann=[]
        # Independent CAR in Qiskit little-endian qubit order.
        for p in range(2):
            a=np.zeros((4,4),complex)
            for col in range(4):
                if (col>>p)&1:a[col^(1<<p),col]=(-1)**((col&((1<<p)-1)).bit_count())
            ann.append(a)
        def quadratic(matrix):return sum((matrix[i,j]*ann[i].conj().T@ann[j] for i in range(2) for j in range(2)),np.zeros((4,4),complex))
        self.eye=eye;self.blocks={"D_0":quadratic(self.h.one_body)}
        fragments=[lam*(quadratic(g)@quadratic(g)) for lam,g in zip(self.h.lambdas,self.h.g_matrices)]
        self.blocks.update(D_1=fragments[0],D_2=fragments[1])
        self.scalar=self.prep.constant_coefficient+self.prep.extracted_identity_coefficient
        self.matrices={**self.blocks,"D":sum(self.blocks.values()),
            "R":fragments[2]-self.prep.extracted_identity_coefficient*eye,
            "physical":self.h.constant*eye+self.blocks["D_0"]+sum(fragments)}
        # Independent orbital projectors; no candidate circuit Operator used here.
        eig,u=np.linalg.eigh(self.h.g_matrices[2]);u=u[:,np.argsort(-np.abs(eig))]
        zs=[eye-2*quadratic(np.outer(u[:,p],u[:,p].conj())) for p in range(2)]
        self.component_operators={}
        for c in self.prep.rte_preparation.symbolic_tail.components:
            matrix=eye.copy()
            for p in c.diagonal_pauli_support:matrix=zs[p]@matrix
            self.component_operators[c.component_id]=c.coefficient_sign*matrix
        self.normalized_tail=sum((c.probability*self.component_operators[c.component_id]
            for c in self.prep.rte_preparation.symbolic_tail.components),np.zeros((4,4),complex))
        self.state=np.zeros(4,complex);self.state[1]=1
        return dict(basis="QISKIT_LITTLE_ENDIAN_FULL_FOCK",sector_projection=[2,1])

    def check(self,a,b,code="B0_CORRECTNESS_FAILURE"):
        value=float(self.np.linalg.norm(a-b,ord=2))
        if not self.np.isfinite(value) or value>1e-12:
            error=Stage1Failure(code)
            def entries(matrix):
                return [[dict(real=float(z.real) if self.np.isfinite(z.real) else None,
                              imag=float(z.imag) if self.np.isfinite(z.imag) else None) for z in row] for row in matrix]
            error.diagnostics=dict(spectral_error=value if self.np.isfinite(value) else None,
                finite=bool(self.np.isfinite(value)),atol=1e-12,rtol=0,phase_fit=False,
                actual=entries(a),reference=entries(b))
            raise error
        return value

    def circuit_gate(self,circuit,reference,*,raw=None):
        try:
            return self.check(self.Operator(circuit.remove_final_measurements(inplace=False)).data,reference)
        except Stage1Failure as error:
            import base64,gzip,io
            from qiskit import qpy
            def encoded(qc):
                stream=io.BytesIO();qpy.dump(qc,stream)
                return dict(encoding="gzip+base64+qpy",data=base64.b64encode(gzip.compress(stream.getvalue(),mtime=0)).decode(),
                    global_phase=float(qc.global_phase),qubits=qc.num_qubits,clbits=qc.num_clbits)
            error.diagnostics["circuits"]={"actual":encoded(circuit)}
            if raw is not None:error.diagnostics["circuits"]["raw"]=encoded(raw)
            raise

    def input_gate(self):
        np=self.np;meta=self.run["inputs"]
        if (self.prep.hamiltonian_hash!=meta["hamiltonian_id"] or self.context.tail_id!=meta["tail_id"]
            or list(self.prep.deterministic_fragment_indices)!=[0,1] or len(self.component_operators)!=3):
            raise Stage1Failure("B0_INPUT_IDENTITY_MISMATCH")
        with np.load(self.run["root"]/SNAPSHOT/"inputs.npz",allow_pickle=False) as archive:
            checks={k:self.check(self.matrices[k][np.ix_([2,1],[2,1])],archive[k]) for k in ("D_0","D_1","D_2","physical")}
            checks["state"]=float(np.linalg.norm(self.state[[2,1]]-archive["state"]))
            if checks["state"]>1e-12:raise Stage1Failure("B0_STATE_MAPPING_MISMATCH")
        checks["R_identity"]=self.check(self.matrices["R"],self.prep.exact_rte_lambda_r*self.normalized_tail)
        checks["H_identity"]=self.check(self.matrices["physical"],self.matrices["D"]+self.matrices["R"]+self.scalar*self.eye)
        return checks

    def config(self,spec):
        return self.cached("config",str(spec),lambda:spec.make_config(self.prep.rte_preparation.symbolic_tail,self.make_config))

    def event(self,spec,descriptor):
        def make():
            config,distribution=self.config(spec);order,indices=descriptor
            if order not in distribution.orders or distribution.order_probabilities[distribution.orders.index(order)]<=0:
                raise Stage1Failure("B0_EVENT_OUTSIDE_SUPPORT")
            event=self.make_event(indices,self.prep.rte_preparation.symbolic_tail.components,distribution,distribution.orders.index(order))
            if event.event_probability<=0:raise Stage1Failure("B0_EVENT_PROBABILITY_FAILURE")
            return event
        return self.cached("event",event_key(spec,descriptor),make)

    def moment(self,spec):
        return self.cached("moments",str(spec),lambda:self.moments(self.normalized_tail,self.config(spec)[0]))

    def event_matrix(self,spec,descriptor):
        return self.cached("event_operator",event_key(spec,descriptor),
            lambda:self.event_unitary(self.event(spec,descriptor),self.component_operators))

    def exp(self,label,time):
        return self.cached("exponential",exp_key(label,time),lambda:self.expm(-1j*float(time)*self.matrices[label]))

    def block(self,label,time,controlled):
        def build():
            qc=self.QC(3 if controlled else 2)
            self.block_builder.append_block_evolution(qc,self.prep.deterministic_blocks[int(label[-1])],float(time),
                controlled=controlled,ancilla_qubit=2 if controlled else None)
            return qc
        return self.cached("block_build",block_key(label,time,controlled),build)

    def rte(self,spec,word,controlled):
        def build():
            from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
            events=tuple(self.event(spec,d) for d in word)
            request=DFRTEEventSequenceCircuitRequest(events=events,component_specs=self.prep.rte_preparation.component_specs,
                controlled=controlled,ancilla_qubit=2 if controlled else None,
                tail_id=self.prep.rte_preparation.symbolic_tail.tail_id,tail_hash=self.context.tail_id,
                occurrence_rte_steps=spec.rte_steps)
            return self.rte_builder.build_sequence(request).circuit
        return self.cached("rte_build",rte_key(spec,word,controlled),build)

    def render(self,row,variant,controlled):
        plan=outer_plan(row,self.context);words=iter(trajectory(row,variant,plan));qc=self.QC(3 if controlled else 2)
        for stage in plan.body:
            if isinstance(stage,DStage):
                for op in stage.append_operations():
                    label="D_"+str(self.context.block_ids.index(op.block_id))
                    qc.compose(self.block(label,op.evolution_time,controlled),inplace=True)
            else:
                spec,word=next(words);qc.compose(self.rte(spec,word,controlled),inplace=True)
        phase=-float(plan.case.total_time)*self.scalar
        if controlled:qc.p(phase,2)
        else:qc.global_phase+=phase
        return qc

    def reference(self,row,variant):
        # Independent mathematical product; rightmost factor acts first.
        plan=outer_plan(row);words=list(trajectory(row,variant,plan))[::-1];tail_index=0;factors=[]
        for kind,w in mathematical_stages(plan.case.method)*plan.case.step_count:
            time=plan.case.total_time*w/plan.case.step_count
            if kind=="D":
                for weight in map(F,symmetric_s2_sequence(INNER[plan.case.inner_arm]())):
                    factors.extend(self.exp(label,time*weight/(1 if label=="D_2" else 2))
                                   for label in ("D_0","D_1","D_2","D_1","D_0"))
            else:
                spec,word=words[tail_index];tail_index+=1
                if spec.duration!=time:raise Stage1Failure("B0_REFERENCE_STAGE_ORDER_MISMATCH")
                factors.extend(self.event_matrix(spec,d) for d in word[::-1])
        value=self.eye.copy()
        for matrix in factors:value=value@matrix
        return self.np.exp(-1j*float(plan.case.total_time)*self.scalar)*value

    def controlled(self,u):
        result=self.np.eye(8,dtype=complex);result[4:,4:]=u;return result

    def wrapper_reference(self,u,axis):
        # Explicit bit embedding and 2x2 H/Sdg, independent of wrapper renderer.
        np=self.np;h=np.array([[1,1],[1,-1]],complex)/np.sqrt(2);tail=h@(np.diag([1,-1j]) if axis=="sine" else np.eye(2))
        x=np.array([[0,1],[1,0]],complex);prep=np.kron(np.eye(2),np.kron(np.eye(2),x))
        return np.kron(tail,self.eye)@self.controlled(u)@np.kron(h,self.eye)@prep

    def expectation(self,circuit,reference):
        value=self.Operator(circuit.remove_final_measurements(inplace=False)).data[:,0]
        actual=float(self.np.sum(abs(value[:4])**2)-self.np.sum(abs(value[4:])**2))
        expected=float(self.np.sum(abs(reference[:4,0])**2)-self.np.sum(abs(reference[4:,0])**2))
        if not self.np.isfinite(actual) or abs(actual-expected)>1e-12:raise Stage1Failure("B0_HADAMARD_SIGNAL_FAILURE")
        return dict(actual=actual,reference=expected,error=abs(actual-expected))

    def analytic_mean(self,row):
        plan=outer_plan(row);exact=[];ideal=[];corrected=[];n=1.
        for kind,w in mathematical_stages(plan.case.method)*plan.case.step_count:
            time=plan.case.total_time*w/plan.case.step_count
            if kind=="D":
                exact.append(self.exp("D",time));value=self.eye.copy()
                for weight in map(F,symmetric_s2_sequence(INNER[plan.case.inner_arm]())):
                    for label in ("D_0","D_1","D_2","D_1","D_0"):
                        value=value@self.exp(label,time*weight/(1 if label=="D_2" else 2))
                ideal.append(value);corrected.append(value)
            else:
                spec=next(s.payload for s in plan.body if not isinstance(s,DStage) and s.evolution_time==time)
                moment=self.moment(spec);n*=moment.normalization_product
                exact.append(self.exp("R",time));ideal.append(self.exp("R",time));corrected.append(moment.corrected_operator)
        def combined(word):
            value=self.eye.copy()
            for factor in word:value=value@factor
            return self.np.exp(-1j*float(plan.case.total_time)*self.scalar)*value
        targets=dict(physical=self.exp("physical",plan.case.total_time),exact_outer=combined(exact),
                     ideal_tail=combined(ideal),corrected=combined(corrected))
        targets["raw"]=targets["corrected"]/n
        with open(self.run["root"]/SNAPSHOT/"result.json") as file:old=json.load(file)
        saved=next(r for r in old["rows"] if r["candidate_id"]==row["candidate_id"] and r["case"]["total_time"]==str(plan.case.total_time))
        signals={t:complex(self.np.vdot(self.state,m@self.state)) for t,m in targets.items()}
        errors={t:abs(z-complex(saved["outputs"]["signals"][t]["X"],saved["outputs"]["signals"][t]["Y"])) for t,z in signals.items()}
        if max(errors.values())>1e-12:raise Stage1Failure("B0_FULL_SECTOR_MEAN_MISMATCH")
        return dict(signals=signals,sector_signal_errors=errors,N=n,representatives_are_not_mean=True)

    def single_mean(self,spec,terms,inv):
        from trotterlib.rte import exact_enumerated_event_mean_operator,finite_taylor_operator
        events=[self.event(spec,inv["events"][k][1]) for k in terms]
        raw=exact_enumerated_event_mean_operator(events,self.component_operators)
        config,distribution=self.config(spec)
        corrected=finite_taylor_operator(self.normalized_tail,config.dimensionless_step_time,spec.cutoff)
        return dict(error=self.check(raw*distribution.exact_finite_distribution,corrected))

    def wrapper(self,controlled,axis):
        from trotterlib.rpe_hadamard_interrogation import build_prepared_hadamard_wrapper
        return build_prepared_hadamard_wrapper(controlled,ancilla=2,axis=axis,occupied_qubits=(0,))

    def compile(self,wrapper):
        from trottertracks.pr_dictionary_validation.gates import compiled_wrapper
        from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
        return compiled_wrapper(wrapper,CompilerProfile(),cache=self.compile_cache,
            identity_context=dict(input=self.run["inputs"]["input_data_sha256"],policy="B0_FULL_PREPARED_WRAPPER"))

    def measure(self,circuit):
        from trottertracks.pr_dictionary_validation.f2_execution import circuit_metrics
        return circuit_metrics(circuit)

    def provenance(self):
        configs={key:dict(config=asdict(value[0]),distribution=asdict(value[1]))
                 for (kind,key),value in self.cache.items() if kind=="config"}
        return dict(preparation_id=self.context.preparation_id,tail_id=self.context.tail_id,
            component_order=[dict(id=c.component_id,sign=c.coefficient_sign,coefficient_abs=c.coefficient_abs,
                probability=c.probability,support=list(c.diagonal_pauli_support))
                for c in self.prep.rte_preparation.symbolic_tail.components],
            configs=configs,compiler=PROFILE,occupation_qubits=[0],ancilla=2,
            bit_mapping="OpenFermion sector [1,2] -> Qiskit indices [2,1]",
            trajectories=[dict(case=case_key(row),representative=variant,
                words=[dict(spec=asdict(spec),events=word) for spec,word in trajectory(row,variant)])
                for row in cases() for variant in range(4)])


def execute(run,ledger,evidence,*,native_factory=Native):
    diagnostic=run["scope"].get("diagnostic_first",False)
    native=native_factory(run,ledger);inv=inventory(diagnostic=diagnostic);reports=[];compiled=[]
    evidence.write("graph.json",dict(cases=cases(),compiler=PROFILE,scope=run["scope"]))
    for key,(spec,descriptor) in inv["events"].items():
        native.event(spec,descriptor);native.event_matrix(spec,descriptor)
    for key,spec,terms in inv["means"]:
        reports.append(native.call("single_mean",key,lambda spec=spec,terms=terms:native.single_mean(spec,terms,inv)))
    for row in cases()[:1] if diagnostic else cases():
        key=case_key(row)
        mean=None if diagnostic else native.call("analytic_mean",key,lambda row=row:native.analytic_mean(row))
        for variant in range(1 if diagnostic else 4):
            item=key+":"+str(variant)
            system=native.call("system_build",item,lambda:native.render(row,variant,False))
            controlled=native.call("controlled_build",item,lambda:native.render(row,variant,True))
            reference=native.call("reference",item,lambda:native.reference(row,variant))
            gates=dict(system=native.call("system_gate",item,lambda:native.check(native.Operator(system).data,reference)),
                controlled=native.call("controlled_gate",item,lambda:native.check(native.Operator(controlled).data,native.controlled(reference))))
            for axis in (("cosine",) if diagnostic else ("cosine","sine")):
                aid=item+":"+axis
                wrapper=native.call("wrapper_build",aid,lambda:native.wrapper(controlled,axis))
                wr=native.call("wrapper_reference",aid,lambda:native.wrapper_reference(reference,axis))
                record=dict(trajectory=item,axis=axis,correctness="IN_PROGRESS",mean=mean,
                    operator_errors=dict(gates),metrics=None,gamma=None)
                ledger.rows.append(record)
                raw_error=native.call("raw_wrapper_gate",aid,lambda:native.circuit_gate(wrapper,wr))
                record["operator_errors"]["raw_wrapper"]=raw_error
                raw_signal=native.call("raw_expectation",aid,lambda:native.expectation(wrapper,wr))
                record["raw_signal"]=raw_signal
                circuit=native.call("compile",aid,lambda:native.compile(wrapper))
                compiled_error=native.call("compiled_gate",aid,lambda:native.circuit_gate(circuit,wr,raw=wrapper))
                compiled_signal=native.call("compiled_expectation",aid,lambda:native.expectation(circuit,wr))
                compiled.append((aid,circuit))
                record.update(correctness="PASS_EMPIRICAL_NO_PHASE_FIT",compiled_signal=compiled_signal)
                record["operator_errors"]["compiled_wrapper"]=compiled_error
    # No metric is released until every trajectory, mean and compiled gate passes.
    for (aid,circuit),record in (() if diagnostic else zip(compiled,ledger.rows)):
        record["metrics"]=native.call("measure",aid,lambda circuit=circuit:native.measure(circuit))
    return dict(study="B0",rows=ledger.rows,single_step_means=reports,input_diagnostics=native.input_diagnostics,
        source=run["source"],provenance=native.provenance(),gamma=None,shots=None,expected_cost=None,
        compile_requests=len(compiled),actual_transpile_count=len(native.compile_cache),
        cache_hits=len(compiled)-len(native.compile_cache),compiled_cache_keys=sorted(native.compile_cache),
        stop="MANDATORY_STOP_NO_RETRY_NO_RESUME",classification="REPRESENTATIVE_PILOT_NOT_EXPECTED_COST")
