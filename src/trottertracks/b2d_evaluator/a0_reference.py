"""Independent 90/128 digit reference; plan construction is stdlib metadata only."""
from dataclasses import asdict
from fractions import Fraction as F
import json
import math

from .outer_rte_execution import candidate_graph
from .outer import mathematical_stages, INNER
from trottertracks.b2d_adapter._published import symmetric_s2_sequence
from .evaluation import Node
from .followup_io import SNAPSHOT, registered_scope, Stage1Failure

IDS = ("S1","F1","B3","R4")
PRECISIONS = (90,128)
TARGETS = ("physical","exact_outer","ideal_tail","corrected","raw")


def binary_fraction(value):
    """No short decimal conversion. Used on each binary64 real/imag component."""
    value = float(value)
    if not math.isfinite(value): raise ValueError("Nonfinite snapshot value")
    return F(*value.as_integer_ratio())


def plan():
    nodes, memo, rows = [], {}, []
    def add(kind, deps=(), params=()):
        node = Node(kind,tuple(deps),tuple(params))
        if node not in memo: memo[node]=len(nodes); nodes.append(node)
        return memo[node]
    cases = [r for r in candidate_graph("SNAPSHOT_METADATA").rows if r["candidate_id"] in IDS]
    for digits in PRECISIONS:
        one = add("identity",params=(digits,))
        def mul(a,b): return add("multiply",(a,b),(digits,))
        def product(factors):
            value=one
            for factor in factors: value=mul(value,factor)
            return value
        def exp(label,time): return add("exponential",params=(digits,label,time))
        def power(matrix,count): return product([matrix]*count)
        for original in cases:
            c=original["case"];rte=original["rte_parameters"]
            key=original["candidate_id"]+":"+str(c["total_time"])
            count=c["macro_steps"]*(c["refinement"] or 1)
            exact,ideal,corrected,normalizations,bounds,d_checks=[],[],[],[],[],[]
            for kind,weight in mathematical_stages(c["method"])*count:
                time=c["total_time"]*weight/count
                if kind=="D":
                    word=[]
                    for w in map(F,symmetric_s2_sequence(INNER[c["inner_arm"]]())):
                        word.extend(exp(label,time*w/(1 if label=="D_2" else 2))
                                    for label in ("D_0","D_1","D_2","D_1","D_0"))
                    d=product(word); exact.append(exp("D",time));ideal.append(d);corrected.append(d)
                    d_checks.append(add("D_diagnostic",(d,),(digits,)))
                else:
                    spec=(digits,time,rte["r_rte"],rte["K"])
                    poly=one;term=one
                    # Independent recurrence in R itself; no DenseBackend/moment helper.
                    for degree in range(1,rte["K"]+2):
                        term=add("polynomial_term",(term,),(*spec,degree))
                        poly=add("add",(poly,term),(digits,))
                    finite=power(poly,rte["r_rte"])
                    exact.append(exp("R",time));ideal.append(exp("R",time));corrected.append(finite)
                    normalizations.append(add("normalization",params=spec))
                    bounds.append(add("taylor_bound",params=spec))
            phase=add("phase",params=(digits,c["total_time"]))
            def phased(word): return add("scalar_multiply",(phase,product(word)),(digits,))
            n=add("normalization_total",normalizations,(digits,))
            values=dict(physical=exp("physical",c["total_time"]),exact_outer=phased(exact),
                        ideal_tail=phased(ideal),corrected=phased(corrected))
            values["raw"]=add("divide",(values["corrected"],n),(digits,))
            outputs={}
            for target in TARGETS:
                request=add("target",(values[target],),(digits,key,target))
                outputs[target]=request
                outputs[target+"_signal"]=add("signal",(request,),(digits,key,target))
            for name,left,right in (("outer_physical","exact_outer","physical"),
                    ("ideal_physical","ideal_tail","physical"),("corrected_physical","corrected","physical"),
                    ("raw_physical","raw","physical"),("D_approximation","ideal_tail","exact_outer"),
                    ("finite_RTE","corrected","ideal_tail")):
                outputs[name]=add("residual",(values[left],values[right]),(digits,key,name))
            outputs.update(normalization=n,taylor_bound=add("bound_total",bounds,(digits,)),
                input_diagnostics=add("input_diagnostics",params=(digits,)),
                D_diagnostics=add("D_diagnostics",d_checks,(digits,)),
                phase_diagnostic=add("phase_diagnostic",(phase,),(digits,c["total_time"])),
                outer_unitarity=add("D_diagnostic",(values["exact_outer"],),(digits,)))
            rows.append(dict(candidate_id=original["candidate_id"],case=c,rte=rte,digits=digits,key=key,outputs=outputs))
    return nodes,rows


def scope():
    nodes,rows=plan()
    calls=[("a0.snapshot","input"),("a0.initialize","reference")]+[("a0."+n.kind,str(i)) for i,n in enumerate(nodes)]
    calls += [("a0.compare",row["key"]) for row in rows if row["digits"]==128]
    return registered_scope("a0",calls,rows=8,precisions=list(PRECISIONS),case_precisions=16,
        targets=80,signals=80,nodes=len(nodes),expm_methods={"90":"taylor","128":"pade"},
        numerical_status="EMPIRICAL_ONLY_U_NUM_NULL",atol="1e-40")


class Reference:
    """Dormant scientific arithmetic. Construct only inside --execute."""
    def __init__(self,arrays,metadata):
        from trottertracks.pr_dictionary_validation.reference import precision_context, exact
        self.contexts={d:precision_context(d) for d in PRECISIONS}
        self.exact=exact;self.data={}
        scalar=sum((F(x["coefficient"]) for x in metadata["scalar_terms"]),F())
        for digits,ctx in self.contexts.items():
            def lift(value):
                value=complex(value)
                return ctx.mpc(exact(ctx,binary_fraction(value.real)),exact(ctx,binary_fraction(value.imag)))
            values={label:ctx.matrix([[lift(x) for x in row] for row in matrix])
                    for label,matrix in arrays.items() if label!="state"}
            values["state"]=ctx.matrix([lift(x) for x in arrays["state"]])
            values["D"]=values["D_0"]+values["D_1"]+values["D_2"]
            values["R"]=exact(ctx,binary_fraction(metadata["exact_rte_lambda_r"]))*values["tail"]
            values["lambda"]=exact(ctx,binary_fraction(metadata["exact_rte_lambda_r"]))
            values["scalar"]=exact(ctx,scalar)
            self.data[digits]=values

    def norm(self,ctx,matrix):
        return max(ctx.svd(matrix,compute_uv=False))  # largest singular value, not Frobenius

    def perform(self,kind,args,params):
        digits=params[0];ctx=self.contexts[digits];v=self.data[digits]
        exact=lambda value:self.exact(ctx,value)
        if kind=="identity":return ctx.eye(2)
        if kind=="multiply":return args[0]*args[1]
        if kind=="add":return args[0]+args[1]
        if kind=="exponential":return ctx.expm(-ctx.j*exact(params[2])*v[params[1]],method="taylor" if digits==90 else "pade")
        if kind=="phase":return ctx.exp(-ctx.j*exact(params[1])*v["scalar"])
        if kind=="scalar_multiply":return args[0]*args[1]
        if kind=="divide":return args[0]/args[1]
        if kind=="polynomial_term":
            _,time,r,k,degree=params
            return args[0]*(-ctx.j*exact(time)*v["R"]/(r*degree))
        if kind=="normalization":
            _,time,r,k=params;x=abs(v["lambda"]*exact(time)/r)
            return ctx.fsum(x**n/ctx.factorial(n)*ctx.sqrt(1+(x/(n+1))**2) for n in range(0,k+1,2))**r
        if kind=="normalization_total":return ctx.fprod(args)
        if kind=="taylor_bound":
            _,time,r,k=params;x=abs(v["lambda"]*exact(time)/r)
            degree=k+2;term=x**degree/ctx.factorial(degree);total=term
            for _ in range(10000):
                term*=x/(degree+1);degree+=1;total+=term
                if abs(term)<=ctx.eps*abs(total):
                    return dict(step=total,r=r,time=str(time),K=k,nominal_only=True)
            raise Stage1Failure("INCONCLUSIVE_REFERENCE_CONVERGENCE")
        if kind=="bound_total":
            return dict(value=ctx.expm1(ctx.fsum(b["r"]*ctx.log1p(b["step"]) for b in args)),
                        stages=list(args),certificate=False)
        if kind=="target":return args[0]
        if kind=="D_diagnostic":return self.norm(ctx,args[0].H*args[0]-ctx.eye(2))
        if kind=="D_diagnostics":return list(args)
        if kind=="phase_diagnostic":
            import cmath
            legacy=cmath.exp(-1j*float(params[1])*float(v["scalar"]))
            lifted=ctx.mpc(exact(binary_fraction(legacy.real)),exact(binary_fraction(legacy.imag)))
            return dict(unit_modulus_error=abs(abs(args[0])-1),double_phase_difference=abs(args[0]-lifted))
        if kind=="signal":return (v["state"].H*args[0]*v["state"])[0]
        if kind=="residual":return self.norm(ctx,args[0]-args[1])
        if kind=="input_diagnostics":
            return dict(H_identity=self.norm(ctx,v["physical"]-v["D"]-v["R"]-v["scalar"]*ctx.eye(2)),
                tail_norm=self.norm(ctx,v["tail"]),state_norm=ctx.norm(v["state"]),
                D_hermiticity=self.norm(ctx,v["D"]-v["D"].H),R_hermiticity=self.norm(ctx,v["R"]-v["R"].H))
        raise ValueError("Unknown A0 operation")

    def encode(self,value,digits):
        ctx=self.contexts[digits]
        if isinstance(value,dict):return {k:self.encode(v,digits) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [self.encode(x,digits) for x in value]
        if isinstance(value,(str,bool,int)) or value is None:return value
        if hasattr(value,"rows"):
            return [[self.encode(value[i,j],digits) for j in range(value.cols)] for i in range(value.rows)]
        return dict(real=ctx.nstr(value.real,digits),imag=ctx.nstr(value.imag,digits))


def execute(run,ledger,evidence,*,reference_factory=Reference,snapshot_loader=None,previous_rows=None):
    def load():
        import numpy as np
        with np.load(run["root"]/SNAPSHOT/"inputs.npz",allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
        if set(arrays)!={"D_0","D_1","D_2","physical","tail","state"}:
            raise Stage1Failure("SNAPSHOT_ARRAY_DOMAIN")
        if any(a.dtype!=np.complex128 or not np.isfinite(a).all() or
               a.shape!=((2,) if k=="state" else (2,2)) for k,a in arrays.items()):
            raise Stage1Failure("SNAPSHOT_NUMERIC_DOMAIN")
        return arrays
    arrays=ledger.call("a0.snapshot",snapshot_loader or load,identity="input")
    ref=ledger.call("a0.initialize",lambda:reference_factory(arrays,run["inputs"]),identity="reference")
    nodes,rows=plan();evidence.write("graph.json",dict(nodes=[asdict(n) for n in nodes],rows=rows))
    values=[]; records={}
    for i,node in enumerate(nodes):
        values.append(ledger.call("a0."+node.kind,lambda n=node:ref.perform(n.kind,[values[d] for d in n.dependencies],n.parameters),identity=str(i)))
        # Persist completed case-precision records even if a later node fails.
        for row in rows:
            if max(row["outputs"].values())==i:
                record={**{k:v for k,v in row.items() if k!="outputs"},
                    "outputs":{k:ref.encode(values[j],row["digits"]) for k,j in row["outputs"].items()},
                    "u_num":None,"certified":False}
                ledger.rows.append(record);records[(row["key"],row["digits"])]=record
    old=(json.loads((run["root"]/SNAPSHOT/"result.json").read_text())["rows"]
         if previous_rows is None else previous_rows)
    comparisons=[]
    for row in rows:
        if row["digits"]!=128:continue
        def compare(row=row):
            ctx=ref.contexts[128]
            def number(record):return ctx.mpc(record["real"],record["imag"])
            a=records[(row["key"],90)]["outputs"];b=records[(row["key"],128)]["outputs"]
            changes={k:abs(number(a[k])-number(b[k])) for k in a if k.endswith("_signal") or k in (
                "outer_physical","ideal_physical","corrected_physical","raw_physical","D_approximation","finite_RTE","normalization")}
            previous=next(x for x in old if x["candidate_id"]==row["candidate_id"] and x["case"]["total_time"]==str(row["case"]["total_time"]))
            bound=number(b["taylor_bound"]["value"]).real
            changes["taylor_bound"]=abs(number(a["taylor_bound"]["value"])-bound)
            signal_differences={t:abs(number(b[t+"_signal"])-ctx.mpc(
                ref.exact(ctx,binary_fraction(previous["outputs"]["signals"][t]["X"])),
                ref.exact(ctx,binary_fraction(previous["outputs"]["signals"][t]["Y"])))) for t in TARGETS}
            bias=abs(number(b["corrected_signal"])-number(b["physical_signal"]))
            operator_keys=dict(outer_physical="outer_operator_error",ideal_physical="ideal_tail_physical_error",
                corrected_physical="corrected_physical_error",raw_physical="raw_physical_error",
                D_approximation="D_approximation_error",finite_RTE="finite_RTE_error")
            operator_differences={k:abs(number(b[k]).real-ref.exact(ctx,binary_fraction(previous["outputs"][old_key])))
                                  for k,old_key in operator_keys.items()}
            return dict(key=row["key"],precision_changes=ref.encode(changes,128),
                saved_signal_differences=ref.encode(signal_differences,128),bias=ref.encode(bias,128),
                saved_operator_differences=ref.encode(operator_differences,128),
                saved_bound_difference=ref.encode(abs(bound-ref.exact(ctx,binary_fraction(previous["outputs"]["taylor_residual"]["complex_signal_bound"]))),128),
                saved_bias_difference=ref.encode(abs(bias-ref.exact(ctx,binary_fraction(previous["outputs"]["signals"]["corrected_physical_signal_error"]))),128),
                stable=all(x<=ctx.mpf("1e-40") for x in changes.values()),
                nominal_bound_consistent=number(b["finite_RTE"]).real<=bound,
                classification="EMPIRICAL_ONLY_UNRESOLVED_IF_MISMATCH",u_num=None,certified=False)
        comparisons.append(ledger.call("a0.compare",compare,identity=row["key"]))
    return dict(study="A0",rows=ledger.rows,comparisons=comparisons,source=run["source"],
                scientific_status="EMPIRICAL_CONSISTENCY" if all(c["stable"] and c["nominal_bound_consistent"] for c in comparisons) else "INCONCLUSIVE",
                stop="MANDATORY_STOP_NO_RETRY_NO_RESUME",u_num=None,certified=False)
