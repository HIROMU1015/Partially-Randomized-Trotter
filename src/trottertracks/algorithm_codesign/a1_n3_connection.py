"""N3 connections. Passive import; real acquisition requires separate permission.

Tests inject invented laws, states and word tapes. No default science runner,
file read, RNG construction, compile, qualification or state solve on import.
The public measurement result contains bits/metadata only. Python privacy here
is an interface boundary, not a sandbox against introspection of engine objects.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import subprocess

from . import a1_control_variates as cv
from .a1_connections import proposed_control_banks
from .df_correction_comparison.intervals import configure, iv, upper, ceil_upper, outward_upper

M = 2**53
METHODS = ('B0', 'B1-O', 'B1-OL', 'A1')
ENGINE_COMMIT = 'fdf00cee3c87d804845497afc89ac6a71f3b3f59'
ENGINE_DIR = Path('artifacts/o2_a1_new_target/2026-11-06/engine_qualification_run01')


def word53(source):
    k = source()
    if type(k) is not int or not 0 <= k < M:
        raise ValueError('independent uniform 53-bit integer word required')
    return k


def below(n, source):
    """Unbiased integer rejection, NOT rare-stratum path rejection.

    Each source call is counted by Streams. The caller fixes a word budget;
    exhaustion aborts the campaign, never changes the conditional law.
    """
    if type(n) is not int or not 1 <= n <= M:
        raise ValueError('integer count total outside grid')
    limit = M // n * n
    while True:
        k = word53(source)
        if k < limit:
            return k % n


@dataclass(frozen=True)
class PathChoice:
    order: int
    labels: tuple

    @property
    def event(self):
        return cv.EventMetadata(self.order, tuple(self.labels), (-1)**(self.order//2))


class ConditionalSampler:
    """Same supplied 53-bit Q for canonical and directly conditioned paths."""
    def __init__(self, law, law_id, a, b):
        from types import SimpleNamespace
        self.law, self.a, self.b = law, a, b
        if law.orders.labels != (0, 2):
            raise ValueError('fixed n0/n2 law required')
        for categorical in (law.orders, law.labels):
            if any((p*M).denominator != 1 for p in categorical.masses):
                raise ValueError('saved actual 53-bit integer masses required')
        self.bank, self.order_bank, self.five_bank = proposed_control_banks(
            SimpleNamespace(law=law), law_id=law_id, a=a, b=b)
        self.patterns = ((1,1,0,0), (1,0,0,0), (0,0,0,0), (0,0,1,0), (0,0,1,1))

    @classmethod
    def from_record(cls, record, law_id, a, b):
        if record['uniform_grid_size'] != M or record['application_label_order'] != 'products_then_rotation':
            raise ValueError('saved law grid/application order mismatch')
        orders = cv.CategoricalLaw(tuple(record['orders']), tuple(F(x['exact']) for x in record['order_masses']), law_id)
        labels = cv.CategoricalLaw(tuple(x['label'] for x in record['label_masses']), tuple(F(x['exact']) for x in record['label_masses']), law_id)
        return cls(cv.OccurrenceLaw(orders, labels), law_id, a, b)

    def cells(self, method):
        if method in ('B0', 'A1'):
            return (('all', F(1)),)
        if method == 'B1-O':
            return tuple((str(n), p) for n,p in zip(self.law.orders.labels, self.law.orders.masses) if p > 0)
        if method == 'B1-OL':
            return tuple((str(i), p) for i,key in enumerate(self.patterns) if (p := self.five_bank.pattern_mass(key)) > 0)
        raise ValueError('unknown method')

    def _pick(self, categorical, source, allowed=None):
        rows = [(label, int(p*M)) for label,p in zip(categorical.labels,categorical.masses)
                if p > 0 and (allowed is None or allowed(label))]
        total = sum(p for _,p in rows)
        if not total:
            raise ValueError('empty positive conditional support')
        if len(rows) == 1:
            return rows[0][0]
        k = below(total, source)
        for label,p in rows:
            if k < p:
                return label
            k -= p
        raise AssertionError('integer mass coverage')

    def cell_of(self, method, path):
        self.law.validate(path.event)
        if method in ('B0','A1'):
            return 'all'
        if method == 'B1-O':
            return str(path.order)
        if method == 'B1-OL':
            return str(self.patterns.index(self.five_bank.values((path.event,))))
        raise ValueError('unknown method')

    def draw(self, method, cell, source):
        if cell not in dict(self.cells(method)):
            raise ValueError('missing/zero positive stratum')
        if method in ('B0','A1'):
            n = self._pick(self.law.orders, source)
        else:
            n = int(cell) if method == 'B1-O' else (0 if int(cell) < 2 else 2)
        labels = []
        for pos in range(n+1):
            allowed = None
            if method == 'B1-OL':
                i = int(cell)
                if pos == 0:
                    equal = i in (0,3,4)
                    allowed = lambda x, equal=equal: (x == self.a) == equal
                elif pos == 1 and i in (3,4):
                    allowed = lambda x, equal=(i==4): (x == self.b) == equal
            labels.append(self._pick(self.law.labels, source, allowed))
        result = PathChoice(n, tuple(labels))
        if self.cell_of(method,result) != cell:
            raise ValueError('conditional metadata mismatch')
        return result


@dataclass(frozen=True)
class Domain:
    method: str
    axis: str
    stage: str
    cell: str

    def __post_init__(self):
        if self.method not in METHODS or self.axis not in ('X','Y') or self.stage not in ('train','cal','prod') or not self.cell:
            raise ValueError('method/axis/stage/stratum domain required')

    @property
    def token(self):
        return json.dumps((self.method,self.axis,self.stage,self.cell),separators=(',',':'))


class Streams:
    """Explicit independent domains. No RNG is created here.

    Disjoint handles/trial IDs detect reuse, but do not prove PRNG independence.
    Each trial's path-word cap is a scientific execution limit, not a CPU cap.
    """
    def __init__(self, path_word_cap):
        if type(path_word_cap) is not int or path_word_cap < 1:
            raise ValueError('preselected positive path word cap required')
        self.cap, self.sources, self.used, self.counts = path_word_cap, {}, set(), {}

    def bind(self, domain, path_source, bit_source):
        old = [id(s) for pair in self.sources.values() for s in pair]
        if domain in self.sources or id(path_source) == id(bit_source) or any(id(s) in old for s in (path_source,bit_source)):
            raise ValueError('stream reuse across stage/axis/method/stratum/domain')
        self.sources[domain] = (path_source,bit_source)

    def claim(self, domain, trial):
        if type(trial) is not int or trial < 0 or (domain,trial) in self.used or domain not in self.sources:
            raise ValueError('unknown domain or repeated trial')
        self.used.add((domain,trial)); self.counts[domain,trial] = [0,0]
        def one(index):
            def draw():
                c = self.counts[domain,trial]
                if c[index] >= (self.cap if index == 0 else 1):
                    raise ValueError('word budget exhausted; STOP without retry')
                c[index] += 1
                return word53(self.sources[domain][index])
            return draw
        return one(0), one(1)


def pcg64_streams(master_seed, domains, *, path_word_cap):
    """DEFERRED N3 RNG construction. Not used by current artificial tests.

    Distinct SeedSequence children for path/bit; IID uniform words remain a
    model assumption, not a theorem about deterministic PCG64.
    """
    import numpy as np
    domains = tuple(sorted(domains, key=lambda d:d.token))
    if len(set(domains)) != len(domains) or type(master_seed) is not int or master_seed < 0:
        raise ValueError('unique frozen domains and master seed required')
    children = np.random.SeedSequence(master_seed).spawn(2*len(domains))
    book = Streams(path_word_cap)
    for i,d in enumerate(domains):
        def source(child):
            generator = np.random.PCG64(child)
            return lambda: int(generator.random_raw()) >> 11
        book.bind(d, source(children[2*i]), source(children[2*i+1]))
    return book


class MeasurementFacade:
    __slots__ = ('__probability','__scope')

    def __init__(self, probability, scopes):
        if set(scopes) != {'X','Y'} or any(s.axis != axis or s.weight_bound != 1 for axis,s in scopes.items()):
            raise ValueError('bound X/Y unit-weight scopes required')
        self.__probability, self.__scope = probability, dict(scopes)

    def acquire(self, sampler, streams, domain, trial, *, sampling_method=None):
        """Exactly one private Born evaluation and fresh bit, never its mean."""
        from .a1_engine_connection import threshold_count
        scope = self.__scope[domain.axis]
        if scope.law_id != sampler.bank.law_id:
            raise ValueError('law identity mismatch')
        method = sampling_method or domain.method
        if method != domain.method and not (domain.method == 'A1' and domain.stage == 'train' and method == 'B1-OL'):
            raise ValueError('only declared conditional A1 training may change acquisition law')
        path_words, bit_word = streams.claim(domain,trial)
        path = sampler.draw(method,domain.cell,path_words)
        threshold = threshold_count(self.__probability(path.order,path.labels,domain.axis))
        bit = int(bit_word() >= threshold)
        identity = hashlib.sha256(json.dumps((scope.target_id,path.order,path.labels)).encode()).hexdigest()
        return cv.Measurement(domain.token,str(trial),(path.event,),bit,F(1),identity,scope.circuit_id)


def validate_batch(sampler, batch, method, cell, stage):
    token = Domain(method,batch.scope.axis,stage,cell).token
    if batch.stage != stage or batch.scope.law_id != sampler.bank.law_id or batch.scope.weight_bound != 1:
        raise ValueError('batch stage/law/weight mismatch')
    for r in batch.records:
        if r.stream_id != token or r.wrapper_id != batch.scope.circuit_id or r.weight != 1 or len(r.events) != 1:
            raise ValueError('batch method/axis/domain/wrapper mismatch')
        path = PathChoice(r.events[0].order,r.events[0].selected_component_ids)
        if sampler.cell_of(method,path) != cell:
            raise ValueError('batch conditional metadata mismatch')


def inspect_certified_bundle(root):
    """Static byte/source checks only; no requalification or numerical load."""
    from . import a1_engine_connection as engine
    root = Path(root); binding = engine.inspect_fixed_bundle(root)
    paths = subprocess.check_output(['git','ls-tree','-r','--name-only',ENGINE_COMMIT,'--',str(ENGINE_DIR)],cwd=root,text=True).splitlines()
    if not paths:
        raise ValueError('missing fixed engine proof')
    for p in paths:
        if (root/p).read_bytes() != subprocess.check_output(['git','show',ENGINE_COMMIT+':'+p],cwd=root):
            raise ValueError('engine proof byte mismatch: '+p)
    profile = engine.read(root/engine.DOC/'PROFILE_LOCK.json')
    for p,h in profile['source_sha256'].items():
        if hashlib.sha256((root/p).read_bytes()).hexdigest() != h:
            raise ValueError('certified SOURCE mismatch: '+p)
    binding['engine_commit'] = ENGINE_COMMIT
    binding['engine_certificate_sha256'] = hashlib.sha256((root/ENGINE_DIR/'ENGINE_ERROR_CERTIFIED.json').read_bytes()).hexdigest()
    return binding


def attach_saved_proof(engine, rows, certificate):
    """Deserialize certified exact matrices; NEVER regenerate their intervals."""
    table = {}
    for row in rows:
        key = row['key'][0], tuple(row['key'][1])
        if key in table:
            raise ValueError('duplicate certified gate key')
        table[key] = dict(key=key,matrix_sha256=row['matrix_sha256'],
            matrix=[[(F(re),F(im)) for re,im in line] for line in row['matrix_complex_rationals']],error_upper=F(row['error_upper']))
    if not table or certificate['status'] != 'CONDITIONAL_UNIFORM_ENGINE_BOUND_CERTIFIED':
        raise ValueError('missing qualified certificate')
    engine.table = table
    engine.certificate = {'max_gate_count':certificate['max_gate_count']}


def verify_runtime(root):
    """Reuse static dependency/native profile, allowing only private cache paths."""
    from . import a1_engine_connection as engine
    old = engine.read(Path(root)/engine.DOC/'PROFILE_LOCK.json')
    now = engine.technical_profile(root)
    for profile in (old,now):
        profile.pop('source_sha256')  # certified source subset was checked separately
        for name in ('MPLCONFIGDIR','NUMBA_CACHE_DIR','XDG_CACHE_HOME'):
            profile['thread_environment'].pop(name,None)
    if engine.canonical(old) != engine.canonical(now):
        raise ValueError('certified runtime dependency/native profile mismatch')


def load_certified_services(root):
    """DEFERRED REAL N3 binding, not invoked in this technical-only task."""
    from . import a1_engine_connection as engine
    from .a1_new_target_n1 import A,B
    root = Path(root); binding = inspect_certified_bundle(root)
    verify_runtime(root)
    saved, same = engine.from_fixed_bundle(root)
    if same['N1_identities'] != binding['N1_identities']:
        raise ValueError('payload changed during binding')
    certificate = engine.read(root/ENGINE_DIR/'ENGINE_ERROR_CERTIFIED.json')
    if certificate['state_identity'] != binding['state_identity']:
        raise ValueError('state certificate identity mismatch')
    attach_saved_proof(saved,engine.read(root/ENGINE_DIR/'GATE_CERTIFICATES.json'),certificate)
    sampler = ConditionalSampler.from_record(engine.read(root/engine.P1/'LAW.json'),binding['N1_identities']['law_sha256'],A,B)
    target = hashlib.sha256(engine.canonical(binding).encode()).hexdigest()
    scopes = {axis:cv.Scope(target,sampler.bank.law_id,binding['engine_certificate_sha256'],'m0',axis,F(1)) for axis in ('X','Y')}
    return MeasurementFacade(saved.probability,scopes), SavedPrices(saved), sampler, scopes, certificate


class SavedPrices:
    """Same saved selected logical circuit; existing compiler/cache, no oracle."""
    def __init__(self, engine):
        self.__engine = engine

    def price(self, path, axis, compiler, *, cache=None):
        from .shared_cpu import module
        # The serialized preparation is not DFPartialS2RepeatedRequest. Reuse
        # certified selected_circuit instead of rebuilding/preparing that input.
        expected = dict(basis_gates=('rz','sx','x','cx'),backend_name=None,coupling_map=None,
            optimization_level=2,layout_method=None,routing_method=None,transpiler_seed=17,qiskit_version='1.3.0')
        if any(getattr(compiler,k) != v for k,v in expected.items()):
            raise ValueError('same fixed compiler profile required')
        circuit = self.__engine.selected_circuit(path.order,path.labels,axis)
        from qiskit import ClassicalRegister
        register = ClassicalRegister(1,'readout'); circuit.add_register(register)
        circuit.measure(self.__engine.n,register[0])
        key = hashlib.sha256(json.dumps((self.__engine.prep['preparation_hash'],self.__engine.policy['policy_fingerprint'],path.order,path.labels,axis,ENGINE_COMMIT,'full_no_prep')).encode()).hexdigest()
        if cache is None:
            cost = module('rte_compiled_cost').transpile_and_measure_cost(circuit,compiler,circuit_fingerprint=key,actual_circuit_fingerprint=key)
            return cost,False
        cost,_key,hit = cache.get_or_transpile(circuit,compiler,circuit_fingerprint=key)
        return cost,hit


def collect(facade, sampler, streams, scope, domain, count, price, *, sampling_method=None, sink=None):
    """DEFERRED fixed-quota stage; tests use mocks only. Charge every trial."""
    if type(count) is not int or count < 1 or scope.axis != domain.axis:
        raise ValueError('fixed positive quota and bound axis required')
    records, costs = [], []
    for trial in range(count):
        record = facade.acquire(sampler,streams,domain,trial,sampling_method=sampling_method)
        if record.wrapper_id != scope.circuit_id:
            raise ValueError('facade/collector scope mismatch')
        records.append(record)
        if sink is not None:
            sink('bit',record)
        path = PathChoice(record.events[0].order,record.events[0].selected_component_ids)
        cost = price(path,scope.axis)
        if type(cost.rz_count) is not int or cost.rz_count < 0:
            raise ValueError('missing/invalid full interrogation RZ price')
        # No T/native synthesis was obtained. Keep unavailable metrics UNKNOWN.
        costs.append(cv.TrialCost(scope,record.stream_id,record.trial_id,domain.stage,cv.FULL_SCOPE,
            (None,F(cost.cx_count) if hasattr(cost,'cx_count') else None,F(cost.rz_count),None,None),F(0),F(0),0))
        if sink is not None:
            sink('cost',costs[-1])
    return cv.Batch(scope,domain.stage,tuple(records)), tuple(costs)


def merge_training(batches):
    batches = tuple(batches); cv.require_disjoint(*batches)
    if any(b.stage != 'train' for b in batches):
        raise ValueError('conditional training only')
    return cv.Batch(batches[0].scope,'train',tuple(r for b in batches for r in b.records))


def freeze(bank, training, range_guard):
    """Reuse fixed 3-control OLS; never drop controls, clip or refit on guard."""
    import numpy as np
    model = cv.fit_and_freeze(bank,training)
    design = np.asarray([[1,*bank.values(r.events)] for r in training.records],dtype=float)
    rank = int(np.linalg.matrix_rank(design))
    if rank != len(bank.controls)+1:
        raise ValueError('rank deficient fixed-control training; STOP')
    if model.output_range.hi-model.output_range.lo > F(range_guard):
        raise ValueError('CV range guard; STOP without clipping/refit')
    return model,rank


@dataclass(frozen=True)
class Calibration:
    count: int
    width: F
    sample_variance: F
    variance_upper: F
    delta: F


def calibrate(values, lo, hi, delta):
    """Maurer-Pontil v1 Thm10; independent fixed-n IID cal, frozen lambda."""
    import mpmath as mp
    values = tuple(map(F,values)); lo,hi,delta = F(lo),F(hi),F(delta)
    if len(values) < 2 or hi <= lo or not 0 < delta < 1 or any(not lo <= x <= hi for x in values):
        raise ValueError('independent bounded fixed calibration required')
    n = len(values); width = hi-lo; mean = sum(values,F(0))/n
    variance = sum(((x-mean)**2 for x in values),F(0))/(n-1)
    configure()
    v = (mp.iv.sqrt(iv(variance))+iv(width)*mp.iv.sqrt(2*mp.iv.ln(iv(1/delta))/(n-1)))**2
    bound = min(width**2/4,outward_upper(upper(v)))
    return Calibration(n,width,variance,bound,delta)


def sufficient_shots(calibration, epsilon, engine_error, centering_error, delta):
    """Two-sided bounded Bernstein/Hoeffding; outward ceil, no science default."""
    import mpmath as mp
    e = F(epsilon)-F(engine_error)-F(centering_error); delta = F(delta)
    if e <= 0 or not 0 < delta < 1 or min(F(engine_error),F(centering_error)) < 0:
        raise ValueError('positive error headroom and failure budget required')
    configure(); log = mp.iv.ln(iv(2/delta)); D = calibration.width
    nb = max(1,ceil_upper(log*(iv(2*calibration.variance_upper/e**2)+iv(2*D/(3*e)))))
    nh = max(1,ceil_upper(iv(D**2/(2*e**2))*log))
    return min(nb,nh), {'Bernstein':nb,'Hoeffding':nh,'headroom':str(e)}


def confidence_cells(alpha, cells):
    """Simultaneous 4 methods x 2 axes, then cal/prod, then all positive cells."""
    alpha = F(alpha)
    if not 0 < alpha < 1 or type(cells) is not int or cells < 1:
        raise ValueError('confidence budget/cell count')
    return alpha/(16*cells), alpha/(16*cells)


def choose_allocation(masses, calibrations, pilot_prices, headroom, delta_prod, cap, *, units):
    """Finite epsilon allocation, all positive cells retained; no prod feedback.

    Candidates: equal epsilon, plus all positive integer j summing to units,
    epsilon_h = e*j_h/(units*p_h). Only pilot/cal may choose the frozen plan.
    """
    masses = tuple(map(F,masses)); headroom = F(headroom)
    if not masses or min(masses) <= 0 or sum(masses) != 1 or headroom <= 0 or type(cap) is not int or cap < len(masses):
        raise ValueError('all positive strata and complete mass required')
    H = len(masses)
    if len(calibrations) != H or len(pilot_prices) != H or type(units) is not int or units < H:
        raise ValueError('complete cal/price rows and finite allocation grid required')
    def compositions(total, length):
        if length == 1:
            yield (total,)
        else:
            for first in range(1,total-length+2):
                for rest in compositions(total-first,length-1):
                    yield (first,)+rest
    candidates = [(headroom,)*H]
    candidates.extend(tuple(headroom*j/(units*p) for j,p in zip(js,masses)) for js in compositions(units,H))
    feasible = []
    for es in candidates:
        counts = tuple(sufficient_shots(c,e,0,0,delta_prod)[0] for c,e in zip(calibrations,es))
        if sum(counts) <= cap:
            score = sum((max(F(1),F(price))*n for price,n in zip(pilot_prices,counts)),F(0))
            feasible.append((score,es,counts))
    if not feasible:
        raise ValueError('sufficient production quota exceeds cap; STOP')
    return min(feasible)


@dataclass(frozen=True)
class ProductionPlan:
    method: str
    scope: cv.Scope
    cells: tuple  # (cell, exact mass, quota, statistical epsilon)
    antecedents: tuple
    calibrations: tuple
    model: object
    epsilon: F
    engine_error: F
    alpha: F


def plan_production(sampler, method, calibration_batches, *, epsilon, engine_error,
                    alpha, cap, pilot_prices, units, model=None, pilot_batches=()):
    """Freeze all quotas before production; only retained pilot/cal bit data."""
    rows = tuple(calibration_batches)
    expected = sampler.cells(method)
    if len(rows) != len(expected) or tuple(k for k,_ in rows) != tuple(k for k,_ in expected):
        raise ValueError('all positive calibration cells in frozen order required')
    batches = tuple(b for _,b in rows)
    if any(b.stage != 'cal' for b in batches) or not batches:
        raise ValueError('calibration stage required')
    scope = batches[0].scope
    if (model is None) != (method != 'A1') or (model is not None and model.bank != sampler.bank):
        raise ValueError('A1 requires the same frozen control bank')
    antecedents = tuple(pilot_batches) + ((model.training,) if model is not None else ()) + batches
    cv.require_disjoint(*antecedents)
    delta_cal,delta_prod = confidence_cells(alpha,len(expected))
    calibrations = []
    for (key,batch),(_,mass) in zip(rows,expected):
        validate_batch(sampler,batch,method,key,'cal')
        if model is None:
            lo,hi,values = F(-1),F(1),tuple(r.raw for r in batch.records)
        else:
            lo,hi,values = model.output_range.lo,model.output_range.hi,model.apply(batch)
        calibrations.append(calibrate(values,lo,hi,delta_cal))
    e = F(epsilon)-F(engine_error)  # exact saved nu gives zero centering error
    if F(engine_error) < 0 or e <= 0:
        raise ValueError('certified engine error leaves no precision')
    _,es,counts = choose_allocation(tuple(p for _,p in expected),tuple(calibrations),
        pilot_prices,e,delta_prod,cap,units=units)
    return ProductionPlan(method,scope,tuple((k,p,n,eh) for (k,p),n,eh in zip(expected,counts,es)),
        antecedents,tuple(calibrations),model,F(epsilon),F(engine_error),F(alpha))


def production_mean(plan, sampler, production_batches):
    rows = tuple(production_batches)
    if tuple(k for k,_ in rows) != tuple(k for k,_,_,_ in plan.cells):
        raise ValueError('positive production stratum omitted/reordered')
    cv.require_disjoint(*plan.antecedents,*(b for _,b in rows))
    result = F(0)
    for (key,batch),(_,mass,quota,_) in zip(rows,plan.cells):
        if batch.stage != 'prod' or batch.scope != plan.scope or len(batch.records) != quota:
            raise ValueError('frozen production quota/scope mismatch')
        validate_batch(sampler,batch,plan.method,key,'prod')
        values = plan.model.apply(batch) if plan.model is not None else tuple(r.raw for r in batch.records)
        result += mass*sum(values,F(0))/quota
    return result


def summarize_costs(records, classical):
    """Realized campaign vector only; no expected-cost confidence assertion."""
    result = cv.account_costs(records)
    names = ('generation','fit','correction','compile','cache','engine_simulation','setup')
    if set(classical) != set(names) or any(v is not None and F(v) < 0 for v in classical.values()):
        raise ValueError('separate complete classical cost categories required')
    result['classical_seconds_by_category'] = classical
    result['classical_cost_complete'] = all(v is not None for v in classical.values())
    result['classical_setup_seconds'] = classical['setup']
    result['classical_online_seconds'] = (None if not result['classical_cost_complete'] else
        sum((F(v) for k,v in classical.items() if k not in ('setup','engine_simulation')),F(0)))
    result['peak_memory_bytes'] = None  # external peak observation, never a zero placeholder
    result['expected_cost_guarantee'] = 'UNKNOWN; realized campaign charges only'
    return result
