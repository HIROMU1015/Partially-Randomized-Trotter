"""Ordered paths and complete all-trial means. No RNG, backend, CI or N selector."""
from fractions import Fraction as F
from functools import reduce
from operator import mul
from family import require
from trottertracks.algorithm_codesign.g6_f3_contract import Evidence, fingerprint

def product(values):
    return reduce(mul, values, F(1))

class WholePath:
    def __init__(self, families, *, schedule_id, independence_assumption):
        require(bool(families) and bool(schedule_id) and bool(independence_assumption), 'PATH_CONTRACT')
        self.families = tuple(families)
        self.contexts = tuple(f.context.copy() for f in families)
        self.context = dict(schema='c1_45_path_context_v1', steps=self.contexts,
            schedule_id=schedule_id, independence=independence_assumption,
            independence_verified=False, scope='SYNTHETIC_ONLY')
        self.context_id = fingerprint(self.context)

    def assemble(self, trial_id, records, *, presampled=True, spent_prefix=None):
        require(type(trial_id) is str and trial_id and len(records) == len(self.families), 'PATH_LENGTH_OR_ID')
        require(type(presampled) is bool, 'ARCHITECTURE_REQUIRED')
        require(fingerprint(self.context) == self.context_id, 'PATH_CONTEXT_CHANGED')
        require(spent_prefix is None if presampled else type(spent_prefix) is dict, 'PREFIX_COST_REQUIRED')
        if spent_prefix is not None:
            require(bool(spent_prefix) and all(type(v) is F and v >= 0 for v in spent_prefix.values()),
                    'INVALID_PREFIX_COST')
        data = []
        for index, (family, record, expected) in enumerate(zip(self.families, records, self.contexts)):
            require(family.context == expected, 'CONTEXT_CHANGED')
            d = family.validate(record)
            require(d['trial_id'] == trial_id + ':' + str(index), 'MICROSTEP_ID_OR_ORDER')
            require(d['selection_scope'] is None, 'CONDITIONING_OR_FORMAL_QUERY_FORBIDDEN')
            data.append(d)
        zero = any(d['kind'] == 'ZERO' for d in data)
        return Evidence.create(dict(schema='c1_45_whole_path_v1', context_id=self.context_id,
            trial_id=trial_id, kind='ZERO' if zero else 'EVENT',
            microstep_ids=tuple(r.digest for r in records), microsteps=tuple(data),
            weight=F(0) if zero else product(d['event']['weight'] for d in data),
            proposal= None if zero else product(d['event']['proposal'] for d in data),
            prequantum_zero_avoids_all_quantum_calls=zero and presampled,
            presampled=presampled, spent_prefix=spent_prefix,
            classical_zero_cost='NOT_ZERO_NOT_MEASURED', conditioning='NONE'))

    def validate(self, record):
        data = record.read()
        require(data['schema'] == 'c1_45_whole_path_v1' and data['context_id'] == self.context_id,
                'PATH_EVIDENCE_CONTEXT')
        rebuilt = self.assemble(data['trial_id'], tuple(Evidence.create(d) for d in data['microsteps']),
            presampled=data['presampled'], spent_prefix=data['spent_prefix'])
        require(rebuilt.digest == record.digest, 'PATH_CONTENT_MISMATCH')
        return data

    def circuit(self, record, adapters, *, before=None, after=None, scalar_phase=F(0)):
        from qiskit import QuantumCircuit
        data = self.validate(record)
        require(data['presampled'], 'STREAMED_PREFIX_NEEDS_EXECUTOR_NOT_IMPLEMENTED')
        if data['kind'] == 'ZERO':
            return None
        require(len(adapters) == len(self.families) and type(scalar_phase) is F, 'PROVIDER_OR_SCALAR')
        n = adapters[0].n
        require(all(a.n == n for a in adapters), 'SYSTEM_WIDTH')
        out = QuantumCircuit(n + 1)
        if before is not None:
            out.compose(before, inplace=True)
        for f, a, d in zip(self.families, adapters, data['microsteps']):
            out.compose(f.build(Evidence.create(d), a), inplace=True)
        if after is not None:
            out.compose(after, inplace=True)
        if scalar_phase:
            out.p(-float(scalar_phase), n)
        return out

class AllTrials:
    def __init__(self, path, trial_ids, axis):
        ids = tuple(trial_ids)
        require(bool(ids) and len(set(ids)) == len(ids) and all(type(i) is str and i for i in ids)
                and axis in ('X', 'Y'), 'FIXED_UNIQUE_PLAN_REQUIRED')
        self.path, self.plan, self.axis = path, ids, axis
        self.seen, self.weighted_sum, self.zeros, self.failure = set(), F(0), 0, None
        self.prefix_cost = {}
        self.prepared = {}

    def prepare(self, trial_id, bit_sources):
        """Connect existing samplers before quantum execution; first failure is terminal."""
        require(self.failure is None, 'ACCUMULATOR_TERMINAL')
        try:
            require(trial_id in self.plan and trial_id not in self.seen and trial_id not in self.prepared
                    and len(bit_sources) == len(self.path.families), 'PREPARE_PLAN_OR_LENGTH')
            records = [f.sample(bits, trial_id + ':' + str(i))
                       for i, (f, bits) in enumerate(zip(self.path.families, bit_sources))]
            result = self.path.assemble(trial_id, records)
            self.prepared[trial_id] = result.digest
            return result
        except Exception as exc:
            self.fail(type(exc).__name__ + ':' + str(exc))
            raise

    def fail(self, reason):
        if self.failure is None:
            self.failure = str(reason)

    def add(self, record, observation=None):
        require(self.failure is None, 'ACCUMULATOR_TERMINAL')
        try:
            data = self.path.validate(record)
            tid = data['trial_id']
            require(tid in self.plan and tid not in self.seen, 'DUPLICATE_OR_UNPLANNED_TRIAL')
            require(tid not in self.prepared or self.prepared[tid] == record.digest, 'PREPARED_PATH_CHANGED')
            require(data['conditioning'] == 'NONE', 'ACCEPTED_ONLY_FORBIDDEN')
            if data['kind'] == 'ZERO':
                require(observation is None and data['weight'] == 0, 'ZERO_HAS_NO_OBSERVATION')
                self.zeros += 1
            else:
                require(type(observation) is int and observation in (-1, 1), 'SIGNED_OBSERVATION_REQUIRED')
                self.weighted_sum += observation * data['weight']
            for key, value in (data['spent_prefix'] or {}).items():
                self.prefix_cost[key] = self.prefix_cost.get(key, F(0)) + value
            self.seen.add(tid)
        except Exception as exc:
            self.fail(type(exc).__name__ + ':' + str(exc))
            raise

    def evidence(self):
        complete = self.failure is None and self.seen == set(self.plan)
        return Evidence.create(dict(schema='c1_45_all_trials_v1', context_id=self.path.context_id,
            axis=self.axis, planned=len(self.plan), completed=len(self.seen), valid_zeros=self.zeros,
            weighted_sum=self.weighted_sum, full_trial_mean=self.weighted_sum / len(self.plan) if complete else None,
            complete=complete, failure=self.failure, spent_prefix=self.prefix_cost,
            denominator_policy='ALL_PLANNED_CLASSICAL_TRIALS_ON_COMPLETE_ONLY',
            confidence_certified=False, shots_acquired=0, scientific_scope='SYNTHETIC_ONLY'))
