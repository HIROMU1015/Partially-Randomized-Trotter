"""Artificial exact-rational omission batches; no fixed scientific kernel."""
from trottertracks import rpe_dp_prefix_reuse as C
from fractions import Fraction


def test_dynamic_residual_batches_preserve_exact_synthetic_group_sums(monkeypatch):
    from types import SimpleNamespace
    from fractions import Fraction
    import multiprocessing as mp
    import pytest
    if 'fork' not in mp.get_all_start_methods():
        pytest.skip('Linux fork fixture')
    monkeypatch.setattr(C,'MIN_DROPS_PER_WORKER',8)
    ordered=[((i%3,i%5),i+1) for i in range(128)]
    def collect(workers):
        residual={'events':0,'groups':{}}
        def add(target,key,mass,coefficient):
            C._grouped_sum(None,target,key,mass,coefficient)
        def drop(flag,phase,mass,completed,target):
            target['events']+=1
            add(target,(flag,phase),mass,Fraction(completed+1,7))
        drop.residual=lambda:residual
        events=[]
        C._drop_many(SimpleNamespace(Mass=Fraction,_accumulate=add),workers,ordered,31,2,drop,events.append)
        # Grouped representation may contain different denominators after
        # partial rational normalization; compare exact values, not layout.
        return residual['events'],{k:sum(Fraction(n,d) for d,n in terms.items())
                                   for k,terms in residual['groups'].items()},events
    serial=collect(1)
    parallel=collect(3)
    assert serial[:2]==parallel[:2]
    assert parallel[2][0]['worker_max_seconds']<=parallel[2][0]['worker_sum_seconds']


def test_persistent_residual_pool_reuses_workers_and_preserves_exact_batches(monkeypatch):
    from types import SimpleNamespace
    import multiprocessing as mp
    import pytest
    if 'fork' not in mp.get_all_start_methods():pytest.skip('Linux fork fixture')
    monkeypatch.setattr(C,'MIN_DROPS_PER_WORKER',8)
    def drop(flag,phase,mass,completed,target):
        target['events']+=1
        C._grouped_sum(None,target,(flag,phase),mass,Fraction(completed+1,7))
    def add(target,key,mass,coefficient):C._grouped_sum(None,target,key,mass,coefficient)
    kernel=SimpleNamespace(Mass=Fraction,_accumulate=add)
    pool=C.ResidualPool();events=[]
    def calculate(workers,ordered,den,completed,persistent):
        residual={'events':0,'groups':{}}
        drop.residual=lambda:residual
        C._drop_many(kernel,workers,ordered,den,completed,drop,events.append,residual_pool=persistent)
        return residual['events'],{k:sum(Fraction(n,d) for d,n in terms.items()) for k,terms in residual['groups'].items()}
    try:
        for i in range(3):
            ordered=[((j%3,j%5),j+i+1) for j in range(128)]
            expected=calculate(1,ordered,31+i,i,None)
            actual=calculate(3,ordered,31+i,i,pool)
            assert actual==expected
            assert events[-1]['persistent_pool']
            assert events[-1]['pool_reused']==bool(i)
        assert pool.pool is not None
        # A changing mutable closure must never enter the retained pool.
        mutable=[]
        def captures(*args):mutable.append(args)
        assert pool.prepare(captures,Fraction,2)==(False,False)
    finally:pool.close()
    assert pool.pool is None
