"""Bounded driver-side compile scheduling; no worker or science launch here."""
from .identity import require
from .ledger import validate_reuse_identity


def compile_wrappers(run, ledger, jobs, compile_worker, options, reuse, *, expected_count, on_complete=None):
    """Consume lazy (expected identity, circuit) jobs and return logical order.

    At most the admitted worker count has an outstanding invocation. Pending
    duplicate owners retain only logical identities, never another circuit or
    reservation. Only the driver publishes completions and cache links.
    """
    pending, in_flight, followers, records = {}, {}, {}, []
    require(type(run.workers) is int and 1 <= run.workers <= 12, 'admitted compile workers')
    iterator = iter(jobs)

    def record_ready(position, record):
        records[position] = record
        if on_complete is not None:
            on_complete(position, record)

    def healthy():
        run.pulse()
        # Inspect every observed failure before generating/submitting more work,
        # including a failure outside the batch returned by wait_any.
        for future in pending:
            if future.done():
                error = future.exception()
                if error is not None:
                    raise error

    def collect():
        healthy()
        done = run.wait_any(tuple(pending))
        healthy()
        require(bool(done) and set(done) <= set(pending), 'owned completion batch')
        # Tie ordering is explicit; aggregation always uses logical positions.
        for future in sorted(done, key=lambda f: pending[f][0]):
            position, key, scope = pending[future]
            metrics = future.result()
            record = ledger.complete(key, metrics)
            ledger.read(key)  # independently check the durable COMPLETE owner
            record_ready(position, record)
            reuse[scope] = key
            del in_flight[scope]
            for follower_position, follower_key in followers.pop(key, []):
                record_ready(follower_position, ledger.complete(follower_key, {}, owner_key=key))
            del pending[future]

    try:
        while True:
            healthy()
            if len(pending) >= run.workers:
                collect()
                continue
            try:
                expected, circuit = next(iterator)
            except StopIteration:
                break
            healthy()
            require(len(records) < expected_count, 'logical wrapper count before registration')
            position, key = len(records), expected['wrapper_key']
            records.append(None)
            ledger.register(expected)
            scope = (expected['geometry'], expected['candidate_template'], expected['axis'],
                     expected['numerical_circuit_fingerprint'])
            owner = reuse.get(scope)
            if owner is not None:
                record_ready(position, ledger.complete(key, {}, owner_key=owner))
            elif scope in in_flight:
                owner = in_flight[scope]
                # Tracking a pending dependency is not RESERVED cache reuse.
                validate_reuse_identity(ledger.expected[owner], expected)
                followers.setdefault(owner, []).append((position, key))
            else:
                ledger.reserve(key)  # durable and charged before pool submission
                healthy()
                future = run.pool.submit(compile_worker, circuit, options)
                require(future not in pending, 'distinct compile future')
                pending[future] = (position, key, scope)
                in_flight[scope] = key
            del circuit
        while pending:
            collect()
        healthy()
        require(len(records) == expected_count and all(r is not None for r in records)
                and not in_flight and not followers, 'all logical wrappers completed')
        return records
    except BaseException:
        run.abort()  # owned children only; charged reservations are never refunded
        raise


def compile_candidates(run, ledger, candidates, compile_worker, options, reuse, *,
                       expected_candidates, expected_wrappers, on_candidate):
    """Fill one bounded compile queue across candidate boundaries.

    Candidate/trajectory/axis generation order is unchanged. Completion callbacks
    publish candidates in that same order, even when workers finish out of order.
    No circuit is retained by candidate bookkeeping.
    """
    contexts, positions = {}, {}
    generated = candidate_count = published = 0

    def jobs():
        nonlocal generated, candidate_count
        for metadata, iterator, count in candidates:
            require(candidate_count < expected_candidates and type(count) is int and count > 0,
                    'candidate count before registration')
            ordinal = candidate_count
            candidate_count += 1
            contexts[ordinal] = {'metadata':metadata, 'records':[None]*count, 'remaining':count}
            actual = 0
            for expected, circuit in iterator:
                require(actual < count, 'candidate wrapper overflow')
                positions[generated] = (ordinal,actual)
                generated += 1
                actual += 1
                yield expected,circuit
                del circuit
            require(actual == count, 'candidate wrapper underflow')
        require(candidate_count == expected_candidates, 'complete candidate stream')

    def ready(position, record):
        nonlocal published
        require(position in positions, 'candidate completion position')
        ordinal,index = positions.pop(position)
        context = contexts[ordinal]
        require(context['records'][index] is None, 'duplicate candidate completion')
        context['records'][index] = record
        context['remaining'] -= 1
        while published in contexts and contexts[published]['remaining'] == 0:
            context = contexts.pop(published)
            require(all(r is not None for r in context['records']), 'complete candidate metrics')
            on_candidate(context['metadata'],context['records'])
            published += 1

    try:
        compile_wrappers(run,ledger,jobs(),compile_worker,options,reuse,
                         expected_count=expected_wrappers,on_complete=ready)
        require(published == expected_candidates and not contexts and not positions,
                'all candidates published in logical order')
        return published
    except BaseException:
        run.abort()
        raise
