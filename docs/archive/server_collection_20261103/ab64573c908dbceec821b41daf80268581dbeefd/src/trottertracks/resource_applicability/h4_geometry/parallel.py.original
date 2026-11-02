"""Bounded driver-side compile scheduling; no worker or science launch here."""
from .identity import require
from .ledger import validate_reuse_identity


def compile_wrappers(run, ledger, jobs, compile_worker, options, reuse, *, expected_count):
    """Consume lazy (expected identity, circuit) jobs and return logical order.

    At most the admitted worker count has an outstanding invocation. Pending
    duplicate owners retain only logical identities, never another circuit or
    reservation. Only the driver publishes completions and cache links.
    """
    pending, in_flight, followers, records = {}, {}, {}, []
    require(type(run.workers) is int and 1 <= run.workers <= 12, 'admitted compile workers')
    iterator = iter(jobs)

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
            records[position] = record
            reuse[scope] = key
            del in_flight[scope]
            for follower_position, follower_key in followers.pop(key, []):
                records[follower_position] = ledger.complete(follower_key, {}, owner_key=key)
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
                records[position] = ledger.complete(key, {}, owner_key=owner)
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
