"""Reuse the fixed DF provider's deterministic S2 flanks within one price task."""
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from unittest.mock import patch


class Flanks:
    def __init__(self, native):
        self.native = native
        self.original = native.build_partial_native
        self.pieces = {}

    def templates(self, blocks, n, signed_time):
        key = id(blocks), n, signed_time.hex()
        if key not in self.pieces:
            native = self.native
            forward, reverse = native.QuantumCircuit(n + 1), native.QuantumCircuit(n + 1)
            half = signed_time / 2
            for block in blocks:
                native._append_block(forward, block, half, 'UNCONTROLLED', n)
            for block in reversed(blocks):
                native._append_block(reverse, block, half, 'DIRECTIONAL', n)
            # Keep a reference to the block tuple for the lifetime of its ID.
            self.pieces[key] = blocks, forward, reverse
        return self.pieces[key][1:]

    def build(self, blocks, *, num_system_qubits, T, q, scalar, controlled_tails,
              control_policy, max_instructions):
        if (type(blocks) is not tuple or type(q) is not int or q != 1
                or control_policy != 'symmetric_directional' or controlled_tails is None):
            return self.original(blocks, num_system_qubits=num_system_qubits,
                                 T=T, q=q, scalar=scalar, controlled_tails=controlled_tails,
                                 control_policy=control_policy, max_instructions=max_instructions)
        native = self.native
        n = native.validate_blocks(blocks, num_system_qubits)
        T, scalar = native._real(T, 'T'), native._real(scalar, 'scalar')
        cap = native._limit(max_instructions)
        if len(controlled_tails) != 1:
            raise ValueError('One explicit controlled tail per outer step is required.')
        tail = controlled_tails[0]
        native._gate_only(tail, n + 1)
        bound = sum(native.block_instruction_bound(b, mode)
                    for b in blocks for mode in ('UNCONTROLLED', 'DIRECTIONAL')) + len(tail.data) + 1
        if bound > cap:
            raise RuntimeError('PRE_BUILD_INSTRUCTION_BUDGET')
        forward, reverse = self.templates(blocks, n, T)
        circuit = forward.copy(name='ax2a_df_partial_s2')
        circuit.compose(tail, inplace=True)
        circuit.compose(reverse, inplace=True)
        if scalar*T != 0:
            circuit.p(-scalar*T, n)
        return native._finish(circuit, n, T, q, 'partial_2nd', control_policy, bound, cap)


@contextmanager
def reuse_df_price_events(rte):
    """Reuse immutable events after each original sampling decision."""
    make = rte._make_event
    active_components = active_distribution = None

    @lru_cache(maxsize=4096)
    def event(indices, order):
        return make(indices, active_components, active_distribution, order)

    def cached(indices, components, distribution, order):
        nonlocal active_components, active_distribution
        if components is not active_components or distribution is not active_distribution:
            event.cache_clear()
            active_components, active_distribution = components, distribution
        return event(tuple(map(int, indices)), order)

    with patch.object(rte, '_make_event', cached):
        yield


@contextmanager
def reuse_df_price_flanks(runner):
    """Warm templates after the existing input checks and before the workers fork."""
    load_runtime = runner.runtime
    with ExitStack() as stack:
        patched = set()
        caches = {}
        warmed = {}
        event_modules = set()

        def runtime():
            provider = load_runtime()
            from trotterlib import rte
            if rte not in event_modules:
                stack.enter_context(reuse_df_price_events(rte))
                event_modules.add(rte)
            runtime_type = provider.D.NativeRuntime
            if runtime_type in patched:
                return provider
            original_context = runtime_type.context

            def context(instance, request):
                result = original_context(instance, request)
                if id(result) in warmed:
                    return result
                native = result.native
                if native not in caches:
                    cache = caches[native] = Flanks(native)
                    stack.enter_context(patch.object(native, 'build_partial_native', cache.build))
                cache = caches[native]
                candidate = request['physical_context']['physical_candidate']
                blocks = result.prep.deterministic_blocks
                if type(blocks) is tuple:
                    steps = result.h['core'].schedule(candidate['formula'], candidate['q'],
                        'ABS_TIME_PER_OUTER', candidate['R_outer'], T=candidate['T'])
                    times = {native._real(step['signed_time'], 'T').hex() for step in steps}
                    for signed_time in sorted(times):
                        cache.templates(blocks, result.prep.num_system_qubits, float.fromhex(signed_time))
                warmed[id(result)] = result
                return result

            stack.enter_context(patch.object(runtime_type, 'context', context))
            patched.add(runtime_type)
            return provider

        stack.enter_context(patch.object(runner, 'runtime', runtime))
        yield
