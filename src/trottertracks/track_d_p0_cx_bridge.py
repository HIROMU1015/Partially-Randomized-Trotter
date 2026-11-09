"""Limited T1 CX-boundary reduction; not an implementation of all P0 passes.

Cancel identical directed CX pairs only when adjacent on both wire frontiers.
Other native operations block cancellation on their wires. Do not commute RZ,
merge 1q runs, lower custom gates, or infer P0 prices outside this rule.
"""
from .track_d_native_resources import NativeCounter, NativeTemplate, count_native


class CXBoundaryBridge:
    """Chunk-independent reduction with an unbounded dependency frontier.

    Input order of surviving gates is retained. Storage can grow with the word;
    an arbitrarily late gate can expose an earlier pair after cancellations.
    Only the reduced native word is NATIVE_EXACT. P0 remains UNQUALIFIED.
    """
    def __init__(self, num_qubits, num_clbits=0, *, global_phase=0.):
        self._input = NativeCounter(num_qubits, num_clbits, global_phase=global_phase)
        self.num_qubits, self.num_clbits = self._input.num_qubits, self._input.num_clbits
        self.global_phase = self._input.global_phase
        self._frontier = [[] for _ in range(self.num_qubits+self.num_clbits)]
        self._nodes = []
        self.cancelled_pairs = 0
        self._live = 0
        self.peak_live_nodes = 0

    def feed(self, operation):
        self._input.feed(operation)  # Reuse T0 schema, bounds and input counting.
        wires = operation.qubits + tuple(self.num_qubits+c for c in operation.clbits)
        if operation.name == "cx":
            a, b = wires
            if self._frontier[a] and self._frontier[b]:
                previous = self._frontier[a][-1]
                candidate = self._nodes[previous]
                if (self._frontier[b][-1] == previous and candidate.name == "cx"
                        and candidate.qubits == operation.qubits):
                    self._frontier[a].pop(); self._frontier[b].pop()
                    self._nodes[previous] = None
                    self._live -= 1
                    self.cancelled_pairs += 1
                    return self
        index = len(self._nodes)
        self._nodes.append(operation)
        for wire in wires:
            self._frontier[wire].append(index)
        self._live += 1
        self.peak_live_nodes = max(self.peak_live_nodes, self._live)
        return self

    def extend(self, operations):
        for operation in operations:
            self.feed(operation)
        return self

    def operations(self):
        """Iterate only after ingestion or while ingestion is otherwise paused."""
        return (node for node in self._nodes if node is not None)

    def result(self):
        return dict(input=self._input.result(),
            output=count_native(self.operations(), self.num_qubits, self.num_clbits,
                                global_phase=self.global_phase),
            cancelled_cx_pairs=self.cancelled_pairs, removed_cx=2*self.cancelled_pairs,
            peak_live_nodes=self.peak_live_nodes,
            rule="identical_directed_CX_adjacent_on_both_wires",
            p0_qualification="P0_UNQUALIFIED", phase_change=0.)

    def template(self):
        return NativeTemplate.from_stream(self.operations(), self.num_qubits, self.num_clbits,
                                          global_phase=self.global_phase)
