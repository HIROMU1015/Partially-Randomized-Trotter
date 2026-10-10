# G10 S3: payload types and dictionary-key audit

This audit reads frozen source and saved technical JSON. It does not regenerate
a live registered G10 payload, events, matrices, samples or synthesis. The v2
bounded traceback did not retain the exact first offending key/path.

The retained non-string keys confirmed by source are **integer provider labels**
in `rows[new m3/m7].events[*].event.provider_calls`. Other inspected retained
dictionaries use string literals, `str(ratio)`, `str(m)`, package/path names or
already decoded JSON keys. Intermediate algebra dictionaries are distinguished
below; they are not sent directly to the result encoder.

The machine audit is
`artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3/payload_type_and_key_audit_v3.json`.
It records 21 source areas, function/line anchors and 68 dictionary AST nodes.
These are static source observations, not a universal proof about arbitrary
Python objects or a trace of a new scientific run.

| Payload area | Producer / retained key and value shape |
| --- | --- |
| Root, provenance, counters | v3 `execute`: string field names; strings/bools/ints and dictionaries |
| Native events | `g7_generator._event`: string outer fields; Fraction scalars, label/phase ints, tuple words |
| `provider_calls` | `_event` constructs `dict(Counter(reduced))`, increments `calls[child]`; **integer label keys**, integer counts |
| Closed P5 / closed P5 tail events | `g9_p5.P5Closed.event` delegates to `_event`; `g10_generator.ClosedP5Tail.event` changes Fraction proposal/weight, retains key shape |
| CTS events | `g10_reference.cts_events`: string fields; Pauli names, phase/sign ints, Fraction coefficients/proposals |
| Native IR | `g9_native.native_ir`: lists/tuples of op, wire indices, ratio strings, signs; integers are values, not dictionary keys |
| Native cost | `g9_native.cost`: string names T/CX/1Q/strict error; int/Fraction values |
| Synthesis cache | outer keys are `str(ratio)`; `numeric.synthesize` records literal string fields, sequence/hash/epsilon strings, int counts, bool/error fields |
| Per-key/total resource | guard receipts use string names and Python int/float values |
| Inventory | string field names and ratio/hash/acquisition strings, bool guards, Python float timings |
| Budget/confidence | `finish_plan`: string field names; Fraction/int/bool/string values |
| Row/event binding dictionaries | `row`: string fields; nested native event includes integer-key `provider_calls` |
| CTS certificates | outer `str(m)`; string fields; exact algebra target converted to `axis+':'+str(phase)` and paired coordinate strings |
| Interface traces | string fields; queried tangents stored as sorted string lists; integer counters and Python float timings |
| Classical stage accounting | string literal fields plus keyword context names; str/list/int/Python float values |
| Saved m5 anchor | JSON decoding already gives string keys; unchanged deepcopy; rebudget adds string-key Fraction/int fields |
| Fixed-policy lower | `affine_policy_lower`: string fields with Fraction/int/bool values; no recomputation here |
| Runtime/protected histories | metadata dictionaries use string package/path/field names; no module object retained |
| Matrix diagnostics | `row` and `circuit_error` explicitly return Python floats; NumPy arrays stay local |
| Exact Pauli algebra | temporary tuple `(axis,phase)` keys and `A` objects; certificate converts keys and calls `A.json()` before retaining data |
| Parent/group/DP/kernel objects | temporary dataclasses/intervals/tuple-key caches; generators export scalar/tuple event fields, not those objects |

`FullReturnGenerator.event`, canonical event constructors and the closed-P5
constructor share `_event`. Child/word labels originate from integer ranges or
dyadic-index choices; this audit invokes none of them. CTS uses its separate
literal string-key event shape.

Saved JSON alone cannot verify a live typed payload's key shape: old
`g10_saved.serial` had already converted integer keys. The synthetic fixtures
therefore explicitly contain integer keys, mixed key types and collisions.

## Output-boundary contract

The unchanged legacy reference normalizes a dictionary as
`{str(k): serial(v) for k,v in value.items()}`. S3 makes only a **shallow local
dictionary** with `str(k)` keys and original child references, then walks retained
children to emit JSON tokens. It never changes event fields or constructs an
entire recursively normalized result.

Python dict assignment preserves the first normalized key's insertion position
and the last value. For `{1:'first', 'middle':0, '1':'last'}`, output order is
`['1','middle']` and `'1'` stores `'last'`. Both collision directions, multiple
collisions, reversed input order and shared subtrees are tested against the
actual unchanged `g10_saved.serial` function.

Value validation still examines **all original values**, including values later
overwritten by a key collision. Nonfinite/unsupported/cyclic original values are
rejected. A legacy conversion could hide a NaN/unsupported value by overwriting
it before JSON validation; that general-object corner is deliberately outside
S3's admitted value domain, preserving S2's rejection discipline. No such value
is an intended scientific output. Nonfinite numeric *keys* stringify to ordinary
strings under the legacy rule; they are separately tested from nonfinite values.

Exact equivalence assumes stable, side-effect-free `str(key)` and no concurrent
input mutation. Pure custom keys are tested, but side-effecting `__str__`, custom
container iteration with side effects and arbitrary object semantics are not
claimed. Conversion exceptions propagate into the existing failure protocol.
The intended schema has plain int/string keys and ordinary containers.

Memory is proportional to the widths of active dictionary maps plus recursion
depth and the bounded write buffer. Wide individual mappings, pathological
nesting and one huge escaped string token are not universally bounded; the fixed
RSS/AS guard remains active. This limitation is distinct from the removed whole
recursive result copy.
