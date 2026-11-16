"""B's normal intake of published PR-Pauli MEAN_HANDOFF packets (accept_saved + total binding), for the Q3 route table.

Runs B's fixed view (SOURCE checked by B's own check_source) with the Q3 route interpreter; no mean, price or Q3.

    python b_intake_pauli_packets.py --view VIEW --packets PACKETS.json --output INTAKE.json
PACKETS.json: [{"name", "commit", "path"}]
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

getattr(sys, 'set_int_max_str_digits', lambda n: None)(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--view', type=Path, required=True);ap.add_argument('--packets', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args();view = a.view
    sys.path[:0] = [str(view / 'scripts'), str(view / 'src')]
    spec = importlib.util.spec_from_file_location('_x', view / 'scripts/run_pauli_ld5_two_vectors_q3_fixed.py')
    X = importlib.util.module_from_spec(spec);sys.modules['_x'] = X;spec.loader.exec_module(X)
    X.check_source();A, T = X.A, X.T
    out = {}
    for p in json.loads(a.packets.read_text()):
        t = time.time();loc = dict(commit=p['commit'], path=p['path'])
        try:
            acc = A.accept_saved(view, integration=T.C.interfaces(str(view)).pipeline, handoff_source=loc)
            res = acc['results'][0];handoff = A.p.SavedArtifacts(view).read(loc)
            cfg_src = handoff.get('configuration_source');original = T._selected(view, cfg_src)
            if original != res['configuration']:
                raise ValueError('CONFIGURATION_DIFFERS_FROM_PACKET')
            binding = T.create(view, original, original_source=cfg_src);bound = T.bind_signal(view, binding, res['signal_packet'])
            out[p['name']] = dict(status=acc['status'], configuration_id=original['id'], configuration_source=cfg_src, signal_packet=loc,
                                  binding_id=binding['id'], total_configuration_id=binding['total_configuration']['id'],
                                  means=len(bound['signal_input']['means']), science_calls=acc['science_calls'],
                                  seconds=round(time.time() - t, 1),
                                  signal_input_digest=hashlib.sha256(json.dumps(bound['signal_input']['means'], sort_keys=True,
                                                                                default=str).encode()).hexdigest()[:16])
        except Exception as exc:  # recorded, never retried here
            out[p['name']] = dict(status='INTAKE_FAILED', error=type(exc).__name__ + ': ' + str(exc)[:500], signal_packet=loc)
        print(p['name'], out[p['name']]['status'], flush=True)
    a.output.write_text(json.dumps(out, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
