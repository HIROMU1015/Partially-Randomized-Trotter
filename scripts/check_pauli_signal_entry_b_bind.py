"""Run B's own Pauli intake ``bind_packet`` on an entry MEAN_HANDOFF (metadata only, no statistics).

B's ``accept_saved`` admits only the fixed LD1/LD2, LD3 and LD5 assemblies; ``bind_packet`` is its
generic part. This script calls it exactly as ``accept_saved`` does for a later prefix: records from
each row's mean_source (Git), dictionary from the descriptor source, LD2's saved target view, and B's
pinned A helpers (``_ld3_metadata_helpers``: P1 ``job`` + 8c8ce542e ``b_intake_dictionary_identity``).

    git archive <B commit> src | tar -x -C <dir>        # e.g. b6d4dea49 (intake of LD3/LD5)
    python scripts/check_pauli_signal_entry_b_bind.py --b-root <dir> --handoff COMMIT:PATH|FILE

Only B's package is imported (from --b-root); this repository is used for Git reads.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
DF = '91349e703e0d09ef702ccd108745dfaf1acc29f7'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--b-root', required=True); parser.add_argument('--handoff', required=True)
    parser.add_argument('--records-from-rows', action='store_true',
                        help='synthetic local packets only: take each record from its row instead of the Git carrier')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.b_root)/'src'))
    from trottertracks import rpe_pauli_signal_intake as I
    store = I.p.SavedArtifacts(str(REPO))
    if Path(args.handoff).is_file():                     # pre-commit check of a local assembly
        assembly = dict(local_file=str(Path(args.handoff).resolve()), committed=False)
        handoff = json.loads(Path(args.handoff).read_text())
    else:
        commit, path = args.handoff.split(':', 1)
        commit = subprocess.check_output(['git', 'rev-parse', commit], cwd=REPO).decode().strip()
        assembly = dict(commit=commit, path=path); handoff = store.read(assembly)
    helpers = I._ld3_metadata_helpers(str(REPO)); out = []
    for packet in handoff['packets']:
        config = packet['configuration']
        if handoff.get('configuration_source') and not args.records_from_rows:
            source = handoff['configuration_source']; value = store.read(source)
            for key in source.get('selector', []):
                value = value[key]
            I._same(config, value, 'configuration SOURCE')
        target_source = dict(commit=DF, path='docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD2.json', selector=['target'])
        target = store.read(target_source)['target']; records = {}
        for row in packet['rounds']:
            if args.records_from_rows:
                records[row['m']] = row['mean_record']; continue
            rec = store.read(row['mean_source'])
            for key in row['mean_source']['selector']:
                rec = rec[key]
            records[row['m']] = rec
        result = I.bind_packet(config, packet, records, store.read(packet['dictionary_identity']['source']), target=target,
                               target_source=target_source, assembly_source=assembly, helpers=helpers)
        view = result['signal_packet']
        out.append(dict(prefix=packet['prefix'], configuration_id=config['id'], means=len(view['means']),
                        binding_status=view['binding_status'], readiness=sorted(result['readiness'])[:8],
                        implementation_sources=sorted({json.dumps(v, sort_keys=True) for v in view['implementation_sources'].values()})))
    print(json.dumps(dict(status='B_BIND_PACKET_PASS', handoff=assembly, records_from_rows=args.records_from_rows, packets=out, B_module=I.__file__), indent=2))


if __name__ == '__main__':
    main()
