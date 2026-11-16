"""A's on-demand PR-Pauli preparation boundary for B2 (metadata by default).

    --request B2_REQUEST.json --output PREPARATION.json [--cache SPLITS.json]

Only --prepare-missing calls the scientific P0 leaf, after separate user
execution permission. It creates one requested split, commits only its fresh
products to supply immutable P0 locators, then continues metadata generation.
It does not start means, prices, compile, Q3, or choose a split/kappa/shot.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from trottertracks import pauli_candidate_preparation as A
from trottertracks import pauli_split_descriptor as D


def materializer(root, directory, source):
    """Future scientific leaf. An already committed split is reused by Assets first.

    Scientific SOURCE is fixed before entry. Only fresh, scoped P0 products are
    committed; the old context, dictionaries and execution evidence stay intact.
    No automatic retry, threshold change or fallback.
    """
    root, directory = Path(root).resolve(), Path(directory).resolve()
    directory.relative_to(root)
    def create(prefix, descriptor):
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip() != source:
            raise ValueError('FIXED_CURRENT_SCIENTIFIC_SOURCE_REQUIRED')
        import run_df_head_pauli_tail_split_stage1 as stage1
        out = directory / ('LD%d' % prefix)
        if out.exists():
            return dict(status='STOP_EXISTING_P0_OUTPUT_NO_RETRY', missing_inputs=[str(out)])
        meta = A.E.p0.metadata(root)
        d = descriptor['descriptor']
        if d is None:
            # Load saved factors only; no SCF/DF/state generation or circuit builder.
            ham, _, _, _, _ = A.E.p0.load_saved(root, meta)
            d = D.derive_descriptor(root, ham, prefix)
        origin = dict(**D.DERIVATION_SOURCE, scientific_source=source,
                      snapshot=meta['binding']['snapshot'], prefix=prefix)
        with D.stage1_metadata(root, prefix, d, descriptor.get('source') if descriptor['descriptor'] is not None else origin):
            result = stage1.execute_p0(out, source, prefix)
        if result['status'] != stage1.success(prefix):
            return dict(status='STOP_P0_FAILED', P0_output=str(out), reason=result.get('error'), missing_inputs=[])
        descriptor_path = out/'DESCRIPTOR.json'
        descriptor_path.write_text(json.dumps(dict(prefix=prefix, binding=meta['binding'], descriptor=d,
                                                   source=descriptor.get('source') or origin), indent=2)+'\n')
        relative = str(out.relative_to(root))
        subprocess.run(['git', 'add', '--', relative], cwd=root, check=True)
        subprocess.run(['git', 'commit', '--only', '-m', 'Save requested PR-Pauli LD%d P0 products [skip ci]' % prefix,
                        '--', relative], cwd=root, check=True, stdout=sys.stderr)
        carrier = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip()
        descriptor_source = dict(commit=carrier, path=relative+'/DESCRIPTOR.json', selector=['descriptor'])
        dictionary = A.E.L.read(root, carrier, relative+'/LD%d_DICTIONARY.json' % prefix)
        row = next(r for r in result['prefixes'] if r['prefix'] == prefix)
        item = dict(payload=dictionary, dictionary_sha256=row['dictionary_sha256'],
                    source=dict(commit=carrier, path=relative+'/LD%d_DICTIONARY.json' % prefix),
                    P0_metrics=row['identity_metrics'], P0_result_source=dict(commit=carrier, path=relative+'/result.json'))
        return dict(status='SPLIT_READY', item=item,
                    descriptor=dict(status='COMMITTED_DERIVED_OR_SAVED', descriptor=d, source=descriptor_source),
                    scientific_source=source, validation_scope='P0_OPERATOR_GATES_NOT_SIGNAL_OR_PRICE')
    return create


def main(argv=None, *, assets_factory=A.Assets, materializer_factory=materializer):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--prepare-missing', action='store_true')
    parser.add_argument('--source')
    parser.add_argument('--p0-directory', type=Path)
    args = parser.parse_args(argv)
    if args.prepare_missing and not (args.source and args.p0_directory and args.cache):
        parser.error('--prepare-missing requires --source --p0-directory --cache')
    request = json.loads(args.request.read_text())
    try:
        assets = assets_factory(ROOT, cache=args.cache, packets=request.get('saved_packets', ()),
                                configurations=request.get('saved_configurations', ()))
    except (KeyError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        result = dict(schema='A_PAULI_CANDIDATE_PREPARATION_V1', status='NEEDS_INPUT_SOURCE',
                      missing_inputs=[str(exc)], items=[], signal_price_Q3_evaluation_performed=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(status=result['status'], missing_inputs=result['missing_inputs'])))
        return result
    if args.prepare_missing and len(request.get('items', [request])) != 1:
        parser.error('--prepare-missing prepares one B2-selected candidate per request')
    create = materializer_factory(ROOT, args.p0_directory, args.source) if args.prepare_missing else None
    result = A.prepare(request, assets, materialize=create)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Complete configurations are directly consumable files even when price evidence is absent.
    for i, item in enumerate(result['items']):
        config = item.get('configuration')
        if config is None:
            continue
        path = args.output.with_name(args.output.stem+'.%d.CONFIGURATION.json' % i)
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('DO_NOT_OVERWRITE_DIFFERENT_CONFIGURATION:'+str(path))
        path.write_text(json.dumps(config, indent=2, allow_nan=False)+'\n')
        assets.remember_configuration(config, path)
        item['configuration_file'] = str(path.resolve())
        packet_file = args.output.with_name(args.output.stem+'.SAVED_PACKETS.json')
        packet_file.write_text(json.dumps(list(assets.packets), indent=2)+'\n')
        command = [sys.executable, '-B', 'scripts/run_pauli_signal_entry.py', '--plan',
                   '--config', str(path.resolve()), '--saved-packets', str(packet_file.resolve())]
        if str(item['L_D']) not in assets.context['prefixes']:
            desc_file = args.output.with_name(args.output.stem+'.%d.DESCRIPTOR.json' % i)
            desc_file.write_text(json.dumps(dict(prefix=item['L_D'], binding=assets.binding,
                descriptor=item['descriptor']['descriptor'], source=item['descriptor']['source']), indent=2)+'\n')
            command += ['--tail-descriptor', str(desc_file.resolve())]
        item['signal_plan_command'] = command
    result['completed_UTC'] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({r['item_id']: r['status'] for r in result['items']}))
    return result


if __name__ == '__main__':
    main()
