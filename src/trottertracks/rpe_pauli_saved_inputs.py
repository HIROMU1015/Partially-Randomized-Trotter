"""Bind the fixed Pauli provider's three saved H4 files on the current host."""
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import subprocess
import sys


PROVIDER_SOURCE = '8478b8200286815ad7ba2652e535d4756ca3c330'
INDEX_SOURCE = '7a8bc7f132e166390c943e6b32c92cc6abe1baf4'
INDEX_PATH = 'artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json'
PATH_MODULES = (
    'df_head_pauli_tail_p0', 'df_head_pauli_tail_ld3',
    'pauli_split_descriptor', 'pauli_candidate_preparation',
)
ENTRY_POINTS = ('prepare_pauli_candidate.py', 'run_pauli_signal_entry.py')


def fixed_blob(workspace, source, path):
    return subprocess.check_output(['git', 'show', f'{source}:{path}'], cwd=workspace)


def provider_files(entry):
    return [f'scripts/{entry}', *(f'src/trottertracks/{name}.py' for name in PATH_MODULES)]


def saved_inputs(workspace, paths):
    index = json.loads(fixed_blob(workspace, INDEX_SOURCE, INDEX_PATH))
    specs = [*index['saved_inputs']['H4']['inputs'].values(), index['T20_reference_files']['H4']]
    expected = {item['path']: item['sha256'] for item in specs}
    if set(paths) != set(expected):
        raise ValueError('Pauli saved-input map must bind exactly the three fixed SOURCE_INDEX paths')
    for original, sha256 in expected.items():
        path = Path(paths[original])
        if not path.is_absolute() or hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            raise ValueError(f'Pauli saved input differs from fixed SOURCE_INDEX: {original}')
    return expected


def configure(entry, paths):
    entry = Path(entry).resolve()
    workspace = entry.parents[1]
    if entry.parent.name != 'scripts' or entry.name not in ENTRY_POINTS:
        raise ValueError('Unknown fixed Pauli entry point')
    files = provider_files(entry.name)
    for relative in files:
        if (workspace / relative).read_bytes() != fixed_blob(workspace, PROVIDER_SOURCE, relative):
            raise ValueError(f'Pauli provider file differs from fixed SOURCE: {relative}')
    saved_inputs(workspace, paths)

    def mapped_path(*parts):
        path = Path(*parts)
        return Path(paths.get(str(path), path))

    sys.path[:0] = [str(workspace / 'src'), str(workspace / 'scripts')]
    for name, relative in zip(PATH_MODULES, files[1:], strict=True):
        module = importlib.import_module(f'trottertracks.{name}')
        if Path(module.__file__).resolve() != workspace / relative:
            raise ValueError(f'Unexpected Pauli provider import: {name}')
        module.Path = mapped_path


def main():
    encoded, entry, *arguments = sys.argv[1:]
    configure(entry, json.loads(encoded))
    entry = Path(entry).resolve()
    sys.argv = [str(entry), *arguments]
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
