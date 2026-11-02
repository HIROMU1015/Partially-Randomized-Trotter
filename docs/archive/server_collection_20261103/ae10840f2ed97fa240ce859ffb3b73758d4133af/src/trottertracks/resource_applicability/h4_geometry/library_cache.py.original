"""Fixed, home-local plotting dependency cache; no scientific cache reuse."""
from importlib import metadata
import os
from pathlib import Path

from .identity import require
from .prelaunch_audit import private_path, streaming_sha

CACHE_CAP = 4 * 2**20
MAX_FILES = 4


def verify(profile):
    require(set(profile) == {'schema_version', 'root', 'matplotlib_version', 'files',
                             'bytes', 'scientific_cache', 'runtime_writes'}, 'library cache profile schema')
    require(profile['schema_version'] == 'h4-library-cache-v1' and
            profile['scientific_cache'] is False and profile['runtime_writes'] is False,
            'read-only library cache only')
    root = private_path(profile['root'])
    require(root.is_dir(), 'library cache directory missing')
    require(metadata.version('matplotlib') == profile['matplotlib_version'], 'library cache version changed')
    files = profile['files']
    require(type(files) is list and 1 <= len(files) <= MAX_FILES, 'library cache file count')
    names = [row['file'] for row in files]
    require(len(set(names)) == len(names) and set(names) == {p.name for p in root.iterdir()},
            'library cache inventory changed')
    total = 0
    for row in files:
        require(set(row) == {'file', 'bytes', 'sha256'} and type(row['file']) is str and
                Path(row['file']).name == row['file'] and row['file'] not in ('.', '..'),
                'library cache basename')
        actual = streaming_sha(root / row['file'])
        require(actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256'], 'library cache bytes/hash changed')
        total += actual['bytes']
    require(total == profile['bytes'] and 0 < total <= CACHE_CAP, 'library cache byte cap')
    return root


def configure(profile):
    root = verify(profile)
    # Own process/children only. Do not change HOME, XDG settings, or the venv.
    os.environ['MPLCONFIGDIR'] = str(root)
    return root


def existing_directory_probe(profile, event, path):
    """Path.mkdir(exist_ok=True) probes an existing directory during import.

    Allow that exact probe, whose mkdir returns EEXIST. All cache file writes,
    removals, renames and new directories remain forbidden at runtime.
    """
    root = Path(profile['root'])
    return event == 'os.mkdir' and path == root and not root.is_symlink() and root.is_dir()
