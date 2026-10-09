"""Fixed, home-local plotting dependency cache; no scientific cache reuse."""
from importlib import metadata
import os
import sys
from pathlib import Path

from .identity import require
from .prelaunch_audit import private_path, streaming_sha

CACHE_CAP = 4 * 2**20
MAX_FILES = 4


def verify(profile):
    version=profile.get('schema_version')
    fields={'schema_version', 'root', 'matplotlib_version', 'files', 'bytes', 'scientific_cache', 'runtime_writes'}
    if version=='h4-library-cache-v2':fields.add('entrypoint_cache')
    require(set(profile)==fields,'library cache profile schema')
    require(version in ('h4-library-cache-v1','h4-library-cache-v2') and
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
    if version=='h4-library-cache-v2':verify_entrypoint_cache(profile['entrypoint_cache'])
    return root


def verify_entrypoint_cache(policy):
    require(type(policy) is dict and set(policy)=={'root','stevedore_version','source_sha256','sentinel_sha256'},
            'entrypoint memory cache policy schema')
    root=private_path(policy['root']);cache=private_path(root/'python-entrypoints')
    require(root.is_dir() and cache.is_dir() and {p.name for p in root.iterdir()}=={'python-entrypoints'} and
            {p.name for p in cache.iterdir()}=={'.disable'},'fixed entrypoint disable inventory')
    require(metadata.version('stevedore')==policy['stevedore_version'],'stevedore version changed')
    source=Path(metadata.distribution('stevedore').locate_file('stevedore/_cache.py'))
    require(streaming_sha(source)['sha256']==policy['source_sha256'],'stevedore disk-disable source changed')
    require(streaming_sha(cache/'.disable')=={'bytes':0,'sha256':policy['sentinel_sha256']} and
            policy['sentinel_sha256']=='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
            'fixed zero-byte entrypoint disable sentinel')
    return root


def configure(profile):
    root = verify(profile)
    # Own process/children only; HOME, user settings and the venv stay untouched.
    if profile['schema_version']=='h4-library-cache-v2':
        require('stevedore._cache' not in sys.modules,'entrypoint policy must precede stevedore cache import')
        os.environ['XDG_CACHE_HOME']=str(verify_entrypoint_cache(profile['entrypoint_cache']))
    os.environ['MPLCONFIGDIR'] = str(root)
    return root


def existing_directory_probe(profile, event, path):
    """Path.mkdir(exist_ok=True) probes an existing directory during import.

    Allow that exact probe, whose mkdir returns EEXIST. All cache file writes,
    removals, renames and new directories remain forbidden at runtime.
    """
    root = Path(profile['root'])
    return event == 'os.mkdir' and path == root and not root.is_symlink() and root.is_dir()
