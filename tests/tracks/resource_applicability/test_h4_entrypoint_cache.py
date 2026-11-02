"""Read-only dependency cache policy fixtures, no circuits or compilation."""
import copy,hashlib,os,sys,unittest
from importlib import metadata
from pathlib import Path
from unittest.mock import patch
from trottertracks.resource_applicability.h4_geometry import library_cache as cache,launch_binding as bind
from trottertracks.resource_applicability.h4_geometry.identity import Stop
EVIDENCE=Path(os.environ['H4_ENTRY_CACHE_TEST_EVIDENCE'])

def fixture(name,v2=True):
    root=EVIDENCE/name;root.mkdir();data=b'{"artificial":true}'
    (root/'fontlist.json').write_bytes(data)
    p=dict(schema_version='h4-library-cache-v1',root=str(root),matplotlib_version=metadata.version('matplotlib'),
        files=[dict(file='fontlist.json',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())],bytes=len(data),scientific_cache=False,runtime_writes=False)
    if v2:
        xdg=EVIDENCE/(name+'-xdg');xdg.mkdir();(xdg/'python-entrypoints').mkdir();(xdg/'python-entrypoints/.disable').write_bytes(b'')
        source=Path(metadata.distribution('stevedore').locate_file('stevedore/_cache.py'))
        p.update(schema_version='h4-library-cache-v2',entrypoint_cache=dict(root=str(xdg),stevedore_version=metadata.version('stevedore'),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),sentinel_sha256=hashlib.sha256(b'').hexdigest()))
    return p

class EntryCacheTests(unittest.TestCase):
    def test_legacy_profile_keeps_existing_xdg(self):
        p=fixture('legacy',False)
        with patch.dict(os.environ,{'XDG_CACHE_HOME':'PRESERVED'}):
            cache.configure(p);self.assertEqual(os.environ['XDG_CACHE_HOME'],'PRESERVED')

    def test_v2_routes_own_process_only(self):
        p=fixture('env')
        with patch.dict(os.environ,{},clear=False):
            home=os.environ['HOME'];cache.configure(p)
            self.assertEqual(os.environ['HOME'],home);self.assertEqual(os.environ['XDG_CACHE_HOME'],p['entrypoint_cache']['root'])
            self.assertEqual(os.environ['MPLCONFIGDIR'],p['root'])

    def test_nonempty_disable_is_rejected(self):
        p=fixture('nonempty');(Path(p['entrypoint_cache']['root'])/'python-entrypoints/.disable').write_bytes(b'x')
        with self.assertRaises(Stop):cache.verify(p)

    def test_extra_file_is_rejected(self):
        p=fixture('extra');(Path(p['entrypoint_cache']['root'])/'python-entrypoints/new').write_bytes(b'')
        with self.assertRaises(Stop):cache.verify(p)

    def test_version_and_source_tamper_are_rejected(self):
        p=fixture('version')
        for key in ('stevedore_version','source_sha256'):
            bad=copy.deepcopy(p);bad['entrypoint_cache'][key]='invalid'
            with self.assertRaises(Stop):cache.verify(bad)

    def test_disable_symlink_is_rejected(self):
        p=fixture('link');file=Path(p['entrypoint_cache']['root'])/'python-entrypoints/.disable';file.unlink();file.symlink_to(EVIDENCE/'elsewhere')
        with self.assertRaises(Stop):cache.verify(p)

    def test_already_initialized_cache_fails_closed(self):
        p=fixture('loaded')
        with patch.dict(sys.modules,{'stevedore._cache':object()}),patch.dict(os.environ,{},clear=False):
            with self.assertRaisesRegex(Stop,'precede'):cache.configure(p)

    def test_disk_writes_remain_denied_with_target_diagnostic(self):
        p=fixture('guard');path=Path(p['entrypoint_cache']['root'])/'python-entrypoints'
        plan={'output_root':str(EVIDENCE/'out'),'control_root':str(EVIDENCE/'control')}
        for worker in (True,False):
            with self.assertRaises(Stop):bind.write_guard(plan,'os.mkdir',(path,0o700,-1),worker=worker,library_cache=p)
            with self.assertRaises(Stop):bind.write_guard(plan,'open',(path/'new','w',os.O_WRONLY),worker=worker,library_cache=p)
        with self.assertRaisesRegex(Stop,'os.mkdir .*python-entrypoints'):
            bind.write_guard(plan,'os.mkdir',(path,0o700,-1),worker=True,library_cache=p)
