"""Cache routing and cumulative no-refund gates, entirely artificial."""
import copy
import hashlib
from importlib import metadata
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from trottertracks.resource_applicability.h4_geometry import library_cache as cache, launch_binding as bind, prelaunch_audit as audit
from trottertracks.resource_applicability.h4_geometry.identity import Stop

EVIDENCE=Path(os.environ['H4_PRELAUNCH_TEST_EVIDENCE'])


def fixture(name):
    root=EVIDENCE/name;root.mkdir()
    data=b'{"artificial":true}'
    (root/'fontlist.json').write_bytes(data)
    return {'schema_version':'h4-library-cache-v1','root':str(root),'matplotlib_version':metadata.version('matplotlib'),
            'files':[{'file':'fontlist.json','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}],
            'bytes':len(data),'scientific_cache':False,'runtime_writes':False}


class LibraryCacheTests(unittest.TestCase):
    def test_private_env_routing_preserves_home(self):
        profile=fixture('cache-env')
        with patch.dict(os.environ,{},clear=False):
            before=os.environ['HOME'];cache.configure(profile)
            self.assertEqual(os.environ['HOME'],before)
            self.assertEqual(os.environ['MPLCONFIGDIR'],profile['root'])

    def test_changed_bytes_fail_closed(self):
        profile=fixture('cache-changed');(Path(profile['root'])/'fontlist.json').write_bytes(b'changed')
        with self.assertRaises(Stop):cache.verify(profile)

    def test_extra_file_and_symlink_fail_closed(self):
        profile=fixture('cache-extra');(Path(profile['root'])/'extra').write_bytes(b'x')
        with self.assertRaises(Stop):cache.verify(profile)
        profile=fixture('cache-link');p=Path(profile['root'])/'fontlist.json';p.unlink();p.symlink_to(EVIDENCE/'cache-env/fontlist.json')
        with self.assertRaises(Stop):cache.verify(profile)

    def test_version_mismatch_fail_closed(self):
        profile=fixture('cache-version');profile['matplotlib_version']='invalid'
        with self.assertRaises(Stop):cache.verify(profile)

    def test_existing_directory_probe_but_no_cache_writes_for_either_role(self):
        profile=fixture('cache-guard');root=profile['root']
        plan={'output_root':str(EVIDENCE/'output'),'control_root':str(EVIDENCE/'control')}
        for worker in (False,True):
            bind.write_guard(plan,'os.mkdir',(root,0o700,-1),worker=worker,library_cache=profile)
            bind.write_guard(plan,'open',(root+'/fontlist.json','r',os.O_RDONLY),worker=worker,library_cache=profile)
            for event,args in [('open',(root+'/fontlist.json','w',os.O_WRONLY)),('os.mkdir',(root+'/new',0o700,-1)),
                               ('os.remove',(root+'/fontlist.json',-1)),('os.rename',(root+'/fontlist.json',root+'/renamed',-1,-1))]:
                with self.assertRaises(Stop):bind.write_guard(plan,event,args,worker=worker,library_cache=profile)

    def test_missing_directory_probe_cannot_create(self):
        profile=fixture('cache-missing');p=Path(profile['root']);(p/'fontlist.json').unlink();p.rmdir()
        self.assertFalse(cache.existing_directory_probe(profile,'os.mkdir',p))

    def test_carry_and_observer_reservations_exceed_ten_GiB_without_refund(self):
        self.assertEqual(bind.PRIOR_CARRY,{'actual_invocations':20,'charged_bytes':4428938712,'wall_seconds':5472.345380863175})
        self.assertEqual(bind.CARRY['actual_invocations'],21)
        projection=audit.storage_projection(library_cache_bytes=1234)
        original=audit.storage_projection(prior_charge=bind.ORIGINAL_CARRY['charged_bytes'])
        self.assertEqual(projection['cumulative_charge_bound']-original['cumulative_charge_bound'],
                         bind.CARRY['charged_bytes']-bind.ORIGINAL_CARRY['charged_bytes']+2*1234+128)
        self.assertGreater(projection['cumulative_charge_bound'],10*2**30)
        self.assertGreater(projection['cumulative_charge_bound'],13*2**30)
        self.assertLess(projection['cumulative_charge_bound'],17*2**30)

    def test_budget_gate_fails_before_one_shot_output_affinity_or_science(self):
        import test_h4_prelaunch_synthetic as fixtures
        plan,auth,review=fixtures.documents('cache-budget-gate')
        plan['storage']=audit.storage_projection();fixtures.rebind(plan,auth,review)
        observation=fixtures.observed(plan);observation['observed_monotonic']=time.monotonic()
        observation['memory']['observed_at']=observation['observed_monotonic']
        with patch.object(bind,'verify_runtime',return_value=({},{})),patch.object(bind,'verify_frozen_receipts'),\
             patch.object(bind,'host_readonly',return_value=observation),patch.object(bind,'claim_once') as claim,\
             patch.object(bind.os,'sched_setaffinity') as affinity:
            with self.assertRaisesRegex(Stop,'cumulative output budget'):bind.launch(plan,auth,review,explicit_launch=True)
            claim.assert_not_called();affinity.assert_not_called()
        self.assertFalse(Path(plan['output_root']).exists())
