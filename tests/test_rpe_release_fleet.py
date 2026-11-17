"""Four-host mapping uses saved profile contracts, without contacting a host."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('fleet',ROOT/'scripts/prepare_rpe_release_fleet.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class FleetConfiguration(unittest.TestCase):
    def test_stage_sources_host_paths_and_original_profiles_remain_separate(self):
        with tempfile.TemporaryDirectory()as temporary:
            base=Path(temporary);profiles=base/'old';prices=base/'price';hosts={};old={}
            q3='2'*40;worker='d'*40;price='a'*40;adapter='3'*40;provider='e'*40
            paths={}
            for host in ['local','server1','server2','local0-wsl']:
                root=base/host
                runtime=str(root/'python')
                rule=dict(stage='compile',provider='PRICE_MODEL_VALIDATION',source_commit=price,
                    profile=dict(interpreter=runtime,preparation_sources={price:str(root/'old-price')}))
                profile=dict(host_id=host,root=str(root/'q3'),interpreter=runtime,route_views={},
                    adapter_root=str(root/'adapter'),adapter_manifest={'commit':adapter,'files':{}},
                    preparation_sources={worker:str(root/'worker'),provider:str(root/'native')},
                    preparation_profiles=[rule,dict(stage='signal',provider='PF_SIGNALS',source_commit=provider,
                        profile=dict(interpreter=runtime,preparation_sources={provider:str(root/'native')}))])
                M.put(profiles/(host+'.json'),profile);old[host]=(profiles/(host+'.json')).read_bytes()
                replacement=json.loads(json.dumps(profile));replacement['preparation_profiles'][0]['profile']['preparation_sources'][price]=str(root/'verified-price')
                M.put(prices/(host+'.json'),replacement)
                paths[host]=str(profiles/(host+'.json'))
                hosts[host]=dict(worker_source=worker,worker_root=str(root/'worker'),python=runtime,
                    repository=str(root/'objects.git'),seed_repository=str(root/'worker'),
                    metadata_root=str(root/'metadata'),install_root=str(root/'releases'),
                    runtime_observations={'preparation_0':dict(version_info=[3,11,0,'candidate',1],modules={'qiskit':'1.3.0'})})
            M.put(base/'ROUND.json',dict(executor=dict(distributed=dict(preparation_profiles=paths)),price_validation={'source':price}))
            M.put(base/'SERVICE.json',dict(round=str(base/'ROUND.json'),computation_source=q3,
                migration={'candidate_completion_source':worker}))
            report=M.prepare('1'*40,'SYNTHETIC_RELEASE',base/'SERVICE.json',prices,hosts,base/'fleet')
            manifest=M.read(base/'fleet/RELEASE_COMPONENTS.json')
            self.assertEqual(manifest['components']['q3']['source'],q3)
            self.assertEqual(manifest['components']['price_worker']['source'],price)
            self.assertIn('price_worker',manifest['stage_components']['compile'])
            self.assertNotIn(str(base),json.dumps(manifest))
            observed=next(c for c in manifest['runtime_contracts']['local']if c['runtime']=='preparation_0')
            self.assertEqual(observed['runtime_observation']['version_info'],[3,11,0,'candidate',1])
            for host in hosts:
                self.assertEqual((profiles/(host+'.json')).read_bytes(),old[host])
                self.assertTrue(report['profile_preservation'][host]['q3_profile_core_unchanged'])
                layout=M.read(base/'fleet/hosts'/(host+'.json'))
                self.assertEqual(layout['host_id'],host)
                self.assertTrue(Path(layout['manifest_path']).is_absolute())
                self.assertTrue(Path(layout['layout_path']).is_absolute())
                self.assertEqual(layout['component_roots']['price_worker'],str(base/host/'verified-price'))
            self.assertEqual(M.read(base/'fleet/hosts/server1.json')['availability'],'DEFERRED_OFFLINE')
            self.assertEqual(M.read(base/'fleet/hosts/local0-wsl.json')['intake'],'OFF')
            self.assertEqual(report['scientific_calls'],0);self.assertEqual(report['hosts_contacted'],[])


if __name__=='__main__':unittest.main()
