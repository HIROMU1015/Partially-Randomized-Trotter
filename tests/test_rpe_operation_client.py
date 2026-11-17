"""Common operation transports reuse the existing central Gateway; no science."""
from pathlib import Path
import json
import sys
import tempfile
import threading
import time
import unittest
import socket
import subprocess
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import rpe_operation_client as C
from trottertracks.rpe_release_registry import register,write
from test_rpe_local0_search import fixture_config,request


class OperationClientTests(unittest.TestCase):
    def test_reconnect_removes_only_stale_owned_socket_and_preserves_active_owner(self):
        with tempfile.TemporaryDirectory()as temporary:
            path=Path(temporary)/'reverse.sock'
            listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.bind(str(path));listener.close()
            subprocess.run([sys.executable,'-c',C.REMOTE_SOCKET_PREPARE,str(path)],check=True,capture_output=True)
            self.assertFalse(path.exists())
            listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.bind(str(path));listener.listen(1)
            try:
                process=subprocess.run([sys.executable,'-c',C.REMOTE_SOCKET_PREPARE,str(path)],capture_output=True)
                self.assertNotEqual(process.returncode,0);self.assertTrue(path.exists())
                self.assertIn(b'preserve its owner',process.stderr)
            finally:listener.close()

    def test_selected_current_is_pinned_before_actual_new_scope(self):
        config={'server_config':'/CENTRAL/SERVICE.json'};calls=[]
        def invoke(config,value):
            calls.append(value)
            return {'release':{'id':'validated-A'},'controller_host':'RomuTail'}if len(calls)==1 else {'status':'STARTED'}
        raw=json.dumps({'scope':{'methods':['PR_PAULI']}}).encode()
        with patch.object(C,'invoke',side_effect=invoke):value=C.operate(config,'start',raw)
        self.assertEqual(calls[0]['action'],'preview')
        self.assertEqual(calls[0]['operation'],'resolve')
        self.assertEqual(calls[1]['release'],'validated-A')
        self.assertEqual(calls[0]['request_base64'],calls[1]['request_base64'])
        self.assertTrue(calls[1]['invoke_result_sync'])
        self.assertEqual(value['operation_controller'],'RomuTail')

    def test_missing_optimizer_holds_only_new_work_before_forwarding(self):
        config={'server_config':'/CENTRAL/SERVICE.json','deployment_layout':'/OWNED/LAYOUT.json'}
        with patch.object(C,'invoke',return_value={'release':{'id':'A'},'components':{'components':{}}})as invoke,\
             patch.object(C,'read',return_value={}),\
             patch('trottertracks.rpe_release_deployment.ensure_components',return_value={'status':'HOLD','holds':['missing fixed source']})as ensure:
            with self.assertRaisesRegex(RuntimeError,'missing fixed source'):
                C.operate(config,'start',b'{"scope":{}}')
        self.assertEqual(invoke.call_count,1)
        self.assertEqual(ensure.call_args.kwargs['components'],['optimizer'])

    def test_legacy_resume_keeps_central_pin_without_installing_latest(self):
        calls=[]
        def invoke(config,value):
            calls.append(value)
            return {'release':{'id':'old-fixed'}}if len(calls)==1 else {'status':'REUSED_COMPLETED'}
        config={'server_config':'/CENTRAL/SERVICE.json','require_components':True}
        with patch.object(C,'invoke',side_effect=invoke):result=C.operate(config,'resume',run_id='old-run')
        self.assertEqual(result['status'],'REUSED_COMPLETED')
        self.assertEqual(calls[1]['run_id'],'old-run');self.assertNotIn('release',calls[1])

    def test_route_failure_never_falls_back_to_a_local_controller(self):
        with tempfile.TemporaryDirectory()as temporary:
            config={'transport':'unix','socket_path':str(Path(temporary)/'absent.sock'),'server_config':'/CENTRAL/SERVICE.json'}
            with patch.object(C,'invoke_endpoint',side_effect=AssertionError('local controller fallback')):
                with self.assertRaises(OSError):C.operate(config,'preview')

    def test_unix_forwarded_requests_deduplicate_and_follow_run_pin(self):
        with tempfile.TemporaryDirectory()as temporary:
            base=Path(temporary);state=base/'central-state';registry=base/'RELEASES.json'
            for label in ['A','B']:
                home=base/label;home.mkdir();service=fixture_config(home)
                service.update(state_directory=str(state),code_source=label*40)
                write(home/'SERVICE.json',service)
                register(registry,label,home/'SERVICE.json',activate=label=='A')
            bootstrap=base/'SERVICE.json';write(bootstrap,{'release_registry':str(registry),'state_directory':str(state)})
            endpoint=base/'endpoint.py'
            endpoint.write_text('''import argparse,base64,json,sys
from pathlib import Path
sys.path[:0]=[ROOT+'/src',ROOT+'/tests',ROOT+'/scripts']
from trottertracks.rpe_release_registry import select
from trottertracks.rpe_local0_search import Gateway,write
p=argparse.ArgumentParser();p.add_argument('--transport-base64');v=json.loads(base64.b64decode(p.parse_args().transport_base64))
config=json.loads(Path(v['server_config']).read_text());q=json.loads(base64.b64decode(v['request_base64']));q['action']=v['action']
for k in ['run_id','release','instruction']:
 if v.get(k):q[k]=v[k]
chosen=select(config,q)
def launch(run,*_):
 count=Path(COUNT);n=int(count.read_text())+1 if count.exists()else1;count.write_text(str(n))
 write(run/'LAUNCH.json',{'pid':0,'invocation_id':str(n)});write(run/'EXIT.json',{'returncode':0,'invocation_id':str(n)})
 write(run/'ROUND_RESULT.json',{'methods':{'PR_PAULI':{'stop_reason':'NO_CANDIDATE_LEFT_IN_RANGE'}}})
if v.get('operation')=='resolve':result={'release':chosen.get('release'),'controller_host':'RomuTail'}
else:result=Gateway(config,preflight=lambda *_:{'status':'PASS'},launch=launch).handle(q)
print(json.dumps({'ok':True,'result':result}))
'''.replace("ROOT+",repr(str(ROOT))+"+").replace('Path(COUNT)',repr(base/'COUNT').replace('PosixPath','Path')).replace('else1','else 1'))
            directory=base/'operation';gateway=base/'GATEWAY.json'
            write(gateway,{'state_directory':str(directory),'socket_path':str(directory/'control.sock'),
                'python':sys.executable,'remote_entry':str(endpoint),'server_config':str(bootstrap),
                'forwards':[{'host_id':'server1','enabled':False}]})
            server=C.OperationGateway(gateway)
            thread=threading.Thread(target=server.serve,daemon=True);thread.start()
            try:
                for _ in range(100):
                    if server.socket.exists():break
                    time.sleep(.01)
                client={'transport':'unix','socket_path':str(server.socket),'server_config':'/IGNORED/CLIENT/PATH'}
                raw=request(4,6);raw.pop('action');raw.pop('instruction');data=json.dumps(raw).encode()
                first=C.operate(client,'start',data);again=C.operate(client,'start',data)
                self.assertEqual(first['run_id'],again['run_id'])
                self.assertEqual(again['status'],'REUSED_COMPLETED');self.assertEqual((base/'COUNT').read_text(),'1')
                register(registry,'B',base/'B/SERVICE.json',activate=True)
                new=C.operate(client,'preview',data)
                old=C.operate(client,'resume',run_id=first['run_id'])
                self.assertEqual(new['release']['id'],'B');self.assertEqual(old['release']['id'],'A')
                self.assertEqual((server.socket.stat().st_mode&0o777),0o600)
                self.assertEqual(server.forward_status['server1']['status'],'DISABLED')
            finally:
                (directory/'STOP').touch();thread.join(3)
            self.assertFalse(thread.is_alive())


if __name__=='__main__':unittest.main()
