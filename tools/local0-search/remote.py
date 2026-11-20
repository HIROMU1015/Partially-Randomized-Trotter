"""Small stdlib bootstrap: resolve service before importing controller code.

CONFIG is supplied by the existing PowerShell JSON transport. No client-side
fixed code_root, separate PlanOnly source or executable request text is used.
"""
import base64
import hashlib
import json
import pathlib
import subprocess
import sys

if 'CONFIG' not in globals():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport-base64',required=True)
    arguments=parser.parse_args()
    CONFIG=json.loads(base64.b64decode(arguments.transport_base64))

try:
    path = pathlib.Path(CONFIG['server_config'])
    config = json.loads(path.read_text())
    while 'service_config' in config:
        path = pathlib.Path(config['service_config'])
        config = json.loads(path.read_text())
    config['config_path'] = str(path)
    request = json.loads(base64.b64decode(CONFIG['request_base64']))
    request['action'] = CONFIG['action']
    if CONFIG.get('instruction'):
        request['instruction'] = CONFIG['instruction']
    if CONFIG.get('run_id'):
        request['run_id'] = CONFIG['run_id']
    if CONFIG.get('release'):
        request['release'] = CONFIG['release']
    selected = config
    if config.get('release_registry'):
        state = pathlib.Path(config['state_directory'])
        pinned = None
        if not request.get('scope'):
            target = state / 'CURRENT_TARGET.json'
            if request.get('run_id'):
                index = json.loads((state / 'RUNS.json').read_text()) if (state / 'RUNS.json').exists() else {}
                row = index.get(request['run_id'])
                if row:
                    pinned = pathlib.Path(row['run_directory']) / 'CONTROLLER_CONFIG.json'
            elif target.exists():
                pinned = pathlib.Path(json.loads(target.read_text())['run_directory']) / 'CONTROLLER_CONFIG.json'
        if pinned and pinned.exists():
            selected = json.loads(pinned.read_text())
        else:
            registry = json.loads(pathlib.Path(config['release_registry']).read_text())
            release = request.get('release') or registry['current']
            selected = json.loads(pathlib.Path(registry['releases'][release]['service_config']).read_text())
    root = pathlib.Path(selected['code_root']).resolve()
    sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
    from trottertracks.rpe_local0_search import Gateway, RequestError
    from trottertracks.rpe_release_registry import select
    if pathlib.Path(sys.modules['trottertracks.rpe_local0_search'].__file__).resolve() != root / 'src/trottertracks/rpe_local0_search.py':
        raise ValueError('Controller module does not match selected deployment')
    try:
        selected=select(config,request)
        for filename,expected in selected.get('fixed_settings',{}).items():
            asset=pathlib.Path(filename)
            if not asset.is_file()or hashlib.sha256(asset.read_bytes()).hexdigest()!=expected:
                raise RequestError('fixed_settings','fixed deployment setting changed: '+filename)
        if CONFIG.get('operation')=='resolve':
            manifest=selected.get('release_manifest')
            components=json.loads(pathlib.Path(manifest).read_text())if manifest else None
            layout=selected.get('host_layouts',{}).get(CONFIG.get('host_id'))
            host_layout=json.loads(pathlib.Path(layout).read_text())if isinstance(layout,str)else layout
            result=dict(status='RELEASE_RESOLVED',release=selected.get('release'),
                components=components,host_layout=host_layout,controller_host='RomuTail',
                controller_source=selected['code_source'],computation_source=selected['computation_source'])
        else:
            if request['action']in('start','resume')and config.get('legacy_relay_command'):
                spec=json.loads(pathlib.Path(selected['round']).read_text())
                if not spec.get('executor',{}).get('distributed',{}).get('relay'):
                    process=subprocess.run(config['legacy_relay_command'],capture_output=True,text=True,timeout=30)
                    if process.returncode:raise RequestError('relay','legacy relay could not be started: '+process.stderr[-1000:])
            result = Gateway(config).handle(request)
            transport_hook=config.get('relay_transport_command')
            if (request['action']in('start','resume')and transport_hook
                    and result.get('status')in('STARTED','ALREADY_RUNNING')):
                # This operational hook replaces only the mailbox relay. The
                # selected controller and scientific run keep their fixed code.
                try:
                    observed=subprocess.run(transport_hook,input=json.dumps(result),capture_output=True,text=True,
                        timeout=config.get('relay_transport_timeout_seconds',90))
                    transport=json.loads(observed.stdout)
                    if not isinstance(transport,dict):raise ValueError('Relay transport hook must return a JSON object')
                    if observed.returncode:
                        raise RuntimeError(str(transport.get('error')or transport.get('status')or
                            'Relay transport hook failed')[:1500])
                    result['relay_transport']=transport
                except Exception as error:
                    result['relay_transport']=dict(status='RELAY_TRANSPORT_PENDING',error=str(error)[:1500])
            hook=config.get('confirmed_run_hook')
            if CONFIG.get('invoke_result_sync')and hook and result.get('status')in('STARTED','ALREADY_RUNNING','REUSED_COMPLETED'):
                problem=(result.get('applied',{}).get('fixed',{}).get('problem')or{})
                artificial=str(problem.get('system','')).startswith('ARTIFICIAL')
                if artificial and not config.get('allow_artificial_result_sync'):
                    result['result_sync']=dict(status='ARTIFICIAL_SKIPPED')
                else:
                    try:
                        observed=subprocess.run(hook,input=json.dumps(result),capture_output=True,text=True,timeout=150)
                        result['result_sync']=json.loads(observed.stdout)
                    except Exception as error:result['result_sync']=dict(status='RESULT_SYNC_PENDING',error=str(error))
    except RequestError as error:
        result = dict(status='NOT_STARTED', started=False,
                      errors=[dict(field=error.field, reason=error.reason)])
    print(json.dumps(dict(ok=True, result=result), ensure_ascii=False))
except Exception as error:
    print(json.dumps(dict(ok=False, error=type(error).__name__ + ': ' + str(error)), ensure_ascii=False))
    raise SystemExit(1)
