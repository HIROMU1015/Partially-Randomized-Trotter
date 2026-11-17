"""Small stdlib bootstrap: resolve service before importing controller code.

CONFIG is supplied by the existing PowerShell JSON transport. No client-side
fixed code_root, separate PlanOnly source or executable request text is used.
"""
import base64
import json
import pathlib
import sys

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
    if pathlib.Path(sys.modules['trottertracks.rpe_local0_search'].__file__).resolve() != root / 'src/trottertracks/rpe_local0_search.py':
        raise ValueError('Controller module does not match selected deployment')
    try:
        result = Gateway(config).handle(request)
    except RequestError as error:
        result = dict(status='NOT_STARTED', started=False,
                      errors=[dict(field=error.field, reason=error.reason)])
    print(json.dumps(dict(ok=True, result=result), ensure_ascii=False))
except Exception as error:
    print(json.dumps(dict(ok=False, error=type(error).__name__ + ': ' + str(error)), ensure_ascii=False))
    raise SystemExit(1)
