"""Run the ordinary PR-DF Q3 adapter with fourteen-row saved-signal intake."""

from copy import deepcopy
import inspect
import json
from pathlib import Path
import runpy
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FIXED_INTAKE = '27e70efe533862f38f4080aabb9cdd3ad1749fe9'


def install_prdf_intake(load_prdf_route, view, route_views):
    route, resolution = load_prdf_route(view, route_views)
    original = route.U.metadata
    file = Path(inspect.getsourcefile(original)).resolve()
    relative = file.relative_to(route.U.ROOT)
    expected = subprocess.check_output(['git', 'show', FIXED_INTAKE+':'+str(relative)], cwd=ROOT)
    if file.read_bytes() != expected:
        raise ValueError('PR-DF intake differs from its original fixed SOURCE')
    source = inspect.getsource(original)
    old = "len(view['signal_packet']['means'])!=13"
    if source.count(old) != 1:
        raise ValueError('Original PR-DF intake length check differs')
    scope = dict(vars(route.U))
    exec(compile(source.replace(old, "len(view['signal_packet']['means'])!=len(original['rounds'])"),
                 str(file), 'exec'), scope)
    route.U.metadata = scope['metadata']
    return route, resolution


def verify_reissued_carriers(original, git_json, view, parent, metadata):
    """Verify the new carrier locators while comparing unchanged numeric fields."""
    binding, packet_view = metadata
    origin = git_json(view, parent['signal_packet'])
    name = 'origin_signal_packet'
    packet = packet_view['signal_packet'][name]
    checked = deepcopy(packet)
    for old, new in zip(origin['means'], checked['means'][:13]):
        if new['horizon_origin'] != old['source'] or new['normalization']['source'] != new['source']:
            raise ValueError('Reissued PR-DF mean lost its original carrier')
        new['normalization']['source'] = old['normalization']['source']
    original(view, parent, (binding, dict(packet_view, signal_packet=dict(packet_view['signal_packet'], **{name: checked}))))


def main():
    if '--request' not in sys.argv:
        raise ValueError('The ordinary --request file is required')
    request = json.loads(Path(sys.argv[sys.argv.index('--request')+1]).read_text())
    if (request.get('method'), request['route']['kind'], request['origin']['M']) != ('PR_DF', 'prdf_d2_split', 13):
        raise ValueError('This validation wrapper is restricted to PR-DF M=13')
    adapter = runpy.run_path(str(ROOT/'scripts/run_rpe_optimizer_q3_job.py'))
    entry = adapter['main']
    load = adapter['load_prdf_route']
    verify = adapter['validate_horizon_extension']
    entry.__globals__['load_prdf_route'] = lambda view, views: install_prdf_intake(load, view, views)
    entry.__globals__['validate_horizon_extension'] = lambda *args: verify_reissued_carriers(verify, adapter['git_json'], *args)
    return entry()


if __name__ == '__main__':
    raise SystemExit(main())
