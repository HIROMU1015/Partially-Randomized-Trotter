"""No-argument scope only; disabled packs refuse before numerical imports."""
import argparse
import json
from pathlib import Path
import h46_standard_y4_contract as c


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    for mode in ('preflight', 'activate', 'launch', 'supervise', 'execute', 'audit', 'evaluate'):
        g.add_argument('--' + mode, action='store_true')
    p.add_argument('--package')
    p.add_argument('--approval')
    p.add_argument('--config')
    a = p.parse_args()
    if not any(getattr(a, m) for m in ('preflight', 'activate', 'launch', 'supervise', 'execute', 'audit', 'evaluate')):
        print(json.dumps(dict(status='SCOPE_ONLY', valid_grants=0, science_started=False)))
        return 0
    try:
        if a.config:
            cfg = c.read(a.config)
            # This boundary must precede importing native modules or runtime.
            c.require(cfg.get('authorized') is True and cfg.get('grant', {}).get('valid') is True,
                'EXPLICIT_STANDARD_Y4_SCIENCE_APPROVAL_REQUIRED')
            c.install_paths(cfg['assets'], cfg['runtime']['site_packages'])
            c.authorize(cfg)
            from h46_standard_y4_runtime import runtime, audit
            if a.audit:
                r = audit(cfg)
                print(json.dumps(r))
                return int(bool(r['errors']))
            rt = runtime(cfg)
            if a.supervise:
                return rt.runner.supervise(cfg, a.config)
            c.require(a.execute, 'STAGE_EXECUTION_MODE_REQUIRED')
            c.require(c.read(cfg['supervisor']['entered']) == dict(SOURCE=cfg['SOURCE'],
                plan_sha256=cfg['plan_sha256'], grant_id=cfg['grant']['grant_id'], retry=False),
                'OWN_EXTERNAL_SUPERVISOR_REQUIRED')
            return rt.runner.execute(cfg)
        c.require(a.package, 'PACKAGE_REQUIRED')
        package = c.read(a.package)
        c.install_paths(package['assets'], package['runtime']['site_packages'])
        if a.preflight:
            from prepare_h46_standard_y4 import metadata_preflight
            print(json.dumps(metadata_preflight(a.package)))
            return 0
        if a.activate:
            c.require(a.approval, 'EXPLICIT_APPROVAL_RECEIPT_REQUIRED')
            print(json.dumps(dict(configs=c.activate(a.package, a.approval))))
            return 0
        configs = [str(Path(r['draft_config_path']).with_name('config_APPROVED_PRIVATE.json'))
                   for r in package['stages']]
        for path in configs:
            c.authorize(c.read(path))
        if a.launch:
            from h46_standard_y4_runtime import launch
            if not launch(a.package, configs):
                return 1
            from h46_standard_y4_evaluation import evaluate
            evaluate(a.package, configs)
            return 0
        if a.evaluate:
            from h46_standard_y4_evaluation import evaluate
            evaluate(a.package, configs)
            return 0
        raise ValueError('EXPLICIT_MODE_REQUIRED')
    except Exception as exc:
        print(json.dumps(dict(status='REFUSED', reason=str(exc), retry=False, mandatory_STOP=True)))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
