"""Default scope only; sampling/compile remain explicitly unauthorized."""
import argparse
import json
import h46_pr_exploration_contract as c

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group()
    g.add_argument('--execute',action='store_true');g.add_argument('--supervise',action='store_true')
    g.add_argument('--audit',action='store_true');p.add_argument('--config');a=p.parse_args()
    if not any((a.execute,a.supervise,a.audit)):
        print(json.dumps(dict(status='SCOPE_ONLY',cohort=c.design.COHORT,stages=c.STAGES,
          scientific_execution_authorized=False,n_E_adopted=None,mandatory_STOP=True)));return 0
    try:
        c.require(a.config,'CONFIG_REQUIRED');cfg=c.read(a.config);c.authorize(cfg)
        from h46_pr_exploration_runtime import get_runtime
        runtime=get_runtime()
        if a.supervise:return runtime.runner.supervise(cfg,a.config)
        if a.audit:return int(bool(runtime.auditor.audit(cfg)['errors']))
        c.require(c.read(cfg['supervisor']['entered'])==dict(SOURCE=cfg['SOURCE'],
          plan_sha256=cfg['plan_sha256'],grant_id=cfg['grant']['grant_id'],retry=False),
          'EXTERNAL_EXPLORATION_SUPERVISOR_REQUIRED')
        return runtime.runner.execute(cfg)
    except Exception as exc:
        print(json.dumps(dict(status='REFUSED',reason=str(exc),retry=False)));return 2

if __name__=='__main__':raise SystemExit(main())
