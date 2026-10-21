"""Default scope only. Explicit future approval required before native imports."""
import argparse
import json
import h246_confirmation_contract as c

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group()
    for name in ('execute','supervise','audit','evaluate'):g.add_argument('--'+name,action='store_true')
    p.add_argument('--config');a=p.parse_args()
    if not any((a.execute,a.supervise,a.audit,a.evaluate)):
        print(json.dumps(dict(status='SCOPE_ONLY',stages=c.STAGES,cohort=c.d.COHORT,
          scientific_execution_authorized=False,mandatory_STOP=True)));return 0
    try:
        c.require(a.config,'CONFIG_REQUIRED');cfg=c.read(a.config);c.authorize(cfg)
        if a.evaluate:
            from h246_confirmation_evaluation import evaluate
            evaluate(cfg);return 0
        from h246_confirmation_runtime import get_runtime,audit
        rt=get_runtime(cfg)
        if a.supervise:return rt.runner.supervise(cfg,a.config)
        if a.audit:return int(bool(audit(cfg)['errors']))
        c.require(c.read(cfg['supervisor']['entered'])==dict(SOURCE=cfg['SOURCE'],
          plan_sha256=cfg['plan_sha256'],grant_id=cfg['grant']['grant_id'],retry=False),
          'EXTERNAL_CONFIRMATION_SUPERVISOR_REQUIRED')
        return rt.runner.execute(cfg)
    except Exception as exc:
        print(json.dumps(dict(status='REFUSED',reason=str(exc),retry=False)));return 2

if __name__=='__main__':raise SystemExit(main())
