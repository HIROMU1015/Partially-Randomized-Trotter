"""Run only an exactly granted H2 stage; scope/default never starts science."""
import argparse
import json
from types import FunctionType
import h2_outer32_contract as c

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument('--execute',action='store_true');g.add_argument('--supervise',action='store_true');g.add_argument('--audit',action='store_true')
    p.add_argument('--config');a=p.parse_args()
    if not (a.execute or a.supervise or a.audit):print(json.dumps(dict(stages=list(c.STAGES),science_started=False)));return 0
    try:
        c.require(a.config,'H2_CONFIG_REQUIRED');cfg=c.read(a.config);c.authorize(cfg)
        if a.supervise:
            import run_h6_m5_saved_phase as original
            fn=FunctionType(original.supervise.__code__,dict(original.__dict__,c=c,__file__=__file__),'supervise',original.supervise.__defaults__,original.supervise.__closure__)
            return fn(cfg,a.config)
        if a.audit:
            from h2_outer32_runtime import get_runtime,extra_audit
            get_runtime(cfg);return int(bool(extra_audit(cfg)['errors']))
        c.require(c.read(cfg['supervisor']['entered'])==dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],grant_id=cfg['grant']['grant_id'],retry=False),'H2_EXTERNAL_SUPERVISOR_REQUIRED')
        if cfg['stage']==c.STAGES[0]:
            from h2_outer32_runtime import signal_stage
            signal_stage(cfg);return 0
        if cfg['stage']==c.STAGES[2]:
            from h2_outer32_pipeline import design_stage
            from pathlib import Path
            pilot=c.read(Path(cfg['run_root']).parents[1]/c.STAGES[1]/'config_APPROVED_PRIVATE.json')
            design_stage(cfg,pilot);return 0
        from h2_outer32_runtime import get_runtime
        return get_runtime(cfg).runner.execute(cfg)
    except Exception as exc:print(json.dumps(dict(status='REFUSED_OR_STOP',reason=str(exc),retry=False)));return 2

if __name__=='__main__':raise SystemExit(main())
