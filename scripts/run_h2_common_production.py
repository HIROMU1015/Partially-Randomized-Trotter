"""Separate production entry. The previous mock --execute guard is unchanged."""
import argparse
import json
import h2_common_production_contract as c


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--contract',required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--approval');p.add_argument('--grant');a=p.parse_args()
    try:
        if a.execute and not (a.approval and a.grant):
            raise c.Stop('EXPLICIT_USER_SCIENCE_APPROVAL_AND_GRANT_REQUIRED')
        cfg=c.read(a.contract)
        if a.execute:result=c.execute(cfg,c.read(a.approval),c.read(a.grant))
        else:result=c.preflight(cfg)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        return 0
    except (ValueError,KeyError,OSError,TypeError) as exc:
        print('STOP: '+str(exc),file=c.sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
