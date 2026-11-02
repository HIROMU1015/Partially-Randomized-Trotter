"""The corrected saved-input kernel under the exact new 216-axis grant."""
import argparse
import pf_direct_retry_contract as r
from pf_direct_production_worker import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('spec','grant','system','candidate','output'):p.add_argument('--'+key,required=True)
    args=p.parse_args()
    raise SystemExit(run(args.spec,args.grant,args.system,args.candidate,args.output,authorize=r.authorize))
