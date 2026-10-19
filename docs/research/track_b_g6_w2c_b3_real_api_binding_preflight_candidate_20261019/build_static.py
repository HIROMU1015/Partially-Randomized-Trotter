"""Build static data only. Never imports or evaluates scientific modules."""
from pathlib import Path
import json,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.dont_write_bytecode=True
from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1 import static as S
from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1.authority import binding

def main():
    c=S.candidate(ROOT)
    outputs={'scientific_api_inventory.json':S.inventory(ROOT),'technical_contract_candidate.json':c,
             'api_binding_manifest_candidate.json':S.binding(),'monitoring_manifest_candidate.json':S.monitoring(),
             'authority_binding_candidate.json':binding(c,Path('/tmp/b3-bundle-artificial-FUTURE_FAKE_EXAMPLE'))}
    for name,value in outputs.items():(HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(S.preflight(c,ROOT)))
if __name__=='__main__':main()
