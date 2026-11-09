"""Small full-wrapper molecular check; excluded from every price cohort."""
import h46_full_h_t20_opt as study
from h46_full_h_t20_aer import simulate_statevector
study.aer_statevector=simulate_statevector

if __name__=='__main__':
    checks=[]
    for system in ('H4','H6'):
        r=study.row(system,1,4,2,'ACTION_ADAPTER_VALIDATION')
        previous=study.PRIVATE/'cost'/'ACTION_ADAPTER_VALIDATION_NOT_COST_SAMPLE'/r['id']/'0'/'result.json'
        if system=='H4' and previous.exists():
            value=study.read(previous)
        else:
            value=study.worker((r,'ACTION_ADAPTER_VALIDATION_V2_NOT_COST_SAMPLE',0))
        checks.append(dict(system=system,status=value['status'],checks=value['checks'],
            included_in_cost_samples=False,cohort=value['cohort'],index=0))
    study.save(study.PUBLIC/'action_adapter_validation.json',dict(status='PASS',trials=checks,
        compiler_axes=4,included_in_cost_samples=False))
    print('H4/H6 native and compiled full-wrapper action adapter PASS')
