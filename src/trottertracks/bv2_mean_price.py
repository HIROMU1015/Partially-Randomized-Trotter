"""C-side passive entry for B PR_RPE_RMSE_V2, using B's existing adapter.

No price calculation, scientific file reader or B controller implementation.
Callers supply the fixed B integration module and already sourced records.
New valued records must explicitly declare a full-wrapper compiled mean unit;
this function never assigns that meaning to S0a or legacy C1 identities.
"""
from copy import deepcopy

B_RELEASE = '6a9d7fd36b687e58aecbed1278c6d7927d6827d6'
QUANTITY = 'FULL_WRAPPER_MEAN_COMPILED_PRICE'
SCHEMA = 'PR_RPE_RMSE_V2'


class V2MeanPriceLookup:
    """Same no-argument result interface as B.MeanPriceLookup.

    PR A-source-bound rows reuse B.MeanPriceLookup/MeanPriceProvider unchanged.
    PF and full-H RTE need their own physical price bridges, so stay missing.
    This stricter new C entry does not change B's legacy quantity-omission rule.
    """
    def __init__(self,config,records=(),*,integration,selections=None):
        B=integration.B
        if B.SCHEMA!=SCHEMA or config.get('schema')!=SCHEMA:raise ValueError('B_V2_SCHEMA_REQUIRED')
        B._validate(config)
        self.config=deepcopy(config);self.B=B;self.lookup=None
        self.method=B.comparison_method(config['spec'])
        self.reason=('USE_EXISTING_PF_PRICE_PROVIDER_WITH_TARGET_RETAINED_H_BINDING'
                     if self.method in ('FULL_PF','TRUNCATED_PF') else
                     'FULL_H_RTE_PRICE_PREPARATION_LAW_EVENT_COUNT_BRIDGE_MISSING'
                     if self.method=='FULL_H_RTE' else 'PR_PHYSICAL_LOOKUP_VIEW_NOT_CONNECTED')
        records=deepcopy(list(records))
        for record in records:
            quantity=record.get('quantity')
            if quantity is not None and quantity!=QUANTITY:raise ValueError('STRUCTURAL_OR_LOCAL_QUANTITY_IS_NOT_MEAN_COMPILED_PRICE')
            if record.get('mean',{}).get('rz_count') is not None:
                if quantity!=QUANTITY:raise ValueError('NEW_C_V2_VALUED_RECORD_REQUIRES_EXPLICIT_COMPILED_MEAN_QUANTITY')
                if not record.get('source'):raise ValueError('PRICE_SOURCE_REQUIRED')
        if self.method=='PR' and config['spec'].get('schedule_adapter')==integration.ADAPTER:
            self.lookup=integration.MeanPriceLookup(config,records,selections=selections)

    def __call__(self):
        B=self.B;B._validate(self.config)
        if self.lookup is not None:
            records=self.lookup()
            for record in records:
                if record.get('mean',{}).get('rz_count') is not None and record.get('quantity')!=QUANTITY:
                    raise ValueError('LOOKUP_CHANGED_COMPILED_MEAN_QUANTITY')
            return records
        return [dict(**B.p.measurement(row,axis,saved_sampler=self.method in ('PR','FULL_H_RTE')),
            identity=deepcopy(row['identity']),law_binding=deepcopy(row['law_binding']),axis=axis,
            source=dict(module='trottertracks.bv2_mean_price',B_commit=B_RELEASE),
            quantity=QUANTITY,cohort='C_V2_UNCONNECTED_METHOD',classification='UNAVAILABLE',
            expectation='UNAVAILABLE',mean=dict(rz_count=None),n_paths=None,model=None,
            price_status='OUTSIDE_DOMAIN',reason=self.reason,
            uncertainty=dict(joint_covariance=None,systematic=None,total_CI=None,all_law_expectation=None),
            normalization_is_price_multiplier=False,full_RPE_price_adopted=False)
            for row in self.config['rounds'] for axis in B.p.AXES]
