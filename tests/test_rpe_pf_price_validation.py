"""Artificial supplied PF price metadata; no price prediction or science."""
from copy import deepcopy
import unittest
from trottertracks.rpe_pf_price_validation import describe_candidate


class Reader:
    def __init__(self, values):self.values=values
    def read(self, loc):
        value=deepcopy(self.values[loc['path']])
        for key in loc.get('selector', []):value=value[key]
        return value


def fixture():
    model_ref=dict(commit='f'*40,path='MODEL.json')
    config_ref=dict(commit='c'*40,path='CONFIG.json')
    price_ref=dict(commit='p'*40,path='PRICES.json')
    config=dict(id='cfg',spec=dict(method='PF',formula='2nd',L_D=7,binding={'snapshot':'same'},
        split={'original_PF_method':'B0I','scalar':'one-time'},profile={'P':0}),rounds=[dict(
            m=m,q_PF=2**m,actual_delta_pf='1/10',T=str(2**m/10),candidate={'id':'physical'+str(m)})for m in range(13)])
    model=dict(configuration_id='cfg',models={a:dict(coefficients={'one':1,'q':2},
        features=['one','q'],domain=dict(route='TRUNCATED_PF',axis=a,implementation={'lowering_id':'fixed'},profile={'P':0}),
        calibration=[dict(cell_id='cfg:m0',source={'commit':'cal','path':'direct0'}),
                     dict(cell_id='cfg:m1',source={'commit':'cal','path':'direct1'})])for a in ('X','Y')},
        calibration_rounds=[0,1],holdout_round=2)
    rows=[dict(m=m,axis=a,kind='DIRECT_COMPILED_DETERMINISTIC'if m<2 else'CONDITIONAL_AFFINE_PRICE_REFERENCE',
        actual_delta_pf='1/10',unit_price_RZ=1+2*2**m,price_evidence={'source':model_ref})for m in range(13)for a in ('X','Y')]
    table=dict(axes=rows,model=model_ref,holdout=[dict(m=2,axis=a,q=4,predicted=9,observed=9,
        receipt={'source':{'commit':'h','path':'receipt'+a}})for a in ('X','Y')])
    candidate=dict(name='TRUNCATED_PF:cfg',method='TRUNCATED_PF',configuration_id='cfg',key={'p':7},
        prices=[r['unit_price_RZ']for r in rows],pf_evidence=dict(configuration=config_ref,prices=price_ref))
    return candidate,Reader({'CONFIG.json':config,'MODEL.json':model,'PRICES.json':table})


class PFDescriptors(unittest.TestCase):
    def test_on_demand_supplied_model_domain_and_holdout(self):
        candidate,reader=fixture();before=deepcopy((candidate,reader.values))
        result=describe_candidate('.',candidate,reader)
        self.assertEqual(len(result),1);desc=result[0]
        self.assertEqual(desc['identity']['coefficients']['X'],{'one':1,'q':2})
        self.assertEqual(desc['identity']['family']['p'],7)
        self.assertEqual(desc['identity']['actual_step'],{'numerator':1,'denominator':10})
        self.assertEqual(len(desc['points']),13)
        self.assertTrue(all(p['R']==0 for p in desc['points']))
        self.assertEqual(desc['points'][2]['predicted_RZ']['Y'],{'numerator':9,'denominator':1})
        self.assertEqual(desc['evidence_sources'][0]['source']['selector'],['holdout'])
        self.assertEqual(desc['evidence_sources'][0]['format'],'PF_HOLDOUT')
        self.assertFalse(desc['qualified_for_candidate'])
        self.assertEqual(before,(candidate,reader.values))

    def test_direct_only_needs_no_model_check(self):
        candidate,reader=fixture()
        for row in reader.values['PRICES.json']['axes']:row['kind']='DIRECT_COMPILED_DETERMINISTIC'
        self.assertEqual(describe_candidate('.',candidate,reader),[])

    def test_model_identity_ignores_configuration_label_shots_kappa(self):
        candidate,reader=fixture();first=describe_candidate('.',candidate,reader)[0]['model_id']
        reader.values['CONFIG.json']['spec']['kappa']=999
        candidate['name']='other-run';candidate['caps']=[10]*13;candidate['saved_points']=['allocation2']
        self.assertEqual(first,describe_candidate('.',candidate,reader)[0]['model_id'])
        reader.values['MODEL.json']['models']['Y']['coefficients']['q']=3
        self.assertNotEqual(first,describe_candidate('.',candidate,reader)[0]['model_id'])

    def test_saved_accounting_row_resolves_primary_closure_source(self):
        candidate,reader=fixture();rows=reader.values['PRICES.json']['axes']
        candidate['price_axes']=[]
        for i,row in enumerate(rows):
            if row['m']<2:
                candidate['price_axes'].append(deepcopy(row));continue
            primary='PRIMARY'+str(i)
            reader.values[primary]=deepcopy(row)
            reader.values['ACCOUNT'+str(i)]=dict(m=row['m'],axis=row['axis'],physical_id='p'+str(row['m']),
                source={'C_price_table':dict(commit='p'*40,path=primary)})
            candidate['price_axes'].append(dict(m=row['m'],axis=row['axis'],physical_id='p'+str(row['m']),
                price_kind='CONDITIONAL_FIXED_COMPILER_CLOSURE_DERIVATION',
                price_source=dict(commit='a'*40,path='ACCOUNT'+str(i))))
        result=describe_candidate('.',candidate,reader)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['identity']['model_source'],{'commit':'f'*40,'path':'MODEL.json'})

    def test_known_pf_compiler_step_representation_preserves_canonical_grid(self):
        candidate,reader=fixture()
        for r in reader.values['CONFIG.json']['rounds']:
            r.update(T=2**r['m'],q_PF=3*2**r['m'],actual_delta_pf='1/3')
        for r in reader.values['PRICES.json']['axes']:r['actual_delta_pf']=float(1/3)
        for model in reader.values['MODEL.json']['models'].values():
            model['domain']['implementation']['lowering_id']='H46_LONG_TIME_NATIVE_COMPOSE'
        result=describe_candidate('.',candidate,reader)
        self.assertEqual(len(result[0]['points']),13)
        self.assertEqual(result[0]['points'][0]['canonical_actual_step'],'1/3')
        self.assertEqual(result[0]['points'][0]['round']['actual_delta_pf'],'1/3')
        self.assertEqual(result[0]['points'][0]['actual_step_rule'],'EXISTING_H46_BINARY64_T_DIV_Q')

    def test_wrong_frozen_domain_rejected(self):
        candidate,reader=fixture();reader.values['MODEL.json']['models']['X']['domain']['route']='FULL_PF'
        with self.assertRaisesRegex(ValueError,'DOMAIN_DIFFERS'):describe_candidate('.',candidate,reader)

    def test_historical_axis_selectors_keep_original_coefficient_sources(self):
        from trottertracks.rpe_pf_price_validation import _HISTORIC_MODELS
        candidate,reader=fixture();axes=reader.values['PRICES.json']['axes'];models=reader.values.pop('MODEL.json')['models']
        frozen={'models':{'B0I|S2|X':models['X'],'B0I|S2|Y':models['Y']}}
        reader.values[_HISTORIC_MODELS['path']]=frozen
        for row in axes:
            if row['m']<2:continue
            row['frozen_model']=dict(_HISTORIC_MODELS,selector=['models','B0I|S2|'+row['axis']])
            row['kind']='MODEL_PREDICTION'
        desc=describe_candidate('.',candidate,reader)[0]
        self.assertEqual(len(desc['evidence_sources']),2)
        self.assertEqual([s['m']for s in desc['evidence_sources']],[6,9])
        self.assertEqual(desc['calibration_sources'][0]['selector'],['models','B0I|S2|X'])
        self.assertTrue(all(s['configuration_id']=='cfg'for s in desc['evidence_sources']))


if __name__=='__main__':unittest.main(verbosity=2)
