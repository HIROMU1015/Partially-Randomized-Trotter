"""Artificial metadata-only checks of the real driver's selected functions."""
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT=Path(__file__).resolve().parents[1]


def selected(path,names,scope):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Missing original metadata function')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return NS(**{n:scope[n]for n in names})


class Driver(unittest.TestCase):
    def fixture(self,step=.2,domain=True):
        def same(a,b,message):
            if a!=b:raise ValueError(message)
        def canonical(value):
            return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        P=NS(ROUTE='ARTIFICIAL_PAULI',LOWERING_ID='ARTIFICIAL_D2',canonical_sha256=canonical,
            load_dictionary=lambda *a:dict(prefix=4,lambda_P=.25,saved_DF_scalar=.125),
            total_scalar=lambda dictionary,constant:dictionary['saved_DF_scalar']+constant,
            SCALAR_QUANTITY='ARTIFICIAL_SCALAR_ONCE',seed=lambda candidate,cohort,i,outer,piece:100+i*8+outer)
        cell_scope=dict(deepcopy=deepcopy,Fraction=Fraction)
        cell=selected(ROOT/'scripts/run_pauli_ld3_cal.py',('cell_request',),cell_scope).cell_request
        spec=dict(prefix=4,config_id='ARTIFICIAL_CONFIG',configuration=dict(id='ARTIFICIAL_CONFIG',spec=dict(
            L_D=4,split=dict(tail_dictionary=dict(kind='ARTIFICIAL'),lambda_R=dict(numerator=1,denominator=4),scalar=.125),
            binding=dict(snapshot='ARTIFICIAL'),profile=dict(P=0)),rounds=[dict(candidate={'id':'m12'},q_PF=4096,R_outer=99)]*13),
            cell=dict(cell_id='Q8R7',q=8,R=7,T=8*float(step),K=2),cohort='ARTIFICIAL_N2',
            m12=dict(id='m12',q=4096,R=99),p0=dict(commit='ARTIFICIAL',directory='artificial/'),
            config=None,total_view=None,old_seed_sources=[],plan=dict(fixture='ARTIFICIAL'))
        if domain:
            f=Fraction(step);spec.update(calibration_domain=dict(actual_step=dict(numerator=f.numerator,denominator=f.denominator),
                actual_step_hex=float(step).hex()),covered_rounds=[1,2,3])
        legacy_spec=spec['configuration']['spec']
        saved=cell(P,legacy_spec,dict(q=8,R=66,T=.8,K=2),.25)
        helper=NS(legacy=lambda:NS(B0=NS(connection=lambda:P,HAMILTONIAN_CONSTANT=2.)),
            cell_request=cell,PIN_ROOT=ROOT,LD2_CONFIG={'path':'config'},LD2_Q8R66={'path':'saved'},
            COMMON={'path':'common'},PIN='ARTIFICIAL_LEGACY',C2_DRIVER={'path':'ARTIFICIAL'})
        sources=dict(deepcopy=deepcopy,Fraction=Fraction,same=same,D3=helper,ROOT=ROOT,
            LD3_DRIVER=dict(path='scripts/run_pauli_ld3_cal.py',commit='ARTIFICIAL'),
            blob=lambda *a:(ROOT/'scripts/run_pauli_ld3_cal.py').read_bytes(),
            read=lambda ref:legacy_spec if ref['path']=='config'else saved,
            old_seeds=lambda spec:set(),COMPILE_MAXIMUM=4,SCRIPT='scripts/run_pauli_split_cal.py',
            request_api=lambda:NS(validate_driver_spec=lambda root,s:s),SPECS={5:{'legacy':'unchanged'}})
        return spec,selected(ROOT/'scripts/run_pauli_split_cal.py',('metadata','select_job_spec'),sources)

    def test_multistep_metadata_and_worker_preserve_exact_domain(self):
        spec,d=self.fixture()
        binding=d.metadata(spec)
        self.assertEqual(binding['request']['T'],1.6)
        self.assertEqual(binding['request']['step_time'],.2)
        self.assertEqual(binding['calibration_domain'],spec['calibration_domain'])
        self.assertEqual(binding['covered_rounds'],[1,2,3])
        self.assertEqual(binding['m12_canonical']['R'],99)
        self.assertEqual(binding['m12_canonical']['relation'],'WHOLE_SCHEDULE_PROVENANCE_NOT_CALIBRATION_R_OR_STEP_ANCHOR')
        self.assertEqual(d.select_job_spec(dict(prefix=4,spec=spec,binding=binding)),spec)
        self.assertEqual(len(binding['step_seeds']),2)
        self.assertEqual(binding['compile_maximum'],4)
        bad=deepcopy(binding);bad['calibration_domain']['actual_step']['numerator']+=1
        with self.assertRaisesRegex(ValueError,'step domain'):
            d.select_job_spec(dict(prefix=4,spec=spec,binding=bad))
        bad=deepcopy(binding);bad['covered_rounds']=[0]
        with self.assertRaisesRegex(ValueError,'covered rounds'):
            d.select_job_spec(dict(prefix=4,spec=spec,binding=bad))

    def test_non_binary64_exact_step_is_not_silently_rounded(self):
        spec,d=self.fixture(Fraction(1,3))
        with self.assertRaisesRegex(ValueError,'exact step'):
            d.metadata(spec)

    def test_legacy_metadata_fields_and_old_spec_selection_unchanged(self):
        spec,d=self.fixture(.1,domain=False)
        binding=d.metadata(spec)
        self.assertEqual(binding['request']['T'],.8)
        self.assertEqual(binding['request']['step_time'],.1)
        self.assertNotIn('calibration_domain',binding)
        self.assertNotIn('covered_rounds',binding)
        self.assertEqual(binding['m12_canonical']['relation'],'SAME_PER_STEP_R_AS_m12_BUT_DIFFERENT_q_T_AND_ID_NOT_A_ROUND_RECEIPT')
        self.assertEqual(d.select_job_spec(dict(prefix=5)),{'legacy':'unchanged'})


if __name__=='__main__':unittest.main(verbosity=2)
