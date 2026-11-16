"""The real ordinary materializer with mocked P0 acquisition and Git publication."""
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import prepare_pauli_candidate as R
from trottertracks import pauli_p0_no_drop as N


class Materializer(unittest.TestCase):
    def test_adopted_default_reaches_new_builder_and_uses_separate_directory(self):
        payload=dict(prefix=7,binding=dict(system='ARTIFICIAL',snapshot='ARTIFICIAL_SNAPSHOT'),builder=N.provenance('FIXED'),
                     lambda_P=.25,saved_DF_scalar=.125)
        captured=[]
        def execute(root,out,source,prefix,desc,desc_source):
            captured.append((Path(out),source,prefix,deepcopy(desc),deepcopy(desc_source)))
            out.mkdir(parents=True)
            return dict(status=N.success(prefix),prefixes=[dict(prefix=prefix,dictionary_sha256='ARTIFICIAL',
                identity_metrics=dict(artificial=True))])
        def blob(root,source,path):return (Path(root)/path).read_bytes()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            directory=Path(temp)/'p0';descriptor=dict(descriptor=dict(scalar=.125,lambda_R=.2),
                source=dict(commit='OLD',path='SAVED_DESCRIPTOR.json'))
            with patch.object(R.subprocess,'check_output',side_effect=[b'DESCENDANT_WITH_RESULTS\n',b'CARRIER\n']), \
                 patch.object(R.subprocess,'run') as git,patch.object(R.A.E.p0,'blob',side_effect=blob), \
                 patch.object(R.A.E.p0,'metadata',return_value=dict(binding=payload['binding'])), \
                 patch.object(R.A.E.L,'read',return_value=payload), \
                 patch.object(N,'execute_p0',side_effect=execute),patch('sys.stderr',new=io.StringIO()):
                product=R.materializer(ROOT,directory,'FIXED')(7,descriptor)
            self.assertEqual(captured[0][0],directory/N.BUILDER_ID/'LD7')
            self.assertEqual(captured[0][1:3],('FIXED',7))
            self.assertEqual(product['builder_id'],N.BUILDER_ID)
            self.assertEqual(product['scientific_source'],'FIXED')
            self.assertEqual(product['item']['source']['commit'],'CARRIER')
            self.assertIn(N.BUILDER_ID,product['item']['source']['path'])
            self.assertTrue(any(c.args[0][:3]==['git','merge-base','--is-ancestor'] for c in git.call_args_list))
            self.assertFalse((directory/'LD7').exists())

    def test_existing_new_builder_output_is_a_stop_not_a_retry(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            directory=Path(temp)/'p0';(directory/N.BUILDER_ID/'LD7').mkdir(parents=True)
            with patch.object(R.subprocess,'check_output',return_value=b'FIXED\n'), \
                 patch.object(R.subprocess,'run'), \
                 patch.object(R.A.E.p0,'blob',side_effect=lambda root,s,p:(Path(root)/p).read_bytes()), \
                 patch.object(N,'execute_p0',side_effect=AssertionError('No scientific retry')):
                result=R.materializer(ROOT,directory,'FIXED')(7,dict(descriptor=None))
            self.assertEqual(result['status'],'STOP_EXISTING_P0_OUTPUT_NO_RETRY')

    def test_bad_builder_is_not_silently_substituted(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            with self.assertRaisesRegex(ValueError,'UNKNOWN_P0_BUILDER'):
                R.materializer(ROOT,Path(temp)/'p0','SOURCE',builder_id='OTHER')


if __name__=='__main__':unittest.main(verbosity=2)
