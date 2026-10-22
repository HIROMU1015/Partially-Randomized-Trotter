"""Artificial stdlib tests only; no molecule, runtime import or old runner."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE=Path(__file__).resolve().parents[1]/'scripts/export_h2_outer32_metadata.py'
spec=importlib.util.spec_from_file_location('metadata_export',MODULE)
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


class ProjectionTests(unittest.TestCase):
    def test_numeric_tokens_unchanged(self):
        raw='{"small":1e-09,"negative":-0.00000000001234567890123456789,"zero":-0.0,"i":12345678901234567890123}'
        value=e.loads(raw);encoded=e.encode(value)
        for token in ('1e-09','-0.00000000001234567890123456789','-0.0','12345678901234567890123'):
            self.assertIn(token,encoded)
        self.assertEqual(e.loads(encoded),value)
    def test_new_float_forbidden(self):
        with self.assertRaisesRegex(ValueError,'NO_NEW_FLOAT'):e.encode(0.5)
    def test_number_injection_forbidden(self):
        with self.assertRaises(ValueError):e.encode(e.Number('1,"x":2'))
    def test_roots_and_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            x=e.Export(tmp,Path(tmp)/'public',Path(tmp)/'private/map')
            x.root('/private/execution','EXEC')
            source={'path':'/private/execution/output/a','grant':{'valid':True},'seed_registry':[123],
                'seed_cohort':'EXPLORATION','seed_transpiler':17,'value':e.Number('1e-09')}
            public=x.public(source)
            self.assertEqual(public['path'],'${EXEC}/output/a');self.assertNotIn('grant',public)
            self.assertNotIn('seed_registry',public);self.assertEqual(public['seed_transpiler'],17)
            self.assertIsInstance(public['value'],e.Number)
    def test_original_hash_and_projection_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'original';p.write_text('{"x":1e-09,"seed":67,"grant":{"valid":true}}')
            x=e.Export(tmp,Path(tmp)/'public',Path(tmp)/'private/map')
            item=x.get(p,'/x');self.assertEqual(item['pointer'],'/x')
            x.write('projected.json',item)
            self.assertNotEqual(e.sha_file(p),e.sha_file(Path(tmp)/'public/projected.json'))
            self.assertEqual(x.origins[str(p)]['original_sha256'],e.sha_file(p))
            self.assertEqual(x.get(p)['saved'],{'x':e.Number('1e-09')})
    def test_no_scientific_import(self):
        tree=ast.parse(MODULE.read_text())
        allowed={'argparse','collections','concurrent.futures','hashlib','json','os','pathlib','re','subprocess'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                self.assertTrue(all(x.name in allowed for x in node.names))
            elif isinstance(node,ast.ImportFrom):self.assertIn(node.module,allowed)
    def test_pointer_escaping(self):
        x=e.Export('.','/tmp/public_unused','/tmp/private_unused')
        self.assertEqual(x.pointer({'a/b':[{'x~y':17}]},'/a~1b/0/x~0y'),17)
    def test_projection_idempotent_and_pointer_not_path(self):
        x=e.Export('.','/tmp/public_unused','/tmp/private_unused')
        x.root('/private/execution','EXEC')
        row={'pointer':'/stage','saved':'/private/execution/stagepacks/output/a'}
        first=x.public(row)
        self.assertEqual(first,{'pointer':'/stage','saved':'${EXEC}/stagepacks/output/a'})
        self.assertEqual(x.public(first),first)


if __name__=='__main__':unittest.main()
