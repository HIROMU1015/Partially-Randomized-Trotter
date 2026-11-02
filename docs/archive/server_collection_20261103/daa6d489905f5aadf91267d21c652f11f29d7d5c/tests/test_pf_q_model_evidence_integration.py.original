"""Artificial documentary/scanner checks; no scientific model or accounting API."""
import ast
import csv
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import integrate_pf_q_model_evidence as doc


class DocumentaryTests(unittest.TestCase):
    def test_internal_path_findings_do_not_republish_literal(self):
        raw=b'example '+b'/home/'+b'example_user/project/private_location'
        finding=doc.scan_content(raw,'.txt')
        self.assertEqual(len(finding['internal_paths']),1)
        self.assertNotIn('example_user',json.dumps(finding))
        self.assertEqual(finding['internal_paths'][0]['category'],'USER_HOME_ABSOLUTE')

    def test_credential_pattern_is_reported_as_hash_only(self):
        raw=b'ghp_'+b'A'*36
        finding=doc.scan_content(raw,'.txt')
        self.assertEqual(len(finding['credential_patterns']),1)
        self.assertNotIn(raw.decode(),json.dumps(finding))

    def test_compressed_PDF_stream_is_scanned(self):
        text=b'/tmp/'+b'example-document'
        raw=b'%PDF-1.4\nstream\n'+zlib.compress(text)+b'\nendstream\n'
        finding=doc.scan_content(raw,'.pdf')
        self.assertEqual(finding['internal_paths'][0]['layer'],'pdf_FlateDecode_stream_0')

    def test_PNG_text_metadata_is_scanned(self):
        data=b'Comment\0'+b'/tmp/'+b'example-metadata'
        raw=b'\x89PNG\r\n\x1a\n'+struct.pack('>I',len(data))+b'tEXt'+data+b'\0'*4
        finding=doc.scan_content(raw,'.png')
        self.assertIn('png_tEXt',finding['inspected_layers'])
        self.assertTrue(any(p['layer']=='png_tEXt' for p in finding['internal_paths']))

    def test_append_preserves_historical_CSV_bytes_and_cells(self):
        old='id,text\nA,"旧値,そのまま"\n'.encode()
        with tempfile.TemporaryDirectory(prefix='pf-document-test-') as directory:
            root=Path(directory);(root/'old.csv').write_bytes(old)
            with patch.object(doc,'ROOT',root),patch.object(doc,'OUT',root):
                doc.append_csv('old.csv','new.csv',[dict(id='B',text='補足')])
            raw=(root/'new.csv').read_bytes()
            self.assertTrue(raw.startswith(old))
            self.assertEqual(list(csv.DictReader(io.StringIO(raw.decode())))[1],dict(id='B',text='補足'))

    def test_wrong_accepted_count_is_rejected_without_reanalysis(self):
        with patch.object(doc,'read',return_value={'all100':{'scope_conditions':99}}):
            with self.assertRaisesRegex(ValueError,'ACCEPTED_SUMMARY_LITERAL_MISMATCH'):doc.verify_saved()

    def test_document_script_has_no_scientific_import_or_evaluation(self):
        tree=ast.parse(Path(doc.__file__).read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [alias.name for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names]
        self.assertFalse(any(x.startswith(('numpy','scipy','fractions','trotterlib','run_h6','pf_selection')) for x in imports))
        forbidden={'eval','exec','compile','fit','bootstrap','weighted','ranking','replay_pf'}
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in forbidden for n in ast.walk(tree)))


if __name__=='__main__':unittest.main()
