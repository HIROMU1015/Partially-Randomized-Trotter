"""Use both production metadata entry points with artificial signal indexes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import run_pr_pauli_candidate_signals as pauli

ROOT = Path(__file__).resolve().parents[1]


class SavedPreparationSignalIndex(unittest.TestCase):
    def make_workspace(self, root):
        scripts = root / 'scripts'
        package = root / 'src/trottertracks'
        package.mkdir(parents=True)
        scripts.mkdir()
        (package / '__init__.py').touch()
        subprocess.run(['git', 'init', '-q', str(root)], check=True)
        return scripts, package

    def test_pauli_fixed_entry_reuses_original_order_and_validates_added_packet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'workspace'
            scripts, package = self.make_workspace(root)
            (package / 'pauli_signal_entry.py').write_text("""
from pathlib import Path
from types import SimpleNamespace
S3=SimpleNamespace(P1_KERNEL='fixed-kernel')
KERNEL_FILES=('artificial.py',)
def saved_index(root, packets):
    with (root/'validations').open('a') as out: out.write(packets[0]['path']+'\\n')
    return {'physical':packets[0]['path']}
def read_ref(root, loc):
    return {'packets':[{'rounds':[]}]}
""")
            (scripts / 'run_pauli_signal_entry.py').write_text("""
import json
from pathlib import Path
import sys
from trottertracks import pauli_signal_entry as E
root = Path(__file__).resolve().parents[1]
print(json.dumps(E.saved_index(root, [dict(commit='a'*40, path=n) for n in sys.argv[1:]])))
""")
            temp = Path(temporary)
            for i, names in enumerate((['A'], ['A'], ['A','B'], ['B','A'])):
                log = temp / f'entry-{i}.log'
                self.assertEqual(pauli.entry(root, sys.executable, names, log), 0, log.read_text() if log.exists() else '')
                self.assertEqual(json.loads(log.read_text()), {'physical':names[0]})
            self.assertEqual((root / 'validations').read_text().splitlines(), ['A','B'])

    def test_pf_original_and_plan_use_same_compact_locator_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            root = tmp / 'workspace'
            scripts, package = self.make_workspace(root)
            (package / 'pf_evidence_preparation.py').write_text("""
from pathlib import Path
def mean_pool(root):
    with (Path(root)/'validations').open('a') as out: out.write('verified\\n')
    return {('H4','B0I',7,0.1):[dict(source=dict(commit='a'*40,path='original-signal'),value={'large': 'metadata'})]}
def make_plan(root,cfg,source):
    return mean_pool(str(root))[('H4','B0I',7,0.1)][0]['source']
""")
            (scripts / 'prepare_pf_candidate.py').write_text("""
import argparse,json
from pathlib import Path
from trottertracks import pf_evidence_preparation as E
ROOT=Path(__file__).resolve().parents[1]
STUDY='study'
MODULE='unused'
def load(path): return json.loads(Path(path).read_text())
def write(path,value): Path(path).write_text(json.dumps(value))
def native_row(*args): pass
def comparison_method(*args): pass
def publish(*args): pass
def head(): return 'ARTIFICIAL'
def source_gate(): return {'commit':head()}
def selected_configuration(item): return item['configuration_body']
def preflight(plan,out): assert plan['path']=='original-signal'
def main():
    p=argparse.ArgumentParser()
    for k in ('request','output','mode'): p.add_argument('--'+k)
    args=p.parse_args()
    write(args.output,E.make_plan(ROOT,{},source_gate()))
""")
            subprocess.run(['git','add','.'], cwd=root, check=True)
            subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],
                           cwd=root, check=True)
            request = tmp / 'REQUEST.json'
            request.write_text(json.dumps(dict(selected=[dict(configuration_body=dict(id='C'))],
                registry_file=str(tmp/'REGISTRY.json'), output_root=str(tmp/'unused'))))
            cache = tmp / 'cache'
            owner = tmp / 'OWNER.json'
            subprocess.run([sys.executable,'-B',str(ROOT/'scripts/check_pf_preparation_owner.py'),
                '--provider-root',str(root),'--request',str(request),'--output',str(owner),'--signal-index',str(cache)],check=True)
            self.assertEqual(json.loads(owner.read_text())['rows'][0]['status'],'ORIGINAL_OWNER_PREFLIGHT_PASS')
            clone = tmp / 'owned'
            subprocess.run(['git','clone','-q','--shared',str(root),str(clone)],check=True)
            output = tmp / 'PLAN.json'
            subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_pf_preparation_stage.py'),
                '--provider-root',str(clone),'--input',str(request),'--output',str(output),'--plan-signal-index',str(cache)],check=True)
            self.assertEqual(json.loads(output.read_text()),dict(commit='a'*40,path='original-signal'))
            self.assertEqual((root/'validations').read_text().splitlines(),['verified'])
            self.assertFalse((clone/'validations').exists())
