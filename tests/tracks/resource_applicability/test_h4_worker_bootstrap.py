"""CLI import regression only: no permit, owned worker, or science job."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).absolute().parents[3]
WORKER=ROOT/'src/trottertracks/resource_applicability/h4_geometry/workers.py'

class WorkerBootstrapTest(unittest.TestCase):
    def test_real_spawn_command_reaches_closed_private_entry(self):
        tree=ast.parse(WORKER.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess' and n.func.attr=='Popen']
        self.assertEqual(len(calls),1)
        values=calls[0].args[0].elts
        command=[]
        for value in values[:4]:
            if isinstance(value,ast.Name) and value.id=='PYTHON':
                command.append(sys.executable)
            elif isinstance(value,ast.Name) and value.id=='__file__':
                command.append(str(WORKER))
            elif isinstance(value,ast.Constant):
                command.append(value.value)
            else:
                self.fail('unexpected real worker command prefix')
        result=subprocess.run(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,timeout=10,check=False)
        self.assertEqual(result.returncode,1)
        self.assertEqual(result.stdout,b'')
        self.assertEqual(result.stderr.decode().strip(),'private owned-worker entry only')

if __name__=='__main__':unittest.main()
