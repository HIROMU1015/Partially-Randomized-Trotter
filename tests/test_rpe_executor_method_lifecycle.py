"""Normal factory ownership survives interim method search; no science."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import run_rpe_method_optimizer as CLI
from test_rpe_optimizer_concurrent import fixture


class MethodLifecycle(unittest.TestCase):
    def exercise(self,distributed):
        with tempfile.TemporaryDirectory()as temporary:
            problem,context,routes=fixture(['PR_PAULI','PR_DF'],1)
            options=dict(scheduler='concurrent',max_concurrent_jobs=2)
            if distributed:options['distributed']={'port':0}
            spec=dict(executor=options,max_new_q3_per_method={'PR_PAULI':0,'PR_DF':0},
                      science_approved=False,price_validation={'store':str(Path(temporary)/'models.sqlite')})
            executor,optimize=CLI.make_executor(ROOT,temporary,routes,{},problem,spec)
            try:
                self.assertIn('finish',executor.__dict__)
                for method in ['PR_PAULI','PR_DF','PR_PAULI']:
                    view=executor.for_method(method)
                    view.finish()
                    self.assertFalse(executor.state.closed)
                    self.assertFalse(executor.state.driver.closed)
                    if distributed:
                        self.assertIsNotNone(executor.rpc_server)
                        self.assertTrue(executor.rpc_thread.is_alive())
                for _ in range(2):
                    result=optimize(ROOT,problem,context=context,methods=list(problem['methods']),
                        initial='all',max_new=spec['max_new_q3_per_method'],
                        executor=executor,science_approved=False)
                    self.assertEqual(set(result),set(problem['methods']))
                    self.assertFalse(executor.state.closed)
                # The same owner can still reserve an artificial preparation
                # job after a method has drained; there is no scientific leaf.
                if distributed:
                    request=dict(method='ARTIFICIAL',route={'kind':'preparation'},
                        preparation=dict(stage='signal',artificial=True,
                            command=['ARTIFICIAL_ONLY'],outputs=['receipt.json']))
                    job=executor.submit_preparation(request,approved=True,budget=1)
                    self.assertEqual(executor.store.row(job)['state'],'QUEUED')
            finally:executor.finish()
            self.assertTrue(executor.state.closed)
            self.assertTrue(executor.state.driver.closed)

    def test_concurrent_method_drain_does_not_close_owner(self):self.exercise(False)
    def test_distributed_method_drain_does_not_close_queue_or_preparation(self):self.exercise(True)


if __name__=='__main__':unittest.main()
