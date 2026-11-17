"""Method drain preserves the normal controller and optional validation service."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import run_rpe_method_optimizer as CLI
from trottertracks import rpe_price_validation as Validation
from trottertracks import rpe_q3_coordinator as Coordinator


class PriceValidationLifecycleTests(unittest.TestCase):
    def test_method_drains_leave_controller_open_then_main_closes_service_once(self):
        source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        for distributed in (False, True):
            with self.subTest(distributed=distributed), tempfile.TemporaryDirectory() as directory:
                service = Mock()
                options = dict(scheduler='concurrent', max_concurrent_jobs=2)
                if distributed:
                    options['distributed'] = dict(bind='127.0.0.1', port=0)
                spec = dict(methods=['PR_PAULI','PR_DF'], science_approved=False,
                    max_new_q3_per_method=dict(PR_PAULI=0, PR_DF=0), executor=options)
                server = SimpleNamespace(server_address=('127.0.0.1',8763), shutdown=Mock(), server_close=Mock())
                server_thread = SimpleNamespace(join=Mock())
                with patch.object(Validation, 'Service', return_value=service), \
                     patch.object(Coordinator, 'serve', return_value=(server,server_thread)), \
                     patch.object(CLI, 'head', return_value=source):
                    executor, _ = CLI.make_executor(ROOT, Path(directory),
                        dict(PR_PAULI=[], PR_DF=[]), {},
                        dict(accuracy=dict(e=1), target=dict(energy_ha=0), rpe=dict(M=1)), spec)
                    try:
                        views = [executor.for_method(name) for name in spec['methods']]
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            list(pool.map(lambda view:view.finish(), views))
                        self.assertFalse(executor.state.closed)
                        self.assertFalse(executor.state.driver.closed)
                        self.assertFalse(executor.state.monitor_stop.is_set())
                        service.close.assert_not_called()
                        executor.finish()
                        self.assertTrue(executor.state.closed)
                        self.assertTrue(executor.state.driver.closed)
                        executor.finish()
                        service.close.assert_called_once_with()
                    finally:
                        executor.finish()


if __name__ == '__main__':
    unittest.main()
