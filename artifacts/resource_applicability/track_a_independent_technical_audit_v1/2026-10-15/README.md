# Track A 独立技術監査 bundle

[主要監査資料](../../../../docs/research/track_a_independent_technical_audit_20261015.md)。修正SOURCE **20f89bc32b161ee22e4a1ed709f9e3414c19633a**、対象SOURCE **60eeb51c148007359cf6e8bcf6288e6837a39701**。6件の再現済み制御不具合を最小修正、新22＋旧41＋変更adapter旧12＝75人工PASS。

| 証拠 | 用途 |
| --- | --- |
| [独立担当所見](independent_reviewer_report_v1.md) | 著者の結論を継承せず、独立コード読取・変異test・修正評価。外部取得は親担当と区別。 |
| [findings](findings_v1.json) | 6 CONFIRMED_DEFECT、修正/回帰、未修正0（限定scope）。 |
| [監視red](execution_reproduction_red.log) / [数値red](numeric_reproduction_red.log) | 修正前の人工再現。実計算失敗ではない。 |
| [回帰green](regression_green.log) / [結果receipt](regression_result_v1.json) | 修正後75PASS・実際のwall/RSS。 |
| [test計画](artificial_test_plan_v1.json) | 1process/thread1、home scratch、上限・所有・cleanup。 |
| [closure](source_closure_v1.json) / [旧new差分](source_delta_v1.json) | actual SOURCEから取得した15file、Git blob/bytes/SHA、変更理由。 |
| [GitHub receipt](github_reference_receipt_v1.json) | 固定16fileを実取得してlocal Gitと照合。独立担当所見の14fileに親が旧D0/H8結果2fileを追補。 |
| [候補binding索引](candidate_binding_update_v1.json) | 旧draftを保持した新SOURCE/hash参照。contract/grant/launch/sealではない。 |
| [保全receipt](preservation_receipt_v1.json) | 73旧worktree HEAD/status不変、新実科学/grant/共有変更0。 |
| [監査資料の形式検査](audit_verification_v1.json) / [manifest](artifact_manifest_v1.json) | 新入口のリンク・test名・固定bytes/SHA検査。 |
| [公開receipt](publication_receipt_v1.json) | push可否・remote確認。SOURCE/REVIEW_BUNDLEの40文字SHAは最終報告で指定。 |

数値testは[独立13tests](../../../../tests/test_independent_track_a_numeric_audit_v1.py)、認可・監視・exitは[新9tests](../../../../tests/test_independent_track_a_execution_audit_v1.py)。旧fixtureは人工scaffoldingだけに利用し、新oracleと変異を独立に加えた。実H6/H8 Hamiltonian・保存科学値の再会計・native build/compile・新grantは0。

## 限定testの再現方法

再現も人工のみ。SOURCEのcheckoutで、下記を実行する場合はscratchをhome配下の新exclusive directoryへ作る。これは既存runの再実行ではない。original SOURCEで新testsを使うとred、修正SOURCEではgreenになる。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/independent-technical-audit-20261015
audit_scratch=/home/AbeHiromu/projects/h4-handoff-evidence/20261015/audit-test-manual-01
mkdir -m 700 "$audit_scratch"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 QISKIT_NUM_PROCS=1 TMPDIR="$audit_scratch" \
/usr/bin/python3.12 -P -B - <<'PY'
import unittest, resource, signal, sys, importlib.abc
resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))
resource.setrlimit(resource.RLIMIT_CPU, (120, 122))
resource.setrlimit(resource.RLIMIT_FSIZE, (67108864, 67108864))
def stop(*_): raise TimeoutError('ARTIFICIAL_TEST_WALL_CAP')
signal.signal(signal.SIGALRM, stop); signal.alarm(180)
class BlockScientificImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy','qiskit','cupy','pyscf','trotterlib','trottertracks'):
            raise AssertionError('REAL_SCIENTIFIC_IMPORT_FORBIDDEN:'+fullname)
sys.meta_path.insert(0, BlockScientificImports())
suite = unittest.TestSuite()
for name in ('test_independent_track_a_execution_audit_v1.py',
        'test_independent_track_a_numeric_audit_v1.py',
        'test_final_pre_review_execution_v1.py','test_final_pre_review_h8_driver_v1.py',
        'test_final_pre_review_h6_bridge_v1.py','test_pre_review_h8_d0_adapter_v1.py'):
    suite.addTests(unittest.defaultTestLoader.discover('tests', pattern=name))
sys.exit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
PY
```

## STOP条件

確認済み不具合の修正成功と、実H8/H6の成立・科学的採択・起動認可は別。長C/IO中の5秒以内Python監視は保証しない。quota-enabledのCLIはfresh receipt未接続でSTOP。H6の永続one-shot・bounded runner接続は未実施。実結果・profile・resource fresh bindingと新明示認可は将来の許可された段階に残す。

approved=false、runtime_authorization=false、allowed_cpus=[]、未seal、absolute command=nullを保持し、今回監査1回を終了してmandatory STOP。
