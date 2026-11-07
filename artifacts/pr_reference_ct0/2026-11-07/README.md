# CT-0保存結果（2026-11-07 JST、SERVER 2）

[REPORT](../../../docs/research/pr_reference_ct0_20261107/REPORT.md)、[coverage](../../../docs/research/pr_reference_ct0_20261107/KEY_COVERAGE.md)、[phase/precision](../../../docs/research/pr_reference_ct0_20261107/PHASE_PRECISION_FEASIBILITY.md)。

- [SOURCE](../../../src/trottertracks/pr_reference_ct0.py) / [runner](../../../scripts/run_pr_reference_ct0.py)：保存データ専用。旧runner/acquire/compileを呼ばない。現在はSTOP済みで再実行しない。
- `SOURCE_INPUTS.json`：基準commit、実際に読んだ固定Git blobのSHA256/bytes、新SOURCE hash、再利用規則。
- `KEY_COVERAGE.json.xz`：全V3 signed→positive/precision/role照合、missing、symbolic、保存word hash/guard/T field、実inverse Dの補助照合。出力JSONの可逆xz圧縮のみを公開時に行った。
- `PHASE_SUPPORT.json`：B′の有限π反復加算supersetと保存record照合、Referenceの全support角度境界。word価格境界はnull。
- `PRECISION_FEASIBILITY.json`：全18round・両arm・両軸の旧disk/N、追加diskの正beta余地/固定N余地、worst-case RZ数、一様guardの十分境界。
- `TECHNICAL_TESTS.json`：人工9検査。key符号、2πを消さないこと、改変spec/hash/PASS/guard/Tの拒否、disk幾何の単調性。架空schema recordはvalidator用で、物理合成wordの生成・性能検証ではない。
- `ENVIRONMENT.json`：専用venv・affinity・資源pressureの観測。共有設定変更なし。
- `RUN.log`：最終保存解析のstdout。`STOP.json`：範囲完了・全support価格未資格によるmandatory STOP。

実行：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-c-h4-parallel-pilots-20261106/.venv/bin/python scripts/run_pr_reference_ct0.py`。同じ出力のSTOPがあれば拒否する。最初の読取試行は4 legacy recordの`guard_method`欠損で停止し、欠損を未資格分類する最小schema修正後、同じ承認範囲で最終解析を完了した。新規合成や科学条件追加を伴うretryではない。

実資源：affinity64、MemAvailable約471GiB、memory/I/O PSI avg10=0を開始時に観測。主処理は小さな保存archiveの読取・有理比較・36条件の区間二分評価で、worker poolは不要。恣意的resource capや共有環境・他processへの変更なし。新build/compile/synthesis/word生成/guard再計算/sampling/simulation/T価格取得/precision最適化/新shot schedule/総T/T-depthは全て0。

**mandatory STOP。** 旧SOURCE・結果・grant・ledger・marker・STOP不変更。次担当GPT／ユーザー。独立レビュー・外部再現を称さない。
