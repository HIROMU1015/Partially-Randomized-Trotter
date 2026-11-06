# Exact Fraction engine O1/O2：人工実装・検証結果

2026-11-07 JST、SERVER1 `GPUA100-gleap-01`。
**独立O1/O2 prototype・人工correctness・人工benchmark完了。既存科学engine未変更。公開照合後mandatory STOP。**

## 結果

1. **O1は人工入力で旧denseとexact同値。** exact零だけを省く疎/対角contractionを実装した。全entry非零は元dense関数へfallbackする。
2. **O2追加後もexact同値。** `(state_dimension, ordered support)`の不変index計画をprivateな遅延LRUで再利用。cold/warm/eviction・再生成を検査した。
3. **最終11 tests PASS。** 1,008局所条件と10人工gate列の120中間更新を3方式で全振幅比較。旧Born/53-bit threshold・固定wordのbit一致、validation・非変異も確認。有限人工testで科学的認証が完了したとは扱わない。
4. **人工性能は構造依存。** 以下は前処理込みcold、12 gate列＋旧Born＋旧thresholdのwall中央値（6回、方式順を全6順列で入替）。

| 人工fixture | Dense ms | O1 ms | O1+O2 ms | Dense/(O1+O2) |
|---|---:|---:|---:|---:|
| diagonal_k3（人工256振幅） | 149.872 | 20.850 | 18.548 | 8.080 |
| mixed_sequence（人工128振幅） | 51.650 | 16.168 | 15.506 | 3.331 |
| dense_k1（人工32振幅） | 7.124 | 7.315 | 7.311 | 0.975 |

疎/対角等8 fixtureでは、coldでDense/O1が1.214–7.188、Dense/(O1+O2)が1.304–8.080、O1/(O1+O2)が1.043–1.129だった。**O2の追加改善を中央値で観測したが、dense fallbackの2 fixtureに改善はない。** 全10 fixtureとmin/max・CPU・warm・前処理時間を [BENCHMARK](EXACT_ENGINE_O1_O2_BENCHMARK.md) に残し、不利な条件を除外していない。H4全engineの倍率ではない。

5. **memoryは増加。** coldのPython allocation peakはO1+O2でdense比+1,800～13,288 bytes。O2 cacheは0–3 entry（0–384 index）を使用した。前処理込みでも上の時間改善は残ったが、bytesと秒を単一尺度に合算していない。fresh subprocessのRSS high-waterは全30測定で27,262,976 bytesとなり、方式差を識別できなかった。多worker・実科学のmemoryは未測定。

## 再利用SOURCEと成果物

設計正本 `fb28b6a287803ce63a51a1513a57828302003090`、旧kernel `560c6092bfb5e71af8eedc4a4be5416922764f98`。元 `ZERO,multiply,add,apply_exact,norm_squared,born_probability,threshold_count` と補助range検査を、[oracle.py](oracle.py) でhash照合した元ASTのまま再利用した。科学依存をimportしないための限定loaderで、既存関数の重複実装ではない。

| 成果物 | 用途 |
|---|---|
| [prototype.py](prototype.py) | O1、O2、exact dense fallback、validation。科学engineから参照されない独立module |
| [fixtures.py](fixtures.py) / [test_o1_o2.py](test_o1_o2.py) | 手書き人工有理数とstrict correctness |
| [benchmark.py](benchmark.py) | 固定10 fixture、wall/CPU、cold/warm、前処理、fresh-process memory |
| [render_reports.py](render_reports.py) | 保存rawの整合確認と報告表生成。kernel再実行なし |
| [CORRECTNESS](EXACT_ENGINE_O1_O2_CORRECTNESS.md) | 数学的同値性、人工範囲、技術修正、認証との区別 |
| [BENCHMARK](EXACT_ENGINE_O1_O2_BENCHMARK.md) | 全fixtureの比較と未測定事項 |
| [raw_run01](raw_run01/) | SOURCE、環境、測定前PLAN/全入力FIXTURES、11-test log、540 timing、前処理、30 memory、SUMMARY、STOP |
| [RAW_REPORT_CHECK.json](RAW_REPORT_CHECK.json) | raw→SUMMARY再導出、SOURCE hash、3方式threshold整合の確認 |

全入力は人工有理数で、実H4 state/gate/LAW/catalogから作っていない。Python3.12.3・stdlibのみ、科学依存module import0。共有venv/package/cacheを変更していない。初回O1でテスト件数の期待値だけを修正した経緯もCORRECTNESSに記載し、初期logを保持した。

## 科学認証・次の判断

実H4で限定した性能検証を設計する価値はある。特に対角/疎gateを対象に、準備費・cache memoryを含む改善が残るかを確認したい。ただし**人工性能を実H4の改善と認定していない**。実gate構造比率・bit長分布・build/hash/Bornの時間比・worker memory・全体wallへの効果はUNKNOWN。

実科学へ進むには、新しい独立SOURCEと環境/inputを固定し、同じcertified matrix値・state・gate順・qubit順・phase/control・Born/thresholdとidentity照合を保つ接続を用意する。有限の事前指定path/軸について旧新の全振幅/p/t、前処理込みwall/memoryを比較するscopeを具体化し、実H4 state作用/Born/必要な実build/compileのcall範囲とSTOPについて別途明示承認を受ける。RNG/測定/追加SCV runまで自動的に含めない。旧qualificationの再実行や承認証明書・grant発行を標準要件にしない。

保存local誤差証明は同じmatrixと誤差意味論を保つ場合に再利用し得るが、新prototypeへ旧certificateが自動移転するわけではない。元`b_engine`・証明・PROFILE_LOCKはそのまま。既存科学engineへの組込みは今回行わない。

## 固定資料・保全・STOP

正式共通指示2026-10-25の§8.1/8.2/8.4/10/10.3、AGENTS、PROJECT_MAP、研究概要と前回設計4文書を確認した。engine認証 `fdf00cee3c87d804845497afc89ac6a71f3b3f59`、752結果 `3c8e55a2728f86a1eda17a2e0105050a637e7c86`、224完了 `ba483ab3038693048e58830e39a676493a892053` の既存契約/STOPを確認した。保存の科学結果を再集計・再ランキングしていない。

保全対象の科学条件は保存linear H4/1.00 Å/STO-3G/legacy DF12/8 system qubits/全4電子sector、`L_D3,m0/q1,δ.02,r1,K2,coefficient_atol0,X/Y`、N1 actual finite-grid Q/selected full、N2保証state、理想gate・保存DFモデル。これは今回benchmarkの入力条件ではない。

専用branch `codex/exact-engine-o1-o2-artificial-20261107`、worktree `/tmp/exact-engine-o1-o2-artificial-20261107`、baseは設計固定commit。新しいsource/test/runner/raw/docsを本directoryにまとめ、既存の4 Markdown indexから入口を追記する。旧Python SOURCE・SCV実装・test・science runner・証明・PROFILE_LOCKは変更しない。科学的判断/科学結果の変更ではないため、研究正本・dated研究ノート・科学manifestを更新していない。

元mainの開始時HEAD/diff/statusと既存2,136ファイルのhash snapshotを別の本人用temporary領域へ保存した。人工実装・benchmark・文書作成後の照合でdiff/status・全2,136 hashの一致を確認し、公開後も最終照合する。旧worktree/dirty/untracked、科学raw/grant/ledger/marker/STOP、他調査branchを変更していない。process attach/profile、他jobの操作、共有OS/scheduler/cgroup/venv/cache変更はなし。CPU/RAM/科学workerへ新たな低い人工上限を設けていない。

**未実施**：実H4 state evolution/Born・実gate投入・実path生成・実build/compile・科学RNG/sampling/measurement・旧752/224再実行・科学結果再解析・engine再認証・O3以降・科学engine差替え・重要科学レビュー・C1解凍。

今回の新SOURCE/人工test/benchmark/raw/Markdownと必要な入口追記だけをcommit・公開し、remote SHA、SOURCE blob、本文、directory tree、rawと表、原本保全を照合する。最終SHAは最終chatに記載する。**照合完了後mandatory STOP。実H4性能検証や組込みへ進まない。**
