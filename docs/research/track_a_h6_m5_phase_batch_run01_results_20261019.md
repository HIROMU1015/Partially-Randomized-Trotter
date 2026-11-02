# H6 full m5_best：正確な位相蓄積を用いた一回再計算の結果

2026-10-19。**再計算は終了したが、compiled vs保存変換後vectorのraw差1.4322891926554257e-9が、固定1e-9未満を満たさずSTOPした。今回の位相蓄積修正だけでは解消していない。** 自動retry・追加compile・費用受理は0。旧STOPと58受理軸を保全し、local commit後mandatory STOP。pushは行わない。

SOURCE：`96cd7cd4c3aff6558d8a6b72e8e6074dcb39a950`。plan SHA256：`96b278e817cf97ec03aedc208d5d0903dfae10a297c8449830a3cafe303e794c`。branch：`track-a-h6-m5-phase-batch-run01-20261019`。

[起動・承認・実装・scope](track_a_h6_m5_phase_batch_run01_start_20261019.md)、[前回の結果](track_a_h6_m5_phase_repair_run01_results_20261019.md)、[今回のsummary](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01/2026-10-19/summary.json)、[終了監査](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01/2026-10-19/end_audit.json)。以下の実数は承認済みrunner出力の転記で、終了監査でvectorを再評価していない。

## 1. 対象と承認済み実施量

H6直鎖・隣接1 Å/STO-3G、保存tol-only1e-8 DF19、full m5_best/p19/q2、T=.8/P=0、symmetric-directional/full Hadamard cosine、保存low状態一つ。公式IDは`PF_M5_BEST_20D64C2_BINARY64_V1`。Qiskit1.3.0・rz/sx/x/cx・opt1・seed17・num_processes1を維持し、project内の新compiler-policy `M5_OPT1Q_EXACT_PHASE_BATCH_V1`で実行した。

利用者の「問題だと思われるところを再計算して」に基づき、新SOURCE・plan・grantを一回使用。前回の変換後QPY/vector/最初のPASSを、旧STOPを保持した別受入条件と保存SHAへ結合して再利用した。重複するlowering/変換後作用は0、新compile一回・compiled回路の作用一回のみ。新signal/sampling/SCF/DF/state/reference生成、sine、shot/費用会計、fit、ランキングは0。

## 2. 固定raw検査

[gate結果](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01/2026-10-19/raw_gates.json)。各差はfiniteかつ厳密に1e-9未満を要求し、phase alignment、vector正規化、compile後の補正、閾値緩和は行わない。

| 照合 | 保存されたraw差 | 扱い・判定 |
| --- | ---: | --- |
| 変換後 vs 元wrapper保存vector | 2.326360070803798e-11 | 前回PASSをSHA結合して再利用、今回の新作用0 |
| compiled vs 保存変換後vector | 1.4322891926554257e-9 | 新たに評価、FAIL → STOP |
| compiled vs 元wrapper保存vector | null | 先行FAILにより未実施 |

前回同じ位置の差は1.4720759108055665e-9だった。今回も基準超過であり、小さくなったことを受入PASSとしない。未実施の第3検査を0やPASSで埋めない。[数値STOP](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01/2026-10-19/scientific_STOP.json)と、判定前のcompiled QPY/vectorを保存した。費用の受理行は0。

## 3. 変更箇所が実行された証拠と限界

[phase ledger全88行](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01/2026-10-19/phase_batches.jsonl)、[ledger要約](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01/2026-10-19/phase_batch_summary.json)。元のaccelerated Rust再合成器を使い、502,636回の位相増分を各pass内で正確な有理数として蓄積した。置換loop内のfloat化はguardで拒否し、全88passが終了時一回の丸め・通常数値phaseへの復帰を完了した。途中の増分値はbit列SHAで記録し、個別値の全件logは作っていない。

新実装の実行を確認できたが、実H6の残差は残った。**この実装で逐次加算をバッチ化する変更だけでは十分でない。** 内部matrix積・Euler位相生成・ZSXXのphase算術・局所角度簡約等の寄与は未分離であり、そのどれが原因かは認定しない。小さい人工全演算・制御付き演算のPASSを、長い実H6回路のPASSに代用しない。

保存[compiler receipt](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01/2026-10-19/compiled_receipt.json)の診断用countsはRZ831,662、SX533,094、X8,792、CX367,008、measure1。gate countsが以前と同じでも、全gate列・角度・位相のbit一致や費用系列の互換性は確定しない。compiled QPYは92,549,282 bytes、SHAは同receiptに結合し、`UNVALIDATED_DIAGNOSTIC_NOT_FOR_ACCOUNTING`のままとする。QPY/NPZはprivate、[artifact bytes/SHA](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01/2026-10-19/artifact_receipts.json)だけを公開資料へ含めた。

## 4. 終了・保全

19:00:52 JST起動、19:08頃終了。外部wait4のwallは439.7739984281361秒（約7分20秒）、ru_maxrssは2,918,240 KiB（約2.78 GiB）。productive worker一つ、監視parentと外部wait4は別process。保存PASS再利用で余分なloweringと約4百万gateの再生を避けた。

[worker/launcher終了](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01/2026-10-19/completion_witness.json)、[外部native exit](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01/2026-10-19/external_exit_witness.json)。workerは数値STOPを保存してexit0、launcherはSTOPを検出してexit1、いずれも回収済み。worker exit0を科学PASSに読み替えない。外部signal転送は0で、資源・PSIによる中断ではない。

終了監査48項目は、SOURCE/入力/runtime/承認/消費済みgrant、3stageの一回性、初回gateのreuse、数値STOPと未実施値、88phase ledger、QPY/vectorのSHA、exit/reap、起動snapshot保全を確認した。これは証拠の監査PASSであり、科学結果のPASSではない。[保全](../../artifacts/resource_applicability/track_a_h6_m5_phase_batch_run01/2026-10-19/preservation.json)は旧2,570file不変、使用closure2,011file一致。終了監査のQPY/NPZ復元、vector/phase/determinant再解析は0。起動前の人工13件・preflight22項目は再利用した。

旧SOURCE・STOP・58軸・grant/ledger/marker、V1/V2等の他作業、共有Qiskit/SymEngine、他worktreeを変更していない。m5_best主比較のAD03と後続Coreは保留のまま。新policyの費用を旧結果へ混ぜない。

## 5. 次の担当

**ユーザー**：再合成内部のphase生成・matrix/Euler計算・局所簡約を、最小人工反例と局所検証で切り分ける調査へ進むか判断する。同じ全H6 compileを修正なしに繰り返すことは勧めない。

**Codex（次の指示がある場合）**：上記の限定調査・修正計画を行い、どの処理を変える必要があるか絞る。新compile・回路再生・追加候補・sine・費用受理・閾値変更へ自動で進まない。

local commit後**MANDATORY STOP**。公開push・後続Phaseは0。
