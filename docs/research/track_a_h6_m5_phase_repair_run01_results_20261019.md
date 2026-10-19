# H6 full m5_best：明示basis修正の一軸再計算結果

2026-10-19。**一回再計算は終了したが、compile前後のraw差が固定基準を満たさずSTOPした。修正候補は実H6の問題を解消していない。** 修正後回路と保存wrapper vectorの照合はPASS。最終compile後の照合はFAILで、費用を受理せず、自動retry・追加compile・後続Phaseへ進まない。終了証拠の監査63項目はPASSしたが、これは科学結果のPASSを意味しない。

SOURCE：`02cb8b124302af334d0b02fe53213d3e5a932b78`。plan SHA256：`9c70e90b354537732cc734ad5bf2a9ec5b440d49d900bd17b3006093ed2f8491`。branch：`track-a-h6-m5-phase-repair-run01-20261019`。

[起動・承認・fresh gate](track_a_h6_m5_phase_repair_run01_start_20261019.md)、[修正候補の設計と人工検証](track_a_h6_m5_phase_repair_design_20261019.md)、[今回のsummary](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01/2026-10-19/summary.json)、[終了監査](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01/2026-10-19/end_audit.json)。以下のraw差は一回runner出力の転記で、報告作成時にvectorの再評価はしていない。

## 1. 対象と実行範囲

H6直鎖・隣接1 Å/STO-3G、保存tol-only閾値1e-8・観測DF rank19、T=.8/P=0、full m5_best/prefix19/q2。公式IDは`PF_M5_BEST_20D64C2_BINARY64_V1`。full Hadamard wrapper・symmetric-directional controlの**cosine一軸・保存low状態一つ**だけを使った。13 qubit・ancilla12・layout None・最終測定のみという構造と、保存wrapper/low/referenceのidentityを維持した。

利用者の「実際にH6の再計算まで進めてよい」に基づき、`H6_M5_NATIVE_PHASE_REPAIR_COSINE_PILOT_V1`の新grantを一度だけ消費。新native政策`M5_DF_EXPLICIT_BASIS_PHASE_ONCE_V1`は、隔離したadapterでのみ使用した。共有Qiskit・旧native・旧契約を変更していない。[承認・消費のprojection](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01/2026-10-19/scope_and_authorization.json)は正本plan・有効grant/configの代わりにならない。

実施はnative→明示basis変換、一回compile、二回のStatevector作用。新SCF/DF/state/reference生成、signal、sampling、sine、shot/費用再会計、fit、ランキングは0。旧58受理軸の再取得も0。

## 2. raw検証結果

[gate記録](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01/2026-10-19/raw_gates.json)。閾値は全てfiniteかつ**厳密に1e-9未満**。phase alignment・vector正規化・compile後の位相補正・閾値緩和は行っていない。

| 登録した照合 | raw vector差 | 判定 |
| --- | ---: | --- |
| 修正後回路 vs 保存wrapper vector | 2.326360070803798e-11 | PASS |
| compiled vs 修正後回路 | 1.4720759108055665e-9 | FAIL → 即STOP |
| compiled vs 保存wrapper vector | null | 先行gateのFAILにより未実施 |

3つ目は0やPASSで補完しない。compile後vectorと修正後vectorは判定前に保存済みだが、今回の終了監査で追加の差・phase診断は計算していない。[STOPのprojection](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01/2026-10-19/scientific_STOP.json)に元private STOPのSHAを結合した。

**今回確かめられたのは、明示basis変換後の一状態照合が基準内であることと、その後のopt1 compileを通すと依然として基準超過が残ること。** 全状態・演算子の等価性、原因pass、旧微小差の原因、修正完了は確立していない。人工のparameter付き等価則におけるphase欠落の再現と、実H6の残差の因果関係も未確立のままである。

## 3. 保存回路・compilerの証拠

修正後候補は4,076,368命令で、登録した展開上界と一致した。[lowering receipt](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01/2026-10-19/lowering_receipt.json)に正確なscalar蓄積と一度の丸めの証拠を残した。固定Clifford角にはbinary64のπを用いるため、厳密等価性のcertificateとは扱わない。

[compiled receipt](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01/2026-10-19/compiled_receipt.json)の条件はQiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、num_processes1。診断用のgate countsはRZ 831,662、SX 533,094、X 8,792、CX 367,008、最終measure 1。**費用としての受理行は0**であり、これらの整数countsをshot込み費用や有効な比較結果として扱わない。

新旧でgate countsが一致しても、回路のgate列・角度・global phaseの同一性や費用系列の互換性は証明されない。今回のpipeline identityは`34aea106059ffb586da47aac1fe14cd2101d4f51792b1af152d248f67855739f`。旧compiler option hashだけで旧費用へ混合しない。

626件の[compiler観測記録の所在と限界](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01/2026-10-19/compiler_observation_summary.json)を保持した。記録はpass名・時間・global phase・gate countsであり、未知gate名はなかった。この観測だけで最初の不一致passを特定したとは主張しない。QPY・NPZ・raw callback/resource logはprivateに保存し、公開資料には[bytes/SHA](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01/2026-10-19/artifact_receipts.json)を掲載した。

## 4. 終了・保全監査

17:06:21 JSTに起動し、17:22頃に終了。外部wait4のwallは938.0646356120706秒（約15分38秒）、ru_maxrssは5,169,024 KiB（約4.93 GiB）。[worker/launcherの終了](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01/2026-10-19/completion_witness.json)と[外部exit証拠](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01/2026-10-19/external_exit_witness.json)を照合した。

workerは数値STOPを保存してnative exit0、parentはそのSTOPを検出してlauncher native exit1。**workerのexit0を科学PASSへ読み替えていない。** worker・launcherは全回収済みで、外部signal転送は0。全5stageは各1回で、compile1回・作用2回を超えていない。

[終了監査63項目](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01/2026-10-19/end_audit.json)はSOURCE/承認/plan/grant/marker、使用input/runtime、保存receipt/QPY/vectorのSHA、stage、raw gate、exit/reap、起動snapshotの不変性を確認した。[保全](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01/2026-10-19/preservation.json)では旧証拠2,422fileと使用closure1,964fileが不変。過去のSTOP・既存58軸・AD03保留を維持した。終了監査でQPY/NPZ復元、回路作用、数値再解析、費用会計は0。

一軸の依存順序を守りproductive workerは一つ、監視parentと外部wait4は独立process。host PSI・CPU競合・無関係なworktree更新による強制停止ではない。共有環境・他ユーザー・他worktreeに変更していない。起動時の人工11件・preflight22項目は既存証拠を再利用し、科学結果を得るための再試行はしていない。作者自身の検証であり、独立レビューではない。

## 5. 現在の結論と次の担当

H6 full m5_best/q2のcompile等価性問題は未解決。今回の修正候補を完成版として採用せず、m5_bestの主比較採用AD03と後続Coreは保留を維持する。旧STOP・今回のSTOPはいずれも保持する。H4および既存H6 PR等の58受理軸を、この失敗だけで再計算する指示は出していない。

**ユーザー**：保存されたcallback・関連SOURCEの限定的な静的調査を続けるか、新native政策を見直すか判断する。

**Codex（次の指示がある場合）**：承認された調査・修正計画だけを行う。新compile・追加vector再生・sine・shot/費用会計は別の明示承認と新契約なしに開始しない。全件を再計算することを既定方針にしない。

資料はlocal commitのみ。公開push・後続Phaseの自動起動は0。**MANDATORY STOP。**
