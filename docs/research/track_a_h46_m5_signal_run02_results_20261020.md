# H4/H6 m5：修正版signal run02の正常終了

2026-10-20。**修正版で固定88候補のsignal評価を正常終了した。H4は37新規＋3保存再利用、H6は45新規＋3保存再利用で、合計82新signal＋6再利用。5精度・両軸の経験的適格性880行を保存した。両worker・launcherはexit0で回収済み、終了監査47項目を通過した。新compile・sampling・shot/費用会計は行わず、mandatory STOPする。**

SOURCE：`bec018c64e7a8458dbf13b1c0e9655d628292857`。準備・STOP報告commit：`adaa094148ce8671bb9a4811f988fe9646580851`。branch：`track-a-h46-m5-signal-empty-tail-fix-20261020`。ユーザーの「修正を行って再実行して」を、[端点修正報告](track_a_h46_m5_signal_run01_stop_and_fix_20261020.md)に対する新しい一回実行の承認として結合した。元run01のgrantを再使用せず、元STOPは保持した。

## 実行条件と実施範囲

保存線形H4/H6直鎖・隣接1.0 Å、STO-3G、tol-only1e-8・rank自動決定の入力を使用した。観測DF rankはH4=10/H6=19で、rankを生成引数として指定していない。T=.8、P=0、通常signed bias、allowance倍率1、ε=.05/.01/.005/.001/.0005、軸別ε/√2。両系の生成規則全体の同一性やサイズの因果効果は今回の結論に含めない。

| 対象 | prefix | q | 候補数 |
| --- | --- | --- | ---: |
| H4 PR：S2/Yoshida4/m5 | 3 | 1/2/4/8 | 12 |
| H6 PR：同3公式 | 5 | 同上 | 12 |
| H4 m5 B0/B0I | 3/8/9 | 同上 | 24 |
| H4 m5 full | 10 | 同上 | 4 |
| H6 m5 B0/B0I | 5/7/11/15 | 同上 | 32 |
| H6 m5 full | 19 | 同上 | 4 |
| 合計 | | | 88 |

PRはK4、`ABS_TIME_PER_OUTER=32`。m5は`PF_M5_BEST_20D64C2_BINARY64_V1`の11段signed S2と既存の絶対時間整数配分を使用した。相対負時間を保持し、係数の再最適化や公式の置換をしていない。隣接prefix等のoptional20、K2、q16、H8は対象外。

再利用した6raw signalは、元pilotの各系B0/B0I anchorとfullのq2である。元raw inventoryのSHA、入力、reference、runtimeへ結合し、保存値をそのまま使った。旧pilot trajectory・旧n8 ordinary-control費用を新outer32標本へ混ぜていない。full-H referenceも各系の保存1件を読み込み、新たなreference作用・SCF・DF・state生成は行っていない。

今回はsignal-onlyであり、symmetric-directionalのinput/native policyと保存preparationの同一性を確認した段階である。Qiskit1.3.0/rz,sx,x,cx/opt1/seed17のcompiler bindingは保持したが、このq/prefix格子での新wrapper compile・compiled state比較は0件。signal完了を、回路費用取得や新格子のnative検証完了とは呼ばない。

## 終了・並列化の記録

新承認plan SHA：`85bb596c3fce5349d2066d05c5dd1a0cbeebfa49e1f3a35db30db729ac1d8f08`。SOURCE・入力・runtime・承認receiptの2,065ファイルを起動直前・終了時に照合した。

| 記録 | 結果 |
| --- | --- |
| fresh資源 | 空きCPU約61.51 core相当、実効空きRAM 502,902,927,360 bytes、disk 477,557,866,496 bytes |
| 並列構成 | H4/H6の2 process、各系30候補thread、ローカルBLASは1thread |
| 再利用・高速化 | snapshot/referenceは各系1回、full preparation/oracleを系内共有、prefix別contextをcache。5精度ごとの再作用なし |
| H4結果 | 40候補完了＝37新規＋3再利用 |
| H6結果 | 48候補完了＝45新規＋3再利用 |
| 候補ledger・結果 | 登録88件・結果88件。各結果SHAと候補/plan/contextを照合 |
| campaign | `SIGNAL_COMPLETE_MANDATORY_STOP`、errorsなし |
| science worker | 両方exit0・回収済み |
| launcher | native exit0・外部wait4で回収済み |
| launcher内wall | 57.6757637秒 |
| 外部supervisor wall | 61.4077753秒 |
| 外部wait4のru_maxrss | 1,500,100 KiB。全workerの同時RSS合計としては扱わない |

前回STOPしたfull-prefix contextを両系とも通過し、登録候補の評価まで完了した。host PSIや無関係なworktreeの状態を停止条件にしていない。今回の実時間と並列稼働は確認できたが、同一条件の直列benchmarkを実施していないため、何倍高速化したかは主張しない。

## 保存されたsignalと適格性

全88候補にreference・exact-tail・corrected/raw signal、通常signed biasと成分、log_B、closure、経験的u/margin/headroomを保存した。以下は**保存済みの軸別ラベルの件数**で、適格性の再計算・新統計推定・最良選択ではない。

| ε | H4 適格/不適格（80軸行中） | H6 適格/不適格（96軸行中） |
| --- | ---: | ---: |
| .05 | 72 / 8 | 84 / 12 |
| .01 | 63 / 17 | 63 / 33 |
| .005 | 63 / 17 | 62 / 34 |
| .001 | 60 / 20 | 58 / 38 |
| .0005 | 58 / 22 | 57 / 39 |

`UNDETERMINED`の保存軸行は0。**軸単位の件数であり、両軸とも適格な候補数ではない。** 不適格な行も全て保持し、候補の除外・差替え・再探索はしていない。全88候補の技術的完了は、全候補が精度適格であることを意味しない。

allowanceは[準備報告](track_a_h46_m5_signal_readiness_20261020.md)の`M5_SIGNAL_EMPIRICAL_OBSERVATIONS_V1_PROPOSED`を、今回のrunに限って承認された経験的診断として用いた。保存データのpolicy名・statusは変更していない。compiled vector閾値1e-7はuへ流用していない。保存referenceの絶対誤差上界はnullであり、uを厳密なreference/丸め/bias certificateとは呼ばない。正式なshot政策への採択や、母平均費用優位の保証を今回の実行承認から導かない。

候補別のshot数・費用は全てnullのまま。samplingをしていないので、新cohort・新seed・標本SEもない。signalの小さいbiasだけで費用winnerを選ばず、m5がS2/Yoshida4や強化決定論的対照より安いという結論は出していない。追加PFの主比較採用AD03も保留を維持する。

## 成果物と監査

[artifact入口](../../artifacts/resource_applicability/track_a_h46_m5_signal_run02_results/2026-10-20/README.md)から以下を参照できる。

- [signal_records.json](../../artifacts/resource_applicability/track_a_h46_m5_signal_run02_results/2026-10-20/signal_records.json)：88候補の保存JSON値を転記。結果SHAは各系completionとprivate inventoryへ結合。
- [signals_88.csv](../../artifacts/resource_applicability/track_a_h46_m5_signal_run02_results/2026-10-20/signals_88.csv)：signal/bias成分・normalization・allowanceを候補ごとに並べた表。
- [axis_eligibility_880.csv](../../artifacts/resource_applicability/track_a_h46_m5_signal_run02_results/2026-10-20/axis_eligibility_880.csv)：5精度×2軸×88候補。shot/costは空欄（JSONではnull）。
- [candidate_precision_status_440.csv](../../artifacts/resource_applicability/track_a_h46_m5_signal_run02_results/2026-10-20/candidate_precision_status_440.csv)：保存real/imagラベルを併記。合成適格性の新規則やrankingは付けない。
- [end_audit.json](../../artifacts/resource_applicability/track_a_h46_m5_signal_run02_results/2026-10-20/end_audit.json)：47終了監査、2系context・台帳・SOURCE・SHA・null条件・exit/reap・旧保全。
- completion/wait4/fresh gate、SOURCE closure、承認の非実行用projection、relative pathのprivate raw inventory。

終了監査では信号の再作用や適格性式の再実行をせず、保存値の転記・identity・finite・既存gate記録の整合を確認した。公開資料の構造・転記検査41項目もPASSした。著者自身による監査であり、独立レビューではない。

旧2,899保護ファイルとrun01 STOPの18ファイルは全てSHA一致。元SOURCE・grant・ledger・marker・STOP、他worktree、共有環境・他利用者のprocessは変更していない。完了済みrun02の196ファイルも別SHA snapshotに保存した。旧run01はSTOPのままで、run02の正常終了とは別来歴である。

## 次の担当と停止

- **ユーザー**：今回のsignal・軸別適格性を確認し、次の費用取得domainと標本設計の準備へ進めるか判断する。全88候補のsignal取得は完了しているため、この段階の再実行は不要。
- **Codex（次段階・未着手）**：指示があれば、両軸の適格性と入力/native/compiler同一性に基づく費用取得計画を作る。保存単価12軸の再利用を照合し、PRのn_E/seed、決定論的不足軸、必要なnative検査を具体化する。既存強化S2/S4対照を削らず、費用欠測qをbiasだけで除外しない。
- **ユーザー**：新sampling・compile・shot/費用会計・独立確認は、その具体的な計画に対して別途実行承認する。今回のgrantは次Phaseを認可しない。

local commit後mandatory STOP。公開push・追加科学計算・次Phaseへの自動移行は行わない。
