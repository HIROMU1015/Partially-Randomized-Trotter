# H4 tol-only DF10 S_STATS 最終技術準備（2026-10-19）

**実行準備完了・数値解析の別途承認待ち、mandatory STOP。** D190とA50をユーザーが正式に受け入れ、独立確認Pの対象2条件とS_STATSの技術準備を指定した。今回の実データによるn・SD・power設計、raw再集計、bootstrap、sampling、compile、P実行、新seed生成はすべて0。有効grantを発行していない。

専用branch `codex/h4-tol-s-stats-preparation-20261019` をA50完了commitから作成した。数値計算SOURCEは **`8752489bde3903d80d306c4500535c79ac65ef31`**。既存D/A科学SOURCE `84db3909639b34ec521004c3d8f7ef9b5e6ad851` は変更していない。旧rank12 S_STATSの統計値・候補・n設計を移送せず、stdlibの共通normal n式・JSON/SHA helperだけを再利用した。

## 固定対象・保存入力

linear H4・隣接1 Å・STO-3G、tol-only threshold1e-8 DF rank10、system8 qubits＋ancilla、sector36、T=.8、P0、通常signed bias・allowance倍率1。PRは既存B2二次。q1のδ=.8とq8のδ=.1を対象とする。保存状態のH4/H6/H8 solver・正規化の差は[Phase 0](track_a_h4_tol_main_phase0_20261019.md)の記録のまま。

| ε | 固定PR | 保存mean exact | 保存paired SE | 保存n | D190登録内最良PF | 保存PF G_RZ |
|---|---|---:|---:|---:|---|---:|
| .05 | `B2_p3_q1_2nd_r4_K2` | `542760385/8` | `1472260.835308543` | 8 | `B0I_p5_q1_2nd` | `70108214` |
| .0005 | `B2_p2_q8_2nd_r2_K4` | `4912729573525` | `43698381292.52138` | 8 | `B0_p7_q8_2nd` | `7856558002614` |

上表は保存値の転記であり、新しい統計計算ではない。両PRは`EXPLORATION_N8`の別候補。保存FP n32・H4 legacy・条件付きpilot・他分子の標本と混合しない。候補や精度を追加せず、再選択・再ランキングを行わない。保存済みwinnerと指定対象の一致だけを検査した。

- A50結果commit：`85cb5b4d1808434165c421115ed09dab3b63dd00`。結果SHA256：`3c217dc6c6cafe3e13d9ee767088d671dee18273dff300dd9bf34cb03d8dbb69`。
- D190結果commit：`406f70570323366da185355550a3900388717fbb`。結果SHA256：`106469f5a604af01ff2cb77012a736646df3291abe60a7c870e77785058ec8a3`。

commitのGit blobと現在の保存bytes、両stageの終了監査・native exit0・worker回収を照合した。入力はD/Aのresult・audit・native exit・execution binding・artifact manifest各5、rank10 S_STATS草案1、逐語selected summary1、今回の準備承認receipt1の計13。元結果を再計算していない。compilerは保存値の来歴としてQiskit1.3.0/rz-sx-x-cx/opt1/seed17・symmetric_directionalをbindingに残す。S_STATS自身はcompilerを呼ばない。

## 数値実行契約

family α=.05、planning power=.8、固定primary endpoint数m=2。片側Bonferroniで`z_(1−α/2)`を使い、`z_power=z_.8`とする。実行承認後に限り、各候補で保存SEから`s_plan=SE_saved×sqrt(n_saved)`を導出し、次式を適用する。

```text
n(f, eta) = max(2, ceil(((z_(1−alpha/2) + z_power)
                         * f * SE_saved * sqrt(n_saved)
                         / (eta * PF_G_saved))**2))
```

SD倍率f=1/2/4/8、相対検出差η=.01/.02/.05。2 endpoint×4倍率×3差＝**24シナリオ**、感度設定12組、固定基準方向2比較を1 passで扱う。[既存rank10草案](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/S_STATS_contract_DRAFT.json)の条件を継承し、基準は**f4・η=.02**。小さなnのscenarioへ差し替えず、nに恣意的上限を設けない。同一候補の複数endpointがあれば最大nを使う一般則を保つが、今回の2候補は別である。

保存meanは`PF_G−PR_mean_exact`の固定方向確認にだけ使い、検出差を探索で観測した差へ置き換えない。方向は有理数で保存し、同費用は0／INCONCLUSIVE。感度ごとの候補別nとP提案call予算は、将来の承認済みS_STATSの中でのみ算出する。bootstrap・CI/p値・実power評価・希少層visibility nを自動追加しない。

| S_STATSの承認対象案 | 予約数 |
|---|---:|
| numerical design pass | 1 |
| normal scenario | 24 |
| 基準方向比較 | 2 |
| 感度設定 | 12 |
| raw再集計・bootstrap・law再構成・sampling・compile・signal・shot再計算・P実行・新seed | すべて0 |

今回の実績はnumerical passを含めすべて0。Pのn・SOURCE・seed・予算はnull/PENDINGのまま。将来S_STATSが出すPのtrajectory・2軸compile・occurrence/RTE event・control・seed slot等は提案値で、Pの科学実行承認を意味しない。

## 希少eventと母平均保証の限界

2候補の保存stratum集計はともに`S0_ONLY=8, S2_SINGLE=0, S2_MULTI=0, S4_PRESENT=0`。これは保存集計の取得であり、trajectory rawから再集計していない。K2候補では未観測S2、K4候補では未観測S2/S4の費用・分散に関する情報が不足している。

保存summaryには候補固有のlaw確率p2/p4・visibilityがなく、今回それらを再構成しない。観測0を確率0、未知費用を費用0と扱わない。n8の保存SEやf4/f8の拡大は、未観測層を含む分散上界・母平均coverage・達成power・選択後の推論を保証しない。n8内にS0以外がない点を保持したまま、近似normalモデルによる計画値として出力する契約である。

このnormal設計に必須のmean/SE/n/PF入力は揃っている。希少層の確率や費用まで評価したい場合は、必要な最小限の追加入力・解析を別の承認対象として指定する。今回のS_STATS承認案にraw再集計・保存lawの新数値評価・条件付きpilotからの確率移送を含めない。S_STATS後のP最終固定は、結果とこの限界を確認したユーザー判断の後に行う。

## SOURCE・runtime・grant・終了監査

[contract/gate](../../scripts/h4_tol_s_stats_contract.py)、[numeric kernel](../../scripts/h4_tol_s_stats_numeric.py)、[runner/witness](../../scripts/run_h4_tol_s_stats.py)、[metadata preparer](../../scripts/prepare_h4_tol_s_stats.py)、[人工テスト](../../tests/test_h4_tol_s_stats.py)を追加。計算SOURCE closureは9 file、入力13、stdlib/共有binaryを含むruntime128 file、Python3.11 binaryをSHAで固定した。数値計算にNumPy/SciPy/Qiskit/native providerを使わない。

私有spec digest：`c21d9b2eb4fad92cabb5b381346e8f41d9bb15f5f92f03cf655cfcf511d11426`。[実行契約projection](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_preparation/2026-10-19/S_STATS_contract_SEALED_PENDING_APPROVAL.json)、[保存summary](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_preparation/2026-10-19/selected_saved_summaries.json)、[INVALID grant](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_preparation/2026-10-19/S_STATS_grant_INVALID.json)、[承認境界](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_preparation/2026-10-19/approval_boundary.json)に固定情報を保存した。私有契約・input snapshot・runtime SHA・保全manifestは`.h4_tol_s_stats_private/`に保持する。

無効grantは最初のgateで拒否し、科学入力の読込・数値module import・lock/output・witness作成へ進まない。有効化には、別の明示承認receipt・fresh資源receipt・新しいexecution spec/grantが必要で、今回のpending specとINVALID grantを上書きしない。one-shot lockと非払い戻しledgerを先に保存し、失敗時はSTOPと予約を保持する。自動retry・次stage開始は禁止する。

24条件のスカラー計算は1 processの1 passとする。CPU/RAM/worker/wallの恣意的上限なし。起動時の実効資源を観測し、affinity・共有設定・他jobを変更しない。PSIは診断情報として扱い、無関係なworktreeのHEAD/statusをgateにしない。

終了受入条件はnative exit0・同一親wait、固定SOURCE/input/runtimeの終了後SHA、24 scenario/2方向/12設定の保存、基準f4/η.02、P未実行・禁止call0、result/terminal/ledger/ONE_SHOTの保存と整合性である。終了監査で数値設計を再dispatchせず、保存結果とmetadataを検査する。Pの最終nや科学grantは今回作成しない。

## 検証・保全・次の承認

人工21＋既存数式回帰14＝**35 PASS**。数式の既知ceiling、family数2、感度12、24 scenario、call予算、exact方向/同費用、n capなし、cohort/入力/契約改変拒否、人工one-shot重複拒否・STOP保持を検査した。実CLIの無効grant拒否はdirect/witnessの2経路PASS。

実際のmetadata準備中は`NormalDist.inv_cdf`、`normal_n_formula`、mean/variance/stdev/covarianceを例外化し、数値module/scientific providerのimportも拒否するguardを使用した。その状態で準備が完了したため、実標本によるn・SD・power計算は呼び出していない。D/A/F/FPを含む旧SOURCE・結果・grant・ledger・marker・STOPの9,178レコードをSHA照合し不変。作者自身の技術確認であり、独立科学レビューではない。

[テスト報告](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_preparation/2026-10-19/test_report.json)、[拒否検証](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_preparation/2026-10-19/refusal_report.json)、[準備終了監査](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_preparation/2026-10-19/completion_audit.json)、[artifact SHA](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_preparation/2026-10-19/artifact_manifest.json)。数値output・ONE_SHOT・native_exit・実行resultは未作成。

専用branchのSOURCE commitと成果物commitとして保存し、今回新規の公開pushは行わない。旧D/Aの公開承認を、新しいS_STATS成果物の公開承認へ自動拡張していない。

**mandatory STOP。** 技術的な未解決事項はない。nとP予算は未計算、希少層の確率・費用と母平均保証は未解決の科学的限界として残る。次は**ユーザー**が、このSOURCE・入力・24シナリオ・2比較・f4/η.02・希少層の限界を確認し、S_STATS数値解析1回を別途承認するか判断する。承認後に**Codex**がfresh receiptと有効grantを固定して一回実行し、保存・監査後に再びSTOPする。sampling・compile・P科学実行は別承認のまま。
