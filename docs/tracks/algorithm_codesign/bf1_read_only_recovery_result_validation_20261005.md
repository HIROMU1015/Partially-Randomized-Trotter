# BF1-R0 read-only recovery result and GPT handoff

2026-10-05 JST。`BF1_READ_ONLY_RECOVERY_COMPLETE`、mandatory STOP。
**事後の機械的replayにより、preregistered primary分類BF-Aを復元した。**
原BF-1は`BF1_INCOMPLETE_MANDATORY_STOP_NO_RETRY`、`INCONCLUSIVE`のまま保持する。
science rerun、新cell/signal、追加science candidateは0。研究方針の変更はGPT側へ戻す。

## 1. 二層のidentityとscope

| 証拠 | Identity |
|---|---|
| 結果前契約で取得された元science source | `e59344a564e70d64dc3ea39d640581c72676df31` |
| 元authorization-only commit | `cc971e4a2bff9b0c5708003fde7b9519eed27241` |
| 元の中断result commit | `09c9e89555032213b52a6a60b38f55563d07a34d` |
| failure後のR0 contract/source commit | `f226f8c81b5ba07b6d0c4b248c0eb15bf080a622` |
| R0 contract SHA256 | `d0c150796e9639c26a41ccc6045d05d2081b5fdd6af511da784664960ff337ae` |
| R0 branch / execution ID | `track-b-bf1-read-only-recovery` / `bf1-r0-20261005-read-only-v1` |

[R0契約](bf1_read_only_recovery_contract_v1.md)とsource/tests/reportを結果前にcommit・pushし、
remote一致を確認してから一回だけreplayした。原Sのrule source 10ファイルを照合し、
cross-scoreだけは承認済みserialization castを許容した。全24 source textと6 input textを
実行前・終了直後・結果文書更新前に照合した。古い契約・preparation・原resultを書き換えていない。

科学入力のscopeは元のknown/development H4 linear 1.00 Å、STO-3G、8 qubits、DF rank12、
generation-prefix `L_D=3`、one-body込みD generators 4、`T=0.8`。
5-stage symmetric fourth-order family、O/L/F各32点、四fixed refs、
`q={1,2,4,8}`、`R_bud={5,10,20,40,80}`、固定allocation、K2と事前triggerのみのK4。
primary `epsilon=.01, alpha=.05`、materiality ratio `<=.95`、元数値guardとboundary規則を維持した。
R0では分子NPZを操作せず、保存済みbias/uとformula-only情報を使った。

これはsource-bound one-shot dataに対する**failure後の復元解析**。
元runで直接保存されたprimary/cross-score、immutable CI、外部再現、independent validationとは呼ばない。

## 2. 復元したprimary判定

| 項目 | 復元値 |
|---|---|
| Primary outcome | **BF-A** |
| 元`classify()`のreason | `less_than_1_percent_finite_decision_gain` |
| F最小 / O∪L∪fixed最小のfinite action proxy | **1.6024124982** |
| Ratio interval | `[1.6021983874, 1.6025769383]` |
| Ratio uncertainty | `0.0003785509` |
| F最小 / L集合のfinite最小 | **1.0000000000** |
| F bestがL集合外か | false |
| ±2% cost四cornerのratio範囲 | `1.5907589599–1.6139736616` |

BF-Aは元sourceの`ratio >= .99`というoperational ruleによる。
今回のF bestは共通参照より約60.24%大きく、共通参照と「ほぼ同じ」という意味ではない。
F対Lのfinite最小値は同じで、この固定scopeでF固有のdecision-relevantな利得は得られていない。

| 集合 | Best formula | q / R / K | Finite action proxy |
|---|---|---|---:|
| F | Suzuki5 | 1 / 10 / 2 | 20,709,936.722970817 |
| L（共通finite採点） | 同じSuzuki5 | 1 / 10 / 2 | 20,709,936.722970817 |
| O∪L∪fixed refs | native S2 | 2 / 5 / 2 | 12,924,223.160897588 |

F/L winner identityは`e7ade3f521e82595cb6b0b2de64a225fbddc793395d8885e23d4771272da8113`。
q変化はsecondaryとして保存したが、primaryの別GO routeにはしない。
指標はshotsとnative actionのproxyであり、compiled gates/RZ、最終RPE総costではない。
限定family・探索budget・development条件外へ一般化しない。

## 3. Attributionとbridge

32点×3armを原探索規則で再構成し、union 50係数をO/L/Fで150 logical cross-scoresした。
originsはreplayed search集合から決め、saved record順から推定していない。
全replayed identityが元cellに一致し、cache keys/counts/nested valuesは不変だった。

cross-score scopeは`POSTHOC_RECOVERED_OBJECTIVE_ATTRIBUTION`。
元`attribute_F_winner()`は`SEARCH_REACHABILITY_OR_BUDGET_EXPLANATION_NOT_EXCLUDED`を返した。
今回は`J_L(w_F) = min J_L(W_L) = 21,206,506.000019286`で、F winner自身がL集合に含まれる。
このflagはsearch到達差が実際に存在したという認定ではない。`establishes_design_principle=false`を維持する。

bridge `epsilon=.05`は`MISSING_FROM_ORIGINAL_RUN`。
不足はfinite cell `q=1, R=5, K=4`、係数identity
`708496f5a73473fb5475f728659f7dba42c51571e4fdf94b542b5057a90be37b`。
既存cacheだけでのbridgeを停止し、穴埋め・再探索を行わなかった。primary/attributionの復元は完全。

## 4. 技術検証と保存

限定23 testsが通過。invented scalar cellsで元ruleとの一致、不足cell停止、bridge分離、
physical providers拒否、draft/input gateを検査した。最終reportは
`artifacts/track_b_bf1_read_only_recovery_contract/2026-10-05/focused_synthetic_report_v2.json`。
full suite、science runner、NPZ resolve/stat/hash/load、Hamiltonian/state/target再計算、
trajectory、circuit/compile、GPU query/useは全て0。runtimeの未宣言file access拒否発生は0。

replay wall 1.2782 s、CPU 1.2776 s、process peak RSS 384,679,936 bytes。
R0上限wall/CPU各120 s、RSS 1GiB、output 16MiB内。科学実行の性能比較ではない。
resultは2,542,288 bytes、SHA256
`013d1b2ea5749c74adf251cb7cd6c7ae5dbfd66f805bb8ca6601e0ea0c01fa53`。

[新result](../../../artifacts/track_b_bf1_read_only_recovery/2026-10-05/v1/result.json)と
[結果audit](../../../artifacts/track_b_bf1_read_only_recovery/2026-10-05/v1/result_validation_audit.json)を参照する。
元200 ideal/1069 finite、原preparation 14ファイル、原science markerは保持した。
R0専用markerもconsumed、retry=false。request原文はCRLF・Markdown空白を含めraw bytesのまま保存した。

## 5. GPT側へ戻す未決事項

元preregistered primary情報とattributionを回収できた。研究全体の方針、RQ、新規性、
論文着地点、追加検証の必要性・範囲は、このBF-Aと証拠scopeを入力にGPT側で全面再評価する。
CodexはB-F main lineの採否や別研究への転換をここで決定しない。
science rerun、別geometry/split/family/精度、BF-2を開始しない。
全結果後mandatory STOP、science_retry_authorized=false、BF2_authorized=false、automatic_next_stage=null。
