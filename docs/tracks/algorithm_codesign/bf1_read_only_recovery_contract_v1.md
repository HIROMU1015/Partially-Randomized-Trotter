# BF1-R0: saved-data recovery contract v1

2026-10-05 JST。受領判断：`PROCEED_READ_ONLY_BF1_RESULT_RECOVERY_BEFORE_REDESIGN`。
**一回限りのread-only recoveryを認可する。science rerunは認可しない。**
利用者が提示したGPT reviewに従う。review原文とraw SHAは新contract artifactへ保存する。
以下をcommit固定してから原BF-1のreplayを一回だけ行う。scopeを結果後に広げない。

## 1. Identityと入力

| 項目 | 固定値 |
|---|---|
| 元science source S | `e59344a564e70d64dc3ea39d640581c72676df31` |
| 元authorization A | `cc971e4a2bff9b0c5708003fde7b9519eed27241` |
| 元result commit | `09c9e89555032213b52a6a60b38f55563d07a34d` |
| serialization修正commit | `abd24254f84f35fa1d942928c4c49a4bda4de194` |
| R0 branch | `track-b-bf1-read-only-recovery` |
| R0 worktree | `/home/abe/Project/prt-worktrees/track-b-bf1-read-only-recovery` |
| R0 execution ID | `bf1-r0-20261005-read-only-v1` |

科学データは原`result.json`と`cells.jsonl`だけ。原cellはideal 200＋finite 1,069。
result raw SHAは`99ef73f93f6303dbb1c85ac887dbfd66d02e006305938da0623f880493ea8250`、
cells raw SHAは`a2974de79d454e062b41c4485e58ef07a0152ba5bf7163912575f0d2af1da0a6`。
domainは原v1 manifestを使い、raw SHA
`caece92bd2d9f827b0f1f17281865869420bf923273ff294fd30cabd24b51efe`を維持する。
原preregistration/v2/v3、Sのsourceとformula-only domain/allocation規則を参照する。
保存された`lambda_r`とD generator数は原resultのinput auditから取得する。
Hamiltonian、state、targetは再構築せず、保存済みbias/uを使う。

元入力scopeはknown/development H4 linear 1.00 Å、STO-3G、8 qubits、DF rank12、
`L_D=3`、one-body込みD generators 4、`T=0.8`。新held-out/独立validationではない。

## 2. 復元規則

元sourceの`search()`、objective、K4 trigger、allocation、`rescore()`、`classify()`、
secondary frontierをそのまま再利用する。physical initializationを呼ばない専用Evaluatorは
保存済みcacheのlookupだけを実装する。sourceとnumerical環境を照合する。

1. frozen 16 initial points、4 fixed refs、domain長さとidentityをformula-onlyで照合する。
2. O/L/Fそれぞれ16 starts＋16 refinementを元のdeterministic規則で再playする。
   record順からarmを推測しない。別optimizer/探索budgetを与えない。
3. 元runnerの順序でO∪L∪fixed、Fをprimary `epsilon=.01`でcommon finite rescoreし、L集合を作る。
4. 原`classify()`とsecondary frontierを適用する。primary 5%、数値guard、±2%感度、boundary規則は維持する。
5. 元unionだけを修正版cross-scoreでcache-only採点する。Sとの差は順位のPython `int`化だけ。
6. optional bridge `epsilon=.05`は既存cacheだけで試す。再探索しない。

各required lookupが不足したら、`RESCUE_INCOMPLETE_MISSING_ORIGINAL_CELL`でその場で停止する。
不足キー、phase、完了済みarmを記録し、holeを埋めない。identity不一致・重複recordも停止する。
全候補identityは既存cellに結び付ける。再構成された係数定義は新しいscience candidateではない。
cacheのkeys/counts/nested valuesが不変で、新cell・signal・追加science candidateが0であることをassertする。

cross-scoreは`POSTHOC_RECOVERED_OBJECTIVE_ATTRIBUTION`として保存し、元runで直接保存された証拠とは呼ばない。
primaryが未復元ならcross-scoreへ進まない。診断はprimary分類を変更しない。
bridgeに一つでも不足cellがあれば`MISSING_FROM_ORIGINAL_RUN`を保存し、primary/attributionの成功を取り消さない。

## 3. Outcomeとevidence境界

- `BF1_READ_ONLY_RECOVERY_COMPLETE`：3×32探索、common primary判定、secondary frontier、cross-score/attributionが復元できた。
  原規則が`INCONCLUSIVE`を返した場合も、復元処理が完全ならこのstatusになる。
- `BF1_READ_ONLY_RECOVERY_INCOMPLETE`：必要cell欠落、identity不一致、処理失敗、上限等で上記が完了しなかった。
  完了済みの段階は保持する。primaryが復元済みでもattribution未完了ならこのstatusと区別して記録する。

原resultは永久に`BF1_INCOMPLETE_MANDATORY_STOP_NO_RETRY`、`INCONCLUSIVE`のまま。
回収できた分類は「source-bound one-shot dataから事後の機械的replayによりpreregistered primary分類を復元した」
と記述する。「original BF-1 was BF-C」と書かない。failure後に作られた解析contractと、
結果前契約で取得された科学データの二層を分ける。CI/外部再現/compiled総cost/new method成立を主張しない。

## 4. 禁止、上限、保存

NPZ resolve/stat/hash/load、Hamiltonian/state/target再計算、signal取得、新ideal/finite cell、
追加coefficient/q/R/K、別探索、geometry/split変更、trajectory/circuit/compile/GPUを禁止する。
Track Aのworktree/API/artifact/runtimeと旧STOP、元preparation/source/resultは変更しない。

1 worker、BLAS/Numba threads 1、wall/CPU各120 s、process RSS 1GiB、新output 16MiB。
実入力1269 cell以外を受け取らず、元400/4000 cell上限も検査する。
new-code synthetic testsはinvented scalar cellsを用い、原replay前にpassさせる。full suiteは行わない。

contract/source/tests/reportは`artifacts/track_b_bf1_read_only_recovery_contract/2026-10-05/`と
本branchへcommit・pushする。runnerはclean HEAD、全text bytesとHEAD blob、原S source一致
（cross-scoreのcast例外だけ）、数値環境、元consumed markerをreplay前に確認する。
HEADの完全SHAをlaunch時に記録するためcontract自身へのcommit自己参照は不要。

R0専用consumed markerをgit commonの`track-b-bf1-read-only-recovery/`へexclusive作成する。
原science markerを保持し、R0のretryも禁止する。新resultは
`artifacts/track_b_bf1_read_only_recovery/2026-10-05/v1/result.json`へexclusive保存する。
原artifactを更新・再生成・移動せず、新契約のsource/inputsを結果後もhash照合する。

## 5. 必ずSTOP

成功・失敗・復元したprimary outcomeのいずれでもmandatory STOP。
science_execution_authorized=false、science_retry_authorized=false、BF2_authorized=false、automatic_next_stage=null。
結果と必要資料をcommit・pushし、完全SHAとGitHub URLでGPT側へ戻す。
次のRQ・新規性・論文着地点の全面再評価、救済失敗後のrerun例外の必要性・scopeはGPT側が判断する。
本指示はrerun例外を認めていない。marker消去、別実行IDへの付け替えや条件追加へ進まない。
