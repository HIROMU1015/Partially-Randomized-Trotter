# BM-0.5 同値性監査・GPT handoff

2026-10-05 JST。`BM05_EQUIVALENT_NO_ESTABLISHED_METHOD_DELTA_MANDATORY_STOP`。
**固定nested列の三次BCH係数は同値。現adapterのnew-method gateは不通過、BM-1は実行しない。**
BM-0を保持し、研究全体を再設計していない。applicationへの縮小・別deltaの採否はGPT側。

## 1. Review入口

| 資料 | 確認すること |
|---|---|
| [利用者review受領記録](inputs/bm05_review_request_20261005.md) | 今回承認された記号監査とpilot案の修正範囲 |
| [同値性・method-delta監査](bm05_equivalence_and_method_delta_audit_v1.md) | 一般sizeの再帰導出、score policy、差分表、new-method gate |
| [限定formal audit script](../../../scripts/tracks/algorithm_codesign/audit_bm05_symbolic_equivalence.py) | 抽象wordsとFractionだけ。physical providerなし |
| [9 fixtureの保存report](../../../artifacts/track_b_bm05_equivalence/2026-10-05/formal_word_audit_v1.json) | 直接exp/log・compact・BMがdegree3で完全一致、誤係数control6件 |
| [BM-1案 v2 amendment](bm1_pilot_scope_amendment_v2.md) | leading heuristic／I2、primary count／secondary人工cost、実行未認可 |
| [audit status／文書inventory](../../../artifacts/track_b_bm05_equivalence/2026-10-05/audit_status_v1.json) | source・report hash、科学操作0、STOP |

## 2. Scopeとidentity

branch `track-b-bm05-equivalence-20261005`、独立worktree
`/home/abe/Project/prt-worktrees/track-b-bm05-equivalence-20261005`。
base：[BM-0 packet at 3b2d624](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b2d624adde979f7d8f983bc7fdbbec88f90c500/docs/tracks/algorithm_codesign/bm0_review_packet_20261005.md)。
元BM-0の数学仕様／claim表／pilot v1／design manifest／入力snapshotは変更しない。
BM-0 design manifestのinventory hashはその固定BM-0 commitに対して照合する。
今回のindex追記は新BM-0.5 manifestで管理し、古いmanifestのhashを現在のindexへ書き換えない。

B原science source `e59344a564e70d64dc3ea39d640581c72676df31`、authorization child
`cc971e4a2bff9b0c5708003fde7b9519eed27241`、原中断result
`09c9e89555032213b52a6a60b38f55563d07a34d`、R0 result
`6d2645a09440f50e5b869ef42a1b73a1b625a1af`は保持する。
A参照`4c23453c541700c6a41ba71fc5ec9323b53858d6`も固定参照だけ。
rootの未commit文書やAのworktreeを編集・copy・stageしない。

## 3. 今回閉じたことと、閉じていないこと

| 項目 | 判定 |
|---|---|
| ideal三次operator係数 | compact=BM。任意group sizeの再帰導出＋限定formal check |
| DF substitution後 | 同じ入力へ同じ式を代入するので同じ。物理行列は未生成 |
| score／候補順位 | backend・aggregation・cost・tieが同じなら同じ |
| triangle policy差 | combined/separatedは違い得るがcompactにも両方可能。独立methodの証拠ではない |
| reuse／formal評価cost | 強いcompact側も同じfloor/internalをcache可能。現案に専有できる削減なし |
| finite-T、finite RTE、実数値、分子、actual compiled cost | 今回未評価／未保証。symbolic等号を転用しない |
| 現adapterのnew-method GO | 不通過 |
| application studyの必要性・scope／別delta | GPT判断待ち。Codexは自動で始めない |

BM-1のnew-method routeを今走らせても、同score対照間の差を作れない。
違うaggregationや弱い再計算baselineを与えて差を作ることはしない。
これをB-M全構成の失敗や、別の評価法の不可能性へ一般化しない。

## 4. 操作と停止

関連一次本文の確認、text／metadata読取、一般sizeの形式導出、
独立な三経路を比較する**一回の形式algebra script**、文書・provenance監査のみ。
9 fixtureは抽象的なgenerator indexで、BM-1 toy/geometryの追加ではない。
science run、Hamiltonian/state/target生成、NPZ resolve/stat/hash/load、signal、trajectory、
circuit build/compile、GPU query/use、保存science値の再集計・再採点・再分類、full test suiteは全て0。
科学source・共有API・旧artifact・marker・global validation manifestを変更しない。

必要資料はAGENTS.mdの継続的な指示により独立B branchからoriginへcommit・pushし、remote SHAを照合する。
公開はBM-1のauthorizationではない。**mandatory STOP、BM1/BM2/BM3_authorized=false。**
次のGPT reviewは、限定applicationに情報価値があるか、または別の具体deltaを必要とするかという判断だけ。
今回その判断に先回りして新しいmodel・method・pilotを追加しない。
