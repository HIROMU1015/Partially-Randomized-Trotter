# N3保存帰属v1：タイマ・出力義務・情報欠落の対応

2026-10-13 JST。[完了済みpilot科学レビュー](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/review_intake.md)第12.2–12.3節への技術資料。
凍結source3eba0814を静的に読み、保存literalとpointerを索引化した。新しい時間集計、幅評価、PSD再認証は実行していない。

対象は非負density、full5/6・active4/5、ν2/3、一外部mode、候補2mode/H、元ε_proj=.01の固定6入力。
geometry/basis/DF rank/policy/L_D/PF窓は適用なし。新ε、入力、solver設定、候補探索は作らない。

## 出力義務

| path | 候補生成・出力 | 同じ物差しではない部分 |
|---|---|---|
| OCC | 32 spectral score更新、端点と丸めκ候補、τ/Z/s、正の幅証拠、有理checker | 負のFantope証拠を取得しない。最適幅完全性はない |
| SDP_EQUIVALENT | δ=.01/.005/.0025各1 solve、κ/τ/Zの正証拠、dual γのtrace補正/混合、可能なら厳密非可行証拠 | OCCより追加の出力義務がある。単に3で割った一回費用を推定しない |
| DENSITY_SOS | 同じ3tier、particle/hole/D/Q/G Gram、number multiplier、ℓ、残差e、正証拠、sector checker | より強い証拠族・大きいGram/sector検査。一般SOSの本質的速度差とはしない |

[保存literal索引](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/saved_literal_index.json)は216 mode-path rows、432 solver events、696候補を参照元file・JSON pointerへ結合する。
sourceのtier loop順と、保存された全3event・3candidate slotの存在を確認してtierへ対応させる。slotが欠ける場合は曖昧な対応を拒否する。
反復は独立6H以上の科学標本に数えない。

## 18 inaccurate eventの所在

[literalによるstatus索引](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/inaccurate_event_pointers.json)の18recordは、すべて`V2_banded_6`、`DENSITY_SOS`、external5。
3tier×kernel/cold×各3反復に対応する。同値SDPの保存eventには`optimal_inaccurate`はない。
原rawにあるiterations=20000もそのまま転記した。数値品質の再判定や、新たなsolverを実行した結果ではない。
この所在から、OCC/同値SDPの幅差の説明に、その18recordを同値SDP側のinaccuracyとして用いることはできない。
SOSの幅・費用へどれだけ影響したかは、statusだけでは判別しない。

## timingの境界

`Engine.phase`は子phaseのwall/CPUを親から差し引くexclusive timer。[静的source位置索引](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/source_static_function_index.json)を参照。

| 保存field | source上の範囲 | 保存値だけでは分けられない部分 |
|---|---|---|
| numerical_proposal | NumPy eigh、またはCVXPY `problem.solve`全体 | CVXPY canonicalization/interface/backend/postprocessの専用timerなし |
| reported_solve_time / reported_setup_time | backendが返す時間 | problem構築、全CVXPY費用、後段有理証拠処理を含む時間ではない |
| bound_acquisition | 上界と外部lowerの取得。nested numerical/repairを除く | phase外のAPI/control overhead |
| rationalization_and_repair | PSD repair probes、SOS係数残差の処理等 | 丸めや行列組立の一部はphase外。SDP dual混合/反証専用timerなし |
| SOS_coefficient_assembly | SOS problem/CAR/CSC構築 | PSD/LMI solver、その後のsector checker |
| independent_checker | 正証拠の独立checker | SDP反証のverifyはこの専用phaseで計測していない |
| width_and_output | 幅の有理区間・witness構築 | H selector/serializationはpath別field、disk IOはrunner共通 |

固定[SCS3.2.8 header](https://raw.githubusercontent.com/cvxgrp/scs/3.2.8/include/scs.h)はsolve/setupをmillisecondsで返す。
[CVXPY1.6.5のSCS adapter](https://raw.githubusercontent.com/cvxpy/cvxpy/v1.6.5/cvxpy/reductions/solvers/conic_solvers/scs_conif.py)は1000で割ってsolver_statsへ格納するため、pilot保存fieldの単位はseconds。
adapterはSCS3系のstatus2をoptimal_inaccurateへ対応させる。solverの数値statusと、有理証拠の健全性は別。

新事後評価でphase合計やbackend時間との差を求める場合も、未割当の混合残差として扱う。負の残差を隠したり0へclipしない。
それをrefutationだけの費用、canonicalizationだけの費用、refutationを除いた反実仮想の費用へ変換しない。
新しい同機能benchmarkや出力義務の変更は、今回の保存整理から自動的に認可しない。

## 保存・未保存

保存：最終κ/τ/Z/s、修復後Gram、丸めℓ、残差bound、number係数、幅区間、solver status/iterationsとreported timing、phase totals、元U/c証拠。
未保存：raw float proposal、修復前Z/Gram、採用repair shift r、修復前objective、SCS primal/dual residual・duality gap、tier別refutation専用time。
`repaired`はrも返すが、producerは最終証拠にrを保存していない。最終Zの対角等から唯一のrを逆算できるとは扱わない。
SOSのeは保存されているが、元solver誤差とPSD修復の寄与をそれぞれ識別した値ではない。
保存されなかった値を再実行で復元し、原記録へ追記することはしない。
