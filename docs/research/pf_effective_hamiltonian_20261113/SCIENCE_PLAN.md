# 未実行：H_eff診断の最小科学pilot案

2026-11-13 JST。今回実行したのは人工testとmetadata確認だけ。**新実分子のstep/固有値/保存signal比較・PF幅選択には別途明示承認が必要。** Claudeレビューの約60点格子やΔ_max/0.8Δ_max案を今回の正式menuにしていない。

## 最初に固定する条件

系・元保存H/state/sector・比較対象Full PF一候補・既存formula・実幅Δ・branchとreferenceの定義/SOURCE・二つの確認時刻を有限計画に固定し、ユーザーがその範囲を承認する。A/Cが対応する数値許容値とSOURCEを具体化し、細かな実装判断ごとの追加承認を要求しない。PF公式・Δ探索窓・energy予算は今回選択しない。

保存H4 DF10を選ぶ場合は既存A91349/C86f37の入力loader・同H/state/snapshot・weighted-Hermitian completion・fragment順・scalar/P0を維持する。参照energyを保存solver proxyとするのか、追加の根拠を持つground energyとするのかを明示する。proxyの厳密ground同定を仮定で補わない。

## 最小scope：Full PF一候補・一幅

| 処理 | 上限案・判定できること |
|---|---|
| PF step | 既存direct-expm builderでscalar込みUを1個。新回路build/compile0 |
| effective spectrum | complex Schur1回、選択branchのH_eff/overlap/再構成診断1件 |
| reference H eigensolve | **既定0**。保存energy/sourceを使う。元固有basisとの対応が必要なら、追加Hermitian eigensolve1回を承認対象へ明記 |
| q power/signal照合 | 指定二時点でU^qのstate action/complex meanを計2件。全spectral measureによる同signal再構成と比較 |
| 保存値の比較再解析 | 使う場合のみ最大2件をscopeへ含める。同identity/law/scalar/source/数値backend条件を保持し、別candidateのmeanを流用しない |
| 新full-H reference | 保存pairを使える場合0。欠測なら追加取得へfallbackせずSTOPし、必要範囲を報告 |
| N/RPE統計・価格/G・幅探索 | 全て0 |

二時点はT=qΔが整数で、同stepのphysical recipeが成立するものを指定する。m12だけや二点だけで全13roundのRMSEを資格付けしない。既存direct expmの正当性検証を再目的化せず、今回新しいlog branch/spectral measureの適用条件を確認する。

## Truncated/PRへの拡張案（自動追加なし）

Full PF診断後に必要と判断した場合、Truncated PF一候補又はPR exact-tail一候補を追加する有限案を別に承認する。Truncatedでは保持H/B0/B0Iと元full-H targetを区別し、identity scalarを残す。PRではunitary controlだけのcall範囲を明示し、finite-K corrected/raw・normalizerの余分な取得や再解析を暗黙に含めない。

共通stepが成立しないearly round/actual幅なら、必要な異なるstepごとに別spectrumを評価する別案にする。old13roundを単一H_effへ一括置換しない。実K/κ/R変更・新H6条件をこのpilotへ追加しない。

## 成功・STOPと未採択事項

成功は、SOURCE/binding・scalar・元順unitaryと選択branchの数値診断が成立し、全spectral measureが指定qの直接signalを数値精度内で再構成できること。minimum eigenvalue shiftだけをRPE biasと認定しない。

source/target/retained H・state/sector/scalar不一致、非unitary・非有限、cut/window/alias未解決、数値residual又は保存値比較のallowance超過で、取得済み残差を保存してSTOP。branchをreferenceに近い値へ無断移動しない。閾値緩和・投影・再正規化・自動retry・別幅への乗換えなし。

方法採択には、正しいenergy branch・ground対応・state overlap・他spectral成分・全decoderと誤差couplingの扱いをユーザー/A/C/Bで明示する必要がある。Minkowski予算への移送とraw直接riskの二重会計を区別し、十分N/global RMSE/Gはこのpilotでも未資格のままにできる。

**ユーザー**が有限入力/branch/referenceとscopeを承認した場合に**A/C**が実行し、結果を**B**の既存risk入口へ渡す。今回の技術成果をcommit/push・remote照合した後はmandatory STOP。
