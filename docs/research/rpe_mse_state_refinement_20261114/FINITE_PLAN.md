# 実H4の有限適用とshot評価の準備（未実行）

2026-11-14 JST。[REPORT](REPORT.md)／[数式・入力](HANDOFF.md)。今回の許可は技術実装・人工検証まで。以下を自動実行しない。

## 第一段階：同じ2入力で上界だけを比較する案

対象はA91349の保存PR-LD1/LD2各1構成。元INPUTSは`43898ada12e405eacda27b5bccf1d464a0362ed2:docs/research/pr_ld1_ld2_deterministic_risk_results_20261113/INPUTS.json`。target proxy、loss、exact-affine確率、ideal finite-K IID、ρ1、全13roundのX/Y vector `64,60,56,52,47,43,38,34,30,25,21,17,12`は変更しない。

**提案する科学実行は各1件、計2 D.evaluateだけ**。1回の評価で同じcount表から旧・新上界を計算する。旧run01を再開・上書きせず、新SOURCE/caller/入力と新出力先を固定する。旧callerのexecuteは元src・既存run01を保護するため、そのまま再起動しない。必要なら既存callerの`evaluate_one`を新しい有限callerから再利用できる。新controllerは不要である。

科学実行前に**ユーザー**が、新しい状態数・遷移組の有限追跡範囲を選び、上記2件の実行を明示許可する。その他の既に採択された条件を再採択させる必要はない。許可のためのgrant/証明書は不要。

追跡範囲の選択肢：

1. **完全状態集約を包含**：C_m=(N_mX+1)(N_mY+1)、q_m=2^mとして、保持数を`max_m q_m C_m`、遷移組数を`C_0+Σ_(m≥1)q_(m−1) C_(m−1) C_m`まで許す。この有限上限は全候補状態を包含し、実数値のraw分析から決めない。ただし計算時間・整数bit長・一時メモリは大きくなり得る。
2. **明示された部分追跡**：保持数/組数を外部指定し、残余に旧L_badを使う。同じ2件から部分上界と残余寄与を報告する。これは計算資源の恒常capではなく、どこまで細分化して評価するかという範囲。改善しない場合もあり、許可された範囲外へ自動拡大しない。

判断目的は、保守性が実解決履歴の一律最大lossに由来するのか、元未解決loss family又は追跡残余が支配するのかを区別すること。実データでの残余・寄与・計算量は未取得であり、今回予測値を作らない。独立2構成は2worker等で並行可能で、共有設定や他担当processを変更しない。

成果は旧/DP/minのMSE上界、η、吸収U、残余r、zero/tie/未解決確率のexact値又は範囲、保持状態数・遷移組数、時間/RSS、UNKNOWN理由。十分N・global bound・G/winnerは引き続きnull。

## 第二段階：必要Nを検討する有限whole-vector案

第一段階の結果を見てから、**ユーザー**がmodel riskの採用水準・比較閾値・候補集合・本数を明示する。現在の`RAW_MODEL_DIAGNOSTIC_ONLY`は、元Minkowskiから自動導いたMSE≤εE²の正式契約ではない。model filter閾値を外部指定しても、元energy taskの資格とは区別する。

候補の作り方は既存W.prepareのallocation schemaとK.prepare/D.evaluateを使う。以下は未採択案で、数値評価していない。

- 元control vectorを基準に、有限集合の共通shot倍率s_Nを指定する。各roundを別々に最適化せず、生成した全13round/X/Yを一組として評価する。
- zero構造を調べる対照として、正本vectorのeven Nだけを次のodd整数に変える共通vectorを候補にできる。一般に各roundのどちらかの軸がoddならX=Y=0 countは起きないが、tie/誤branch・riskの十分性は別である。これはloss変更でもNの単調改善保証でもない。
- 適格価格が揃った後、有限のcost-aware whole-vector候補も同じdecoderで評価する。shot総数順やround別の独立選択をG最小の代用にしない。

各配分は独立allocation ID/SOURCEを持ち、元physical configuration IDは変更しない。Nはexact整数。偶奇・branch/tieのため、Nや追跡budgetに対するrisk上界の単調性を仮定した二分探索を行わない。候補数・倍率集合・X/Y規則は勝手に固定しない。

上界≤採択したmodel閾値なら、その条件付きモデルについて十分条件を示せる可能性がある。上界超過だけでは真のrisk超過や物理候補の不可能性は判定できない。有限menu内の結果からglobal最適Nを主張せず、未探索域を残す。正式N packet producerは今回追加していない。

## G・誤差契約・STOP

Gへ進むには、対象Nの資格と同identity/law/compiler/lowering/full-wrapper/P0の適格価格が必要。既存D.compare→W.compare/B.selectとexact会計を再利用し、無資格価格・構造proxy・shot総数でwinnerを作らない。C/Dへ今回新価格要求は出さない。

元full-H proxyに対するraw riskへAのH_eff由来δEを再加算しない。物理/数値/実lawの資格とMinkowski接続は別の未解決事項である。

SOURCE/identity/law/target/vector不一致、旧証明条件の欠落、非有限遷移・質量不整合ではUNKNOWN又は明示エラーで停止。状態budgetで落とした質量は消さず、旧lossで囲む。MC fallback、追加配分、自動再試行・budget拡大、実N/G認定をしない。

**現時点はmandatory STOP。** 次はユーザーが第一段階の追跡範囲と有限2適用を許可した場合だけCodex Bが進める。
