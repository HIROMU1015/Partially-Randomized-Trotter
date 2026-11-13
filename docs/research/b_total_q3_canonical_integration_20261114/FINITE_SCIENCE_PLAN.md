# 最初の有限H4 total/Q3評価案（未承認・未実行）

目的：保存Pauli LD1/LD2の各1 whole-shot vectorについて、Z1＋T1の直接モデルMSE上下界とbias²/Var診断を評価できるか確認する。必要Nの探索や正式資源比較ではない。

## 提案する有限範囲

| 条件 | 提案と採択状況 |
|---|---|
| 対象 | A `714247619afb4714fd663eb15c0470ef32bdc7af` のPauli LD1/LD2、各13round、合計2評価。旧DF signalとは別 |
| configuration | 本統合のopt-in canonical total/Q3 ID（[対応metadata](CANONICAL_BINDING_METADATA.json)）。元H/state/snapshot、S2、h=.1、M12、K2、共通κ、Pauli辞書・roundごとのq/R/allocationを保持 |
| estimator | `consecutive_dyadic_nearest_branch_principal_energy_total_v1`。zero保持・tie computed argmin。主推定量への正式採択はユーザー判断 |
| shot vector | X/Y各軸 `64,60,56,52,47,43,38,34,30,25,21,17,12` を共通の**USER_SUPPLIED control vector**として1本ずつ使う案。旧DFの十分NやPauliのnominal Nとは呼ばない。今回Pauli logBからNを計算していない |
| target | 元full DF-Hの保存solver energyを条件付きproxyとして使う案。値・SOURCE・範囲を明示したevaluation_targetを別入力。ground認証や元packetのproxy採択falseを変更しない |
| risk | 指定raw/IID/targetに条件付く直接モデルMSE。PF/finite-Kの位相ずれを含み得る。純粋shot誤差と同一視せず、AのδEを加えない。元Minkowskiは参照のみで変更しない |
| loss | total decoderの全事象二乗energy loss。旧部分decoderの未解決loss familyは参照coarse証明に使用するだけ。model_loss_contractを新scopeで明示採択する必要あり |
| law | 保存ideal finite-K rawを真値と仮定したshots/axes/roundsのIID、`EXACT_AFFINE_OF_BINARY64_RAW`案。実samplerへの保証なし |
| phase安全集合 | 全round rho=1.0 radの案。新Pauli signalへの閉包成立は未評価。不成立はUNKNOWN |
| count/閉包検査 | `maximum_round_outcomes=4225`、`maximum_verification_pairs=55043173`案。旧固定配分のmetadata上限を参考にした有限範囲で、成立の先取りではない |
| total DP | `maximum_frontier_states=8192`、`maximum_transition_pairs=141559036`を**構成ごとの追跡範囲案**とする。資源capではない。total DPで新たにzero/tieを状態として保持するため、旧DPと同じ到達状態数や残余になるとは仮定しない |
| 旧部分DP | `options.state_refinement`は指定不可。本callerが拒否する。旧固定run01を再実行・上書きしない |
| 出力 | total MSE上下界、lemma/DP/minの適用条件、zero保持/tie採用、追跡残余とそのloss、moments区間、UNKNOWN理由、計算負荷。N/global RMSE/G資格はnull |

上の具体値は**未承認の有限案**である。過去に旧DFの同配分・追跡範囲が許可されていても、新しいPauli/total評価の許可へ流用しない。信頼水準・MC反復数は不要で、整数/有理数による決定論計算と、未認証binary64 allowanceを区別して記録する。

## 実行前にユーザーが決めること

1. Z1＋T1を診断用opt-inとして上記2評価に使うか、主推定量として正式採択するか。後者ならerror contractとcanonical版を確定して別SOURCEにする。
2. 直接モデルMSEのclaim範囲、target proxy、全事象loss、IID/exact-affine、raw数値誤差を今回の条件付き仮定として扱うか。
3. 上記1vector×2構成と、rho・count/遷移・total DP追跡範囲を許可するか。新配分や探索を含めない。
4. 目標energy精度に対するmodel-only filterを記録するか。filter通過だけでは十分N・物理精度保証にならない。allowance/target/state/lawの移送証拠が別途必要。

承認証明書・grant新発行は不要。ユーザーの明示指示で範囲を確定する。今回は実行しない。

## SOURCE・実行とSTOP

- Bの本統合commitを固定し、元A packet・元configuration selector・decoder/risk実装 `710aa69b663ef3313439762d25c80a652a03948f`・C `c628e9e8447c11c984064b922cbce6c52dc72d48`を記録する。
- 既存intake→canonical bridge→`T.evaluate`を使用。count・total DP・lossを新しく実装しない。独立2評価の並列化は可能で、共有環境に変更しない。旧DPの所要時間は参考に留め、新total/Pauliの所要時間を保証しない。
- kernelの非有限・質量不保存・上下界不整合・閉包不成立・source/physical不一致はUNKNOWN/停止。残余を落とさず既存の正当な上界を保持する。失敗後に無断retry・範囲拡張しない。
- 新しい出力先へ保存し、旧DP raw/結果/契約/STOPを保全。今回の2件から追加shot探索、十分N認定、価格・G・winnerへ進まない。
- 公開・remote照合後mandatory STOP。

**現時点は技術統合完了後STOP。科学実行は別途明示許可待ち。**
