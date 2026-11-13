# 有限部分追跡：承認後の2評価への接続

2026-11-14 JST。[範囲・根拠](REPORT.md)／[FINITE_SCOPE.json](FINITE_SCOPE.json)。現時点で科学実行は未許可・未実行。今回の利用者指示は範囲の具体案までである。

## 固定する入力とAPI

数値kernelは06db9aefeの既存D.evaluate、物理入力/target/loss/IID/whole-vector/ρは43898ada1の既存INPUTSを継承する。A91349のLD1/LD2 canonical ID、元physical candidate、SOURCE/law/H/state/snapshot、prior/branch/alias/shift0、normalization、元Minkowskiと資格flagは変更しない。

追加するoptionだけを、新しい入力/SOURCEへ明示する。

```python
options['state_refinement'] = {
    'source': new_fixed_scope_source,
    'maximum_frontier_states': 8192,
    'maximum_transition_pairs': 141559036,
}
```

新SOURCEで既存callerのevaluate_one等を再利用できるが、旧execute/run01を再開・上書きしない。新callerを必要最小限で用意・固定し、同じ原rawをK.accept_saved/K.prepareへ渡し、**D.evaluateを各1回だけ**呼ぶ。旧結果の数値を新sourceへ付け替えず、同じcount表から旧上界と新DPを計算して比較する。

事前SOURCE/metadata照合は値の欠測・canonical/physical/law/target/整数Nの混在を検出する。既知のmean UNQUALIFIED/proxy未認証は、採択済み条件付き診断として記録を保つ。これを未承認の科学的資格へ変更せず、診断を止めるための新たな資格要件も追加しない。許可はユーザーの明示指示で足り、証明書/grant等を作らない。

## 報告する分解

1. `coarse_MSE_upper`、DPの`MSE_upper`、採用するmin上界と改善幅。旧値は同じ新評価内で算出する。
2. `state_refinement.resolved_MSE_contribution`=η。実解決履歴のlossであり、純shot項と同定しない。
3. `absorbed_unresolved_probability`=u と、そのloss上界寄与u L_U。
4. `residual_probability`=r と、追跡外寄与r L_bad。rは捨てたtailの近似ではなく、未追跡履歴の正確な質量。
5. `any_zero`のexact確率、未解決/tieの区間。r>0なら実全体tieをreference-oracle tieで代用しない。
6. round別保持状態数・群数・累積遷移組、時間/CPU/RSSとUNKNOWN理由。

partial DPでは、吸収したprefixが後でzeroを含む可能性もあり、rにも未解決になる履歴が含まれる。u L_U、r L_badに加えてglobal zeroのlossを別加算してはいけない。global zeroは確率の分離報告に使い、二重計上を避ける。実全体tie/未解決の厳密値は、既存実装でr=0の場合のみ出す。

旧top-levelのunsafe/reference-oracle-tie/未解決範囲は旧診断のまま、新しいDPの確率はstate_refinement側で区別する。保持数Kはround末で、prune前の一時状態数の実測は現APIにない。保持数の記録・count幾何の一時状態数上限・実RSSを分け、実測していない値を埋めない。

## 続くshot評価に向けた課題

- この2件でDPの時間/整数bit長・RSS・残余寄与を把握するまでは、多数配分を処理する費用を楽観視しない。
- 残余が上界を支配するなら、同じNで追跡を増やす技術と、N変更の科学的効果を区別する。勝手にK/組数/配分を増やさない。
- 既存の共通count幾何・整数演算の再利用/高速化は将来の技術候補。今回kernel変更・依存追加・数値近似はしない。
- whole-vectorごとにdecoderを評価し、Nや偶奇・budgetの単調性を仮定しない。元RMSE契約への接続・model閾値・Nの主張水準は別に必要。
- 適格なN/価格が揃うまでは、既存W/B会計へ人工又は未資格の値を入れてG/winnerを出さない。

## STOP

今回の範囲案は未採択。ユーザーの別明示許可前にはH4 probability/risk計算を起動しない。

許可後も、対象は各LD1件だけ。SOURCE/入力不一致、非有限遷移、質量不整合・既存証明条件の失敗はUNKNOWN又は明示エラーとして残す。改善なし・大きなrは自動再試行の理由にせず、旧上界を含む結果を報告する。原raw/結果/STOPを保全し、他担当や監査のbranch/worktree/processへ操作をしない。

2構成を独立2workerで処理できる。監視は自分のworkerだけの読取観測とし、共有環境の設定変更や他processの停止を行わない。正常なworkerを一時的な共有負荷だけで中断・再起動しない。例外/OOM等の未確認結果をPASSにしない。

正式な十分N/global energy保証・shot最適化・価格/Gは対象外。AのδEをraw riskへ加算せず、元契約と未資格を維持する。**現時点はmandatory STOP。**
