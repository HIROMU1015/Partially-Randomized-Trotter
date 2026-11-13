# 状態集約上界：成立条件・数式・API

2026-11-14 JST。[REPORT](REPORT.md)／[次の有限範囲](FINITE_PLAN.md)。既存[decoder/law/旧上界の仕様](../rpe_deterministic_risk_20261113/HANDOFF.md)を保ち、追加部分だけを定義する。

## 状態と集約が正当な条件

固定M≤12、consecutive dyadic schedule、元scalar互換R3、同じcount変換・prior/alias/shift0を使用する。元decoderは現在roundのprincipal測定位相と前round推定値だけで次branchを決め、priorはclipに使わない。履歴中にzero又はtieがあれば、後で位相候補を選んでも最終energyはNoneのままである。元wrapperのzero優先とtie許容差≤1e−12を保持する。

従って、round mまで未解決になっていない履歴は、**推定値のbinary64 bit列が同じ場合だけ**まとめられる。float.hexをkeyとし±0も分け、近似bin/区間化を使わない。round countもzero flagとprincipal位相の完全一致だけでまとめる。IIDに条件付けば同じ状態の将来分布は同一なので、整数確率質量を合算できる。branch履歴は将来のdecoder規則に影響しないが、誤branchやaliasによる実直線energy lossは終端値へ残る。

未解決zero/tieには**共通の採択済みBOUNDED_INTERVAL [l,u]**を適用するため、未解決履歴は吸収できる。zero/tieで別loss・履歴依存のlossを採用する場合は、この集約証明をそのまま使えない。現APIはそのような変更を実装していない。

## 残余を含む上界の証明

元round表の共通分母D_mと整数weightを再利用する。prefix全体も共通分母∏D_mを持ち、遷移時は整数weightを乗算・加算する。吸収済み/残余質量にもD_mを掛け、毎roundで質量和=分母を確認する。確率のfloat化・tail切捨てはない。

解決中の状態を元R3._nearestで遷移させる。前状態方向と同様に、count位相方向をNumPyでまとめて評価する。外部の遷移組範囲で処理できない前状態は、将来全履歴を残余へ移す。round末の保持状態数を超える場合も、共通分母の整数質量が大きい順に保持し、残りを残余へ移す。順序の同率はhex keyで決定する。

最終的に、保持した解決済み履歴の状態sと確率w_s、吸収した未解決質量u_0、追跡しなかった残余rが互いに素に全履歴を覆い、Σw_s+u_0+r=1となる。η=Σ_s w_s(−s/h−E*)²とする。Ehatは元binary64出力、差と二乗の集計はexact rationalである。

既存の主値由来resolved loss上界をL_D、L_bad=max(L_D,u)とすると、任意の元loss family内の全事象risk Rについて、

`η + u_0 l ≤ R ≤ η + u_0 u + r L_bad = U_DP`。

保持した解決履歴では実際のloss、既知未解決では同じloss契約、未知残余では旧最大lossを適用するため成立する。元上界U_oldも同じモデル・契約で有効なので、

`U_selected = min(U_old,U_DP)`。

r=0では全解決履歴のlossと未解決確率がexactに計算でき、[η+u_0 l,η+u_0 u]はloss familyの両端である。これは未解決energyを補完した単一推定器のRMSEを定義しない。l=uの場合だけ指定loss下のMSEが一点に定まる。

## zero/tieの範囲

独立round count表からz=P(any zero)をexactに求める。最終未解決確率p_Uは`max(u_0,z) ≤ p_U ≤ u_0+r`。zero優先と未解決statusがzero/tieのみという既存規則から、実全体tieの確率はp_U−zであり、範囲は`[max(0,u_0−z),u_0+r−z]`。r=0ならexact値となる。reference-oracle tieを代用していない。

## 既存APIの最小拡張

```python
# 値・SOURCEは外部で指定する。科学実行の許可や追跡範囲を自動採択しない。
options['state_refinement'] = {
    'source': refinement_scope_source,
    'maximum_frontier_states': declared_states,
    'maximum_transition_pairs': declared_pairs,
}
records = D.evaluate(config, packet, allocations,
    evaluation_target=evaluation_target, statistical_contract=statistical_contract,
    integration=I, source=result_source, options=options)
```

両maximumは明示した非負exact整数。0なら全質量を残余で囲める。CPU/RAM/workerの制限ではない。状態数は**round末の保持数**であり、round内の一時メモリ上限ではない。transition_pairsにはm0の仮想初期状態×count組も含む。旧maximum_round_outcomes/maximum_verification_pairsは別に維持する。

既定なしでは旧経路を保持。指定ありでは同じround表から旧・新を計算し、二項PMFは二重計算しない。現在の実装は旧count/good閉包/明示loss契約が成立した後だけDPへ進む。DP自体は実遷移を使うが、旧閉包が不成立の場合を新たに迂回する機能は今回追加していない。

出力`diagnostic.deterministic_risk`：

- `coarse_MSE_upper`：旧上界。`MSE_upper`：成立したmin上界、既存のexact model filterへ渡せる。
- `state_refinement`：DPのloss下上界、既知解決寄与、吸収未解決/残余確率、保持状態数/遷移組数、改善した未解決/tie範囲。`residual_probability=0`の場合だけ実tie/未解決のexact値が入る。
- 元`rounds.unsafe/zero/reference_oracle_tie`とトップの`unresolved_probability_bounds`は旧診断のまま。改善した値は上記の子fieldを参照し、別概念を上書きしない。
- count質量不整合・非有限遷移・loss上下矛盾等は全体をUNKNOWN/MSE=nullにする。旧値はcoarse fieldへ記録するが、検証失敗を隠して選択へ進まない。scopeにより残余を囲めた正常な部分DPはUNKNOWNではなく、有効な保守的上界である。

scientific_energy_qualification=UNKNOWN、n_packet/sufficient_N/global_RMSE_bound/Gはnullのまま。deterministic_input.optionsに新scopeも結び付き、既存D.compare/W.compare/B.selectの外部N/SOURCE検査を再利用する。人工の資格fixtureを実Nへ移さない。rawを観測に使い、normalizationの再乗算やAδEの加算はしない。

**限界**：確率・loss集計がexactでも、binary64 raw/solver値・保存state・actual lawの科学的資格や実数decoderとの包含は取得しない。元Minkowski契約も変更しない。上界改善は科学的energy精度保証ではない。実H4への適用は別明示許可後のみ。
