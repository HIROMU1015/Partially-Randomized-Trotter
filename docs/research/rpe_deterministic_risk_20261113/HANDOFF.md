# 決定論的risk：条件・数式・入出力

2026-11-13 JST。[REPORT](REPORT.md)／[テスト](TESTS.txt)／[次の科学条件](NEXT_SCIENCE_PLAN.md)。元A→K→Cの[signal/target受入](../pr_ld1_ld2_statistical_caller_20261113/HANDOFF.md)を作り直さず使用する。旧MC/risk_intervalsは照合用に保全し、新経路から呼ばない。

## 1. 定義とdecoder SOURCE

count平均は既存S.count_meanの`(2k−N)/N`をexact整数/Fractionで作ってからbinary64にする。位相はmath.atan2(Y,X)、主値化は元wrap、候補は`wrap((phase+2πk)/2^m)`、前round推定との円周距離が最小のbranchを選ぶ。scalarの完全tieはargminの最小index。旧vectorized argsort規則へ置換しない。wrapperは距離差≤1e−12でUNRESOLVED_BRANCH_TIE、どこかのroundでX=Y=0ならUNRESOLVED_ZERO_COORDINATESを優先する。未解決energy=None、Ehat=−final_phase/h、priorは診断のみ。元energy shift/alias規則を変えない。

実数での最近branch則では、前roundまでの円周誤差≤ρ_(m−1)/2^(m−1)、現在の測定位相誤差≤ρ_mなら、正しい現在候補と前推定の距離は高々`(2ρ_(m−1)+ρ_m)/2^m`。隣接候補間隔は2π/2^mなので、`2ρ_(m−1)+ρ_m<π`なら一意な最近候補になる。理想距離gapの下界は`2(π−2ρ_(m−1)−ρ_m)/2^m`。一律ρ<π/3はこの実数条件の特殊例だが、実wrapperの1e−12と浮動小数点境界を無視して採用しない。円周上の正しい候補でも主値aliasによる実直線energy誤差は別である。

## 2. 今回用いる有限support検証

評価target E*からθ=−hE*を定める。各roundの非零確率countに対して、zeroでなく、宣言した`|wrap(phase−wrap(2^mθ))|<ρ_m`を満たし、target最寄りのlattice oracle自体もtieでないものをG_mとする。これは安全集合の定義であり、rawが単一固有位相モデルであるという資格を付ける操作ではない。

現在位相からtargetに最も近い候補をc_mとし、全good countのc_m集合をV_mとする。m0はwrap(phase)が元初期推定そのもの。以後、**全v∈V_(m−1)と全good現在位相**について、元R3の実際の遷移がc_mに一致し、距離gap>1e−12となることを検査する。この集合は到達可能good履歴のsupersetなので、検査成立なら帰納的に全good履歴は元decoderで解決され、最終出力がc_Mとなる。全count履歴の指数的列挙は不要。

この検査が失敗することは、その量子候補やNが不可能という意味ではない。安全集合・証明方法が不足しているのでUNKNOWNにする。空のgood集合ではall-good事象が空となり、得られるのは全loss範囲による粗い上界である。検証範囲は元scalar互換性を確認したM=0〜12。より長いscheduleはUNKNOWNとして追加の技術検証を要する。

`transition_distribution(table, previous_estimate)`は、**前状態が解決済みで固定**という条件でcount別のzero/tie/branch・oracleとの不一致を返す。roundの`reference_oracle_tie`を全RPEの実tie確率と混同しない。global未解決確率は`P(any zero) ≤ P(unresolved) ≤ P(any unsafe)`というモデル内の範囲で報告する。

## 3. 二項確率とrisk上界

各軸のpをA/Dとし、Binomialの整数分子`w_k=choose(N,k) A^k(D−A)^(N−k)`、共通分母D^Nを保持する。隣接w_kの整数漸化式と総和=分母を確認する。p=0/1は非零support1点だけ扱う。極小確率も0に落とさず、全計数・集計で整数を保持する。log_probabilityは診断表示であり上界の算術には使わない。

probability_ruleは外部指定で、次を区別する。

- EXACT_AFFINE_OF_BINARY64_RAW：与えたbinary64 rawを正確な有理数とみなし、p=(1+raw)/2をexactに計算。
- LEGACY_BINARY64_AFFINE_RAW：旧MC/S.exact_countsと同じbinary64加算・除算後のpを有理数化。

ULP付近では両者が異なり得る。ruleはdeterministic_inputに記録し、後から混ぜない。どちらも真の物理rawやactual samplerへの移送は別資格である。shot数をfloatへ変換しない。degenerate pではint64を超えるNも扱えるが、一般の全count列挙は外部指定の有限範囲を必要とする。

元のIID shots/axes/rounds契約を明示入力する。g_m=P(G_m)、`J_M=E[(−c_M/h−E*)² 1_GM]`として、上記検証成立時のall-good寄与は`(Π_(m<M)g_m) J_M`。初回unsafeの確率は`p_m=(Π_(j<m)g_j)(1−g_m)`であり、その和は`1−Πg_m`。

decoded energyは現実装の主値範囲にある。±π/hを外向き1ULPで広げた端点とE*からresolved loss上界L_Dを求め、未解決lossの外部上界L_Uと合わせ`L_bad=max(L_D,L_U)`とする。したがって、

`model MSE ≤ (Π_(m<M)g_m) J_M + L_bad Σ_m p_m`。

この式は原IID契約の独立性を使用する。全roundの相関が未規定なら使わない。未解決lossは既存summary_optionsのBOUNDED_INTERVALに従うloss familyで、外部model_loss_contractの採択宣言/SOURCEが必要。未採択/UNDEFINEDならMSE上界はUNKNOWN。未解決energyを0等で補完しない。π/T_m規模のdriftは参考式にとどめ、今回の精密上界に使わない。

確率・MSE・閾値比較はexact rational。Ehatは元binary64 decoderの出力値を有理数とみなし、loss差・二乗の集計はexactに行う。RMSE表示はDecimalの平方根近似後にexact MSEとの比較で外向き調整し、MSEを先にfloat化してunderflowさせない。数値raw/solver targetの誤差や、理想実数decoderとの差の包含は未取得のまま保持する。これをHamiltonian ground energyに対する科学的保証へ昇格しない。

## 4. API：既存whole-shotと接続

```python
from trottertracks import rpe_deterministic_risk as D

# config/packet/targetは既存K/C受入、allocationsは全round N_X/N_Yの有限menu。
# options中の全値・SOURCEは外部から明示する。ここでは科学条件を設定しない。
results = D.evaluate(config, packet, allocations,
    evaluation_target=evaluation_target, statistical_contract=statistical_contract,
    integration=I, source=result_source, options=options)
```

optionsは`{source, probability_rule, phase_radii, maximum_round_outcomes, maximum_verification_pairs, model_loss_contract}`。phase_radiiは全round別の明示ρ、二つのmaximumは実施する有限検査の範囲で、上限外はUNKNOWN。model_loss_contractは既存risk機能と同じ`{adopted,source}`の宣言を使い、科学的採択をコードが代行しない。

K.prepareのtarget欠測・canonical/source/law・whole-vector検査をそのまま使う。rawだけを観測確率にし、corrected/B/価格標本数を追加乗算しない。元A target/proxy=false/UNQUALIFIED、元Minkowski、H/state/snapshot/prior/decoderは変更しない。packetはKで受入可能なSOURCE付きviewを渡す。今回の保存Aへの数値適用や、他baseline producerの追加受入実行は含まない。

resultsにはallocation/source/evaluation_target/signal_inputとdeterministic_inputを残し、`diagnostic.deterministic_risk`へ上記結果を入れる。MC_calls=0、confidence=None。n_packet/sufficient_N/global_RMSE_bound/Gはnull、科学的energy qualificationはUNKNOWN。実入力のevaluateは新しい科学計算であり別明示許可が必要。

## 5. 価格・有限選択との接続

```python
comparison = D.compare(config, results, prices,
    model_limit=model_limit, qualifications=external_N_packets,
    **existing_W_cost_options)
```

model_limitは外部の`{MSE_upper,source,scope:'MODEL_MSE_FILTER_ONLY'}`。元εEから直接総risk契約を自動設定しない。exact MSE上界≤limitというfilterを通過した配分だけに、外部N packetを付けてW.compareへ渡す。上界超過は必要十分な不成立証明ではなく、UNKNOWN_OR_UPPER_BOUND_NOT_SUFFICIENTとして残す。

external_N_packetsはallocation_id→既存形式N packet。rmse_evidence.deterministic_inputは当該resultの同名fieldと一致させ、元SOURCE/law/target/全N・optionsを混ぜない。新moduleはQUALIFIED flagを生成しない。人工試験は既存のSYNTHETIC_ONLY fixtureで資格を供給し、科学packetに流用しない。

既存W/BがNの全energy契約、finite-K、物理identity、価格SOURCE・compiler/lowering・full wrapper/P0、model/cohort資格を検査し、exact積和/選択を行う。資格N/価格が不足すればGはUNKNOWN。N総数だけでwinnerを選ばない。複数configurationは各内側結果を既存B.selectへ渡す構成を維持し、新controllerやKLY探索族は追加しない。選択範囲は外部から供給された有限whole-vector集合だけである。

**mandatory STOP**。元Minkowskiと未取得の科学的資格を保持し、実H4 N/G評価へ進まない。
