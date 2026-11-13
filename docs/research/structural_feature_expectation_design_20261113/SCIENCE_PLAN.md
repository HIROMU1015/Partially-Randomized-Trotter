# 必要になった場合の最小有限科学案（未承認・未実行）

本案は実H4平均の計算許可ではない。Dの具体的用途がない場合は実行しない。Dの直接paired残差法が予算を満たすなら、機構model/fitや余分な特徴量を要求しない。

## F1a：線形個数・basis groupだけ

- 目的：Dが指定した**対象round/featureだけ**の平均を、保存確率と解析式から得る。上限は既存1 preparation/13 law。新候補・H6なし。
- SOURCE：Cの9d95d3ea/de2427、law調査d72dc4e6、本実装commit。実行前にreaderと元raw probability/nominal値の保存field・bytesを固定。表示値から確率を再生成しない。
- 入力：保存324 componentの確率・support・basis ID/hash、指定roundのorder0/2 weight。元exactκ/identity/scheduleを保全。preparationを生成し直さない。
- 最初はideal source ratioだけをtargetとし、Fractionで計算。出力は解析的count mean、law/単位/入力SOURCE、表示値の外向きrounding。実sampler距離・compiled価格資格はnull。
- uniform-grid lawも必要な場合だけ、既存F0を同じ有限scopeに明示して組み込む。CDF差は元helperで算出し、PCG64 IIDの数学的証明とはしない。F0を暗黙に開始しない。
- sampling/build/compile/K0/K1/E[S]/実C−S fit/価格/N/Gは0。1回のbatchで評価し、欠測・identity/role/law不一致なら代替生成せず保存してSTOP。
- 成功条件：元D feature定義と同じ単位、指定lawの解析式条件が成立し、数値表現の誤差が記録される。compiled Cや残差modelの精度資格を意味しない。

## F1b：新しい状態依存rewardが本当に必要な場合だけ

- Dが採用候補featureを固定し、まず最小の代表1 lawで検査する。新native featureのstate/reward/entry/close/outer/scalar/axis/terminalを元Accountingから定義する。
- 元S kernelのtransition/状態辞書/境界を再利用できる条件をSOURCEで確認。新featureを失う縮約なら状態を拡張し、その閉包とreward parityを有限supportで再検査する。
- physical reward生成と平均評価は新科学計算なので、F1aの許可に暗黙で含めない。対象feature・law・parity件数・数値契約・実行SOURCEを固定した有限案を改めて提示する。
- 元Bandの外向き区間・mass検査、binary64演算順、literal zeroを保持。大きなjoint状態や計算負荷が必要なら、モデル/特徴量を単純化するかDの限定直接法を比較する。近似・閾値緩和を自動採用しない。
- 失敗/UNKNOWN時は近似kernel・直接compileへfallbackしない。旧K0/K1と26 E[S]は不変。指定rewardのmeanまででSTOPし、価格fit/全Gへ進まない。

いずれも、law誤差、order2条件付き残差、fit/移送精度の不足を解決したことにはしない。追加科学実行は**ユーザーの明示承認後のみ**。
