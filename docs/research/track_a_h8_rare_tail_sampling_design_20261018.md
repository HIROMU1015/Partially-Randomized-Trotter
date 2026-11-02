# H8 rare-tail：条件付きsamplerとnative接続の設計

準備のみ。実H8へ本書の数式・kernelを数値適用していない。SOURCE・保存lawは[統合報告](track_a_h8_rare_tail_selection_readiness_20261018.md)と[入力manifest](../../references/h8_rare_tail_selection_inputs_v1.json)。

## count → 一様位置 → 元component生成

元lawは独立occurrence k∈{0,2,4}、確率(p0,p2,p4)。R=q*r。R!/(n0!n2!n4!)個のorder wordは同じcountなら等確率。

1. s4はn4=1…R、n2=0…R−n4を列挙。s2_multiはn4=0、n2=2…Rを列挙。
2. `log C(R,n4)+log C(R−n4,n2)+n0 log p0+n2 log p2+n4 log p4`をlog-sum-expで正規化。確率が0のcategoryは正のcountを持たない。πが小さいための無条件rejection samplingは使わない。
3. 条件付きcountを一度sample。一様なlabelled occurrence permutationの先頭n2をorder2、次n4をorder4とする。位置固定・順序のsort・ちょうど1/2個への固定なし。
4. occurrence iをstep floor(i/r)へ戻す。元component probabilityでk+1個のindexを生成。**index0がrotation、その後k個がproducts**。元`_make_event`が実際のapplication sequenceをproducts→rotationへ構成する。ここを取り違えない。
5. 元_make_eventはphase、rotation angle、coefficient sign、basis/fragment、normalizationを保持。条件付けは費用標本設計だけなので、量子演算子の係数にπでの再重み付けをしない。
6. 同じevent列をcosine/sine両方のwhole-wrapperへ渡す。別seed・再samplingは拒否する。

countの多項係数と一様位置の逆数が打ち消し合い、各order wordの確率は元の積確率/層確率になる。component抽選が次数から独立な元確率を保ち、元event変換が同一なら、event列の条件付きlawも一致する。これは数学的導出であり、実H8での新しい数値検証ではない。

## 数値安定性と同値性の限界

log-spaceで小さいπを大きい数の差から作らない。s2_multiも差分式ではなく正のcount重みの和を使用。πそのものがunderflowしても`log_stratum_mass`を保持する。

double precisionで正規化後の極小cellが0に丸まる場合は`underflowed_count_cells`へ記録。将来native adapterは`require_representable`で拒否し、任意のcellを黙って省かない。高精度化が必要なら別SOURCEで準備・検証する。人工極小fixtureはこの拒否まで確認する。実H8のcount値をここで評価して「問題なし」とは認定していない。

標準libraryの人工RNGはfixture用。将来のnative版では固定PCG64と独立したorder/component streamを使い、元component index生成のAPI・順序を保つ。新namespaceはstage/candidate/replicaで分離し、旧探索/確認/pilot/STOP済み試行のmaster/order/step seedとの衝突をfreshに拒否する。元seedで元eventをbit再現することと、新条件付き標本の分布同値性を区別する。

O(R²)のcount tableは同一R/law/層でキャッシュし、trajectoryごとに作り直さない。nativeへの接続前にPython/NumPy/PCG64・providerを封印する。今回numpyや分子packageのimport、native build、実H8 samplingは0。

## native adapter blueprint

元SOURCE`03479a75…`のadapterは、保存DF/preparationの検証、`h8_fast_cost_port_v1.cost_task`への限定注入、同trajectory evolutionの両軸再利用、固定`compile_owned`、元whole-wrapper metrics抽出を持つ。既存s2 `validate_orders`は0/2専用なので、そのままs4に流用しない。

将来の新adapterで差し替えるのは条件付きorder/count生成とそのmetadataだけ。`make_rte_config`/分布literal、tail component、JW順、scalar、basis cancellation、symmetric-directional、state P0、compilerは元providerに固定する。bindingにSOURCE closure、入力16roleのbytes/SHA、tail/preparation identity、compiler/profile、event digest、seed群を含める。

新SOURCEで全wrapperを実際にbuild/compileして初めてMEASURED_NATIVE_WRAPPERと表示。今回のcallbackはMEASURED_MOCK_WRAPPERしか受け付けず、MODEL_ESTIMATEとは別区分。mock PASSはnative cost/fingerprint/controlの再現証拠ではない。

先頭trajectory/qで元control検査を行い、それがPASSした後だけ同候補の残りを並列dispatchする案。q4完了後q8という旧pilotの順序を再採用するかは新契約で固定。1e−8不一致、compiler/input/event drift、native異常終了、実際のOOM/継続不能、call超過でfirst STOPを保存し、owned processだけ回収。無関係なHEAD変更・一時的PSI/CPU競合では強制終了しない。

## 後続の感度stage：式と入力だけ

保存shotで重み付けたtrajectory費用をG、対照点費用をDとする。候補ごとに

`μ_G = π0 μ_0 + π2[(1−w) μ_2_single + w μ_2_multi] + π4 μ_4`、
`w=P(N2≥2|s2)`。

例えば、他成分を明示的な仮定値として固定するなら、逆転境界は

`μ_4* = [D−π0 μ_0−π2((1−w)μ_2_single+w μ_2_multi)]/π4`。

同様にμ_2_multiの閾値もその係数で解く。π4=0、w=0、欠測入力・不適格shotは別statusで停止し、除算・0補完しない。μ0は元n32全0の条件付き実現値であって既知母平均ではない。今回のs2 n8を無条件n32に混ぜず、N2=1の事後部分標本とs2母集団estimandを混同しない。

必要入力は保存law/count vector、候補別保存shot、固定対照費用、各層の実測/仮定label、bias namespace・倍率・ε。出力案は仮定ごとのsymbolic/数値閾値と、欠測層・保証不能欄。母平均推定やPR winnerの再会計へ自動で接続しない。**この式の実H8数値評価は未実施・別承認。**
