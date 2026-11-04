# 追加samplingと、sampler-lawからの決定論的平均確認

2026-11-03 JST。**次は決定論的な局所平均照合を推奨する。大量shotの追加は既定路線としない。**
対象は[選定案](ROUND_PRIORITIZATION.md)のPartial m17/q131072/r32/K2、両軸。
条件はlinear H4/1Å/STO-3G/legacy DF12、physical ground state、L_D3、δ=.02、4電子sector70。
実験は未実施。本書の数式・計算量は静的整理で、H4の新しい性能値や残差を求めたものではない。

## 1. 検証する問いを分ける

1. 宣言されたcategorical event lawと条件付き測定式を正しく平均したとき、V2の保存meanになるか。
2. 実際のPRNG/choice/key分割・event生成・測定streamが、その法則を実現しているか。
3. その測定平均への近さが、固定N/αのRPE energy成功条件に十分か。

決定論的方式は1を精密に調べ、[精度要件](RPE_MEAN_ERROR_REQUIREMENTS.md)の残余評価と組み合わせて3の局所条件を調べる。
samplingは2の実行経路を通るが、必要精度が小さい場合に1や3を強めるには多数の標本を要する。
二つの方式は同じ証明の代替ではない。

## 2. SOURCEから導く一eventの平均

carrier `4591b8d75a2a6a55c74a51225f0c61648023c8ee`の
[rte.py](../../../src/trotterlib/rte.py)を参照した。
`iter_sample_rte_events`は先に偶数order nを選び、その後n+1個のcomponentを独立に選ぶ。
全supportはn=0,2,...,Kと、全てのretained componentの順序付きn+1 tupleである。
identity/sign、productsの作用順、phaseは`_make_event`と`event_unitary`に定義される。

signed involutionをP_i、実際に渡されるcomponent確率を

    p_i = component.probability_i / sum_j component.probability_j,
    A_prob = sum_i p_i P_i

と置く。order確率π_n、theta_n=atan(tau/(n+1))に対し

    M_law = sum_even_n π_n (-1)^(n/2)
             [ cos(theta_n) I − i sin(theta_n) A_prob ] A_prob^n .       (3)

これはsamplerの宣言された分布を直接平均する式である。
P_iが非可換でも、独立なcomponent選択により積の期待値はA_prob^nになる。
同じPをn回使い回す仮定は置かない。

canonical lawでは

    w_n = |tau|^n/n! sqrt(1+(tau/(n+1))²),
    B = sum_even_n w_n,  π_n=w_n/B,
    M_law = [ sum_(k=0)^(K+1) (-i tau A_prob)^k/k! ] / B .             (4)

SOURCEの`finite_taylor_operator`はdegree K+1までのpaired polynomialを作り、
`finite_rte_operator_moments`がcorrected/attenuatedを明示して返す。
order2イベントはdegree2だけでなくpaired degree3も含む点を保つ。
イベントのrotationは`cos(theta)I−i sin(theta)P`であり、angleの二倍解釈や符号反転を持ち込まない。

実装上はπ_nをSOURCEのorder_probabilitiesから、p_iをsamplerと同じ再正規化から取り、式(3)を独立に評価する。
係数由来のA_coeff=sum_i coefficient_i Q_i/lambda_Rと同一になるかも確認する。
同じ近似正規化を無意識に両側へ適用して差を消さない。
`_make_event`のprobability記録と`choice`へ渡す確率の対応をSOURCEで照合する。
float64の確率配列と、有限bit PRNG/choiceの厳密な出力lawを同一と証明した式ではない。

## 3. 一trajectoryの平均を列挙しない

[rpe_hadamard_failure_validation.py](../../../src/trotterlib/rpe_hadamard_failure_validation.py)の
`_trajectory_signal`は、各PF stepでforward half F、r個のevent、reverse half Rを順に作用させる。
matrixとしては右から作用するので

    S_law = R M_law^r F,
    z_law(m) = <psi | S_law^q_m | psi> .                              (5)

独立な全event位置に対して期待値の線形性を使う式であり、一つの乱数回路を多数shot再測定するモデルではない。
tau=lambda_R*delta/r、F/Rは既存partial-S2の同じδのhalfである。
Mは既に1/Bを含むので、式(5)にさらにB^(−q*r)を掛けない。

保存signalの生成元
[finite_rte_signal_validation.py](../../../src/trotterlib/finite_rte_signal_validation.py)は
`corrected_tail=finite_rte_corrected_operator(...)`、`corrected_step=R corrected_tail F`、
`attenuated_round=finite_rte_attenuation(...,tail_evolutions=q) * corrected_step^q`を用いる。
式(5)はcanonicalな式(4)と同じものになるはずで、その接続を**別式・別集約経路**で検査する。

H4の全event tupleや全trajectoryを明示列挙する必要はない。
70次元のAを構成し、少数orderのmatrix polynomialと反復二乗を使う。
概略はbasisのgroup化・cacheでAを一度作り、dense演算でO((K+log r+log q)d³)、d=70。
literal Monte Carloは各標本でq*r回のevent位置のstate-actionを通る。
saved Nをそのまま検証用nに流用する理由はなく、nが小さな許容差に対して二乗逆数で増える点が差になる。
実測時間・speedup倍率は今回算出していない。

matrix-free/sparse作用の既存経路も候補だが、ここでは小さなsectorのdense平均行列を第一選択とする。
長qの作用をq回ループすることや、全event supportをH4で列挙することは最小案にしない。
既存`_normalized_symbolic_tail_in_sector`はbasisごとにdiagonalをまとめてから共役し、閾値で落とした成分を除外する。
このgroup化をp_i重みの経路にも適用できる。

## 4. 共通バグを見逃さない範囲と、残る共通性

保存signalと同じ`finite_rte_operator_moments`だけを再呼出しして同じ値を得ても、独立検証には弱い。
推奨する新しい最小部分は式(3)のliteral-law集約と、probability重みからのA_prob構成である。
これを係数重みのA_coeff/Taylor経路・保存μへそれぞれ比較する。
新しいDF engine、RPE decoder、sampler、compiler、管理基盤は不要。

| 比較 | 検出し得ること | 共通して残ること |
|---|---|---|
| literal-law平均 対 Taylor参照 | order重み、KとK+1、phase/sign、probability正規化、attenuationの二重適用、r内側/q外側の作用順 | 宣言されたモデル・共通metadata |
| A_prob 対 A_coeff | component probabilityと保持係数/符号/閾値の接続差 | 同じDF抽出metadata自体の誤り |
| 二経路の最終complex 対 保存μ | 同じstate・δ・tuple・基底規約でのsaved-model再現 | 元H/DFモデル、Gaussian基底定義・state入力に共通する誤り |
| 小さい人工supportの全列挙 対 式(3) | 非可換product順、負tau、event phase、期待値の合成 | 実H4・長qの数値精度は確認しない |

第一行の「検出し得ること」は同一コードを両側で共有しない場合の条件付きの能力である。
PRNGのstream相関、choiceの実装不良、backend/noise、全controlled回路の正しさは決定論的平均だけでは認定できない。
候補Bも6固定requestの局所意味論で、全event supportやr32への一般証明ではない。

## 5. 方式比較

| 判断項目 | 追加Monte Carlo | 決定論的なlaw平均 |
|---|---|---|
| 直接の問い | 実装されたrandom経路の標本平均・測定過程 | 宣言されたevent lawと保存平均の対応 |
| 小さいmean差 | τとη/powerからnを設計。単なる非棄却は不十分 | 浮動小数点・入力不確かさを除けば全supportの期待値を計算できる |
| Rare event | 観測なしでも平均寄与0とはしない。必要nや既知質量の境界が必要 | 全orderと全componentの重みを含める。未観測tupleも平均に入る |
| 計算負荷 | run02 pipeline再利用可能だがm17のq*rを各shotに掛ける | 保存sector/state、basis cache、matrix polynomial/反復二乗を利用 |
| 独立性 | PRNG・choice・stream配布を実際に通る。ただし同じstate-actionの共通誤りは残る | Taylor経路と別の集約式を追加できる。ただしDF metadata/basisは共有 |
| 残る限界 | 有限標本・選抜/pooling・系統差と偶然の区別 | PRNGや物理測定の実行lawを全部検査するものではない |
| V2への価値 | 選択したτ内を区間で認定できれば局所marginal仮定を支持 | 小さい許容差に対して先に局所モデル接続を精密に検査できる |

Monte Carloを後続で行うなら、検証用のn・η・τ・seed recipeを先に固定し、対象を選んだ理由を記録する。
V2のN/α/βを置換しない。run02は結果を見たpilotなので、確認用標本への無断poolingや
そのrawに対する事後的な同等性判定を行わない。新しい独立streamでの確認を基本案とする。
複数axis/roundと再試行を含めた同時評価を設計し、都合のよい結果までsamplingしない。
条件付きsignalだけを平均する補完案は仮想Bernoulli noiseを省けるが、law/PRNGの検査という問いは残る。

## 6. 人工検証と推奨の限界

[test_design_identities.py](test_design_identities.py)の8 tests PASS。
既存`tests/test_rte.py`の小さい非可換tail・全列挙対paired Taylorの検査を参考に、
standalone rte.pyの既存関数と、元trajectory/reconstructorの変更なしFunctionDefを使った。
追加したliteral-law式を小さい全support、負tau、確率mutation、q/rの合成、phase wrap、zero radius、
同等性と不一致、strict branch境界、最終Bernoulliの一段の周辺化で検査した。
科学module全体・H4 runner・saved rawは使わず、RNG drawもしていない。

推奨は**局所の決定論的平均照合を先に実施し、random経路の追加検査は別の実装上の問いが残るときに補完する**こと。
解析式があるというだけで、実H4数値値やsampler実装を検証済みとはしない。
次の具体的なcall・数値判定の固定は[NEXT_STAGE_DECISION](NEXT_STAGE_DECISION.md)へ。
