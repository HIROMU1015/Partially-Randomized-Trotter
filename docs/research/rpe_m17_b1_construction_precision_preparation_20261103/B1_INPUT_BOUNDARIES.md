# B-1の固定入力と構成境界

2026-11-03 JST。準備のみ。以下の実H4の演算は未実行。
対象はlinear H4/隣接1.00 Å/STO-3G/保存legacy DF12、Partial L_D3、単一δ=.02、
Bの保存physical ground state、8 qubit・4電子sector70、m17/q131072/r32/K2。
identity policy `extract_identity_phase`、`coefficient_atol=1e-12`を保持する。

## 原資料の役割

| 役割 | 固定commit |
|---|---|
| 旧library carrier（実装の出所） | `4591b8d75a2a6a55c74a51225f0c61648023c8ee` |
| 元m17科学実行SOURCE | `128e9a5bf2630cd0333ec5f69b81b25def78eb79` |
| 元m17結果・7行列の公開 | `cd749066904d3f29b28b03c19ab69370372b157f` |
| Task A科学実行SOURCE | `4479a1c9cf150250254e2c2650f70fb43841fed1` |
| Task A結果・S64の公開 | `a5304864ec5e1412a36c31574bcb95fb13f80202` |
| Bの保存physical state | `e22d7718c47358ff650dbc04e8e72f6ab1c8bf41` |
| 元のround/shot/予算 | RPE V2 `78c49d0b5f1d71e63b5959e78813237923bbc648` |

[元m17報告](../rpe_m17_mean_validation_execution_20261103/REPORT.md)と
[Task A報告](../rpe_m17_numerical_precision_execution_20261103/REPORT.md)を参照。
Task Aの保存factor固定gapは約6.2431e-10。元診断INCONSISTENT、u_num_bound=null、
RPE_tolerance_certified=false、scientific_status=UNDETERMINEDを変更しない。

## 実際に保存されているもの

元m17の`operators.npz`は下記7キーだけで、すべて70×70の`<c16`。
ここではbyte/hash、NPY headerとJSON項目を確認し、配列を数値としてロード・評価していない。

| 要求上の名称 | 保存先・実際のキー | B-1aでの扱い |
|---|---|---|
| A_prob / A_coeff | NPZ `A_probability` / `A_coefficient` | 二つを独立に固定。相互置換・再構成なし |
| F / R | NPZ `forward` / `reverse` | Task Aと同じ固定factor |
| Reference tail | NPZ `reference_tail` | Task A基準と新tailとの差に使用 |
| Law one-event / tail | NPZ `law_event_mean` / `law_tail_mean` | 同上。既存キーを改名しない |
| τ | RESULT `context.distribution.dimensionless_step_time` | JSONのbinary64へround-tripして固定 |
| Event order / π_n | 同`orders` / `order_probabilities` | 保存0/2と保存確率を固定。再正規化なし |
| B | 同`exact_finite_distribution` | 保存binary64を固定。分布式から再生成しない |
| λ_R / step_time / r / K | RESULT `context.config` | 由来と条件の記録。τをλから再計算しない |
| Order非正規化weight | 同`distribution.unnormalized_order_weights` | 保存あり。π/Bの再構成には使わない |
| 成分probability sum | RESULT `context.component_probability_sum` | 保存集約値のみ。B-1aでは再評価なし |
| 成分別係数・確率・basis | 上記NPZ/RESULTには218成分の完全な構成入力はない | B-1bに不足。aggregate Aから逆算しない |
| ψ | B `physical_state.npz`の`state` | complex128のまま固定。再正規化・再solveなし |
| S64 | 元NPZにはなし。Task A `frozen_steps.npz`の`law/reference` | Task Aで新生成された値。今回はbyte保全確認だけ |

保存Aの既知のFrobenius差約4.80e-16は元結果の引用であり、最終meanのgapへの寄与率ではない。
N411/axis、α=.016957089147912167、ε=.15244325851620388と保存βはそのまま保持する。

## 将来のSOURCE replayで初めて作るscalar

θ64、cos64、sin64、logB64、exp(r logB64)は原NPZ/RESULTに保存されていない。
将来の`--run`内で元SOURCEと同じ`math.atan(tau64/(n+1))`、`math.cos/sin`、
`math.log/exp`を再評価し、値と`float.hex()`、由来を新RESULTへ保存する。
これは**新しいSOURCE replay入力**であって、元保存値の復元ではない。
Taylor float係数も旧`finite_taylor_operator`内で新たに演算される。

B-1aはこの新θ64を正確なdyadicへ持ち上げてHP cos/sinを評価する。
cos64/sin64はfloat replayの比較情報であり、HP経路への固定入力ではない。
Taylorは保存τ64からHPで`−i τ64/k`を評価し、normalizerは保存B64からHPで
`exp(r log(B64))`を評価する。HP atan、τ/λ/B/πの定義式からの再生成はしない。
全保存行列・ψ・π64・τ64・B64・新θ64の持上げは`as_integer_ratio()`を使い、
`str(float)`を正確な入力と取り違えない。保存前に失われた桁は復元されない。

## LawとTaylorの段階対応

| 段階 | Law | Taylor参照 |
|---|---|---|
| Component平均（今回は固定） | `grouped_averages`：p_i / math.fsum(p)とsign、basisごとに加算しA_prob | 独立のc_i/λ_Rを同じbasisで集約しA_coeff |
| Event/order | 保存π_n、phase `(−1)^(n/2)` | K2はdegree3までのpaired Taylor |
| Rotation・積順 | θ64 SOURCE replay。HP `π_n phase (cosθ I − i sinθ A_prob) A_prob^n`を和 | HP `term = term * ((−i τ64/k) A_coeff)`、Iからdegree1–3を和 |
| 内側・normalization | event平均のr乗。保存πにattenuationが含まれ、Bで追加除算しない | polynomialのr乗をHP `exp(r logB64)`で一回除算 |
| 決定論的half | `(R * tail) * F`の左結合 | 同じF/R、同じ順序 |
| Outer/mean | `ψ† [(R tail)F]^q ψ` | 同じq/ψ、再正規化なし |

独立なsigned involution成分の平均をAとしたとき、独立なn個のproduct平均はA^nとなる。
個々の成分同士の可換性は不要。rotationのsignはA_probに含まれ、event phaseは別に一回適用する。
元のhalf構成でidentity/scalar位相を処理済みなので、ここでglobal/scalar補正を追加しない。

代数的に両event式が等しくなるのは、A_prob=A_coeffかつ
`θ_n=atan(τ/(n+1))`、`π_n=[τ^n/n! sqrt(1+(τ/(n+1))²)]/B`、同じ有限K/Bを
整合的に使う場合。保存A、π64、B64、新θ64はそれぞれ丸め済みなので、HP化だけでこの恒等式は強制されない。
πのfloat処理・名目categorical lawと実PRNGの有限表現も、この検証だけでは厳密に同一と証明しない。
LawをTaylor関数へ置き換える処理はない。

## 次の境界

B-1aは保存A以降のtail構成の算術をまとめて変更する最初の検証。
matrix積、cos/sin、Taylor係数除算、normalizationの個別の寄与まで一意には分離しない。
B-1bは、固定した成分別binary64係数/確率・sign/support・basis変換入力を取得し、
追加情報が得られる見込みを確認した場合に別途提案する。SOURCEを再実行して成分を新生成したものを保存原本とは呼ばない。
B-2はπ/B/θ/係数等を元の定義式から整合的に高精度生成する別検証。今回はどちらも未実装・未実行。
