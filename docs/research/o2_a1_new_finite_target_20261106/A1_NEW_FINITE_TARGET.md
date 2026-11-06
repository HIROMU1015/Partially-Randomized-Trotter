# A1 新finite targetの定義案

2026-11-06 JST。**契約設計案。新targetの正式採択・科学実行許可・取得済み数値ではない。**
元SOURCE候補は固定[6381f816306a6f98d6ea3abc13c1eb5ee8bdf9e6][BASE]。旧来歴の結論は[固定REPORT][OLD]を引き継ぎ、再調査しない。
比較・実行条件は[契約草案](A1_NEW_SCIENCE_EXECUTION_CONTRACT_DRAFT.md)、判断事項は[REPORT](REPORT.md)を参照する。

## 1. 入力とstate

| 項目 | 第一候補（採択前） |
|---|---|
| モデル | 保存linear H4、1.00 Å、STO-3G、legacy DF rank12、8 system qubits |
| NPZ | `artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz` |
| NPZ SHA256 | `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b` |
| 時間条件 | L_D=3、m=0、RPE repetition q_m=1、t=δ=0.02、r=1、K=2、X/Y |
| 初期状態 | **全保存DF Hamiltonian**の全4電子number sectorにおけるground。H_D/PF有効固有状態へ変更しない |
| 対象モデル | 理想gate・fresh IID random-word/Born測定モデル。実backend/noise・元未近似Hの保証は別 |

`DFHamiltonian`の規約は
\(H_{DF}=cI+\sum h_{pq}a_p^\dagger a_q+\sum_l\lambda_l A_l^2\)、\(A_l=\sum G^{(l)}_{pq}a_p^\dagger a_q\)。追加の1/2を入れない。
sectorは既存ablation runnerと同じ `PhysicalSector.number_sector(n_qubits=8,n_electrons=4)` を第一候補とする。暗黙にS_z=0のspin sectorへ狭めない。sector indicesとQiskit/OpenFermionの並び替えを保存する。

研究全体の元Hの厳密ground利用仮定と、本診断の保存DF groundを区別する。状態準備費は除外する。
対応するstate payloadは本契約では未固定。fingerprintだけでvectorの存在・同一性を認定しない。後日、同じH/sector/並びの保存payloadを検証して再利用できなければ、全Hを渡した既存 `solve_df_ground_state` 等の限定solveを別承認scopeに含める。
一意groundが未確認、縮退またはstate精度の保証が不足する場合はSTOPし、別stateやground混合へ自動変更しない。
state norm/sector/最低固有空間への対応と誤差をengine側で確認する。residualだけからgapなしにstate誤差を発行しない。
vector、E0、gap、path別信号、真variance/covarianceはestimator・control選択・配分へ渡さない。state検証用gap等が必要でもengine内に隔離する。

## 2. Partitionとpreparationの定義

`split_df_hamiltonian_by_ld`のweight \(|\lambda_l|\|G_l\|_F^2\) 降順、同weightではoriginal index昇順を使用する。先頭3 fragmentをD、残りをRとする。今回は順位・weight・indicesを計算していない。
one-bodyはD、constantはscalar。`prepare_df_partial_s2`に `partition_policy='weight_ranked_prefix'`、`identity_policy='extract_identity_phase'`、`coefficient_atol=0.0`、`diagonal_sort='descending_abs'`、`basis_construction_policy='fermionic_gaussian_jw'` を明示する。
random側の `lam*(sum eta_k n_k)^2` を既存 `exact_df_diagonal_coefficients` のI/Z/ZZ展開で得る。threshold条件は `abs(coefficient)>0.0`、exact zeroだけを除く。WP04の1e-12処理は混在させない。

展開されたidentityをκIとしてscalarへ抽出する。非identity coefficientをc_i、sign-absorbed involutionを \(P_i=\mathrm{sign}(c_i)U_i Z_{S_i}U_i^\dagger\) と書く。
`exact_rte_lambda_r` はSOURCEの `math.fsum(abs(c_i))` をbinary64へ格納したλ_f。component確率、basis metadata/runtime操作、符号、support、並びを同じpreparationから保持する。
「symbolic/exact」という実装名を、元Hとの厳密一致や浮動小数点誤差0の証明とは扱わない。

後日のN1は完全partition/weight、pre-threshold係数（identity/exact-zeroも含む）、retained catalog、D block payload、全basis操作・選択policy用basis操作、λ_f、κ、config/B、環境・source hashを保存する。
既存 `preparation_payload` と、extraction内で計算済みのpre-threshold/hash payloadを読み取り専用で返す最小差分を設計対象とする。後者は現返却値だけでは完全取得できないため、新SOURCEに差分を固定して受け取る。serializationのために再度eigh/係数生成を行わず、旧診断のPREPARATIONを新inputへ転記しない。

## 3. 一次target：実law・selected full interrogationの複素信号

canonical samplerのactual lawをQと書く（RPE repetition q_mと区別）。
\(\omega=(n,L_0,\ldots,L_n)\)、n∈{0,2}。Lは**適用順**products→rotation。n=2のL0/L1はproducts、L2がrotationである。
実抽出はorder→label配列、配列内部はrotation→products。`_make_event`には `(L_n,L_0,...,L_{n-1})` を渡す。
full-basis参照eventは
\[
V_\omega=(-1)^{n/2}\exp[-i\phi_{n,f}P_{L_n}]P_{L_{n-1}}\cdots P_{L_0},\quad
\phi_{n,f}=\mathrm{float}(\mathrm{atan}(\tau_f/(n+1))),\quad\tau_f=\mathrm{float}(\lambda_f t).
\]
n=0のproductはI。signとevent phaseは回路に含め、測定後に再度掛けない。

D blockのSOURCE格納順をD0,...,Dsとし、forward halfは \(F=e^{-itD_s/2}\cdots e^{-itD_0/2}\)、reverse halfは \(R=e^{-itD_0/2}\cdots e^{-itD_s/2}\)。指数表記は構成を示す。一次targetのF/Rは最終gate列の各binary64 parameterをexact angleとして解釈して定義し、再計算した指数行列で置換しない。SOURCEが生成するhalf-stepの格納角度・scalar位相の丸めも記録する。
`support_run_le_1` と `boundary_optimized` を同じωへ適用したsystem unitaryを \(U^{sel}_\omega\) と定義する。
格納scalar位相をθ_c,f=float(−t*c)、θ_I,f=float(−t*κ)として、full-basisの参照構成は \(U^{full}_\omega=e^{i(\theta_{c,f}+\theta_{I,f})}R V_\omega F\)。selected policyの数値基底操作はこの参照との厳密同値を未検証のまま仮定せず、一次targetでは**実際に凍結するselected gate列**でUを定義する。

各ωにordinary controlled \(\mathrm{diag}(I,U^{sel}_\omega)\)、ancilla H、XはH readout、Yは**Sdg→H** readoutを用いる。bit0→+1、bit1→−1で
\[
f_X(\omega)=\Re\langle\psi_{DF}|U^{sel}_\omega|\psi_{DF}\rangle,\quad
f_Y(\omega)=\Im\langle\psi_{DF}|U^{sel}_\omega|\psi_{DF}\rangle.
\]
scalar/identity位相はcontrolled枝のPhaseGateに入り、global channel同値だけではこの信号の同値にならない。
\[
Z_{new}=\sum_\omega Q(\omega)\langle\psi_{DF}|U^{sel}_\omega|\psi_{DF}\rangle,
\qquad\mu_b=\mathbb E_Q[Y_b]=\mathbb E_Q[wY_b],\quad w=1.
\]
**一次targetはattenuatedなこの複素信号**。B_fをshot weightに掛けていない。平均channel \(\mathbb E[U\rho U^\dagger]\) は別物であり、本Hadamard targetを決めない。

compiled wrapperは同じlogical full回路から生成して価格を測る。engineがcompiled回路を使うならlogical targetとの誤差をb_engineへ含める。無視する場合は理想compiler同値という条件付きモデルに限定し、実機精度保証とは呼ばない。

## 4. Actual finite-grid law

NumPy1.26.4の既存 `iter_sample_rte_events` をunconditional streamに使用する。
`finite_rte_distribution(λ_f*t,2)` のorder入力と `_validate_components` のfsumで再正規化したlabel入力を、同じ演算順の `connect_sampler_inputs` へ渡す。
各choiceはbinary64 `p.cumsum(); cdf/=cdf[-1]` とright-search。M=2^53、cdf境界C_jをexact binary64値として
\[
Q_j=(\lceil MC_j\rceil-\lceil MC_{j-1}\rceil)/M,\qquad
Q(\omega)=Q_n\prod_{j=0}^{n}Q_{L_j}.
\]
`law_from_choice_cdf` のexact Fractionをν・層質量・estimatorに共用する。zero grid massを持つsymbolic componentを黙って再正規化しない。
`event.event_probability`はnominal floating値でありactual Qそのものではない。これをcontrol平均やimportance weightに代用しない。
PCG64の固定seed列をIIDの証明とせず、fresh independent uniform wordsという統計モデルを明記する。stream名・seedの相違だけでは独立性の証明にならない。

## 5. Algebraic referenceとbiasの位置付け

\(S=\sum_i|c_i|\)をexact記録係数の和、\(\widehat H_R=\sum_i|c_i|P_i\)をfull-basis参照とすると、理想paired K2平均はdegree3の \(P_3(-it\widehat H_R)/B(tS)\)。\(B(x)=\sqrt{1+x^2}+x^2\sqrt{1+(x/3)^2}/2\)。
この参照、保存DFの真時間発展、一次targetを区別する。SOURCEのλ_f/order weight/atan/B_f、finite-grid law、selected basis操作の数値差は後日この参照との接続で評価する。

| 接続 | 初期1-round診断での扱い |
|---|---|
| 元未近似H→保存DF H/state | 範囲外。誤差0とも元Hへの保証とも言わない |
| PF・有限RTE cutoff/attenuation・Q対algebraic law | 一次targetの定義に含む。μ_newを推定するための追加biasではない。物理誤差が消えた意味ではない |
| ideal full basis対selected policy | 一次targetはselected実列。参照へのb_selは別途取得が必要 |
| engineのstate/probability/浮動小数点/測定bit-grid/必要ならcompile誤差 | μ_newの保証に必要なb_engine。未取得値を0にしない |
| CV centering | exact Q/ν/有理数補正ならそのモデル内で0。丸める実装なら \(b_{center}\le\sum|\lambda_j|e_j\) |

[旧state-free bound実装][BOUND]はalgebraic接続の部品として再利用可能だが、旧run・旧target用guardを削除して成功扱いしない。新targetのPF/finite-RTE/selected/数値を全て保証する関数ではない。
初期の複素mean比較に半径下界は不要。将来RPEへ接続するには新targetでのPF/RTE/normalization/law/実装誤差、正の半径下界、branch margin、round間の同時失敗配分、energy換算を別に閉じる。

## 6. 同じtargetで比較できる条件

B0/B1-O/B1-OL/A1でsnapshot、state/sector/order、preparation、Q、selected policy、phase/readout、w=1、precision/α、ideal/native境界を共有する。
B1はQ(ω|h)を取得し既知Q(h)で合成する。A1は同じQに対するνでcenterする。条件付きsamplerは別のPR targetや別weightを導入するものではない。
新contractのidentityはN1で生成したpayloadを基準にする。以降は同一run内でこのidentityを要求し、別環境で差が出たときは許可なく比較対象を置換しない。
旧partition/preparation/tail hashとの一致は要求しない。旧artifactは履歴と実装設計の参照に留め、旧価格・ν・state半径・ε/α・N=510を新結果として移植しない。

[BASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/6381f816306a6f98d6ea3abc13c1eb5ee8bdf9e6
[OLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6381f816306a6f98d6ea3abc13c1eb5ee8bdf9e6/docs/research/o2_a1_opt2_provenance_20261106/REPORT.md
[BOUND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6381f816306a6f98d6ea3abc13c1eb5ee8bdf9e6/src/trottertracks/algorithm_codesign/a1_law_validation.py
