# M1-S Gate 0：source事前調査とheadroom事前登録草案

2026-10-16 JST。**DRAFT／NOT ADOPTED／NO SCIENTIFIC EXECUTION AUTHORIZED**。

**既存H4入力の取得性と変換仕様を整理した。Pauli-tail表、Gate 0の評価runner、科学量W/v/rhoは未作成・未評価。** 本書は[Checkpoint 2 GPT判断][GPT2]に従う技術準備で、実験の正式採択・結果前事前登録の成立・科学実行許可ではない。[候補仕様JSON][SPEC]も**草案、runner入力/authorization/markerではない**。

## 1. 候補目的と判断の範囲

提案する問いは「事前固定した非冗長Pauli-tailとoracle参照stateで、small-step先導減衰係数に追加研究を検討するheadroomがあるか」。主診断候補はW=1−<h²>、v=Var_j<P_j>、rho=(W+v)/(2W)、h=HR/lambda_R。

これらはoracle診断量で、量子回路で無料に使えるpredictorではない。Gate 0だけでRPE精度・shots・native総資源・DF-native PRの改善を認定しない。rhoによる総資源no-go/自動終了、0.7/0.85/abs(alpha)sigma=0.1等の正式閾値は採用しない。

## 2. 取得できたH4入力と旧計算の履歴

正本は下記固定GitHub refの保存NPZ。今回は**raw bytes/SHA256、ZIP/NPY header、metadata_jsonのみ**確認した。数値配列のdecode、stateやHamiltonianの再構成、行列演算、旧runnerの実行はしていない。[source availability][SOURCES]に詳細を保存した。

| 候補 | 固定資料・既存仕様 | 今回確認した範囲 |
|---|---|---|
| H4-DEV-1p00 | [NPZ][DEV]。linear 1.00 Å、STO-3G、8 spin orbitals、DF rank12、charge0/multiplicity1の保存metadata | 8,652 bytes、raw SHA下記、全8 member/header。新Gate 0への正式採択なし |
| H4-HOLDOUT-1p30 | [NPZ][HOLD]。linear 1.30 Å、STO-3G、8 spin orbitals、DF rank12、別Hamiltonian | 8,687 bytes、raw SHA下記、全8 member/header。旧held-outの履歴を保持。今回の追加科学対象には採択しない |

H4-DEV-1p00：

- raw SHA256：`3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。
- 保存Hamiltonian hash：`de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。

H4-HOLDOUT-1p30：

- raw SHA256：`ad7e3e7165c55dbaa395eef7a1dd74db89e1f7ab29a69ac64333f4aebf8b3e37`。
- 保存Hamiltonian hash：`a70d9619e794a0238aae57096a33759bf72d7b294a251b235508cd2ccc6b16c0`。

両方ともN=4、N_alpha=N_beta=2、Sz=0のsector metadata。NPY headerはconstant scalar、one_body(8,8)、lambdas(12)、g_matrices(12,8,8)、sector_basis_indices(36)、state_vector(256)、sector_state_vector(36)、metadata_json scalarを示す。**full Fock長のstateを保存していることと、全Fock groundであることは別**。このspin sectorだけでsinglet証明も得られない。

旧生成は `build_df_h_d_from_molecule(4,distance=d,basis='sto-3g',df_rank=12)`。元のS0はsource `c644925b50587072784846df09bf02c39e8453e1`、authorization `e9bffb85f9ed57712bb83150172a6a4662cecf7f` に従い入力を保存したが、旧pilotのHamiltonian hashとは一致せず `STOP_INPUT_REPRODUCTION_MISMATCH`。後続V1–V3はsource `ef86868a95e9b77eb5cb521ec156f66c7d6c25f6`、[別authorization][OLD_AUTH]の下で保存development入力を検証した。[結果原本][V13]のstatusは `S0_PRIME_PASS_V4_REVIEW_REQUIRED`。当時のlocal/dirty-worktree証拠であり、immutable CIや今回の再検証ではない。旧承認をM1-S Gate 0へ移さない。

## 3. SOURCE・分割の所在と利用できない近道

[DFHamiltonian source][DF]が表す演算子は `constant+sum one_body[p,q] a_p^dagger a_q+sum_l lambdas[l] A_l^2`、`A_l=sum_pq g_l[p,q]a_p^dagger a_q`。保存されたrank12 DF Hamiltonianそのものを将来の再構成targetとし、元の非圧縮integral Hamiltonianへ黙って変更しない。

| 既存source | 静的確認 | Gate 0への境界 |
|---|---|---|
| DFHamiltonian／build／sector solver | DF source §§class/build/solve。solveは固定physical sector | 今回import/solveしない。新geometry・SCF/DF再生成は禁止 |
| [generation_partition][S0SOURCE] | 保存順indexのprefixを明示渡しする | 重みで再順位付けして結果選別しない |
| [split/select][SPLIT] | selectはone_bodyとconstantをHDへ保持、DF block prefixを分割 | 既存lambda_rはDF fragment weight。Pauli lambda_Rへ流用不可 |
| 既存chemistry_hamiltonianのjw_hamiltonian_maker | sourceを読むとMolecularData/run_pyscfを呼ぶ | **凍結DFの変換adapterではない**。今回も将来の同入力変換でもそのまま呼ばない |

元sourceの保存SHAと固定Git sourceを9件照合し、一致を確認した。現在基点の該当sourceもその履歴sourceとbyte一致。sourceそのものの実行・数式の数値検算ではない。

V1–V3の過去結果にはprefix長3/6/9、B2-G/B2-Wの対応がある。草案候補はこの既存集合から、**development 1.00 Å、保存生成順prefix3/6/9の3分割を全件報告**とする（未採択）。

| ID | HDに置くDF block index（草案） | HRのDF block index（草案） |
|---|---|---|
| G3 | [0,1,2] | [3,4,5,6,7,8,9,10,11] |
| G6 | [0,1,2,3,4,5] | [6,7,8,9,10,11] |
| G9 | [0,1,2,3,4,5,6,7,8] | [9,10,11] |

これをユーザーが採択するまでは対象固定ではない。後からrhoが好都合になる分割を選択・追加しない。1.30 Å、endpoint0/12、別rank、別geometry、H6はこの最小草案の科学評価集合へ入れていない。追加するなら結果前の別採択が必要。

## 4. Pauli-tail変換仕様案：まだ実行していない

**DF-native eventとPauli-string eventは別law・別normalizer・別価格**。既存DFのlambda/normalizer/event数/costをPauli側の数値として転記しない。

1. 保存DF表現のselected/unselected blockとone_body/constantから、固定分割のHD/HRを定義する。A_l²の積で生じるfermionic contractionを保持する。
2. 保存spin-orbital順のJordan–Wigner規約を使う。state_vectorはOpenFermionのoccupation basis、mode0が上位bitというsource規約をbindingへ記録する。Qiskit little-endianへ無断で転置しない。変換backend/version/source SHAとstate orderを将来固定する。
3. HRの同じPauli文字列に対する係数は**符号付きで先に全て集約**し、canonical文字列順へ並べる。+P/−Pを別eventのまま残さない。丸めで小係数を削る規則とimaginary residual許容値は未決で、無断の近似cutoffは入れない。
4. **identityをHDの既知scalarへ移す案**を候補とする。移動前後のconstant・controlled relative phase台帳を保持し、Hの意味を維持する。これは元DF分割のscalar配分を変えるため、正式採択時に明示承認が必要。identityをrandom tailへ残す別案と結果を見て切り替えない。
5. identity処理後の非零係数c_Pについて、lambda_R=sum_P abs(c_P)、p_P=abs(c_P)/lambda_R、P_signed=sign(c_P)P、共通θ=lambda_R δ/(2r)を仕様とする。λ、p、Pauli表は今回生成しない。λ=0は空tail／ratio適用外。
6. full Fockで固定DF-Hに対するHD+HR再構成、Hermiticity、basis/state順、identity/phase、確率正規化を将来検査する。DFだけの過去再構成PASSを新PauliモデルのPASSにしない。

**不足している新SOURCE**は、凍結DFからのPauli-tail変換adapter、canonical表と変換receipt、M1-S headroom評価runner、対応する固定source/runtime binding。今回新規実装・科学runner作成はしていない。

## 5. State・全空間・spectral条件の棚卸し

| 資料で確認できたこと | 今回確認できないこと／後続作業 |
|---|---|
| 全DF-Hを表すinput、full長とsector長の保存state、N=4/Sz=0 metadata | 数値stateを数学的にexactな非縮退全H eigenstateと認定する証拠 |
| 旧solverは固定sectorのgroundを求めるsource | 全Fock ground/端点性、全空間でのenergy非縮退性 |
| 旧V1に保存stateの整合・residual記録がある | 今回のPauliモデルでのstate/error/reconstructionの新しい数値検査 |
| full入力から全spaceを定義できる | 保存された完全spectrum、pair-sum gap、near/far coupling tableは今回未取得 |
| DF blockはnumber-conservingな構成 | 個々のPauli-tail eventのN/Sz保存、最小到達spaceはPauli表生成後の別確認 |

Jensen floorは全Hの対象energy非縮退という条件付き。sector groundだけから適用を認定しない。一方、headroom診断を行うために大きな全spaceスペクトル計算へ自動拡張しない。最小Gate 0は**保存数値stateによるoracle診断**として状態精度・証拠限界を報告できるかをユーザー/GPTが判断する。定理の厳密domain認証とoracle数値診断を別labelにする。

## 6. 事前登録案の決定欄

| 項目 | 草案・未決状態 |
|---|---|
| 分子・input | H4-DEV-1p00/rank12を候補。正式selected_input=null |
| 分割 | G3/G6/G9全件を候補。selected_split_ids=null |
| law | 非冗長・符号集約Pauli、identity→HD案。正式identity_policy=null |
| state | 保存full長sector-ground数値参照。exact/nondegenerate認証は未確定 |
| 主量 | W,v,rho。数値は全て未取得、0≤v≤Wは受理済み理論の整合範囲 |
| W=0/λ=0 | ratioを補完せずNOT_APPLICABLE。Wの誤差区間が0を含むならUNDETERMINED |
| 精度 | complex128を技術候補。再構成/state/係数/確率の許容値は全て未採択 |
| 比較 | 同H・law・state・相対時間。総位置Rを揃える場合independent各枝R/2 |
| R,r,δ,s | 主headroomのW/vには不要。実信号を評価する許可はない。有限step/等位置例を置く場合の具体値は未決 |
| dressing | sigma_k(HR)、abs(alpha)sigmaは補助候補。参照α/δ・許容値は未決で、今回計算しない |
| pair-sum/coupling | 最小草案では追加spectrum取得を自動実行しない。必要性・scopeを先に採択 |
| 科学判断閾値 | rho、必要削減率、dressing、総資源目標の正式値はnull |
| 実行source・資源 | runner/version未固定。将来CPU/GPU/メモリと加速経路を確認。科学wall-time capは既定なし、他の資源/出力上限は別途準備 |

将来の入力確認精度には、raw file hashの厳密一致、固定metadata/hashとの同一性、operator再構成許容値eps_H、state norm/residualのeps_state、確率eps_prob、imaginary coefficient eps_imagの**異なる検査**が必要。旧PRの1e-10等を今回の科学閾値として黙って流用しない。値・norm・state誤差伝播・零分母規則は実行前採択の未決事項。

## 7. 将来の結果表とSTOPの意味

結果は採択後の全分割を順序付きで報告し、成功行だけを選ばない。行の候補schemaは input/ref/hash、split index、Pauli table hash、identity台帳、lambda、state/space/誤差、W/v/rho（domain label付）、補助α/sigma、未実施項目、technical status、解釈留保。数値を今回埋めない。

| 状況 | 報告・停止の意味 |
|---|---|
| SHA/再構成/law/basis不一致、必要SOURCE欠損 | TECHNICAL_STOP。当該実験を止める。研究全体no-goではない |
| λ=0、W=0、state/domain条件が未認証 | NOT_APPLICABLEまたはUNDETERMINED。ratioを補完しない |
| headroomが小さい | 当該lawの研究優先度を判断へ戻す材料。全資源改善不能の証明ではない |
| headroomがある | 有限stepや実装の追加研究を検討する材料。科学GO/資源認定ではない |
| dressing警告が大きい | 固定幅T1cの必要性を判断へ戻す。T1cは開始しない |
| 総資源改善が未認証 | Gate 0の正常な限界。旧native価格を足して結論を作らない |
| 研究全体の正式STOP/継続 | ユーザー/GPTの科学判断。自動閾値にはしない |

失敗時に別geometry・別分割・別変換・別閾値へ自動迂回しない。retry、source改訂、対象追加は結果保存後に既存契約と個別承認に従う。現在の草案に実行marker、予約job、scientific authorizationは存在しない。

## 8. 何を決めれば実行準備へ進めるか

まずユーザー/GPTが入力、3分割か別集合か、非冗長化/identity、oracle stateの証拠区分、比較規約、精度と判断基準を採択する。その後、必要なsource/runtime・加速/メモリ設定・技術確認を準備し、**Gate 0科学計算はさらに別の明示実施承認後**とする。草案の公開を採択・結果前登録・実行許可と扱わない。

本調査で新たに必要と判明した科学作業はPauli表生成/再構成、必要なstate精度確認、W/v/rho、任意のsigma/α、採択した場合だけのgap/coupling。**今回の実施は0件**。新SOURCE、T1c、Q4/Q5、DF拡張へ進まず、技術資料の公開後mandatory STOP。

[GPT2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62554ee127a2bec1206367cf1754df2ff49fa000/docs/research/track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md
[DEV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz
[HOLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p30_rank12_held_out_v1.npz
[V13]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/artifacts/pr2_new_series_validation/2026-09-28/pr2_v1_v3_result_v1.json
[OLD_AUTH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54336028edd7ef759f8a0c73bd0a96ade13f47f9/artifacts/pr2_new_series_validation/2026-09-28/pr2_v1_v3_authorization_v1.json
[DF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c644925b50587072784846df09bf02c39e8453e1/src/trotterlib/df_hamiltonian.py
[S0SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c644925b50587072784846df09bf02c39e8453e1/src/trotterlib/pr2_s0_s1_validation.py
[SPLIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ef86868a95e9b77eb5cb521ec156f66c7d6c25f6/src/trotterlib/df_partial_randomized_pf.py
[SOURCES]: ../../../artifacts/track_b_m1s_post_checkpoint2_preflight/2026-10-16/v1/source_availability.json
[SPEC]: ../../../artifacts/track_b_m1s_post_checkpoint2_preflight/2026-10-16/v1/gate0_candidate_spec.json
