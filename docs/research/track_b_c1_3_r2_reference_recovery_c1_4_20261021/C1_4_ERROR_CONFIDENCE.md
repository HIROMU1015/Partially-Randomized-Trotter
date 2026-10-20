# C1-4：誤差・confidence契約草案（未採択）

共通の最終target候補は `z_H(T)=<psi|exp(-i T H_DF)|psi>`。保存DF、R1 exact model、有限Taylor mean、有限bit/実装meanを別の対象として扱う。`epsilon_complex=0.05`、`alpha_familywise=0.05`は既存の未採択設計候補であり、今回採択・最適化・十分Nの計算は行わない。C1-3-R2のT=4/5、q=1、r=4、m=5も限定correctness用で、C1-5の最適条件を意味しない。

| 項目 | 分類 | 既存根拠・成立条件 | C1-5までの不足 |
|---|---|---|---|
| 1. 保存DFとR1 exact model | 必要SOURCE/証拠不足・未解決 | R1のrelative spectral residual `4.609337135740557e-13`は数値診断。exact dyadic係数が正しく戻ることとrigorousな分子operator norm bindingは別。[R1] | `MODEL_BINDING_UNCLOSED`。固定DFとの厳密/認証誤差上界、またはtargetをR1モデルへ限定するユーザーの科学判断 |
| 2. Outer PF | 条件付き適用可能・証拠不足 | 既存symmetric S2のforward deterministic half→tail→reverse half。今回scheduleは確認済み。[S2] | R1各blockとT/qに適用するPF norm bound・定数・strict実装誤差のsource-bound結合。旧経験的C_useをrigorous boundにしない |
| 3. Taylor truncation | 既存保証で適用可能（formal exact-query model） | 正有理p、Hermitian involution、x<=1、m=5に対する既存step bound `epsilon_T=3*x^6/6!`。[A]の`error_budget` | 各実microstep xとの結合、外側/model/native誤差と別計上。式の数値評価は今回行わない |
| 4. Nonunitary meanの積 | 条件付き適用可能 | 既存D1の`(1+epsilon_T)^r-1`と`(1+epsilon_T+e)^r-(1+epsilon_T)^r`。meanをunitaryと仮定しない。[A] | 独立stepの全trial law、各stepの一様e、deterministic halves・全scheduleへの結合。D1実runnerの固定p/r/rowsを無断変更して流用しない |
| 5. Coefficient/finite-bit/root | 既存保証で条件付き適用可能 | 実pi*W=alpha_midはexact。確率丸めetaはweightで補正し、root biasは別。formal mean差<=rho*B_ord、range/momentは既存式(17)(18)。今回FullのP0-P5と6 eventの実weightを確認。[M§4]、[F3] | 全比較方式の実law/precision/terminal結合。FullのP0-P5をcanonical他方式の証明へ移さない。未知global normalizerを予算へ無料注入しない |
| 6. Angle | 条件付き適用可能・nativeは証拠不足 | F3 half-angle/atan enclosure、固定256bit/64項/tau=10^-12の6件guard通過。float変換の有理誤差も保存。[F3] | 全実eventのguard/coverage/終端、binary64時間・scalar phase丸め、native RZ/RZZ strict誤差までの配分。今回の小さいfloat残差は証明の代用にしない |
| 7. Controlled-Q/native strict | 実装が必要・必要証拠不足 | G8はuniform CQ誤差を仮定したconditional boundを持つ。up-to-global-phase誤差はcontrolled相対位相に使えない。[G8B]、[G8P] | DF basis/中央Z・ZZ/rotation/actual adjoint/scalarを含む全supportのstrict native certificate。新direct DF経路へG7 helperの費用・定数をそのまま移さない |
| 8. State preparation | 必要SOURCE/証拠不足 | 今回は固定保存stateのhash/normとstatevector診断。物理準備を実施・認証していない | 実準備回路、誤差、T/CX/1Q/depth/workspace・準備頻度。無料ground-state oracleにしない |
| 9. Readout | 実装が必要・必要証拠不足 | XはH、YはSdg/H、bit0→+1/bit1→-1という既存Hadamard規約を接続。今回はshots 0 | Readout error/校正/費用と実backend。state preparationと二重計上/無計上を避ける |
| 10. Coherent X/Y・familywise | 条件付き適用可能・未解決 | bounded weighted全trial estimatorと既存Bernstein十分予算。[A]の`whole_budget`、[M§6.3] | 誤差配分、比較row/axis family、fresh IID仮定、全N完遂、resource failure配分、全path estimatorの実装・Evidence。新Nは未計算 |

`pi*W=alpha_mid`から、有限bit meanがideal有限Taylorとexact同一になるとはいえない。etaによるproposal変更とroot/angle mean biasを二重加算しない。有限meanからexponentialへの差、DF/model差、PF差も別ledgerに残す。[M§4]

統計契約では各axisの総bias `b_axis`を確定し、残余 `s_axis=epsilon_axis-b_axis>0`、認証second moment `M2_plus`、range `W_plus`、failure配分 `alpha_axis`を先に固定する。既存Bernstein/log/ceil式へこれらを結合するが、今回数値Nを取得しない。十分Nと必要最小N、期待accepted数と実accepted数、tail capとhard capは別である。[A]、[G8B]

Complex精度にはX/Yの誤差からcomplex normを満たすaxis配分、familywiseには全比較row/axisとresource-tail等のfailure配分を定める。旧G7/G8/G10/D/Pilotの固定familyのalpha値・factor 2・primitive errorを今回へ混入しない。deterministic certificateと確率的failureの区別も固定する。[M§6.2,9,10]

全trial estimatorはraw nonreduced/coin rejectを量子実行前zeroとし、分母は全古典trial数。technical failure、precision不足、保存障害はzeroではなく未完了STOP。whole-pathでは各microstepの実weightの積と独立性・zero path判定を実装し、accepted-onlyへ再正規化しない。完走への事後conditioningで未完了prefixへconfidenceを移さない。[F3]、[A]

現時点で新しい非自明定理が必須だとは認定していない。既存のnorm/telescoping/有限bit保証へDF providerを結べない、新しいfamily/all-support保証が必要と判明した場合は`PROOF_REQUIRED`として別承認へ返す。今回、新定理・新保証・新selectorを作成していない。

次の契約ではtarget（保存DF/R1）、state・準備モデル、T/q/r/m、全方式、精度・bias/failure配分、uniform provider誤差、全trial law、RNG/測定仮定、終端/Evidence、one-shot/STOPをユーザーが採択して固定する。必須certificateが欠ける、残余幅が非正、law/support/identity不一致、technical failureなら実行GOにしない。

[R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/model_binding.json
[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_partial_s2.py#L945
[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/rte_development/accounting.py
[M§4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L212
[F3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g6_f3_generator.py
[G8B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g8_bounds.py
[G8P]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g8_pipeline.py
[M§6.3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L480
[M§6.2,9,10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L651
