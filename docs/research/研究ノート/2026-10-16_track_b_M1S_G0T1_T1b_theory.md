# 2026-10-16：Track B M1-S／G0-T1 T1b解析的移送検証

Checkpoint 1 GPT原本をユーザー指定pathから全文確認・byte copyした。43,022 bytes、SHA256 `f24bf90e3991cd4ed7f7513a75bc1380c03f30ccd90fc09f4d16efea3ba5a32d`、保存commit `4899608ea222777e64e706fda2b7e54b84c86247`。T1aは固定有限次元・有限減衰予算・C-secで限定受理済み、独立新規性未認定、限定T1b推奨である。過去T1aの著者pending記録は履歴として保持した。

今回ユーザーは添付指示でT1bの新規解析的研究を明示承認した。数値・H4/H6新科学量・simulation/sampling・回路/compile・費用・Gate 0/Q4/Q5/DF拡張は未承認。指示原本と承認記録を[証拠directory](../../../artifacts/track_b_m1s_g0t1_t1b_theory/2026-10-16/v1/evidence_index.json)へ保存した。

[証明本文](../../tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md)はS0のexact-half、固定r、小δ_rel、sδ_rel²有界、全H固有stateを維持する。正確なΨから一次A=J_H/2、二次Q=beta D_R、beta=lambda_R²/(4r)を導出。HS散逸性とunitary-product Taylor積分剰余によりT1b固有のKを与え、telescoping・near/far平均化で全s一様O(abs(delta_rel))の複素信号上界を著者証明として提出した。T1aの定数18は転用していない。

単一pair-sum条件の下では、非直交Riesz射影、左右weight、全s residual、finite-step mode log誤差を明示。全H stateでW=1−<h_R²>、v=Var_j<P_j>を使い、先導rateはbeta(W+v)。ordinary片枝r・角2thetaでは2beta W、独立±各枝rではbeta W。W>0で候補rho^(H)=(W+v)/(2W)は指定ordinaryとの先導係数比としてHOLDS。状態dressing、mode差、rate絶対/相対差、ratio分母の誤差を分けた。Outer symmetric splittingはexact HDでもlocal三次・累積O(s abs(delta_rel)^3)として残り、tail有限角とmode-mixingの三次位相も区別した。

二準位Z/Xの非可換分割でWと1−<h_R>²の単純代入の違い、非空のfloor付き位相保証域を手計算で示した。さらに許可された三準位Hermitian involution classで、非零HRと非可換HDを持ちながら対象pair-sumが共鳴する例を構成。C-secは二つの減衰指数の正weight和になり、孤立stateだけからのsingle-rate一般化は成立しない。反例は異符号の同じinvolutionを含む固定lawであり、非冗長Pauli表のみのsubclassの反証と称しない。block上界はその例にも適用される。

著者の区分は主定理/条件付きmode/rhoがPROVED_WITH_DOMAIN、共鳴single-rate一般化の反例がCOUNTEREXAMPLE_IN_DEFINED_CLASS。一般共鳴の代替ratioは未解決、W=0やsingle-rate不能域で元ratioは適用外。size/gap一様性、分子の実用定数、RPE/shot/native総費用、独立新規性は未確立。

[Checkpoint 2 Decision Packet](../../tracks/algorithm_codesign/m1s_g0t1_t1b_checkpoint2_decision_packet_20261016.md)に独立検算箇所、既知性、適用域、反例、継続/縮小/technical note/STOPの判断材料をまとめた。Codexが重要科学的受理・全研究GOを代行していない。新解析的研究あり、新科学数値・科学test・compileは0件。文書/hash/remote byte検査は科学的validationではない。旧source/result/classification/auth/marker・親dirty worktreeは保持した。

現在状態：`T1B_ANALYTIC_TRANSFER_SUBMITTED_WITH_DOMAIN_CHECKPOINT2_PENDING_MANDATORY_STOP`。正式S0と今回指示に従い、次は別途開始承認後のGPT重要科学レビュー。ここでmandatory STOPし、Gate 0/E0、Q4/Q5、DF拡張へ自動進行しない。
