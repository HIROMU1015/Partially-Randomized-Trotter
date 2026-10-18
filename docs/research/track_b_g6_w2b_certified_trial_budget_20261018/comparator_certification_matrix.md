# 対照方式の認証・実装・比較可能性

`CERTIFIED_BUDGET_FORMULA`は条件付き数学budgetの所在、`CONDITIONAL_COMPARISON`は同じ前提での比較式、`IMPLEMENTATION_INCOMPLETE`は実装/物理層の不足、`INSUFFICIENT_CERTIFIED_INFORMATION`は具体値・identity不足を示す。`REQUIRES_NEW_PROOF`を実装不足の同義語にしない。今回は比較を閉じるために必須と判明した新しい非自明証明は0件。科学取得・W2cは`OUT_OF_SCOPE`。

| 方式 | finite target / precision / 認証式 | SOURCE状態・必要情報 | 今回の比較statusと限界 |
|---|---|---|---|
| Ordinary | 同じP_m、η/ρ/H/K/角誤差/ε/s/ell/α。κb_O²、λb_O、ceil | [G] CanonicalGenerator＋budget。b_O=oは同じ固定root列で一致。全canonical law guardとfresh IIDが前提 | CERTIFIED_BUDGET_FORMULA / CONDITIONAL_COMPARISON。固定同K経路ではN_F≤N_O。strict N差はceil条件。F3非zero入口はIMPLEMENTATION_INCOMPLETE |
| Partial Return＋ordinary tail | 同じP3＋tail=P_m、同条件。κb_P²、λb_P、ceil | [G] partial_return_tail、group root/label/first/conditional law。b_Pは実B.hi | CERTIFIED_BUDGET_FORMULA / CONDITIONAL_COMPARISON。C2/I1条件を使用。具体inputの上端未評価でINSUFFICIENT_CERTIFIED_INFORMATION、全域優位は未認定 |
| Closed P3＋ordinary tail | 同じP_m、同条件。κb_3²、λb_3、ceil | [G] closed_P3_tail、root_collected/fixed_first群、対応child law。Partialと同じlawとしない | CERTIFIED_BUDGET_FORMULA / CONDITIONAL_COMPARISON。同じC2/I1条件を使用する強い対照。F3 canonicalは未対応 |
| Full RA | 同じP_m、同条件。κou、λo、ceil。B_F²へ置換しない | [G] FullReturnGenerator/upper_normalizer＋budget。[F3]のFull preflight/angleは実装済み、IID/物理provider/全N測定は別 | CERTIFIED_BUDGET_FORMULA / CONDITIONAL_COMPARISON。未知B_F/Z不要、u/oとinput/policy/source identityが必要 |
| P5Closed（別枠、m=5のみ） | 同じ理想P5、B_F,5=B_5,5。ただしG7と同じbias/confidenceの固定N contractは今回確認した資料にない | [P5]にclosed_P5_full、有限bitgroup samplerあり。**[G9PLAN] planに既存認証Nはあるが別policy** | G7主表ではCERTIFIED_N_NOT_COMPARABLE。G9内の同contract比較式は存在するがG7のNと混ぜない。理想normalizer一致からdigital/N一致を主張しない |

全方式に同じ合法なinput/Green query、factor/closed式、precision、cache/streaming、IS/CTS layeringの機会を与える。今回は新たなproposal/CTSや最適化を作らず、固定4方式の既存budgetを比較する。最適な全既知方式に対する優位ではない。Fullに未知B_F/Z/実測varianceを使わせたり、対照の情報取得/精度だけを弱めたりしない。

## m=5 P5Closed：取得できた既存Nの所在

対象契約は固定[G9C]であり、SOURCEだけの推測で採択内容を補わない。

[P5]はroot/ordered-pair/長さ4群の有限bit実装を持つが、そのfile自身にG7互換budget関数はない。[G9PLAN] `plan`を静的に確認し、P5を含むG9内の既存Nがあることは保持した。

| G9既存設定 | G7主比較との差 |
|---|---|
| 登録p=(1/5,3/10,1/2)、x=5/7、m=5、η=ρ=10^-12、H160/K256 | 固定登録inputとexact Clifford+T providerを持つ別契約。一般m≥5の新policyへ拡張しない |
| β_9=2[8ρ+8(1+ρ)2ε_Rz]、s_9=1/200−β_9 | G7の係数3によるβと異なる。CTSも含むrepresentation norm<8の共通policy |
| α_axis,9=49/22000、22 axesの推定failureと11 rowsのresource failureを分離 | G7のα=1/320・16 axesとは異なる |
| ell_9^+=log_upper(2/α_axis,9)、[G8LOG]のoutward log | G7 log640_upperと同一の関数/定数ではない |
| M2_F,9=κ_9 o_9u_9、M2_5,9=κ_9 b_5,9²、W=λ_9 B.hi | **moment/rangeの形は既存G9内でも保持**。κ_9=(1+ρ)²/(1−η)^7、λ_9=(1+ρ)/(1−η)^7 |
| N_9=ceil(ell_9^+[2M2_9/s_9²+4W_9/(3s_9)]) | 実SOURCEにある式の転記。新しいP5 budgetの作成ではない |

G9に登録されたFull/P5双方の上端と同G9 policyを使う場合だけ、[比較条件](trial_count_inequality_conditions.md)のC1/I1という**同一budget間の直接差分/ceil操作**を別枠で適用できる。ここではその入力のNを再計算・再ランキングしない。G7 FullのNとG9 P5のNを並べて優劣を出さない。

G7と完全に同じerror/confidenceを持つP5 Nを新たに認証するには、P5のgroup/word/child精度・range/moment/biasと測定scopeを同policyへ接続する別作業が必要。その接続は今回行わない。単にG7 `budget(P5Closed(...))`がduck typingで呼べそうであることを、採択済み認証契約の存在としない。

m≥7の形式P5＋ordinary tailは[W2A]の形式benchmarkにとどまり、G7にarmがない。完成したdigital samplerや認証Nとして追加しない。[PU]反例の形式量を新実装へ格上げしない。**mandatory STOP。**

<!-- FIXED_REFERENCES -->

[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[W2A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/43d6533970e0eb4c430fe1cff2e798d681b707d4/docs/research/track_b_g6_w2a_general_order_normalizer_20261018/baseline_normalizer_definition_matrix.md
[PU]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/850ac89e1a6ea955def805e5462b8694ec0840b4/docs/research/track_b_g6_pu_analytic_proof_20261018/proof_or_counterexample_certificate.md
[F3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/docs/research/track_b_g6_f3_preflight_angle_20261018/README.md
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
[G9C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md
[G9PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_comparison.py
[G8LOG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g8_bounds.py
