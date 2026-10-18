# G7/G9の比較条件対応表

[入口](README.md) / [既存結果](existing_strong_baselines.md) / [不足証拠](evidence_gaps_and_minimal_plan.md)。根拠は[G7契約][G7C]、[G7 SOURCE][G]、[G9契約][G9C]、[v2 machine契約][G9MC]、[G9比較SOURCE][G9CMP]。数式・条件の静的対応であり、新しい予算・証明・数値評価ではない。

## 1. 数学的対象・方式

| 項目 | G7（主にP5_general_order） | G9 v2 | 比較上の扱い |
|---|---|---|---|
| 推定対象 | P_m(−iσxR)、R=Σp_iQ_i、Q_i²=Iのfinite operator mean | 同じ有限P5(−ixR)のfull first operator moment | P5 anchorの**理想対象**は一致。channel平均で代用しない |
| p/x/m | P5=(1/5,3/10,1/2)、5/7、5。別P3 controlも登録 | P5の同じ値だけ | 同じP5 inputでもN/費用規約は別。P3 controlは主横断比較外 |
| Q_i | abstract Hermitian involution、exact controlled-Q query assumption | 明示3-qubit Q0=Z0、Qi=Vi†ZiVi、V1=R_XX01(π/4)、V2=R_XX12(π/4)R_ZZ01(π/4) | G9はG7形式の一つの実装context。一般Qへnative価格を移植不可 |
| σ | primary費用は+1。−1は取得列actual adjointによるoff-domain意味論確認 | primaryは同+1。negative RZはactual adjoint | 任意σの新native費用表取得済みとはしない |
| Ordinary | even/odd次数pair＋IID raw labels/child | 同じG7 generatorを再利用 | source/入力/precision一致なら4方式の生成law対応を追える |
| Partial | P3の部分return吸収＋次数4/5 ordinary tail | 同じG7 generator | P5 targetを三次に縮小していない |
| Closed P3 | 全P3 closed form＋次数4/5 ordinary tail | 同じG7 generator | 同じ有限P5を実現する対照 |
| General Full | Green/local return、raw-reduced→coin→child | 同じG7 Full generator | reference全表をproductionへ渡さない |
| Closed P5 | G7の登録4方式に含まれない | [P5Closed][P5]の独立degree-five grouping、正規化group law、sample APIあり | 強い同family fast path。一般Fullの独立競合法という新規性評価にしない |
| matched CTS | G7には未登録 | [CTS SOURCE][G9N]のfinite-P5 Pauli収集。real correctionはidentityを含め別event、odd部を一angleにpair | G9が確認した既知構成の特殊化。新測定/新理論として再認定しない |

Closed P5の31 parents/63 events/10 groupsに対するexact係数一致は[G9準備証拠][G9MATH]と[G9結果][G9R]に保存済み。理想full-return ensembleはGeneral Fullと同じだが、group/rootの丸め・proposalは別。理想一致からdigital係数・N・全費用の一致は導かない。

## 2. Law・重み・誤差・confidence

| 項目 | G7 | G9 v2 | 結論 |
|---|---|---|---|
| zero-fill | Fullはraw nonreduced/coin rejectを量子前zero。4方式の全trial分母 | 同じFull zero。Closed P5/CTSはcanonical event lawで受理質量1 | accepted-only正規化は禁止。failureをzeroにしない |
| digital weight | c̃/π、exact rational。canonicalも実proposalで補正 | 同じG7 4方式＋Closed P5/CTSの固有group/Pauli law | 理想Zとdigital受理率を混同しない |
| m2/range | Full κ B_O.hi U.hi、canonical κ B_a.hi²、range λB_envelope.hi | 4方式は同じ形。Closed P5はgroup B、CTSはcollected係数のB | 同じm=5ならκ/λの次数因子7は共通でも、B/残余幅/Nは別 |
| root/dyadic | outward isqrt K256、H160、η=ρ=10^-12 | 同K/H/η/ρ。CTSにはQ(√2)係数intervalとnormの丸めもある | precision数値の一致だけで全係数処理同一とはしない |
| coefficient/norm envelope | norm<3、mean係数error≤3ρ | 全representation共通norm<8、mean係数error≤8ρ | G7の3ρをCTSへ無根拠に適用不可 |
| RZ・provider | ε_RZ=10^-6、event rotation≤2ε_RZ。CQ誤差0は抽象仮定 | 同ε_RZ、composition≤2ε_RZ。CQ誤差0はこのexact Clifford+T model | hardware誤差の実証ではない |
| axis bias | β7=2[3ρ+3(1+ρ)2ε_RZ] | β9=2[8ρ+8(1+ρ)2ε_RZ] | **異なる**。同ε_axisでも残余s=ε_axis−βが違う |
| ε_axis | Re/Im各1/200 | 同1/200 | 精度目標は同じ。Taylor tailやQPE/RPE全体を保証しない |
| estimator confidence | 8 rows×2 axes、α_axis=1/320、familywise≤1/20 | 11 rows×2 axes、α_axis=49/22000、estimator合計49/1000 | **異なる**。旧16軸を拡張せず22軸の枠も新集合へ自動転用しない |
| resource failure | G7はNのdeterministic quantum-call hard cap、別accepted tail取得なし | β_resource=1/11000/row、11 rows合計1/1000、total1/20 | 同じtotal0.05でも配分/保証対象が違う |
| logとceil | `log640_upper`固定96項、exact rational ceil | `log_upper(2/α)`、exact ceil、accepted tail t=10 | log SOURCEも別。新たに同一値と評価していない |
| N | 各axisの認証十分**全trial数** | 同じ意味だが上記policyで別N | G7 N×G9 native単価/総量をそのまま混合しない |
| accepted/hard cap | 期待N z、hard N/axis | 期待N z、別resource tail capとhard 2N/two axes | 期待費用・tail upper・hard upperを別座標にする |

κ=(1+ρ)²/(1−η)^7、λ=(1+ρ)/(1−η)^7はP5での記法。両者ともBernstein型の既存十分予算を使い、exact signal/参照moment/未知B_RAでNを下げない。G9 CTSのPauli収集Bはその方式の合法なI1情報であり、Full local budgetへ全列挙normalizerを注入したものではない。[G9CMP][G9CMP] [G7C][G7C]

## 3. Provider・compiler・費用座標

| 項目 | G7 | G9 v2 | 横断比較の条件 |
|---|---|---|---|
| 回路実装 | [generic helper IR][CP]。word CQ＋child compute/rotate/actual-adjoint | primaryは各方式共通direct V/P controlled lowering。return5方式はhelper診断も別保存 | G7 helperとG9 directを同じ実装としない。Fullにもdirectを許す |
| phase/adjoint | 元degreeのouter Z、逆時間word、σ、W scalar込みnegative列のactual adjoint | exact V…V†、controlled相対phase、outer Z、negative列actual adjoint | scalarをphase最小化で落とさない。G9 CQはexact Hermitian involution構成 |
| compiler | CQ内部は未指定、RZ列は取得済み | CZ→H CX H＋adjacent inverse cancellation、全方式同条件 | 全回路最適・最良compilerではない。新optimizationへ既存順位を外挿しない |
| T | RZ切片＋label別CQ T symbol＋prep T | exact modelのnative-event T切片＋K T_prep/readout | RZ-onlyをnative-event全Tと同一視しない |
| CX/1Q | helper skeleton CX2/H4等＋未知CQ価格、readoutは別 | native event CX/1Q、outer prep/readout 1Qは別欄 | skeleton-onlyと総nativeを分離。1Q欄のscopeも照合 |
| depth | 確定native depth表なし | `cost`はT/CX/1Qを数えるがdepthを保存しない | **未取得**。gate数からdepthを作らない |
| ancilla | outer+helper=2、provider追加は未知 | directはsystem3+outer1、helperはsystem3+outer/helper2 | model内workspace差は既知。hardware peakやstate workspaceは別 |
| state/readout | unspecified exact state、共通prep symbol、Re/Im readout2/3の1Q | 共通非負T_prep/readout symbol、native eventとouter1Qを分離 | 実state・backend・readout価格は未取得。未知を0にしない |
| quantum実行 | 実measurement0 | 実measurement/trajectory0 | native資源会計とhardwareでの実費測定を区別 |

## 4. Classical・cache・情報access

G7にはsetup、128-trial固定SHA interface/CPU診断、cold key取得、小support reference時間がある。G9は全event reference/native会計を含むguard区間CPU等を保存するが、6方式の全N production sampling・F3監査込みCPU profileではない。両者の約秒数を全trial古典順位へ変換しない。[G7R][G7R] [G9R][G9R]

G9は19 keysのうち18をG7からangle/ε/tool/strict-phase/hash一致で再利用し、CTS用1 keyだけ新取得した。**旧wrapper費用は再利用していない**。同じ旧sequenceを持つことはprovider/native wrapperが同じという意味でも、独立再現でもない。cold/warm・取得済み資産・arm単独取得・共有key pool・lookup/出力を同じ規則で扱う。[G9MC][G9MC] [G9AUD][G9AUD]

G9はQ(√2) Pauli情報を安く取得できる明示**I1** contextで、全方式に同じ情報を公開する。matched CTSの収集費用・dictionaryが一般Qでも無料とはしない。G7の抽象query-only/I0状況へ、そのCTSを追加oracleなしに持ち込めない。逆に、明示I1を採る場合にCTSだけPauli情報やdirect/基底共有を禁じて弱くしない。

固定[F3][F3]はFull非identityのみで、Closed P5/CTSの同等入口や全N accumulatorではない。[F2][F2]の4方式bit会計もClosed P5/CTSの完成監査CPU測定ではない。Codex Aの新入口実装の有無・完成度は本作業では調べていない。比較準備が4方式で完成しても、強い対照まで自動的に費用認証が広がるわけではない。

**一致するのは理想P5 anchorと一部precision/既存primitive。policy・実装・情報accessまで合わせて初めて新しい横断費用比較が可能。G9内の既存matched比較の引用には新しい横断計算は不要。mandatory STOP。**

<!-- FIXED_REFERENCES -->

[CP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_provider.py
[F2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/full_bit_complexity_accounting.md
[F3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/docs/research/track_b_g6_f3_preflight_angle_20261018/implementation_contract.md
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[G7C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[G7R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md
[G9AUD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/saved_output_audit_v2.json
[G9C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md
[G9CMP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_comparison.py
[G9MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/artifacts/track_b_g9_p5_native_preparation/2026-10-10/mathematical_preparation_v1.json
[G9MC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/contract_v2.json
[G9N]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_native.py
[G9R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
