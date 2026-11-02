# A2 暫定比較契約・費用台帳の設計

2026-10-18 JST。**`DESIGN_ONLY_NOT_EXECUTION_CONTRACT`。** [接続仕様](certificate_connection_spec.md)を受ける比較の型・判定・費用scopeを定める。科学条件は未採択、d/α/分子/geometry/rank/算術解像度/科学call範囲は未設定。正式科学実行契約・authorization・grantは作成していない。

## 1. 共通条件と実行guardの設計

比較の入力`ComparisonPolicy`は、共通bundle_id、固定DF target、固定N、固定PSD分割/順序、exact ground sector支持の条件付きaccess、energy-windowのd>0、0<α_spec<1、採用する保証version、算術enclosure停止幅、情報取得policy、費用scopeを持つ。d/α等は正確なRat。未採択fieldは`UNSET_REQUIRES_ADOPTION`という状態で保存し、0や慣例値を仮入力しない。

全routeは同じ情報pool、証拠取得policy、scalar/energy原点、d、α_specで比較する。相対側が得たPSD/sector/構造情報を対照側へ隠さない。追加norm情報を得た場合も両者へ同じ価格で公開する。結果を見て対照のboundを粗い候補へ戻したり、relativeに有利なPF順序だけを残したりしない。

将来のguardは、(1)ユーザーによる科学条件/契約採択、(2)SOURCE・入力・契約の固定、(3)明示的な別途科学実行承認、(4)全certificate/contextの受理、を別々に検査する仕様とする。今回の文書やevidence indexを(1)–(3)の代用にしない。one-shot/retry、call範囲、output/log制限は後日の正式契約で決める。既存grant/ledger/markerを再利用しない。

## 2. 既存保証への接続

[EW2監査 `db181958…` §§1,4–6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/db1819587df36ee8833b37227ca0083faa0408ec/docs/research/qc_relative_form_ew2_step_audit_20261017/mathematical_audit.md)を式・branchの正本とする。

\[
\eta_0(h)=\frac{32(h\Lambda)^2}{1-16(h\Lambda)^2},\qquad
\eta_1(h)=h^2K_{\rm SOS}^{\rm upper}+\frac{512(h\Lambda)^4}{1-16(h\Lambda)^2},
\quad\eta=\min(\eta_0,\eta_1).
\]

証拠のK_SOS_upperを使うため、その粗さを出力に残す。ηは近似leading式へ置換しない。RF/EW2は固定正stage S2、同一N不変sector、原点からの解析的log、h>0、hΛ<1/4、η<1を要する。exact ground支持の下で

\[
B_R(h)=\min\left(1,\frac{\eta(h)^2W(W+d)}{(1-\eta(h))d^2}\right).
\]

強化対照は同じcapからβ=Λ/2を使い、

\[
b_C(h)=\frac{h^2\overline C_2}{1-h\beta},\qquad
B_C(h)=\min(1,b_C(h)^2/d^2),\qquad h\beta<1.
\]

こちらはcenterしたunitaryのlogと既知scalarを戻したK_cを用いる。RF領域内の比較点では元監査の条件でK_c=K。領域外もcentered branchとそのunwrapを明示して扱い、RFの1/4半径へ機械的に制限しない。βは最適bandwidthだとはしない。元Hやcontrolled絶対位相を変更したという意味ではない。

`ApplicableRoute`は、route ID、固定式/監査commit、必要certificate IDs、domain predicate、branch ID、state/access仮定、units、算術証拠、適用/不適用理由を持つ。K_SOS等がない場合は個別RF0の部分記録は可能だが、完成したRF0/RF1/EW2比較としてsealしない。

## 3. 最大十分stepと開区間の扱い

\[
\mathcal A_R=\{h>0:h\Lambda<1/4,\ \eta(h)<1,\ B_R(h)\le\alpha_{\rm spec}\},
\quad h_R^*=\sup\mathcal A_R,
\]
\[
\mathcal A_C=\{h>0:h\beta<1,\ B_C(h)\le\alpha_{\rm spec}\},
\quad h_C^*=\sup\mathcal A_C.
\]

いずれも**選択した証拠が許す最大十分step**であり、真のPF誤差の最大stepではない。非退化なΛ,W,d,Cbar2>0等の条件では、前回監査のh0/h1、h_R=max(h0,h1)、強化h_Cの式と同じendpointである。既存閉式をroot生成の補助に使えるが、float sqrtや丸めた閉式だけでendpointを認定しない。

### 検証可能な数値出口の仕様

`StepCertificate`は`route_id, context_ids, definition_of_set, endpoint_bracket, endpoint_attained, accepted_probe, rejected_probe_or_domain_edge, exact_predicate_witness, formula_ref`を持つ。

- 正確なRat hに対して、上の分母/domain/ηと確率条件を有理数演算で検証できる。generatorは既存単調性・root式に基づくrational bracketを提案し、verifierは受理側と棄却側を再検査する。
- 非退化時はh_loが受理、h_hiが同じ単調判定で棄却、または厳密なdomain edgeであることを示し、endpointを区間で囲む。domain edgeではゼロ分母を評価せず`DOMAIN_EDGE_OPEN`を記録する。
- `h_lo`は実際に保証が受理するcandidate、endpoint bracketの上端は実行点と同一視しない。strict domainを満たさないhで、丸めた`≤`判定へ切り替えない。
- enclosureの目標幅・bit/精度policyは後日採択する。未採択なら`NOT_EVALUATED_PENDING_POLICY`。指定幅で分離できない場合は`UNRESOLVED_ARITHMETIC`を保存し、結果を見た自動precision増加や閾値変更を行わない。

比較で`h_R,lo > h_C,hi`が証明できたら`RELATIVE_LARGER_SUFFICIENT_STEP_AGAINST_DECLARED_COUNTER`、逆が証明できたら`COUNTER_LARGER_SUFFICIENT_STEP`。区間が重なるだけなら`ORDER_UNRESOLVED`。等値は別の正確なendpoint証拠がある場合のみ`EQUAL_SUFFICIENT_STEP`とする。同じmax endpointを持つのか、一つの有限probeが不受理なのかを区別し、上界が不合格だったことから実際のPFが不正確だとは結論しない。

### 退化case

| 条件 | 保存する結果と制限 |
|---|---|
| W=0 | E0=L既知。EW2の0除算・資源改善例へ進めず`TRIVIAL_KNOWN_ENERGY` |
| Λ=0 | 全F_j=0、H=LI。`1/Λ`を評価しない。同じく既知energy |
| Cbar2=0 | 非負norm和が0の証拠から既知S2保証は全実hでexact。b_C式のdomain内supと、別route`COMMUTATOR_ZERO_EXACT_S2`の全hを区別。後者へのrelative step優位を作らない。実QPE alias/readout制限は残る |
| M=1または可換証拠 | exact S2の既知対照を許す。raw RFの保守的な正剰余を改善実例と誤認しない |
| K_SOS_upper=0 | RF1の四次剰余を保持。二次leading/Xiだけでstepを選ばない |
| 真のe=0、W>0 | eをoracleにしない。証拠からe=0を確認できない限りWの一般判定を使う。W=0等の既知caseとは区別 |
| d≤0、α_specが(0,1)外、負cap、L/U不整合 | `INVALID_TASK_OR_CERTIFICATE`。数式へ投入しない |

今回はh、Xi、成功率、費用比を計算していない。

## 4. 比較recordと主張の上限

`ComparisonRecord`は、共通context、全route条件/StepCertificates、情報poolと各生成費、両側のenclosure、order判定、失敗理由、partial evidence、仮定、未評価scopeを保存する。relativeが不利・未解像でも出力を残し、良いrecordだけをevidence indexに登録しない。

以下を独立fieldとして必須にする。

| field | 初期比較で許される意味 |
|---|---|
| `certificate_verdict` | 型/hashの受理と数理証拠の再検証を区別 |
| `step_comparison` | 宣言したRF/EW2とcentered Cbar2対照の十分証拠上の比較 |
| `conditional_on_exact_ground_access` | true。state preparationを完成・無料と扱わない |
| `original_molecular_guarantee` | `NOT_ESTABLISHED`。固定DFの結果を元分子へ移さない |
| `other_baseline_envelope` | `NOT_EVALUATED`。適用可能な低energy保証、QPE回路安定性、さらにtightなlog/構造対照は残る |
| `end_to_end_execution_step` | `NOT_ESTABLISHED_WITHOUT_COMMON_IMPLEMENTATION_READOUT_GATES` |
| `total_resource_improvement` | `NOT_EVALUATED`。state/native/readout/取得費の不足を0扱いしない |
| `scientific_go_no_go` / `A2_adoption` | 自動出力しない。ユーザーの別判断 |

構造Cbar2の未完・未使用情報や有利な未知比較がある場合は、その限界をrecordへ付ける。今回の設計はbest-known envelope全体への優位を認定する手順ではない。

## 5. state accessとQPE接続

| access型 | 仮定・必要情報 | 許される報告 |
|---|---|---|
| `EXACT_GROUND_COPIES_CONDITIONAL` | H_DFのN-sector ground支持の正規化state、縮退ground混合も固定EW2条件内で可。copy取得方法/費用は未知として別記 | ground access条件付きのenergy-window/step比較。準備込み総費用ではない |
| `PREPARATION_UNITARY_AVAILABLE` | 具体的な準備unitary/逆、SOURCE・入力・精度・成功率・費用。controlled/inverseを必要とするprotocolなら追加仕様 | これらを検証した別taskに限る。初期比較から自動移行しない |
| `APPROXIMATE_STATE` | overlap/支持/品質と失敗率の正当な証拠、新しいtaskへの接続が必要 | 保存solver残差や高visibilityだけでexact ground条件を満たしたとしない。今回新しい近似state保証は作らない |

Uを認証するtrial determinantは変分上界用であって、EW2へ入力するground stateではない。この二つのstate refsを別fieldにする。

各方式で採用するhを固定した後、QPE powersは同じkernel `S2(h)^r` のinteger powersである。roundごとにhや分割を変更して単一generatorの保証を流用しない。元energy原点Lを復元する。unshifted S2の位相規約を`θ=−hy mod 2π`とするなら復元は`L−θ_unwrapped/h`。centered counterpartは`θ_c=−h(y_c−β) mod 2π`、復元は`L+β−θ_c,unwrapped/h`と記録する。controlled phaseのscalarを無言でglobal phase同値へ落とさない。

log branchの数学的存在、測定位相のalias/unwrap、QPE読出し精度・成功率を別gateにする。readout protocol、power列、対象窓、精度・失敗率、basis/bit orderingが未指定なら実行準備完了としない。

共通taskにはenergyのspec/readout/model/実装予算と、α_spec/readout/準備等の失敗率を定める欄を設ける。数値配分は今回未採択。exact certificateのinvalidは技術拒否であり、独断で確率α_certを割り当てて吸収しない。固定DF内のtargetを定義したことと、元分子model誤差がないことは別。合成/回転残差をhやquery数へ戻す換算とQPE回路安定性は、使う既知保証・scopeを別途固定する。未検証の誤差をenergyへ機械的に足して実QPEを保証しない。

## 6. 費用台帳

`CostRecord`の型は`id, context_ids, stage, method_scope, status, quantity, unit, measurement_or_symbolic_basis, SOURCE_ref, input_ref, reuse_group, amortization_count, notes`。`status`は`MEASURED / SYMBOLIC / UNKNOWN / NOT_EVALUATED / NOT_APPLICABLE`。未測定quantityは理由付きnullであり0ではない。quantityには正確なcountまたは単位付きmeasurement/enclosureを使う。

| 区分 | 個別保存する費用 | 共有・償却の条件 |
|---|---|---|
| 入力 | 読込み、decode/hash、raw→model規約確認 | 同じraw/model意味論が一致するときのみ再利用 |
| 共通certificate | PSD/caps/L/U/W/Λの生成と**別検証**、scalar/sector検査、保存 | generator費・checker費・witness bytes・bit長・再検証費を分離 |
| route別certificate | K_SOSとCbar2各経路の生成/検証、root/step enclosure | 情報は両側へ公開。対照を強くする構造取得も記録し、一方だけ無料にしない |
| 古典回路前処理 | Gaussian proposal、残差検証、phase台帳、回路生成 | model/分割/精度/backendが変われば再利用を再判定 |
| 量子access | ground copies数、準備/逆/成功処理の費用またはUNKNOWN | 準備はshot/protocolの実際の回数で数え、全base-queryごとへ無条件に重複課金しない |
| controlled S2 | onebody、signed平方、Gaussian・対角・scalarのlogical作用とnative価格 | PSD外側shiftの復元を二重計上しない。boundary fusionはSOURCE/同値性/精度を固定した場合のみ |
| QPE/readout | power別全base適用、readout/reflection/測定、confidence反復 | query数、logical gate数、native T/CX/depthを別座標で保存 |
| workspace | mode/control/QPE register、state prep/合成のworkspace | 古典sector圧縮をqubit削減として数えない |

共通の前処理を両候補の独立総費用へ同じ基準で含め、共有実行なら実際に一度生成したことも別に記録する。比較差分では共通費の相殺を説明できるが、初回取得が無料とは表現しない。償却回数・再利用task数はユーザー採択まで未設定。exact証拠でもchecker更新・context変更時の再検証は必要である。

CPU/wall、peak RAM、整数bit演算、witness/output bytes、quantum queries、T/CX/RZ/depth/qubitsを無定義に合算しない。同じ資源座標gでは、既存監査どおり前処理に反復ごとのstate準備・base適用回路・readoutの費用を加える形を使えるが、換算価格なしの「総cost」単一値は作らない。並列wall-timeを逐次時間の和と誤記しない。

証拠生成・exact grouped展開はbit長/語数で高価になり得る。理論的取得可能性は安価な取得の証明ではない。今回、実測・shot/query forecast・native compileを行っていない。将来のbreak-evenは、共通task・実装/readoutが成立し、比較対象の費用座標と再利用条件が揃い、未知費用の扱いが採択された場合のみ評価する。

## 7. resource・保存・STOP

共通運用指示2026-10-17 §8.4を適用する。将来の新契約で人工的なCPU/RAM/worker/wall上限を既定にしない。dominant workload、整数bit長・語数増加、cache、独立group/証拠の並列化と実際の空き資源から構成を選ぶ。再現性・科学call範囲・保存・精度を維持し、共有環境/他jobを変更しない。具体的なcall/output/log/instruction制限、失敗時保存、retry policyは後日採択する。今は数値実行を準備・発行していない。

失敗時も入力/全context、partial certificate、verifier理由、SOURCE/hash、stdout/stderr/cost、達した段階を新しい記録へ保存する設計とする。旧証拠・既存grant/markerを上書きしない。準備完了や有利なstep判定をauthorizationへ変換しない。

**今回の暫定比較仕様の設計は完了。科学契約未採択・実装未承認・科学実行未承認のままmandatory STOP。**
