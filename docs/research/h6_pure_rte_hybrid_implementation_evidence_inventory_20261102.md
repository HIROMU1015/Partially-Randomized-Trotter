# H6純RTE対Hybrid：実装精度・価格tailの既存証拠調査

2026-11-02 JST。**P3は固定H4 matched分割の条件付きcase studyとして一区切りとし、追加のnative/compiled保証は保留する。**
次の調査対象を元のPauli純RTE対Hybridへ戻すというユーザー指示に沿い、下記2比較の保存証拠を対応付けた。
本調査はSOURCE・保存結果・archive/QPY bytesの読取と技術的整理だけである。
新しい誤差値・確率・bias/headroom・感度・N/G/SE・順位の計算、sampling/RNG replay、状態作用、回路生成/compile、人工数値testsは0。
主要claimの正式採択や独立科学レビューではない。全旧SOURCE/raw/結果/STOPを維持する。

**主な発見：H6の保存会計には既に53-bit実効lawとbinary64 event角度が入っている。**
P3と同じ有限bit評価を一から行うことより、保存targetの数学的作用からliteral native/compiled Fullへの未取得の接続を確認する方が次の課題に合う。
一方、現在の仮想ρ=0・有限価格標本という主張を報告するために、追加科学計算は必須ではない。

## 1. P3の区切りと元研究との関係

| P3証拠 | 固定結果commit | 確認できたこと / 保留すること |
|---|---|---|
| Stage 1 | `5708e37080738c4ebf74590d799a3741246483b2` | H4案Aの共通Hamiltonian/T/V分割の数値整合。一般的な表現優位の証明ではない |
| Stage 2 | `8bcf5851f99f980dc08c5885ace54874b554c815` | 補正済みnominal平均の整合、B_P≈39.1259/B_DF≈34.5280。finite Taylor biasは未解像 |
| Stage 3 run04 | `e44e87a306d05c8549f1a044b7fd056c2e1055a3` | 共通Gaussian保持、独立32 trajectory/arm、128 compile。保存標本ではPauli単回価格が低い。母平均tail/compiled作用保証はない |
| Stage 4 | `a1d3f32aba6416fd1e200cd66a97d1a825b5dd73` | 5精度で条件付きG_P/G_DF≈.45273。finite-bit gap=0/ρ=0/B engineering候補の仮定を明示 |
| 有限bit | `426ccda5df044d2e94d4a18d98cde75a96ab1e6b` | base基準law/角度上側候補P9.35e-14/DF1.80e-13。保存nominalへのbridge込み候補P1.43e-9/DF1.78e-9。全signalはengineering評価 |

P3の全条件はlinear H4/1Å/STO-3G/full256/保存DF tol-only1e-8 rank10、案A/T.8/S2/q8/r4/K4/M_B・M_C/P=0。
M_B/M_Cは共通の非自明なT_effを保持する残差RTEであり、**L_D=0のPauli純RTEではない**。
有限bitの20 headroomが正という結果もnative bridge=0/compiled ρ=0を仮定したpreviewで、実装保証ではない。
[有限bit報告][P3]と[Stage 4報告][P3S4]を固定case studyの入口とし、任意の保証追加を現作業の前提にしない。

元のDF B2・決定論的PF・RPE全roundの目的は維持する。
Pauli-onlyの固定時間signal/状態準備P=0の比較、P3の共通分割、DF B2、RPE全roundの総資源は別の研究範囲である。
今回それらの最終費用を統合・再会計しない。

## 2. 固定H6対象と来歴

linear H6、隣接1.00Å、STO-3G、中性singlet、保存Pauli-JW918非identity項＋constant、system12/full4096。
T=.8/S2、q8/K2、central-rotation制御、system12＋ancilla1（wire12）、cosine/+Im sineのfull Hadamard wrapper、P=0。
Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、backend/coupling/layout/routing追加指定なし。DF rankは適用外。

| 比較 | 純RTE | Hybrid | 意味 |
|---|---|---|---|
| 主要family比較 | R64=`H6_p000_q8_r64_K2`、L_D=0、512 event位置 | H32=`PR_p0078_q8_r32_K2`、L_D=78、256 event位置 | 既存有力候補の費用比較。rも異なる |
| q/r/K一致対照 | 同じR64 | H64=`PR_p0078_q8_r64_K2`、L_D=78、512 event位置 | prefix/tail/lawも変わるため一因子比較ではない |

constant phaseは純RTEでも保持する。非自明な決定論的保持がないことを、scalarまでないことと混同しない。
同じindex0000は両方式のpaired科学標本を意味しない。

| 役割 | 固定結果 / 科学SOURCE | 保存入口 |
|---|---|---|
| H32/H64 | `00958ae1945c780718b2deef07d5dab1096a5cd8` / `887b69e449cb3c378286f09d0d7626599850fb67` | [H6拡張REPORT][HREPORT]、SOURCE.json、candidate_results.json、price_statistics.json、data_archives.json |
| H32/H64会計 | 同じ結果 / 解析`af891fd45805454c6592bbee5441266f14235ffd` | CANDIDATE_RESULTS.csv、COMPARISON_RESULTS.csv、comparison_details.json |
| R64 | `10cb3e5cca7834ebf75301bd91bd093ca51d4a6b` / `e755ecf581600192fd2745d7066e410cebd933d3` | [event拡張][RREPORT]、SOURCE_e755…json、candidate_results.json、data_archives.json |
| P1 | `bfb4ca75437a9b3605a8f57b5f82100615dd034f` / `a0a3aff915f98456733d6bf09ed641a1be423628` | [保存bias分離][P1]、source_and_inputs.json、signals_and_signed_differences.json |
| P2 | `c9073274daee69c9b69d8db3dc332bf6132c2259` / `11e50e262f95630b14fa4070ad30894ba3f64b74` | [固定programの価格・意味論][P2]、saved_full.json、semantic_checks.json、source_and_inputs.json |

H6 model SHA256は`6c66a2eeb4d99cf5bbd9137bbcb067275536be3201d5cc047225266dc5f58681`、
reference SHA256は`9956466a5cf3286af9960b792b455bb8bacc2532a34ac7a28b470475f92fcb7a`。
共通Hamiltonian hashは`37b2747eaafacef8b4b470e5f4d6e195ac0602dc8e59953e3b76152d4576c847`、
raw state hashは`ff1ba6309c6b17a263195eea718f1d5243cbe6b3422c6a9d1fbb0fe8b212ff0c`。
これらはP3 H4の入力とは異なる。今回raw stateの新しいnorm計算やreference生成は行わない。

現在checkoutにH6 Hybridの元ディレクトリ全体はないが、固定commitのGitオブジェクトとGitHubから取得できる。
欠落と誤認して再計算せず、既存P2の`selected_members`と既存`recover_member`で選択memberだけを読んだ。
3候補のlaw/spec・target・confirmation/0000 result・両軸input/compiled QPY、計21 memberをarchive/part/member SHAと照合した。
12 QPYのgzip/raw SHA・bytes・QPY magicは元resultと一致。**今回はQiskitによるQPY deserializeや状態作用も0**である。
元reportのpublic取得と固定refを確認した。機械可読の参照・値・hashは[INVENTORY.json](../../artifacts/h6_pure_rte_hybrid_evidence_audit/2026-11-02/INVENTORY.json)。

## 3. 既存証拠だけで支持できる費用の主張

[既存論文用統合][SYNTHESIS]では、H2/H4/H6の15精度条件中14条件で純RTEの条件付き点Gが低い。
H4 ε_sig=.05のHybrid低Gを対照として維持し、H6だけから一般化しない。5精度は同じ価格標本を共有し、15独立実験ではない。
H6の最新family比較はH32対R64、確認標本n64対n48で、保存H/Rは約1.129/1.152/1.183/1.489/2.092。
以下は**既存confirmation行のexact文字列を引用**し、G・倍率・SE・順位を再計算していない。

| 候補 | 確認n | 保存B exact | 保存C_cos=C_sin exact | ε=.05 G exact | ε=.0005 G exact |
|---|---:|---|---|---|---|
| R64 | 48 | 1661454695514047/1125899906842624 | 149519/48 | 570714023/6 | 45657735316457/48 |
| H32 | 64 | 5747355790514877/4503599627370496 | 299249/64 | 3435079271/32 | 63691899506611/32 |
| H64 | 64 | 1271969024947877/1125899906842624 | 432931/64 | 973661819/8 | 72211615818205/32 |

`comparison_details.json/primary_comparisons/versus_RTE`はH32/H64の保存比較を持ち、最厳精度でいずれもREFERENCE_LOWER_EMPIRICAL_G。
H64対R64も新しく選抜・再会計した比較ではない。
H32対R64では、粗精度で純RTEはB/Nが大きいがCが低く、厳精度では小biasによるheadroomとCの双方が費用差に対応するという既存の会計説明がある。
保持を除いたことだけの寄与率や、全パラメータ最適性を示すものではない。[既存原因調査][MECHANISM]。

主張できるのは、同じ保存H6/model/state、固定候補、固定compiler、**仮想ρ=0・保存価格標本**に条件付けた低Gである。
観測SEや記述的2 combined-SEを、母平均順位の厳密保証に使わない。

## 4. 保存signal/lawは既に何を評価しているか

3 lawは`pauli_pr_search_law_v1`、NumPy1.26.4、uniform_bits=53、orders=[0,2]。
R64の成分数は918、H32/H64は840。component/orderのexact grid masses、binary64 angles、scalar_step_binary64、B_savedが保存されている。
SOURCEの`h4_pauli_pr_search.law_descriptor`と`pauli_vector_tight_enclosure.evaluate/event`は、
これらの実効質量と角度からmoment平均を伝播する。

`target_semantics=C_EFFECTIVE_GRID_LAW_EXACT_RETAINED_PRODUCTS_NATIVE_STEP_SCALAR`。
targetはscalar128bit・vector中心53bitのtight L2評価、保存raw ν、B一回、非再正規化、`actual_rho=null`。
**既存H6 N/Gはこの実効law targetのbias/数値allowanceに接続しており、P3 Stage 4のfinite-bit gap=0仮定と同じ出発点ではない。**
P1のΔ_gridが未解像であることは、実効law平均そのものが未取得という意味ではない。
保存targetは平均、固定trajectoryのinput/compiled QPYは個別programである。両者を直接引いてcompiled誤差としない。

保持rotationはtargetでexact binary64係数×exact δ/2として扱う一方、literal D2回路にはbinary64で形成した倍角operandが入る。
さらに負product/order phaseのπ加算・modulo集約、制御分解や相対phase、wrapperの作用までtargetの包含だけから保証されるわけではない。
これがnative接続に残る境界である。full-wrapper全体global phaseと、controlled bodyのancilla-relative phaseも区別する。

## 5. P1/P2の到達点と未取得の実装精度

| 境界 | 既存証拠 | 残るもの |
|---|---|---|
| 保存model/state/reference→law target | 同じHamiltonian/raw state/ν、実効CDF/角度・B一回、tight L2 bias/allowanceを保存 | literal回路の全作用差ではない |
| 理想分割のbias原因 | P1：Hybrid共通Δ_outer≈7.0164e-5＋1.13627e-4i。I/Z保持のB/Cは整合。R64 finite項は解像 | 全grid差とH32/H64 finite項は未解像。P1の1e-10はengineering分解能で、targetの包含や実ρと別 |
| 固定Full inputの意味論 | P2：3候補×2軸で保存token・順序・wire・angle・phase・wrapper一致、最大parameter差0 | input作用と理想selected-targetの数値差は未取得 |
| 固定Full compiledの実在と価格 | input/compiled両QPYが保存され、P2でmetricsとcompiler profileを照合 | compiledとinputの作用誤差は未取得 |
| compiled精度→N/G | 保存ρ_guard・仮想ρ=0会計がある | actual_rho/actual_N/actual_Gはnull。実ρの所属確認や新感度会計は未実行 |
| 個別誤差→全law | 同じsampling law/phaseに接続するSOURCEは読める | 一様boundまたは妥当なΣ p_i error_iがない。数本のPASSから全lawへ外挿しない |

P2の固定program値と標本平均は異なる。confirmation index0000のFull価格は次の保存値である。

| 候補 | event位置数 | 固定Full RZ（両軸） | CX | total depth | P2報告にあるorder内容 |
|---|---:|---:|---:|---:|---|
| H32 | 256 | 4732 | 7762 | 8064 | order2を1個含む |
| H64 | 512 | 6735 | 10934 | 11659 | 全order0 |
| R64 | 512 | 3081 | 4344 | 5084 | 全order0 |

P2はHybrid保持単独2500 RZ、結合の正負の非加法性も測定済みである。これを一から再compileする必要はない。
H64/R64ではevent-only価格も違うので、Full差を保持78項だけに帰属しない。
今回のarchive/QPY byte照合は、P2の過去の意味論検査を新しく実施した検査や、新しい作用誤差値として数えない。

## 6. 保存許容幅と価格tail：どの主張を制限するか

下表はε_sig=.0005の`axes.*.rho_guard`保存decimalをそのまま引用する。
既存式はh=ε_a−|bias|−u−ρ、(u+ρ)/h≤.01。ρ_guardはこの条件の許容端であり、実測誤差でも順位逆転の閾値でもない。

| 候補 | cosine rho_guard | sine rho_guard | actual_rho |
|---|---|---|---|
| R64 | 3.5004127876769173e-06 | 3.5004418213125615e-06 | null |
| H32 | 2.8058046093851577e-06 | 2.375479343411928e-06 | null |
| H64 | 2.8058066327023485e-06 | 2.3754802450308678e-06 | null |

個別trajectoryのB補正前の差や平均targetのuを、そのままρ_guardと比較して所属を認定しない。
仮に実ρを取得してもρ=0の保存Nがそのまま有効とは限らず、必要なら別許可の会計接続を行う。
今回、新しいrho許容幅、順位感度、損益分岐、G_guard_rangeからの再比較は作っていない。

| 確認cohort | n | 保存order0 / order2 event count | 母平均価格上界 |
|---|---:|---|---|
| H32 | 64 | 16374 / 10 | null |
| H64 | 64 | 32765 / 3 | null |
| R64 | 48 | 24563 / 13 | 未保証 |

H32/H64は[元REPORT][HREPORT]、R64は元verification.jsonの`observed_orders`から引用。
**K2のorder4はsupport外**。P3 K4のorder4未観測問題を持ち込まない。
ただしorder2・成分組合せ・support・回路最適化応答の希少な費用寄与は残り、order2を観測しただけでtailが閉じるわけではない。
同一trajectoryのcos/sin共分散は元会計の規約を維持し、方式間をpairedにしない。標本SEと母平均上界も別である。

| 強めたい主張 | 現在の制約 / 優先する確認 |
|---|---|
| 固定候補・ρ=0・保存標本の条件付き低G | 既存結果で報告可能。全native/compiled/tail検証を必須にしない |
| 保存programで実装が期待通り作用する | selected-target→input→compiledの接続が未取得。小さい固定trajectory診断が直接対応する |
| 実装精度保証を含む比較 | native/compiled誤差を全lawへ転送する証拠が必要。個別診断だけでρ=0仮定を解除しない |
| 真の期待compiled価格・母平均Gの順位 | 価格tail/選抜後の不確かさも独立に制限する。局所精度PASSだけでは解決しない |

SOURCE読取だけでは、実装誤差と価格tailの**実際の大きさのどちらが順位を変えるかは判断できない**。
実装を含む主張を次に調べるという目的にはnative/compiled接続が先に対応する、という作業上の優先順位である。
tailの全保証を初期の局所精度診断の実施条件に加えない。

## 7. 再利用できるコードと持ち込まない条件

| 資産 | 利用できる部分 / 今回確認した制限 |
|---|---|
| P1 `run_h6_pauli_p1_bias.py` / `pauli_mechanism_bias.py` | law/target/reference対応、ν/B所有、理想split/finiteの定義。8伝播を再実行する必要はない |
| P2 `selected_members` / `restore_pauli_h2_h6_evidence.recover_member` | 今回実際に使用したSHA付き選択archive読取。2読込fileはP2固定SOURCEのbytesと一致 |
| P2 `pauli_mechanism_cost.recipe/source_tokens`・D2 descriptor | compact/full eventの共通view、固定selected-target/tokenの接続、phase所有とinput意味論。sampling/native再buildなしで参照できる |
| H6 `h4_pauli_pr_search` / `pauli_vector_tight_enclosure` | 保存実効CDF/角度、matrix-free signed Pauli作用、数値包含。既存law平均を再実装しない |
| P3 `matched_partition_finite_bit` | CDF整数mass、scalar角度区間、差の上側転送の考え方は再利用候補。ただしORDERS=(0,2,4)、q8/r4、dense H4、Gaussian前提のままH6へ使わない |
| `h4_r0_empirical.propagate` | 保存QPY逐次statevector作用とraw observable/ν。既存実装自体はEMPIRICAL_ONLYで数値誤差上界なし。H4固定preflightはH6用に新規接続が必要 |
| `h4_arb_bridge` / selected-target core | native rz/sx/x/cx逐次作用、unitary L2 recenter、区間差の規約は参考になる。H4固定adapter/旧grant/controllerや旧科学承認を移植しない |

H6はsystem12/Full13 qubitで、P3のfull dense operator/SVD経路を無理に使わずvector作用を優先する。
旧H4の制限を外してprivate entry pointを無断実行しない。必要な場合はH6専用の小さいadapterを新SOURCEへ固定する。
新たな承認証明書・grant発行・approval用ledgerを作らない。共有library/環境も変更しない。

## 8. 新科学計算が必要な場合の最小候補案（未実行・未承認）

保存index0000の3候補×両軸=6組を出発点にできる。R64を両比較で二重に評価せず、P2のQPY/意味論/価格を再利用する。
次の案は局所的な実装診断を目的とし、全law保証や費用再会計を目的にしない。

| 項目 | 限定案 |
|---|---|
| 対象 | R64/H32/H64、confirmation/index0000、cosine/+Im sine |
| selected reference | 同じ保存event列・理想係数/時間・binary64 event角度・scalar規約からC_iを構成。平均targetを個別targetに流用しない |
| 評価件数案 | selected C_iのsystem作用3複素値＋保存native Full6軸＋保存compiled Full6軸＝最大15状態作用、6比較組 |
| 初期精度案 | complex128による局所診断。演算に応じた数値不確かさを別欄とし、厳密上界がなければEMPIRICAL/nullを保持 |
| 参照budget | 保存候補・精度・軸ごとのrho_guardを単位合わせして引用。新閾値・新N/Gは生成しない |
| 差 | selected C_i→inputとinput→compiledを分け、B_savedは一度だけ。raw φ/νを保ち、full-wrapperの測定observableで比較 |
| 位相 | 全体global phaseの非観測性とbody相対phaseを分離。誤差を事後phase alignmentで除かない |
| 禁止 | 新sampling/compile、Full再生成、SCF/FCI/JW/DF/state/full-H reference生成、N/G・感度会計、候補探索 |
| 終了 | SOURCE/保存入力不一致・意味論不一致・非有限値・分解能不足は記録して停止。自動retry/高精度化/追加trajectoryへ進まない |

これは実装可能性と必要件数の案で、今回の実行許可を意味しない。正確なSOURCE・入力hash・誤差伝播・数値判定政策を実装時に固定する。
新しい局所誤差値の評価にはユーザーの明示的な科学実行許可が必要。
H64/R64のindex0000はorder0だけであるため、この6組のPASSをorder2を含む全lawへ一般化しない。
局所診断後に全lawや価格tailをどこまで求めるかは、目指す主張に応じて別途判断する。

## 9. 今回の完了範囲と次の担当

今回の成果は本調査書、3候補の保存law/target/価格/ρ_guard/QPYへの機械可読対応表、P3の区切りと元研究への作業対象の切替である。
技術照合は固定21member・12 QPY bytes、SOURCE/参照の読取と保存文字列の転記に限定した。
新科学的数値解析・状態作用・sampling・回路生成/compile・感度/N/G更新・新誤差/tail上界は0。
過去の人工PASSは過去の証拠として引用し、今回の新testsと数えない。

次の担当は**ユーザー**：現状の条件付きcase studyとしての報告を維持するか、§8の局所実装診断を行うか判断し、実行する場合だけ対象・件数・精度政策を明示許可する。
許可後の担当は**Codex**：既存P1/P2/逐次作用を最小接続し、人工検査・科学SOURCE固定・指定範囲の実行と報告をまとめて行う。
P3の追加精密化、H6の新探索・tail実行・費用再会計、DF B2/RPE全round計算は今回開始しない。
公開・remote照合後、全旧mandatory STOPを維持して終了する。

[P3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/426ccda5df044d2e94d4a18d98cde75a96ab1e6b/artifacts/df_pauli_matched_partition_validation/2026-11-02/finite_bit/run01/REPORT.md
[P3S4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a1d3f32aba6416fd1e200cd66a97d1a825b5dd73/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage4/run01/REPORT.md
[HREPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/REPORT.md
[RREPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/docs/pauli_rte_event_extension_20261026.md
[P1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bfb4ca75437a9b3605a8f57b5f82100615dd034f/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/bias/REPORT.md
[P2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c9073274daee69c9b69d8db3dc332bf6132c2259/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/cost/REPORT.md
[SYNTHESIS]: pauli_h2_h4_h6_paper_evidence_synthesis_20261101.md
[MECHANISM]: pauli_pure_rte_cost_mechanism_inventory_20261101.md
