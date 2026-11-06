# 研究C：再配置ENTRY＋PRと保存CDF8/XDF8＋強Dの比較条件の確定

調査日：2026-11-07 JST。依頼指定のファイル名の日付は保持した。**新規科学計算・保存数値の再集計・回路build/compile・NPZ/QPY数値decodeはすべて0。** 保存JSON、固定契約、SOURCEと関数の静的照合による技術報告である。

**結論：保存CDF8＋D、XDF8＋Dは、比較B「試験済み強実装の条件付き点値比較」ではともにD1として利用できる。** 同じ元物理H、精度目標、RPE処方、compiler、serial会計への接続を確認した。保存値はENTRY E2＋PR/r16が75,621,395,471 RZ、CDF8＋Dが102,056,444,569、XDF8＋Dが100,267,421,030である。ただし前者はideal条件のfinite-bit価格proxy、後二者はlevel B数値合成誤差を含む旧D価格であり、同じ誤差資格の独立比較結果ではない。再配置・候補・アルゴリズムをまとめた保存構成の点値比較に限定する。

比較Aの共通条件の多くは既に一致する。一方、**「同じ線形Z再配置を与えたCDF8/XDF8の強Dにも有利か」は未評価**である。再配置はPF分割と一体basisを変え、Dのkernel、PF誤差、shotsを変え得る。この問いを採用する場合に限りD3であり、§10に未承認の最小案を示す。限定した比較Bを提示するための追加計算は不要である。

## 1. 目的と固定資料

対象は直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2、4 spatial/8 spin orbitals、物理sector36の次の3構成である。

| 構成 | 固定された対象 |
|---|---|
| ENTRY E2＋PR/r16 | C0、ENTRY strength .03、64更新の保存number-CDF8。元順LD2のtail線形Zを一体headへ再配置。partial S2/s1、h=δ=.1、K0、m0..12、own-DF ground oracle、状態準備除外 |
| CDF8＋D | 保存`B2_CDF_relative`。通常number-CDF8、元frame順、再配置前の全frame ordered symmetric S2/s1、h=.1、m0..12、同じoracle/除外条件 |
| XDF8＋D | 保存`B3_XDF_spin_relative`。保存OpenFermion eigh元順prefix8、spin-lift＋relative Gaussian、その他は旧CDF8＋Dと共通 |

以下を固定証拠として参照する。本文の[R]、[D]、[L]、[C]、[H]はこの表を指す。

| 記号 | 固定資料・実行との関係 |
|---|---|
| [R] | [再配置後結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/docs/research/research_c_h4_relocated_downstream_results_20261106.md)。同commitの[comparison.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/artifacts/research_c_h4_handoff/2026-11-06/science/run01/comparison.json)、[公開summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/artifacts/research_c_h4_handoff/2026-11-06/publication/summary.json)、[800標本CSV](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/artifacts/research_c_h4_handoff/2026-11-06/publication/native_price_samples.csv)、[result audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/artifacts/research_c_h4_handoff/2026-11-06/execution/result_audit.json)、[STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/artifacts/research_c_h4_handoff/2026-11-06/science/run01/STOP.json) |
| [D] | [旧強D結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/docs/research/research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md)。同commitの[実行契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/run01/contract.json)、[SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/run01/SOURCE.json)、[STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/run01/STOP.json)、[CDF結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/run01/candidates/B2_CDF_relative.json)、[XDF結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/run01/candidates/B3_XDF_spin_relative.json) |
| [L] | [既存比較可能性調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/77ded5c12db21572933653cb26758436da3d9ecd/docs/research/research_c_baseline_comparability_20261106.md)、特に§6〜10。本報告は残っていた条件対応を具体化する。RC-DF/SCDF/UWCの文献調査は繰り返さない |
| [C] | [a670固定契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a67036a6baafc00deaa38a1b0b801d313d519ad5/docs/research/research_c_h4_relocated_downstream_contract_unapproved_20261106.json)。元の16入力・科学条件の定義 |
| [H] | 実際に採用された[引継ぎ契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/artifacts/research_c_h4_handoff/2026-11-06/science/run01/contract.json)と[64-file SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cfc64b70861a6482dddd042da8a110abebd6ad7a/artifacts/research_c_h4_handoff/2026-11-06/technical/SOURCE.json)。実装SOURCE `cfc64b70861a6482dddd042da8a110abebd6ad7a`、SOURCE SHA256 `6bd88bc86bd1da2876f87b8ae67be509b15727f01517cea9c7219f91c350c4cc`、契約SHA256 `5454c539bc52cbf51fad6a8af6802d5acbe48ff1e341e1edf12ae60f19a66bbb` |

旧強Dの科学実行SOURCEは`1aeedd96178a3f50a15bc2701c3c332f4f391b64`、数値実装は`969c17d623d56eb645417b281deadb1be4757699`。実行契約はbinding v2、SHA256 `1bf82b39693f07503eec2879c93cb17a734ccdfdcdca03b50544a15444a56cd2`、31-file manifest SHA256 `881694bde50cfa1ad50a67fef86c0bffc42c165559a6f8be56e148573e7cad8c`である。B2 nativeは`880006396326a4d46f4a4030564279e92fadb9cf`、B3 nativeは`b40fce5c62c9fff97f68073cbc02182e4bbf2b9b`。契約の歴史的`UNAPPROVED`名・metadataだけで未実行と判定せず、保存結果とSTOPまで接続した。

[R]の実行は検証済み288標本の再利用＋512標本の新計算、合計800標本とD3件で完了した。元processの終了143・元STOP/completion欠落は別のtermination記録に残り、旧PASSは作られていない。継続runの`STOP.failed=false`、retry/resume0を確認した。[D]は86/86 COMPLETE、`STOP.failed=false`、retry/resume0。今回これらのrunner・preflight・auditを再実行していない。

正式共通運用指示はworktrees共通の`research_project_instructions_unified_2026-10-25.md`原本を確認し、特に§8、§10を適用した。本調査はその範囲での条件照合であり、重要GPT科学レビュー、新規性・論文化GO/NO-GOの採択ではない。

## 2. 物理Hamiltonian・DF精度の整合

### 2.1 同じ元Hへの来歴を接続できる

旧強Dの`original_input`は元P-H4 snapshot、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`、公開commit `86b8b002704aced8ca239414080f53793de81658`、生成SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12`である。[元receipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86b8b002704aced8ca239414080f53793de81658/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/receipt.json)はz=0,1,2,3 Åの直鎖H4、STO-3G、SCF収束、4 spatial/8 spin/sector36を記録する。

新しい`original_h_g_constant.npz`のSHA256は`1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`であり、旧snapshotと同じコンテナbytesではない。しかし[Stage A準備SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5a185964090f2cb8d19d445d41cf520c4d8ed2c7/src/trottertracks/c_pr_aware_df_pilot.py)の`prepare_worker`は、同じ元snapshotの`one_mo → h`、`eri_chemist → g`、`nuclear_repulsion → constant`だけを抽出する。保存Stage A契約・`prepare/science_calls.jsonl`の`original_input_load COMPLETE`と停止記録がこの実行を接続し、[Stage B契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/docs/research/research_c_pr_aware_df_stage_b_contract_draft_20261027.json)にも元8a9…と抽出後1d608…の両identityが残る。再SCF・軌道回転・再尺度化を挟む処理ではない。

[旧fit_export](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/artifacts/research_c_c1_stage1_compat_science/2026-10-25/run01/fit_export.json)は元snapshotの`one_mo`、chemist `eri_chemist`、核scalar、`mo_coeff`のmember identityを保存する。元のcanonical real RHF MOの順序・位相をそのまま継承するため、候補ごとのUは同じ元MO座標からのframe回転として解釈できる。spin順は2p=α、2p+1=βであり、XDFのspin-liftはこの順序を保つ。これは保存来歴と抽出コードからの同一性確認であり、本調査で配列を開いてtensor等値を再測定したものではない。

κは元gから補正された`ffsim_corrected_kappa`を一度使い、切り詰めたgや最適化後Jから再補正しない。旧exportと[H].`kappa_array_identity`のκ identityはともにshape[4,4]/`<f8`/SHA256 `5f0ca073f19fcc99c5e1c81b13910d527717ad4758236f59f714d47dba8ed32f`、核scalarはshape[]/`<f8`/SHA256 `874b34c93c04576c14050d91ecace903d7b2693560bf77a05b2178fc48fb232b`である。旧XDFの保存one-bodyと元g補正κの対応も既存数値gateの範囲で確認されており、全候補のκ配列がbitwise同じという主張はしない。

### 2.2 近似Hとrankの意味は異なる

| 項目 | ENTRY E2 | 旧CDF8 | 旧XDF8 |
|---|---|---|---|
| 保存表現 | number-CDF、一般のsigned symmetric J＋orbital U | ffsim number-CDF、同じ一般J形式 | signed λとone-body Gのsquare fragments |
| rank8 | spatial4上のJ/U各8 frame、一体項は別 | actual rank8、一体項は別 | 元のeigh順λ/Gの最初の8因子、一体項は別 |
| 候補生成 | C0/ENTRY/.03/64更新の有限endpoint | 保存ffsim fit、max_vecs8、tol1e-8 | 保存XDFを再fitせずprefix8 |
| DF sector誤差の保存値［Ha］ | error `2.319854032718789e-9`、guard `3.3210657426701382e-9`を別fieldで保持 | norm＋guard `7.666215916301914e-5` | norm＋guard `9.533901455091894e-5` |
| 共通予算 | `.00019875`、保存`budget_PASS=true` | `.00019875`、保存PASS | `.00019875`、保存PASS |

E2の候補ファイルSHA256は`0e1119dfb6be9b75fb5e592b7a65c76cda61e348fd7cad65d7f104bf997929f5`。[H].`candidates.E2.physical`が候補J/U、元入力、sector error、guardを結び付ける。E2のguardは`sector_operation_scale_guard8_proposal_v1`、旧側もoperation-scaleのengineering guardである。いずれも厳密rounding enclosureではない。表の新しいerror＋guardの和や新しいnormは計算していない。

XDFのrank8は保存順で予算を最初に満たしたprefixであり、任意順序・任意XDFでの最小rankではない。squareのcore制約と一般Jの自由度は異なるため、rank8だけで表現能力・最適化条件の同等性を認定できない。旧CDFのoptimizer終了とE2の有限更新も、共通の収束・最適性証明ではない。

**元の物理Hは共通だが、候補ごとの近似H_DFとown-DF groundは共通でない。** この差はDF energy reserveの消費、固有energy・alias、PF分割誤差、PRのmass/attenuation/RTE誤差、shots/W、native priceに影響する。同じ予算に収まる異なる近似を比較することは可能であり、実際のDF誤差まで等しくする再fitは比較の必須条件ではない。

### 2.3 線形Z再配置はHを保つがPF分割を変える

実装は[c_h4_relocated_downstream.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cfc64b70861a6482dddd042da8a110abebd6ad7a/src/trottertracks/c_h4_relocated_downstream.py)の`prepare`、`raw_generators`、`prepare_candidate`。元frame順LD2でtail F2〜F7だけの線形Zを一体headへ移し、F0/F1、各frame identity、ZZを保つ。J/Uは変更しない。`rows=J.sum(axis=2)`を用いてκへの一体補正とscalar shiftを組にし、tailのRZ primitiveだけを除く。核scalarは一回、追加shiftは移送した項との恒等式の一部であり、核反発の二重加算ではない。

独立CAR経路で再配置前後の全Hと保存sector Hを照合したのが[R]の証拠である。同J/U Dも同じLD2で作った再配置モデルを受け取り、DだからといってLD8で再配置し直さない。Dの全frame ordered S2は、その後で同モデルの全8 frameを使うという別の操作である。

旧CDF8/XDF8強Dにはこの再配置がない。全Hが保たれても、分割されたgeneratorのcommutator、S2 kernel、head Gaussian、PF13 round誤差は保たれるとは限らない。したがって旧D値を「再配置済みDの価格」に読み替えることはできない。

## 3. PF/RTE・RPE政策の比較

### 3.1 共通の最終課題と相違する時間発展

[D].`accuracy_target`と[H].`target`はJSON値として一致した。共通条件は化学精度`.00159 Ha`、DF reserve fraction`.125`、base time`.1`、β_total=π/6、β_stat=π/12、α_axis=.01/26、候補ごとのα_total=.01、m=0〜12、t_m=.1×2^m、own-DF ground oracle、alias確認、状態準備費用除外である。PF誤差やDF誤差を実測energy誤差と呼ばない。

旧Dは元順F0〜F7＋一体項の**全frame ordered symmetric S2、s1のみ**。PR/rやRTEはない。再配置E2＋PRはLD2のpartial ordered symmetric S2＋signed corrected RTE、s1/h=.1/K0/r16である。rは一つのkernel内のRTE処理で、round反復数に追加乗算しない。[R]の同J/U Dも全frame ordered S2/s1であるが、旧Dとは候補と再配置後generatorが異なる。

旧Dのordered schedule・全frame構築は`c1_stage2_native.NativeBuilder.ordered`を継承した`c1_h4_strong_baseline.StrongBuilder`、PRは`c_pr_aware_df_stage_b.build_kernel → c1_stage3_native_bridge.partial_kernel`を使う。元順と対称scheduleを保持し、結果を見てframeを並べ替えていない。表現に適したsquare/general-J primitiveの相違を、compiler不一致と混同しない。

### 3.2 同じshots式へ入れる誤差の資格が異なる

[共通accuracy実装](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cfc64b70861a6482dddd042da8a110abebd6ad7a/src/trottertracks/c1_stage3_accuracy.py)の`assess_round`は両系で同じbytesである。raw complex signalに対して

\[
\delta_m=A_m(\epsilon_{PF,m}+\epsilon_{RTE,m})+\epsilon_{syn,m}+\epsilon_{law,m},\quad
\rho_m=A_m-\delta_m,\quad \beta_{det,m}=\arcsin(\delta_m/A_m)
\]

を用い、coordinate tolerance `ρ sin(β_stat)/sqrt(2)`から同じlegacy Hoeffding整数shotsを求める。`assess_energy`はDF誤差［Ha］を一度だけ加え、DF＋β_total/(h×2^M)と共通目標を比較する。DF誤差をdimensionless PF/signal誤差へ混ぜない。全13 roundの適格性が先で、その後に費用を出す。

| 項目 | 再配置E2＋PRの主価格lane | 旧CDF8/XDF8＋D |
|---|---|---|
| lane | `ideal_exact_native`、`ASSUMED_SCENARIO` | `B_NUMERICAL_OPERATOR_UPPER_BOUND`、level B pilot |
| PF/RTE | 候補別PF、corrected RTE、attenuationを数値評価 | 候補別PF、RTE誤差0、A=1 |
| synthesis/law | exact native synthesis・ideal finite-RTE IID lawを仮定し0。実装law証拠が0と測れたわけではない | 保存full512 compiled/reference Frobeniusとnorm-growthを各roundへ伝播。非零synthesisを計上。deterministic lawのlaw誤差0 |
| 実装証拠lane | 全law synthesis・implemented-law誤差等が不足し実費用は未資格 | 数値十分条件`NUMERICAL_PILOT_CA_PASS`、厳密enclosure・actual CAは未資格 |

同じ最終energy estimation課題を共通のconditional oracle/serialモデルで価格化しているが、実機・実装lawまで同じ保証で課題を達成したという比較ではない。旧Dの非零合成誤差を黙ってゼロにしたり、PRの未知の実装誤差を0で補完したりしない。

### 3.3 比較AとBの区別

**比較A：共通条件を揃えた比較。** 物理入力、DF予算、RPE target、compiler、資源会計は既に共通である。ただし同じ再配置機会と同じ誤差laneに揃えたCDF8/XDF8対照は保存されていない。DF表現・PR/Dのアルゴリズムの違いまで消す必要はないが、この相違を残したまま「再配置・誤差政策も統制した比較」と呼べない。

**比較B：試験済み強実装の比較。** 旧CDFはgeneral-J/legacy bases＋relative Gaussian、旧XDFはsquare/spin-adapted bases＋relative Gaussian、E2は再配置＋partial PRの既存実装をそれぞれ使い、同じ資源会計で保存点値を並べる。この用途では旧Dの再fit・再compileは不要である。ただし誤差lane差を併記した条件付きpackage比較であり、各表現の大域最良実装やPRだけの因果効果ではない。

## 4. Native実装・総RZ会計の比較

### 4.1 Compiler・control・融合

旧新の契約`compiler`は一致する：Qiskit1.3.0、basis `[rz,sx,x,cx]`、optimization_level1、seed_transpiler17、approximation_degree1.0、qubits_initially_zero=false。Gaussian version `nature_jw_exact_zero_lapack_atan2_v1`とP0 `qiskit_P0_mpfr160_declared_phase_v1`を使用する。

両方とも8 system qubits＋ancillaのcontrolled evolutionで、basis回転をuncontrolledの対として置き、中央のdiagonal primitiveをcontrolする。frame identityと核scalarの位相をancilla側に保持し、global phase fitはしない。`StrongBuilder.sequence/transition`は`V_following† V_previous`のrelative Gaussianとanalytic vacuum phase補正を実装する。旧CDFはlegacy spin-orbital eigensolver basis、旧XDFは`prepare_xdf_basis(...SPIN)`でexact interleaved spatial-spin liftのbasisを使う。PRのheadは同じprecision builderのrelative融合、tailはfull-frame basisと同一basis連続runの結合を使い、head/tail境界の追加融合はしない。

**SOURCE系列の相違を、科学条件の不一致と一括判定する必要はない。** 旧31-file manifestと新64-file manifestで、次の再利用実装はそれぞれ同じhashだった。

| 共通ファイル `src/trottertracks/` 以下 | 対応する処理 |
|---|---|
| `c1_stage3_accuracy.py` | round精度・energy・総資源集計 |
| `c1_stage3_connection.py` | serial repeatとshell合成 |
| `c1_h6_ca_resources.py` | D round費用・保存packet接続 |
| `c1_h4_strong_baseline.py` | spin-lift・relative Gaussian強builder |
| `c1_h6_native_precision.py` | precise Gaussian emitter |
| `c1_h6_compiler_precision.py` | P0 compiler/phase政策 |

また`pr_dictionary_validation/f2_execution.py`のmetrics実装は旧B2 native SOURCEから新SOURCEまでdiffなしだった。これは当該ファイルのGit内容照合であり、旧manifestに直接載ると偽っていない。`c1_h4_native_fair.py`の旧数値実装から新SOURCEへの差はfull-unitary decode backend追加であり、compile・kernel builder・metricsの変更ではない。新経路も全512 coverageを維持する。backend高速化の差だけを理由とする科学再計算は不要である。

RZは最終basis gateの`rz`個数、CXは`cx`個数であり、角度精度をT countへ変換した値ではない。depth系も両方で同じ6指標を出す。FT費用・state preparation込み価格へ外挿しない。

### 4.2 W・kernel・shellは実質的に同じ会計

旧Dは`c1_h6_ca_resources.serial_round_cost(packet,m)`から、PRは実行経路の`c_pr_aware_df_m3.serial_cost(packet,kernel,m,axis)`から、共通の`c1_stage3_connection.compose_serial_cost`を呼ぶ。旧packetの`s=1`、PR実行経路の`substeps=1`を確認した。別の旧helperのdefault substepsを当該実行へ混入させない。

各round/axisのnative concatenationは、`2^m × kernel + head shell + axis shell`。同じ`aggregate_totals`がaxis shotsを掛けて合計するため、記号上は両方とも

\[
C_{total}=W\overline C_{kernel}+C_{shell},\qquad
W=\sum_{m,a\in\{X,Y\}}N_{m,a}\,s2^m
\]

である。Dではkernel単価が確定値、PRでは一つの登録価格分布からの標本平均となる。depthはfencesで分けたscheduled serial sumであり、長いround全体をglobal transpileした最小depthではない。

旧packetのhead/X/Y shell 6指標と新`shell_reports.json`は対応する値が一致する。RZはhead2、X2、Y1、CXはすべて0。headは測定なし、各X/Y readoutは1測定。head/kernel/readoutを同じ範囲で数えるため、measurementをkernelに含めた差やshell二重計上ではない。

以下は**既存fieldの引用だけ**である。旧DのWを新しく求めたり、保存価格へ別係数を掛けたりしていない。

| 保存量 | E2＋PR/r16 | CDF8＋D | XDF8＋D |
|---|---:|---:|---:|
| kernel RZ | 平均5,512.3125、独立final64 | 11,062、確定回路 | 10,885、確定回路 |
| planned shots合計 | 15,118 | 13,606 | 13,600 |
| W | 13,718,624 | 本調査では新規算出しない | 本調査では新規算出しない |
| shell RZ | 52,913.0 | 同じshell規則で保存総費用に計上済み | 同左 |
| 条件付き総RZ | **75,621,395,471** | **102,056,444,569** | **100,267,421,030** |

E2の出典は[R]と[E2/final_r16/summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/artifacts/research_c_h4_handoff/2026-11-06/science/run01/E2/final_r16/summary.json)の`statistics.calibration`、`statistics.lanes.ideal_exact_native`、`bindings.ideal_exact_native`。Dは[D]の候補JSONと対応する`.packet.json`。800/64というclassical価格標本数をshots/Wへ掛けない。planned shotsは実量子shot実行数ではない。

## 5. 統計と証拠資格

1. **保存Dの確定価格**：固定回路metricsと登録shots処方から決まる費用。保存intervalがpointと一致するのは価格分布の標本誤差がないためで、numerical error enclosureやhardware cost確定を意味しない。DにPRと同じ64価格標本を新しく要求する理由はない。
2. **E2の価格推定**：selection32でrを固定した後の独立final64。finite-bit lawで生成したclassical native価格の標本平均を、ideal finite-RTE費用に接続したproxyである。`ideal_law_expectation=null`、all-law synthesis/implemented-law error/ideal-law price transferは未資格。
3. **区間の適用先**：[R]の主比較A/Bは登録2比較の各名目97.5% Welch差区間。同J/U PR/Dの個別名目95%比区間とは別である。E2の保存PR/D比`.7187167479963066`、区間`[.7050069520663036,.7324265439263095]`は分母105,217,244,042の**E2自身のD**だけに対応する。CDF8/XDF8への区間や独立final比較へ付け替えない。
4. **今回得ていない統計**：旧D対E2の差・比・savings%・CI・p値・新しい同時資格を計算していない。旧Dは新しいfinal標本ではなく、跨SOURCEの独立確認実験も実施していない。保存点値の大小の記述を、母平均・全lawの統計的優位と呼ばない。
5. **数値correctness**：両系列はfull512 raw/compiled/reference、実際に構築したinverse、return、phase、XYを検査した数値pilot。新系列は800個別標本＋Dのcorrectness FAIL0。個々のPASSは未観測trajectoryすべての合成誤差上界を証明しない。旧Dはfull512の誤差とnorm guardを13 roundへ伝播した点で誤差証拠が異なるが、全演算の厳密丸めenclosureではない。

[R]のP/r16副比較とbest-policy比較の`UNRESOLVED_DISTRIBUTION_DIAGNOSTIC`はそのまま維持する。診断を通過した別の比較で上書きしない。どちらの系列もown-DF ground準備はoracle扱い、actual CA・hardware/FT資源保証・一般分子での優位は未認定である。

## 6. 公平比較条件の対応表

表のD1は限定した保存比較、D2は歴史的文脈でのみ使う範囲、D3は明示した追加の科学的問いに必要な検証、D4は重要情報未取得を表す。用途を分けて判定する。

| 比較軸 | ENTRY E2＋PR | CDF8＋D | XDF8＋D | 判定・根拠 |
|---|---|---|---|---|
| 物理H | 元snapshotのh/g/核scalar抽出 | 同じ元snapshot | 同じ元snapshot | D1：元8a9…→抽出1d608…の保存来歴と`prepare_worker`で接続。再測定ではない |
| DF誤差 | error/guard別、予算PASS | norm＋guard PASS | norm＋guard PASS | D1：共通CA/8。実誤差とguard資格は同じでない。§2 |
| Rank・表現 | 一般J/U 8 frame、有限ENTRY endpoint | 一般J/U 8 frame、保存fit | square λ/G prefix8 | D1 B：8因子対照。表現能力・収束公平性はrankだけでは保証しない |
| 再配置・scalar | F2〜F7のZ→κ/head＋対応scalar shift、I/ZZ保持 | 未適用、元κ＋核scalar一回 | 未適用、元κ対応one-body＋核scalar一回 | D1 B。再配置機会を揃えるAにはD3。H保存とPF保存を区別 |
| PF政策 | partial S2/s1、h=.1/K0/r16、LD2 | 全frame ordered S2/s1、h=.1 | 同左、元順prefix8 | D1 B：アルゴリズム差。再配置済み強DのPFは未評価 |
| RPE | m0..12、共通target、PF/RTE/attenuationを入力 | 同target、PF＋合成誤差を入力 | 同左 | D1：課題・式は共通。laneまで統一した資格は未成立 |
| Ground | E2自身のDF ground oracle/alias | CDF自身のDF ground oracle/alias | XDF自身のDF ground oracle/alias | D1：同じ条件付きoracle政策。状態そのものは異なる。準備費用除外 |
| Native | precise head＋relative、tail full-frame run融合 | general-J、legacy basis＋relative | square、spin-lift basis＋relative | D1 B：表現固有の試験済み強実装。無理な回路統一は不要 |
| Compiler | Qiskit1.3/opt1/seed17/P0、同basis | 同じ | 同じ | D1：契約dict、precision/builder SOURCE一致。全512 backend変更は価格定義変更でない |
| W・Shell会計 | s1のW×平均kernel＋shell | s1のW×確定kernel＋shell | 同左 | D1：同じcompose/aggregate、同shell範囲、同6指標 |
| 統計 | selection後final64のproxy平均 | 確定D価格、歴史的実行 | 同左 | D1は点値比較のみ。跨baseline CI・独立確認資格は新設しない |
| Correctness | 個別full512 PASS、ideal lane仮定、all-law未資格 | full512＋13 round伝播、level B pilot | 同左 | D1 Bの資格注記必須。actual CA/all-lawの比較資格は双方なし |
| 総合比較資格 | 条件付きpackage価格proxy | **D1 B**。同一再配置・laneのAとしてはD2、必要時D3 | **D1 B**。同一再配置・laneのAとしてはD2、必要時D3 | SOURCE別系列だけではD2限定にしない。厳密Aの欠落は具体的に特定済み。主要条件未取得を理由とするD4ではない |

## 7. 既存証拠だけで成立する主張

使用できる記述は次のとおりである。

> 同じ保存H4・1.00 Å/STO-3G入力、DF reserve、RPE target、compilerとserial資源会計の下で、再配置ENTRY E2＋PR/r16の保存条件付き総RZ proxy点推定75,621,395,471は、再配置前の試験済み強CDF8＋Dの102,056,444,569および強XDF8＋Dの100,267,421,030より小さい。PR側のideal synthesis/law仮定とD側のlevel B数値誤差政策は異なる。この結果は保存構成どうしの限定した点値比較である。

この記述には新規の価格算出・比率・統計検定を必要としない。「別SOURCEだから比較不能」という未確定部分は、元入力への接続、target/compilerの一致、共通関数とshellの照合によって解消した。[L]の歴史的対照としての価値を保ちながら、限定比較Bとして直接並べられる条件を明確にした。

既存の同一J/U D対照は、E2の再配置後モデルにおけるPRの条件付き名目低費用を支える別の証拠である。ただし「E2でPR/Dが低い」と「別J/Uの旧強Dより点値が低い」を接続しても、旧CDF/XDFの再配置済みDにも低費用だという推移的な結論にはならない。

成立しないのは、PRだけ・ENTRY目的だけ・線形Zだけの因果的節約、同一再配置を与えた全強Dへの勝利、全law母平均の統計的優位、厳密CA/実機/FT優位、一般的最適性・独立新規性の認定である。保存DF誤差の差を費用差の唯一の原因とする説明もしていない。

## 8. 不足する証拠

| 不足 | 科学的な重要度と原因 | 保存資料・関数で解決できる範囲 |
|---|---|---|
| 再配置済みCDF8/XDF8強DのPF/native/RPE価格 | **順位を変え得る。** 一体basis・S2分割・PF誤差・shotsが変わる。比較Aとして再配置機会を揃える問いではD3 | 元J/Uまたはλ/G、DF誤差、own-ground/reference、builder、RPE・会計は再利用できる。旧点値から新PF/native値を代数的に復元できない |
| 同じlaneでの跨baseline価格 | **誤差資格の主張に重要。** PRのideal仮定と旧Dの数値synthesis伝播が異なる | 共通ideal laneのD評価は既存accuracy engineで可能だが、今回未評価の値を埋めない。PRの実装law資格はDを追加評価しても取得できない |
| 跨baselineの独立統計確認・CI | 母平均優位を正式主張する場合に重要。既存finalは別の登録比較 | 同じDを64回数える必要はない。保存PR/D CIの分母差し替えは禁止。本調査では新CIを作らない |
| 実装law/全law synthesis、actual CA、厳密enclosure | hardware/保証主張には重要だが、今回の限定比較Bの不足を超える | 個別800 PASSや旧D level Bで補完できない。今回、大規模な保証研究を追加案にしない |

反対に、SOURCE名、manifestのファイル数、準備契約のdraft表示、JSONのpath表記、full-unitary backend高速化の違いは、それだけでは新規科学計算の理由にならない。物理H・RPE・shell会計の未取得を理由にしたD4は、今回の静的調査では残らなかった。

## 9. 追加科学検証の要否

**既存の試験済み構成に対する限定点値比較Bを報告する目的なら不要。** 旧値をD1として利用し、誤差laneと再配置の差を同じ箇所に明記すれば研究判断の材料になる。精度target・compiler・会計を再確認するだけの再計算、DF誤差を等しくする再最適化、確定Dへの価格標本追加は提案しない。

**「標準的な再配置を強Dにも与えた後に追加利益が残るか」を中心の判断にする場合だけ限定的に必要。** これは単なる厳密化ではなく、対照側にも適用できる具体的な操作が費用順位を変え得るためである。現在の保存結果だけではこの判断の肯定・否定はできない。§10はこの問いをユーザー／GPTが採用した場合に限る未承認案であり、今回の自動的な次作業ではない。

## 10. 必要な場合だけの最小実行案（未承認・未実装・未実行）

目的は**固定した一つの共通再配置政策の下で、保存CDF8/XDF8のD対照を埋めること**に絞る。各表現の全実装・全再配置政策の最適化は含めない。比較Bの既存強実装を置き換えず、比較Aの追加対照として別に記録する。

| 項目 | 必要最小限の案 |
|---|---|
| 構成 | 保存CDF8＋再配置D、保存XDF8＋再配置Dの2件。対比先は保存E2/r16のfinal64と保存同J/U D。新fit・新候補・PR再探索なし |
| 物理入力・候補 | 元8a9…/抽出1d608…のH4入力、旧CDFの保存J/U（export file SHA256 `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08`）、旧XDFのsigned λ/G元順prefix8、各候補の保存ground/reference/DF guardを再利用。核scalar/元g補正κは維持 |
| 再配置政策 | E2との共通操作として元順LD2、F2〜F7だけの線形Z→一体head、frame I/ZZを保持。全frame Dの評価時にもLD8で再配置し直さない。これは比較A用の固定政策であり、各Dにとって最良という仮定を置かない。全frame Z移送や別順序の探索は含めない |
| PF/RPE/会計 | 全frame ordered symmetric S2/s1/h=.1、m0..12、同target/own-DF oracle/状態準備除外/P0/fences/shell。まずE2と同じ`ideal_exact_native`条件のD価格を別laneで取得し、同じnative検査から数値synthesis伝播laneも区別して保存する。PRの全law資格が得られたとは扱わない |
| 新数値評価 | 変更後の一体basis、sector/full-CAR分割とS2 reference、候補ごと13 roundのPF/alias・shots/費用、各候補1 native kernelとraw/compiled/actual inverse。H_DF自体のfit・DF norm・ground eigensolveは原則再利用し、新規計算しない。保存H/referenceが必要な形で揃わなければ欠落を明示し対象を再固定する |
| correctness | 再配置前後Hのsector/full一致、scalar/κ一回、XDFのsigned squareとspin順、raw/compiled/実逆の全512、return/phase/XY、shell、PF/roundの適格性。unqualified時は価格・優位をPASSにしない |
| 再利用関数 | CDF再配置は`c_h4_relocated_downstream.prepare/raw_generators`。強nativeは`StrongBuilder`、XDFは`prepare_xdf_basis(...SPIN)`、PFは既存`PF_rounds`、D評価は`c_update064_strong_d.evaluate`のprepared-model経路、精度/会計は`assess_round/assess_energy/serial_round_cost/aggregate_totals`を再利用 |
| 最小変更 | 現在のrelocation関数はnumber-J/U専用でありXDF λ/Gをそのまま受けない。XDFには保存square由来の線形Zを同じ意味で移す小さなadapterと独立CAR照合が必要。spin-lift/relative Gaussianは保持し、任意のgeneral-J builderへ無検証に置き換えない。既存封印SOURCEは変更せず、新しい差分として固定する |
| 統計・価格標本 | Dは確定回路なので新しい64価格標本は不要。E2の保存final64は再利用可能だが、新しい確認標本にはならない。最小案の判定は共通laneでの点値比較までとし、新CI・同時優位・独立追試の認定は含めない |
| 計算規模・資源 | 対象2、sector36/system256/control512、PF13 round×2、native build/compile各1、raw/compiled/実逆4 decode×2が中心。これは案の対象数で、実測call数ではない。complex128の512×512 matrixは4 MiB、8個の返却matrixは計32 MiBで、reference・expm/検査workspace・Qiskit・worker本体は別途必要。全体RSSとwallは未測定で、既存800標本の実行時間から比例推定しない。800標本を再計算する案ではない。2候補・独立decodeを空きCPU/RAMの範囲で並列化し、共通入力/cacheを再利用。恣意的なworker/RAM上限や共有環境変更を追加しない |
| 判断への接続 | どちらかの再配置DがE2の点値以下なら「その対照を含めても追加利益が残る」は支持できない。両方が上でも結論は固定2候補・政策・ideal価格proxyの点値まで。PRのみの因果効果・actual CA・全DF/PF最良性は認定しない |
| 実行許可・STOP | ユーザー／GPTが上の科学的問い・再配置政策・lane・点値判定を採用し、ユーザーが具体的SOURCE/入力/範囲の新科学実行を明示許可した場合のみ進める。結果前に2候補・閾値・再実行条件を固定し、結果後STOP。旧run/STOP・消費済み実行範囲を再開しない。今回はSOURCE/契約/runnerを作成していない |

この案を実行する必要があるかは、限定比較Bで十分か、共通再配置の比較Aまで研究上の判断を進めるかに依存する。単にSOURCEを一本化するための実行ではない。

## 11. 最終結論とSTOP

| 問い | 回答 |
|---|---|
| Q1 同じ物理Hamiltonianか | **元のHは同じ。** 元snapshot→h/g/c抽出の保存来歴を接続。近似H_DFまで等しいとはしていない。§2 |
| Q2 DF精度・rank・表現の影響 | 共通予算PASSだが実誤差、square/general-J自由度、有限最適化政策は異なる。own-ground/PF/PR/shots/nativeへ影響し、同rankだけの因果比較にはならない。§2 |
| Q3 PF/RTE・RPEの共通性 | target、h/s、13 round、oracle/alias/除外、shots/energy式は共通。partial PRと全frame D、再配置の有無、合成/law資格は異なる。§3 |
| Q4 Native条件は一致するか | compiler・precision/phase/control/計数は対応。CDF legacy basis、XDF spin-lift、PR head/tailの融合範囲は表現・アルゴリズム固有の差として保持。§4 |
| Q5 総RZ会計は直接比較できるか | **同じserial式・shell範囲として可能。** shots/W自体は候補別誤差に依存し等しくない。新係数・総費用の算出なし。§4 |
| Q6 旧Dは歴史値だけか | **CDF8/XDF8とも比較BでD1。** 共通再配置・同laneのAを確認した結果として使う場合は歴史的参考D2にとどまり、その問いを解くにはD3。§6 |
| Q7 correctness・統計・actual CAの違い | PRは独立final64のfinite-bit proxy、旧Dは確定価格＋数値合成誤差伝播。個別full512 PASSと全law保証は別。跨baseline CI・actual CA保証なし。§5 |
| Q8 再計算せず何が言えるか | 保存構成の条件付き点値ではE2が両旧強Dより小さい。PRだけの効果、再配置強Dへの優位、母平均/実機/FT一般優位は言えない。§7 |
| Q9 追加検証の理由・最小対象 | 比較Bには不要。同じ再配置を与えた強Dへの追加利益を問う場合だけ、PF/native/shotsが変わるため保存CDF8/XDF8の2 D対照が不足。未承認案§10 |
| Q10 新DF最適化は必要か | **不要。** 保存J/U・λ/G・reference/DF guardを再利用できる。新しいfitや候補探索を必要条件にしない |
| Q11 旧資料・共有環境の保全 | 独立worktree/branchでこのMarkdown一つだけ追加。旧SOURCE/契約/raw/STOP/grant/ledger/marker、他worktree/process/cache・共有設定へ書込みなし。科学call、数値decode、再集計0 |
| Q12 次の担当・承認 | **ユーザー／GPT**が比較Bの限定記述で十分か、§10の問いが研究判断に必要かを決める。Codexは公開照合後STOP。新科学実行や重要科学レビューはそれぞれ明示許可が必要で、自動開始しない |

静的確認では旧[D]の実行契約・SOURCE・STOP・候補JSON、新[R/H]の実行契約・SOURCE・STOP・比較JSON・公開summaryを指定固定commitと照合した。accuracy/compiler dictと主要6共通実装のSOURCE identity、shell・accountingの呼出し対応を確認した。数値evaluator、科学runner、回路処理、価格再集計は呼んでいない。文書のみの追加であるため、新しい数値test結果や独立レビューPASSも作っていない。

本報告を指定branchへcommit/pushし、remote SHA・tree・公開本文を照合した後に**MANDATORY STOP**。H2/H6、baseline fit、PF/RTE/RPE/native、新規集計、研究C正式新規性判断、論文化GO/NO-GO、重要GPT科学レビューへ進まない。次の担当は**ユーザー／GPTによる主張範囲・追加検証要否の判断**。Codexの自動的な次作業なし。
