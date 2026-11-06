# 新C：適用条件・SOURCE再利用・費用・最小検証の対応表

2026-11-06 JST、**SERVER 1 / GPUA100-gleap-01**。
branch：`codex/new-c-control-free-survey-server1-20261106`。
固定SOURCE：`2397c5ac511148423ae89f25f74f1de967d1d39a`（以下 **S**）。
主報告：[REPORT.md](REPORT.md)。文献の確認版・深度、固定5資料、実行禁止・保全の範囲は同書§1–2を参照。

今回の表は静的調査結果であり、実装や科学的GO/NO-GOの採択ではない。

- **適用可**：記載した既存機能・元条件内の再利用。新Cの成功や科学実行許可を意味しない。
- **適用不能**：記載した単純置換・入力条件では要求信号/情報を満たさない。方式family全体の不可能性を意味しない。
- **未資格**：追加access・性質・保証・費用の少なくとも一つを確認できていない。欠損費用を0としない。

## 1. 方式ごとの要求条件と適用判定

本研究の共通条件は元Hの厳密ground供給、元state準備費の除外、同じE₀・energy精度・成功確率である。新しい分子・geometry・basis・DF rank/L_D・delta窓・精度・round数は固定していない。

| 方式・一次資料のlocator | 得る測定量・必要な入力 | 数学的条件・不定性 | 固定groundとPRへの判定 | 費用への経路・不足 |
|---|---|---|---|---|
| **既存PR＋fresh-shot RPE**：[PR App.A (A18)–(A31)][PR]、S [rte][rte] / [accounting][account] | 各pathのcontrolled Hadamard Re/Im。元groundとancilla。raw/corrected規約を保持 | 線形coherent mean、独立bounded outcome、正半径・bias・alias/branch。PFと統計誤差は別 | **適用可：元条件内のbaseline**。exact groundは各pathの固有stateである必要がない。既存の条件付き保証を一般化しない | diagonal-only control・basis共有・phase保持を共通に与える。既存長q価格、元H表現包含等の未資格を維持 |
| **C-1 VPR**：[§II A、App.B (B1)–(B4)][C1] | target/secondaryと2種interferenceの絶対値二乗。追加stateと相対位相付きsuperposition、target projector、各normが必要 | DFTの有限支持・spectral independence、一意性はshift等を除く。零点、leakage、noiseでconditioningが変わる | **適用不能：同じexact ground braの直接構成**ではsecondaryはtargetに比例し、確率にE₀が残らない。**未資格：PRへ一般化した構成**は二次moment/一次meanの差とanchorが未解決 | `1+3R` 系列、全時間点shot、追加state/projector、古典固有値問題・支持探索。RPEのshot式は移植不可 |
| **C-1 2DPR/HIO**：[§II B Eq.(6)、App.C][C1] | 2D絶対値、dummy evolution、必要なboundary情報 | 可換dummy、非自明な第二方向、共役対称性・実非負spectrum/window。generic一意性とHIO収束は別 | **適用不能：exact groundの直接構成**ではH方向が単一phase因子。固定粒子数上の粒子数dummyもscalar。**未資格：PR**ではpositive spectrumと時間構造も未確認 | N×M時間点、dummy回路、測定、HIO反復/restart・memory。基底stateを別入力へ変更する費用とtarget同一性が必要 |
| **C-2 虚時間log微分**：[本文 (1)–(9)、Supp.A.3/B/C][Yang] | 正負短虚時間、real-time、投影確率、未正規化norm、既知積分初期phase | local H/短相関などの効率条件。logの解析性・zero-free近傍/小確率対策、差分・積分・Trotter/統計誤差 | **適用不能：normalized ground survivalだけ**ではenergy情報なし。**未資格：有料normalizer取得を含む経路**。DF/nonlocal groundとPR log微分の条件は未確認 | QITE unitary取得、tomography/期待値、normalization、両符号×時間grid、古典積分を課金。normalizerをE₀から生成しない |
| **local sequential Hadamard**：[§II/III.1、App.A][local] | 単一gate controlと各prefix間の複素積、initial-state projector | 固定unitary gate列、prefix overlapの下界、state inverse。途中が小さいとshot増 | **未資格**。固定pathへの既知手順の適用余地はあるが、fresh pathの確率を平均した後の再構成保証とは違う | 全prefix測定、projection・ancilla、pathwise reconstruction誤差、外側平均の費用。回路長だけで利益を判断しない |
| **local論文の完全control-free variant**：[App.F][local] | 既知prefix複素振幅、追加Pauli rotation後の確率群 | phaseを逐次引き継ぐ基準、例外/符号条件。任意の未知振幅を無条件に復元しない | **未資格**。PR pathのscalar/order・projectionとmeasurement conditioningを含む別照合が必要 | 追加rotation・測定family、例外処理・prefix数。詳細PR価格/finite-confidence成功は未取得 |
| **新A reference/CSWAP**：[新A固定調査§3][AB]、[SE §II (1)–(3)][SE] | 2 register間の量子干渉。reference固有作用の既知path phase、actual adjoint | canonicalの共通固有基底条件とfresh-shot単一block条件を区別。reference半径とpath phaseの補正が必要 | **未資格：新A性能**。固定静的仮説のみ参照。SERVER 2 pilot結果を取得・変更していない。新Cと同じ方式にまとめない | reference準備/register/reset、CSWAP/routing、補正、両lane。既存basis control削減を二重に節約へ数えない |

C-1は出版社最終PDF、C-2とlocal-controlは固定著者版の必要箇所を確認した。原論文での有効性と、本SOURCEに対する資格は別である。C-1/C-2の直接代入式はREPORT§3.3/4に示す。

## 2. PRの信号・保証と必要な追加性質

`a_ω=⟨g|U_ω|g⟩`、`μ=E_p[w_ω a_ω]` とする。符号をunitaryへ含めるcanonical raw meanではw=1である。sourceのBを外へ戻すcorrected表現と混同しない。

| 対応項目・SOURCE | 現在確認できること | 単純移植が適用不能となる理由 | 新Cで必要になる追加確認 |
|---|---|---|---|
| coherent mean / [rte.py][rte] `finite_rte_operator_moments` | finite polynomial、corrected/raw operator、B/attenuationを別出力 | `E abs(a)^2` は `abs(E a)^2` にvarianceを足した量。channel一致もcoherent meanを定めない | 測定が実際に返すmomentと同じtargetへの関係。追加干渉accessを使う場合はその実現・費用 |
| VPR interference / [C-1 App.B][C1] | 単一unitaryのamplitude間のinterference | 同じpathの確率平均は `E[a conjugate(b)]` を含み、`E[a] conjugate(E[b])` とは異なる | target/secondary/全interferenceを**同じ平均信号族**として復元できるか |
| sign/scalar / [builder][builder]、[partial][partial] | event・component sign、identity/constant phase、actual inverseを保持 | survival確率はpath scalarに不感。metadataだけで失われたphaseは測れない | 複素weight/符号を一度だけ配置し、測定range/momentとbiasへ反映する |
| finite K/r / [rte][rte] `finite_rte_distribution`, `compose_finite_rte_occurrences` | even Kのpaired degree K+1、signed time、個別B、順序 | meanは一般に非unitary。時間点ごとのschedule変更は単一unitary groupを保証しない | positive spectrum/支持・共役時間反転、C-2の解析性が成立する対象と誤差規約 |
| groundと絶対energy / [SOURCE保証][guarantee] `ground_certificate` | 元の固定保存行列モデルへのoffline照合とalias情報 | 保存ground vector/energy/gapは検証情報でありcandidateへの無料入力ではない。`f(0)=1` もenergy shiftを固定しない | 許されるanchor・projector・追加state/情報の取得。E₀/gapをcandidateだけへ渡さない |
| phase/radius / [finite signal][signal] `_state_result`、[保証][guarantee] `phase_upper`, `evaluate` | 元のcomplex signal単位でdisk/半径/phaseを分離 | 二次momentやlog確率、solverの残差を同じcomplex diskへ無条件変換できない | PR bias、復元誤差、zero/conditioning、統計誤差、normalizationを一貫した単位にする |
| shot / [accounting][account] `RPEHadamardSamplingPolicy`, `evaluate_rpe_round_candidate` | bounded IID X/Y、元のpartial-S2・基準半径1モデル | VPR/HIO・weighted outcome・複数点連動復元は同じ測定分布でない | 全点同時confidence、復元失敗を含む総alpha、整数shot。実測定rangeを使う |
| alias/decoder / [phase validation][phase] `reconstruct_rpe_phase` | 既存q列と対応phaseへのnearest-branch復元 | spectrumのshift/reflection不定性は通常のmod-2π aliasと同じでない | anchor/反転解決後に同じE₀へ接続できるか。既存decoderのoracle比較を推定入力にしない |
| native誤差/価格 / [compiler A][compiler]、[価格][price] | 元full joint/control/wrapper/profileに限定したcorrectness・価格接口 | 制御後jointの共通global phase同値を、後からcontrolするsystemのscalar同値へ移せない。新wrapperは未検証 | 新しいfull測定回路のphaseとnative誤差、T精度、routing。旧priceを新構成へそのまま当てない |

`C_use`は実行delta窓上の経験的包絡であり厳密上界ではない。既存の条件付き固定行列保証・local evidenceを、C-1/C-2のenergy成功定理、元H表現の無条件保証、外部再現へ格上げしない。[status][status]、[manifest][manifest]

## 3. 再利用する関数・test・pipeline

以下は**Sの本文・定義を静的に確認した入口**である。関数名とtest/runnerの所在は確認したが、起動していない。再利用区分は「そのまま＝元意味論を保つ部品」「変更必要＝新I/O・誤差/測定規約」「新規必要＝選択後に初めて必要となる欠損機能」「保証適用不可」を表す。今回、新規実装は0件である。

| 固定SOURCE / 関数 | 入力→出力・再利用区分 | test / runner・検証入口 | 制約・将来必要な差分 |
|---|---|---|---|
| [rte.py][rte] `finite_rte_distribution`, `_make_event`, `iter_sample_rte_events`, `event_unitary` | tail/時間/Kとordered component→finite law・signed event。**そのまま：law/phase定義** | `tests/test_rte.py`, `tests/test_rte_truncation_budget.py`, `tests/test_rte_integer_counts.py` | sampling/行列評価は今回は実行禁止。classical seed法は独立量子bitsの実装証明ではない |
| [rte.py][rte] `finite_rte_operator_moments`, `compose_finite_rte_occurrences`, `finite_rte_combined_attenuation` | 各occurrence H/time/r/K→corrected/raw mean・normalization。**そのまま：定義/reference** | `tests/test_finite_rte_signal_validation.py`、`scripts/run_finite_rte_signal_validation.py` | 一次meanのreferenceでありcontrol-free確率meanではない。新measurementとの照合が必要 |
| [df_rte_tail.py][tail] `extract_df_diagonal_tail`, `extraction_to_normalized_rte_tail`, `prepare_df_rte_event_inputs` | 固定DF/basis/identity policy→signed involution・component spec。**そのまま：固定入力の表現** | `tests/test_df_rte_tail.py`, `tests/test_df_rte_event_preparation.py` | scalar/identity抽出とbasis hashを保つ。古いDFを再生成しない。generic registryに特定physical basisの対称性を仮定しない |
| [df_rte_qiskit.py][builder] `QiskitDFRTEEventCircuitBuilder.build_event/build_sequence` | event/spec/basis/control→ordered phase保持回路。**そのまま：元builder**、**変更必要：新測定wrapper** | `tests/test_df_rte_qiskit_builder.py`：sign/full controlled operator、sequence/basis共有、central-only control | 新C用回路は作成しない。uncontrolled出力のglobal phaseを消すと、後段の干渉方式には使えない |
| [df_partial_s2.py][partial] `prepare_df_partial_s2`, `QiskitDFPartialS2CircuitBuilder.build_step`、[repeated][repeated] `make_df_partial_s2_repeated_request`, `QiskitDFPartialS2RepeatedCircuitBuilder.build` | 固定H/split/step/event列→D/R構成・反復・scalar metadata。**そのまま：順序/phase/共有** | `tests/test_df_partial_s2.py`, `tests/test_df_partial_s2_repeated.py`：actual product・control0/1・phaseを一度だけ適用・attenuation別合成 | 新Cの時間gridやdummy/ITE構成の保証は含まない。反復を一つのHのevolutionと同定しない |
| [track_c_h4_finite_bits.py][bits] `integer_cdf`, `draw_cdf`, `weight_interval`, `proposal_budget` | canonical probability/bit数/normalizer→CDF・weight区間・proposal disk。**そのまま：元lawの生成規約** | `tests/test_track_c_h4_pilots.py::test_finite_cdf_retains_rare_support_and_rounding_bound`、[保証runner][guarantee-run] / checker | SUPPORT消失拒否、全drawの誤差を保持。新weight・多系列・path共有時の独立性/誤差配分は変更必要 |
| [track_c_source_guarantee.py][guarantee] `basis`, `tail_task`, `ground_certificate`, `phase_upper`, `evaluate` | physical SOURCE/保存reference→generator・finite mean・alias・radius/phase。**そのまま：元保証の証拠**、**保証適用不可：新測定** | `scripts/run_track_c_guarantee_cn_price.py`, `scripts/check_track_c_guarantee_cn_price.py`、[元A報告][guarantee-report] | RZ/XXPlusYYのphysical SOURCE限定。保存state/E₀/gapはoffline参照。checkerも再実行しない。新CをcertifyするAPIではない |
| [finite_rte_signal_validation.py][signal] `_state_result`, `validate_finite_rte_signals` | exact/PF/corrected/raw operatorとstate→複素bias・半径・phase。**そのまま：将来のreference比較**、**変更必要：新観測量** | `tests/test_finite_rte_signal_validation.py`, `scripts/run_finite_rte_signal_validation.py` | 数値評価を伴うため今回は呼ばない。pure-state survival probabilityが複素信号と一致するtestではない |
| [rpe_hadamard_failure_validation.py][failure] `_axis_distribution`, `_explicit_fresh_iid_validation`、[phase validation][phase] `reconstruct_rpe_phase` | X/Y分布・fresh path・phase列→失敗率/branch検査。**変更必要：新測定分布** | `tests/test_rpe_hadamard_failure_validation.py`, `tests/test_rpe_four_round_phase_validation.py`、同名`run_*.py` | 二項bounded outcome・指定q/axisの保証。VPR/HIO復元・ITEのlog誤差へは保証適用不可 |
| [rpe_resource_accounting.py][account] `RPERoundCostProvider`, `evaluate_rpe_round_candidate`, `build_rpe_resource_summary` | round/β/α/実価格→shot×cost。**そのまま：provider分離・会計規約**、**変更必要：family/grid/成功条件** | `tests/test_rpe_resource_accounting.py`, `tests/test_rpe_resource_accounting_guards.py`：欠損/未検証proxy・empirical/rigorous区別 | partial-S2、Cδ²、基準半径1、fresh IIDの制約を保持。新Cのshotをこの関数で生成しない |
| [df_rpe_resource.py][cost] `DFLevel5RCompiledCostProvider.evaluate`、[track_c_h4_price.py][price] `trajectory_parts`, `wrapper`, `metrics` | 元request→full測定回路の価格。**そのまま：baselineの取得接口**、**新構成価格は新規必要** | `tests/test_df_rpe_resource.py`, `tests/test_df_partial_s2_repeated_cost.py`、`scripts/run_track_c_h4_pilots.py` | `trajectory_parts`/関連helperはcompileを呼ぶ経路を含む。samplingだけのsafe APIとは扱わず、今回は呼ばない。価格MC標本数は量子shot倍率ではない |
| [compiler A][compiler] `compile_reference`、[correctness][correctness] `synthetic_checks`, `h4_check`、[gates][gates] `controlled_gate`, `signal_gate`, `compiled_wrapper` | registered IR/full wrapper→native・operator/信号照合。**そのまま：元correctness規約**、**変更必要：新register/readout** | `scripts/run_track_c_compiler_a_correctness.py`、[元correctness報告][correctness-report] | already-controlled joint全体の共通global phaseだけを許す。native途中のsector射影、control後付け、新T価格への無条件転用は不可 |
| **欠損機能：新Cの観測モデル・復元・入力準備** | 選択されたVPR/HIO/ITE/local方式→同E₀の推定。**新規必要となり得るが今回は未実装** | 対応する新test/runnerは作らない | まずaccess/mean/anchor/安定性を判断する。既知solverやstate回路の実装を先行させない |

既存pipelineは「固定入力/表現→ordered finite trajectory→phase保持wrapper→native correctness/価格→signal disk/radius→整数shot/alias→全費用」と辿れる。新Cで欠けるのは主にmeasurement accessと復元の保証・費用であり、既存pipelineを再実行することでは埋まらない。旧native priceやstrict合成recordは元SOURCE・角度・precision keyの資格範囲だけで再利用候補となる。

## 4. 最小確認案と反証・STOP条件

**以下は将来の判断材料であり、実行計画・契約・許可を新設する表ではない。今回はいずれの人工simulation/測定/solver/科学計算も行っていない。** REPORTに書いた恒等式・groundへの直接代入で判別できる点を先に使う。新しい物理条件・数値thresholdは固定しない。

| 段階・最小の問い | 最小の確認対象（必要なら別許可後） | 判別できること | 反証・STOP条件／次に必要なこと |
|---|---|---|---|
| **M0 accessとenergy基準** | 供給state以外に、何のstate/readout/norm/referenceをどの費用で取得するか、既知値と未知値を列挙 | E₀・gap・古典vector・無料projector・制御state oracleの先取りを識別 | 許されるanchorがなくenergy shiftを識別できない案は、その主張の範囲でSTOP。ground供給だけで追加oracleを無料化する案もSTOP |
| **M1 固定groundの退化** | REPORT§3.3/4の式。必要な場合のみ、最小eigenstateの人工例を別途選ぶ | VPRの比例signal、2DのH方向factor、ITEのnormalized-state/norm分離 | 同じground bra＋絶対値だけの直接構成は適用不能。PR誤差の作る変化をE₀情報とする案は保証が無ければSTOP |
| **M2 一次/二次moment** | 記号的に `U` と `-U`、variance恒等式を確認。必要なら小supportで別許可の照合 | 同じtrajectory確率から異なるcoherent meanが生じ得る。interferenceのcross-momentも区別 | probability平均をcoherent meanの絶対値へ置換する案はSTOP。coherent accessや別targetなら費用と同E₀接続を示す |
| **M3 finite-PR意味論** | 適合する既存のordered event/finite law/testを使い、sign・identity・負時間・B・noncommuting順序を最小supportで確認する案 | 理想unitary、raw/corrected mean、channel、測定momentの相違を検出 | support/phase/weight/normalization不明、actual adjoint不一致、元targetへ戻らない場合STOP。solverの性能試験へ先に進まない |
| **M4 文献の条件** | VPR支持/独立性、2D正値性/非退化、ITE local性/zero-free/normalizerを対象候補で確認 | 文献のどの保証を使え、どこから追加研究になるか | 固定ground退化、zero/conditioning、dummy/ITE取得不能を無視した適用主張はSTOP。新保証を今回作らない |
| **M5 noiseと成功条件** | 将来採択された最小人工signalで、全点confidence、復元誤差、PR biasとalias/energy接続を区別 | 小残差と正しいphase/energy成功を区別。biasと統計の二重計上や欠落を検出 | 正半径/残余誤差予算・anchor・成功率が示せない場合はenergy保証の主張をSTOP。既存RPEのNをコピーしない |
| **M6 full回路と費用** | 上記が成立した後だけ、強Baseline 1/2と同精度のfull wrapperを比較する案 | control削減が追加state・測定・shot増・古典費を上回る因果経路を確認 | 追加qubit/normalizer/投影/postselection/価格欠損を0にする比較はSTOP。短回路CXやquery数だけの改善は総資源優越にしない |
| **M7 実H4等** | access/mean/誤差と人工例が成立し、実DF basis・分割・時間窓の妥当性やnative費用が判断を変える場合のみ | 物理groundのoverlap・locality、実SOURCEと全時間grid費用の適合 | 元H/geometry/basis/rank/L_D/delta窓/精度/成功率/比較scopeをユーザーが選び、必要な実行を別途許可するまで着手しない |

ここでSTOPは、記載した接続・claimをそのまま進められない条件である。新C全体の科学的NO-GOの宣言ではない。旧Track C/C2/新B等のSTOPはこの表とは独立に維持する。

## 5. 強baseline・総資源の照合項目

| 比較軸 | Baseline 1：PR＋RPE | Baseline 2：新A | Candidate：新C |
|---|---|---|---|
| evolutionとphase | diagonal/中央だけcontrol、basis共有、scalar保持、適用可能なcompiler最適化 | reference干渉、actual inverse、pathwise phase補正 | 文献が測るmoment、必要なphase基準、復元条件を明示 |
| 追加入力・測定 | 元state＋ancillaのX/Y | reference state/register、CSWAP/reset/routing | secondary/superposition、projector/inverse、dummy/ITE、normalizer/tomography |
| shot/時間点 | 元round/両軸、fresh IIDと正半径 | 同じ複素targetを実現した場合の実radius/誤差から評価。pilot性能は未取得 | 全時間grid/measurement family、prefix・norm取得・受理失敗も含め、同時成功条件から別評価 |
| 誤差 | PF、finite-RTE、finite-bit、compiler/synthesisと統計を区別 | 上記＋reference/adjoint/CSWAPの実native誤差 | 上記＋復元conditioning、log差分/積分・zero・支持/window/anchor |
| 会計 | 全shot×実wrapper資源。既存価格の資格範囲内 | 追加lane/qubit/control/readout全体 | 全shot×追加処理込み資源。RZ/CX/T/depthとqubit peakを別記 |
| 古典/情報費 | 既存sampling・前処理/価格取得も別記 | reference phase・生成/補正・native価格 | reconstruction time/memory、支持探索/restart、norm・unitary・state情報の取得 |
| 未資格を残すもの | 長q価格等を元statusのまま扱う | 新A pilot結果・性能を推測しない | 文献depth改善やcontrol-freeだけから利益を推測しない |

## 6. 引渡しとmandatory STOP

GPT／ユーザーへ渡す問いは、①許されるmeasurement/accessとenergy anchor、②PRの一次meanへの対応、③固定groundでの情報と追加費用、④文献保証の対象と新しい追加性質、⑤強baselineに対して利益が残る機構である。応用実装と独立した追加研究の境界を判断する。新規性や改善率は未判定である。

Codexは2文書のcommit/pushとremote SHA・内容一致の照合まで実施し、**公開照合完了時点でmandatory STOP**する。その後に新しいsolver、builder、state準備、optimizer、科学実行・GPT科学レビューを自動開始しない。新Aや旧STOPを変更しない。次の担当は**GPT／ユーザー**。

[C1]: https://journals.aps.org/prxquantum/pdf/10.1103/7qcr-znl2
[Yang]: https://arxiv.org/html/2308.10796v2
[local]: https://arxiv.org/html/2506.18765v2
[PR]: https://arxiv.org/pdf/2503.05647v2
[SE]: https://arxiv.org/pdf/2604.14921v2
[AB]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/docs/research/pr_reference_signal_static_investigation_20261106/REPORT.md
[status]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/VALIDATION_STATUS.md
[manifest]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/artifacts/validation_manifest.json
[rte]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trotterlib/rte.py
[tail]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trotterlib/df_rte_tail.py
[builder]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trotterlib/df_rte_qiskit.py
[partial]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trotterlib/df_partial_s2.py
[repeated]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trotterlib/df_partial_s2_repeated.py
[bits]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trottertracks/track_c_h4_finite_bits.py
[guarantee]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trottertracks/track_c_source_guarantee.py
[guarantee-run]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/scripts/run_track_c_guarantee_cn_price.py
[guarantee-report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/docs/research/track_c_guarantee_cn_price_20261106/A_REPORT.md
[signal]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trotterlib/finite_rte_signal_validation.py
[failure]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trotterlib/rpe_hadamard_failure_validation.py
[phase]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trotterlib/rpe_four_round_phase_validation.py
[account]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trotterlib/rpe_resource_accounting.py
[cost]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trotterlib/df_rpe_resource.py
[price]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trottertracks/track_c_h4_price.py
[compiler]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trottertracks/track_c_compiler_a.py
[correctness]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trottertracks/track_c_compiler_a_validation.py
[correctness-report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/docs/research/track_c_compiler_a_correctness_20261106/REPORT.md
[gates]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/src/trottertracks/pr_dictionary_validation/gates.py
