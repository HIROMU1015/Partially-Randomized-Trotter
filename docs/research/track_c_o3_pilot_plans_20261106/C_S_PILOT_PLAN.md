# C-S pilot案：合成精度と同位置residual補完

2026-11-06 JST。**DRAFT / 未実行・未採択。** [共通契約案](SHARED_PILOT_CONTRACT_DRAFT.md)を適用する。新sampling law、SP toy延長、Parent旧実験再開は計画しない。

## 1. 問い・入力・固定する層

既知の粗い角度合成＋randomized residualが、normalization/weight/shot増加を支払っても、同じ元H energy精度・成功条件で通常合成を下回るかを問う。

H4共通入力候補、LD3、canonical partial-S2、h=δ0=.02、既存original-tailのK2/round別r、18roundを固定する案。processorはなし。PF係数/順序・H_D/H_Rの科学的分割・RPE時刻はarm間で変更しない。元H/DF/state接続を満たさないまま主要結果を出さない。

**主比較はprofile T候補**：strictなRZ合成、Clifford+T/CXとdepthを個別に報告する。旧Qiskit `rz,sx,x,cx` のRZ価格へT価格を接ぎ足さない。profile QのRZ/CXのみで実施する場合は、T削減の主張を外して、その限定目的をユーザーが選ぶ。

## 2. target operatorと有限arm表

| arm案 | 変更するもの | 役割 |
|---|---|---|
| S0 通常合成 | 各eligible RZを選んだ通常precision policyで合成 | 同じenergy targetに適合する強baseline。exact Clifford/T fast path・fusionも共通 |
| S1 一様residual補完 | 同じeligible RZ class全体に既知の粗角＋同位置残差処理を適用 | 「Dだけを選ぶ」利点と区別する既知法の一様適用対照 |
| S2 D側residual補完 | Dのdiagonal phase rotation classだけS1と同じ処理。R側はS0と同じ | 配置による単独効果。basis/Givensやprocessorを同時最適化しない |

SPの多数mask、全角度notch sweep、両側の全組合せを繰り返さない。S2で選ぶclassは回路roleから事前定義し、E₀/exact signalや価格を見て個別gateを後から選ばない。S1/S2は同じcoarse catalogue・残差law・precision policyを使う。

### 分割を変えないresidual接続

各primitiveの数学的targetを `R_A(θ)=exp(-i θ A/2)` とする。同じgenerator・同じ回路位置で、既知の分解 `R_A(θ)=R_A(θ̃) R_A(θ-θ̃)` を使い、残差rotationを既存canonical operator-RTE等で補完する案。合成された粗角operatorと数学的粗角rotationの差も別途支払う。

有限RTEなら補正meanは残差rotationの**有限近似**であり、exact一致と記さない。ideal PF列→coarse/residual finite mean→native実現meanという比較を行う。θ/2・符号・scalar/controlの規約を共通にする。

元PR論文の「D係数を丸め中央Rへ加える」方法は近い既知対照だが、central Rへ移動するとPFのD/R分割誤差も変わる。本pilotの**主比較へ無条件に入れない**。必要なら別の分割変更ablationとして別範囲を定める。同位置補完のoperator接口が実装できない場合、中央移転に黙って差し替えず未実装/未資格として報告する。

## 3. mean・normalization・weight・誤差

1. original finite-RTE pathと追加residual eventのordered productを定義する。実proposalとcoefficient/sign/phaseから`Q=E_p[W V]`を追跡し、元H groundの`Zcorr=⟨E₀|Q|E₀⟩`へ戻す。確率を二度掛けない。
2. original tailとresidualのnormalizer/weightは実装規約に従って合成する。独立性がある場合だけ積のmoment式を使い、同じgate keyを共有することを独立drawの根拠にしない。
3. 未補完の通常合成bias、coarse sequenceのstrict operator error、有限residual truncation、coefficient/finite-bit誤差、PF/元H近似を別記する。誤差が未知なら0補完しない。
4. corrected estimatorのrange Lまたは必要なmomentと有限confidence保証を記録し、共通契約案の十分Nを使う。Bによるraw減衰とcorrected rangeを同時に掛けない。二次moment改善だけでRPE成功を主張しない。
5. systemのquantum channel一致はHadamardのcoherent mean一致を保証しない。TE-PAIのsystem-channel式やSPのchannel価格をそのままoperator補完へ代入しない。full ancilla+systemの測定等価性を使う別方式なら、その保証対象を明示する。

既知PAI/TETRISは[固定文献調査][survey]のoperator/channel対応を引継ぐ。初期実装はcanonical operator-RTEによる同位置補完に絞る案とし、PAI/TETRIS全pipelineを同時導入しない。必要な場合の追加対照は、同じcoherent target/情報/費用境界が成立するものに限る。

## 4. 固定SOURCE・最小実装差分

| 固定SOURCE・関数 | 再利用範囲 | 足りない箇所・最小差分案 |
|---|---|---|
| `4ccc1832f63cd323b23fc05623f7e0b58cf34f37`：`parent_allocation.py`の`event_statistics/error_accounting/native_plan/adjoint_plan` | 与えたfinite tableのweight/range/moment、外部誤差、実順序/相対phase/actual adjoint | 新しいglobal law最適化は不要。H4から得るsame-position residualの記述と、元tailとの順序接続を渡す |
| `546bca7ed2cece59b714b986a264d7b0b0871a01`：`parent_allocation_native.py`、`rte_reallocation/native.py` | strict RZ key、signed inverse、control lowering、exact symbolic inverse cancellation | saved keyの角度/precision/toolが一致する範囲だけ再利用。分子の新角度は新合成callとして扱う |
| 同版：`synthesis_placement/economics.py` | exact-angle fast path、`synthesize`、区間guardの既存実装 | `pai`はchannel interpolation。主coherent residual式には流用しない。`error_guard`のprojective許容をcontrolled system gateへ持ち込まず、strict相対phase guardを接続 |
| 同版：`synthesis_placement/wrapper_adapter.py`、`wrapper_accounting.py` | 保存weight/coefficient/誤差の区分と反証。新mean/priceへの入力形式の参考 | `StoredLibrary`は固定8角度/23sequence等に拘束。H4を通すために元inventoryを改変せず、新しい限定native-angle providerのみ作る |
| `f29eafbf4412193e8e8660d9f051889c2cba684b`：`rte.py`、DF builder、RPE provider | canonical分布、signed heterogeneous occurrence、basis/control、軸別full費用 | original PF列を維持して追加residual occurrenceをその位置へ接続。weighted実半径の一般会計差分は共通部だけ |

既存testsは`test_parent_allocation.py`のmean/実proposal/phase drop/adjoint、`test_parent_allocation_native.py`、`test_sp1_wrapper_adapter.py`、`test_sp1_wrapper_accounting.py`が候補。toy専用の保存数/価格をH4の正解にせず、新接口の人工controlだけ必要に応じて追加する計画。今回testも実装も行っていない。

既存scriptは同546bca版の`scripts/tracks/algorithm_codesign/run_sp05_synthesis_economics.py`、`scripts/tracks/algorithm_codesign/run_sp1_wrapper_pilot.py`、`scripts/tracks/algorithm_codesign/run_parent_allocation_synthesis_key.py`を参照する。旧toy inventory/one-shot条件を持つためH4で起動しない。新しい有限angle/provider差分へ必要な関数だけを接続し、旧scriptやmarkerは変更しない。

## 5. precisionとcorrectnessを確定する順序

precisionは旧1e-6/1e-7のどちらかを根拠なく固定しない。ユーザーが選ぶ全interrogation synthesis/error budgetと同じfinite policyを全armに与え、旧catalogueで対応する値を再利用できるか判定する。長qで適合しなければ新精度取得が必要と報告する。新精度を価格だけに使うか、bias/shotにもcreditするかを混ぜない。

correctnessは以下を計画する。

- 正/負角、exact端点、identity、同generatorのcoarse×residual、controlled相対phaseを独立operator参照で照合。
- original PFの同じ位置と順序を保つことをliteral列で確認。candidateだけbasis共有/融合を追加しない。
- ideal、finite、nativeを区別し、full joint operatorまたは明示されたfull measured wrapperの期待値を照合。projective/global-phase fitでcontrol誤差を隠さない。
- 同じ元H groundへの全18round complex signal、phase/radius、weight range/moment、十分N/branchの適合を確認。raw/correctedのどちらで会計したか保存する。

少数の回転レベル・短qだけで不成立を識別できれば、長回路の合成・価格取得を増やさない。一方、短q成功だけでlong energy成功とはしない。

## 6. 価格取得・全RPE・古典費

同じH4のfull interrogationを、通常合成とresidual armの両方で同じnative profileにlower/compileする。T/T†、CX、他Clifford、個別depthを保存する。加法primitive Tは予備診断であり主価格を代用しない。controlled primitiveをlowerした後のglobal phase処理と、system gateをcontrolする前のphaseを区別する。

q1/2/8は共有計画の短い較正/holdout案。original tailのrare orderとresidual eventのrare branchを条件付き層別にし、実確率を掛けて期待価格へ戻す。全supportを列挙できない場合の未取得strata/tail価格は上界またはUNKNOWNのまま残す。元SP/D1/D3の保存価格は新角度・新whole-wrapperの母平均ではない。

全roundのNはS0/S1/S2それぞれの実mean/error/rangeから同targetで評価する。同じ配分/precision最適化機会を与える。`G_T`、`G_CX`、`G_depth`は別の合計。主指標に利益があっても他資源の増加を隠さない。長q価格が外挿だけなら条件付きGに限定する。

古典費には角度inventory/coarse catalogueの取得、normalizer/rangeの計算、strict合成とguard、per-shotのevent生成、lookup/cacheアクセスを記録する。古典費をgateへ無根拠換算しない。価格を得てから有利なgateだけを選ぶ選択器なら、その取得費と事後性を明示する。

## 7. 科学call・保存物・判断

| call群 | 必要範囲案 | 残すもの |
|---|---|---|
| 共通入力 | H/state/DF primitiveは共有して再生成を避ける | 元H/DF/stateの対応、sector、SOURCE/input |
| residue/mean | S0/S1/S2の登録class、coarse policy各一つ。全18roundへのmean/error/range接続 | θ/θ̃/残差、coefficient/proposal、順序、B/weight/range/moment、各mean/誤差 |
| 合成 | unique positive angle/precision/tool keyごと。負角はactual adjoint。旧key一致だけ再利用 | full sequence、strict guard、T/CX/depth、tool/seed/precision、失敗/未資格 |
| full価格 | 登録したq1/2/8、両軸、全lawを覆う価格標本/strata | paired/covariance、未知rare平均、model幅、compile context |
| 会計 | 同じenergy成功条件で3armのNと全G | 十分N/β/α、classical取得/生成費、長qscope、比較不確かさ |

n/precision/残差r/Kは実行前に一つの有限policyへ確定する。旧SPのmask全探索、全notch幅、Parentの新law探索、旧D1/D3の再実行、RNG replay/仮想量子shot大量samplingを初期必須にしない。

**継続材料**：strict coherent target/元H energy接続を満たし、normalization・必要N・rare価格と不確かさを戻しても、強いS0/S1に対する全G改善が残る。
**縮小/STOP材料**：単価だけ低い、B/range/shot増でGが増える、system-channel一致しかない、unknownを0補完しないと比較できない、元H/長q/数値誤差の資格がない。新sampling lawを探して救済しない。

## 8. 独立性・未決条件

C-NのPF winnerを待たずcanonical S2を固定でき、C-Pのprocessorは使わない。共通H/state/phase保持nativeと一般RPE会計は共有可能。C-N/C-Pと異なるprofile Tを選んだ場合、3pilot間の費用順位を作らず、それぞれの固定baseline内の効果を比較する。

**具体化済み**：同位置residualで分割誤差を混入させない定義、3arm、coherent meanとweighted十分N、strict-phase合成、full価格/rare/classical費の接続。
**未決**：共通元H/target、profile T採否・主資源、eligible rotation class/coarse catalogue、全round合成予算とprecision/residual r/K policy、価格精度。ユーザーの選択と別の実装・科学実行許可後にCodexが最小接続を作る。今回mandatory STOPは文書公開照合まで。

[survey]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106
