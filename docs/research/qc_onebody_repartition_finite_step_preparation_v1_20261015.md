# 一体項再配分PF：有限step最小pilotの実装・実行前契約

2026-10-15 JST。**実装・人工検査・identity-only preflightまで。分子の新しい数値結果は取得していない。** [受領済みGPTレビュー原文](../reviews/quantum_chemistry_pf/qc_onebody_repartition_final_audit_scientific_review_2026-10-14.md)に従い、応答生成の専有的高速化を主張する経路を終了したまま、生成されたPFの有限stepでの有用性を一回だけ判別できる準備を行う。研究の正式採択・性能優位を認定しない。

基準は `9e288089ca22e2729976ae75d096c4df00e49a7d`。独立branchは `qc-onebody-repartition-finite-step-preparation-v1-20261015`。旧Phase A監査・STOPを編集せず、新しいユーザー指示と完了済み科学レビューによる別scopeを追加した。[固定契約JSON](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/contract.json)が全数値条件・意味論の正本である。

## 1. 一件の既存入力と供給状態

| 条件 | 固定内容 |
| --- | --- |
| Hamiltonian/model | 保存済みHermitian分子DF Hamiltonian、linear H4 |
| geometry/basis | H間距離1.00 Å、STO-3G、charge0、multiplicity1 |
| DF | 保存rank12、generation順0…11を変更しない。保存truncation metadata0.0を新しい原積分精度証明としない |
| sector | 8 spin/4 spatial、Nα=Nβ=2、36次元。interleaved α/β、mode0がMSBのOpenFermion basis |
| PF/split | 全12 density fragmentを決定論的に使用。PRのL_D探索ではない。一体Aが最外側、最初のdensity Bが直内側 |
| 元pack | `artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz` |
| pack SHA256 | `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a` |
| Hamiltonian identity | `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424` |
| supplied state | 同packのsector_state_vector。state hash `31e63b0104126c85136ee173f1dce7642aee2d272924e70e8b120ac340ab45bd` |
| blind性 | 既存development inputである。held-out/新分子移送ではない |

[入力identityと保存metadata](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/input_identity.json)。今回読んだのはpackのbytes、NPY headers、metadata文字列だけで、係数・state arrayのdeserialize、RDM、期待値、固有値、PF、compileを実施していない。保存metadataのhermitizedや状態hashは過去の記録であり、今回新しく数値保証したとはしない。

将来の認可済み実行ではcached stateをそのまま供給する。新しいground-state準備/最適化は0件で、truth専用の36次元H diagonalizationを1回だけ行い、cached stateのresidual、基底branchとのoverlap、gapを検査する。不一致・gap不確定ならretryせず停止する。

この保存packは旧PR S0で旧pilotとのbyte identity不一致を起こした `de7a` 側である。今回そのpackそのものを独立入力として明示固定し、旧 `d8b` Hamiltonianとの一致を要求・主張しない。旧PRの不一致STOP、held-out未開封、Track A/Bのsource・認可は変更しない。

## 2. 限定constructorと情報隔離

保存表現は

\[
H=cI+d\Gamma(a)+\sum_{\ell=0}^{11}\lambda_\ell\,d\Gamma(g_\ell)^2.
\]

ここで元packのone-bodyには既存normal-order補正が含まれる。平方fragmentへ新たな1/2を付けたり、補正をもう一度加えたりしない。scalarも残す。

最初のdensity frameだけを、spatial g0の昇順Hermitian diagonalizationとspin-paired embeddingで固定する。Xはそのframeのdiagonal、spin-paired spatial値の平均0とする。A→A+X、B→B−X、tail/frame順/sectorは不変。DF因子回転、非隣接更新、複数fragment更新はない。

前回支持された1-RDM応答をそのまま使う。数学的監査や旧27検査を再実施しない。

\[
d(x)=d_0+g^Tx+x^TQx.
\]

d0は元S2 leading operatorの独立期待値取得、γは供給された状態の1-RDM取得として記録する。取得1回は一つの古典contraction操作の呼称であり、量子1shotという意味ではない。将来pilotは理想的な古典情報による機構層で、量子測定取得費用はUNKNOWNとして別会計に残す。

方向はspatial projected gradient（g_even+g_oddの平均を除き、infinity normで正規化）1本のみ。|t|≤0.25 max(10^-12,max|a_pq|)。候補はzero、両endpoint、二次応答の実root、vertexのうち区間内のもの、最大6件。|leading coefficient|を最小化し、10^-12以内のtieは|t|最小、次に小さいsigned tを選ぶ。gradientが小さい場合はzero fallback。QのPSDや全方向のconvexityを仮定しない。iterative optimizerはない。

既知再配分対照は最初のdensityのnormal-order linear term λ0η²を同じmean-zero/infinity-bound内で一体側へ移す手順である。FFFのvariance最適化を実装したとは称さない。等情報B*も同じ応答と同じPFを生成できることを保持し、generatorの独占的速度差を再benchmarkしない。

constructorに渡るのはa、γ、d0、係数、合法boundsのみ。E0、gap、finite-step bias/overlapは渡さない。selector_sealを保存・hash固定してからtruth-H solverとPF評価を開始し、seal後の再選択はない。

## 3. PF・強対照と固定評価grid

| 構成/方法 | 実行上の役割 |
| --- | --- |
| original、response_selected、known_linear_extraction | 固定3構成。first density以外を動かさない |
| S2、S4 | 全3構成×2次数×6h=36 primitive cells |
| scalar-corrected S2 | 全3構成でE_PF(h)−h²dを同じ情報accessで評価、追加18records |
| two-point Richardson | 各構成の隣接h/h/2で(4E_fine−E_coarse)/3、追加15records |
| original subdivision | original S2に同じ6点の細分化を許し、目標精度を満たす最小費用を使う |

合計69 method records、target2×price2で276 cost rows。S4はp,p,1−4p,p,p、p=1/(4−4^(1/3))のsigned S2合成であり、S2応答でS4誤差を最適化したとはしない。全方式に同じsector scalar extraction、Gaussian basis共有、隣接同fragmentのfusionを許す。

hは `h_s=1/(8 Λmax 2^s), s=0…5`。Λmaxはsealed構成に共通の、one-body spectrumとdensityのtriangle boundでsourceから定義する。真のenergy/gap/PF errorからhを調整しない。登録formulaによる局所6点窓であり、全許容hやoptimal precisionの探索ではない。最大hがすでにeligibleな場合にも、全域最適性や他の刻みの不能性を結論しない。境界を理由に追加hを実行しない。

truth側はunitary complex Schurとprincipal phasesを使う。近い固有値をcluster化し、供給exact-input stateに対する重みが一意に最大のclusterを採る。ambiguity、unitarity/eigenpair defect、wrap条件、overlap不足は未判定または不適格。独立checkerは保存された固有対を検証し、新しいPF eigensolverを呼ばない。

targetは10^-4、10^-5 Ha。予算をdeterministic/statistical/synthesis/informationに1/2、1/4、1/8、1/8で分ける。deterministic biasに数値allowanceを加えた値がその予算より厳密に小さく、overlapから数値marginを除いた値が0.99以上の場合だけeligibleとする。double precisionと独立残差検査による登録marginであり、外向き区間計算による厳密certificateではない。

比較はresponse S2/rawまたはcorrectedの最小eligible費用対、original S2/corrected、全S4、known linear S2/corrected、全Richardsonの最小eligible費用。h選択は全方法に等しく与える**固定gridの事後oracle**で、truthなしの運用可能なprecision selectorを検証したことにはならない。

## 4. controlled資源・情報費用と成功範囲

資源部はcompileを行わない。spatial Givensをspin二つへ埋め込み、fermionic parity linksを記録する。固定Nによるscalarの抽出、scalar-only fragmentのbasis gate省略、隣接relative-basis共有、repeated-query境界の厳密fusionを全対照に適用する。densityのlinear/pair phase、known controlled scalarを数える。

価格はcontrolled Givens/linear phase/pair phaseについてP1=(4,1,2)、P2=(8,1,4)の**登録価格仮説**。Givens価格にそのparity構造を含める。このvectorは実CX/T/depthのcompile値でも、最適回路の下界でもない。`controlled_Givens` fieldはcontrolled wrapper内のGaussian primitiveの価格単位で、全basis gateへcontrolを付けることを義務化するものではない。conjugationのbasisをuncontrolledにしてcontrol-0側をuncomputeする合法な実装も対照から除外しない。その実単価や小角のT合成・精度依存のgate elisionはbenchmarkしない。

coherent query数は `ceil(π/(h ε_stat))` のHeisenberg型会計仮説。Richardsonは二つの実験の費用を加え、|weight|和5/3をprecision allocationへ戻す。overlapに由来する入力ground branch抽選のrepeat数も別に戻す。QPE/RPEの具体的統計手順・全confidence overheadを構成した定理ではない。log/depth/ancilla/測定/state情報をゼロとした最終total-cost claimへ拡張しない。

Gaussian/pair/phaseのoccurrence数とuniform synthesis-angle予算を出すが、T合成やbackend費用はUNKNOWN。元raw S2/S4に不要なγ/d0取得を強制しない。corrected originalはd0だけ、known rawは係数だけ、responseはγ+d0、corrected knownは自身のleading contractionを取得できる。これらを別の取得費用ラベルで出す。

全2target・2priceで強対照比が0.90を数値marginの外で下回る場合に限り `CONDITIONAL_COMPONENT_GAIN_IDEAL_INFORMATION` とする。10%は今回新たに結果前に固定したmaterial-effect gateであり、C1の5%gateを流用したものではない。一部差・小効果・missing/境界は別分類。主要leading改善だけではこの判定を通さない。

positiveの差額は同じ価格単位内の追加情報費用break-even予算として出せる。取得費用の単価が未知なら利益を確定しない。否定結果も、登録grid/constructor/価格仮説に限定し、量子化学PF一般の不能性を主張しない。新規性・全ground-energy資源優位・研究採択は結果の有無によらず未決である。

## 5. 実装・独立checker・実行上限

libraryは [pilot kernels](../../src/trottertracks/ground_pf_repartition/qc_onebody_repartition_pilot_v1.py)、[pipeline](../../src/trottertracks/ground_pf_repartition/qc_onebody_repartition_pipeline_v1.py)、[資源会計](../../src/trottertracks/ground_pf_repartition/qc_onebody_repartition_resources_v1.py)、[独立audit kernels](../../src/trottertracks/ground_pf_repartition/qc_onebody_repartition_checker_v1.py)、[認可gate](../../src/trottertracks/ground_pf_repartition/qc_onebody_repartition_execution_v1.py)。重い旧trotterlib initializerを実行せず、Track A/Bのsourceを変更しない。

- [runner](../../scripts/run_qc_onebody_repartition_pilot_v1.py)：stdlib identity-only preflight、別明示認可、固定source/environment/input照合、一回marker消費後の数値import。
- [independent checker](../../scripts/verify_qc_onebody_repartition_pilot_v1.py)：独立なJW tensor ladder、Padé expm再合成、direct one-body commutator収縮、保存eigenpair/branch/補正/外挿/controlled primitiveと価格の照合。
- [準備checker](../../scripts/check_qc_onebody_repartition_preparation_v1.py)：source/input/review/protection/manifestのhash/byte/header照合だけ。
- [人工tests](../../tests/test_qc_onebody_repartition_pilot_v1.py)：登録外4-mode手書きfixture、37tests PASS。分子性能結果やcurrent CIではない。

| 項目 | 将来一回の上限 |
| --- | --- |
| input/state | 既存pack load1、新分子/新ground-state preparation0 |
| candidate | 方向1、最大6 leading候補、反復optimizer0 |
| spectral calls | spatial eigh15、truth-H eigh1、PF Schur36 |
| acquisition/contraction | γ取得1、direct leading3。checker独立γ収縮1/direct leading3 |
| PF/audit | producer seed36、checker再合成36、checker expm≤1024、新audit eigensolver0 |
| workers/threads | 1/1、BLAS/OpenMP/Numbaすべて1 |
| RSS/output/log | 512 MiB/32 MiB/2 MiB |
| compile/T synthesis | 0/0 |
| attempt/retry | 1/0。auditも一回、marker後のtechnical failureを救済しない |
| wall/CPU time cap | なし。tool polling timeoutとは別 |

[performance plan](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/performance_plan.json)。CPU32 logical/RAM約64GBをread-onlyで確認し、GPU driverは使用不可だった。36次元一件のsector matrix、CAR tensor、batched exterior frame、共通density cache、spectral seed、closed-form repeated cost、独立Padé cacheを使う。小さい単一入力なのでIPCよりcache共有を優先してserialにする。実速度倍率は測っていない。

## 6. source固定・認可境界・STOP

実行source commitは `19d5bd74db020564ea9a4b1882e102911e983cdb`。[source manifest](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/source_manifest.json)は21filesを固定する。manifest SHA256は `8b4a3ed3034c0931dd33cff60cb27cdce974f8ca5dc316bab71d2d0ed29d8fe1`、contract SHA256は `c3e9d2028ca74a65d7f781d48a02033755d92919a24be3415097fa573b97161d`。

[preflight](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/preflight.json)、[人工37tests](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/synthetic_tests.json)、[未発行認可target](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/authorization_target.json)、[実行引継ぎ](qc_onebody_repartition_finite_step_execution_handoff_v1_20261015.md)。targetはapproved=falseであり、そのままexecuteへ渡しても拒否される。消費済み940-call認可やC1事後診断の認可を使えない。

旧417files、旧150manifest records/non-result fields、global description欠落FAIL、元worktree137paths、受領原文34,108bytesを保持する。N1STOP/N2HOLD/N3CLOSED/THRIFT現template終了、C1–C4、Track A/B、第二候補を変更・再実行しない。global validator全体を通過したとはしない。

GitHub commit/push・再取得byte/SHA照合後Mandatory STOP。**次はこの固定one-shotの別明示実行承認であり、同じ科学レビューの再実施ではない。** 実行認可後のデータ取得・独立監査・公開も一回で停止し、その重要科学結果の解釈はGPTの別開始承認後に戻す。

## 公開確認（2026-10-15追記）

payload `fd44510328d8ab3068634a9906d40de16ca3ec43` を独立cacheへGitHub SSH depth3で再取得し、47変更blobs/1,859,892bytesをcommit/worktreeとbyte/SHA照合した。[receipt](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/publication/receipt.json)。source21files/旧417files/6固定参照/受領原文34,108bytes/150旧manifest records/non-result fields/global FAIL/root137pathsも一致。外部科学再現・current CIではない。

実行前準備完了・Mandatory STOP。科学実行認可は未発行false、attempt未消費、実入力の新eig/RDM/expectation/PF/compile0。source/contract/inputは変更せず、次は固定one-shotの別明示実行承認。
