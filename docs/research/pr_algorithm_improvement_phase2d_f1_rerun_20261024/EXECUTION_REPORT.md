# Phase 2D-F1：修正版による登録済み科学検証結果

**F1_REGISTERED_CORRECTNESS_PASS_UNQUALIFIED_PRICE_FALLBACK_MANDATORY_STOP**。
登録された正しさ検証は完了し、数値的不一致・科学的FAIL・実行中の技術的エラーは0件。
価格・intervalの未資格を認証済み最適選択やshot数のPASSと扱わない。

## 実行範囲と固定条件

Repository：HIROMU1015/Partially-Randomized-Trotter。
結果branch：`research/pr-algorithm-improvement-phase2d-f1-rerun-20261024`。
修正済み科学SOURCE：`8192530b17d6bc4af195c734ff43a6e78c506942`。
[metadata修正](../pr_algorithm_improvement_phase2d_f1_metadata_fix_20261024.md)以外の科学処理は既存F1-PRE／Iを再利用した。
SOURCE closure 97file（既存96file＋管理回帰test）を実行前・処理境界・終了時に照合した。

対象はP0固定の4 mode・2粒子・occupied(0,1)・L_D=1・残差density-square 2fragmentの合成fixture。
分子geometry・AO basis・分子DF rankは非該当。
T=1/5、q=2、delta=1/10の一点、r=2、h=1/20、K=2、full16／controlled32。
JW little-endian、ancilla-last、faithful_identity_in_tail、scalar・符号・phase・products→child順を維持した。
catalogは保存4label、scalar=-1/31、T=h=1/20、q=r=1、K=2。
B保存入力13fileを同一hashでコピーして使用し、Hamiltonian・辞書・lawを再生成していない。

P0許容差を維持した。rtol=0、phase fittingなし、NaN/InfはFAIL。
H／Hermiticity／involution=1e-11、controlled／unitarity=1e-10、mean／signal=2e-10、compiled=5e-10、
probability／mass／normalization L1=1e-12、coefficient=1e-12×max(1,reference)、angle=1e-12、
DP double=1e-10×max(1,price)、高精度DP=1e-40×max(1,price)、90/128 reference一致=1e-60。
compilerはQiskit1.3.0、opt1、seed17、rz/sx/x/cx、topology-free、approximation_degree=1、qubits_initially_zero=False。

今回ユーザーが新しい実行と技術的中断時の修正・新規実行を承認した。
旧grantを再利用せず、新grantを一回消費して新outputへ全登録caseを実行した。
今回の実行は正常終了し、追加の技術修正・別attempt・retry／resumeは不要だった。
一つのgrantを使った再試行を許可するようgateを緩和していない。

## 登録件数と実績

| family | 登録 | 完了 | 判定 |
|---|---:|---:|---|
| F1.CATALOG.EVENT | 136 | 136 | PASS |
| F1.CATALOG.MEAN | 2 | 2 | PASS |
| F1.COMPILE | 64 | 64 | PASS |
| F1.COMPONENT | 447 | 447 | PASS |
| F1.EXHAUSTIVE | 1 | 1 | fallback処理PASS／認証選択は未評価 |
| F1.H | 4 | 4 | PASS |
| F1.LAW | 348,232 | 348,232 | PASS |
| F1.MEAN | 4 | 4 | PASS |
| F1.MOTIF | 96 | 96 | PASS |
| F1.PRICE.ARMS | 4 | 4 | fallback処理PASS／認証選択は未評価 |
| F1.PRICE.COMPARE | 1 | 1 | PASS |
| F1.PRICE.DP | 1 | 1 | PASS |
| F1.PRICE.ENUM | 68 | 68 | PASS |
| F1.REFERENCE | 506 | 506 | PASS |
| F1.SELECTOR | 1 | 1 | fallback処理PASS／認証選択は未評価 |
| F1.WRAPPER | 64 | 64 | PASS |

全caseのID・重複・attempt/completion・family件数を保存journalと登録inventoryで照合した。
単位が異なるため、この表を科学call総数へ単純合算しない。
COMPONENTは447build＋operator check、MOTIFはraw／optimized／actual adjoint各32、
wrapper64、compile request64（完全IR一致cache hit6、実transpile58）、catalog controlled build136。
独立referenceは90digits253＋128digits253、全253組で精度間一致を確認した。

## 4arm比較と主要残差

全armでH-ID、fragment S/P、finite mean、LAW最終mass/L1、component、8motifの3variant、X/YとcompileがPASS。
表のPhiは既存literal RZ-token grammarの期待値であり、compiled期待costやshot加重資源値ではない。

| arm | LAW event数 | 全体mean残差 | compiled operator残差 | compiled signal残差 | Phi X | Phi Y |
|---|---:|---:|---:|---:|---:|---:|
| SS | 2,758 | 7.77226e-16 | 5.04633e-15 | 1.16747e-15 | 245.415917705 | 246.415917705 |
| SP | 195,170 | 7.77226e-16 | 5.13231e-15 | 1.16747e-15 | 125.889229236 | 126.889229236 |
| PS | 17,602 | 1.44337e-15 | 4.59435e-15 | 9.99201e-16 | 156.405846341 | 157.405846341 |
| PP | 132,702 | 5.55114e-16 | 5.0243e-15 | 7.77156e-16 | 71.567246528 | 72.567246528 |

最終LAW probability massは4armすべて1.0、component normalization L1はすべて0。
joint probability L1の最大は9.442571e-17、各eventのprobability／coefficient差の最大は2.775558e-17。
独立CAR／frame referenceへの照合はfull空間で行い、Hの2粒子sectorも別に照合した。

| 検査 | 全arm／catalogの最大残差 | 固定許容差 |
|---|---:|---:|
| 元Hと辞書H | 5.63536e-132 | 1e-11 |
| component／motif／catalog controlled operator | 2.20745e-15 | 1e-10 |
| controlled unitarity | 4.21885e-15 | 1e-10 |
| product Hermiticity | 2.44929e-16 | 1e-11 |
| product involution | 1.33227e-15 | 1e-11 |
| one-step corrected mean | 2.22046e-16 | 2e-10 |
| 全体 corrected mean | 1.44337e-15 | 2e-10 |
| raw motif | 2.20745e-15 | 1e-10 |
| basis共有 motif | 1.88738e-15 | 1e-10 |
| actual adjoint | 1.68731e-15 | 1e-10 |
| Hadamard X | 2.88658e-15 | 2e-10 |
| Hadamard Y | 4.78784e-16 | 2e-10 |
| compiled full wrapper operator | 5.13231e-15 | 5e-10 |
| compiled signal | 1.16747e-15 | 5e-10 |
| catalog corrected mean | 5.55122e-16 | 2e-10 |

control0/1 block、符号、relative/global phase、演算子順序をphase合わせなしで照合した。
Gaussian／Pauli basis共有のoptimized回路と、その実際のinverseも独立referenceに一致した。
raw signalとcompiled operator/signalを同じ独立referenceへ照合した。
90/128digits間のoperator最大差は5.598592e-91、signal最大差は3.006805e-91（閾値1e-60）。
詳細なcase ID・全残差の最大値は[要約JSON](../../../artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24-r1/F1_RESULT_SUMMARY.json)にある。

## Literal価格・DPと未資格事項

catalog68eventの直接価格を列挙し、DP期待値と照合した。
直接期待値36.611400465200184、DP 36.61140046520014、差4.263256e-14。
90/128digitsの直接列挙対DP残差は各7.854550e-90／1.384947e-127で、固定閾値内。
全4armのPhiも登録price case内でbinding・評価した。

interval providerは今回も未資格であり、4armとも `UNPRICED_FALLBACK`。
selector／exhaustiveも既存契約どおり `UNPRICED_FALLBACK, arm=SS` を返した。
**SSは未資格時の既定fallbackであり、認証された最適辞書ではない。**
Phiの大小だけから認証score、最適選択、shot数、最終資源優位性を結論しない。
以下は未評価：資格済みintervalでのscore enclosure／比較・tie処理、認証価格最適化、shot数、F2実費用・性能、新規性。
登録済み検証の未実行caseは0件だが、この未資格branchをPASSへ読み替えない。

## 実際の処理費用

開始2026-10-24 01:36:54 JST、終了01:37:23 JST、exit0。
wall=29.588294秒、CPU user82.35秒＋system1.94秒、GNU time CPU284%。
GNU time最大RSS595,704 KiBは最大観測値で、並行workerの合算RSSではない。
利用可能CPU affinity32、実行前MemAvailable約52.7GiB、disk空き約345.5GiB。
4arm＋catalogの独立5datasetを5spawn workerへ分割し、BLAS/OpenMP/Numba/Qiskit各1、Rayon各6。
wall／memory capは設けず、共有環境や他jobを変更していない。

保存case内の処理時間合計ではLAW34.916秒、reference30.430秒、COMPONENT1.147秒、COMPILE0.797秒。
これは並行実行された処理の合計であり、job wall timeとは異なる。
reference／basis／lawと完全IRのcacheを再利用し、比較用の反復benchmarkやspeedup測定は行っていない。
rawはprivateに1,930file／185,942,352bytesを保存した。
公開するのは科学判断に必要な要約と[raw相対path・hash台帳](../../../artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24-r1/OUTPUT_HASH_MANIFEST.json)。

## Provenance・不具合・引継ぎ

B科学SOURCE `3de42d3a3efe9ebc74dde8a1d0e1f1bdbadd607c`、B結果 `32c76bc6392ba277ce9fdd8d4bc54ffb5339b767`、
P0 `b8d54708f90ccef275eda3757143100b2d0c677e`、I `bfaab3831042f64a7112296437c0001e34920209`、
F1-PRE `31e8391a56fbb1a0e64f477f78e242655329de99`、旧停止結果 `594bab9e1441699474f262b075390db3ab8723c6` を保全した。
専用CPython3.11.9／NumPy1.26.4／Qiskit1.3.0／mpmath1.3.0を再利用し、環境17,617fileをpreflightで照合した。
SOURCE closure・保存入力13file・環境lockのSHA256と依存versionは要約JSONに固定した。
承認原文・grant・binding snapshot・raw・認証情報・内部pathは公開しない。

今回の科学実行に技術的エラーや未解決の数値的不一致はない。
報告スクリプトの初期assertがfallback armをNoneと仮定していた点は、既存仕様のSSへ修正した。
保存診断の集計だけの修正であり、科学SOURCEやrawを変更せず、科学計算を再実行していない。
旧停止runを成功へ書き換えず、今回の結果を独立した新しい証拠として追加する。
ローカル実行の結果であり、CIや独立した外部再現と呼ばない。

次の担当はユーザー／GPT：F1の範囲付きPASSと未資格事項を確認し、
[F2条件付き契約案](../pr_algorithm_improvement_phase2d_preparation_20261023/F2_CONDITIONAL_CONTRACT_DRAFT.md)で
finite-target比較／physical同精度比較のどちらを採るか、必要なinterval・統計・bias資格と実行範囲を判断する。
F2案はT=.2、q=1、delta=.2、r=2、h=.1、K=0であり、今回F1のlaw／Phi／meanをそのまま移送しない。
Codexの次の提案作業は、その判断に基づくF2入力・正確なcase件数・SOURCE・環境・資格条件の実行前固定。
F2科学実行は別の明示承認後に限る。新しい重要科学レビューも自動開始しない。

**COMPLETED_MANDATORY_STOP_F1_REVIEW_REQUIRED。F1終了後のmandatory STOPを維持し、F2へ進まない。**
