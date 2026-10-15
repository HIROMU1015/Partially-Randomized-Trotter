# H-E1 Stage 0：補正情報取得とPF構成の共同設計・技術事前監査

2026-10-15 JST。ユーザー指定の文献・固定研究資料・静的数式による一回の技術監査。**K0を完了して停止する。現定義から具体化できる既知の処理系は `REDUCES_TO_KNOWN`、H-E1全体の独立寄与は `UNDETERMINED`。独立した生成規則を特定できず、K0の進行条件は不成立。K1〜K3は未実施。** 不明であることを継続根拠にせず、追加候補探索・pilot準備へ進まない。

これはGPT科学レビュー、新規性の正式認定、研究投資・終了の決定ではない。新しい科学計算と保存数値の再集計は0。入力は厳密**量子**基底状態の利用権であり、古典ベクトル、RDM、基底エネルギーが無償で与えられる設定へ変更していない。

## 1. 完成版レビューの保存と研究状態

ユーザーが本文で指定した[完成版レビュー](../reviews/quantum_chemistry_pf/qc_onebody_repartition_h_boundary_final_scientific_review_20261015.md)の実ファイルを取得し、100,617 bytesをそのまま保存した。SHA256は `a88a7db0281db3ef26c1f919379802209e18bf42691786e9eea0ffebcfd0394d`。[原文identity・作成経緯](../../artifacts/qc_he1_stage0_prevalidation/2026-10-15/review_identity.json)を分離した。本文・数式・数値・引用の編集はない。

完成版自身はClaude原レビューとGPT注記を統合した資料と記しており、Claudeによる修正版の再承認を意味しない。参照名 `レビュー_outer.md` と `レビュー_修正版.md` の別ファイルは今回の添付・指定workspaceでは取得できなかった。完成版は取得済みであり `SOURCE_MISSING` ではない。未提供の原資料を復元・捏造せず、既存レビューは上書きしない。

| 区分 | 今回の記録 |
| --- | --- |
| 完成版科学レビューの推奨 | `CLOSE_CURRENT_CONFIGURATION_RECOMMENDED` |
| 第一候補の正式な採否・終了変更 | `AWAITING_USER_ADOPTION_DECISION` |
| 一体項再配分一般の不能性 | `NOT_ESTABLISHED` |
| H-E1の研究採択 | 未採択 |
| 第一候補の追加h・別分子・DF回転 | 起動しない |

固定結果は [500ea6の結果本文](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/500ea6ece429d6951f710cd99ac7e46337fcf017/docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_results_20261015.md)と[引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/500ea6ece429d6951f710cd99ac7e46337fcf017/docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_gpt_handoff_20261015.md)を正本とする。保存summaryと独立auditも読取り確認した。linear H4、1.00 Å、STO-3G、保存DF rank12・generation順全12 density、Nα=Nβ=2・36次元、固定先頭pair一方向X、比較h=0.002745119306432635〜1 Ha⁻¹、381比較点／補助含762点、target=10⁻⁴/10⁻⁵ Ha、価格P1/P2という限定結果である。PRのL_D・delta探索ではない。登録10%改善未達と独立監査PASSを保持し、今回それらを数値再認証・再ランキングしていない。

完成版§1.6が挙げたcommit差は、[commit役割表](../../artifacts/qc_he1_stage0_prevalidation/2026-10-15/commit_roles.json)に整理した。`3dd8acec…` は修復準備が参照する旧停止結果、`3953db12…` はchecker v2固定source、`d96c9c6c…` は修復準備の公開完了、`d7038d4d…` は実行前の別認可公開、`500ea6ec…` は結果公開である。異なる段階のidentityを同じ役割と解釈しない。完成版の「要確認」という原文は変更しない。

## 2. 仮説、入力・出力、今回確認できた範囲

H-E1の仮説は、PF構成と主要固有値biasの補正情報を取得する方法を共同設計し、同じ元Hamiltonianのground-energy精度・成功確率に必要な総資源を減らすこと。

入力として許すのはHの係数・実装可能なfragment、厳密量子基底状態の利用、指定精度・失敗確率・価格条件である。E0は設計入力にしない。期待値、RDM、covariance、gap下界、PF branch情報は、それぞれ取得方法と費用を要する。新たな状態準備法、block-encoding/qubitization/QSVT中心の設計、PR内部改良への移行は対象外。

独立算法として必要になる出力は、少なくとも具体的な写像

\[
\mathcal G:(H,\ \text{state access},\ \epsilon,\alpha,\text{prices})
\longmapsto
(S,\ \text{correction estimator},\ \text{acquisition policy},\ \text{validity evidence})
\]

である。今回受領したH-E1は、この写像の目的と候補要素を指定するが、既知PF・測定法と異なる状態、遷移、係数生成、測定回路、保証の具体的な規則までは指定していない。未知の写像について、既知法との完全な同値性も、優れた計算量も証明できない。

以下では「既知の部品」「それらを使う既知の費用評価」「まだ与えられていない新しい生成則」を分離する。文献本文を取得したことと、全証明の再証明・初出性の網羅調査を区別する。

## 3. 外部一次文献との直接比較（K0）

関連本文、版、URL、SHA、取得失敗と代替取得は[literature registry](../../artifacts/qc_he1_stage0_prevalidation/2026-10-15/literature_registry.json)に記録した。論文全文は再配布しない。以下の各行は本文の指定箇所に基づく技術的対応付けである。

| 本文・確認箇所 | 入力情報 | 目的／最適化自由度 | 保証の種類 | 情報取得費用 | 回路資源 | H-E1との一致・限界 |
| --- | --- | --- | --- | --- | --- | --- |
| [FFF、Quantum 7, 889](https://quantum-journal.org/papers/q-2023-01-03-889/pdf/)、§2.1–2.3、Eqs.4,9–14,22 | fragment、測定stateの分散・covariance。計画用近似state | Hの期待値測定費用。one-body再配分cとshot配分mを交互更新 | 分散会計は条件付き。c,m同時問題の大域最適性は保証しない | shot数を明示。covariance準備・近似誤差を量子state accessだけで無料にしない | basis測定とstate利用回数 | 「演算子の分割と情報取得配分を共同設計」は既知。H測定のFFFを高body-rankのPF誤差Dへそのまま適用できる保証はない |
| [Hamiltonian partitioning、2210.10189v2](https://arxiv.org/html/2210.10189v2)、one-body自由度・cost sections | fermionic係数、分解・回転frame | Trotter誤差指標とfragment構造、one-body再配分 | 規定したnorm/range boundsと数値比較を区別 | 係数処理。量子d測定の一括価格保証は与えない | fragment exponential、Gaussian基底変換 | 自由度・既存frameの回路保持は既知。新しい補正情報生成則を追加したことにはならない |
| [Mehendaleら、2312.13282v3／2025出版版](https://arxiv.org/html/2312.13282v3)、II Eqs.5–9、Appendix A.7 Eqs.31–34 | 係数とCISD等の近似state、BCH主要項 | PF eigenvalue誤差を予測し、step・QPE・合成誤差配分でT費用を評価 | 主要項・近似stateによる予測。全有限hの厳密上界ではない | classical proxy取得とexact量子stateからのd推定を区別 | rotations、T合成、phase-estimation回数 | 誤差係数をQPE資源へ戻すことは既知。候補のd取得法や総費用定理がなければ差は残らない |
| [Maxwellら、2606.30738v1](https://arxiv.org/html/2606.30738v1)、III.1–3 Eqs.9–12、IV.1 | fragmentとclassical MPS/MPO、state近似 | grouped/compact BCHとleading energy係数、重要項評価 | perturbative/asymptotic結果と数値heuristicを分離 | grouped contractionの古典費用。top-k切断は量子有限shot保証ではない | 主に誤差評価と実用予測 | O(L)等のgroup数削減を量子コピー数削減とみなさない。同じ恒等式・cache・RDMを強対照にも認める |
| [CasaresらSPRINT/GRADE、2606.30741v1](https://arxiv.org/html/2606.30741v1)、III.2–3、Appendices D/E.3/F.2 | 分子係数、norm階層、classical state estimator | grouping、ordering、near-integrable kernel、processing、費用 | 構成上のorder条件と経験的estimatorを区別 | estimator・補正構成は既存。全ての量子情報取得単価が無償とは読まない | controlled Givens/phase、Toffoli等 | PFと誤差・回路のcodesignは既知。量子d取得まで安くなる特定の新生成器は未指定 |
| [Corrected Product Formulas、2409.08265v4](https://arxiv.org/html/2409.08265v4)、§§1.1,2,3.2–3.4,5 | 二つのexactly simulatable partition、必要なら小摂動階層 | symplectic／symmetric／composite corrector、higher order | 明示条件下のorder/errorとcorrector分解誤差 | state依存dを取得せずoperatorを補正する方式を含む | exponential数、boundaryとstep内correctorの費用 | 強い対照。同じ「補正PF」でもscalar補正と別物。一般DFのB全体が安価にexact可解とは限らない。表記上の留保は§4 |
| [Rendon–Watkins–Wiebe、2212.14144v4](https://arxiv.org/pdf/2212.14144v4)、§3、Lemma6、Theorems7/15/17、Appendix D | controlled term evolution、isolated branch・gap/state access条件 | Chebyshev nodes、PF幅、推定precision、eigenvalue外挿 | 解析domain・準備・統計モデル等の条件付き定理 | uncertainty伝播と準備・推定費用を扱う | controlled evolution、ancilla、precision依存gate費用 | 外挿と情報精度の配分は既知。元Hのexact groundが全nodeのPF groundと一致するとは仮定しない |
| [Watson–Watkins、2408.14385v2](https://arxiv.org/pdf/2408.14385v2)、results・Richardson/interpolation・sampling sections | H、初期state、observable、PF | 時間発展observableの外挿、stepと推定誤差 | commutator条件等を伴うrigorous解析 | 統計増幅・反復を会計 | circuit depthとrepetitionsを分離 | 情報なし外挿を無料のbias除去としない。observableの定理をそのままground eigenphase定理へ移さない |
| [Well-conditioned MPF、Quantum 7,1067](https://quantum-journal.org/papers/q-2023-07-25-1067/pdf/)、§2.1–2.2、Appendices C–E | base PF、observable、選んだ指数・重み | 外挿係数のconditioning、回路深さ・統計負担 | 展開次数・noise増幅の解析と有限例 | 重みL1による統計・非Trotter誤差増幅を明示 | 独立回路の深さ、shots | 重み・幅の選択は既知。signed合算を単一unitary PFの固有位相と同一視しない |
| [Blanes–Casas–Ros、2000 processing](https://personales.upv.es/~serblaza/2000CelMechNI.pdf)、pp.1–5 Eqs.2–5 | 可解部分flow、near-integrable構造 | kernelとprocessor、Lie order条件 | classical symplectic/near-integrable解析 | ground-state情報取得の理論ではない | kernel反復＋端processorのflow費用 | processingとboundary cancellationは既知。量子への変換条件・controlled費用を別に要する |
| [Blanes–Casas–Shaw、2311.11581v3](https://arxiv.org/pdf/2311.11581v3)、§§1–3 | symmetric second-order map、合成係数 | generalized extrapolation、order、並列計算負担 | classical composition/order解析 | quantum finite-shot保証を自動供給しない | map数と並列latency | 合成・外挿の新しい名称だけでは差にならない |
| [Ikeda–Kono–Fujii、2307.05406v3](https://arxiv.org/pdf/2307.05406v3)、II–III、V.B Eq.33 | 現stateの回路access、低/高次PF | 低高次差を測ってstepを適応選択 | higher-order remainder条件と統計precisionの扱いを区別 | 測定回数と二期待値測定を明示 | PF深さ×測定回数 | 「測った誤差でstepを選び、情報費用を戻す」は既知。ground energyのscalar d補正そのものと同じtaskではない |
| [WangらHNCC、2607.11856v2](https://arxiv.org/pdf/2607.11856v2)、II、IV–V | Pauli/local Hamiltonian、stateとobservable、係数sampler | BCH remainderをchannelで補償し、sampling overheadを制御 | locality・強度条件付きcomplexity theorem | LCQC norm、sampling回数、preprocessing | Pauli回転channelsのgate数と反復数 | 情報・回路の交換を持つ強い関連法。channel observable推定をcoherent ground-eigenphase補正へ同一視しない |

この対応表から、**PF構造・誤差係数・推定精度・回路単価の同時評価という目的だけには独立性を認められない**。一方、各論文がH-E1の全自由度を同時に最適化する一つの定理を既に証明した、という結論でもない。特にFFFのmeasurement objective、compact BCHの古典計算、Chebyshevのeigenvalue推定、HNCCのchannel taskは異なる。

## 4. 理論上の混同を除くための静的確認

### 4.1 Processingの相似変換と固有値補正

既知のexact processor Pに対して `S'=P S P⁻¹` はSと同じ固有値を持つ。`P(h)=exp(h²K+…)`、`H_eff=H+h²D+…` なら主要演算子の変化は `[K,H]` 型であり、Hの固有状態ではその期待値は0である。この二つは相似変換と固有状態交換子恒等式の直接的帰結で、新定理ではない。

したがって、operator normの誤差が小さくなることと、同じkernelのground固有位相biasが小さくなることを区別する。processorによるoverlap・測定表現・varianceの変化はあり得るが、低費用化の具体的生成則は別に必要。近似processorやstep内correctorでkernelを変更する場合、その誤差と費用を落とさない。[SPRINT Appendix E.3](https://arxiv.org/html/2606.30741v1)、[CPF Eqs.4–6](https://arxiv.org/html/2409.08265v4)

CPF最新版の取得版は**v4、2026-07-28**。v3を最新版として引用していない。arXiv submission history、[出版社ページ](https://journals.aps.org/prxquantum/abstract/10.1103/phk5-slkl)、明示したerratum/correction検索では別の正式訂正文を特定できなかった。これは訂正不存在の証明ではない。

v4のHTMLとPDFのEq.28には、symmetricと記しながら右側を `exp(-C_sym)` とする表記を確認した。`C_sym=O(λ³)` のconjugationでは `[C_sym,K_2]=O(λ⁴)` であり、一般のλ³項を除去できない。本文Eq.5の同符号sandwichと整合しないため、**表示Eq.28をそのまま保証根拠に採用しない**。この表記を黙って修正したり、全CPF結果の無効性を主張したりしない。比較では定義Eq.5と明示的composite Eq.27、§5の分解誤差条件を区別して使う。これは取得本文の整合性留保であり、新たな訂正論文の発見ではない。

### 4.2 Scalar補正・外挿・信号の区別

`E_PF(h)=E0+d h²+O(h⁴)` を補正するにはdの推定値と有限h残差が別々に必要。dを測れたことだけではbranch、overlap、高次残差、controlled synthesisを保証しない。完成版の `η_d≲ε_info/h²` は今回のK1入力条件として保持し、新たな分子への数値代入・損益計算はしていない。

Richardson/Chebyshevの複数energy値の統計合算、MPF observableのsigned合算、単一unitaryのQPE固有位相は異なる出力。signal channelの保証をenergy bias保証へ変換するには対応定理が要る。exact量子groundの提供だけから、PF固有状態・単一位相・信号半径1・無料RDMを導かない。[既存RPE会計](../rpe_resource_accounting.md)

## 5. 第1・第2研究と本repository内研究との境界

同名main branchのREADMEだけでなく、専用研究branchの固定本文を取得した。[参照identity](../../artifacts/qc_he1_stage0_prevalidation/2026-10-15/reference_identity.json)には、当repository27本文と第1・第2研究repository14本文のcommit・path・SHA・取得範囲を保存した。別repositoryへ変更・commit・pushは行っていない。

| 研究・固定本文 | 入力／主目的と自由度 | 保証・情報費用／回路指標 | H-E1との境界 |
| --- | --- | --- | --- |
| 第1研究：[最終synthesis](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/cd81018932b9557b7a55e2e1cd8c5e3988245409/PF_first_study_final_synthesis_20260925.md)、[claim ledger](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/cd81018932b9557b7a55e2e1cd8c5e3988245409/PF_first_study_paper_claim_ledger_20260926.md) | approximate/exact state proxyとfinite-time PF branchを分離。PF・時刻・QPE配分、校正・domain・safety margin | developmentの機構・decision sensitivity。continuous rotation proxyであり全FT費用ではない。practical selectorにexact truthを渡していない | 校正精度を残余誤差予算・regretへ戻すことは既存。H-E1の名前だけで新しい予算研究にしない |
| 第2研究：[Direction C](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/1d01e77247364b49dbce2e9ce2fe7c32af0fb937/docs/second_study_v2/budget_safety_mechanism_20261005/final_direction_and_claim_scope.md)、[quantity dictionary](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/1d01e77247364b49dbce2e9ce2fe7c32af0fb937/docs/second_study_v2/budget_safety_mechanism_20261005/quantity_dictionary_and_analysis_spec.md)、[closure](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/1d01e77247364b49dbce2e9ce2fe7c32af0fb937/docs/second_study_v2/evidence_integration_20261006/approved_closure.md) | cheap predictor、spectral point/width、abstention、候補時刻、fixed-budget decision window | empirical widthとcertificateを区別。取得情報の正確さと資源上のdecision valueは別。cold/conditional/cache費用未確定を保持 | 「不確かさを狭くして総費用を下げる」は既存RQと重なる。量子d測定を安くする別操作列があれば差の候補だが今回未特定 |
| 第1研究の後続条件拡張：[19362cの結果](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/19362c23ccdc16b6ad0ff547cc363e138a3930d2/docs/research_outcomes/20261007/validation_expansion_results.md) | 同じ機構・枝・時刻手順の条件変更 | 数値成立とmethod benefitを分離。元予測・予算・分類を変更しない | 旧結論だけで後続を無視しない。既存手順の拡張であり、新しい共同生成器の証拠とはしない |
| Track A：[claim map](track_a_post_pm2_claim_evidence_map.md)、[RPE会計](../rpe_resource_accounting.md)、固定H4/H6確認本文 | DF-prefix・PF/finite-RTE・precision・shot/wrapper費用 | complex signalのbias・normalization・十分shots×compiled native費用。全ground taskや無料state情報の保証ではない | PR resource assessmentとtask接続をH-E1の成果として移管しない。bias/stat/confidence配分そのものも既知 |
| Track B：[D1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md)、[D1–D3固定照合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md) | finite operator表現、law、r、保証とnative価格。explicit synthetic provider | range/moment/bias、Bernstein十分予算、T/CX/1Q、prep/readout。online classical費用未取得 | PF/PR内部codesignは別担当。同finite mean、channel、ground phaseを同一にしない。別branchの現在作業へ介入しない |
| C1：[保存oracle診断](qc_c1_saved_oracle_diagnostic_v1_results_20261014.md) | 固定macrocycleとstate依存selector | 既知compositionと固定catalogの限定的費用診断。情報取得・運用precisionは未解決 | state-dependent leading coefficient・誤差配分だけでは新しいselectorにならない。再監査・再集計なし |
| C2/C3：[完了技術監査](qc_c2_c3_final_technical_audit_v1_20261014.md) | C2はPF eigenvalue外挿・branch、C3はSchur-root | C2の外挿・precision allocationは既知。情報・保証・全費用の低費用取得差は未特定 | H-E1をC2の再命名にしない。C3と除外したQSVT設計は起動しない |
| C4：[完了監査](qc_c4_finite_rte_method_efficiency_technical_audit_v1_20261014.md) | 固定finite meanのcost-aware proposal | 最適則・DP等は既知、効率的access・全費用は条件付き | 「測定負担と回路費用の積を最小化」自体を新規性にしない。production・再探索なし |
| 第一候補：[最終方法差監査](qc_ground_pf_final_method_difference_audit_v1_20261014.md)と今回保存レビュー | exact-state一体再配分・1-RDM応答 | 同情報compact BCHに独占的計算量差なし。有限h検証は別の限定結果 | 1-RDMの名称を変えて再提案しない。元の数学的成果と費用結果を保全する |

第1・第2研究はclassical surrogate/校正の価値を検証しており、H-E1の量子補正情報生成と完全に同じ実装ではない。しかしexact stateへ前提を替え、既存のerror budget・width・cost最適化を繰り返すだけでは独立した生成アルゴリズムの差を示せない。過去のnegative resultからH-E1一般の不能性を導くこともしない。

## 6. 同情報の強対照と、還元できる範囲

**既知の処理系B\***を、次の明示した既知libraryに限って構成できる。

1. S2/S4、適用条件を満たすSPRINT/near-integrable/CPF、合法な既知partitionを候補とする。未知のH-E1固有kernelを無料で対照へ渡す、という定義にはしない。
2. 各候補のDをgrouped BCH・同じground-state恒等式・cacheで表現する。古典RDMを使うなら取得精度・費用も候補と対照で共通に戻す。
3. Dの取得には適用可能な既知のobservable分解・測定・allocationを使う。FFFを用いる場合は、対象Dでそのfragment/測定条件が成立するか確認する。classical contraction数とstateコピー数を同じ指標にしない。
4. 推定dによるscalar subtraction、または適用可能な既知外挿を行い、statistical error・remainder・branch・overlap・合成誤差を分離する。
5. 同じ情報・価格・成功率条件の下で、既存の配分と費用比較を行う。T、コピー数、classical時間は根拠なしに加算しない。

H-E1がこの処理系の候補名・目的関数・最適化変数を変更しただけなら、**その部分は `REDUCES_TO_KNOWN`**。これは全組合せに一つの最適性定理があるという主張ではなく、各操作と情報条件が既知手順で実現できるという限定的な還元である。

独立差としてなお検討し得るのは、具体的なPF生成により補正用Dを低取得費用のclassへ写す規則、同じ情報で測定量や必要copiesを削減する手順、またはその総資源保証。今回、そのいずれについても**具体的な構成を特定できなかった**。「測りやすいDを探す」「共同最適化する」は目的の指定であって生成手順ではない。未知の探索を開始しない。

## 7. K0判定、未解決事項、K1〜K3の停止

| 対象 | 技術判定 | 根拠・停止理由 |
| --- | --- | --- |
| PF/partition選択、BCH d評価、補正・外挿、情報精度配分、費用会計の一般原理 | `REDUCES_TO_KNOWN` | 外部本文と既存第1・第2研究に具体的対応あり |
| exact quantum stateからのRDM/dを無料にしない条件 | `SUPPORTED` | state accessと測定取得の区別。既知取得法の適用にはvariance・precision等が必要 |
| near-integrable/CPF/外挿の適用 | `CONDITIONAL` | 可解partition、norm階層、branch/gap、測定モデル、分解誤差等 |
| H-E1固有の生成規則・情報量／計算量／全費用の改善 | `UNDETERMINED`、具体的寄与 `NOT_IDENTIFIED` | 新写像・出力class・対照との差・保証が未指定。未知自体は新規性ではない |
| K0全体 | `UNDETERMINED`、`gate_pass=false` | 現在具体化可能な範囲は既知。一般仮説の全還元・不能性までは証明しない |
| K1 | `NOT_EXECUTED_K0_NOT_PASSED` | 非自明な方法差が未特定。新しい取得価格・損益分岐評価を展開しない |
| K2 | `NOT_EXECUTED_K0_NOT_PASSED` | 強対照の**同定**はK0で実施。新性能上限・費用利益は計算しない |
| K3 | `NOT_EXECUTED_K0_NOT_PASSED` | 新しいscaling理論やH4からの移送主張を作らない |

重要な必須文献本文は取得済み。Blanes processingのlocal直接取得403はwebによる19-page PDF本文取得で補い、local PDFのSHAを捏造していない。停止理由は未取得abstractを残したことではない。不足しているのは **(i) H-E1固有の実行可能な生成則、(ii) 出力する誤差演算子の取得方法と保証、(iii) 同情報B\*で再現できない差の根拠**。これらを数値pilotで先に埋める計画は作成しない。

完成版にある1.5倍目標／1.3倍screeningは仮の例のまま。達成値、上限証明、登録契約、実行認可へ変更しない。研究として継続するか、既存研究へ包含するか、別方向へ移るかは[GPT引継ぎ](qc_he1_stage0_gpt_handoff_20261015.md)の科学判断事項とする。

## 8. 証拠、保全、停止

[scope](../../artifacts/qc_he1_stage0_prevalidation/2026-10-15/scope.json)、[stage/method status](../../artifacts/qc_he1_stage0_prevalidation/2026-10-15/method_status.json)、[原文identity](../../artifacts/qc_he1_stage0_prevalidation/2026-10-15/review_identity.json)、[固定参照](../../artifacts/qc_he1_stage0_prevalidation/2026-10-15/reference_identity.json)、[一次文献](../../artifacts/qc_he1_stage0_prevalidation/2026-10-15/literature_registry.json)、[旧保全索引](../../artifacts/qc_he1_stage0_prevalidation/2026-10-15/preservation_index.json)を保存した。

新H・分子・eig・RDM・期待値・PF・数値最適化・保存数値集計・sampling benchmark・compile/T合成・pilot・科学runner/test/checker・新GPT科学レビューはいずれも0。実施した検査はdocument/JSON/link/byte/SHA・Git来歴の整合確認だけ。科学sourceや本番サンプラーは追加していない。

基準500ea6の既存5,161保護files、162旧manifest recordsとnon-result fields、既存 `relevant_commits[7].description` 欠落FAIL、親workspaceの442件の既存編集を[metadata-only監査](../../artifacts/qc_he1_stage0_prevalidation/2026-10-15/static_integrity.json)で照合し保全した。global validator PASS・新CI・外部科学再現とは呼ばない。N1 STOP、N2 HOLD、N3 CLOSED、THRIFT現template CLOSED、C1〜C4判断、旧結果・認可・STOP・Track A/Bの独立性を保持する。

独立branch `qc-he1-stage0-prevalidation-20261015` に公開し、GitHubからのbyte再取得後 **Mandatory STOP**。第一候補の正式終了、H-E1採択、GPT重要科学レビュー、K1以降・数値pilotを自動開始しない。同じ一般調査を繰り返す作業にも進まない。
