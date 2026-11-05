# O1：既存のHamiltonian表現・分割と証拠範囲

2026-11-06 JST。SERVER 2（GPUA100-gleap02）での静的技術preflight。新科学計算、保存値の数値再解析・再ランキング、回路生成・compile、科学moduleのimport、test/runner実行は行っていない。結果欄は固定資料の引用である。候補採択・独立新規性の最終判断はしない。

## 1. 読み方と固定資料

基点は文献調査commit `e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8`（以下S）。3環境統合は `3571477093441ac4852166fe829f8ccc5db5ec4f`（以下I）を指定通り別途参照した。**資料収録commit、科学実行SOURCE、結果commitは別物**である。台帳でSOURCEがUNKNOWNの項目をSやIで埋めない。

正式共通運用指示 `research_project_instructions_unified_2026-10-25.md`、共通・repositoryのAGENTS.md、PROJECT_MAP.md、研究概要全文、研究目的・研究方法の該当節、VALIDATION_STATUSとmanifest構造を確認した。研究概要の旧時点の本文に対し、後続固定台帳の更新を優先した。既存台帳や規範文書は変更しない。[入口][entry]・[概要][overview]・[状態][status]

必須4資料ずつの参照先：

- I：[台帳][ledger]、[Gap][gap]、[再利用][reuse]、[残検証][remaining]。CSVはO1関連行の科学条件・SOURCE・結果・statusを静的に読んだ。全runの再監査ではない。
- S：[O1][opportunity]、[先行研究比較][prior]、[全体費用接続][costmap]、[文献・locator][sources]。Phase 2Aの強対照、2Cの方法差・終了判断、2DのF1/F2を必要な範囲で確認した。
- 外部一次本文の今回の確認範囲は§4。既存調査から継承した評価と今回直接確認したSOURCEを区別する。

本調査の共通taskは元の物理Hの厳密基底状態を受け取り、そのE₀を所定の精度・成功確率で推定すること。状態準備費は除外するが、E₀・gap・近似Hの基底状態・古典全状態ベクトルを無料情報にしない。新しい分子、rank、精度、失敗確率、compilerは固定していない。

## 2. 表現と分割の比較

| 方式 | 表現・係数norm | splitとPF/RTE | 元Hとの関係・主な費用境界 | 確認済み／不足 |
|---|---|---|---|---|
| Pauli-JW | `H=cI+Σ h_j P_j`。同一global Pauliをcollectした後のtail `λ_R=Σ_R |h_j|`。identityの扱いを明示 | weight順prefixのPauli項をD、残りをpaired finite-RTE。全非identity randomのLD0は、one-bodyを保持するDFのfragment LD0と異なる | 元積分・index・spin/JW順・scalar一致が必要。Pauli回転、制御、parity、合法な隣接簡約をfull X/Y wrapperに含める | H2の固定4方式、H4/H6の別Pauli検証が存在。同じLD値や異なる物理splitを「表現だけ」の比較にしない。希少次数の母平均価格・全RPE優越は未確立 [ledger][ledger] IDX-REP-08/SUP-PAULI |
| DF / explicit DF（X-DF） | 一体補正＋回転基底のdensity-square fragment。RTE用には各fragmentをI/Z/ZZ involutionへ展開。fragment ranking proxyと展開後exact λ_Rは別 | 保存rank/orderのfragment prefixをD、tailをRTE。`prepare_df_partial_s2`はone-body correctionを別のD blockに保持。D forward/reverse half sweep＋central tail | 打切りなしの代数、tol-only DF、legacy固定rankを分ける。基底往復/Givens、対角回転、control、scalar、step境界が価格を決める | ライブラリ・Track A・信号検証あり。旧screeningは失効。DFのnormをPauli normやqubitization用λへ無変換で代入不可 [DF準備][dfprep]・[tail][dftail] |
| C-DF / RC-DF | 一般対称Jと直交Uで圧縮。X-DFのouter-product制約を緩め、RC-DFは係数正則化も使う | 各回転基底の対角項をPF/PRに接続可能。ただしqubitization向けλと本paired RTEのexact λ_Rは同じ定義ではない | fitのtensor残差、sector/energy誤差、scalar・一体補正の対応が必要。少ないleafでもbasis/control費が減るとは限らない | 原著の圧縮・norm改善と、repository研究CのPR接続は既存。CCSD(T) heuristicは厳密E₀保証ではない。新rank探索や再fitを本調査で実施していない [RC-DF原著][rcdf]・[研究C既存調査][cfeas] |
| UWC | OO＋対象blockを保存するBLISSによってPauli係数を集中。λ最小化だけを目的にしていない | 変換後のweight順prefixとRPE費用を結合。論文の具体化はPauli LCU | OOはHと状態の座標を共変換。BLISSは対象sectorの保存であり全空間ground順位の保存ではない。logical回転価格／STARモデルと本native wrapper価格を区別 | 外部原著を強baselineとして扱う。repository旧grouped-UWCの数値はFOUND_UNVERIFIEDを維持し、復活させない [原著][uwc]・[継承比較][prior]・[ledger][ledger] UWC-DISCOVERY |
| 混在S/P辞書 | 固定残差fragmentをframe Z/ZZ（S）またはphysical Pauli（P）へ表し、global operator一致に基づきcollect。exact-zeroとidentityを元契約通り処理 | 同じD/Rを保持してtail辞書を変える。λ、component確率、event角度、finite normalizationはarm別 | 異basisの同じZ labelを同一operatorとしてcollectしない。typed dispatch、actual adjoint、controlled相対phase、全wrapper価格が必要 | F1登録correctness、F2登録全wrapper数値と単回費用が完了。NとN加重費用はnull。独立新規性未確立、singleton/PS48の正式一時終了を保持 [Phase 2C][mixedprior]・[F2原結果][f2result] |
| prefix / 非prefix | prefixは指定順の切断。Pauli weight順、DF weight proxy順、generation順は異なる | split変更はHD/HRだけでなくpartial-PFの非可換構造・位相・半径・順序も変える | 非prefixは物理splitの自由度であり新算法名ではない。追加探索・誤差情報取得・価格評価の費用も比較 | `explicit_ordered_partition`のadapterは存在するが新selectorではない。PR-2のprefix成功を非prefix成功へ移さない。P-C/P-D/BF/BM/R3の限定負結果・既知対応を維持 [DF準備][dfprep]・[Gap][gap] |
| 既存PR最適化 | 固定表現/split上のδ、round別r/K、PF family、β/α、shot、価格model | RTE次数とmicrostep、PF step/repetitionを区別。決定論・discard+scalar・partial・full-randomのendpoint比較 | biasを減らすためのstep増と、normalizationを抑えるためのr増は単価を上げ得る | Track Aの固定T信号比較、条件付き18round会計・energy接続は実施済み。最終元分子energy・全実装成功・不確かさ込みの最終優越は未確立 [Track A][tracka]・[ledger][ledger] SUP-RPE-* |

DF squareへの並べ替えでは `T_eff=T_bare+一体収縮補正` が必要。`V_square`をbare quarticと同一視するとHが変わる。核scalar、JW identity、Gaussian回路に含まれるphaseを一度ずつ数える。同一保持T／残差VをPauliとDFで比較する既存の静的設計があり、再設計は不要である。[matched partition既存設計][matched]

## 3. 保存済み結果・負結果の条件とSOURCE

下表の数値・順位は原報告の引用。新しい比、集計、順位は計算していない。記載のない詳細は固定原結果へ戻り、別系列の条件で補わない。

| 記録 | 保存科学条件 | SOURCE → 結果 | 到達点・保持する制限 |
|---|---|---|---|
| H2 Pauli/DF 4方式（IDX-REP-08） | linear H2、1Å、STO-3G、同MO積分、Nα=Nβ1の保存厳密最低固有vectorを共有、会計はρ₀仮定。tol-only1e-8 DF rank3、T=.8、q4、δ=.2、S2。PRはPauli LD7／DF LD1、r2/K4、64 paired価格trajectory。Qiskit1.3/opt1/seed17、topology-free | [比較carrier][h2result] `c8990de…`内SOURCE=`7a6c4263d96153ecbe721dcfb6941d0d2684b3b0`。case別にはPauli-PR/DF-PFが同SOURCE、Pauli-PF/DF-PRは`b5ed2c6e2399b55a0c24eb8437f842892a6703ef` | 条件付き比較完了。固定時間複素信号精度の結果であり全RPEではない。未観測rare-orderの母平均価格は未同定。carrierのSOURCEを全4caseの実行SOURCEにしない |
| F1修正版 | 4mode/2粒子/occupied(0,1)、density-square2fragment、LD1、SS/SP/PS/PP。T1/5、q2、δ1/10、r2、h1/20、K2。90/128桁reference、rtol0、phase fitting禁止 | SOURCE `8192530b17d6bc4af195c734ff43a6e78c506942` → 結果 `b0dad8cc2f7a4cb821f53185c4a65f3a42e8f381`（台帳MIX-F1） | 登録correctness PASS。未資格fallbackは最適selectorでない。分子geometry/AO basis/分子DF rankは非該当 |
| F2修正版 | 同合成fixture、T1/5、q1、δ1/5、r2、h1/10、K0。binary64 production＋90/128桁reference。Qiskit1.3/opt1/seed17、rz/sx/x/cx、topology-free、phase fit禁止 | SOURCE `9468c7089727f5648d9f59cddbaebe0a42f5f0a0` → [結果 `dc7378ab…`][f2result] | 全登録raw/compiled operator・signal・4arm mean PASS。保存単回7指標はPP<SP<PS<SS。`N=null`、`weighted_costs=null`、`physical_same_accuracy_claim=false`を実際に確認。occupied fixtureの検査を今回の物理H厳密ground条件の検証へ移さない。混在が必ず勝つ証拠ではない |
| core/pair（IDX-REP-09） | synthetic3mode JW正平方DF2、physical3＋aux1 pair。q1価格、q1/2/4 mean、複素精度.05/.02、原固定compiler | 実行SOURCEは台帳UNKNOWN。結果 `ae7434452450e80ec007d65417335e18a063fa1e`、`artifacts/representation_attribution_pair/2026-10-10/` | whole-Pauli強対照でcore利得が消失、pairのshot節約はbasis/central費に負けた。反可換pairの別名再開をしない |
| N1（IDX-REP-04） | synthetic JW4、ν2、signedDF2、T=.6、q1/2/4/8、δ=.6/.3/.15/.075、複素精度.04 | SOURCEは台帳UNKNOWN。結果 `4c719a72eb8caae2f77fc746bd4672d610d97610`、`artifacts/n1_decode_recovery_v1/2026-10-11/run1/` | decode修正後も固定contextでexact fallbackを選択。近似coreが常に有利とはしない |
| PR-2 / Track A | PR-2はH4/STO-3G/legacyDF12、1/1.30Å、保持rank3/6/9等、T=.8、q1/2/4/8、固定r/K表。後続Track AのH4/H6/H8は1Å/STO-3G、legacyDF12とtol-onlyDF19/28等を分離 | PR-2結果 `e3531956af6a6d359e2f1fb320e9a5179f41eb98` / `f29eafbf4412193e8e8660d9f051889c2cba684b`。各実行SOURCEは元結果へ。[Track A固定監査][tracka] | prefix transferとmatched信号比較を保持。strong D強化で差が縮む。rare費用、候補境界、非Clifford、欠測を含む全登録比較に留保。全energy taskへ一般化しない |
| P-C/P-D/R3（IDX-OPT-03〜05） | P-C：H4/STO-3G/DF12/LD3の局所距離窓からstretchへ。P-D：H4/DF12/LD3/4/5、5公式、S1 T=.8等。R3：静的検討のみ | P-D関連結果 `d4163acd7b577afaa7d6db8c29db64f76ec03ae7`。他SOURCE/結果の未結合は[台帳][ledger]通り | P-C広域予測破綻、P-D強baselineとの選択一致、R3具体的方法差不足。grid拡張だけを新機構にしない |
| 研究C update005 / update064 | linear H4、1Å、STO-3G、numberCDF8、Nα=Nβ2、sector36、元順LD2、partial S2/s1、δ=h=.1、r32、K0、m0..12。同candidate強D、own-DF ground、prep除外、P0/Qiskit1.3/opt1/seed17 | 005 SOURCE `62fa4c0144b9e36f9d96b7dff67628051946632d` → `459c8472d66689bf149a60fd6cc7311ff5237c65`。064 SOURCE `9837eac59b94113bf057fcc311a27b35e60343f5` → `3718d369b3db515b2dd13b2cdbb6c49d57880121` | 005独立640価格は符号/精度未解決。064独立256価格はnominal model限定支持。異cohortをpoolしない。元Hの厳密state・actual CA・全law成功を確認した結果ではない |
| 研究C固定word | 上記064条件、1 h-step、4構造word各32event、full256/controlled512 | SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7` → 結果 `3649b75af191916daf44abb91e14d34c43b83e8f` | 全512入力列のraw/compiled/actual inverse/X-Y数値PASS。全合法word・全law・13round・actual CAは別。有限集合correctnessを母平均保証へしない |
| 18round / energy | H4、1Å/STO-3G、legacyDF12、LD3対Full12、δ=.02/K2/round別r、18round/36軸、保存同H ground条件付き | 会計SOURCE `4591b8d75a2a6a55c74a51225f0c61648023c8ee`。energy後続は台帳SUP-RPE-ENERGYの固定chain | 条件付き成功とenergy bridgeは実施済み。直接compile q≤32と長q外挿、ideal IID・数学的decoderと実装全体を分離。正式未決定を維持 |

**研究Cの先行NO-GOを保持する。** [11/2既存調査][cfeas]は、basis-switch/overlap/凍結event価格、support/rank、relative Gaussian等について、complete P0価格への操作可能な接続不足を理由に新cost-aware fitの準備候補を選んでいない。同調査のr128/旧candidate条件を後続r32/064の結果で上書きしない。今回も新η-fit、switch-only fit、singleton selectorを推薦・再開しない。

## 4. 強baselineを弱くしないための文献照合

[UWC原著][uwc]の§II.2.4 Eqs.(17)–(18)、§III.2–3 Eqs.(28)–(35)、App.D.1を直接確認した。Pauli係数を降順に並べ、`G_m=min_LD[g_det,m LD+g_rand,m λ_R²]`、全体は`Σ_m e N_m G_m`。OO/BLISSとsplit最適化が既知であり、最終roundを重視する近似とround別表現の可能性も記載されている。したがって表現・split・RPEの結合そのものを新規機構に数えない。詳細な比較条件は既存[先行研究比較][prior]を継承する。

[PR v2][prpaper]は§VI.B・VII.A–B・App.Eの該当箇所を確認。DFで保持した残差をPauliへ戻す／回転基底involutionを使う発想は既知。[RC-DF][rcdf]はfactorizationとerror-corrected計算の節を確認し、fit・norm・CCSD(T) heuristicの範囲を区別。[BLISS][bliss]は対象sector保存、[SCDF][scdf]は圧縮とsymmetry shiftの既知結合を確認した。新たな定理証明・重要科学レビューは実施していない。

公正な強baselineは単一の旧DF点ではない。同じ元H/stateアクセス・error情報・前処理予算に対し、適用可能なUWC変換、collected Pauli、DF/X-DF/RC-DF、discard+既知scalar、full deterministic、partial/full-randomを含める。既存PF familyとδ/r/K/β/α再最適化、phase保持control・共有・compiler簡約を双方へ与える。baselineに同じ辞書・core候補まで開放すると同じ解になる場合は、独立新規性を主張せず、既知法の適用条件の知見として扱う。

UWCのlogical回転count／STARのangle依存実装、repositoryのtopology-free RZ/CX、Clifford+T合成は別会計である。原著の係数をこちらのfinite-RTE/shot式へ無条件代入したり、原著の定数sampling overheadを本実装にも仮定したりしない。旧grouped-UWCの来歴未結合を外部原著で補わない。

OOの座標上の表記変更と、実際に量子stateを回転する操作を区別する。後者が必要なら費用境界へ明示し、状態準備除外を追加演算すべての無料化へ広げない。厳密入力groundを理由に原著のhigh-overlapパラメータを0へ代入して、深さや総費用の特異な極限を採用することもしない。

## 5. SOURCEから見つかった再利用上の制約

- [RTE][rte]のpaired cutoff Kは偶数で、補正meanの多項式次数はK+1。`finite_event_mean_operator`は曖昧な旧wrapperでdeprecated。将来は列挙meanとsample meanを別APIで扱う。
- `normalize_involutory_tail`はdense operatorのHermitian/involutionを検査する小規模入口。科学的に必要なinvolution条件は再利用できるが、dense行列を大規模用の無料oracleにしない。既存symbolic tail/typed metadataを先に使う。
- [RPE会計][accounting] `evaluate_rpe_round_candidate`はDFPartialS2Preparation・二次PFに型を限定し、`reference_radius=1`を置く。新表現／splitに対する一般の信号半径保証を持つAPIではない。厳密Hのground stateが近似PFの固有状態とも限らないことは[研究方法][method]にも明記済み。
- 混在辞書の[価格DP][pricing]はliteral grammarの期待価格でありnative compile価格ではない。F2のactual価格と区別する。既存関数名・過去test PASSを未資格Nの代用品にしない。
- 研究Cの[performance SOURCE][cperformance]はphysical groundを別評価した後、candidate Hの`eig[1][:,0]`を下流stateに採用する。今回の元H state条件には直接適合しない。

再利用候補・必要最小差分は[最小検証計画](O1_MINIMUM_VALIDATION_PLAN.md)へ。以上は静的な存在・interface確認であり、新条件での動作検証ではない。

[entry]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/PROJECT_MAP.md
[overview]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/研究概要・現状.md
[status]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/VALIDATION_STATUS.md
[ledger]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/IMPROVEMENT_VERIFICATION_LEDGER.csv
[gap]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/IMPROVEMENT_GAP_MATRIX.md
[reuse]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/DUPLICATION_AND_REUSE_MAP.md
[remaining]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/REMAINING_VALIDATION_MAP.md
[opportunity]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/RESEARCH_OPPORTUNITIES.md
[prior]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/PRIOR_ART_COMPARISON.md
[costmap]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/END_TO_END_COST_ANALYSIS_MAP.md
[sources]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/SOURCES.md
[dfprep]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/df_partial_s2.py#L328
[dftail]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/df_rte_tail.py
[mixedprior]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_algorithm_improvement_phase2c_design_20261023/METHOD_DELTA_AND_PRIOR_ART.md
[f2result]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc7378ab34015287215bcf4ce52147e46d94ecd9/artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/RESULT_SUMMARY.json
[h2result]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8990de7721545faff88b636108034dedb80a463/artifacts/non_df_h2_process_recovery/2026-10-23/comparison.json
[tracka]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/track_a_final_technical_audit_20261019.md
[matched]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/h4_df_pauli_matched_partition_static_design_20261102.md
[cfeas]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/research_c_pr_native_cost_improvement_feasibility_20261102.md
[cperformance]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trottertracks/c_pr_integrated_performance.py#L114
[rte]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/rte.py
[accounting]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/rpe_resource_accounting.py#L564
[pricing]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation/pricing.py
[method]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/研究方法・解析手順.md
[uwc]: https://arxiv.org/html/2603.22778v2
[prpaper]: https://arxiv.org/pdf/2503.05647v2
[rcdf]: https://arxiv.org/html/2212.07957v3
[bliss]: https://arxiv.org/html/2304.13772v3
[scdf]: https://arxiv.org/html/2403.03502v1
