# V3 preflight：保存費用不確かさの意味と適用域

2026-11-03 JST、LOCAL。既存値の転記とSOURCEの静的確認だけを行った。新しい平均・SE・fit・誤差伝播・N×C・順位を計算していない。固定ファイル、Git blob、SHA-256、URLは[EVIDENCE_INDEX.json](EVIDENCE_INDEX.json)、72軸行の対応は[PRICE_EVIDENCE_MAP.csv](PRICE_EVIDENCE_MAP.csv)にある。

対象はlinear H4、隣接1.0 Å、STO-3G、8 system qubit、legacy DF12、Partial L_D3／Full L_D12、δ=.02、controlled partial-S2のcosine/sine Hadamard wrapper。状態準備を除くcompiled RZ数であり、実backend料金・時間ではない。Qiskit1.3.0、rz/sx/x/cx、opt2、seed17、backend/coupling/layout/routingは保存nullを保持する。[OPT2DOC][OPT2DOC]、[OPT2JSON][OPT2JSON]

## 1. 種類を分けた対応

| 区分 | 保存計算方法・所在 | 対象domainと意味 | 適用できないこと |
|---|---|---|---|
| 直接価格のsampling SE | COMPSRC6 `_statistics`：標本分散ddof=1、SE=√(s²/n)。OPT2JSON `direct_compiled_rz_measurements`にmean/SE/min/max/n | 固定compiler下の古典random trajectory価格平均の標準誤差。Partial r1/2/4はn8、r8/16/32はfresh n32、いずれも各q・軸・policyの記録 | 量子測定shot誤差、PF/RPE誤差、compiler/backend系統差ではない。両軸・両policyはpairedでありnを倍にしない |
| Fullの直接SE=0 | 同tableのdeterministic_tail_free、n1 | seed17の固定決定論回路を各qでcompileした保存値 | 母集団の価格が無不確かさ、全qの式が証明済みという意味ではない |
| affine単価の較正SE | PRESRC `affine_prediction_with_standard_error`、ABLATIONSRC `AffineAxisCostModel`。各round/axesの `propagated_calibration_standard_error` | 価格式 `(2-q) C1+(q-1) C2`。保存SOURCEはq1/q2較正を独立として `hypot((2-q) SE1,(q-1) SE2)` を用いる | 構造的model discrepancyを含まない。q>2は較正点の外。q4/8が他holdoutの間にあってもq1/2 fitからは外挿 |
| 較正由来の総費用half-width | ABLATIONSRC `evaluate_ablation_scenario`：round・軸のN×SEを保守的に加算し、1.96倍。保存名 `conservative_calibration_95_half_width` | 選択済みplanの較正誤差スケール。Partial 58532541516.722305、Full 0.0（保存値） | 1.96という係数だけで、選抜・未知系統差を含む厳密な同時95% CIにはならない。今回加算し直していない |
| q8 holdout discrepancy | OPT2SRC `evaluate_proxy_cell`、FITSRC `fit_affine_holdouts`。`abs(prediction-direct_mean)/direct_mean`。各r・軸・metric・policyを分離 | Partial δ.02の各rにq8固定holdout。全12 randomized groupの保存最大selected RZ .044897726326746004、全metric .04748804666729306 | 有限holdoutで観測された点残差。信頼上界やq>8全点の一様保証ではない。holdout自身のsampling変動も残る |
| A0 q16/32 discrepancy | A0 `fixed_holdout_rows`、LINEAGESRC。最新fresh q1/2較正を固定legacy opt2 q16/32へ照合。fitには追加しない | **L_D3、δ.02、r32、K2、selected RZだけ**。q16約3.345%、q32約4.911%。較正n32・holdout n8 | 最終r1/q16,32、全r、全metric、q>32への保証ではない。最終r32 round自体もq32768以降であり直接一致しない |
| A0標準化残差 | A0保存 `absolute_standardized_residual`：予測SE・直接SEのhypotで残差を割る | selected q32=2.012152530848661。5%点誤差gateはPASSだが1.96以内ではない | 統計的同一性、独立した新compile実験、全長qの信頼区間としない |
| `local_5_percent` | OPT2SRC `_comparison`：候補ごとの保存sampling half-widthに0.05×点費用を加える | 想定した5%の対称model幅と較正誤差を足した感度scenario | 確率95%・全長qの実測上界ではない。5% gateを通ったことだけでは正当化されない |
| `per_r_selected_measured_discrepancy` | OPT2SRC `_measured_model_half_width`：保存round費用に対応rの最大selected RZ holdout残差を掛けて足し、sampling幅を加算 | 現在のcoherent比較は**q8診断**に基づくper-r経験的帯。Fullはq16/32の保存RZ最大残差0.0を使用 | A0の4.911%へ自動置換済みではない。短q残差を最終長qへ移したscenarioであり全長q CIではない |
| `per_r_full_basis_measured_discrepancy` | 同じ関数のfull_basis_shared診断 | 補助policyの観測残差を幅として用いた感度scenario。選択policyそのものの再測定とは別 | full-basisを選択policyの独立した反復と数えない。A0 full-basis q32の5.598% FAILも保持 |
| `transfer_25_percent` | `_comparison`：sampling half-width+0.25×点費用 | 長q・移送不確かさを想定する25%感度scenario | 実測された25%上界、普遍的CI、backend/noise/別分子への保証ではない |
| q>32の系統誤差 | OPT2JSON scope `q_above_32_directly_validated=false` | 保存された保証・直接compiled価格は未発見。構造、境界相殺、transpiler最適化、trajectory長の影響が未検証 | 0と補わない。長q物理信号のV1/V2 PASSで価格の未知幅を埋めない |
| compiler/backend変更 | compiler固定条件とscope | opt1→opt2の同trajectory再compileは別の保存実験因子。coherent最終比較はopt2へ統一 | opt1とopt2の区間を混合しない。実coupling、noise、別Qiskit、別seed、別rank・分子へ無断移送しない |

式はSOURCEに記載された方法の説明である。今回それらの数式を保存数値へ新たに適用していない。[PRESRC][PRESRC]、[ABLATIONSRC][ABLATIONSRC]、[OPT2SRC][OPT2SRC]、[OPT2EXTSRC][OPT2EXTSRC]、[A0][A0]

## 2. 共有較正と依存関係

同じ `(δ,L_D,r,policy,axis)` のq1/2を複数roundへ再使用するため、予測誤差は共通較正成分を持つ。SOURCEのround・軸を横断するSE加算は、全round独立として二乗和を取る処理ではない。一方、q1とq2の単価SE伝播自身は独立較正という前提を持つ。保存manifestではq別task seedが異なり、監査はinitial/fresh間のseed非重複を記録しているが、それだけで全価格誤差の独立性を証明したとはしない。

同一trajectoryのcosine/sineとfull/selected policyはpairedである。コンパイル後RZが両軸で同じでも、二つの独立価格母集団の反復ではない。PartialとFullは同じcompiler/Hamiltonian/wrapper条件を共有するが、量子shotやrandom trajectoryも同じという意味ではない。候補間の系統誤差共通性・相関係数は保存比較で推定されていない。正の相関による費用差の打消しを仮定した新しい区間を作らない。

initialの一部trajectoryはWP05-bのもの、legacy opt2のq1/2はWP05-bR、q16/32はM08と同じphysical trajectoryを再compileしたもの。fresh32 extensionは失敗groupの**置換**であり、旧標本とのn40 poolではない。AUDIT・OPT2JSON・A0・旧compiler-transfer解析を別々の独立価格取得と数えない。[INITIALMAN][INITIALMAN]、[EXTMAN][EXTMAN]、[AUDIT][AUDIT]、[LEGACY][LEGACY]、[TRANSFERDOC][TRANSFERDOC]

## 3. 保存された比較と重要な回路群

次はOPT2JSON `coherent_reoptimization/best_by_ld/*/best` と `comparison` の**保存値**。このpreflightの再会計ではない。

| 保存欄 | Partial L_D3 | Full L_D12 |
|---|---:|---:|
| 状態準備なし点費用 | 1263314146772.9375 | 1327822299806.0 |
| shot総数 | 13588 | 11162 |
| 最終round寄与率 | .6012425822697168 | .4407641524844915 |
| 後半3round寄与率 | .9139324437497927 | .8302589410503727 |
| 較正95%と名付けられたhalf-width | 58532541516.722305 | 0.0 |
| per-r selectedモデルhalf-width | 53258783757.6017 | 0.0 |
| per-r full-basis補助half-width | 60636748803.40786 | 0.0 |

Partial点比 `.9514180827942961`、約4.858%低いという保存点選好は維持する。主要回路群はPartialのm15–17 `(q,r,K)=(32768,32,2),(65536,32,2),(131072,32,2)` と、それに対応するFull `(q,0,0)`。両軸を含み、いずれも直接q32域の外にある。Partial r8/16などのグループ間や具体的な次holdoutの情報価値順位は、保存された寄与率だけで数値的に確定できない。新しい感度解析候補へ残す。

| 保存scenario | Partial区間 | Full区間 | 保存判定 |
|---|---|---|---|
| local_5_percent | [1141615897917.5684,1385012395628.3066] | [1261431184815.7,1394213414796.3] | overlap |
| per_r_selected_measured_discrepancy | [1151522821498.6135,1375105472047.2615] | [1327822299806.0,1327822299806.0] | overlap |
| per_r_full_basis_measured_discrepancy | [1144144856452.8074,1382483437093.0676] | [1327822299806.0,1327822299806.0] | overlap |
| transfer_25_percent | [888953068562.9808,1637675224982.894] | [995866724854.5,1659777874757.5] | overlap |

Fullの縮退区間はこの経験的scenarioで保存sampling SEとholdout残差が0だった結果であり、全長qの厳密費用確定ではない。保存 `best/scenario_intervals` と `comparison/scenarios` には最下位桁の丸め差がある。表は後者を転記し、丸めの統一や再生成はしない。

A0は後続の別reconciliationで、coherent artifactを改変・再最適化していない。旧compiler-transfer focused値約1.215990e12、約8.422%差はmixed opt1/opt2であり、上記約4.858%の比較とは別。q16/32の旧2.340%残差とA0の新しい4.911%残差も、使用較正が異なるため交換しない。[OPT2JSON][OPT2JSON]、[A0][A0]、[TRANSFERANALYSIS][TRANSFERANALYSIS]

## 4. 価格だけでは解決しない境界

V2は保存物理平均・IID測定・外部alias window等を条件に、両候補の18roundを `CONDITIONAL_SUCCESS` とした。V1の状態identity、PF有効固有状態との区別、厳密単位半径未保証、Full半径7点の不一致は残る。これらを価格検証で解消したとはしない。[V1][V1REPORT]、[V2][V2REPORT]

状態準備は未測定。保存shot総数が異なるので共通準備価格Pも相殺しない。保存 `preparation_break_even` の点境界は26590335.133166734 RZ-equivalent/shotで、全scenarioはP=0で既に重なり、非負の共通Pで頑健な区間優位を支持しない。準備手順が候補ごとに違えば共通P仮定自体が追加前提になる。新しい準備価格・break-evenを計算していない。

長q価格を改善しても、実backend/noise、別分子・別DF rank、手法一般の最適性、最終総資源優位は自動的に成立しない。局所compiler精度向上の既存STOPを維持し、後続候補は[最小検証案](MINIMAL_V3_VALIDATION_PLAN.md)で分離する。

## 固定参照

[A0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/m06f_a0_proxy_lineage_reconciliation_v1.json
[ABLATIONSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_ablation.py
[AUDIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_fresh32_audit_20260925_065827.json
[EXTMAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-24/wp11_all_r_opt2_extension_20260924_222811.manifest.json
[INITIALMAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-24/wp11_all_r_opt2_initial_20260924_023012.manifest.json
[LEGACY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_compiler_transfer/2026-09-23/m06_l08_opt2_same_trajectory_compute_v1.json
[OPT2DOC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research_direction_full_opt2.md
[OPT2EXTSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_opt2_extension_analysis.py
[OPT2JSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json
[OPT2SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_opt2.py
[PRESRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_prevalidation.py
[TRANSFERANALYSIS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_compiler_transfer/2026-09-23/m06_l08_opt2_focused_analysis_reaggregation_v1.json
[TRANSFERDOC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research_direction_compiler_transfer.md
[V1REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md
[V2REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
