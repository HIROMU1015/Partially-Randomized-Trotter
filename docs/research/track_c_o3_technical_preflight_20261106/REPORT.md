# Track C（O3）：PF・回路合成・誤差配分の技術プレフライト

2026-11-06 JST。静的SOURCE読解、既存報告の引用、直接関係する一次文献の限定照合。
**総費用改善は未実証。候補3件と共通接続作業を整理した段階であり、研究の正式GO/NO-GOは出していない。**

## 1. 目的・環境・固定入力

元の物理Hamiltonian `H` の厳密基底状態 `|E₀⟩` を初期状態とし、同じ基底エネルギー精度・成功確率で、状態準備を除くPR＋RPE全体の総回路費用を下げる。近似DF Hamiltonian、`H_D`、PF有効Hamiltonianの基底状態へ置換しない。診断に真のE₀を使うことと、実行時の選択器へE₀・gap・exact biasを無料で与えることを区別する。

| 項目 | 今回の実施内容 |
|---|---|
| 環境 | **SERVER 2 / GPUA100-gleap02**。通常利用者所有のproject領域。SERVER 1・LOCALへの切替なし |
| 専用branch | `codex/track-c-o3-technical-preflight-20261106`。基点 `e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8` |
| 3環境統合 | **`3571477093441ac4852166fe829f8ccc5db5ec4f`**、[指定5成果物][integration]。基点に継承された同directoryとのGit差分なし |
| PR＋RPE先行調査 | **`e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8`**、[指定6成果物][survey]。O3・比較表・費用地図・文献本文案内・出典を参照 |
| 正式指示 | user-wide・projects・repositoryの`AGENTS.md`、`PROJECT_MAP.md`、利用可能な正式共通指示2026-10-25を実読。研究概要全文と関連status/manifestも確認 |
| 保全 | 既存checkout/index/branchを変更せず新worktreeを使用。他jobの停止・再起動・設定変更なし。旧SOURCE/raw/result/grant/ledger/marker/STOPは不変更 |
| 実行範囲 | 科学moduleのimport、研究runner/testの起動、新計算、保存数値再解析・再順位付け、simulation、sampling、回路build/compile、algorithm実装は**実施していない** |

shellの既定sandboxが起動時のnetwork namespace初期化で失敗したため、実行環境の承認機構を通して同じ通常UIDでファイル/Git操作を行った。sudo・OS設定変更はない。計算環境の新規構築やraw全量回収は不要だった。AI sub-agentは使用していない。

成果物は本書、[候補比較](CANDIDATE_MATRIX.md)、[再利用・強対照](REUSE_AND_BASELINES.md)の3点のみ。以下の数値・PASS/STOPは保存報告の引用で、今回の新評価ではない。

## 2. 既存PF・検証の到達点

### 2.1 次数と係数版

一般の整合的な次数pのPFでは、十分小さいstepの局所演算子誤差は `O(h^(p+1))`、固定総時間でのglobal誤差は `O(h^p)`。これは有限hの基底位相bias、半径、native費用、全RPE費用の優越を保証しない。丸めた係数には元の精度・検査窓を添える。

| 方式・改善対象 | 静的SOURCEと既存検証 | 有利・不利／現在の境界 |
|---|---|---|
| 対称S2、Suzuki高次構成、Yoshida型4/8次 | [PF係数SOURCE][pf-source]、[P-D registry/原報告][pd-pilot]。本repositoryの`4th`は**Yoshida triple jump**。SPRINTの5-composition Suzuki再帰と同じ係数名で扱わない | 高次ほどstage・負時間が増え得る。D内部だけの四次化では外側D/R Strang誤差は二次のまま。S2も有力対照として残す |
| Morales 8次・公開高次・processed YP8m8 | 旧m8は[P-D固定SOURCE][pf-source]。別PF repositoryの[公開係数版][external-pf]にはY8m10b、YP8m8 kernel/processor等が別関数で存在 | **旧`8th(Morales)`、公開Y8m10b、YP8m8を同一候補にしない**。P-Dでは旧m8、研究Bでは公開版を使う。10次関数の存在はPR全費用検証済みを意味しない |
| 既存4次改善・PFと費用の共設計 | P-Dのnew_2は8桁係数による実用次数候補。別PF repositoryにはeigenvalue-optimized m6とactual-circuit-optimized m5もある。新係数探索は今回行わない | 旧係数最適化・回路最適化を新算法として再命名しない。new_2の数値次数検査を厳密な記号的証明へ格上げしない |
| P-Dのenergy/tail-aware選択 | H4 linear 1Å、STO-3G、8 qubit/4電子、DF12、LD3 development/4 holdout。δ=.4、energy tolerance 1e-6 Ha、branch weight≥.9995。[原pilot][pd-pilot]はMorales8対new4の選択差、[現実化][pd-real]は負時間小行列・内部D近似を確認 | 後続S1は同じH4/LD3、T=.8、δ=.1/.2/.4、K2と限定K4。[強対照B1b/B2/B4は同じ候補][pd-fair]。[事後解析][pd-post]もfinite補正固有の選択利益を支持せずS2へ進まない。初期の有望表示だけを採用しない |
| BCH・compact commutator・BM | [BM-0.5原監査][bm]は固定nested列の三次係数が同情報compact再帰と一致することを保存。符号・inner/outer規約とcacheも対応済み | 三次の形式log同値は有限RTEの信号保証ではない。group別triangleとcombined評価の違いを新operatorや独占的評価法にしない。BM-1の新手法gateは未成立 |
| BF finite-RTE係数選択 | [BF1-R0原報告][bf]。H4 linear1Å/STO-3G/DF12、generation-prefix LD3、T=.8、5-stage symmetric4次、q1/2/4/8、Rbudget5/10/20/40/80、K2/trigger K4 | 原runはserialization中断・INCONCLUSIVE。別SOURCEの保存復旧は事後解析で、F最良がL集合にも含まれF固有decision gainなし。欠落bridgeを埋めず、retryしない |
| PR-2 | [M2報告][pr2]。H4 linear1.30Å/STO-3G/DF12、T=.8、事前固定5構成。B2 LD3、B0 LD6、B1 LD12、B3 LD0。Qiskit1.3/opt1/seed17/free topology、状態準備なしHadamard | 固定構成のmatched **complex signal** accuracyとheld-out移送を支持。全energy/RPE、高次PFを再最適化した強対照、一般最適rankの証拠ではない |
| PR-3 | [固定pilot][pr3]。2qubit toy、T=.8、tail qDRIFT levels4/8/16、complex RMSE .05、分子/DFなし | 外挿biasは低下したがvariance＋backbone費用で通常PR最良の**7.557倍**。`STOP_PR3_VARIANCE_BACKBONE_DOMINATES`を維持。同じ外挿を候補にしない |
| Processing・near-integrable・研究B | [研究B一時保留報告][research-b]。synthetic2mode/N1・mode0入力、3DF因子、D=onebody+DF0/1、R=DF2、T=±.18。D内部／outer／finite-RTEを分離し公開Y8m10b/YP8m8・SABA2も照合 | D内部高次化の局所改善はあるが、A3/A4差は丸め水準で未識別。corrected改善とraw改善も一致しない。Full Yoshida4 q4の強対照が重要。rare価格・γ等未資格、同精度G未確定、**一時保留**。元H分子ground検証ではない |

P-Dの明確な教訓は「演算子・固有値の良い公式を、そのまま安いPFと呼べない」こと。後続強対照が既に同じ候補を選ぶ領域で、同じscore探索を再提案しない。BF/BM/PR-3/研究Bの旧STOPをTrack Cという名称で解除しない。

別PF repositoryの有限時間校正・response/BCH高次系列については、統合台帳`IDX-PF-04〜07`の帰属付き要約を継承した。S4 `complete_no_benefit`、S5へ進まないSTOP、モデル増量の単調改善未確立を保持する。全原結果を今回新規精読・再認定したとはしない。

### 2.2 合成・誤差配分・RPEの既存証拠

| 系列 | 既存到達点 | 未解決／誤用を避ける境界 |
|---|---|---|
| SP0.5/SP1 | [SP1報告][sp1]：2qubit synthetic coherent wrapper、12wrapper/48mask、εcomplex=.05、αfamily=.05、native operator error1e-6。fusion後の**加法primitive Tモデル**で局所gain、長い列でloss/shot cap | Cのrandom coinはactual finite-RTEではない。whole-wrapper compileでも分子groundでもない。SP toyの同じmask探索を再開しない |
| R1/Parent・ALLOCATION | 統合`SP-R1/ALLOCATION`に訂正S4/KKT、finite-P5再配分、controlled phase/actual adjoint、native接続が存在。[訂正版SOURCE][parent]を確認 | cost×二次moment最適とfinite-confidence全RPE最適は別。全trial・weight・非unitary mean・古典取得費を保持。新IS式は本候補にしない |
| FR phase/radius分離 | [FR-R1b][fr]：非一様4×4 toyの同情報bound改善witnessあり | 固定位相予算1e-2/1e-3/1e-4 radではFRだけの認証0件、`MECHANISM_ONLY_NO_PRACTICAL_GO`。scalar分離だけの旧課題を再実行しない |
| 短round・全round | [旧研究概要][overview]・[status][status]に短4round、round別r/K/β/α。[後続V3-R1][rpe-r1]はH4 linear1Å/STO-3G/legacyDF12、LD3対Full12、δ=.02、K2/round別r、18round/36軸、Qiskit1.3/opt2/seed17/no routingの保存モデル | 約4.858%のPartial点選好は保存結果であり、共分散・長q systematic込みでは**UNDETERMINED**。条件付き18round成功・energy接続は未着手ではない一方、最終資源優位・全実装保証ではない |

## 3. PF誤差から全費用へ接続する際の区別

全方式の比較対象は、資源aごとの

\[
G^{(a)}=\sum_{m,b\in\{X,Y\}}N_{m,b}\,\mathbb E[C^{\mathrm{full,no\text{-}prep},(a)}_{m,b}]
\]

である。RZ、CX、T、depthを別々に扱い、換算率なしに合算しない。classical取得/最適化/生成費、qubit数は別記する。

| 量 | 意味・必要な対応 |
|---|---|
| 演算子ノルム誤差 | 全状態の差の制御。小さくなるだけでは元H groundのenergyやGの改善証拠にならない |
| PF有効Hamiltonianの固有値bias | 適切なlog branch上でのE_eff−E₀。D/R外側とD内部の誤差を分ける。純粋unitary processingではkernelの固有値集合は不変 |
| 元H groundから見た位相 | `arg ⟨E₀\|U_PF(h)^q\|E₀⟩`。複数PF固有位相の混入・branch・有限時間を含むため、固有値biasとは別 |
| finite-RTE複素信号 | 順序付きrandom pathの**振幅平均**。`Q_m=E_p[wV]`、`Zcorr=⟨E₀\|Q_m\|E₀⟩`。system channelの一致だけではHadamard信号を保証しない |
| 半径・attenuation | finite-meanの混入による半径と正normalizationによる減衰を区別。`Zraw=Zcorr/B`という規約ならraw半径とcorrected estimator rangeの負担を二重計上しない |
| weight・shot | 有効な正半径下界と実際のestimatorのrange/moment、X/Y別αを使用。bounded IIDの十分Nをweighted/correlated測定へ無変更で移さない |
| full回路・全round | D backbone、controlled PF、basis往復、相対phase、合成、readout、processor端点を含める。単step価格や中央RTEだけではGを決めない |

processingを `U′=P†UP` と書けば、有限時間signalは `⟨E₀|P†U^qP|E₀⟩` となる。これは同じ入力状態に実際にPを作用させる回路の記述で、元Hの基底状態の仮定をPで変換した別の無料state oracleへ置き換えない。作用し得るのはoverlap・有限時間phase混入・半径であり、**同じkernelに対する固有位相bias改善とは区別する**。processorの共通共役による内部相殺も、RTEを挟む別D runを無条件に跨がない。

静的SOURCE照合で、[既存`evaluate_rpe_round_candidate`][rpe-code]には`partial-S2`限定と `C δ²`、基準半径1のmodelがある。高次/processed/weighted mean用の一般入口とみなせない。元の短round検証の不具合認定ではなく、**Track Cへ移す際のinterface・適用条件の不足**である。`C_use`は実行窓の経験的envelopeであり厳密上界ではない。

## 4. PFとnative synthesisの協調で確認すべき点

| 変更 | 費用・誤差へ効く箇所 | 現在の証拠境界 |
|---|---|---|
| PF次数・係数・stage | D単価、tail occurrence数、負時間、absolute tail exposure、整数r配分 | 既存P-Dが指標を持つ。高次の短いstep数だけで比較しない |
| controlled PF・basis共有 | diagonal-only control、actual adjoint、basis往復/隣接融合、identity相対phase | [既存DF builder][builder]で実装済み。全armに同じ最適化を与える。P-Aの一区間同値を新機構にしない |
| synthesis precision/残差化 | deterministic synthesis bias、finite-RTE mean/normalization、sampling weight、T/CX等 | PR原著App.E・既存SP/Parentを対照にする。RZ個数の不変性はClifford+T合成費の不変性ではない |
| processor端点 | full interrogation内で共有できる範囲、q依存の加法費、D/R境界での再出現 | [既存processed費用helper][processed-cost]はkernel反復とprocessor pair数を分離。構造価格でありcompiled wrapper価格ではない |
| 誤差配分 | PF/finite-RTE/synthesis/statistical phase、αの配分→r/K/N→G | fixed-split配分は既存。表現/splitをroundごとに変えるO4は今回提案の要件にしない |

既存priceは元snapshot/compiler/control/wrapper/precisionの範囲だけ再利用候補になる。新しいcompiled資源取得、保存price再集計、候補ランキングは今回行っていない。後続長q価格の未知幅を0としない。

## 5. 直接関係する先行研究との照合

下表は関連箇所の限定確認。全定理の再証明・完全な新規性審査ではない。詳細な元調査locatorは[固定SOURCES][survey-sources]へ戻る。

| 既知法 | 今回直接確認した範囲 | Track Cへの含意 |
|---|---|---|
| [PR原著v2](https://arxiv.org/pdf/2503.05647v2) | App.E E21–E24周辺：全round費用、normalizationとshot、係数丸めのresidual化 | PF/synthesis/shotの結合自体は既知。原RMSE会計を任意成功確率保証に読み替えない |
| [SPRINT v1](https://arxiv.org/html/2606.30741v1#S3.SS3) | §III.2/3、App.F.3：near-integrability、processing、合成のspectral shift/減衰 | 単純な高次化・共役・合成最適化は既知。spectroscopyのlinewidth解析をRPE成功定理へ移さない |
| [高次PF原著v1](https://arxiv.org/abs/2210.15817v1)・公開係数SOURCE | 原著metadataを確認。係数内容は旧P-Dと公開repositoryの固定関数を直接区別 | 旧m8と出版版を区別する。原著の係数表を今回独立再検証してはいない |
| [TE-PAI v3](https://arxiv.org/html/2410.16850v3#A1)、[continuous-time PAI v3](https://arxiv.org/html/2604.02854v3#A3) | channel法とoperator平均法の区別、後者App.CのHadamard用TETRIS | gate削減とsampling overheadを一緒に比較。状態準備改善は今回の目的から除外。channel一致だけでPR oracleを置換しない |
| [UWC v2](https://arxiv.org/html/2603.22778v2#S3.SS3) | §III.3の費用目的・round別表現への言及と古典費 | 全費用最適化もround依存構成も既知。表現変更はO1/O4との共通境界であり、本候補の独自性にしない。repository旧UWC数値はFOUND_UNVERIFIEDのまま |

## 6. 残る候補・技術的成立性

[CANDIDATE_MATRIX.md](CANDIDATE_MATRIX.md)で仮説・反証・最小検証・STOPを比較する。

1. **C-N：既知PF間の非加法なnative境界価格を利用できる条件。** 同じ適格信号・全Nで、加法stage proxyでは捉えないD/R/control/合成の価格差が利益として残るか。単なる並べ替え・既知fusionだけなら応用実装へ限定する。
2. **C-S：actual finite-RTEに接続した合成精度・残差配置の有利条件。** 安い合成と増えるB/weight/shotをfull Gで比較する。SP toyの再探索ではなく、元H ground・coherent mean・長roundの適合が未検証という限定した接続課題。
3. **C-P：固定kernelのprocessingによる有限時間信号の有利条件。** kernelの固有値を変えずに、半径/多位相混入の改善がprocessor費を上回るか。既知processed法に対する追加性質は未確立。

3候補は変更する層が異なるため、将来別途許可された場合に固定対照から個別に検証可能。C-NのPF価格、C-Sの合成・残差、C-Pの信号/端点は各々単独ablationを持てる。一方、元H/state・共通target/情報・phase-preserving full wrapper・finite mean・半径/shot/branch・全round会計を共有する。候補を最初から混合しない。**ここでいう並列研究は科学的独立性の整理であり、計算実行・sub-agent利用の許可ではない。**

**共通接続A**はPF/RTE/synthesis/statの誤差予算とα配分であり、独立した第4の新算法候補には数えない。既存fixed-split r/K/β/α資産を用い、強対照にも同じ最適化機会を与える。各候補単独に利益がなくても、既知法の適用結果として報告できる。形式上3候補を残したことは科学GOを意味しない。

## 7. 次の判断に必要な情報・STOP

ユーザーが将来進める場合に指定すべきものは、候補または応用評価の範囲、元H/sector/入力stateと近似誤差の関係、energy精度・失敗確率、主資源指標、compiler/native精度、利用できる誤差情報、baselineへ与える最適化範囲である。旧P-D/研究B/SP/FR等を再開するなら対象と変更範囲の明示が別に必要となる。

Codexの次の科学作業には、新係数/新bias/半径評価、保存結果の再会計、simulation/sampling、合成・回路build/compile、必要なadapter実装等が含まれ得る。**今回は全て未実施・未承認**。採択が実行許可を代行しない。重要な詳細科学レビューを行う場合も別の開始許可が必要で、今回自動開始していない。

公開対象は3文書だけ。技術検査はUTF-8/Markdown構造・固定参照/ローカルGit blob・差分の範囲確認と、公開後のremote SHA/取得本文/Git blobの照合に限定する。新しい科学的PASSや改善率を出すためのtestは起動しない。公開照合結果は最終チャットでremote commitとともに報告する。

**mandatory STOP：技術調査・候補整理・3成果物の公開照合まで。旧研究STOPを維持し、自動実行する後続研究なし。**

[integration]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105
[survey]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106
[survey-sources]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/SOURCES.md
[pf-source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/product_formula.py
[external-pf]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/product_formula.py
[pd-pilot]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_energy_tail_pareto.md
[pd-real]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_pd_realization.md
[pd-fair]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_pd_fair_comparison.md
[pd-post]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_pd_s1_posthoc.md
[bm]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md
[bf]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d2645a09440f50e5b869ef42a1b73a1b625a1af/docs/tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md
[pr2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md
[pr3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/pr2_pr3_minimal_pilot_validation.md
[research-b]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9adc2ce1c373c7ba311864c733ac6f40b575fbc/docs/research/research_b_final_outcomes_and_pause_handoff_20261026.md
[sp1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md
[parent]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ccc1832f63cd323b23fc05623f7e0b58cf34f37/src/trottertracks/algorithm_codesign/parent_allocation.py
[fr]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/fr_revision_nonuniform.md
[overview]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/研究概要・現状.md
[status]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/VALIDATION_STATUS.md
[rpe-r1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/REPORT.md
[rpe-code]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/rpe_resource_accounting.py
[builder]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/df_rte_qiskit.py
[processed-cost]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/processed_cost.py
