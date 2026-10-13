# 量子化学mixed-access feasibility v1：固定H4一回結果

2026-10-13 JST。branch `qc-mixed-access-feasibility-run-v1-20261013`。
**`NO_POSITIVE_BUDGET_IN_REGISTERED_MODEL`、Mandatory STOP。**
固定scopeの154 PF queriesと1386 partial resource rowsを完了。273 eligible R/frontier recordsの全てで、κ3/6/12の登録費用hypothesisによるremaining budgetが負だった。
これは指定回路template・精度配分・価格hypothesis・有限gridの観測であり、THRIFT全般の不能、最適回路のlower bound、量子総資源の科学的結論ではない。研究継続・詳細backend投資判断は未実施。

## 固定sourceと認可

- 科学source payload：`102e6b31401356d6eea775169041ebdf2cfa6ba0`。技術準備公開head：`18a47a4b0052e1c443e97f0317aa7e574ddcf562`。
- 認可・host preflight固定commit：`214f9ab2d7d3e0b64b287f14f49795c99a1a8244`。
- [ユーザー実行承認記録](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/user_authorization.json)：直前の固定H4一回scope説明に対する「次の作業に進んで大丈夫」。GPT新レビューの承認を含まない。
- [原科学契約](../../artifacts/qc_mixed_access_feasibility/2026-10-13/feasibility_contract.json)、[source/lock binding](../../artifacts/qc_mixed_access_source_v1/2026-10-13/source_manifest.json)、[固定実行計画](../../artifacts/qc_mixed_access_source_v1/2026-10-13/execution_plan.json)。古い未認可status/null欄は履歴として変更していない。
- [launch receipt](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/launch_receipt.json)、[実行log](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/science_run.log)。attempt1/retry0。source16 locked filesは実行前後とも不変。

実行commandは、固定CPython3.12.3 venvで、全BLAS/OpenMP/NumExpr系thread変数を1にした
`python scripts/run_qc_mixed_access_feasibility_v1.py --execute --authorization artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/user_authorization.json`。
他のH2/H4旧案、N1/N2/N3、Track A/B計算は開始していない。

## 対象・入力・実行窓

H4直線鎖、隣接1.00Å、z=(-1.5,-.5,.5,1.5)Å、STO-3G、charge0/spin0、実RHF canonical MO、4 spatial/8 spin、ν4、frozen coreなし。
新規RHF一job、conv_tol1e−10/max_cycle50/minao/標準DIIS、通常iteration5で収束。標準extra convergence diagonalizationも残した。FCI・多体ground/gap対角化は0。
[入力pack](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/input.npz)、[input seal](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/input_seal.json)。
SHA256：`f345af12281836731c3122bb2a4c0c2b389d7c90ded7ba16976d3cfeeb7bd951`。既存rank12 snapshotを再利用していない。

spin順は`2pα,2p+1β`、bit0=mode0、chemist ERI、nuclear scalarを保持。
DFは全16 factorsと補正one-bodyを使用。1e−12 reporting thresholdでのnumerical rank10は表示だけで、16→10へのtruncationは行っていない。内部rankも同thresholdの表示のみ。L_Dなし、PRなし。
geometryはdevelopmentで既知でありblind transferではない。

P2（JW）、E2/E4（excitation）、D2/D4（full DF）、R2/R4（known compressed THRIFT）の7方式。
T={1,4} atomic time、r={1,2,4,8,16,32,64,128,256,512,1024}、7×2×11=154 evaluations。
全256 output rows×ν4の70 input columnsで`||(S_m(T/r)^r-exp(-iTH))Qν||₂`を計測。各Pauli stageをsector投影せず、phase最小化なし。
εU=.001はoperator errorであり、ground-energy1mHa達成ではない。
PF fractions .4/.6/.8、Rの残余coefficient/function/synthesis三等分、B二等分、Dのmodel residual控除、numerical margin1e−8を固定。
次の2rと全ての大きい登録rが通る場合だけeligible。未実行rの外挿・PF係数fit・追加precision/geometry/rank scanなし。

## 保存された構造量

[structure](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/structure.json)、[derived pack](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/derived.npz)。

| 量 | 値・意味 |
|---|---|
| M_raw canonical oriented monomials | 324。随伴側も数えるnormal-ordered offdiagonal monomial数 |
| M_eff | 90。同一single transitionのdensity-assisted項だけexact group |
| z_j | 4…6。spectator detuningのliteral非zero数。定数は別 |
| λ_A / λ_B | 22.7290115796 / 6.12755273356。descriptorのL1 envelopeで、真のnorm比やαの証明ではない |
| `||B Qν||₂` | 0.941797465695。補助SVD一回、真のground/gapを用いない |
| DF coefficient max residual | 8.32667e−16 |
| full-Fock DF matrix residual Frobenius | 7.87555e−14。operator差の保守的上端として予算控除 |
| factors / numerical reporting rank | 16 retained / 10 at1e−12 |

signed densityとliteral nonzeroを保持した。同じgeometry/basisでも旧rank policyとの無条件の費用比較は行わない。

## ideal PF step数の観測

[全154 probes](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/probes.json)、[保存集計](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/descriptive_summary.json)。
下表は全方式に同じPF fraction .6（PF予算.0006）を与えた最小eligible r。coefficient/function/synthesisを加えた実compiled誤差の測定ではない。

| 方法 | T=1：r | 同rのoperator error | T=4：r | 同rのoperator error |
|---|---:|---:|---:|---:|
| P2 | 8 | 5.57418e−4 | 64 | 2.69999e−4 |
| E2 | 8 | 5.66599e−4 | 64 | 2.74722e−4 |
| E4 | 2 | 5.16830e−5 | 8 | 4.77571e−5 |
| D2 | 16 | 2.23571e−4 | 64 | 3.85259e−4 |
| D4 | 2 | 5.11161e−5 | 8 | 5.37890e−5 |
| R2 | 8 | 1.64213e−4 | 32 | 3.94818e−4 |
| R4 | 1 | 2.30620e−4 | 4 | 1.89449e−4 |

同次数E4/D4と比べR4は両Tで半分のr、R2はT4でE2の半分だった。T1・fraction.6のR2/E2は同r。
これはこの有限doubling gridのideal PF観測。step低下を回路総費用低下やground-energy全taskの利益へ読み替えない。
全配分のnumerical ambiguity recordsは0。P2の保存complete-unitary leakageは極小でも、個別Pauli stageの非保存を省略する根拠にはしない。

## 費用frontier

[resource evaluation](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/resource_evaluation.json)：1386 rows、273 eligible R frontiers、正のremaining budget0。
一次データには`7 N_Toffoli + N_RZ T_RZ(η)`のsymbolic差をη付きで保存した。`T_RZ`はUNKNOWNで、以下の数値は登録した補助価格hypothesis `ceil(κ log2(1/η))`によるもの。
baseはP2/E2/E4/D2/D4の全eligible r/配分から同model価格最小を選び、Rにも全登録r/配分とcenter theta sharingを与えた。

下表は各(T,κ)のbaseline最小とRの**known成分の総T最小**。未知function費用を含むfull costではなく、実際のT合成・compile達成値でもない。

| T | κ | baseline：方法/r/fraction | baseline T hypothesis | R known成分最小：方法/r/fraction | R known T hypothesis |
|---|---:|---|---:|---|---:|
| 1 | 3 | E2/8/.6 | 612,208 | R2/4/.8 | 19,906,056 |
| 1 | 6 | E2/8/.6 | 1,123,408 | R2/4/.8 | 36,163,544 |
| 1 | 12 | E2/8/.6 | 2,145,808 | R2/4/.8 | 68,678,520 |
| 4 | 3 | E4/8/.4 | 2,905,776 | R4/4/.4 | 102,453,120 |
| 4 | 6 | E4/8/.4 | 5,353,136 | R4/4/.4 | 185,302,656 |
| 4 | 12 | E4/8/.4 | 10,247,856 | R4/4/.4 | 351,001,728 |

全frontierで`(C_BASE - C_R_known)/r_R<0`。κ3での最大remaining per-stepでも、T1は−4,621,146（R2/8/.4）、T4は−5,253,902.5（R2/32/.4）。これはknown総T最小のR選択とは別の集計。
F_remainingはcompute/erase、zero/sign/quadrant、guard、time scale、cleanupの未会計部分で、UNKNOWNを維持した。0を実費用として代入していない。
価格・templateは保守的envelopeでありlower boundではない。負の数値frontierだけから、より安い別実装、他分子、全THRIFTに対する一般的不能を証明しない。total depth/workspace、function backend、ground-state準備・QPE統計を含む最終総費用は未評価。

## 監査と実行資源

[raw seal](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/raw_seal.json)はselection前に163 filesをSHA封印。
[独立saved audit](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/independent_audit.json)、[別CLIからのdisk replay](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/independent_checker_replay.json)は全154 PASS。
raw h/ERIから独立CARでHを再構成し、DF係数・coverage・全output rows/columns・norm witnessを検証。追加SVD/eigh/分子/PF queryは0。worst witness上端と宣言normの差3.73035e−14。
[std-lib保存整合性監査](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/saved_execution_integrity.json)は1386 rowsのeligibility/価格、273 frontierのFractionによる厳密な保存整数差/割算、source/input/raw bindings、capsを照合した。新templateや候補を生成していない。
checkerは全PF trajectoryの独立再生成ではない。全roundoffのinterval certificate・外部科学再現・CI証拠とは呼ばない。

[reference checks](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/reference_checks.json)：Padé/TaylorのFrobenius差T1=1.05070e−14、T4=3.97411e−14（tol1e−10内）。reference unitarity≤2.52965e−14、全seed unitarity≤2.56789e−13。
[execution metrics](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/execution_metrics.json)、[performance config](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/runtime_performance_config.json)。

| 実行項目 | 実績／cap |
|---|---|
| RHF/input | 1 / 1、通常SCF iterations5 / 50 |
| DF decomposition / small eigh | 1 / 1、18 / 66 |
| FCI/ground/gap eig、compile/T synthesis、shots/PR | 全0 |
| PF errors / norm SVD | 154 / 154、155 / 200（coupling1含む、saved checker0） |
| distinct seeds / dense power products | 91 / 91、770 / 2000 |
| primitive applications | 107,315 / 4,000,000 |
| resource rows | 1386 / 1386 |
| 最大common word / function angle bits | 88 / 128、40 / 128。Rのgroup input word22…31 |
| workers / thread | 2 / 1、CPU。GPU driver probe exit9でGPUなし |
| aggregate RSS sampled peak | 290,676,736 bytes / 4GiB、.2秒sample、tracker等を含む厳密連続peakではない |
| main preprocessing peak RSS | 101,339,136 bytes |
| 一回wall time | 13.1890秒、科学wall/CPU capなし |
| attempt output | 63,811,461 bytes / 64MiB。audit metadataは別に保存し、生成prefixの合計もcap内を照合 |
| 実行log | 730 bytes / 4MiB |

row/Gaussian kernels、13-h seed cache/method、binary power、2spawnを固定sourceのまま適用。D4 worker6.218秒が最長。別backendとのspeedup倍率は測っていない。
科学attemptは一回消費済み、failureなし、再試行なし。結果・input・source・原科学契約・旧124 result sets・non-result-set fieldsを保護する。既存global manifestのdescription欠落FAILは保持。

## 次の判断

[GPTへの引継ぎ](qc_mixed_access_feasibility_v1_review_handoff_20261013.md)。研究としての継続・詳細function backend投資・別用途/方向へ移るかは、新結果の重要科学判断としてGPTへ戻す。開始にはユーザーの明示承認を別に得る。
Codexは同型scan、backend/compile、H2/H6/他geometry、PR、N1/N2/N3を自動起動しない。元量子化学Hのground-energy低コスト化という目的は維持し、中心テーマの採択や研究終了を代行しない。
次の担当：GPTの科学判断開始承認待ち。Mandatory STOP。

## 公開・remote取得確認

固定証拠payload `ae3975de24bff6ab6aec3494b522851db2a8f3dd`。初回data commit `c23b3f1422fdf3956a823af6fc7197a9942194f8`へ、ignoreされていた保存済み実行logを追加して証拠索引を完成した。source・科学結果・raw seal・生成索引は変更せず、科学再実行0。[補完記録](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/publication/retrieval_precheck_failure.json)。
[GitHub固定結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae3975de24bff6ab6aec3494b522851db2a8f3dd/docs/research/qc_mixed_access_feasibility_v1_results_20261013.md)、[固定引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae3975de24bff6ab6aec3494b522851db2a8f3dd/docs/research/qc_mixed_access_feasibility_v1_review_handoff_20261013.md)、[一次resource JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae3975de24bff6ab6aec3494b522851db2a8f3dd/artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/resource_evaluation.json)。
[remote receipt](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/publication/receipt.json)：GitHub196変更blobs/65,061,432bytes、生成索引182files、raw163files、source16locks、旧優先12blobs、追加relative links48件一致。local object alternatesなし。新しい科学計算・tests・GPTレビューを公開確認のために実行していない。
これは資料の取得性・byte保全の確認。CI/外部科学再現とは異なる。次はGPT科学判断の開始承認待ち、Mandatory STOP。

## 2026-10-13 完了GPTレビューの受領・成果固定への追補

[完了GPT科学レビュー原文](../reviews/representation_exploration/qc_mixed_access_feasibility_v1_scientific_review_2026-10-13.md)、[成果固定・現系列終了](qc_mixed_access_feasibility_v1_closure_20261013.md)。ユーザーの採用・反映指示に従い、現templateの詳細function backendへの追加投資を見送り、現feasibility系列を終了する。ideal PFのstep改善と登録費用モデルの負remainingを保持し、THRIFT全般の不能へ広げない。

元source/input/result/auth/raw/STOP・原statusは不変。本追補は原文保存・文書反映のみで、新科学レビュー/計算/再集計/証拠replayは行わない。同じレビューの再実施は不要。現時点の次は、別の開始承認を得た新研究方向のGPT比較。上のレビュー開始待ち記述は当時の履歴、現在状態はclosureを参照。Mandatory STOP。
