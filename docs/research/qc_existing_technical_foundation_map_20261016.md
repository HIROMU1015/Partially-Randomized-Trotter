# 新アルゴリズム研究に向けた既存技術基盤対応表

2026-10-16 JST。既存source・索引・保存検証記録の読取りだけによる技術整理。Claudeの科学レビューから独立し、候補の生成・採択・優先順位付け・科学的GO/STOP判断を行わない。

D1（block-encoding・qubitizationおよびそれらを中心とするQSVT・量子walkを新規構成対象にしない）、D2（元の量子化学Hamiltonianの絶対基底エネルギー推定）を固定する。既存CLOSED・STOP・PAUSED・HOLDは不変。SOS-PFの現探索CLOSED、K0技術判定`UNDETERMINED_DATA_REQUIRED`、A2未採択、K1-F未認可・未実施も[既存採用decision][a3]どおり保持する。

既存の[統合資料][integrated]・[API再利用索引][reuse]・[事前検証証拠索引][catalog]に情報の大部分が揃っていたため、調査を拡張せず、入出力と利用条件の接続だけを補足した。古い統合資料の「判断待ち」等を現在状態として再採用しない。

## 参照する固定点

| 記号 | Repository / 固定commit | 用途 |
| --- | --- | --- |
| P | `HIROMU1015/Partially-Randomized-Trotter` / `78b50c515d9df9556f261e74ca7c57af461b7199` | 共通ライブラリ・統合索引・保存検証記録。全独立branchの最新状態を表すcommitではない |
| H | `HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae` / `e0e82f6c7bee73d2360b77456bd22f7356572e01` | 高次PF研究repositoryの今回確認したclean HEAD。入口は[README][h-readme]。trackedなPROJECT_MAP・専用testsは見つからなかった |
| B | PR repository / `2dcd56737be35526f6436a52835f59d817adf30c` | 別branchにあるnative IR・strict-phase合成・Bernstein資源会計の参照点。実行sourceはD1報告記載の`874ca12d347f8f35e5d8d64f65f24abc4a2df523`と区別 |
| A | PR repository / science `5a903aea2492c2d211c1e15e7737df0d50694c12`、実行source `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7` | H6の別source系列。保存報告は`0dd5c3d204dab9b731268a1aa2add621080f3d20`。PのAPIと同一検証だとは扱わない |

commitはローカルGit objectから確認した参照固定点であり、今回の新しいsource freeze・GitHub独立取得・CI結果ではない。親workspaceの作業開始HEADは`e098c54c78f589055082f9cfc2b13de50c90ca94`。既存dirty checkoutをimmutable evidenceとして扱わず、別branchのmerge・checkoutはしていない。

## 機能と再利用条件

「実装済み」は静的にsource/APIを確認した意味。「限定検証」は既存資料に記録されたscopeだけを指し、今回の再検証ではない。「未実装」は当該固定API・経路にない機能、「条件未確定」は追加の入力・根拠が必要な接続を指す。

| 機能・状態 | source / 固定点 | 入力 → 出力 | 既存検証の範囲 | 再利用時の制約・不足要素 |
| --- | --- | --- | --- | --- |
| Hamiltonian・DF・固定粒子sector：実装済み／限定検証 | P: [`df_hamiltonian.py`][p-ham]（`DFHamiltonian`, `PhysicalSector`, `df_linear_operator`）、`df_partial_randomized_pf.py`（hash・分割）、`df_trotter/decompose.py` | constant・一体/二体積分・DF rank/tol・sector → signed平方DF、normal-order補正、sector作用・古典参照state | 固定入力・sectorの検査は`tests/test_df_hamiltonian.py`、後述H4/H6保存記録。個別実行のsource/input fingerprintは別管理 | 元H・定数・軌道/スピン順・粒子数・切断誤差を固定。matrix-free solverや保存stateは古典参照であり、安価な量子ground oracleではない。新候補への変換adapterと元H誤差台帳は条件未確定 |
| JW Hamiltonian・旧行列参照：実装済み／歴史的証拠 | H: [`chemistry_hamiltonian.py`][h-ham]、`matrix_pf_build.py`、`matrix_multiply.py` | H鎖サイズ・distance・既定basis → JW `QubitOperator`、HF energy、名前、qubit数。別関数で疎行列固有対 | 保存notebook・係数artifactを伴う旧高次PF研究。全入力・生成来歴の再認証は今回行っていない | `jw_hamiltonian_maker`はPySCFを呼ぶ生成関数であり、今回実行0。返すenergyはHF。`ham_ground_energy`に物理sector引数はない。Pへの同一Hamiltonian変換・sector制限は自動接続されない |
| 決定論PF係数・回路：実装済み／限定検証 | H: [`product_formula.py`][h-pf]、[`pf_decomposition.py`][h-steps]、`qiskit_time_evolution_grouping.py`／P: [`df_trotter/circuit.py`][p-df-circuit] | PF label・順序付きterm/block・time → `(index, weight)`列、Pauli/DF回路 | Hは旧研究の記録。Pの[固定5公式pilot][p-five]はtoy・exact二blockの限定検証 | label一覧を全次数・全DF・controlled検証済みとはしない。負時間・順序・隣接融合の意味論を保持。Hのgrouped回路は恒等項を省略するため絶対energyにはscalar復元が必要。Pの`energy_shift`と同一視しない |
| partial-S2・finite RTE：実装済み／限定検証 | P: [`df_partial_s2.py`][p-s2]、`df_partial_s2_repeated.py`、[`rte.py`][p-rte]、`df_rte_tail.py` | DF・partition・time・r/K・seed → hash付きpreparation、event/law、回路、有限剰余・normalization/attenuation | [partial-S2定義][p-s2-doc]、[finite-signal検証][p-signal]と対応tests。一般新分割の検証ではない | このpartial-S2 APIは二次専用。一体/constantは決定論側。ranking proxyと展開後の`exact_rte_lambda_r`を分離。RTEの入力は正規化Hermitian involution、対象は平均振幅演算子。一般DF平方をそのままinvolutionと扱わない。別候補の生成器は未実装 |
| ordinary controlled evolution・X/Y Hadamard：実装済み／限定検証 | P: `QiskitDFPartialS2CircuitBuilder`、[`rpe_hadamard_interrogation.py`][p-wrapper] | controlled trajectory・ancilla/axis指定 → 測定付きX/Y wrapper、phase/identity metadata | Pの直接wrapper APIはq=1/2/4。別系列Aの[q≤8のH6記録][a-h6]をPの一般保証へ移さない | `diag(I,U)`のancilla-relative phaseを保存。constant・tail identity・Taylor符号を別会計。wrapperは既制御回路を再制御しない。高次PF/new kernelの制御adapter、負時間・scalar・逆回路の同一性は未確認 |
| RPE誤差・shot・round会計：実装済み／条件付き限定検証 | P: [`rpe_resource_accounting.py`][p-account]、`df_rpe_resource.py` | round・PF係数/rigor区分・r/K・位相/failure配分・cost provider → feasibility、十分shots、round/総cost、条件付きenergy幅 | [会計仕様][p-account-doc]とguard tests。unit-radius参照、独立測定・fresh IID trajectory、alias-free等を仮定 | PF係数を自動認証しない。元Hのexact groundだけでは有効PFの固有state・半径1を保証しない。一般半径の認証/adapter、全round decoder・実shot実行はこのAPIに未実装。詳細条件は下記 |
| QPE分布・β解析：実装済み／歴史的モデル | H: [`qpe_beta.py`][h-qpe] | **既知**spectrum・population・T/N → Fejér分布、抽出位相、βの経験fit | 旧notebook/解析経路。β試行の再実行・現在の実行証拠確認は0 | 制御QPE回路・inverse QFT・state準備を生成しない。関数名の`ground_state_energy`は最大頻度binの位相であり、groundの自動同定や絶対energy成功保証ではない。energy/time/phase branchの変換契約が必要 |
| 高次PFの解析費用外挿：実装済み／歴史的モデル | H: [`cost_extrapolation.py`][h-cost]、`config.py`、`io_cache.py` | 保存fit係数・PF label・H鎖範囲・β/target・分解数/RZ層表 → Pauli回転数・RZ layer/T-depthのモデル外挿・図 | 旧保存係数・notebook。各artifactの生成commit・入力・fit窓の完全性は個別確認が必要 | 実native compilerや準備・全failure会計ではない。既定`CA/10`や経験βを新しいtask契約と同一視せず、古い係数を現行DFへ流用しない。H12の新係数や最終total-cost評価はここから認定しない |
| compiled RZ/CX/depth・proxy：実装済み／限定検証 | P: [`df_rpe_hadamard_compiled_cost.py`][p-compiled]、`rte_compiled_cost.py`、`rpe_hadamard_validated_proxy_provider.py`／A: `ax1b_evaluation.py` | preparation・trajectory/law・compiler・構築policy → exact/MC per-axis期待metric、paired記録、来歴 | [H4証拠索引][catalog]、別sourceの[H6確認][a-h6]。直接domain・proxy較正domain・外挿を区別 | `rz_count/cx_count/rz_depth/cx_depth/total_depth/circuit_size`。MC標本数は量子shotsではない。compiler/basis/precision/routingを固定。未観測rare-tailの母平均費用保証、長round全wrapper、状態準備込み総費用は未確定 |
| native T/1Q会計・strict-phase RZ合成：実装済み／synthetic provider限定検証 | B: [`g9_native.py`][b-native]、[`rte_reallocation/numeric.py`][b-numeric]、`rte_development/worker.py` | event → literal IR；angle・ε・tool設定 → sequence/hash/strict誤差上界；固定cache → T/CX/1Q | [D1保存報告][b-result]のexplicit 3-qubit provider・固定angle inventory。全化学DF compilerではない | `up_to_phase=False`、実sequence/随伴、ε・tool identityを保持。Tは1Qに含まれる。新angleの価格は未知、合成実行は別認可。一般DF/高次PF→このIRの接続、routing・全FT費用は未取得 |
| PF誤差・分枝診断／成功確率：実装済み／限定検証 | P: [`pf_delta_validation.py`][p-delta-src]、`finite_rte_signal_validation.py`／H: `eig_error.py`／B: [`rte_development/accounting.py`][b-account] | 固定H/state/PF又はlaw・剰余/丸め/合成予算・moment/range → 位相/分枝診断又は十分shots/resource-tail予算 | [H4 PF検証][p-delta-doc]、finite-signal、D1 syntheticの各別scope。operator/signal/eigenphase/統計を混同しない | Hの位相追跡はE・参照stateを入力に使う診断。`C_use`は実行窓の経験envelope。Bは契約をliteralに固定した会計で汎用solverではない。元H ground branch・有用な有限step certificate・全task failureの接続は条件未確定 |
| 実行・cache・計算資源管理：実装済み／限定運用検証 | P: [`parallel_validation_executor.py`][p-executor]、`io_cache.py`、`df_gpu_statevector.py`／H: `config.py`, `io_cache.py` | source/input hash・task/seed・compiler・memory/worker/GPU指定 → deterministic task ID、checkpoint/resume、資源guard | [executor仕様][p-executor-doc]と保存H4運用検査、H6のcache運用記録。科学的優位の検査とは別 | GPUは既存kernelを呼ぶjob割当で、単一stateの多GPU分散は未実装。CPU/RAM/GPU空き・依存版・実backendは将来実行前に確認。保存host値を現在の専有資源としない。古い既定wall制限は将来の新契約（phase/total capなし）と整合させる |

## 検証scopeと証拠の読み分け

- **H4 PF/signal参照**：linear H4、1.00 Å、STO-3G、DF rank12の固定snapshot。PF分枝診断は`L_D=0…11`、δ=`0.0125/0.025/0.05/0.1/0.2/0.4`、q=`1/2/4`、中央tailはsector exact exponential。finite-RTEやcompiled総費用を含む結果ではない。finite-RTE/回路側は別仕様・別入力hashを確認する。
- **H4 compiled索引**：同索引記載の代表linear H4/1 Å/STO-3G/DF12、主`L_D=3/12`、partial-S2、δ=`0.01/0.02`中心の登録窓。q≤32の一部holdoutやopt1/opt2を全cell・全metricへ拡張しない。これはH4 geometry別系列やH6との共通snapshotを意味しない。
- **H6別source**：linear H6/1 Å/STO-3G、tol-only `1e-8`・rank19・cutoff0、Nα=Nβ=3/sector400、generation順prefix5、partial-S2、T=`0.8`、q=`1/2/4/8`、δ=`0.8/0.4/0.2/0.1`、r=`4/4/4/2`、K=`2/4`。固定8構成の32独立trajectory、Qiskit1.3/opt1/rz,sx,x,cx/no routing、準備費用P=0の**complex-signal**資源評価。母平均winner・絶対ground-energy taskではない。
- **B native会計**：explicit 3-qubit involution、p=`(1/5,3/10,1/2)`、τ=`5/7`、state000、P5、r=`1/2`、axis ε=`1/200`、family failure=`1/20`、primitive strict ε=`1e-6`。geometry/basis/DF/L_Dは非該当。Bernstein十分予算・加法native価格であり、実shots・分子全回路の結果ではない。

Pの[`VALIDATION_STATUS.md`][p-status]と[`validation_manifest.json`][p-manifest]は個別result-setのstatus・来歴を優先する。`legacy_high_order_pf=historical_evidence`、旧DF screening=`invalidated_pending_regeneration`、UWC=`prose_only_missing_raw_artifacts`を保持する。局所source/testsの存在はimmutable CIや外部再現を意味しない。既知global manifest FAILを修正・再判定していない。

## 候補提示後に必要になる接続情報（実装・実行計画ではない）

1. **入力同一性**：元Hのscalar/核反発・normal-order補正、積分/DF表現、sector、rank/tol・誤差、軌道順、hashを渡す。両repoのPython package名は同じ`trotterlib`なので、環境とimport解決先を固定する。
2. **evolution契約**：time符号・適用順、`diag(I,U)`、scalar phase、ancilla、制御/随伴、固定kernelのpowersかround別近似か、random draw/weight/normalizationを明示する。既存partial-S2へ一般kernelを無条件に渡せない。
3. **絶対energyと成功確率**：位相cutと一意unwrapping区間、ground branch/overlap・半径の根拠、PF/表現/合成biasと統計/family failureを分離する。Pの`certified`は、外部のrigorous PF入力、unit-radius等の仮定、strict `β_RPE<π/3`、独立bounded測定、毎shot fresh IID RTE、切断誤差会計・全round union条件を満たす場合だけ。原Hのexact groundを与えることだけでは満たさない。
4. **費用と不足部分**：準備・情報取得、制御全wrapper、必要全round、rotation合成精度、量子shots、古典前処理/online、memory/depth/FT費用を別々に接続する。RZ外挿、MC費用点推定、Bernstein十分T予算の間に自動換算はない。全項目を閉じた元Hの絶対ground-energy推定pipelineは、今回確認した固定経路では未完成。

sourceの読取り以外の実装・数値テスト・科学計算・保存数値の再集計・compile/T合成・文献追加調査はすべて0件。新規ローカル文書は本表だけ。既存source/results/authorization/STOP/manifestを変更せず、commit・pushなし。既存資料で今回の技術確認に必要な情報は揃ったため、追加探索せず**Mandatory STOP**。Claudeレビュー終了を待った自動K0/K1開始は行わない。

[integrated]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md
[reuse]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/artifacts/qc_algorithm_candidates/2026-10-13/reuse_and_future_validation.json
[catalog]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/prevalidation_catalog_evidence_map.md
[a3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dbe51f1870cd069b952cc92593833d355da10e28/artifacts/qc_sos_pf_a3_state_adoption/2026-10-16/decision.json
[h-readme]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/e0e82f6c7bee73d2360b77456bd22f7356572e01/README.md
[h-ham]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/e0e82f6c7bee73d2360b77456bd22f7356572e01/src/trotterlib/chemistry_hamiltonian.py
[h-pf]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/e0e82f6c7bee73d2360b77456bd22f7356572e01/src/trotterlib/product_formula.py
[h-steps]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/e0e82f6c7bee73d2360b77456bd22f7356572e01/src/trotterlib/pf_decomposition.py
[h-qpe]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/e0e82f6c7bee73d2360b77456bd22f7356572e01/src/trotterlib/qpe_beta.py
[h-cost]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/e0e82f6c7bee73d2360b77456bd22f7356572e01/src/trotterlib/cost_extrapolation.py
[p-ham]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/src/trotterlib/df_hamiltonian.py
[p-df-circuit]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/src/trotterlib/df_trotter/circuit.py
[p-five]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research_direction_energy_tail_pareto.md
[p-s2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/src/trotterlib/df_partial_s2.py
[p-rte]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/src/trotterlib/rte.py
[p-s2-doc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/df_partial_s2_compiled_cost.md
[p-signal]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/finite_rte_signal_validation.md
[p-wrapper]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/src/trotterlib/rpe_hadamard_interrogation.py
[p-account]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/src/trotterlib/rpe_resource_accounting.py
[p-account-doc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/rpe_resource_accounting.md
[p-compiled]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/src/trotterlib/df_rpe_hadamard_compiled_cost.py
[a-h6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/docs/research/track_a_h6_independent_confirmation_result_v1.md
[b-native]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/src/trottertracks/algorithm_codesign/g9_native.py
[b-numeric]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/src/trottertracks/algorithm_codesign/rte_reallocation/numeric.py
[b-account]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/src/trottertracks/algorithm_codesign/rte_development/accounting.py
[b-result]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md
[p-delta-src]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/src/trotterlib/pf_delta_validation.py
[p-delta-doc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/pf_delta_validation.md
[p-executor]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/src/trotterlib/parallel_validation_executor.py
[p-executor-doc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/server_parallel_validation_execution.md
[p-status]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/VALIDATION_STATUS.md
[p-manifest]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/artifacts/validation_manifest.json
