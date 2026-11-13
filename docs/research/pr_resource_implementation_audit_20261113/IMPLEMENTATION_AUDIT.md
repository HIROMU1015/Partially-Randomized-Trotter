# PRリソース評価：実装・公開ライブラリ活用の監査結果（2026-11-13）

調査範囲は、現在使われているコードとその直接依存に限った。作業は次の3つで、コードの変更、ライブラリの導入、科学計算、実回路のcompileはしていない。

- SOURCEを静的に読む
- 人工入力・公開metadataで軽い計測をする
- Qiskitのgate定義をintrospectionで確認する（transpileはしない）

## 0. 調査開始時点の固定SOURCE・環境

| 担当 | branch | head | 主なPR資源評価module |
|---|---|---|---|
| A | `codex/pr-r-dependent-n-model-20261111` | `169d3316dd008eb98dd9be56fe4ecefc5c1a7d60` | `pr_rpe_rmse`、`pr_rpe_power_mean`、`h4_pr_m1_rmse`、`h4_pr_rmse_completion`、`pr_r_*`、`pr_rpe_rmse_{pilot,preflight,s0_v2}` |
| B | `codex/r-variable-rpe-cost-pipeline-20261111` | `3216d27a69ac8532d2a22e78f5f94491d202d238` | `pr_rpe_rmse_controller`、`pf_pr_rpe_pipeline`、`pr_rpe_rmse_{evidence,integration}`、`h4_s0_v2_intake`、`abc_s0_published_intake`、`rpe_{diagnostic,d_evidence}_intake`、`bv2_mean_price` |
| C | `codex/r-variable-compiled-price-design-20261111` | `83d219cc68c2…`（K0 PASS・E[S]4値を公開） | `mean_compiled_price`、`h4_structural_kernel`、`h4_pr_physical_closure`、`s0a_*`、`r_price_phase1{,_science,_runner}`、`r_variable_compiled_price` |
| D | `codex/d-full-pf-full-h-rte-v2-paths-20261113` ／ `codex/d-compiled-residual-c-minus-s-20261113` | `e7dca5f7…` ／ `af6f27d7…` | `full_baseline_v2_requests`、`full_pf_saved_inventory`、`full_pf_price_receipts`、`compiled_residual` |

共通の物理・回路SOURCE：
- 保存carrier：`155529e2`の`h46_long_time_{signal,native}`、`h46_new2_core`、`h46_m5_plan`
- provider：`c31ef424`の`trotterlib.rte`、`df_rte_qiskit`、`rte_compiled_cost`、`ax2b_native_df_v5`
- Full-H：`9b83b96b`の`h46_full_h_pure_rte`

研究venv（`h6-environments/20261011/venv`、Python 3.11）：

| 区分 | package |
|---|---|
| 導入済み | numpy 1.26.4、scipy 1.14.1、qiskit 1.3.0（`native-dependency-v2`にはqiskit-aer 0.15.1も）、pyscf 2.7.0、openfermion 1.6.1、mpmath 1.3.0、sympy 1.14.0、numba 0.60.0、networkx 3.6.1、rustworkx 0.18.1、pandas 2.3.3、h5py 3.16.0、pytest 8.3.5、psutil（v2のみ） |
| 未導入 | python-flint、statsmodels、hypothesis、pytest-benchmark、joblib、threadpoolctl |

## 1. PF・PR・RTEの物理計算 — 原則として現状維持

| 要素 | 現状（確認したSOURCE） | 判定 |
|---|---|---|
| Hamiltonian・DF分解 | 保存snapshot・receiptから読み込む（`h46_m5_tol_saved_input.load`等）。SCF/DFは再生成しない | **現状維持**。pyscf/openfermionで作り直す利益はなく、保存入力の同一性が研究の前提 |
| PF・PR時間発展 | `h46_long_time_signal.evolve_batch`：block別の`eigh`分解と`spectral_apply`による密行列作用。q外側ループ、最大中間normの記録 | **現状維持**。sector次元はH4 36、H6 400。round当たりの行列ベクトル積は少なく、binary64の演算順・norm記録が数値gateと結び付いている |
| finite-K RTE | `trotterlib.rte.finite_rte_distribution`（paired Taylor weight、fsum正規化）、`polynomial`（K+1次までの逐次和） | **現状維持**。小さな解析計算 |
| spectral power | Aの計画（`c76ffe5d`）：tail polynomialを固有値上でR乗し、event loopを除く。3 spectraを共有 | **現状維持**（既に適切に高速化されている） |
| reference | `scipy.sparse.linalg.expm_multiply`、保存reference | **現状維持** |
| normalization・誤差 | log Bの和、`Decimal`による二精度でのceiling、u floor・closure gate | **現状維持**。binary64への変換を避ける設計は妥当 |

## 2. RPE・RMSE評価

| 要素 | 現状 | 所見 |
|---|---|---|
| nominal shot数 | `pr_rpe_rmse.nominal_axis_shots`：`exp(2 log B)(11+4(M−m))`を`Decimal`の二精度でceilし、一致を検査 | 妥当。**現状維持** |
| decoder（scalar） | `rpe_four_round_phase_validation.reconstruct_rpe_phase`：round mごとにq_m=2^m個の分岐をPythonで列挙する（O(q)） | 単発のdecodeなら問題ない |
| decoder（vectorized） | `_vectorized_reconstruction`：`(repetitions, q_m)`の複素配列を作りargsortする | **計算量とmemoryの問題**。人工位相で実測：M=12、2000反復で1.47 s、最大複素一時配列125 MB。10⁵反復なら1 roundで約6 GB（**推定**）。Aの計画S3（全13round decoder検証）の前に、閉形式の分岐選択へ置き換えるべき（候補2） |
| 有限shotのMonte Carlo | `_marginal_branch_monte_carlo`：PCG64 Generatorによる`binomial`のvectorized生成 | 生成部は妥当。上記の再構成部がボトルネック |
| 信頼区間 | Aの計画S3：MSE標本平均にHoeffdingを使い、`n_val ≥ ⌈L_max² log(2/δ)/(2η²)⌉`。未解決率の二項区間は未定 | SciPy 1.14.1の`binomtest(...).proportion_ci(method='exact'/'wilson')`が**導入済みで利用可能**（動作確認済み）。statsmodelsは不要 |
| AST経由の科学SOURCE再利用 | `h4_pr_m1_rmse.function`（固定blobから関数をASTで抽出しexec）。`h4_pr_rmse_completion._LocalEvidenceAdapter`（`ast.NodeTransformer`で固定関数の定数tupleと比較式を書き換え、本体を途中で切る） | **一致した件数を検査しない**。patternが変わると変換が黙って適用されない（候補4） |

## 3. 量子回路・compiler・資源評価

| 要素 | 現状 | 所見 |
|---|---|---|
| 決定論PF block | `ax2b_native_df_v5.append_directional_diagonal`：CX ladderで中央RZだけをancilla制御 | 効率的。**現状維持** |
| **RTE tailの制御ZZ回転** | `df_rte_qiskit._append_central_application`：`RZZGate(2θ).control(1)`。Qiskit 1.3.0の汎用`add_control`を経由する | **重要所見**。introspection（transpileなし）で、definitionは`ccx, crz, ccx`。CCX 1個は標準定義で`cx 6、t 4、tdg 3、h 2`。中央だけを制御した等価形`cx·crz·cx`（CX 4個）と比べ、1回転当たりCX +10とT型・H由来のRZが十数個以上多い（optimization前）。PRとFull-H RTEのtail（両方とも`controlled=True`）だけに掛かり、PF blockには掛からない。C1の構造S（`Accounting`の'crzz'）は`cx·crz·cx`で数えており、gate名は同じでも分解が違う。保存C1の凍結θL（H4 18.29/event）は同じ桁（候補1） |
| 1q制御Z回転・product | `RZGate.control(1)`→`CRZGate`（rz 2、cx 2）。productは`cz` | 問題なし |
| Hadamard wrapper | `build_native_hadamard_wrapper`：h、Y軸はsdg、h、measure | 問題なし |
| compiler | `rte_compiled_cost.transpile_and_measure_cost`：`transpile(basis=rz/sx/x/cx, opt1, seed17)`。level 1は、初期のInverseCancellation→BasisTranslator→〔Optimize1qGatesDecomposition＋InverseCancellation〕の固定点反復（Qiskit 1.3.0の`builtin_plugins.py`で確認）。1q置換の基準は、targetがない場合run長 | 科学profileそのものなので**現状維持**。passを変える高速化は研究結果の意味を変える |
| compile経路の重複 | ほぼ同じ「compose→wrapper→compile→metrics」が4つ：`pf_common_run_native.cost_one`、`h46_long_time_native.cost_trial`、C1 `r_price_phase1_science.Backend.price_requests`（同一pathのS/C対）、`h46_full_h_pure_rte`（`cost_trial`経由） | 固定SOURCEは**現状維持**。新しいcompileは`Backend.price_requests`を再利用すべき（候補9） |
| 並列化・cache | 経路単位の`ProcessPoolExecutor`（fork）。compilerの`num_processes=1`で二重poolを避ける。fingerprint cache（binding検査付き） | 妥当。**現状維持** |
| 構造S counter | `h4_df10_price_v1.Accounting`と`h6_df19_price_v1.Accounting`は、ancilla wire番号（8と12）以外は同一の複製 | C1の凍結SOURCEなので**現状維持**。新規counterでは引数化する |
| 過去の関連監査 | 研究C `research_c_rustiq_interface_audit_20261024`：Qiskit 1.3.0同梱の`synth_pauli_network_rustiq`をPR controlled回路へ使う最小adapterを提案（`READY_FOR_LIMITED_IMPLEMENTATION_PROPOSAL`、未実装） | 回路合成の改善候補として既に存在。候補1と同じ「回路定義の変更」に当たり、科学判断が必要 |

## 4. 平均価格・残差評価

| 要素 | 現状 | 判定 |
|---|---|---|
| `mean_compiled_price.MeanKernel` | numpyの密なP/g、二乗法による`repeat`、mass検査 | **現状維持**。状態数は小さい |
| C `h4_structural_kernel` | `nextafter`とγ_n＝nu/(1−nu)による手書きの方向付き丸め区間（`dot_bounds`/`Band.repeat`）、longdoubleでの`matrix_power`照合、Fractionによるlaw因数分解の丸め検査。K0がPASS済み | 結果は既に出ている。**置換不要**。必要なら独立の照合としてpython-flintの`arb_mat`（ball演算）を使える（候補8、未導入） |
| `local_residual_mean` | 旧`_iid_sequence_form`（純関数）をlazy importする | 呼出し時に`trotterlib/__init__`がpyscf/qiskitを含めて全importする。純関数1個のために重いimportが走る（候補20、低） |
| D `compiled_residual` | 同一pathのD、識別可能性（Fractionによる階数）、結合規則 | 小規模で妥当。sympyへ置換する利益はない |

## 5. 共通基盤・A/B/C/D接続

| 要素 | 現状 | 所見（実測は人工・metadataのみ） |
|---|---|---|
| `pf_pr_rpe_pipeline.candidate_id` | 呼出しのたびに`scripts/h46_long_time_signal.py`をread・`ast.parse`・compileして`cid`を作る | **実測7.3 ms/回**。memo化した同じ関数では0.005 ms/回、出力は同一。B `_validate`（13round canonical config）は0.86 s/回で、その**約90%がこの関数**。`check_rpe_new_evidence_intake`はmemo化（in-process patch）で**75.1 s→16.5 s**、13件ともPASS（候補3） |
| `same()` | `json.dumps(sort_keys)`の文字列比較 | 13round config（約240 KB）で約5 ms。1と1.0を区別する厳格な仕様。**現状維持** |
| identityの埋め込み | 各roundのidentityにspec/configuration全体をcopyする（1行約14.5 KB） | Bの契約なので**現状維持**。menuが大きくなったら再検討 |
| 共有moduleの分岐 | `pf_pr_rpe_pipeline.py`がA/B/Cで3版（標準4次・H6 provider hookの差）。`full_baseline_v2_requests.py`はB版（eecaa803系）・D統合版（e7dca5f7）・Bがexecで読むD `0536646b`版の3通り。`mean_compiled_price.py`はB（旧）とC（stream_structure付き）の2版 | Cの`check_r_variable_compiled_price`は`pf_pr_rpe_pipeline.py`をC `4cb61ef4`にbyte固定しており、**B tree上では以前から1件FAIL**（候補5） |
| 固定SOURCEのAST/exec loader | 約30箇所で個別に実装（`functions`、`function`、`_constant`、`pilot_group_function`、`fixed_module`、test helper等）。git show helperも10以上のmoduleで重複 | 自由変数の検査がなく、`NameError`が実行時に初めて出る。**実例：C S0a run01はexec scopeの`ast`欠落（`NameError`）でSTOP**した（候補4） |
| D inventory | `full_pf_saved_inventory.coverage`：14.6 s中8.3 sが`saved_grid()`の繰返し（候補ごとにgit show 3回＋AST）、3.2 sがendpoint listの線形探索 | D所有。cacheとindex化で短縮できる（候補10） |
| データ検証 | 独自validatorとexact identity比較 | pydantic/jsonschemaの導入は具体的な不具合がなく、frameworkを増やすだけ。**現状維持** |
| テスト | `unittest` scriptと人工fixture。pytest 8.3.5は導入済み | **現状維持**。property testは閉形式decoderの同値性検査に限り、numpyの乱数生成で足りる（hypothesisは任意） |
| BLAS thread | 長時間runnerは`OPENBLAS_NUM_THREADS=1`等を環境変数で指定（実行中processで確認） | 妥当。threadpoolctlは不要 |

## 6. 確認範囲の限界

- 旧研究branchの全fileは監査していない（指示どおり）。`algorithm_codesign/*`・`representation_exploration/*`等の旧系統は、git helperの重複数を数えただけ。
- 実測はすべて人工入力または公開metadataに対する技術計測で、科学pipelineの実行時間ではない。10⁵反復のmemory等は**推定**。
- 制御ZZのoverheadは、gate定義（transpile前）による。compile後の正確なRZ増加量は未測定で、候補1の検証P0が必要。
