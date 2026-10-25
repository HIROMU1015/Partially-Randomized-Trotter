# 研究B：Outer Splitting 22行限定検証の実行準備・固定契約

2026-10-25 JST。**実装・人工検証・read-only preflight完了。今回の22行の科学計算は未実行。科学実行には研究Bチャットでの別途明示許可が必要。公開確認後mandatory STOP。**

## 1. 起点と科学SOURCE

作業起点は[限定設計の公開commit df5c97d](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/df5c97dfeb1a8e6d2d49ef71588640febf2f5837/docs/research/research_b_outer_rte_same_accuracy_cost_design_20261025.md)。独立worktree・branch `codex/research-b-outer-rte22-ready-20261025` で準備した。

今回固定する**科学SOURCEは `26e8734aff3969dc8077cb058826d12394be28e1`**。[実装・人工テストのcommit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/26e8734aff3969dc8077cb058826d12394be28e1)を先に固定し、本書と必要な索引だけを後続文書commitに置く。科学実行には文書commitのHEADを使用しない。

旧16行の科学SOURCE `b85dd328423de5b1159f11939835f04db90f3187`、[結果公開 b1d9f46](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b1d9f46ee583d86666deb433a4fc4c20dad20225/docs/research/research_b_outer_splitting_stage1_execution_20261025.md)、旧36行のSOURCE・契約・結果・grant／ledger／marker／STOPは保全する。今回、保存済みCSV／raw数値を開いて再解析していない。共通運用指示・AGENTS.md・他研究の規則を変更していない。

## 2. 固定22行と共通条件

各T=+0.18、−0.18について下記11条件、計22行。並びは正Tの表順11行、負Tの表順11行。

| ID | Outer | D内部 | q | f | r_RTE | K | R occurrence／行 |
|---|---|---|---:|---:|---:|---:|---:|
| B1 | S2 | A2 | 2 | — | 2 | 2 | 2 |
| B2 | Refined-S2 | A2 | 2 | 2 | 2 | 2 | 4 |
| B3 | Yoshida4 | A2 | 2 | — | 2 | 2 | 6 |
| B4 | SABA2 | A2 | 2 | — | 2 | 2 | 4 |
| D1 | Yoshida4 | A3 | 2 | — | 2 | 2 | 6 |
| R1 | Yoshida4 | A2 | 2 | — | 4 | 2 | 6 |
| R2 | Yoshida4 | A3 | 2 | — | 4 | 2 | 6 |
| R3 | Yoshida4 | A2 | 2 | — | 2 | 4 | 6 |
| R4 | Yoshida4 | A3 | 2 | — | 2 | 4 | 6 |
| S1 | SABA2 | A2 | 4 | — | 2 | 2 | 8 |
| F1 | Refined-S2 | A2 | 2 | 8 | 2 | 2 | 16 |

Synthetic 2-mode、one-body＋3 DF因子、N=1、mode0 occupation、L_D=2 weight-ranked prefix、D内部n_D=1。A2は既存Yoshida4、A3はpublished Morales Y8m10b。分子geometry／AO basis／分子DF rank探索は非該当。既存scalar／identity分離とDF factor選択を維持する。

Complex128、correctness atol=1e-12、phase診断無効、analytic moments、独立RTE occurrences、samplingなし、記述的診断のみ。q=2のmacro幅は±0.09、q=4は±0.045。F1はqf=16実stepで、別のphysical時間窓を追加しない。

入力recipeは既存 `InputRecipe`：fixture commit `52ba88687f92d089921b207e60aec006f04141de`、blob `0c92ae9c1e8c5721d9c8629a83435b855f5d6043`、`tests/test_df_partial_s2_repeated.py::_nontrivial_basis_case`。Preflightでは固定bytes・AST literalの参照のみで、fixture関数やHamiltonian constructorを呼ばない。実際のH／D／R／sector／state identityは次の許可後、登録Input packageで解決・数値検査する。

追加のr/K=(4,4)、f=3/4、n_D=2、A4、別T・sector・state・fixture、係数探索、stage別r/K最適化は含めない。未達でも集合を自動拡張しない。

## 3. 再利用と最小拡張

- `outer.py::build_outer_graph` にcaseと整列するr/K、candidate ID、analytic診断の任意引数を追加。既定の16行graphは変更しない。
- 新しい `outer_rte_execution.py` は上表・22行scope・数値予算metadataを固定し、既存 `outer_execution.py` のSOURCE／input binding・scope照合・read-only preflightを再利用する。
- `OuterDenseBackend` を拡張し、既存 `make_rte_config`／`finite_rte_operator_moments`、DF Input package、A2/A3列、exp cache、数学gate、5対象signalを継続使用する。
- Boundは既存 `rte_occurrence_truncation_from_config` と `compose_truncation_residual_bounds` を使用。前者内部のstep／occurrence boundを再実装しない。新規wrapperはその記録・DAG接続だけ。
- `launcher.py` の保存・排他的mkdir・失敗処理・STOPをそのまま使用し、固定22行を選ぶ入口だけを追加。新CLIは `scripts/run_b2d_outer_rte22.py`。`--inspect` はmetadataのみ、科学実行は `--execute --source-commit … --output …`。

旧36行・16行の条件を新集合へ置換せず、専用namespaceと別SOURCEに分離した。新規科学backend、authorization／grant／監査framework、承認証明資料は作らない。

## 4. r／K・signed stage・normalization・bound

r_RTEは時間分割数、Kは非負偶数finite-Taylor cutoff。K=2/4で通常の保持多項式は3/5次、paired event次数は0,2／0,2,4。明示Kを渡し、既存 `truncation_tolerance=1.0` はlawのmetadataとして保持する。これはphysical signalの目標精度ではない。

各caseのr/Kを全R stageに渡す。Yoshida4中央Rの負時間を保持し、Tの符号反転後も各stageの符号を保つ。負時間finite meanを正時間meanのinverseに置換しない。既存event phase・scalar phase・controlled relative phaseの規約を保持するが、新しいcontrolled回路は生成しない。

MomentのDAG identityはtail／Hamiltonian／入力array identity、signed duration、r、K、tolerance、seed、precision、analytic lawを含む。Seedは既存41＋occurrence位置のmetadataであり、乱数は引かない。実入力へのbinding後も同じscopeを照合し、backendでsigned configを再確認する。

22行のR occurrence参照は計140。共有後のmoment nodeは96だが、ordered mean、N_j、Taylor boundのoccurrence列は省略しない。例えば同じYoshida4のA2/A3でmomentを共有しても、各行の6 occurrenceは残る。`rte_stages`／boundの `append_occurrence` は、その行のplan.bodyからRを抜き出したappend順の位置である。数学的積順はappend順の逆。Config・event distribution・N_j・attenuationを保存し、全体Nは各occurrenceの積で計算する。

全行で `M_corr=(∏N_j) M_raw` を検査する。Raw X/Y／半径／複素signalとcorrected meanを区別し、Nの増減からshot数・variance・native費用を認定しない。

Bound適用の理論前提は、`R=λ_R H̄_R`、H̄_Rがsigned Hermitian involutionの凸結合で `||H̄_R||≤1`、D因子がunitary、同一ordered productであること。入力Hermiticityと追加tail norm gate（許容1e-12）を先行する。既存step boundをρ_jとすると、各occurrenceは `(1+ρ_j)^r_j−1`、全体は `∏_j(1+ρ_j)^r_j−1`。Repeated occurrenceをcount=1の反復列として既存合成関数へ渡す。これはfinite corrected対ideal-tailのoperator／共通正規化stateのsignalに対する**nominal analytic bound**で、physicalとの全biasでも、浮動小数点誤差を含むcertificateでもない。Norm gateのatolを厳密な数値保証へ転用しない。

## 5. Metadataから固定した科学DAG・scope

実Hamiltonian、expm、RTE moments、signal、bound、Nを計算せずにgraphを展開した。Scope schemaは `b2d_outer_rte22_scope_v1`。正規化layout fingerprintは `1ca646f0d954b8ceeffa47f8ddfd609d1e32a1a0439391b4158a18a0a8121e8c`。実入力identityは別途binding・correctness検査する。

| 境界 | 固定値 |
|---|---:|
| 比較行 | 22 |
| DAG dispatch | 1,239 |
| Unique exponential key／backend.expm | 274 |
| RTE config／moments／truncation record | 各96 |
| Bound composition | 14 |
| Input package高位境界 | 10 |
| Signal expectation | 110（5対象×22行） |
| D数学gate | 36 unique node |
| Outer数学順gate | 12 unique node |
| Signed RTE config／normalization gate | 96 unique node |
| Full raw／corrected rescaling gate | 22 |
| Input／weak診断／tail norm gate | 各1 |

全dispatch内訳：exponential 274、moment_field 192、operator_error 128、outer_bias_qualification 22、outer_d_check 36、outer_diagnostics 1、outer_normalization 14、outer_order_check 12、outer_rescale_check 22、outer_rte_check 96、outer_rte_moments 96、outer_rte_records 14、outer_signals 22、outer_tail_bound_check 1、outer_taylor_bound 96、outer_taylor_total 14、physical_scalar_phase 70、product_append 106、rescale_raw 22、validate_input 1。

同一R列のnormalization／bound合成等も共有するため、14 nodeでも全22行へ出力される。Bound内のstep計算・Taylor／normalizationの内部処理は既存高位境界内に含む。親子・入れ子境界を加算した値を独立科学call総数と呼ばない。旧906／490／80は新scopeへ転用していない。

入力境界はfixture／split／preparation／sector／full各1、block3、tail／state各1。既存Input package内部のGaussian basis準備・tail基底／sector変換を含む。将来の許可後にだけ実行し、native full PF wrapperのbuild／compileは含めない。

## 6. Correctnessの実行順とSTOP

将来の科学実行は次の順序とする。

1. Fresh outputを排他的に作成し、SOURCE・scope・recipe・環境をbindingする。
2. 固定Input packageを一度生成し、H／D/R分割／sector／state identity、Hermiticity、state norm、必要な非可換性を検査する。
3. Weak-residual診断とtail norm前提を記録し、A2/A3全D列を独立数学的積と照合する。
4. Exact-D/R outer全列をcaseから独立構築した数学的作用順と照合する。
5. 全signed RTE config・N_j・attenuationを照合し、全行のraw／corrected rescalingを検査する。
6. **共有後169個の全gate nodeがPASSした後だけ**、operator／signal比較とbias予算診断を実施し、登録scope完遂を確認する。
7. 成功・correctness FAIL・INCONCLUSIVE・例外・保存失敗の全経路で結果／部分結果・理由・実績・STOPを保存して終了する。

科学的未達・改善なし・識別不能を実装不具合と同一視しない。r/K、精度、候補集合、Tを自動変更しない。Retry／resume・上書きは禁止。別attempt、追加grid、native費用・shot評価、Stage 2を自動開始しない。

## 7. ε・u_num・confidence

対象は同一physical時間の複素coherent signal `z_phys=<ψ|exp(-iHT)|ψ>` の絶対誤差で、energy／RPE位相精度とは区別する。ε={1e-8,1e-9}を固定する。

| 項目 | 規則 |
|---|---|
| Algorithmic bias b | `|z_corr−z_phys|` |
| Deterministic予算 | `b+u_num≤ε/2` |
| 実装・合成誤差γ | ε/10、未取得 |
| 統計誤差s | 2ε/5、未取得 |
| Confidence | 95%、α=0.05 |
| X/Y配分 | 共通のequal union allocation、α_X=α_Y=0.025／candidate・T |
| 複数candidate・両Tの同時主張 | familywise budget未配分、同時保証なし |

現backendは**u_numのqualified評価器を持たない**。したがって `u_num=null`、`deterministic_qualified=null`、`UNRESOLVED_NUMERICAL_UNCERTAINTY` を両εに保存する。`observed_bias_within_budget` は観測bとε/2の大小だけで、適格性の保証ではない。仮にb=0でもu_num=0とはしない。Correctness atol=1e-12をu_numへ代入しない。丸め誤差水準・予算境界付近も保証未確定として報告する。

γ、s、shot数はnullのまま。End-to-end statusは `NOT_CERTIFIED_NO_TOTAL_RESOURCE_CLAIM`。Analytic biasが小さくても同精度達成・総資源最小・科学的GOを認定しない。Familywise confidenceやu_numを解決する追加計算は今回準備にも次の固定22行にも自動追加しない。

## 8. 保存する指標・出力保全

5対象はexact physical、exact-D/R outer、近似D＋ideal-tail、finite corrected、raw mean。各行にphysicalとのoperator差、corrected対outer、ideal対outerのD近似差、corrected対idealのfinite-RTE差、5対象のX/Y・半径・複素signal直接差を保存する。Phase診断は行わない。

さらにstage別config／event distribution／N_jと全体N、attenuation、Taylor residual記録・合成bound、R occurrence／short-step数、D run／kernel反復／block指数数、D/R signed総時間・絶対時間、弱摂動commutator診断、εごとの未確定なbias適格性を保存する。これは構造指標でありnative wrapper期待費用ではない。

新namespaceは `artifacts/research_b_outer_rte22/`。候補outputは末尾commandに記す。**今回namespace・科学outputは未生成。** 過去directory・STOP・原本を流用しない。保存は既存9ファイル：binding.json、inputs.json、inputs.npz、graph.json、events.jsonl、partial_results.json、result.json、status.json、STOP.json。Bindingに科学SOURCE・scope・環境version・入力recipe、graph/resultに解決後identity・精度・条件を保持する。

既存出力枠を継続：総16 MiB、JSON各1 MiB、event各64 KiB、9ファイル、STOP reserve256 KiB。新graphの未解決metadata JSONは383,745 bytes、既存枠内。最大event数は3,902。実績は各登録境界と照合し、失敗時は完了済み行・gate記録・call履歴を残す。これらは出力／call制御であり、CPU・RAM・wall-time上限ではない。

## 9. 人工テストとread-only preflight

**既存105＋追加14＝119件PASS**。ローカル人工検証で、CI・独立外部再現・科学的correctness／性能の証拠ではない。新テストは `tests/management/test_b2d_outer_rte22.py`。22行表、candidate別r/K、S1/F1のstage数、負時間、cacheのr/K・入力binding、normalization重複因子、DAG変異拒否、mock全登録実績、gate失敗で比較0、部分保存、INCONCLUSIVE、保存失敗、fresh衝突、retry拒否、CLI誤起動防止、u_num未確定、既存bound関数へのstub接続を確認した。

```bash
/usr/bin/python3.11 -I -B -S tests/management/test_b2d_outer_rte22.py -v
```

同じ `-I -B -S` で既存8ファイルも個別実行：adapter15、evaluator14、input_preflight16、stage1_launcher15、execution_ready4、simple_execution8、outer_preparation19、outer_execution14。計105件PASS。旧16行906／490／80と旧36行941／448のmetadata／実行mock回帰を維持した。CPython3.11.0rc1、標準ライブラリ・opaque label／callbackのみ。Science import barrier下で実Hamiltonian／operator／RTE moments／signal／Gaussian circuit生成・native build・shot simulation・保存値の再解析を実施していない。

科学SOURCE commit後、独立detached checkout `/tmp/prt-research-b-outer-rte22-science-20261025` のHEAD・clean状態、fixture bytes参照、22行とscope、依存availability／version、command引数、fresh output非存在を**read-only metadata preflightでPASS**した。科学packageはspec／version参照のみを許し、実module loaderを禁止した。科学module load 0、科学output作成0、checkoutは確認前後ともclean。科学constructorは呼んでいない。

予定環境は既存 `/tmp/pr-phase2df1pre-private-20261023/venv/bin/python`、CPython3.11.9。NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、Qiskit Nature0.7.2、OpenFermion1.6.1、OpenFermion-PySCF0.5、PySCF2.7.0、Numba0.60.0、Matplotlib3.9.2。共有環境のinstall／設定変更なし。実行時にもavailabilityを確認し、依存versionをprovenanceへ記録する。不要なbitwise環境固定は加えない。

32 CPU、空きRAM約52.7 GiBを確認した。対象は2×2 sectorで、既存vectorized演算・274 exponential cache・96 moment共有が主要な再利用経路。小行列ごとのprocess分割は追加せず、人工的なCPU／RAM／worker／wall-time上限も設けない。将来の実行直前に空き資源・共有環境非干渉を再確認する。科学速度倍率は未測定。

## 10. 許可後にCodexが使う単一command

ユーザーが研究Bチャットで**この固定SOURCE・22行条件の科学実行を明示的に許可した後だけ**、Codexが以下を一度起動する。今回は起動していない。`--execute` は誤起動防止であり、ユーザー許可の証明ではない。承認原文ファイル、certificate／receipt／grant／packet、端末での特別な承認操作は不要。

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-research-b-outer-rte22-science-20261025/scripts/run_b2d_outer_rte22.py \
  --execute \
  --source-commit 26e8734aff3969dc8077cb058826d12394be28e1 \
  --output /tmp/prt-research-b-outer-rte22-science-20261025/artifacts/research_b_outer_rte22/20261025/analytic-once
```

今回の成果は実行準備である。新22行の入力／数学／signed RTE／rescalingの数値correctness、boundと実bias・Nの値、complex128での識別力、εへの適格性・方式間改善は未確認。Native費用・γ・必要shot数・finite-bit certificate・RPE総資源・Stage 2も未取得。今回の公開・remote照合後はmandatory STOPし、別途許可が来るまで新しい科学計算を開始しない。
