# 改善候補一覧（2026-11-13）

根拠は[IMPLEMENTATION_AUDIT](IMPLEMENTATION_AUDIT.md)。数値は次の2種類に分けて書く。

- **実測**：人工入力・公開metadataに対する技術計測。
- **推定**：実測からの外挿。

優先度は、研究目的への寄与・計算量の削減・実装の複雑さ・科学的リスクを総合して決めた。行数の削減や形式的な共通化だけの候補は、優先度を上げていない。

## 一覧

| # | 対象 | 判定 | 優先度 | 規模 |
|---:|---|---|---|---|
| 1 | RTE tailの制御ZZ回転（汎用control→CCX·CRZ·CCX） | 追加調査→科学判断 | **高** | 実装は小、価格の再取得は大 |
| 2 | RPE decoderの分岐選択（O(q)配列→閉形式） | 変更推奨（新関数＋同値性test） | **高**（S3の前提） | 小〜中 |
| 3 | `candidate_id`の毎回AST parse | 変更推奨 | **高** | 小 |
| 4 | 固定SOURCEのAST/exec loader（自由変数の検査、変換件数の検査） | 変更推奨（新規codeから） | 中〜高 | 小 |
| 5 | 共有moduleの3版分岐と、既存のbyte固定test FAIL | 変更推奨（所有・固定方針） | 中 | 小〜中 |
| 6 | S3の二項区間・未解決率 | 変更推奨（SciPy既存APIを使用） | 中 | 小 |
| 7 | S3のMSE信頼区間の方式 | 追加調査（ユーザー・Aの判断） | 低〜中 | 小 |
| 8 | C K1の区間演算の独立照合 | 追加調査（任意） | 低 | 中 |
| 9 | compile経路の4重複 | 新規作業では再利用（固定SOURCEは現状維持） | 中 | 小 |
| 10 | D inventoryの繰返しgit/AST | 変更推奨（D所有） | 低 | 小 |
| 11 | git show helperの重複 | 現状維持 | 低 | – |
| 12 | identityの埋め込み・`same()` | 現状維持 | 低 | – |
| 13 | H4/H6 Accountingの複製 | 現状維持（新規counterは引数化） | 低 | – |
| 14 | 物理計算の核（eigh/spectral、finite RTE、expm_multiply、Decimal） | 現状維持 | – | – |
| 15 | 大round（m6以上）のcompile性能 | 追加調査（P1実測の後） | 中 | – |
| 16 | 構造S計数のJIT（numba） | 現状維持 | 低 | – |
| 17 | データ検証library（pydantic/jsonschema） | 現状維持 | 低 | – |
| 18 | BLAS thread制御（threadpoolctl） | 現状維持 | 低 | – |
| 19 | property-based test（hypothesis） | 任意 | 低 | – |
| 20 | `local_residual_mean`の重いpackage import | 現状維持（必要時に軽量化） | 低 | 小 |

## 詳細

### 1. RTE tailの制御ZZ回転（最重要）
- **対象**：provider `c31ef424`の`df_rte_qiskit._append_central_application`（`RZZGate(2θ).control(1)`）。PR・Full-H RTEのcontrolled tailすべて（`h46_long_time_native.compose`、`h46_full_h_pure_rte`）。
- **現状**：Qiskit 1.3.0の`RZZGate.control`は汎用`add_control`へ委ねる。そのdefinitionは`ccx, crz, ccx`（introspectionで確認）。CCX 1個は`cx 6, t 4, tdg 3, h 2`を含む。
  - 等価で安価な形は`cx(b,c)·crz(a,c)·cx(b,c)`（中央だけ制御）。決定論PF blockは既にこの方式（CX ladder＋中央RZだけ制御）を使っている。
  - C1の構造S（`Accounting`の`'crzz'`）は安価な形で数えており、gate名が同じでも分解が違う。
- **代替案**：ZZ支持のrotationを`cx·crz·cx`で生成する。Qiskit既存gateで足り、新libraryは不要。過去監査の`synth_pauli_network_rustiq`（Qiskit 1.3.0同梱）による網羅的な合成も候補になる。
- **効果（推定）**：ZZ rotation 1回当たり、optimization前でCX −10、RZ −十数個以上。PR・Full-H RTEのcompiled価格が下がり、PFとの比較条件（制御実装の効率）が揃う。保存C1の凍結θL（H4 18.29/event）は同じ桁だが、これは検証ではない。
- **根拠**：provider SOURCE、Qiskit 1.3.0の`rzz.py`／`add_control.py`／CCX定義（venv内のSOURCEとintrospection）。
- **リスク**：compiled価格Cの定義そのものが変わる。過去の全価格（C1、long-time、Full-H、K1との対応）と比較できなくなり、旧結果は旧SOURCEとして保全が必要。演算子の同値性（controlled-phaseを含む）を検証する必要がある。
- **検証方法**：(a) 小回路で、2つの形のunitary同値性（statevector）とcompile後のRZ/CX数を比較する。(b) 同値なら、新SOURCEで代表round（m0、m6）のpaired S/Cを取り直す。いずれも実行承認が必要。
- **作業規模**：コード変更は小。価格の再取得と比較の再構築は大。
- **優先度**：**高**。研究目的である公平な資源比較に直接影響する。
- **判定**：**追加調査→ユーザーの科学判断**。自動で変更しない。

### 2. RPE decoderの分岐選択
- **対象**：`trotterlib.rpe_four_round_phase_validation._vectorized_reconstruction`（S3の全13round decoder検証で再利用予定）。scalar版`reconstruct_rpe_phase`。
- **現状**：roundごとに`(反復, q_m)`の候補配列を作り、`exp`・`angle`・`argsort`を計算する。O(反復·q)のmemoryと、O(反復·q log q)の時間がかかる。
- **代替案**：候補は円周上に等間隔（2π/q）に並ぶので、k=⌊(q·e−φ)/2π⌋の近傍だけを評価すれば最近傍が決まる（O(1)/round）。元の`_wrap_phase`・距離式・ambiguous判定（1e−12）をそのまま使う。新libraryは不要。
- **効果（実測）**：人工位相、2000反復で次のとおり。分岐・推定値は全件一致。
  - M=8：14倍
  - M=12：1.47 s→0.008 s（**180倍**）。既存の最大一時配列は125 MB/round。
- **効果（推定）**：10⁵反復なら既存は約6 GB/roundの一時配列が必要で、新方式は数MB。
- **根拠**：[benchmark script](bench_decoder_artificial.py)と上の実測。
- **リスク**：境界（ちょうど中間、wrap境界、1e−12近傍のtie）で、argsortと選択順が違う可能性がある。元decoderは科学条件の一部である。
- **検証方法**：既存実装を参照として残し、乱数位相・意図的なtie・wrap境界・小q全探索で、分岐・推定・ambiguousの完全一致を人工testで確認する。科学実行は不要。
- **作業規模**：小〜中。
- **優先度**：**高**（AのS3が現実的なmemoryで動くための前提）。
- **判定**：**変更推奨**（新関数として追加。元関数は変更しない）。

### 3. `candidate_id`の毎回AST parse
- **対象**：`pf_pr_rpe_pipeline.candidate_id`（A/B/Cの各版）。呼出し元はB `generate/_validate`、A `pr_rpe_rmse.schedule`、D inventoryなど。
- **現状**：呼出しのたびに`scripts/h46_long_time_signal.py`の読込・parse・compileを行う。
- **代替案**：抽出した`cid`関数を`functools.lru_cache`で一度だけ作る。標準libraryだけで足りる。必要なら、固定blobのSHAとの照合を一度だけ行う。
- **効果（実測）**：
  - 1回当たり7.3 ms→0.005 ms。
  - B `_validate`は0.86 s/回で、うち約0.78 sがこの関数。
  - `check_rpe_new_evidence_intake`は75.1 s→16.5 s（in-process patch、13件PASS）。
- **リスク**：ほぼない。同じ関数object・同じ出力。process内で`h46_long_time_signal.py`が書き換わる場合を除く（その場合は以前も不定）。
- **検証方法**：既存B/A/C/Dのtest一式。cacheあり・なしで、人工candidate集合のIDが一致すること。
- **作業規模**：小（数行）。**優先度**：**高**。**判定**：**変更推奨**（共通APIの所有者であるBで実施し、A/Cへ配布）。

### 4. 固定SOURCEのAST/exec loader
- **対象**：約30箇所の「固定blobから関数を抽出してexecする」実装。
  - 主なもの：`h4_pr_m1_rmse.function`、`r_price_phase1_science.functions`、`D._constant`、`compiled_residual.pilot_group_function`、test helper。
  - A `h4_pr_rmse_completion._LocalEvidenceAdapter`（`NodeTransformer`による固定関数の書換え）。
- **現状**：
  - exec scopeの依存を事前に検査しない。C S0a run01は`NameError: name 'ast' is not defined`で、科学処理の前にSTOPした。
  - `NodeTransformer`は、patternが一致しなくても黙って何も変えない。
- **代替案**：新規code向けに小さな共通helperを作る。標準libraryだけで足りる。
  - `ast`と`symtable`で関数の自由変数を列挙し、scopeに無ければexec前に失敗させる。
  - 変換が何件一致したかを数え、期待件数（例：tuple 1件、比較1件）と違えば失敗させる。
  - 読込みは`lru_cache`で1回にする。
- **効果**：実行時の設定失敗とsilent no-opを、metadata段階で検出できる。
- **リスク**：既存の固定SOURCEを変えなければ小さい。
- **検証方法**：既知の欠落例（S0a run01と同じ欠落）の人工再現で事前検出されること。既存loaderの出力と一致すること。
- **作業規模**：小。**優先度**：中〜高。**判定**：**変更推奨（新規codeから）**。実行済みの固定SOURCEは変えない。

### 5. 共有moduleの分岐とbyte固定test
- **対象**：`pf_pr_rpe_pipeline.py`（A/B/Cで3版）、`full_baseline_v2_requests.py`（B版・D統合版・BがexecするD `0536646b`版）、`mean_compiled_price.py`（B旧版・C新版）。Cの`check_r_variable_compiled_price.test_original_science_modules_unchanged`。
- **現状**：
  - 差分は小さい（標準4次の対応、H6 provider hook、stream_structure）。各担当が別commitを固定している。
  - C testはC `4cb61ef4`へbyte固定しているため、B tree上で**既存FAIL**になる（D監査で確認、本branchの変更とは無関係）。
- **代替案**：所有と固定の方針を決める。
  - 共通moduleの正本はBとし、A/C/DはB commitを固定参照する。
  - D統合版はB版と同一のV2 IDを生成することを確認済みなので、次のB統合で採用できる。
  - byte固定testは「固定commitのblobとの比較」へ書き換える。
- **効果**：保守性の向上、誤検出FAILの解消、版の取り違え防止。
- **リスク**：統合の際、各担当の固定結果が参照するcommitを変えないこと（新commitで行う）。
- **検証方法**：統合後、全担当のtest一式と、既存固定結果の再現照合（metadataのみ）。
- **作業規模**：小〜中。**優先度**：中。**判定**：**変更推奨**（Bが統合時に実施）。

### 6. S3の二項区間・未解決率
- **対象**：Aの計画S3（未解決trial率、axisごとの二項区間）。
- **代替案**：導入済みSciPy 1.14.1の`scipy.stats.binomtest(k, n).proportion_ci(confidence_level, method='exact'|'wilson'|'wilsoncc')`。動作確認済み。statsmodels（`proportion_confint`：beta=Clopper–Pearson、wilson、jeffreys等）は同等で、追加導入は不要。
- **効果**：自前の区間実装を避け、正確性と保守性が上がる。
- **リスク**：区間の方式（exact／wilson）は事前固定が必要（Clopper–Pearsonは保守的）。
- **作業規模**：小。**優先度**：中。**判定**：**変更推奨**（S3を実装する時）。

### 7. S3のMSE信頼区間の方式
- **現状**：Aの計画はHoeffding（loss上限L_maxが必要）。
- **代替案**：分散を使う型（empirical Bernstein、betting型の信頼系列）にすると、標本数が少なくて済む可能性がある。外部library（例：confseq）の採否は未調査。
- **リスク**：停止規則・同時信頼度を事前に固定しないと妥当性を失う。
- **判定**：**追加調査**。ユーザーとAが、判断目的とδ・ηを決める時に比較する。優先度は低〜中。

### 8. C K1の区間演算の独立照合
- **対象**：`h4_structural_kernel`の手書き方向付き丸め（`dot_bounds`、`Band.repeat`）。
- **代替案**：python-flintの`arb_mat`（ball演算）で、同じP/gの`q`乗区間を独立に再計算する。python-flintのbindingはMIT、FLINT/ArbはLGPL v2.1+。PyPIにCPython 3.11以上のLinux wheelがある。現venvには未導入。
- **効果**：K1 E[S]区間の独立確認（置換ではない）。
- **リスク**：導入には環境の変更が必要（承認が要る）。結果の意味は変わらない。
- **判定**：**追加調査（任意）**。K0 PASS・K1が既に出ているので、優先度は低。

### 9. compile経路の重複
- **現状**：`cost_one`、`cost_trial`、`Backend.price_requests`、Full-Hの4経路が、ほぼ同じcompose→wrapper→compile→metricsを持つ。いずれも固定SOURCE。
- **代替案**：新しいcompile（例：C−SのP1、候補1の検証）は、同一pathのS/Cを同時に取る`Backend.price_requests`を再利用し、5つ目を書かない。固定SOURCEは変更しない。
- **判定**：**現状維持＋新規作業での再利用**。優先度は中。

### 10. D inventoryの繰返しgit/AST
- **実測**：`coverage()`は14.6 s。うち`saved_grid()`の繰返しが8.3 s（408回のgit show＋AST）、endpointの線形探索が3.2 s。
- **代替案**：`saved_grid`とselection IDsを`lru_cache`、endpointを候補IDでindex化する。
- **判定**：**変更推奨**（D所有、優先度は低）。結果は不変。

### 11〜20（現状維持・任意）
- **11 git show helper**：10以上のmoduleに重複がある。`SavedArtifacts`がpathごとにcacheしており、性能上の問題はない。統合は形式的な共通化なので**現状維持**。
- **12 identity・`same()`**：config 1件で約5 ms。Bの契約なので**現状維持**。menu規模が大きくなった時に、hash参照化を再検討する。
- **13 Accounting複製**：C1凍結SOURCE。新規counterではwire番号を引数化する。
- **14 物理核**：eigh/spectral、paired Taylor、expm_multiply、Decimal ceilingは、精度・演算順・速度とも適切。**現状維持**。
- **15 大roundのcompile**：m6（8192 event位置）以上の回路生成とtranspileの時間・memoryは未測定。P1（C−S pilot）の実測後に、回路の生成方法（DAGの直接構築など）を検討する。pass構成は科学profileなので変えない。
- **16 numba**：長いpathのSはCのclosed-state kernelで扱うので不要。
- **17 pydantic/jsonschema**：具体的な不具合がなく、frameworkを増やすだけ。
- **18 threadpoolctl**：runnerは環境変数でBLAS threadを固定済み。
- **19 hypothesis**（MPL-2.0、未導入）：候補2の同値性testはnumpyの乱数で十分。任意。
- **20 `local_residual_mean`**：純関数1個のために、`trotterlib/__init__`がpyscf/qiskitを含めてimportする。必要になったらfile単位のimportへ変える。現状は低優先。
