# B2-D 5方式評価器：接続準備・技術報告

**B0復旧後も未完了・STOP：** [修正・診断・全履歴](../../../docs/research/research_b_outer_b0_recovery_20261025.md)。最終SOURCE5c1fc5bc3、失敗matrix／raw・compiled QPYと途中rowを保存する。初期B3検査は改善したがD1の残差が固定tol未達。人工30件PASSを全B0 correctnessとせず、費用0件のままSTOP。以下の原本・旧結果を保全。

**B0 compiled correctness STOP（2026-10-25）：** [停止報告・実績・保存不足・原本](../../../docs/research/research_b_outer_b0_execution_20261025.md)。SOURCE8d0cb96dfを一回実行し、最初のcompiled gateでFAIL。24 single-step meanと最初のraw側gateは通過したが、費用0件・pilot未完了。数値差・回路は未保存で原因未確定。SOURCE修正・retryなし、mandatory STOP。以下のA0結果・準備履歴を保全する。

**A0実行完了・B0 STOP（2026-10-25）：** [結果・精度限界・原本・STOP](../../../docs/research/research_b_outer_a0_execution_20261025.md)。SOURCE1e3921ba1、8行×90/128桁を一回完了し、全8比較で経験的安定性・nominal bound整合を確認した。u_num=null／certified=false、B0実回路build／compileは未実行。旧SOURCE・結果・STOPを保全し、追加科学計算へ進まない。以下は準備履歴。

**A0／B0準備完了（2026-10-25、科学実行0）：** [`a0_reference.py`](a0_reference.py)／[`b0_native.py`](b0_native.py)／[`followup_io.py`](followup_io.py)。[固定SOURCE・scope・人工138件・個別command](../../../docs/research/research_b_outer_a0_b0_execution_preparation_20261025.md)。A0 SOURCE1e3921ba1、8行×90/128桁。B0 SOURCE8d0cb96df、48代表・96 compile要求。旧科学結果・STOPは保全し、次はA0の実行を先に判断する。B0は別許可。

**Outer/RTE固定22行の科学実行完了（2026-10-25）：** [結果・SOURCE・数値的限界・STOP](../../../docs/research/research_b_outer_rte_22row_execution_20261025.md)。SOURCE26e8734afを変更せず、全correctness先行で22行・1239 dispatch・274 key・RTE各96・110 expectationsを一回で完了した。u_num未評価、nominal boundは浮動誤差を含む証明ではない。精度／総資源の認定や追加計算へ進まない。原本は `artifacts/research_b_outer_rte22/20261025/`。以下の準備記録は保全する。

**Outer/RTE固定22行の準備完了（2026-10-25、科学計算未実行）：** [`outer_rte_execution.py`](outer_rte_execution.py)、入口 `scripts/run_b2d_outer_rte22.py`、人工test `tests/management/test_b2d_outer_rte22.py`。[固定SOURCE 26e8734af・契約・単一command](../../../docs/research/research_b_outer_rte_22row_execution_ready_20261025.md)。候補別r/K、1239 dispatch／274 key／RTE各96／110 expectations。既存105＋追加14件PASS、read-only preflight PASS。u_num未評価を残し、科学実行は別途明示許可後だけ。旧16行・36行の結果・契約は保全する。

**Outer固定16行の科学実行完了（2026-10-25）：** [結果・SOURCE・STOP](../../../docs/research/research_b_outer_splitting_stage1_execution_20261025.md)。SOURCE b85dd3284を変更せず一回完了。全gate先行、16行・906 dispatch・490 key・80 expectations。Corrected signal改善は限定観測、raw改善／総資源優位は認定しない。原本は `artifacts/research_b_outer_splitting/20261025/`。追加科学計算なし、mandatory STOP。以下の準備記録は履歴として保全する。

**Outer固定16行の接続完了（2026-10-25、科学実行未承認）：** [`outer_execution.py`](outer_execution.py) と `scripts/run_b2d_outer_stage1.py` を追加し、既存Input package・scope・launcher・失敗保存／STOPへ接続した。[SOURCE b85dd3284・最終契約・単一command](../../../docs/research/research_b_outer_splitting_stage1_execution_ready_20261025.md)。906 dispatch／490 key／RTE各24／80 expectation。追加14＋既存91＝105人工件PASS、科学処理0件。以下は前段の履歴。

**Outer splittingの独立拡張（2026-10-25、準備のみ）：** [`outer.py`](outer.py)／[`outer_backend.py`](outer_backend.py)、人工test `tests/management/test_b2d_outer_preparation.py`、[技術報告・未承認科学契約案](../../../docs/research/research_b_outer_splitting_stage1_preparation_20261025.md)。4方式のstageとsigned RTE／normalizationを既存DAGへ接続。追加19＋既存72＝91人工test PASS。科学計算・回路生成なし。旧launcher／scopeは保全し、新outerの実行bindingは条件採択後に接続する。

**Stage 1実行済み（2026-10-25）：** [全結果・技術復旧・科学SOURCE758a05ed・STOP](../../../docs/research/b2d_stage1_execution_20261025.md)。入力・独立A4数学参照・published列がPASSし、36行・941 dispatch・448 keyを完了。A0／D-S2からのfull signal誤差改善は観測したがA3との優劣はINCONCLUSIVE。入力loaderの2参照と保存後のCLI表示を修正、人工72件PASS。保存済み原本・CSV・provenanceは `artifacts/research_b_b2d_stage1/20261025/`。以下は準備段階の履歴。追加科学計算・Stage2は開始しない。

**現在の実行経路（2026-10-25）：** [簡素化した科学実行契約・単一command・技術報告](../../../docs/research/b2d_stage1_simple_execution_20261025.md)。科学SOURCE `54008c697e686a42ec48941d25219d523bafd454`。チャットでの明示許可後、Codexが `run_b2d_stage1.py --execute --source-commit … --output …` で一回実行する。承認原文・ID・packet・receipt・grant発行／消費・manual authorityは不要。科学条件・scope・gate・provenance・fresh output・失敗保存・retry禁止・STOPを維持。70人工test PASS、科学計算未実行。以下は過去段階の履歴であり、記載された承認管理や削除済みscriptを将来の実行要件にはしない。

**最新（通常ユーザー方式、2026-10-25）：** [新契約・発行／実行の二段階手順](../../../docs/research/b2d_stage1_user_approval_20261025/CONTRACT.md)・[技術報告](../../../docs/research/b2d_stage1_user_approval_20261025/TECHNICAL_REPORT.md)。新科学SOURCE `91f45aeb50275bb6d22050a304d0a9b1ff025d70`。root所有receiptを本人の0700／0600 private記録へ変更し、approval_policyを契約へ束縛。科学条件・scope・環境は不変。80人工test PASS、正式発行・科学実行なし。同一OSアカウントの本人性を独立認証するものではない。以下は過去段階の記録であり、旧root／管理者要件を今回へ適用しない。

**最新（2026-10-25一括準備）：** [最終契約・ユーザー承認手順](../../../docs/research/b2d_stage1_execution_ready_20261025/CONTRACT.md)・[最終技術報告](../../../docs/research/b2d_stage1_execution_ready_20261025/TECHNICAL_REPORT.md)。SOURCE候補348e08に手動ユーザー承認record接続、実環境binding、published規約の追加gate4件を登録した。74人工test PASS。技術準備は完了、科学実行・正式grant発行は未承認・未実施。下記は以前の段階記録であり、最新状態はこの契約・報告を優先する。

**最新の実行準備（2026-10-25）：** [最終提案](../../../docs/research/research_b_b2d_stage1_final_execution_proposal_20261025.md)・[技術報告](../../../docs/research/research_b_b2d_stage1_final_preparation_technical_report_20261025.md)・[scope JSON](../../../docs/research/research_b_b2d_stage1_scope_20261025.json)。SOURCE候補 `97a449ad53c74361ffc5c4cffd6de5f3f0f0a72e` に `scope.py`／`launcher.py`／`errors.py` とCLIを追加し、入力・DAG・内部数値境界の実績照合と失敗保存を接続。既存46＋追加16＝62人工test PASS。production authorityは未接続、実科学経路・grantは未起動。最新状態はこの後続報告を優先する。

**後続準備（2026-10-25）：** [Input package／preflight技術報告](../../../docs/research/research_b_b2d_input_preflight_technical_report_20261025.md)と[第一段階契約案](../../../docs/research/research_b_b2d_stage1_science_execution_contract_proposal_20261025.md)を参照。
`input_package.py`／`preflight.py` を追加し、位相無効・全correctness先行・科学claim境界を最小拡張した。既存29＋新規17＝46人工test PASS。実入力・科学計算・新grantなし。
以下は起点52baでの準備記録であり、未接続欄の現状は後続報告を優先する。

2026-10-25 JST。**Scienceは未起動。** 起点 `5d3ec883eeeabf69abfa128c9e169c59fc572149` の既存adapter・15人工testを変更せず再利用した。新規追加だけで作業し、旧SOURCE・結果・草案・grant・STOPを保全する。

## 追加した接口

- [`plans.py`](plans.py)：A0 current partial-S2、A1 standalone D-S2、A2 Yoshida4、A3 published Y8m10b、A4 published YP8m8。数学的積とappend順、同方向D半時間、opaque tail barrierを扱う。
- [`_baselines.py`](_baselines.py)：固定PF repository `20d64c2fea1ed2d2188777df4f3d962ee94df486` の `product_formula.py`（blob `70d68691a1dcdedea5447aac4b8bbfa51141d4cb`）からS2／Yoshida4／Y8m10bの3関数をbody変更なしで移植。実取得SOURCEとのAST一致を確認。A4は旧adapterをそのまま使う。
- [`evaluation.py`](evaluation.py)：必須科学条件を持つSettings、共有参照DAG、call予算、出力schema、承認境界。`build_graph` は純粋な記述処理。`evaluate` は承認参照・grant参照・SOURCE／grid／call一致を確認してからbackendを作る。
- [`dense_backend.py`](dense_backend.py)：承認後に限り生成する数値backend。外部で準備したsector行列・state・symbolic tailを受け取る。importだけではnumpy／Qiskit／trotterlibを読まない。今回instantiate・performしていない。
- [新規人工tests](../../../tests/management/test_b2d_evaluator.py)と[実行前提案](../../../docs/research/research_b_b2d_science_preflight_proposal_20261025.md)。

## 再利用と追加の理由

既存PRの `rte.make_rte_config`／`finite_rte_operator_moments`、`finite_rte_signal_validation._signal`／`_phase_distance`／`_array_fingerprint` をlazyな数値backendから接続する。finite Taylorやsampler、signal estimatorを新しく作っていない。`_state_result` はshots／暫定RPE予算等まで計算し、既存validation runnerもHamiltonian・eig・回路を起動するため、第一段階へ丸ごと流用しない。

新コードは5armの接続・共有・identity・出力と必要なmatrix積のglueである。旧partial-S2は任意nested D列を受け取らず、旧PF namespaceのimportも科学依存が広いため、stdlibの独立namespaceを維持した。価格table・係数optimizer・新しいcorrector・circuit renderer・grant発行処理は追加しない。

## 作用順とphase

A0は `reverse @ tail @ forward`。A1–A4は同じGを両側へ置く。A4の数学的 `P K^n P†` に対するappend順は旧adapter同様 `P† → K…K → P`。tail越しのP相殺はしない。隣接D融合の既存規則は旧15testで確認し、本数値経路はnative列最適化を加えずordered積／powerを扱う。

Scalarはcontextのconstant＋抽出identityをphysical時間に対して付ける。tailのevent位相・normalizationは既存analytic momentの意味論を保持し、新samplingをしない。controlled wrapperの数値位相照合は第二段階。A4は独立した数学的S2/P/K積との将来数値gateを用意したが、**旧validatorとの数値operator同値性は引き続き未確認**である。

## 実行しない経路と未解決の接続

今回の人工確認ではbackendのfactoryもmockへ置換する。`DenseBackend` の科学依存import、matrix exponential、operator error、signal、commutatorを一度も実行していない。fixture loader／D blockの行列化／sector・basis順の独立照合はまだ接続が必要で、`PreparedDenseInputs` を名前だけ埋めても科学資格は成立しない。

未来のInput packageには、同じsector順序のblock行列、retained normalized tail、full physical H、state、symbolic tailとresolved identityが必要。backendは配列hash・H=D+R+scalar・shape／Hermiticity／state norm・D／D-R非可換性を数値gateで照合するが、それ自体が未実行である。schemaにSOURCE・設定・input／graph fingerprint・call・全rowを保存する。

未設定のradius／branch／数値条件を拒否し、承認なしではbackend生成前に拒否する。ただし `ExecutionApproval` は呼出し元が渡すattestationであり、外部ledgerの真正性・未使用grant・ユーザー承認を暗号学的に認定する仕組みではない。未来のlauncher側でSOURCE・実入力・fresh出力・新grantを確認する。science runner／CLIは今回作っていない。

## 人工test結果

Python 3.11.0rc1、stdlibのみで次のfocused filesを実行した。

```bash
/usr/bin/python3.11 -S tests/management/test_b2d_adapter.py -v
/usr/bin/python3.11 -S tests/management/test_b2d_evaluator.py -v
```

**既存15件＋追加14件＝29件PASS。** 5family／固定係数AST、A0 sweep、A4旧列との同一性、signed順序、actual descriptor inverseとtail逆の拒否、metadata共有、grid公平性、参照共有、未設定条件／未承認要求の拒否、mock出力schemaを確認した。import barrierでnumpy／scipy／Qiskit／化学／trotterlibを読まないことも確認した。既存science testsは実行していない。

提案gridの36行・941 dispatchはmetadataから数えたもので、科学call消費は0。low-levelの全expm数、Input package作成を含む全job予算、科学性能・速度倍率は未確定。人工PASSはoperator精度・数値位相・full signal改善・YP8m8優位の証拠ではない。

## 次の担当・STOP

研究Bチャット／ユーザーが、[提案条件と未確定事項](../../../docs/research/research_b_b2d_science_preflight_proposal_20261025.md)、Input package接続、科学SOURCE・新grant・全call範囲を確認し、必要な追加準備と限定科学実行を別途明示承認する。科学的materiality、radius／branch条件、精度資格、retry／保存scopeは未採択である。

C1圧縮DF最優先を維持し、研究Aはnormalization／finite-bit／range・shots、研究Cは表現／native合成／制御位相を分担する。通常の研究B進行に統括報告を必須としない。共有環境変更、他研究merge、科学計算・回路build／compile、保存数値の再解析、新grant発行はない。

専用branch `codex/research-b-b2d-evaluator-preflight-20261025` を公開し、remote SHAと追加内容を確認後 **mandatory STOP**。第一段階の未承認科学実行や第二段階へ進まない。
