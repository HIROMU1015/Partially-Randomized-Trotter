# H4 Pauli／DF PR母平均価格：第一段階の静的確認と限定scalar仕様

2026-10-25。**追加compileなしで行う限定scalar batchの仕様を確定した。固定SOURCEモデル・正常終了を前提とする比較としては準備できている。ただし、Pauliの実compilerが全allowed入力で正常完了することまでは証明していないため、無条件の全law upperを保証済みとは宣言しない。** 今回はH4のU・e・L・差・順位を一切数値評価していない。

基点branch `codex/non-df-h4-pr-mean-cost-design-20261025`、commit `a5d96ae54d8cc3bc891d684570ad7e82160bab6c`。GitHub ref・固定ファイルを実取得しlocalと照合した。今回の専用branchは `codex/non-df-h4-pr-mean-cost-stage1-20261025`。元[設計報告](non_df_h4_pr_mean_cost_design_20261025.md)・[非実行案](../artifacts/non_df_h4_pr_mean_cost_design/2026-10-25/next_validation_plan_DRAFT_DISABLED.json)・[統合カタログ](../artifacts/non_df_h4_data_catalog/2026-10-25/README.md)を再利用し、変更していない。

## 1. 対象と判定範囲

固定linear H4／隣接1.00 Å／STO-3G、system8＋ancilla8。Pauli-JW184非identity項、PR prefix27、central-rotation制御。DF-native tol-only1e-8／rank10、one-body＋fragment0保持。S2／T=.8／q8／r4／K4。Qiskit1.3.0、basis `rz/sx/x/cx`、opt1、seed17、backend/coupling/layout/routing指定なし。費用はstate準備を除くfull-wrapper compiled RZ数。

目標は各軸aについて `μ_P,a=E_pP[C_P,a] < μ_D,a=E_pD[C_D,a]`。Pauli/DFのlawは異なる。generic PauliやPFの価格は入力に含めない。保存256本のmeanを再計算・再ランキングする作業でもない。

今回ユーザーが固定した価格confidenceは95%、`δ_cost=0.05`、`δ_cos=δ_sin=0.025`。仕様では十進指定をexact有理数文字列として扱う。これらのδは**価格標本の推論**に用いる。量子shotの `α_axis=0.025` は独立の意味を持ち、shot会計関数は呼ばない。

## 2. Pauli uniform count upper：SOURCE対応

### 2.1 入力familyと一回の費用

`pauli_signal_adapter.py:append_pauli_event`はproduct列を発生順に作用させ、最後にrotationを一つ作用させる。k∈{0,2,4}、product数はk以下、符号・order phaseはbody phaseへ集約する。保持部は各stepにforward/reverseが一回ずつある。

`h4_d2_pauli_control.py:_tokens/_rotation/central_control_body`は、非identity単一Pauli evolutionとlocal X/Y/Z productを受理する。XのbasisはH、YはSdg→H、parity CX往復の中央だけCRZ、productはlocal CX/CY/CZへ移す。body phaseはancilla P一つへ移す。cosine wrapperは両H、sineは終H前にSdgを一つ加える。位相を無視してPを削除する数え方はしない。

全identity evolution、未対応gate、symbolic/classical条件、非有限operandはbuilderが拒否する。固定Pauli184項／tailの保存familyは非identityであり、既存SOURCEの符号・順序規則がorder4にもそのまま適用される。256本の観測範囲を全lawと置き換えるのではなく、この生成規則を対象にする。未観測回路についても同じbuilderとcompilerを使うことが条件である。

保存prefix27のZ/ZZ・support1〜2は[既存構造報告](non_df_h4_cost_reversal_structure_20261024.md)とcatalog内の保存modelで確認する。一回の保持rotationは、basis変換なし、parity CX往復、CRZであり、候補native総数≤6となる。

### 2.2 標準libraryの候補経路

[保存公式Qiskit1.3.0 SOURCE一覧](../artifacts/non_df_h4_r2c_compiler_phase/2026-10-25/upstream_sources.json)と、その `upstream/equivalence_library.py` を読み直した。

| 入力gate | SOURCE上のnative到達候補 | native総数upper |
|---|---|---:|
| H | H→S,SX,S；S→U1→RZ | 3 |
| S/Sdg | ±π/2のU1→RZ | 1 |
| P | Phase→U1→RZ | 1 |
| CRZ | RZ(θ/2),CX,RZ(−θ/2),CX | 4 |
| CY | Sdg,CX,S | 3 |
| CZ | H,CX,H | 7 |
| X/SX/CX/RZ | target basis | 1 |

各経路のglobal phaseはSOURCEに残り、body control後のrelative phaseも保持する。ここで示すのは**個数**であり、πやparameter bindingをexact実数演算と仮定したoperator誤差証明ではない。

Rust `basis_search.rs`はtarget basisのcost=1からrule内instruction cost和を使って探索する。正常なcost探索・非overflow・固定標準libraryのもとでは、採用native展開の総数は上記候補の総数以下になる。同costの別経路を選んでもcount upperは同じ。`compose_transforms.rs`の置換・parameter bindingはgateのoperandとphaseを更新し、数値値を理由に新しい長いruleを任意に挿入する経路ではない。

### 2.3 実際のopt1経路

central価格は `h4_a_cohort.py`／`h4_a_axes.py` → `h4_d1_adapter.py:compile_saved` → `rte_compiled_cost.py:transpile_and_measure_cost` を使用する。最後は `default_compiler()` の設定で `transpile` を呼び、posttranspileの `count_ops()['rz']` を価格にする。

既存runtimeのPython presetを実際に読み、既存保存SHAとも一致した。`generate_preset_pass_manager.py:_skip_target`により今回Targetなし。`level1.py`のlayout/routingはcoupling_map/initial_layoutなしで省略される。initは逆gate消去。今回のstandard gate族には3qやUnitaryGateがなく、translationのUnitarySynthesis/HighLevelSynthesisが追加の一般unitary入力を作る対象はない。BasisTranslatorがnative化する。

opt1の反復は `Optimize1qGatesDecomposition → InverseCancellation → GatesInBasis → 必要時translation → Depth/Size FixedPoint`。保存Rust `euler_one_qubit_decomposer.rs`ではTargetなしのold scoreもnew scoreもnative **総gate数**を比較する。native runはoutside_basis=false、再合成出力はRZ/SX/X。new scoreが改善する場合、または空列化の場合のみ置換する。消去passも削除だけであり、再度basis外になる経路はない。

従って正常returnするこのSOURCE経路では `RZ_out ≤ total_native_out ≤ total_native_initial`。RZ数だけの単調性や、保存cohort最大の全lawへの外挿は使用しない。native gate数が減る採用rewriteしかないので、反復回数の有限性にも接続できる。Euler角の数値精度やphase丸めの誤差証明は今回再開していない。

### 2.4 記号式と未閉鎖の前提

system qubit数n、保持項数Lとして、一般Pauli rotationの候補総数は `≤10n+2`、signed product一つは `≤7n`。Z/ZZ保持rotationは≤6、保持部は2qL回、eventはqr回・productは各K以下、aggregate Pとwrapper両Hが≤7なので、

**`U_P,a = 12qL + qr[(10n+2)+7nK] +7+1_{a=sine}`。**

cosineとsineそれぞれのfull wrapperを被覆するSOURCEモデル上の式である。n/q/L/r/KのH4値を代入・評価していない。

| 前提 | 確認結果 | 結論への扱い |
|---|---|---|
| 固定builderのgate族・保持構造・全orderの回数 | SOURCE・保存modelで対応 | 観測頻度ではなく生成規則からcountする |
| native lowering候補、正常探索時のcost upper | 公式SOURCE・installed library対応 | 最短cost探索／整数cost正常動作を前提とする |
| Targetなしopt1のnative総数非増大 | presetとaccelerated実装を確認 | 正常returnした出力の構造補題として支持 |
| 拒否・例外・非有限値・資源中断 | 保存成功例はあるが、全allowed入力で発生しないことの全域証明なし | **失敗価格を0にせず、成功条件付きlawへ再正規化しない** |
| Rust tag SOURCEと実wheelの完全な対応 | version・既存runtime固定記録を再利用。binary再build一致証明なし | 通常のSOURCE-to-runtime前提として明記。証明済みへ格上げしない |

判定は **`SOURCE_COUNT_BOUND_SUPPORTED_ON_NORMAL_RETURN`**。全allowed trajectoryに対する実wheelのtotalityまで含めた無条件の保証は **`NOT_ESTABLISHED`**。SOURCEモデルと全入力正常完了を前提にした条件付きscalar診断は準備できるが、その前提を受理しない場合は保証付き母平均順位のGO条件が不足する。少数追加compileを行っても、この全域の前提を証明できるわけではない。

## 3. DF DKW lower：保存標本の適合性

### 3.1 固定lawと標本の取得・再利用

元科学SOURCEは `0c329b30a81accb35c212be8defa66fee0dd0f12`。GitHubから元 `h4_audited_evaluation.py` と保存 `run01/SOURCE.json` を実取得した。現在参照したevaluator・audit・sampler・DF repeated builder・compiler等7 sourceは元SOURCE manifestと同SHAである。

- `evaluate_audited_signal`は固定 `cost_samples=256` を先に決め、SeedSequenceの子seed群を全件順に走査する。価格を見て候補を選ぶ、追加抽選する、失敗を捨てて次を引く経路はない。
- DFの同じtrajectoryにcosine/sine wrapperを作り、両価格を保存する。DF repeated builderはmaster seedからstep seedを作り、各stepの有限RTE lawを使う。samplerはPCG64でorderを選び、続いてcomponentを選ぶ。
- Audit IDはcase/binding/**index/seed**から作る。同じevent列が再度出たことを理由に重複を捨てる仕組みではない。validatorやcompiler不一致時はcaseを停止し、resamplingで埋めない。
- Cacheは実circuit fingerprint／compiler／backendで同じ価格関数を再利用する設計。元DF PRの保存会計はcache_hits=0、new_compiles=512、paired_trajectories=256、automatic_retry=false。今回はこの保存値を引用するだけで、新たに数え直していない。
- `status.json`は `COMPLETE_FIXED_CASE`、event_join_count=256。raw resultはpaired sample_count=256。今回、256レコードのIDと両軸fieldの構造だけを確認した。全eventの再読込・RNG replay・価格再集計は行わず、過去のrecovery照合を再利用した。

保存完了データの整合性と元監視の完全性は別。既存[recovery verification](../artifacts/non_df_h4_pilot/2026-10-23/postprocessing_recovery01/verification.json)は `SAVED_DATA_VALIDATED_SUPERVISION_INCOMPLETE`、scientific_retry=false、original_native_completion_verified=falseである。元native UNKNOWNを今回の確認でPASSへ変えない。

固定lawとの対応は、保存distribution/config・同一SOURCE・既存Cのfinite-grid証拠から支持される。`h4_ball_evidence.py:cdf_binary64/grid_masses`は保存binary64 CDFと53-bit wordのeffective質量を対応させる。nominal確率とeffective質量を混ぜない。SeedSequence・DF step用Python `random.Random`・PCG64の階層が数学的IIDを証明するとは言わない。**固定effective lawのIID trajectoryモデルは今回ユーザーが明示した統計上の仮定**であり、そのまま限定する。

SOURCE上に費用に応じた選抜・除外はなく、固定256本の保存を支持する。未記録の外部選抜まで不存在と証明するものではない。95%は今回指定した固定DF価格関数とnamed comparisonの両軸のcoverageであり、多数の事後選択した実装・他研究全体への同時保証ではない。Pauli centralのcohortからDF minimumを選ぶこともしていない。

### 3.2 Full-wrapper価格と保存minimum

`hadamard_wrapper`はH、既に制御されたDF body、sineならSdg、H、ancilla測定を作り、state準備を含まない。compiler価格はposttranspile RZ数。raw resultのscopeも `full_cosine_sine_hadamard_wrapper_no_state_preparation` で一致する。

[保存cost_summary](../artifacts/non_df_h4_conditional_cng_results/2026-10-24/run01/cost_summary.json)の `/df_native_PR/n=256`、`/df_native_PR/observed_min/cosine=21754`、同sine=21754を入力にする。`h4_conditional_cng.py:cost_summary`が各軸の元価格からminimumを保存するSOURCEであることを確認した。既存要約とraw/configの来歴は照合したが、今回 `min(prices)` や新しい統計量を計算していない。

### 3.3 Minimumの事後選択と両軸coverage

非負のDF価格C、CDF F、経験CDF Fhatについて二側DKW–Massart bandを使用する。元[設計§4](non_df_h4_pr_mean_cost_design_20261025.md)と[一次資料の証明](https://arxiv.org/html/2403.16651v1)を確認した。

`e_a=sqrt(log(2/δ_axis)/(2 n_s))` とすると、確率≥1−δ_axisで全threshold tに同時に `|Fhat(t)−F(t)|≤e_a`。left limitを用いて `E[C]≥t P(C≥t)≥t[1−Fhat(t−)−e_a]_+`。

tを保存minimumに取れば `Fhat(t−)=0` なので **`L_D,a=c_min,a max(0,1−e_a)`**。同じ標本でminimumを選べるのはbandが全thresholdに同時成立するためであり、事前固定thresholdの検定へ事後minimumを挿入しているのではない。

各軸δ=.025を割り当てたunion boundで両軸coverage≥.95。軸間独立性を仮定せず、両軸が同じtrajectoryで相関していても適用できる。保存Uのcomponent集合を独立層とは扱わない。未観測order4／componentの費用を0にする操作もない。非負性とCDF bandだけで母平均lowerを作るので、未知の高価格tail上限を必要としない。

## 4. 次の限定scalar batch：実行無効の仕様

[scalar_batch_SPEC_DISABLED.json](../artifacts/non_df_h4_pr_mean_cost_stage1/2026-10-25/scalar_batch_SPEC_DISABLED.json)に入力・精度・許される演算・前提・STOPを固定した。**実行承認はまだない。** 新しいgrant、承認認証、active contract、runnerは作成しない。

1. 入力は保存条件 n=8/L=27/q=8/r=4/K=4、n_s=256、保存両軸minimum、ユーザー指定のexact十進δだけ。元raw価格vectorやQPY/stateを計算に読み込まない。
2. U_P,cos/U_P,sinをPython整数で各一回評価する。共通式にsineのindicatorだけを加える。ここではまだ式を展開・代入していない。
3. 既存 `h4_conditional_cng.py:trans_bounds('ln', exact(2)/exact(δ_axis))` を再利用する。この純粋関数だけを使い、`reuse/shot/scale_bounds/rare_law/cost_summary`を呼ばない。旧shot用のbinary64 literal変換をδ指定へ持ち込まない。
4. 分母 `2 n_s` はexact整数。ln区間をexact有理数で割り、Decimal 96桁で有理数端点を外向きに変換し、各sqrtの正しく丸められた値にnext_minus/next_plusを一つずつ置いてe区間を包含する。Decimalのsqrtは丸め方向指定で直接外向きになると仮定せず、両端を広げる。sqrt helperの新規部分が必要なら人工有理数で技術検証してから使用する。
5. exact有理数として `L_lo=c_min max(0,1−e_hi)`、`L_hi=c_min max(0,1−e_lo)` を作る。U整数と比較し、各軸 `U<L_lo` のときだけ十分条件の分離を記録する。
6. `U≥L_hi` はこの十分条件で未分離、`L_lo≤U<L_hi` は数値包含幅による未確定。いずれも逆順位や同順位を意味しない。96桁一回のbatchでSTOPし、昇精度・law-weighted・追加データへ自動移行しない。

lnとeは両軸で同じため一回ずつの共通評価を再利用できる。予定範囲は**一つの限定scalar batch、両軸U/Lと判定のみ**。H4 build/compile、sampling/RNG replay、Hamiltonian/state作用、新N/G・ρ再会計は0。

両軸の分離は条件付き母平均価格の十分条件だけを示す。full-law正常完了／SOURCE-to-runtime前提が未解消・未採用なら、数値不等式が分離しても最終all-law判定は `PREMISE_UNRESOLVED` のままとする。失敗や未対応gateを費用0とする補完、失敗を除外したlawへの変更、正式certificate受理はしない。

## 5. GO／STOP・次の担当

| 項目 | 現在 | 次の扱い |
|---|---|---|
| DF推論の固定law/IIDモデル・95%配分 | ユーザー指定済み。SOURCE経路に費用選抜なし | 新たにIID証明を要求しない。統計仮定を記載 |
| Pauli countの構造 | 正常returnする固定SOURCEモデルで支持 | 全law正常終了・wheel対応の前提を明示 |
| 無条件の実wheel全law upper | 未確定 | 保証済みとは宣言しない。例外の価格を捏造しない |
| 限定scalar実行 | **未承認・未実行** | ユーザーが上記条件付きscopeを含めて明示承認した場合だけCodexが一batch実行 |
| 両軸分離 | 未評価 | 前提が満たされる範囲の結論を保存しSTOP |
| 未分離／前提不成立／非有限／例外 | 未評価 | 原因と保守性を報告してSTOP。追加手法へ進まない |

**ユーザー**：このSOURCEモデル・全入力正常完了という明示前提付きのscalar診断を進めるか、無条件保証に必要な未閉鎖前提を先に扱うかを判断し、実行範囲を承認する。**Codex**：承認された場合だけ指定batchを行い、数値分離と前提成立を別fieldで報告する。現時点で新compileが必要という結論はなく、少数compileでtotalityを証明する提案もしない。

母平均Gの順位、ρを含む実精度、Pauli B0/B0I切り捨てPF、最良決定論的PF、他分子への一般化は別問題。価格順位が将来分離しても「Pauli PRが全Pauli系手法で最良」とは結論しない。

## 6. 検証・保全・公開

既存の純人工checks4件を同じコードで再実行してPASS。新しいテストframework・回路object・科学pipelineを作らなかった。CDF検査は与えられたband上の補題であり、実PCG64のIIDや統計coverageを実験的に証明したものではない。

[verification.json](../artifacts/non_df_h4_pr_mean_cost_stage1/2026-10-25/verification.json)にGitHub blob対応、元SOURCE対応、保存レコードの構造、人工検証、未解決前提と今回の科学call0を記録した。主agent自身の技術確認であり、正式な科学レビューではない。旧SOURCE/raw/結果/grant/ledger/marker/STOP、旧population upper/null、ρ/null、N/G/null、元native UNKNOWNを保全した。

最新版 `research_project_instructions_unified_2026-10-25.md` §8.4・§10・§10.3を適用。共有環境・他ユーザー・他jobへの操作なし。新しい科学実行や承認証明の仕組みは追加していない。

[publication_status.json](../artifacts/non_df_h4_pr_mean_cost_stage1/2026-10-25/publication_status.json)に公開状態を記録する。専用branchでcommit・GitHub公開・remote SHA照合後、**mandatory STOP**。次の限定scalar実行にも自動移行しない。

主成果物commit **`3b26f3af3ae43514ea8b9cc4841a817b8685074c`** をGitHub接続のGit data APIで公開した。local準備tree `100b94a5213d668494e4599dd0340820211ddcdd` と公開treeが一致。同commitをgit fetchしlocal HEAD／origin ref／GitHub refの同SHA・clean状態を確認した。この公開記録だけを追記し、最終HEADもremoteと照合してチャットへ記載する。**第一段階を完了してmandatory STOP。H4 scalar実行は未承認・未実行のまま。**
