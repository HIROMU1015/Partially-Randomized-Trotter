# Track D C3：raw構築前の技術エラーでSTOP

2026-11-06 JST。**q=4・8の科学的correctness・価格は未評価。** 新runnerのcontract参照先の誤りにより両workerがraw構築前に停止した。raw/compiled回路、CAR積、χの数値、資源価格は生成されていない。compile 0、retry/resume 0。失敗を科学的不成立やReference方式の不利と解釈しない。

実行SOURCEは [`f18adc46dffcb69a6b1028d5c38793f9b4c0b948`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/f18adc46dffcb69a6b1028d5c38793f9b4c0b948)。[実行前PLAN](PLAN.md)を8公開SOURCEファイルとともに固定し、remote/localのbytes一致を確認してから一度だけrunnerを起動した。C2結果 `c8fe5091c42ec786688b6931105ad64d10c87d18` を親とする専用branch `codex/track-d-c3-q4-q8-validation-20261106` を使用した。

## 固定条件・入力identity

linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、各h=δ=.1。元H厳密ground仮定、状態準備費除外。保存cohort `near_practical_PR_r32_fixed32_v1`、endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。

|q|固定時間順fixture|固定global cut|到達点|
|---|---|---|---|
|4|sample00→01→02→03|complete step2後|contract読出し失敗、raw構築前STOP|
|8|sample00→01→…→07|complete step4後|contract読出し失敗、raw構築前STOP|

[SOURCE.json](../../../artifacts/track_d_c3_q4_q8/2026-11-06/run01/SOURCE.json)に元science `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7` の143 SOURCE・16入力、追加保存law、carrier `3649b75af191916daf44abb91e14d34c43b83e8f` の全96 cohort bindings、package pin、新SOURCE hash、既存A/C1/C2保存資料のidentityを記録した。実行前の既存preflightは完了した。[fixture_identity.json](../../../artifacts/track_d_c3_q4_q8/2026-11-06/run01/fixture_identity.json)は固定対象8 sampleのevents/raw/CARファイルhashを抜粋した静的記録であり、新計算結果ではない。

## 原因・保全・実行された範囲

新runner [raw_worker](../../../scripts/run_track_d_c3_q4_q8.py) の53行はcontractを `(origin / runner.CONTRACT)` から読もうとした。`origin`は旧143 SOURCEと固定科学入力を保存した読取用treeであり、この後段のcontract自体はexecution worktreeの `artifacts/research_c_fixed_word_pilot/2026-11-04/run01/contract.json` にある。既存preflightはworktree側から正しく読んでいたが、新workerが参照先を誤った。Codexによる実装上の誤りであり、元SOURCEや入力の科学的な不整合は検出されていない。

両workerとも`FileNotFoundError`、各約3.5秒、runner wall 4.392778133973479秒。`prepare_common`、`prepare`、raw correctness harness、compile/postcheckには到達していない。新sampling、DF refit、ground solve、B2、N、shot、q>8、全RPE費用評価はすべて未実施。

元の[result.json](../../../artifacts/track_d_c3_q4_q8/2026-11-06/run01/result.json)、q4/q8のfailure、[STOP.json](../../../artifacts/track_d_c3_q4_q8/2026-11-06/run01/STOP.json)を変更せず保全した。`source_input_preserved=false`は、成功経路の終端照合に到達しなかったことを表し、入力破損を示す検出結果ではない。停止後、科学計算を呼ばずに既存byte preflightだけを再照合し、実行前SOURCE/input metadataと完全一致したことを[post_stop_identity.json](../../../artifacts/track_d_c3_q4_q8/2026-11-06/run01/post_stop_identity.json)へ別記した。旧grant・ledger・marker・STOP・結果は変更していない。

資源観測はaffinity64 CPU、idle CPU相当22.594167995746947、MemAvailable493960429568 bytes。raw2worker×11threads、PASS後は最大8worker、Reference postcheck4threads、direct1threadという観測値に基づく配分を記録した。後半は未開始。人工的資源上限、共有設定変更、他ユーザーへの操作は行っていない。

## 新実装の状態とstrong対照

新moduleは、Stage Aのprepare_common・saved event復元・full-frame tail共有、StrongBuilderのGaussian basis/位相保持relative transition、既存対角emitter、C1のCAR積とwhole-path分割、Stage A/C2のcorrectness/compile/postcheck harnessを呼ぶ構成にした。

既存StrongBuilder.sequence/directional_headは一種類の制御modeしか受けないため、mode混在headを扱う最小拡張を追加した。既存repeated builderと人工Reference fixtureの規則に従い、step境界のdirectional half＋uncontrolled halfをordinary controlled fullへ置き換える。設計上はsymmetricのq−1境界、ReferenceのA/B内部合計q−2境界を融合し、global cutの1境界は保持する。scalar/identityはsymmetricのancilla P、ReferenceのA/B global phaseに集約する。head相対融合、uncontrolled Gaussian、directional中心、tail同basis共有を使い、架空の節約値の加算はしない。

人工test3件はPASS：固定recipe/cut、非零scalarを含む境界融合の両制御branch一致、保存比較データの分離。**これはH4 q4/q8のcorrectness・fairness成立の証拠ではない。** 実回路を構成して検証する前に停止したため、strong baselineの分子fixtureへの適合性、AB=V、actual adjoint、真空χ、X/Y、compile後精度、fusion損失の価格は未確認。SOURCEは失敗時のまま保全し、今回修正・再実行していない。

## q依存性：今回取得できなかったもの

|項目|q=4|q=8|
|---|---|---|
|raw/compiled QPY・CAR積・χ|未生成|未生成|
|correctness / postcheck|未実施|未実施|
|RZ count / RZ depth|未取得|未取得|
|CX count / CX depth|未取得|未取得|
|total depth / qubits / circuit size|未取得|未取得|
|ReferenceのCX・depth削減維持|判定不能|判定不能|
|step融合・global cutの実価格への影響|判定不能|判定不能|

以下は既存保存値の再掲であり、新compileや新しい集計ではない。X/Yは各々同値。

|保存fixture・方式|RZ|CX|RZ depth|CX depth|total depth|qubits|size|
|---|---:|---:|---:|---:|---:|---:|---:|
|q1 sample00 symmetric|5709|2140|1387|1034|3207|9|11594|
|q1 sample00 Reference|6129|2136|753|495|1679|17|12244|
|q2 sample00→01 symmetric|10773|4100|2624|1996|6096|9|21970|
|q2 sample00→01 Reference|11114|3860|1357|860|3014|17|22263|

出典は[Stage A](../track_d_h4_paired32_reference_20261106/RESULT.md)の[保存CSV](../../../artifacts/track_d_h4_paired32_reference/2026-11-06/run01/paired_resources.csv)と[C2](../track_d_c2_q2_compile_20261106/RESULT.md)の[保存比較JSON](../../../artifacts/track_d_c2_q2_compile/2026-11-06/run01/resource_comparison.json)。q1はevent16後cut、q2はstep1後cut。C2には今回設計したcross-step手動融合が含まれず、q4/q8との最適化条件は異なる予定だった。qごとにpath/cut/fusionが違うため、結果が取得できてもq単独の因果効果とはならない。

全lawの期待価格・信頼区間、長q、全13-round、元HのN・共通エネルギー精度/成功確率に接続する新証拠は取得できなかった。RZ→T換算、topology-free depth→実機時間換算、routingや合成精度の影響評価も未実施。q1/q2の限定結果から全RPE費用優位を主張しない。

## 終了と次の判断

状態は **STOP_FAILURE / mandatory STOP**。SOURCE・入力identity・失敗・未達成果物を公開し、remote SHAと公開ファイルbytesを照合して終了する。raw/compiled回路は存在しないため、架空の回路archiveや資源値は作らない。

- **ユーザー／GPT**：contract参照先をexecution worktreeへ修正する新SOURCEと、同じq4/q8 fixture/cut/閾値による新しい一回の実行を許可するか判断する。今回のSTOPを暗黙に解除しない。
- **Codex（新たな明示承認後のみ）**：参照先の最小修正、科学計算を呼ばないcontract解決テスト、SOURCE再固定後に、許可されたraw検証とPASS時だけ最大8 wrapperを実施する。今回の出力・SOURCE・STOPは保持する。
- B2、追加q、新sampling、全RPE費用、詳細科学レビューには進まない。
