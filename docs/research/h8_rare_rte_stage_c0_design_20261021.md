# H8希少RTE event費用評価：段階C-0技術調査・実行契約案

2026-10-21 JST。**C-0設計・人工検証・technical preflight完了 / MANDATORY STOP**。新しい分子sampling/signal/simulation/SCF/DF/state/回路生成/compile、保存値の新規数値解析は全0。有効な科学実行grantを発行・消費していない。

## 結論と今回の到達範囲

H8 p8/q2/r4/K4とp7/q16/r4/K4の未観測3層を対象とする、条件別の未承認契約案を保存した。S2_SINGLEを第一候補にし、multi/S4は未知項として必ず残す。既存静的資料は再利用可能だが、現証拠だけで両層のfull-wrapper compiled RZを十分に制限できたとは判断しない。追加費用情報を必要とする主張では、新しい同一条件のwhole-trajectory両軸compileが候補になる。

[条件別入力・B保存値](../../artifacts/resource_applicability/h8_rare_rte_stage_c0/2026-10-21/conditions_reference.json)、[必要層一覧](../../artifacts/resource_applicability/h8_rare_rte_stage_c0/2026-10-21/required_information.csv)、[再利用・不足の分類](../../artifacts/resource_applicability/h8_rare_rte_stage_c0/2026-10-21/evidence_reuse.csv)。必要n・誤差・信頼水準・完全な数値law処理・新SOURCE/runtimeは未確定なので、technical preflight PASSを科学実行準備完了へ格上げしない。

## 固定条件と実取得した証拠

共通：linear H8、隣接1 Å、STO-3G、tol-only DF28、T=.8、P=0、二次partial PF、通常signed bias、allowance1、元native/scalar/control/full Hadamard wrapper、Qiskit1.3.0/rz,sx,x,cx/optimization1/seed_transpiler17、backend/coupling/targetなし。候補・PF・analytic shots・元Pを変更しない。

|条件|PR候補|原Pの保存観測|段階Bの保存U（概数）|保存境界/S0平均（概数）|
|---|---|---|---|---|
|ε=.05|B2_p8_q2_2nd_r4_K4|n48、S0のみ|.007160|5.22|
|ε=.0005|B2_p7_q16_2nd_r4_K4|n48、S0のみ|.008808|4.88|

この表は[段階B結果](pr_rare_rte_stage_b_results_20261021.md)の保存値転記であり、新しく計算していない。層確率・mean/SE・shots・PF費用は機械可読条件表に元の文字列のまま保存した。今回の順位再評価は0。

基準`3d5ab5bb1cf0d78a3157ad7f742cfa5c7b58efbb`、先行調査`42ed0876c0505502428629c1fb4863233247ebaf`、H8確認SOURCE`745a674546f2cb6258526fbfde1d9a00601689f8`、packet`a6246f03c6297438ab624fc62e5061a58e53145f`、原profile補足`5f267e96ea313f373eb78a75040a43fac3a2ee1b`を起点に取得・照合した。GitHubから6固定fileを実取得しGit blobとbyte一致。固定Git 33 blobをcacheへ取得してSOURCE・law・native・並列・監査を調査した。[来歴SHA](../../artifacts/resource_applicability/h8_rare_rte_stage_c0/2026-10-21/source_provenance.json)、[remote取得](../../artifacts/resource_applicability/h8_rare_rte_stage_c0/2026-10-21/remote_acquisition.json)。取得ファイル全ての科学的正当性を再レビューした意味ではない。

既知のprivate H8 indexのみで、公開input登録123行に対応する82原ファイルをSHA照合した。snapshot/state/preparationをdecode・再構築していない。profileは5f267e96で**原契約内JSONとして取得確認済み**、canonical digestとhistorical file bytesを混同しない。旧packetのprofile未取得欄は当時の履歴として保全する。PR回路本体0/192軸・旧fingerprint再導出不能・旧compiler出力bit同一性未確認も維持する。[input保全](../../artifacts/resource_applicability/h8_rare_rte_stage_c0/2026-10-21/input_binding_audit.json)。

## 既存証拠で代替できる範囲

|証拠区分|再利用できる内容|不足・適用しない主張|
|---|---|---|
|数学的関係|元finite-law normalization/Taylor誤差規約、Bの層質量という保存評価|normalization/operator誤差boundはcompiled費用上界ではない|
|条件付き静的compiler事実|Tier R L1：特定pipelineで完全basis translation後総gate数Tを最適化が増やさないという既存source調査|native命令数や理論RZをTへ置き換えない。実行時library/Rust/全input閉包は別問題|
|部分的L2|固定template map下のtranslation加法性・native controlled-RZZ経路の同定|全family/rule/再帰closure未完成。CRZZ理論モデルと実CCX–CRZ–CCXの違いを維持|
|解析的gate-countモデル|nativeモデルA/Bの期待値API、basis再利用・phase幅の構造|特定分解のproxyであり、full wrapper実compileの数値上界・certificateではない|
|保存compiled実測|今回PのS0両軸、元PF直接単価、旧q4/q8 conditional pilot|p/q/cohortが違うrare費用を移送・poolしない。保存最大は全support上界でない|
|経験推定|Bのpaired mean/SE/条件付き境界、旧pilotの精度/負荷設計の参考|未観測supportの費用・分散・厳密coverageは得られない|

Tier Rは`bf8c19057fd82c900c6234152e152e79cb52e707`の`A_rare_event/report.md`を再利用した。[先行調査の整理](pr_rare_rte_existing_evidence_20261021.md)。新しい全support boundの導出・採択・数値評価は行わない。将来、完全closureと妥当な重みが得られて残余寄与を制限できればmulti/S4のcompileを減らす案を優先する。現段階ではその代替を充足済みとしない。

旧H8 p7/q4,q8のsingle/multi/S4 conditional pilotは取得済みで、存在しない扱いにはしない。しかしp8/q2とはprefix/qが異なり、p7/q16ともqが異なる。新費用・分散の代用として数値移送しない。既存資料の再取得以外の重複科学解析は0。

## 条件付きsamplingと標本設計

[詳細設計](h8_rare_rte_stage_c0_sampling_inference_20261021.md)に、全multinomial count→一様位置→元component選択→元_make_event→元native loweringの接続を記した。S2_MULTIをN2=2、S4をN4=1へ固定しない。層別化は古典的費用標本の取得だけで、無条件PR法則・量子係数・normalization・shotsを変更しない。

新しい層平均を原P S0と同じ無条件期待値の別項として会計する。原P/旧pilot/新pilot/productionはrowをpoolしない。paired Gの層別分散に加え、原S0の標本誤差も評価式に含める。必要nは許容G誤差・信頼水準・分散・compile負荷で決める。分散不明なら承認済みvariance/load pilotで一度止め、その後にproduction nと新予算を固定して別承認する案が基本となる。

今回、具体n・信頼水準・h・precision/toleranceは未設定。pilotを自動実行したり、結果の符号が分かるまで追加する停止規則は採らない。multi/S4に妥当な制限がなければ統計的取得または未知項の保持が必要。有限pilotの結果だけで全support上界にはできない。

## 並列化・高速化・入力共有

[具体的実装計画](h8_rare_rte_stage_c0_parallel_20261021.md)は、trajectory process並列と両軸fork/COW、親でのpreparation/state一度復元、全体tokenによるnested並列制御、identity固定cache、worker別atomic I/O、task ID順集計、native wait4/孫回収を採用候補とする。

旧loaderは保存blocksからprefixのtailを純粋に再構築し、p7/p8のexact hashを照合する。元receiptは両prefixの`保存prefix完全一致export=false`を記録しているため、「完成済みprefix archiveを読むだけ」とはしない。将来のdecode/prefix復元も予算と承認へ含める。C-0ではprovider import・復元をしない。

read-only資源snapshotを保存した。CPU affinityは空きcoreの保証ではなく、現在のPython/package metadataも歴史的H8環境の同一性を証明しない。full runtime34,273 file・provider931 fileの過去照合receiptは参照したが、live環境の全closure・SessionEquivalenceLibraryをC-0で再認証していない。worker数・RAM/wall capは固定せず、将来fresh resourceと登録taskのprivate memory実測から動的admissionする。共有設定・他process/研究worktreeへ変更0。

## 条件別の未承認契約と科学gate

- [粗精度契約案](../../artifacts/resource_applicability/h8_rare_rte_stage_c0/2026-10-21/contract_H8_0.05_PROPOSED.json)
- [細精度契約案](../../artifacts/resource_applicability/h8_rare_rte_stage_c0/2026-10-21/contract_H8_0.0005_PROPOSED.json)

各案は`approved=false/runtime_authorization=false`。有効grant・approval receipt・実行SOURCEはnull。別conditionに別grant/ledger/marker/STOPを使い、同時実行と開始順は明示承認まで未設定。compile中心のCであっても旧control APIが行うsimulation/oracle等を無断で起動しない。必要な検証callも承認範囲と予算へ明記する。

symbolic予算はtrajectory N、event qrN、occurrence qN、evolution N、測定wrapper2N、compile miss最大2N等を区別し、control・prefix復元・会計・数値照合は別枠。実call数はnと検証設計が確定するまでnullである。予約は科学call前に非返還ledgerへ記録し、cache hitも記録、失敗後同grant retry/resumeなし。既存STOPは保持する。

次のユーザー承認事項は[一覧](../../artifacts/resource_applicability/h8_rare_rte_stage_c0/2026-10-21/approval_requirements.md)にまとめた。契約案の未確定値を埋め、新SOURCEへ本番bridge/schedulerを接続・人工検証し、live runtime/input/preparation/profileを照合してからでなければ、科学実行承認を求められる状態にはならない。

## preflight・人工検証・保全

[technical preflight](../../artifacts/resource_applicability/h8_rare_rte_stage_c0/2026-10-21/preflight.json)は**PASS_C0_METADATA_DESIGN_SYNTHETIC_ONLY**、科学実行readinessは**BLOCKED_PENDING_SEPARATE_APPROVAL_AND_CONTRACT_COMPLETION**。実分子への本番integration testを実施していない。

人工15件はR=3・p=(1/2,1/3,1/6)のtoyのみ。count lawを全word列挙で照合し、multi/S4支持保持、同event両軸、cache/registry分離、O_EXCL mock予約、全体resource token、toy child exit0/7の同親wait4回収を検査した。[テスト記録](../../artifacts/resource_applicability/h8_rare_rte_stage_c0/2026-10-21/synthetic_tests.txt)。toyの成功・失敗各1子processは回収済み、実科学worker0。担当Codex自身の技術検査であり、独立担当者の科学reviewではない。

旧Bのgrant/ledger/marker/input/inventory5fileと原input82fileを保全し、旧科学SOURCE/raw/results/STOPは書き換えない。既存索引は旧bytesを末尾に保持してC-0入口だけ追加する。SOURCE/結果SHA・構造/リンク/private patternの限定検査と終了監査を別receiptに保存する。raw/内部locator/実seed/認証情報は公開候補へ含めず、新GitHub push0。

専用branch `codex/h8-rare-rte-stage-c0-20261021`。成果物をローカルcommitして**mandatory STOP**する。次の担当は**ユーザー**：推論目的・精度/信頼水準・pilot/本番の承認範囲・S0/残余層・公開範囲を判断する。**Codex**は明示承認まで新sampling/compile/保存値解析・詳細科学review・最終freezeへ進まない。
