# Phase 2D-F1：技術的停止と部分的な科学検証結果

**F1_TECHNICAL_FAILURE_PARTIAL_RESULTS_MANDATORY_STOP**。

承認された一回のF1を実行したが、最初のSS compile caseでQiskit metadata処理のTypeErrorにより停止した。retry／resume、SOURCE修正、追加科学処理は行っていない。F1全体の正しさはまだ検証完了していない。

固定4mode・2粒子・L_D=1・残差2fragmentの合成fixture。分子geometry／AO basis／分子DF rankは非該当。T=1/5、q2、delta=1/10の一点、r2、h1/20、K2、full16／controlled32。保存B入力とF1-PRE SOURCEを変更せず再利用した。

- H-IDはSS/SP/PS/PPの4件すべてPASS。
- SSのactual component由来finite meanはone-step／全体ともPASS。SSのM0 raw／optimized／actual adjoint、X wrapperもPASS。
- LAW metadata 47,889件、component54件、motif3件、wrapper1件、catalog controlled60件を完了した。これらの単位は異なり、科学call総数へ合算しない。LAWのpartition終了時mass gateは実行・保存されておらず、全LAW_PASSとはしない。
- 独立referenceは506precision evaluations中109件を計算し、54組の90/128整合を確認。残る1件は90digitsだけで128評価前に停止した。
- compileは1requestをattemptし0completed。例外はtranspiler呼出し前なのでactual transpiler invocationは0。
- catalog直接価格は30recordのみ。DP／比較、physical arm期待価格、selector／全列挙／fallbackは未実行。価格認証、shot数、資源優位性のPASSはない。

[実行結果・残差・件数](EXECUTION_REPORT.md)、[引継ぎ・STOP](handoff.md)、[機械可読要約](../../../artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24/F1_RESULT_SUMMARY.json)、[非公開rawのhash台帳](../../../artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24/OUTPUT_HASH_MANIFEST.json)を参照。

次の担当はユーザー／GPT。metadataの技術修正と、その後の新しいF1実行を承認するか判断する。今回のgrantは消費済みであり、旧結果を置換したり自動再実行したりしない。F2と重要科学レビューは未承認。
