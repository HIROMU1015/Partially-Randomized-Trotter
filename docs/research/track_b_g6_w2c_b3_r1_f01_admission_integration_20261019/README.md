# W2c-B3：F01・R1 Admission・hostの技術統合

2026-10-19 JST。**TECHNICAL_INTEGRATION_COMPLETE / FINAL_SCIENCE_RELEASE_PENDING_PERFORMANCE_DECISION**。
F01修正版driverを固定Git objectから取り込み、R1 admissionと外側STOPを新hostで接続した。
既存209件＋新66件＝**275件PASS（FAIL/ERROR/SKIP各0）**。実G7 science・実authorization/claim発行・消費は**0件**。
これは高速化変更前のR1技術baseline。科学実行用の最終release・正式契約digest・実承認は未確定。

基点 `1c15bacab349b8686c09d2981e1c0bc70e9c3ec9`、F01 SOURCE `a777a2ab54d85a95dce91a06e65d4ec0511403ae`、F01公開最終 `07f65831f045bf1fdbd4b370e44c7f59a9901ddd`。
専用branch `track-b-g6-w2c-b3-r1-f01-admission-integration-20261019`。無条件merge/rebaseなし、他worktree/index/lock/process/未commit成果へのアクセスなし。

- [SOURCE差分](integration_source_diff.md)
- [F01/Admission/STOP契約](f01_admission_integration_contract.md)
- [host・実承認の境界](host_and_authorization_boundary.md)
- [275件の人工回帰](integrated_artificial_regression.md)／[machine結果と全test ID](artificial_test_results.json)
- [技術SOURCEとSHA](technical_release_candidate.md)／[未認可の技術契約候補](technical_contract_candidate.json)
- [高速化担当への引き渡し](performance_handoff.md)／[Q1–Q8・次の担当](handoff.md)
- [SOURCE inventory](source_inventory.json)／[manifest](integration_manifest.json)／[保全](verification.json)／[公開receipt](publication_receipt.json)
- [新host](../../../src/trottertracks/algorithm_codesign/g6_w2c_b3_host.py)／[新統合tests](../../../tests/tracks/algorithm_codesign/test_g6_w2c_b3_host_integration.py)
- [v2 schema差分](admission_host_v2.schema.json)／[読み取り検証器](verify_integration.py)

対象G7 P5_general_order、p=(1/5,3/10,1/2)、x=5/7、m=5、σ=+1、4方式×A/B×X/Y。
A6 math-only（τ10^-6 rad/Kφ32/J32）、既存seed/setup/cache/N/二pass/保存scheduleを維持。
分子・geometry・basis・DF rank・split L_D・δ窓は非該当。固定workloadの記述的古典費用比較の準備。
母期待費用confidence、物理RZ保証、実CPU優位、科学的GOは追加していない。公開・保全後 **mandatory STOP**。
