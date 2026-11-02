# B3採択準備：Angle・実行policy・Admission/STOP

2026-10-19 JST。状態：**TECHNICAL_ADOPTION_PREPARATION_COMPLETE_PENDING_USER_DECISIONS**。
既存F3の数学的条件から具体的Angle候補3組を整理し、既存schedule/seed/保存を維持した新しい未採択契約を作成した。
既存driver・科学SOURCE・旧契約は変更していない。本番admissionは未実装・無効のまま。
実G7 correctness/sampling/CPU profiling、保存科学値再解析、B1/B2、compile/synthesis、実authority/claim発行・消費は**0件**。

直接基点：`e4af6c064db805cbec611bcc8f0ff75a4754d38b`。
専用branch：`track-b-g6-w2c-b3-adoption-readiness-20261019`。
必要な41件の固定公開資料はGit blobとbytes/SHA256を照合した。取得元commitを基点と区別して記録した。

- [Angle具体候補と十分条件](angle_parameter_candidates.md)／[機械可読表](angle_parameter_candidates.json)
- [seed/setup/cache/X/Y/保存/二pass](execution_policy_adoption_candidates.md)
- [external admissionとSTOPの準備](one_shot_admission_and_stop_readiness.md)
- [新しい未採択契約](execution_contract_candidate_refined.md)／[JSON](execution_contract_candidate_refined.json)
- [Q1–Q7・次の承認文](handoff.md)
- [証拠索引](evidence_index.json)／[source inventory](source_inventory.json)
- [新規人工30＋既存84の検査](artificial_test_results.json)／[preflight](preflight.json)
- [準備guard・人工atomicモデル](../../../src/trottertracks/algorithm_codesign/g6_w2c_b3_readiness.py)／[新規tests](../../../tests/tracks/algorithm_codesign/test_g6_w2c_b3_readiness.py)
- [技術・保全確認](verification.json)／[公開receipt](publication_receipt.json)

対象は既存G7 P5_general_order、p=(1/5,3/10,1/2)、x=5/7、m=5、σ=+1、4方式、A/B、X/Y。
分子・geometry・basis・DF rank・split L_D・δ窓には非該当。G9/P1-final監査は実施していない。
固定環境の記述的古典実費比較が目的。母期待費用・一般計算量・native/End-to-End優位は対象外。

候補R1の採択内容は最小1件にまとめられる。科学実行の明示承認は別に必要。
本番portの実装・固定・正式契約digest確定は、その作業を承認された後に行う。
現在の候補は`adopted=false / contract_grants_execution=false / separate_execution_approval=false`。
公開照合・保全後、**mandatory STOP**。
