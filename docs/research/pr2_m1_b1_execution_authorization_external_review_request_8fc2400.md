# GPTへのPR-2 M1-B1 execution authorization最終レビュー依頼

## 依頼

PR-2 matched-accuracy resource studyのM1-B1について、実行sourceを先に固定した後で作成したresult-prior
execution authorizationをレビューしてください。このreviewでは契約、source identity、plan、resource cap、
terminal statusだけを確認し、trajectory sampling、circuit build、compile、held-out accessは実行しないでください。

## 固定identity

- repository：`HIROMU1015/Partially-Randomized-Trotter`
- M1-A result commit：`3c1831e326c27c5f679b3820997f27916d26ed9f`
- M1-A result SHA-256：`1f960d7a33296e2dcb74d497e360572b26409dc9aeae01522335e7b91ed81086`
- M1-A result fingerprint：`422f898bba1e3849d0f45830082b76d4f42da436e2b49796e562cd79fc716c9e`
- actual execution source commit：`33f436bb3a7d5b9cefa23604bb22c8d1fb17cd62`
- authorization bundle commit：`8fc24000b49b6cdb146c3f084891a2f42898214f`
- execution plan v2 SHA-256：`5afc94fac0571b38c74b0b00cfcf68e34e491a5fc65ef579d3ec884d079e5aa5`
- execution plan fingerprint：`17c91d41e77d7c085629b60ea470abc87e9590f448ba9bcdfa4342410cd89607`
- execution authorization JSON SHA-256：`7d188f782354609fec1ea83e289e9872c6a801c3e0e9493f93753fe7cae9d93a`
- result schema v2 SHA-256：`035d9b1e48d8d02718f8c7218ea49d28297b3b3f29c39c4232cdef83c7c1d081`
- focused/benchmark tests：`92 passed`
- authorization gate：PASS、science counterは全て0

主要ファイルは次である。

- `docs/research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md`
- `docs/research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md`
- `src/trotterlib/pr2_matched_accuracy_m1_b1_execution.py`
- `scripts/run_pr2_matched_accuracy_m1_b1.py`
- `tests/test_pr2_matched_accuracy_m1_b1_execution.py`
- `artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/pr2_matched_accuracy_m1_b1_execution_plan_v2.json`
- `artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/pr2_matched_accuracy_m1_b1_execution_authorization_v1.json`
- `artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/pr2_matched_accuracy_m1_b1_result_schema_v2.json`

## 前回reviewの2修正への対応

1. 12,448 wrapperを実際に生成・compileするmodule/runner/testをsource commit `33f436b`へ先に固定した。
   planとauthorizationはその子孫commitで作り、source byteが変わればrunnerが科学計算前に拒否する。
2. result schema v2のterminal statusを
   `M1_B1_COMPILE_MAP_COMPLETE_AWAITING_REVIEW / IMPLEMENTATION_GATE_FAILED`だけにした。
   science runnerは研究四分岐を選ばず、成功時も`research_decision=null`で停止する。

旧result schema v1はzero-compute監査履歴として保持するが、authorizationはv2だけを指定する。

## 今回認可する範囲

- M1-A適格random B2 145＋B3 49を追加・除外せず使う。
- 各random cellは32 trajectory、同じtrajectoryをcosine/sineで共有し、12,416 wrapper。
- B0 12＋B1 4は全件二軸で32 wrapper。不適格B0 4件はfrontierへ入れない。
- 総上限12,448 full wrappers、最大6 spawned workers、各BLAS thread 1。
- signal再評価、追加96、held-out、transfer、winner精密化、S3、GPU、研究自動判定は禁止。
- source/compiler/candidate/axis/trajectory identity一致だけを再利用し、cross-cell reuseを禁止する。
- 32-trajectory actual compiled resource map完成時に必ず停止する。

## 確認してほしい点

1. actual execution sourceをauthorizationより前のcommitに固定した順序で、前回指摘を解消しているか。
2. planのsource hash集合とauthorizationのsource hash集合を完全一致させるgateが十分か。
3. benchmark実装が実際に生成するtrajectory seed列をplanへ固定し、cosine/sineで共有する方法が妥当か。
4. candidate別persistent cache、axis/trajectory別wrapper identity、cell task fingerprint checkpointでunsafe reuseを防げるか。
5. 194 random＋16 baseline、12,448 wrapper、6 workers、32 trajectory停止がbounded pilotとして妥当か。
6. result schema v2の2 terminal statusと`research_decision=null`が、結果後の恣意的自動分類を防ぐか。
7. このbundleのまま一回のM1-B1を実行してよいか。

## 回答形式

次のいずれか一つを先頭に示してください。

- `APPROVE_M1_B1_EXECUTION`
- `REVISE_M1_B1_AUTHORIZATION_BEFORE_EXECUTION`
- `STOP_OR_NARROW_BEFORE_M1_B1`

その後、重大な問題、必要な最小修正、実行前に追加固定すべきidentity/resource/testだけを列挙してください。
review回答だけでは本計算を開始せず、`APPROVE_M1_B1_EXECUTION`を受けてから固定commandを実行します。
