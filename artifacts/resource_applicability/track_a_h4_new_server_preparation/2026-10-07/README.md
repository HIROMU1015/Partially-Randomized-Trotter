# H4 新host取得・監査・解決案（2026-10-07）

現在状態: **H4_NEW_HOST_SOURCE_RECEIVED_ENVIRONMENT_DECISION_REQUIRED_STOP**。
Git取得とsource/environment読取監査は完了。source修正・新人工tests・production準備の完了は宣言しない。

[全体説明](../../../../docs/research/track_a_h4_new_server_preparation_20261007.md)、
[exact source/environment](exact_source_environment_audit_v1.json)、[Git receipt](git_source_receipt_v1.json)、
[資源/容量](resource_readonly_observations_v1.json)、[環境解決案](environment_resolution_proposal_v1.json)、
[監視/serialization実装案](SOURCE_FIX_PLAN_v1.md)、[未seal binding草案](binding_draft_v1.json)を一組で読む。
`requirements_old_reference_v1.txt`は旧45 versionsのreferenceで、install認可や旧environment同一性を意味しない。

originのrun05 branchは指定8f77bebf99c5bd15fa6419c1f58556c3bd2837a9と一致。
補足資料branch c0a5b9692778878865cfa31d7df9a3bcc259f6c4をsource変更なしで取り込んだ。
source19/installed source11は全て一致する一方、既存private venvは18 version differences/45 RECORD differences。
compiler defaults/plugin metadata一致はfull environment/compiler/binary equivalenceの証明ではない。
6 NPZ/freeze/runtime/controlは未受領。旧75 testsは新hostで再実行していない。

20 invocations/165214360 bytes/5466.188392877579 sをcarry、残74764。
allowed_cpus=[]、approved=false、new source/selected Python/input/output/launch commandは未固定。
observer追加roleと候補120.25GiB admissionは未承認で、既存上限は変更していない。
本計算・入力再生成・GPU・共有環境/他job変更・install/upgrade・worker/observer起動なし。
