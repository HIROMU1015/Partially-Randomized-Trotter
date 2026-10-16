# H7中性開殻：準備完了・実科学計算未開始

- SOURCE：`f93ad35b05a75c5e17e2c53b0f3de2af4fc443f2`
- branch：`track-a-h7-neutral-open-shell-readiness-20261016`
- [説明・条件・実行計画](../../../../docs/research/track_a_h7_neutral_open_shell_readiness_20261016.md)

Neutral linear H7 / 1Å / STO-3G / ROHF共通MO / 7e・Nα4Nβ3 / 14modes・sector1225 / T.8/q1、tol-only1e-8/cutoff0/signed order/weighted DF。実rank未知、prefix ceil(L/4)。H5と同じ6構成・最大20group/40compile/16trajectory/32occurrence、通常4並列/S4単独・thread1・hash/preparation/paired evolution reuse。

**54人工/mock検査PASS（53件＋追加1225高速経路1件）。実H7科学call・grant・one-shot消費0。** 別担当独立reviewや実H7成功を示すものではない。

| 資料 | 内容 |
|---|---|
| [技術検査結果](technical_validation_v1.json) | PASSの範囲と未検証事項 |
| [53件log](artificial_tests_final.txt)、[追加1件log](artificial_1225_fast_path_test.txt) | H5回帰とH7人工検査 |
| [人工検査計画](artificial_test_plan_v1.json) | 単一process・thread1・資源/出力上限 |
| [実行計画](execution_plan_v1.json) | 二段階pilot・call・資源案・STOP |
| [固定契約draft](execution_contract_draft_v1.json) | 科学条件/対象/未認可 |
| [binding draft](binding_draft_v1.json) | SOURCE/profile/carry固定、input/output未seal |
| [source closure](source_closure_v1.json) | source267fileのblob/byte SHA |
| [環境summary](environment_profile_summary_v1.json) | 14817file・H5時と同じprofile |
| [保全監査](source_preservation_audit_v1.json) | SOURCEは9追加のみ、旧source/results不変 |
| [carry draft](carry_binding_draft_v1.json) | 旧actual/failed/reservation/unknown保持・reset/refundなし |
| [rank依存call例](rank_dependent_call_plan_examples_v1.json) | rank予測ではない静的例 |
| [host読取snapshot](readonly_host_snapshot_v1.json) | 資源予約や将来の起動保証ではない |

実H7のrank、spin純度、reference/native全coverage、時間・RSS、PRの精度/費用優位性は未知。科学計算の別承認とfresh gate、新H7専用one-shotが必要。`approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`、`absolute_launch_command=null`、未seal。

NPZ・runtime/cache・credential・内部SSH情報はGitに含めない。H7の実行開始や全面mapに進まずSTOP。
