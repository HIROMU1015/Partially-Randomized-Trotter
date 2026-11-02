# H6 s2条件付き費用pilot：準備・未承認の入口

**技術準備完了、実trajectory/compileは未開始。別途明示承認までmandatory STOP。** 対象は `B2_p5_q8_2nd_r2_K2` のwhole-trajectory s2（16 occurrenceにorder2が1個以上）、条件付き8本・測定込みcosine/sine計16 wrapper。

SOURCE **843265b624eac8fbb9acea47ef21cab1207e1880**、専用branch `track-a-h6-conditional-s2-pilot-preparation-20261013`、基点 `eee1343f6e99aa7661892ddfbe1067e974e2b74a`。新SOURCEにはsource/testsだけ、REVIEWには下記軽量準備資料とindexを固定する。

- [実行契約・固定科学条件・sampling・残るリスク](../../../../docs/research/track_a_h6_conditional_s2_pilot_execution_contract_v1.md)
- [技術seal](sealed_v1/manifest_pilot_v1.json)、[未承認authorization草案](sealed_v1/authorization_draft_v1.json)、[8登録task/seed](sealed_v1/pilot_registrations_v1.json)
- [SOURCE closure257・旧251 blob不変](source_closure_v1.json)、[profile/入力33 byte/元proof/carry整合](technical_binding_review_v1.json)、[read-only auditor結果](binding_audit_v1.json)
- [条件付き法則・80新seed・母平均未保証](conditional_sampling_plan_v1.json)
- [58人工/mock tests](test_results_v1.json)、[実行計画](limited_test_plan_v1.json)、[log](limited_tests_v1.log)、[JUnit](limited_tests_v1.xml)、[default dry](default_dry_run_v1.log)
- [CPU/memory/PSI/OOM/FS/inode/quota読取snapshot](resource_snapshot_v1.json)、[容量・block/inode/cache余裕](capacity_plan_v1.json)
- [将来の起動template・fresh gate・新grant・mandatory STOP](execution_handoff_v1.json)
- [公開/再取得証跡](publication_receipt_v1.json)（公開後の別receipt commit）

既知消費compile1700/trajectory834、旧予約20/10を保持。新16/8で必要上限**1736/852**、1720/844からの変更は未承認。失敗返却/reset0、unknown actual=null、STOP時の未受理groupも全額予約。

資源提案：worker1・CPU20・AS32/RSS9 GiB・thread1、coordinator22/23・ASsoft2/hard32/RSS2、aggregate40/RAM64 GiB。output2 GiB/log1 MiB/journal32 MiB、fresh free10 GiB/100000 inode。5秒監視・pressure/OOM/headroom・own-run STOP/cleanupを元H6から維持。現snapshot合格は将来の起動許可ではない。

既存科学224/旧runtime251、592探索/256確認、environment/compiler/input/signal/B/shot、旧roundoff受理とbit同一性未検証を保全。新caseは別seed/phase/output/grantで分離する。原component確率/量子event重みは変更しない。旧supervisorのgroup leaf名は新conditional parent内にだけ残る。

実H6のs2 compile・peak RSS/wallは未測定。8本では複数order2や希少componentを網羅できず、mean上界/正式winnerは未証明。新H6 science/actual transpile/grant、H4/他job/共有環境/venv変更0。NPZ/native runtime/checkpoint/cache/credential/SSH情報を今回commitしない。

次回承認対象は **この固定SOURCE・8条件付きtrajectory/16 compileを一度だけ実行、compile1736/trajectory852と記載資源・STOP政策の採用**。承認後に新exact grant・byte SHA・remote/fresh/未使用output/lockを結合する。自動retry/resume・次stage/H8への進行なし。approved=false、runtime_authorization=false、allowed_cpus=[]、absolute_launch_command=null。


## 実行後の追記

利用者承認後の一回runは8/8・16/16完了、native exit0・cleanup/残存0。[結果入口](../../track_a_h6_conditional_s2_pilot_result_v1/2026-10-13/README.md)。上の未承認記載は準備時点の履歴。SOURCE/科学条件/旧消費・予約を保持し、追加実行なし・mandatory STOP。
