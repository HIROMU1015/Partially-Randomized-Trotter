# H4 run03 最終認可・一度の起動artifact

入口：[本計算認可とruntime入口](../../../../docs/research/track_a_h4_production_run03_20261009.md)。
SOURCE `6e68fd9bcc68e788db6f5d43eaa6a03866e53d3b`、closure44。

- [利用者の上限変更・本計算起動認可](USER_AUTHORIZATION_v3.md)
- [sealed plan](plan_authorized_v13.json)、[認可](authorization_v13.json)、[別担当の最終review](review_v13.json)
- [独立最終整合review](independent_authorized_run03_review_v1.json)
- [最終binding](authorization_binding_v13.json)、[未実行argv](runner_argv_v13.json)
- [先の技術review・SOURCE/profile/input/STOP/carry](../2026-10-09/README.md)

累積17GiB/74805承認済み、carry21/8692723164B/5766.582514658794s upper返却なし。
全map charge bound17429694796B、余裕823916212B。人工compile省略。
SOURCE/profile/input/carry一致・独立review・固定artifact・直前fresh資源/容量/quota・未使用one-shotが合格後、一度起動。
driver16/observer18、worker12 CPU2/4–6/8–15、own-run affinity・内部thread1。observer AS256MiB/RSS64MiB、admission120.25GiB。
完了/STOP後終了、自動retry/次stage/旧partial-cache混合なし。runtime原本はhomeのprivate evidenceに保持してcommitしない。
