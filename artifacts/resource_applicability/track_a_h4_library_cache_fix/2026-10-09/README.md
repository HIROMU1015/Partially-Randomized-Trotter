# H4 library cache fix・再実行条件

一つの資料入口は[修正・限定検証・予算と承認案](../../../../docs/research/track_a_h4_library_cache_fix_20261009.md)。
SOURCE `b8b3ce6e8c98f1ec0419a7af79c5d7c5f3a3b9bb`、closure36。

- [旧/new source blob/hash](source_freeze_v1.json)、[library cache profile](library_cache_profile_v1.json)。
- [限定47回帰＋3 import cases](limited_tests_v1.json)、[独立review](independent_fix_review_v1.json)。
- [STOP/carry receipt](stop_receipt_v9.json)、[plan](plan_draft_v9.json)、[auth](authorization_draft_v9.json)、[review](review_draft_v9.json)、[binding](binding_audit_v9.json)。
- [再実行承認案](relaunch_proposal_v1.json)、[commit対象](commit_inventory_v1.json)。

library import修正PASS、累積予算12.261694GiB>承認10GiBでlaunch条件FAIL。
未seal、approved/runtime_authorization=false、再起動0。既存CPU/environment/observer/actual74804の承認を保持。
13GiBは未承認proposalであり、SOURCE現gateは10GiBを強制する。
cap改定が明示承認されれば、そのgate更新・新SOURCE固定・binding・独立review・fresh gateが必要。
raw dependency cache/NPZ/runtimeはGit外、旧証拠・one-shot・全課金を保持する。
