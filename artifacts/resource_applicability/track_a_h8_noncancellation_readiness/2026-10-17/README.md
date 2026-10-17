# H8 非相殺bias診断：準備bundle（実行未承認）

[資料入口](../../../../docs/research/track_a_h8_noncancellation_readiness_20261017.md)。SOURCE `d567e6e1f3b1713b455a80da9b5c5f9e7848f2de`。

- `execution_plan_UNAPPROVED_v1.json`：固定science・凍結shot rule・bias区分・STOP/未解決規則・費用優先順位・SOURCE9 file・入力manifest・B2依存（未結合）。
- `grant_INVALID_DRAFT_v1.json`：approved=false/runtime_authorization=false、stage空、入出力・B2 receipt root・CPU・command未指定。
- `source_closure_v1.json`：SOURCE9 fileのbytes/SHA/Git blob。
- `evidence_audit_v1.json`：保存入力22 roleのbytes/SHA、来歴join16項目、PR union6・決定論184・費用入力・B2状態のmetadata照合。科学値・floatを含まない。
- `PR_confirmation_evidence_v1.csv`：PR6候補の成分/端点/normalization/u/登録seed/32 pair/D0行の有無と判定。
- `limited_validation_v1.json`：新54人工tests（Python3.12.3・3.11.0rc1）＋既存回帰113件PASS。
- `draft_rejection_v1.json`：公開planと無効grantを固定runner CLIへ渡し、入力読取前にexit2で拒否。
- `preparation_environment_v1.json`：D0契約と同じPython3.12.3・statistics.py hashへの固定。
- `preservation_receipt_v1.json`：B2保護692 files/127 worktree・既存証拠の前後照合、B2 attempt02 STOPとcgroup PSIの記録。
- `preparation_summary_v1.json`：未実行scopeと次の担当。
- `publication_inventory_v1.json`：この準備で追加・変更した公開filesのbytes/SHA（自身を除く）。

実H8保存値による非相殺bias・shot・費用・ランキング、新signal/compile/sampling/fitは0。
B2 attempt02はSTOP（MEMORY_PRESSURE）でB2全4軸PASSは未受領のため、費用比較stageは実行不能のまま。
private入力snapshot・log・test一時領域はhome配下の準備領域に置き、公開しない。
別sessionのB2保全snapshotがこのworktreeのHEAD/statusを記録したため、資料commitは一時indexで作成し、worktreeのHEAD/statusを動かしていない。mandatory STOP。

## v2：レビュー指摘2点の修正（SOURCE `cbfd7cacad8a5828c0dbfd284a99ef0871860e66`）

v1のfilesは旧SOURCE `d567e6e1…` の証拠として残し、上書きしない。

- `review_fix_receipt_v1.json`：指摘原文・該当code・再現・影響・修正・修正前test失敗。
- `behaviour_diff_v1.json`：人工データでの修正前後比較。計算値は同一、追加はgate記録2件と `S4_model_status`。
- `execution_plan_UNAPPROVED_v2.json`／`grant_INVALID_DRAFT_v2.json`／`source_closure_v2.json`：新SOURCEとplan digest `002f9edb…` への結合。gateはstage別に記載。
- `consistency_check_v2.json`：Git object・manifest・plan/grant・v1 files不変・v1との差分field。
- `limited_validation_v2.json`：人工56件（2 interpreter）＋回帰113件PASS。
- `draft_rejection_v2.json`：v2 draftを固定runnerが入力読取前に拒否。
- `preservation_receipt_v2.json`：元worktree不変、全B2 snapshot照合、並行B2への干渉事案A〜Dの記録。
- `publication_inventory_v2.json`：v2で追加・変更した公開files。
