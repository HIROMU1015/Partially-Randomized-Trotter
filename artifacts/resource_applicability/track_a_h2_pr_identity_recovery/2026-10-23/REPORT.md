# H2保存PR単価identity：原証拠回収準備・限定再調査

**PR単価再利用はBLOCKED。可能な技術調査・回収実装・人工回帰を完了したが、O00002の実保存先と完全identity証拠は未取得。科学実行・詳細科学レビューへ進まずmandatory STOP。**

基準commit `4008ce45d26bb858e092d8654187bfcda2dbe180`。専用branch `codex/h2-pr-identity-recovery-20261023`、技術回収SOURCE `9ed3c76c463b7822882bc1827927d5d0fc7c2a40`。旧COMPARE SOURCE `46a3b33e3717c4fadf11846e922caf17d10156d9`、契約digest `a65f3e90b7ed70701f57bcf1519e1d74acdfd1c45c320a5fe91d00d4d44d9ef2`を変更していない。

対象はlinear H2/隣接1 Å/STO-3G/tol-only DF1e-8/rank3、T=.8、ε_sig=.0005、α_axis=.025、P=0、system4/ancilla4。Full S2 `B1_p3_q4_2nd` とFull Yoshida4 `B1_p3_q2_4th` はprefix3、PR S2 `H2O32_c7ec590a32a859bc`/q4とPR Yoshida4 `H2O32_8a2ecdd0294ffb74`/q2はprefix1/K4/outer32。採択済みの共通QR native・経験的allowance・旧FP reference条件付き再利用を保持した。費用比較や順位の新しい結論はない。

## 今回確認したlocator経路

[限定調査](locator_followup_audit.json)と[固定SOURCE根拠](source_boundary_proof.json)を保存した。元outer32の開始/終了Git資料、既存exporter、後続H246確認設定への参照、環境移行STOPを追った。`origin_config_path`から元exploration設定へ戻る経路はSOURCEにあるが、その値を持つ旧private proposal02設定は公開Gitにない。CLI引数で渡されたprivate rootをSOURCEから一意に復元できなかった。

既知のmain project、metadata送付worktree、旧H2 worktreeの直下名だけも確認した。旧private親の直下は2026-10-19だけであり、新しいouter32 rootやtoken mapを示すentryは得られなかった。他ユーザー、server全域、/tmpやagent履歴の横断検索は行っていない。移行時の欠落記録は過去の状況として扱い、現在の全保存先にデータが存在しないとは主張しない。

必要な正確な情報は次のいずれか。これらのpathはtokenized locatorであり、既知の絶対パスではない。

| 原本 | 必要なlocator | 期待SHA256 | byte |
|---|---|---|---:|
| O00002 | `${NEW_EXECUTION}/stagepacks/H2_PR_EXPLORATION/config_APPROVED_PRIVATE.json` | `f011e12b5df6c534fd610899abd2f0f8a17dd05e1cfefaf418a7cb09c0c9a8c7` | 1,331,184 |
| O00001（補助） | `${NEW_EXECUTION}/config_PIPELINE_APPROVED_PRIVATE.json` | `f4da8233474bf8e9f43784e80028948d62d8dd8f43a3f8d82f4c1136ccbdfc0d` | 1,483,023 |

senderのprivate token mapでもよい。exporterの実schemaは `roots: {"/absolute/path": "NEW_EXECUTION", ...}` と `original_sha256`。map自体の保存先も未提供。別環境・未mount媒体・archiveなら、所有者が指定する正確な保存先と対応関係が必要。

## 回収できた証拠と、未取得の証拠

[SOURCE byte照合](source_byte_joins.json)では歴史的に封印されたacquisition SOURCE70ファイルが固定Gitと全て一致した。現COMPAREが使うacquisition8＋provider8の主要計算依存16ファイルも保存封印SHAと全て一致。これは実load、event、回路streamの完全identity証明ではない。

原SOURCEは各taskの `events_PRIVATE.json` に完全eventを保存している。公開exportはそれを取得せず、resultのdigest/stratumだけを投影していた。したがって、現時点の「全文未取得」と「元SOURCEが保存しない」を区別する。全文ファイルの元raw SHAは既存inventoryに登録されていないため、取得後は受入済みresultのevent digestに結合する。

| 項目 | 今回の判定 |
|---|---|
| H/state/snapshot・prep hash・signed schedule | 基準commitのmetadata一致を継承。新prep生成なし |
| 計算依存SOURCE | 16/16 byte一致。元acquisition Git照合70/70一致 |
| 原event全文 | 67ファイル、6,400 event、336 signed piecesの正確なlocator/content digestを特定。本体未取得 |
| archive | 402 manifest、134 compiled raw SHAと保存axis単価のmetadata結合を継承。実archive/raw未取得 |
| operation stream・Gaussian rotation/wire/scalar/wrapper | 原QPY等との完全join未完。SOURCE一致で代用しない |
| runtime | 保存Python3.12と採択3.11の宣言byte差あり。完全なworker実load manifestは公開資料でnull |
| compiler | settings・guard SOURCE・134 guard projectionあり。原file/実使用closure/streamのjoin未完 |
| correctness | 保存された単一入力状態の受入scopeを保持。新閾値での数値再判定・全状態証明への昇格なし |
| cohort・単価 | S2 33 tasks/66 axes、Y4 34 tasks/68 axesのpaired projectionを保持。新平均・u/N/G・順位0 |

[67 task判定](task_identity_67.json)、[134 axis判定](axis_identity_134.json)、[回収件数](request_summary.json)を保存した。原metadata/イベント/archive等2,554ファイルのexact requestを[recovery.json](recovery.json)に列挙した。必要性が未確定な全taskの再compileを計画したものではない。

## 実装とCOMPARE準備

`scripts/h2_pr_identity_recovery.py` は期待metadataを固定Gitから取得するstdlib専用ツール。ownerが明示するroot、O00002、またはtoken mapを受け、O00002の期待byte/SHA・固定67 taskを最初に照合する。元metadataはSHA、全文eventは保存digest/順序/component/phase/schema、archiveは圧縮SHAと伸長raw SHAで照合する。伸長量は元manifestのbyte数に結合し、QPY/NPZを解釈・再保存せず、原fileも書き換えない。NaN/duplicate key・path traversal・symlink逸脱・既存report上書き・他worktreeへのreport出力を拒否する。

原設定・raw archive・event内容を公開Gitへコピーしない。回収toolは成功したfileを記録しても完全identity witnessやgrantを生成せず、部分回収をPASSにしない。実運用のroot指定経路は人工fixtureで検証し、今回は原root未提供なので実原本の回収0。現在のdefault実行は公開metadataのrequest作成のみ。

専用worktreeで `python3 -I -S -B scripts/h2_pr_identity_recovery.py --report NEW_REPORT.json` を使うと、private原本を開かずに回収一覧を再作成できる。ownerから保存先が示された後だけ `--config /owner/provided/.../stagepacks/H2_PR_EXPLORATION/config_APPROVED_PRIVATE.json`、`--execution-root /owner/provided/root`、`--token-map /owner/provided/map.json` のいずれか一つを追加する。reportは同worktree内の新規pathを指定する。これらはmetadata/byte照合であり、科学実行への入口ではない。

[COMPARE preflight](COMPARE_preparation_status.json)は元の封印済みlauncherをmetadata-onlyで再検査し、technical binding確認後 `BLOCKED_PR_IDENTITY` / execution_ready=false。診断4/path8/reference4は再実行しない。旧SOURCE/契約を新branch向けに書き換えず、完全witnessがない段階で新しい実行可能契約を発行しなかった。

将来COMPAREのexact予算は旧契約のまま：bootstrap1/setup1/prepare2/bundle_load1、Full native_validation2/compose2/lowering10、wrapper4/compile4/transpile4/simulate12、allowance4/accounting4/ranking1。diagnose/path_signal/reference_rebind/sampling/request_factory/rte_event/occupation_trajectoryは各0。現在の実消費は全0。

## 残る技術判断と最小追加案

rootが判明しても自動で実行可能にはならない。現行の厳格runtime identityに対して宣言byte差が残り、完全な歴史的実loadをSOURCE宣言や新規に作ったJSONで代用できない。compiler関連closureへ同等性条件を限定するなら、その証拠・政策の明示採択・新SOURCE/契約が必要。今回は条件変更をしていない。

[最小追加案](minimal_additional_validation_DRAFT.json)はまず科学call0で原本を回収・照合する。確認されたstream/compiler差だけからaffected task/axis集合を確定し、全67の再sampling/recompileへ自動移行しない。単なる欠測はconfirmed mismatchではない。

別承認が得られた場合の限定probe例は、S2 sample0（128 event/4 piece）とY4 sample0（64 event/6 piece）の計2保存軌道・4軸。bootstrap1/setup1/prepare1、archived_request_load2/event_metadata_deserialize192/occupation_trajectory2、native_validation2/compose2/lowering10、wrapper4/compile4/transpile4/simulate12、sampling/request_factory/診断/allowance/会計/順位は0。これは新実装・SOURCE・固定契約が必要な未実装DRAFTで、live予算ではない。原event回収とRNGなし復元の人工検証が前提。2代表の成功だけでは67 task全体の単価再利用を証明できず、全cohortに必要な追加proof/call数は未確定として残した。

## 検証・保全・終了

[人工回帰174件PASS](test_results.json)（新36＋既存138）、[全ログ](technical_tests.txt)。source/config/hash/event/原archiveの改変拒否、部分回収時BLOCKED、未承認科学importなし、予算/旧grant/診断非重複の既存回帰を確認。実H2 native/compileを通したという意味ではなく、Codex主担当の技術チェックである。

[保全監査](preservation_audit.json)は旧原本454＋基準成果物37＝491項目PASS。実行へ直接結合する40項目を必須照合し、残る451項目は観測として比較して、今回は全て一致した。無関係な資料更新を停止条件にはしない。旧封印10,767ファイル・SOURCE271ファイルのruntime/preflightもPASS。DIAGNOSTICS SOURCE35444654…、元契約f71e52ce…、completion cc7ebfee…、結果・grant・ledger・marker・STOPは保持。共有環境/venv/他job/他worktreeを変更せず、恣意的な資源上限も追加していない。

[最終照合](final_verification.json)と[成果物inventory](file_inventory.json)に、SOURCE/テストhash、旧manifest項目、限定変更範囲、リンク、STOPとCOMPARE禁止状態を記録した。

担当ユーザー：O00002のcanonical locatorまたはsender mapの保存先を提供。担当Codex：別作業で原file回収と全field joinを行い、厳格identityの成否と残る政策選択を具体化。担当ユーザー：必要な政策変更または追加科学witness/COMPAREについて、具体的なSOURCE・対象・exact予算を確認して別途明示承認する。

今回は新科学計算・保存科学値の数値再解析・sampling・compile・u/N/G・再ranking・live grant・詳細科学レビューは全0。専用branchへlocal commitし、push/PR/mergeなしでmandatory STOP。
