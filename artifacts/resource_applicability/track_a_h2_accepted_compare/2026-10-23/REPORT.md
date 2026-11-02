# H2受入済み診断からのCOMPARE：技術実装・PR identity監査

**専用COMPARE分岐の実装と人工検証、実metadataの受入・封印準備を完了。PR単価再利用は `BLOCKED`、比較科学実行は未承認・未実行。local commit後mandatory STOP。**

branch `codex/h2-accepted-compare-20261023`、基準a6788bf710c16e179a644263952c19711c64a2b2。新SOURCE `46a3b33e3717c4fadf11846e922caf17d10156d9`。旧実行SOURCE35444654…、DIAGNOSTICS契約f71e52ce…、completion cc7ebfee…は不変の正本として扱った。

対象はlinear H2/1 Å/STO-3G/tol-only DF1e-8/rank3、T=.8、ε_sig=.0005、α_axis=.025、P=0、system4/ancilla4、固定Full S2/q4・Y4/q2/prefix3、PR S2/q4・Y4/q2/prefix1/K4/outer32のみ。通常signed biasと共通QR/経験的allowance/旧FP referenceの条件付き政策を継承する。科学的な費用比較や順位の新しい結論は出していない。

## A. 元設定・PR証拠の取得と判定

[locator監査](locator_audit.json)、[送付worktreeの限定確認](sender_locator_check.json)、[67 tasks/134 axesのfield判定](PR_identity_67_tasks_134_axes.json)を保存した。既知Git exporter、origin inventory、受領metadata、旧H2の既知rootだけを調査し、送付branchの登録worktreeも特定した。送付rootの直下と.gitignore、固定READMEに正確なtoken mapの保存先は得られなかった。取得SOURCEの`--private`も必須CLI引数で固定defaultではない。

`${NEW_EXECUTION}`のcanonical rootは引き続き未取得。O00002は`${NEW_EXECUTION}/stagepacks/H2_PR_EXPLORATION/config_APPROVED_PRIVATE.json`、期待SHA `f011e12b5df6c534fd610899abd2f0f8a17dd05e1cfefaf418a7cb09c0c9a8c7`。補助O00001は`${NEW_EXECUTION}/config_PIPELINE_APPROVED_PRIVATE.json`、期待SHA `f4da8233474bf8e9f43784e80028948d62d8dd8f43a3f8d82f4c1136ccbdfc0d`。これらを既知絶対pathとみなさず、他private領域・他server・agent履歴の探索はしていない。

追加回収した公開Git e3531956…のstorage_records.jsonは1,269,353 byte、Git blob `0ffb12f0e52d382e4b1e99dc8a74d9d2030663c4`、SHA `29c5388af211c730ca4f0f7e981261e9c36a37547c893f02016e180c160a9642`。公開projectionのまま[evidence_sources](evidence_sources/storage_records_e353.json)へ保存。元private設定やcompressed/raw archiveを新Git履歴へ追加していない。

| 項目 | 今回の判定 |
|---|---|
| H/state/snapshot | 保存metadataとして67 task全て一致 |
| prefix1 prep hash | 新DIAGNOSTICSの85b8243d…と67 task全て一致。ただしhash単独で完全identityとはしない |
| signed piece/time/rte_steps | 保存lawと新診断scheduleが67 task全て一致 |
| archive/raw SHA | 公開402 archive metadataを回収し、134 compiled raw SHA全てが保存axis cost receiptと一致。実archive bodyのローカル照合ではない |
| 新common rotation/operation stream | DIAGNOSTICSのreceipt取得済み、hash acb9c34e…。元PR streamとの完全結合は未完 |
| event sequence | 保存event digestとN2/N4/stratum要約はあるがfull body未取得。SOURCE上のnested ordered events digest規則を確認 |
| runtime | 保存Python3.12 binary e50d468e…と採択3.11 ebc2fae9…に宣言byte差。当時worker実load証拠は未取得 |
| correctness / compiler / scalar / wrapper / lowering | 保存state限定acceptanceとsettings/guard資料はあるが、全fieldの新系列とのidentity joinは未完 |
| cohort / paired RZ | S2 33 tasks/66 axes、Y4 34 tasks/68 axesをそのまま保持。新平均/u/N/G/順位評価0 |

一致・宣言上の不一致・未知を分離した。runtime宣言差だけで全比較が不可能とは断定せず、現在の保存C再利用条件が未成立と判断する。全134 raw compiled SHAは互いに異なり、現metadataから完全byte一致による縮約は得られなかったが、全軸の再compileが必要だとは決めていない。

## B. 新COMPARE実装

[受入仕様・exact予算](ACCEPTANCE_AND_CALL_CONTRACT.md)と専用launcher・受入検査・control・backend・準備器を実装した。旧runnerや実行済み契約は編集していない。新backendへ旧数値methodを固定コピーし、旧diagnose/reference入口を除去、Full-only native guardを追加した。

元契約/completion/元SOURCE・input/runtime、原grant/approval/marker/ledger、4候補と2 prep、JSON/NPZ SHA、candidate/prefix/prep/operation、adaptive schema、raw/corrected列を検査する。現在はNPY headerと整数basis metadataだけを読み、複素signal vectorのdecode・再評価はしない。[実metadata受入](origin_acceptance.json)でこの区別を記録した。

将来の科学承認後に限り、`bundle_load=1`で保存vectorを読み、fresh setup/prepのidentityへ結合してFull correctness/4候補会計へ渡す。元path/referenceを再実行しない。消費済み元grantを新権限にせず、新SOURCE/契約/承認/unused grant/run_rootを要求する。

PR witnessは旧16 identity fieldに加え、原accepted resultのhash、完全event payload、各fieldのhistorical-to-common証拠fileを必須とする。policy名・prep hash・settings hashだけの受入は拒否。会計欠測・不適格ならrankingへ進まずpartial STOP。

## C. exact契約準備と実行状態

[実パス入りBLOCKED契約](COMPARE_contract_BLOCKED.json)はSOURCE46a3b33e…と10,767 fileを結合した。契約digestは `a65f3e90b7ed70701f57bcf1519e1d74acdfd1c45c320a5fe91d00d4d44d9ef2`。[metadata-only preflight](preflight.json)は技術bindingを確認し、PR witness2件不足により `BLOCKED_PR_IDENTITY` / execution_ready=falseを返した。科学import、live grant、実run_root作成は0。

予算はbootstrap1/setup1/prepare2/bundle_load1、diagnose/path_signal/reference_rebind各0、Full native_validation2/compose2/lowering10、wrapper4/compile4/transpile4/simulate12、allowance4/accounting4/ranking1、sampling/request_factory/rte_event/occupation_trajectory各0。内部callを含むため単純合計を独立実験数と解釈しない。曖昧な追加reserveはない。

元prep objectが必要なためprepare2は残すが、既存の8 pathは再計算しない。追加witnessが必要な場合の[最小案](MINIMAL_WITNESS_PLAN.md)は、原本metadata回収を先行し、影響task/axisを確定した後の別契約とする。全67 taskの自動再取得はしない。2候補sample0/4軸の代表probe例も全cohort証明ではなく、今回の実装・live予算には含めていない。

## D. 検証・保全・次の担当

既存98＋新40＝**人工138件PASS**。[全ログ](technical_tests.txt)、[結果](test_results.json)。人工decoderでload/adapter復元を通し、診断重複0、元grant再利用/入力改変/closure欠測/型不正/予算超過/partial/不適格/PR証拠欠測の拒否を確認。旧数値methodのAST一致と旧execute拒否を回帰した。人工temp内以外のgrantは発行していない。Codex主担当の技術チェックであり、独立科学レビューではない。

[保全監査](preservation_audit.json)の454項目全てPASS。旧SOURCE・元契約/completion・原結果/NPZ/prep・旧approval/grant/ledger/CONSUMED/STOPと既存科学入力の照合を保存した。原本・他worktree・共有venvは変更しない。元DIAGNOSTICSの正常completion/旧STOPを保ったまま、新しい技術準備STOPを保存する。

[最終照合](final_verification.json)と[成果物inventory](file_inventory.json)にSOURCE271 file、封印10,767 file、旧manifest項目の不変性、公開projectionの固定Git byte一致、新grant/run/cacheの不在を記録した。

必要なユーザー情報はO00002のcanonical locatorまたはsender token map。担当ユーザーが情報を提供し、担当Codexが別作業として原event/archive/runtime/operation/correctness証拠を結合する。完全identityが成立した場合だけ新witnessを封印し、新契約digestと正確な対象/予算についてユーザーの比較実行承認を受ける。runtime証拠/同等性の扱いが解消しなければBLOCKEDを維持し、必要なら最小追加witnessの別契約を提示する。

今回は新しい科学計算、実分子回路build/compile/sampling、保存値の新しい数値解析/u/N/G/順位、科学レビューを全て0で終了。専用branchへlocal commit後mandatory STOP。push・PR・mergeなし。
