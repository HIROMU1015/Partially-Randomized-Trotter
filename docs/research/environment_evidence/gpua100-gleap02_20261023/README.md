# GPUA100-gleap02：PR資源評価の既存証拠集約（2026-10-23）

公開承認待ちのローカル共有候補。専用branch `codex/pr-evidence-gpua100-gleap02-20261023`、公開main `0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf`を起点とする。新規科学計算、保存値の再解析・再ランキング、実験再実行、論文執筆、pushは行っていない。

## 論文データ整理担当の入口

- [公開済み固定資料の索引](public_index.csv)：275ファイル、22の取得可能commit。GitHub treeのblob/sizeと固定ローカルbytesのGit blob/SHA256を結合した。代表8ファイルはGitHubから本文を全取得してbyte一致も確認。別の科学SOURCE索引40 pathでは37 pathがGitHubと一致し、H6 run02の3 pathはローカルのみ。全runtime閉包を新たに再監査したものではない。
- [ローカルのみの既存資料](local_only_index.csv)：H246独立確認の10ファイル。result commitはGitHubで取得できなかった。公開済みreadinessを実行結果と扱わない。
- [系列・条件・cohort](series_manifest.json)、[SOURCE対応](source_result_links.csv)、[科学SOURCEの固定path/SHA](science_source_index.csv)、[不足・次担当](limitations_and_requests.csv)、[検証範囲](remote_verification.json)。索引は既存公開資料の参照だけで、公開済み数値表は複製していない。
- [新規共有候補の来歴](candidate_provenance.json)：H246の6ファイルは固定ローカルGit原本とbyte完全一致。H6 run02の単価64軸は保存result32件の既存fieldをそのまま転記し、completion witnessの各result SHAと照合した。

## 優先して参照する固定資料

|系列|固定commit|必要な資料|
|---|---|---|
|H4/H6/H8既存独立確認とH2診断・原証拠回収|[e7a63b2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e7a63b2b06a124f9305483aa39bcf1ba8cafdd0d)|主表6比較、H2 diagnostic projection、回収依頼|
|H2 outer32取得|[a05f4ed](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a05f4ed19e87e1eb65f7fc346385b5b4eae69845)|525探索trajectoryの取得summary・終了監査参照|
|H4/H6 outer32取得|[605ca07](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/605ca07a653241db49348fbd7f4e091f9f18ced7)|793/607探索trajectoryの取得summary|
|H246保存値評価|[8613b3e](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/8613b3ecda2030a677b1690fe7aa3da23dec4873)|1000候補精度行、30公式比較、36単価統計、60費用分解、終了監査|
|H2 metadata export|[e353195](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e3531956af6a6d359e2f1fb320e9a5179f41eb98)|67 taskのmetadataと公開guard参照|
|既存技術監査|[df7beb8](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/df7beb8588a3866a7dac2f29add4be7493d092db)|希少event、domain、費用定義、H6来歴、Claim–Evidence Matrix|

H246独立確認run01のローカル固定commitは`0d3578a4584a924ed10bdd124db6093e02d90c94`、科学SOURCEは`8825391ff73b531be5834bf852b9bf2479d1799b`。H2/H4の保存確認結果とH6のSIGBUS・NOT_EVALUATEDを同時に保持する。[6候補ファイル](candidates/h246_confirmation_run01/confirmation_results.json)は未公開原本の共有用projectionの転記であり、今回数値を再計算していない。

後続H6 run02は科学SOURCE`01d12e3384f8c06a31cf9da0c8149c86ec4b6ee2`。[単価表](candidates/h6_recovery_run02/axis_costs_64.csv)と[receipt projection](candidates/h6_recovery_run02/receipt_projection.json)を用意した。保存native exit0とunit-cost completionは確認したが、run rootの`end_audit.json`は未発見。SOURCEもGitHubで取得できなかった。固定確認の平均・SE・改善率を新算出せず、最終PASSへ昇格していない。

## 数値・定義・未解決事項

数値と単位の正本は各固定表。Gは既存の`N_cos C_cos + N_sin C_sin`というanalytic-shot会計、単価は保存compiled RZである。H6新候補表のrz_count/cx_count/circuit_sizeは保存整数、depthは元の回路深さ、N2/N4は保存event label件数。単位換算や平均化は行っていない。bias・経験的allowance・normalization・headroom・shot式は各系列の固定SOURCE/政策を参照し、旧系列から新系列へ無条件に移送しない。

探索、pilot、旧FP、独立確認、H4 DF12/DF10を混合しない。H8のMechanism aliasは追加独立標本ではない。H4粗精度のNOT_REPRODUCED、H6粗精度のPF低費用、missing/null、H6 SIGBUS、非相殺の不足成分を保持する。finite-sampleの点比較・保存SEを母平均保証や厳密bias証明へ変えない。古いouter32政策未採択STOPと、その後の保存値評価完了も別来歴として保持する。

## privateと公開承認

O00002は期待SHA256に一致する原本をこの環境で発見した。原config・内部絶対パスは公開候補へ含めない。[非機密な状態](private_evidence_status.json)のみ記録し、full historical stream/runtime/correctness identityが解決したとは主張しない。

private設定、seed原本・full event、raw QPY/NPZ/archive、private grant/ledger/marker/STOP、絶対locatorは別の非公開監査領域に保持する。公開候補にはsource/result SHA・失敗経緯・標本単価と対応IDが含まれるため、その開示はユーザー判断を要する。[開示検査](disclosure_review.json)がpattern上PASSでも公開承認の代わりにはならない。

hostnameは現在観測した環境。ローカル保存だけを根拠に全既存計算の実行hostを断定しない。今回の索引は既知の研究領域と既存回収依頼に限定し、全historical rawの網羅的監査ではない。[保全・検証結果](verification.json)も参照する。

次の担当：ユーザーがこの専用commitの候補ファイル全体の公開可否を判断する。承認後の別作業でCodexが対象branchだけをpushしremote内容を照合する。科学計算・H6 retry・論文執筆の承認にはならない。mandatory STOP。
