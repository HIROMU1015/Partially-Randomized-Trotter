# H8 B1：完了済み正本と保存STOP受入契約の比較

B1の正本は **`83a3c4a6688e66576f85471bf41a45e29cbfb705`、全4層PASS・完了済み**。
本書は正常完了gateを要求した準備案と、実際に使用された独立の保存結果受入契約の差分資料である。
正本・実行済み契約・認可・科学的結論を変更しない。新しい受入承認手続きに置換せず、B1を再実行しない。
M1-S V0は別研究であり、SOURCE・実装・契約・成果物を本作業へ取り込まない。

- [完了済み正本の結果報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/83a3c4a6688e66576f85471bf41a45e29cbfb705/docs/research/track_a_h8_phase_b1_run01_results_20261017.md)
- [正本の実行契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/83a3c4a6688e66576f85471bf41a45e29cbfb705/artifacts/resource_applicability/track_a_h8_phase_b1_run01/2026-10-17/execution_contract_v1.json)
- [機械可読の比較・証拠・限定検証](../../artifacts/resource_applicability/track_a_h8_b1_stop_acceptance/2026-10-17/README.md)

## 元run02のSTOPと保存8軸

[元STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af10b55420d31e227905f7762971e4b39c6c6a05/artifacts/resource_applicability/track_a_h8_b0_anchor_run02_stop/2026-10-16/STOP_assessment_v1.json)は
`STOP_AFTER_EIGHT_SAVED_MEASUREMENTS`、取得契約はSTOP、最初のobserver理由は `RuntimeError:PROCESS_THREADS`。
driver exit −9、launcher exit 1、終了時のprotected-worktree監査は未完了、元の正常完了gateは存在しない。
違反PID・thread数・executableは未記録で、Gitが原因だったと断定できない。過去のthread遵守を認定しない。

一方、4構成8軸のcompile workerは全てexit0・reap済み。TRAIN6/HOLDOUT2、未確定compile予約0、固定呼出会計を保存している。
[事後監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b19c337f0b16513d68b59eb3366dc6eff175c313/artifacts/resource_applicability/track_a_h8_b0_saved_anchor_audit/2026-10-17/saved_anchor_audit_v1.json)は、SOURCE/input/runtime/preparation/compiler、前後fingerprint、control、原config/grant、終了・保全証拠の結合を確認した。
その監査の状態 `SAVED_EIGHT_AXES_STRUCTURALLY_REUSABLE_PENDING_B1_APPROVAL` は**監査時点の履歴**であり、後続B1の完了を否定する現在状態ではない。
事後監査は元run02をPASSへ変更せず、失われた正常完了gateを補作しない。

今回、固定commit上の14資料をbytes/SHAへ結合した。B0原本53ファイル（9,402,930 bytes）、B1原本34ファイル（8,068,879 bytes）は旧公開inventoryとstreaming SHAが一致。
原config/grantのraw SHAとB1のcanonical SHAも公開bindingに一致した。
これは限定した保存証拠の照合であり、venvの34,273ファイル等を再走査したという意味ではない。
科学arrayの読込・数値fit・holdout誤差の再計算は行っていない。

## 並行して存在したB1実装

|項目|実装A：正常完了receiptを要求する準備|実装B：正本run01で使用済み|
|---|---|---|
|SOURCE|`1e25d5691b2a5c4738ca8b67c398ae06b70c4e0f`|`6fb8a0a71986fda571d82b5658686f5e74a32f8b`|
|公開資料|準備 `21e3d13469b508545b28415b13eaf25c60a9d38f`|結果 `83a3c4a6688e66576f85471bf41a45e29cbfb705`|
|入口|`h8_phase_b1_model.py` と validation/runner/launcher|`h8_phase_b1_fit.py` と `h8_phase_b1_execution.py`|
|scope|`H8_B1_NEW_B0_8_B0I16_FIT_VALIDATION_V1`|`H8_B1_REUSED_B0_RUN02_B0I16_FIT_V1`|
|B0受入|元取得COMPLETE、exit0、final gate PASS等が必須|固定保存STOP監査＋保存8軸採用を明示した別grant|
|元run02への挙動|正常完了gate欠測により拒否する設計|元STOPを保持したまま保存成果物だけを受領して実行済み|
|来歴確認|24正規化軸のprep/input bindingをfit前に確認|準備段階の原証拠・preparation監査と、数値moduleのeligible検査を組み合わせる|
|B2出力|数値予測を出力しない|事前登録4軸のモデル予測のみ。compile認可false|
|source closure|準備17ファイル|実行19ファイル。今回Git blobと全19 SHAを照合|

共通祖先は `847266323ea3d7bd29ac42322b7819b0b2a6c43f`。
両者はS2の `a+b*q+d*prefix*q`、Fractionによる3列OLS、method×axisの独立4層、固定B0 TRAIN3/HOLDOUT1、B0+I各軸8点のtraining残差診断、measuredを分母とする各軸2%以上STOPを共有する。
静的な契約・source比較であり、出力schemaやbytesの同一性、実行全経路の同値を新たに証明したものではない。
grant/config/model schemaとscopeが異なるため、相互に流用しない。各outputのone-shot lockも、両実装をまたぐ新しい実行許可にはならない。
既存runnerは統合・改変していない。実装Bが正本の使用SOURCEである事実を記録し、今回新たに実装を選定する操作は行わない。

## 受入契約案と実行済み契約の差分

[比較JSON](../../artifacts/resource_applicability/track_a_h8_b1_stop_acceptance/2026-10-17/acceptance_contract_comparison_v1.json)は新しいgrantや起動gateではない。
正常完了経路を緩和する案を採用せず、実行済みの別経路が何を要求したかを整理した。

1. 対象はSOURCE `db559a518214fd27fa96e2f1451911df05c4f4a8` のrun02保存B0 S2固定8軸だけ。旧B0未解決16軸、H6係数は混合しない。
2. 元STOP・欠測final gate・thread原因未解決を保持する。worker全exit0/reap、固定TRAIN/HOLDOUT、controls、prep、SOURCE/profile/input/compilerと原本bytesを別監査へ結合する。
3. 実行済みB1 grantは `accept_saved_B0_with_original_STOP=true` を明記した新scopeのone-shotだった。raw SHAは `13c124893d97df5218627ab6abe146088f99227b7438b2d6b78988e50b2adbff`。
4. その公開認可記録は「その方針で進めて」と対応するB1実行文脈を保存している。今回の利用者指示に従い正本83a3…を完了済みとして扱い、この認可の取り直し・追認手続きは作らない。
5. B1 run01自身のterminal、exit0、cleanup、保全PASSは元B0 run02の状態と区別する。既存model canonical SHA `8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0` は保存bytesから照合した。fit・残差は再計算していない。

H8直線1Å/STO-3G/16 modes/sector4900/DF28/T=.8、S2 p14/21、B0 q1/8、B0+I q1/2/4/8、科学条件・compiler optionsは不変。
完全profile `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`、compiler `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` を維持する。
既存契約のdriver/observer各AS512MiB・RSS256MiB、driver CPU240秒・observer120秒、wall600秒、admission1GiB等も変更しない。ここに記載した過去の上限は新しい実行認可ではない。

## 今回の固定物と検証範囲

新しいmetadata照合コードのSOURCEは **`ccbe5653361271cf6afd659a916df360a1cb1f75`**。
これはB1数値実装SOURCE `6fb8a0a7…` と異なる資料照合用SOURCEであり、正本のSOURCEを置き換えない。
新module・CLI・人工metadataテストの3ファイルだけをsource closureに固定した。CLIにfit・grant発行・起動機能はない。

限定21 tests PASS。STOP保持、gate補作拒否、重複/欠測軸、profile/prep/fingerprint不一致、worker exit/reap/identity/hash、control非finite、原model hash、既存B1 cleanup、認可再開や二重fitの抑止を人工receiptで確認した。
単一test process/thread1、AS512MiB/CPU120秒/wall60秒/output4MiB、temp/logはhome配下。主担当のローカル確認であり独立担当reviewではない。既存の大規模検証campaignは反復していない。

資料専用branchは `track-a-h8-b1-stop-acceptance-reconciliation-20261017`。
既存SOURCE・結果・grant・ledger・marker・lock・他worktree・private venv・共有環境へ書込なし。
V1とV2 CLOSED_NO_GOを維持する。新科学計算・fit・holdout数値検証・compile・費用再会計・grant・原稿改訂・subagentsは全て0。
B1完了からB2/CやM1-Sへ自動移行しない。資料の公開、または認証失敗時のpush待ち報告後にmandatory STOPする。
