# RomuTail価格検証統合・4環境配置（2026-11-18）

技術統合の記録。controller SOURCEは `e917165ba1f36b7627614cd25e88b5108d9b20b9`、
release/currentは `romutail-price-fleet-e917165ba` に採用した。
旧currentからの照合付き切替は [採用記録](checks/ADOPTION.json) を参照する。中央状態の正本はRomuTailの
`/home/abe/Project/Partially Randomized Trotter/.rpe-central/state`、
価格検証storeは同baseの `price-validation/models.sqlite3` である。
ローカル0は操作・閲覧clientを維持する。

## 採用元と保全

実際のcurrent `romutail-6da8554e0-r4` とremote後続を確認して着手した。
`6da8554e0040f98578667973e8802c680f87982e` のWindows Git SSH通信、
WSLへのfile-backed mailbox、method view終了時の共有queue維持、Stop時の戻り値、
Stop/Resumeと切断回収を保持した。価格側 `c4e90c9e5ffc7230583567500a99c4ec5c255f7b`
から共有store、保存LD7受入、明示的preparation_budgets、共有モデル公開、metadata-only commit後の再開、
run所有relayを採用した。後続 `7249f008c` の旧test OUTPUT修正と `8ba82646e` の資料も統合した。
合流commitは `79e304e410fcc8daeac37960e64ab04be97e539b`。

追加変更は既存release registry、JobStore、DistributedExecutor、worker profile、relayを拡張した。
新queueや並行controllerは作っていない。固定設定のhash検査も行い、metadataだけのcommitを許容しても
科学実装・設定の差を受け入れない。relayはrunで所有し、各methodの終了では共有queueを閉じない。
旧release・SERVICE・run入力・未確定job・元成果・STOPは保持する。

固定科学SOURCEはQ3 `28d976b99cdf8329adcc1b79327e329e947ef774`、
Q3 timed adapter `349558b748a82f384a4589434a63ef9b5572032c`、
準備wrapper/worker `d887abdcadf47a969aa2f69c1dd3bdc88c20f1c2`、
価格worker `a18d5a04df5ec8c789f274730bff87a2c0a7cc89` を維持した。
provider・AST guard・入力・工程別runtimeは [公開component設定](fleet/RELEASE_COMPONENTS.json) と
各hostの配置設定が対応を記録する。全工程を同一SHAに変更していない。

## 共有価格store

既存の13モデル（保存独立検証10、待機3）を正本として保持し、旧snapshotの9検証/3待機/1未完へ巻き戻さない。
繰り返し取込では既存rowを保ち、要求・owner・attemptの矛盾はそのモデルだけHOLDにする。
通常候補と他の価格検証は継続可能である。

保存PR-Pauli LD2は通常入口と共有storeから再利用し、新しいcompile要求を作らない。
PR-DF LD7はc4の固定SOURCE・係数・receiptの照合を保持する。
保存検証の再読を新規独立検証と数えず、価格資格・係数・順位・総コスト評価は変更しない。
欠けていたq/Rは、固定evidence anchorと元receiptが一致する場合だけ補うopt-in処理とし、
現在のκや適用pointから推定しない。測定値・係数・選択点・所有者等は変えない。
詳細とlive backup/差分証拠は [store記録](store/README.md) を参照。
固定版での最終検査は [store/final-e917165ba](store/final-e917165ba/) に保存した。
live補完は1モデルの欠損6項目のみで、他12モデルはrecord byteまで不変。再補完は0項目。
通常Service.exportの全13モデル・元receipt参照48件（異なるselector42件）は採用済みCと一致し、人工結果は0件。

`scope.preparation_budgets` はsignal/compile/auditを個別に指定する。
Q3予算をcompile許可へ変換しない。compileの単位はstage taskであり、PR標準検証の
独立2経路×X/Yの4wrapperを4stage taskと数えない。通常探索・通常準備・先読みを優先する。
未検証モデルの代表1点、重複防止、低優先queue、復帰回収は既存実装を使用する。

## 配置と操作

一つの公開manifestがoptimizer、準備、Q3、worker、adapter、provider、工程別runtimeを固定する。
環境別path/Pythonはprivate host layoutで解決する。新規Preview/PlanOnly/Startとworker参加時は
同じmanifestを解決し、assignmentに科学REQUESTの外側で要求releaseを付す。
不一致は科学workerへsubmitする前にHOLDとなる。
既存runのResumeはrunに記録されたrelease・SOURCE・設定を選び、新currentへ置換しない。

RomuTail/server2/local0-wslは新しいoptimizer一式を配置し、既存の科学コード・依存環境・入力を再利用する。
不足コードだけを専用配置へ取得する。古いcheckoutをresetしない。
停止中server1は配置packと同じ追随手順を用意し、実機配置・検収は復帰後とする。
配置済みと受付許可を分離し、WSL通常受付OFF・server1停止を維持する。

共通入口 `scripts/run_rpe_search.py` は同じ中央Gatewayへ要求を送る。
RomuTailはlocal、server2は既存RomuTail→qg2接続の逆向きprivate UNIX socket、
WSLは既存Windows Git SSHのRomuTail aliasを使用する。鍵の複製や新認証基盤はない。
中央へ到達できないclientがローカルcontrollerを代替起動することはない。
操作・復旧は [HANDOFF](HANDOFF.md)、[共通入口](common_entry/HANDOFF.md)、
[配置手順](FLEET_HANDOFF.md) を参照。

## 技術検収と限界

[固定版の技術テスト](checks/TECHNICAL_TESTS.log) は141件実施（140 PASS、任意1件skip）。
共有queueのmethod終了、Stop/Resume、価格の低優先・重複防止・回収、明示予算、保存値再利用、
manifest改変拒否、固定設定変更拒否、metadata commit再開を対象とした。
実装者自身と担当sub-agentによる技術確認であり、独立の科学レビューとは扱わない。

変更のない科学worker/profile/入力の既存検収を再利用する。
6daの [稼働中Stop/Resume実機検収](../rpe_romutail_controller_20261118/direct_transport/ACTIVE_STOP_FINAL.json) と
[Windows同期monitor検収](../rpe_romutail_controller_20261118/checks/WINDOWS_SYNC_MONITOR_FINAL_V3.json) を再利用し、
今回のコード差分を固定版の技術テストと新しい共通入口の人工操作で確認した。
新しいcomponent envelopeは保存REQUESTのmetadata検査と人工ジョブで実機を確認する。
[実3環境の人工検収](checks/ACCEPTANCE.json) は準備4件（信号2、共有compile1、別host監査1）とQ3 3件を一度ずつ採用した。
切断中も他hostは進行し、WSL完了結果は元のattempt/内容で回収した。完了Resumeの追加起動は0件。
[実assignment照合](checks/RELEASE_BINDING.json) は全7件で要求releaseと使用SOURCEが一致し、誤ったreleaseは3環境すべてで拒否した。
最初の人工fixtureは宣言した2方式とrequestの1方式が合わずqueue作成前に停止したため、元入力を残して人工fixtureを修正した（[記録](checks/DIAGNOSTIC_FIXTURE.json)）。
人工runは専用registry・queue・spool・出力先に隔離し、Cや利用者の通常マップに混ぜない。
[Windows explicit/default・3環境Preview一致](checks/PREVIEW_EQUIVALENCE.json) は、
同じrelease/SOURCE/runを解決した。PR-Pauli LD2..3、絶対κ[1/10,1]の両端込み区間、
shot auto、signal/compile/audit/Q3予算0を [最適化器が読む解決済み設定](checks/EFFECTIVE_SETTINGS.json) まで確認した。
実機の最終結果は `checks/`、`fleet/`、`common_entry/` の証拠に記録した。
[環境をまたぐ人工操作](common_entry/ARTIFICIAL_COMMON_OPERATIONS.json) では、server2からStop、
WSLからResume、RomuTailから同じStartを再送し、完了runを再利用した。
その後も全LAUNCH、attempt、成果、ASSIGN/ACCEPT/ADOPT数は不変だった。
[旧r4 runのStatus](common_entry/OLD_R4_PIN_STATUS.json) はcurrent採用後も6da/旧releaseを返した。

[Cの通常同期と実閲覧ファイル](common_entry/REAL_MAP_CHECK.json) は、採用済みC
`ca5905e79d7654bdfe42a679b7b5e74aef80b1be` の13モデルと元receipt URLを実HTMLで確認した。
通常Syncのhashは一致し、既に最新のため追加転送は0件。人工結果を公開していない。
開始/再開の既存OnConfirmedRun、実行中10分同期、終了時の最終同期を維持し、別monitorを増やしていない。

server1の実機参加は未確認であり、復帰を今回の完了条件としない。

| host | optimizer/共通操作 | 信号・compile・audit | Q3 | 通常受付 |
|---|---|---|---|---|
| RomuTail | 固定版配置・照合PASS | 既存保存入力検収を再利用、人工信号PASS | 保存REQUEST metadata・人工Q3 PASS | 既存設定 |
| server2 | 固定版配置・照合PASS | 既存保存入力検収を再利用、人工信号/compile PASS | 保存REQUEST metadata PASS | 既存設定 |
| local0-wsl | 固定版配置・照合PASS | 既存保存入力検収を再利用、人工audit/回収PASS | 保存REQUEST metadata PASS | OFF維持 |
| server1 | 同版pack準備、実機は保留 | 復帰後検査 | 復帰後検査 | OFF/停止 |

新しいWSL provider配置は欠けた10個のGit blob（122,180 byte）だけを取得し、旧checkoutは保全した。
工程ごとのコード・runtime・入力の再利用と残不足は [配置証拠](fleet/DEPLOYMENT_RECEIPTS.json) に記録した。
変更したWSL provider配置は、既存のPauli LD2/DF価格2要求（固定LD7 scalarを含む）の
[metadata検査](fleet/WSL_PRICE_STAGE_READINESS.json) もPASSした。
[保全確認](fleet/PRESERVATION.json) は旧r4 SERVICE・新配置の固定設定・実WSL受付OFFを照合する。

新規実H4信号0、実compile0、実Q3 0、本番探索0、価格再fit0。
技術検収は科学性能・価格資格・総コスト評価のGOを意味しない。
公開・current採用の後はmandatory STOPを維持する。
次の科学実行は利用者による対象・範囲の明示指示と§10.5重複照合の後に行う。
