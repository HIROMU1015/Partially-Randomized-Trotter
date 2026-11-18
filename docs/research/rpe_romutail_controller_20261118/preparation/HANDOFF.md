# PR-DF保存入力の補充と工程別inspection

2026-11-18。担当：Codex sub-agent `prep_inputs`。独立科学レビューではない。

RomuTailとlocal0-wslのPR-DF signal・compile・auditの各3セルは、保存入力の配置後に
`SAVED_INPUT_RUNTIME_INSPECTION_PASS` となった。新規信号・compile・Q3・本番queue投入は0。
[工程×host表](HOST_STAGE_READINESS.json)、[入力の出所](INPUT_PROVENANCE.json)、
[RomuTail結果](local-prdf-dry/READINESS.json)、[WSL結果](local0-wsl-prdf/dry-prdf/READINESS.json)。

元 `dea1cb530` のserver2 12 PASS、RomuTail/WSL各9 PASSは保存記録を再利用し、再検査していない。
再検査したのは不足していたPR-DFの計6セルだけ。
これにより保存入力metadata inspectionはserver2/RomuTail/WSLで各12 PASS。
server1は停止中のため接続せず、12セル未確認を維持した。
**inspection PASSは、任意の新しい完全科学REQUESTの実行admissionを意味しない。**

## 補充した既存入力

- descriptor：`62ee29f280bd4774b208259b5ee73c64229cb9cd:`
  `docs/research/h4_pr_multiwidth_resume_20261117/run01/completion/candidates/503d195e3975103a4f12/DF_TAIL_DESCRIPTOR.json`。
  SHA256は `5a3e24d591922c49a7c447584210323439e14eafaf964568e5da6af216e4ac14`。
  qg2の既存 `codex/b2-rpe-pr-multiwidth-resume-20261117` refからGit objectsを取得した。
  WSLへは前提commit `5829a728...` を持つ4.7 MiBのGit bundleで転送し、元commitを保持した。
- 専用config：元の `config_APPROVED_PRIVATE.json` をbyte同一で配置。
  SHA256 `52e221c7e3d56dc8c75f5df0b208c60bb0c3d3c8885c14a331e7cdc86c4e24c4`。
- provider：元の固定manifestに列挙された278ファイルだけを取得し、全件hashを照合した。
  configを含む279ファイル、約6.8 MiB。元ファイルの記載path・過去の契約・STOPを変更していない。

config/provider本文はGitへ追加していない。認証情報や新しい承認証明は作っていない。
元configの過去の許可を新規実行許可へ転用しない。

## 再利用する配置

RomuTailのprivate入力：

```
/home/abe/Project/Partially Randomized Trotter/.romutail-controller/preparation-inputs/
  descriptor.bundle
  df-runtime.tar
  runtime/runtime_input_config_52e221c7e3d5/config.json
  runtime/runtime_provider_root_b714e6f05181/
  PROFILE.json
  PLACEMENT.json
```

WSLのprivate入力：

```
/root/projects/Partially-Randomized-Trotter-local0-wsl/.romutail-controller-preparation-20261118/
  runtime/
  PROFILE.json
  TASKS_PR_DF.json
  dry-prdf/
```

Windows転送先は `C:\Users\romuk\rpe-romutail-preparation-20261118`。
RomuTailから既存 `Romulaptop` SSH経路でWindowsへ接続し、`wsl.exe -d Ubuntu -- python3 -`
を使って保存入力配置・metadata inspection・結果回収を実施した。
WSLの既存科学checkout・固定transport f956・元profileは変更していない。
local0-wslの `.local0-wsl/SETTINGS.json` は `accepting_new_jobs=false` のまま。
[受付観測](local0-wsl-prdf/PARTICIPATION.json)と[最終配置記録](local0-wsl-prdf/PLACEMENT.json)にhashを保存した。

新しいprofileは [RomuTail用](PROFILE_local.json)、[WSL用](local0-wsl-prdf/PROFILE.json)。
各profile内の `preparation_paths` と工程別 `preparation_profiles[*].profile.preparation_paths` に
`runtime_input_config_52e221c7e3d5` / `runtime_provider_root_b714e6f05181` のhost別pathを追加した。
科学JSONの絶対pathは書換えず、既存path resolverを使う。

## 新しい所有場所またはserver1復帰時

`scripts/prepare_romutail_preparation_inputs.py` は、この保存inspection用の279ファイルの集合・hashと
Git input hashesを照合し、所有場所へ配置してprofileのコピーを作る。入力packageは上記private保存場所から取得する。
元profileやSOURCEファイルは変更せず、Git bundleはobjects/専用refだけを追加する。
新規科学REQUESTや `dry_check_only` のないtaskは受理しない。

```bash
python3 -B scripts/prepare_romutail_preparation_inputs.py \
  --tasks docs/research/rpe_romutail_controller_20261118/preparation/TASKS_PR_DF.json \
  --profile /ABS/EXISTING_PREPARATION_PROFILE.json \
  --package /ABS/df-runtime.tar --git-bundle /ABS/descriptor.bundle \
  --repository /ABS/REPOSITORY --directory /ABS/OWNED_INPUT_DIRECTORY
```

server1復帰時は、既存の固定f956 transportと工程SOURCE・Python/compiler配置を確認し、
同じ入力mappingを作成して、新しいinspection directoryへ既存12 taskを検査する。
今回server1をrequired-host待ちや自動再試行へ追加していない。

今回再検査で使った既存入口：

```bash
/ABS/EXISTING_PYTHON -B /ABS/FIXED_F956_TRANSPORT/scripts/check_rpe_preparation_host_readiness.py \
  --tasks /ABS/TASKS_PR_DF.json --profile /ABS/OWNED_INPUT_DIRECTORY/PROFILE.json \
  --output /ABS/NEW_INSPECTION_DIRECTORY
```

配置helperの人工検査は2件PASS：正しいhashのmappingと元profile保全、変更されたconfigの拒否。
保存ログは [TESTS.log](TESTS.log)。元inspectionのSOURCE・入力・compilerの照合を緩めていない。

中央管理の統合版へ移行する際は、これらhost path mappingを新profileへ引き継ぎ、
新しい完全REQUESTについて通常のSOURCE/fingerprint/admissionを行う。
今回のinspectionを実科学性能・新Q3 PASS・実compile成功と呼ばない。

## 準備入力の転送量

中央が準備jobのGit入力を、checkoutに共通の履歴と、そのjobが参照する固定入力に分ける。
先行jobの結果refは、現在の入力が参照するときに同梱する。同じcheckoutと入力集合から同じbundleを作り、
既存のSHA-256キャッシュでworkerに不足しているbundleだけを送る。workerは共通履歴を先に取り込み、
その後で固定入力とSOURCEを照合する。旧jobを再開するときは保存済みのREQUESTとbundleを使う。

保存済みPR-DFのREQUESTを元のcheckoutと固定SOURCEで再構成した。以下は同じキャッシュへ届けるGit bundleの
合計サイズで、転送時のJSON等を除く。

| 対象 | 保存済みbundle | 新方式 |
| --- | ---: | ---: |
| 同じ固定入力を使うdescriptor 4件 | 180.0 MiB | 45.3 MiB。2件目以降は0 bytes |
| 較正1件の初回 | 66.6 MiB | 81.4 MiB |
| 信号1件の初回 | 110.1 MiB | 140.0 MiB |

保存6件で、固定SOURCEだけの独立したGitリポジトリから新規workspaceを作り、
元REQUESTの固定入力を延べ80件照合した。人工検収では、別候補の履歴を同梱せず、
候補が変わっても共通履歴を再利用できることと、旧bundleを読み込めることを確認した。
初回の容量は入力の履歴に依存する。表の較正と信号では、外部デルタ参照を使わないpackの容量が増えた。
実workerへの配置とcurrent採用は、末尾の容量対策releaseを参照。

## workerの実行ディレクトリ容量

準備jobは実行ごとに独立した作業領域を作る。事前照合で固定したGitオブジェクトは、worker内の
`preparation-metadata-cache` を各作業領域から参照する。展開済みの作業ファイルと新しい成果commitは
各attemptに保存する。共有Gitキャッシュは、参照する作業領域が残っている間は保持する。

保存PR-DF descriptor `prep-e30f836cc9686b7c646df3ba` の入力で、独立した固定SOURCEから新規作業領域を
作成して比較した。従来は作業領域ごとにGit packを47,521,081バイト作成した。変更後は2つの
作業領域のどちらにも入力packの複製がなく、同量をworkerの共有キャッシュに1回保持した。
両方で固定SOURCEと入力が一致した。人工jobでは別のprovider作業領域からの成果公開・回収も確認した。
対象は新規attemptの準備Git pack。既存attemptの容量削減には、回収済み成果と参照の別途照合が必要。

## 容量対策releaseの配置

2026-11-18 JST。RomuTailで `romutail-worker-storage-dd832d360` をcurrentに採用した。
中央とworker通信の固定SOURCEは `dd832d360ee600335b81280b0891b55ade44b5d7`。
前current `romutail-work-conserving-5c8ced4b2-laptop4x4` の固定SOURCEに容量対策のみを取り込んだ。
科学Q3は28d976b99、timed adapterは349558b74、準備wrapperと各provider・価格workerは既存契約を使う。
PBS gatewayと回収用Python、ホストごとの受付・資源設定を引き継いだ。

- 中央は受領済みファイルをハードリンクで公開する。同一ファイルシステムでリンクできない場合は既存のコピー経路を使う。
- 新規準備jobは、事前照合で取り込んだGitオブジェクトを各workerのキャッシュから参照し、実行に使わない `inputs/` の複製を省く。
- 作業開始時に照合したGit commitを基点に、workerはそのjobで増えたGitオブジェクトを返す。
  中央で前提が不足すれば、元REQUESTのSHA-256で固定された入力bundleを先に取り込む。
  開始時と終了時のcommitが同じ場合は、そのcommitの参照を返す。旧方式の返却bundleも回収できる。

server2の保存5件で、同じ結果commitとtreeを新規受け側に復元し、返却量は約134.0 MiBから約408 KiBになった。
RomuTail、server2、Romulaptop WSL、PBSゲートウェイの専用配置で同じ人工検査を通した。
WSLの通常spoolには比較できる準備bundleがなく、PBSで確認した既存の人工準備4件もGit返却0バイトだった。
両環境の実ジョブにおける削減量は作業内容に依存する。今回はPBS計算ノードへの新規投入を行っていない。

RomuTailとWindowsの通常入口で、明示releaseとrelease指定なしのPreviewが同じ版とrun IDを返した。
既存runのStatusは保存済みreleaseを返し、既存設定等35ファイルのSHA-256が一致した。
人工回帰は113件PASS、1件skip。旧版でも再現したStatusのfixture不一致1件を別に記録した。
詳細は `.rpe-central/deployment/romutail-worker-storage-dd832d360/` の `VALIDATION.json` と
`CURRENT_ADOPTION.json`、および同名fleetの `PLACEMENT-*.json` に保存した。

既存の作業ファイルはそのまま保存した。server2の削除候補161ファイルと照合記録は、別の監査
`.rpe-central/validation/worker-spool-audit-20261118/` にある。実信号・実compile・実Q3・本番探索の起動は各0件。
server1が復帰したら、Codexは同じfleetのhostレイアウトで既存科学資産を照合し、専用checkoutを配置する。
配置には必要に応じて `PORTABLE_SOURCE.bundle` を使い、保存入力と人工検査を通してから参加状態を照合する。
