# PBS保存実入力の配置・dry-check

担当はCodex sub-agent `pbs_readiness`。科学評価・新規信号取得・回路compile・Q3本体は全て0件。
結果は[ACCEPTANCE.json](ACCEPTANCE.json)、PBS終了確認は[QSTAT_FINAL.json](QSTAT_FINAL.json)。

- `267191.CPUcluster-PRISM`：`workq`, 1 node / 2 CPU / 4 GB / walltime 5分。
  既存の4方式×signal/compile/auditの12保存入力と、RomuTail統合版のPR-DF追加3入力、計15件が
  `SAVED_INPUT_RUNTIME_INSPECTION_PASS`。この表示は保存入力・固定SOURCE・runtimeのinspectionであり、
  将来の科学REQUEST全体に対する実行許可・実行済み証拠ではない。
- 同ジョブの4方式Q3 dry-checkは、manifest外の歴史的Git参照blobが未配置で失敗した。
  元SOURCEと入力は編集せず、元のGit objectを追加配置した。失敗記録も`checks01/`へ保存した。
- `267192.CPUcluster-PRISM`：1 node / 2 CPU / 4 GB / walltime 3分。
  失敗した4方式だけを再検査し、全て`REAL_HOST_DRY_CHECK_PASS`。`checks02/`に保存した。
  いずれも実行nodeは`CPUcluster035`、PBS状態`F`、`Exit_status=0`、検査jobの残存は0件。
  PBSが報告した実使用walltimeは初回5秒・再検査6秒であり、科学評価時間ではない。

固定科学Q3 SOURCEは`28d976b99cdf8329adcc1b79327e329e947ef774`、timed adapterは`349558b748a82f384a4589434a63ef9b5572032c`。
工程別SOURCE・入力・request hash・runtimeは保存した`inputs/`と各READINESSの原記録を参照する。
現在の4方式dry-checkは既存の保存REQUESTをそのまま使用した。新候補、新方式、新入力、別SOURCEの適格性へ自動転用しない。

配置先は`/home/AbeHiromu/projects/rpe-pbs-20261118/readiness`。
`objects.git`には元のcommit/tree/blobを保った疎なGit object集合、`views/<sha>`には通常のsparse detached worktreeを作成した。
既存28 GBのGit履歴全体を複製せず、科学SOURCE、保存入力、元guardの参照blobを配置した。
途中で不足した原Git参照は[HISTORICAL_LOCATORS.json](HISTORICAL_LOCATORS.json)の118件に固定している。
Pythonはprofileから`/home/AbeHiromu/projects/rpe-pbs-20261118/runtime/venv311rc1/bin/python`を解決し、
元の固定SOURCEのAST/runtime条件を変更していない。

新規runへ渡すprofileは[PRODUCTION_PROFILE.json](PRODUCTION_PROFILE.json)、配置と未設定項目は
[PROFILE_LAYOUT.json](PROFILE_LAYOUT.json)。これは`inputs/PROFILE.json`の科学SOURCE・工程profileを保ち、
PBS transport pinとrun別work_dirの置換箇所を加えたtemplateである。
配置済みのruntime providerファイルはinspection済みだが、task artifactCASはdry-checkで事前充填していない。
採択後の実タスクでは既存workerのhash照合・不足blob転送を使う。
実H4の工程別実行時間見積もり`task_seconds`は未設定であり、別途採択した探索条件から設定する。

配置helperは`scripts/prepare_rpe_pbs_readiness.py`。新workerの固定commitを追加するときは、RomuTailの専用worktreeから次を実行する。

```bash
python3 scripts/prepare_rpe_pbs_readiness.py \
  --stage-commit FULL_FIXED_COMMIT \
  --destination /home/AbeHiromu/projects/rpe-pbs-20261118/releases/SHORT_FIXED_COMMIT \
  --output /ABS/OWNED_STAGING_RECORDS --ssh-alias prism
```

既存object storeへ不足するobjectだけを送り、既存viewは固定HEADとSOURCE差分を照合して再利用する。
このコマンドはworker、科学タスク、PBS jobを起動しない。実行した新transport `fe1b47a50` の初回追加は14 object / 124,329 B、
補完後の再利用は0 object / 0 Bだった。
[STAGING_TIMING.json](STAGING_TIMING.json)の各durationは単一hostのmonotonic時計で測定し、異なるhost時計を差し引いていない。
最初のarchive転送単体の時間は未計測で、archive展開とview配置15.125秒、後続の不足object転送と再利用時間は測定した。
人工workerのタスク転送・採用時間、実H4本体時間はこのmetadata検査の時間に含めない。
