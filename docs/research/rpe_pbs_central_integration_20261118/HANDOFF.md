# 中央統合担当へのPBS合流版引継ぎ

**検証済み統合SOURCE：`925985b4ea76a15f8a07c507179d4fa6c492eabd`。**
専用branch：`codex/rpe-pbs-worker-20261118`。
中央674（実装6da）、価格8ba（実装c4）、PBS5e6を実mergeした版で、同じ925により実PBS人工検収・回収・採用・再開を完了した。
詳細は[REPORT](REPORT.md)、[接続検査](CONNECTION_TESTS.txt)、[所有権検査](OWNERSHIP_TESTS.txt)、
[実PBS結果](cluster/COMPARISON_AND_RESUME.json)。旧PBS22の結果を925へ付け替えていない。

## 取込み・共有currentの担当

取込み可能なmerge commitは`65d989a40`と`a6e99f46f`、接続検査を含む固定版は925。
925は中央tip674と価格tip8baの両方を祖先に含む。既存中央branchへこの専用branchをmergeする際、
中央側の後続変更があれば、その差分を保って接続検査を行う。
後続mergeでcodeが変わった新SOURCEを925と同一の検収済み版として登録し直さない。

この固定版は既に次のreleaseとしてVALIDATED登録済みで、未activateである。

```
romutail-pbs-integrated-925985b4e
/home/abe/Project/Partially Randomized Trotter/.rpe-central/
  releases/925985b4e/
  deployment/romutail-pbs-integrated-925985b4e/SERVICE.json
  RELEASES.json
```

**共有current切替は中央統合担当だけが行う。PBS担当は実施していない。**
[登録記録](deployment/REGISTRATION.json)の前後はともに`romutail-6da8554e0-r4`。
新SERVICEにはc4の共通モデルstore・工程予算・run所有relay、最新中央の起動終了・Windows/WSL修正、PBS925が入る。
元の科学組立てSOURCE d887・Q3 28d・timed adapter349・旧run pin・原結果・STOPは保全している。

## 中央担当の切替手順

1. 現在のcontroller/受領済みtask/旧relayの所有とSOURCEを照合する。既存runはその固定版でdrain/回収する。
2. 新SERVICEは**run専用relay**を使う。旧常駐relayとの同時所有を避け、中央担当が旧relayの状態と受領済みtaskを確認して移す。
   PBS担当は共有`.rpe-central/RELAY.json`、そのSTOP、WSL受付、既存workerを操作していない。
3. 新SERVICEと固定profile・入力・共通storeを確認し、下記登録済みreleaseを中央担当がactivateする。
4. 新しいPreview/PlanOnlyは新current、既存runのStatus/Stop/Resumeは元`CONTROLLER_CONFIG.json`へ戻ることを確認する。

切替コマンド例は中央担当用であり、この作業では実行していない。

```bash
cd '/home/abe/Project/Partially Randomized Trotter/.rpe-central/releases/925985b4e'
PYTHONPATH=src python3 -m trottertracks.rpe_release_registry \
  --registry '/home/abe/Project/Partially Randomized Trotter/.rpe-central/RELEASES.json' \
  --release romutail-pbs-integrated-925985b4e \
  --service '/home/abe/Project/Partially Randomized Trotter/.rpe-central/deployment/romutail-pbs-integrated-925985b4e/SERVICE.json' \
  --evidence '/home/abe/Project/Partially Randomized Trotter/.worktrees/rpe-pbs-worker-20261118/docs/research/rpe_pbs_central_integration_20261118/cluster/COMPARISON_AND_RESUME.json' \
  --activate
```

独立relayを使う中央既定もcode上は保持した。`distributed.relay`未指定なら共有relayを起動・停止しない。
別のrelay方式へ変更する場合は、登録済みSERVICEを編集せず、新しい配置/IDで照合する。

## 通常入口

既存local0の`Invoke-RpeSearch.ps1`を使用する。925のWindows側toolsは中央tip674と同一である。
実機では次の明示release Previewが925を返した。これは科学Startの代用ではない。

```powershell
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Preview `
  -Release romutail-pbs-integrated-925985b4e -RequestFile .\scope.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Status -RunId <既存run_id>
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Stop -RunId <そのrun_id>
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Resume -RunId <そのrun_id>
```

Start/Resumeの科学的実行範囲は別途採択された指示に従う。現在のテンプレートは科学false・Q3/準備予算0。
`scope.preparation_budgets`の明示値、Q3予算、SOURCE/input、one-shot/UNKNOWN/STOPを独立に保持する。
Q3指示から信号・compile許可を推定しない。

## PBSの配置・再開

SSH aliasはRomuTailの`prism`、接続は既存akeno経由。
cluster rootは`/home/AbeHiromu/projects/rpe-pbs-20261118`。
PBS transportは`releases/925985b4e`、Pythonは`runtime/venv311rc1/bin/python`。
科学入力/view/runtimeは前回の保存条件を再利用し、[PBS_PROFILE](deployment/PBS_PROFILE.json)で解決する。
SOURCE925のcontroller/PBS gateway/workerを同時に使う。

通常入口が自動qsub/qstat/成果回収を担当する。workerはPBS割当nodeで実CPU/mem境界を読み、
Q3/準備を同じ枠poolで会計する。管理hostではworkerを実行しない。
要求/結果は共有homeのrun別spoolへ保存し、標準出力返送を待たず回収できる。
qsub応答欠落やcontroller再起動は元のtoken/job/attemptへ戻る。途中終了は元契約のUNKNOWN/retry/STOPを保持する。

配置済み通常資源の例はworkq 1node/96CPU/350GB/24hだが、科学task時間見積りは未設定。
実H4の新規指示を採択した担当が、task時間/memory・PBS資源・完全REQUESTのhost適格性を設定する。
今回は2CPU/4GB/4分の人工jobだけを実行し、実H4の最適枠・性能・科学結果は認定していない。
旧PBS単独版の詳しい仕組みと保存metadataは[旧HANDOFF](../rpe_pbs_worker_20261118/HANDOFF.md)を参照する。
今後の選択対象はこの合流版であり、旧22のSOURCEへ戻して統合完了と扱わない。

## 完了した検収と次の担当

92人工接続検査は91 PASS・1既存配置依存SKIP。単機と実PBSで各14taskを採用し、8結果と方式別採用が一致。
原compile attemptを保ったcontroller再起動、完了run再開で追加起動/採用/割当0を確認した。
PBS job267195はF/exit0、worker EXITED、残存0。[最終job確認](FINAL_JOBS.json)。

次は中央統合担当が共有relay/controllerの所有を照合し、登録済み合流版のcurrent切替を担当する。
実H4探索は利用者の別途採択後に、担当Codexがその固定scopeだけを起動する。
