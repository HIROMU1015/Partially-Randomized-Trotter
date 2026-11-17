# 固定releaseの配置と保全

制御SOURCEは `95a74e4486131a97bd5235221bf2d8ab9f99f591`、release IDは `romutail-pbs-current-95a74e448`。実際のcurrent採用は中央の既存RELEASES.jsonと本作業のREPORTを正本とする。新規runは選択版を記録し、既存runは自身に固定したrelease、科学SOURCE、入力、runtimeを使う。

[RELEASE_COMPONENTS.json](RELEASE_COMPONENTS.json)が公開対応表。optimizerとPBS gateway/workerは新SOURCE95、既存4環境のworker/preparationはd887、科学Q3は28d、timed adapterは349、価格workerはa18、native providerは既存固定版を維持する。host固有pathはRomuTailの `.rpe-central/fleet/romutail-pbs-current-95a74e448/hosts/*.json` で解決する。版選択・queueは既存registry/JobStoreを再利用する。

| 配置先 | optimizer/transportの新checkout | 確認 |
| --- | --- | --- |
| RomuTail | `.rpe-central/releases/95a74e448-pbs-current` | 全component/runtime PASS、反復取得0 |
| server2 | `/home/AbeHiromu/projects/rpe-release-fleet/releases/95a74e448-pbs-current` | 全component/runtime PASS、反復取得0 |
| PBS | `/home/AbeHiromu/projects/rpe-pbs-20261118/releases/95a74e448-current` | optimizer/PBSworker/gateway全SOURCE95、科学16component再利用、47runtime alias PASS |
| local0-wsl | `.rpe-release-fleet/releases/95a74e448-pbs-current` | 必要なoptimizerだけ更新、反復取得0。科学側の既存e917検収を再利用、通常受付OFF |
| server1 | 同じreleaseのpackageを用意 | 停止中、接続なし。復帰後に配置・保存入力・人工参加を検査 |

実測receiptは[DEPLOYMENT_RECEIPTS.json](DEPLOYMENT_RECEIPTS.json)と各hostのINVENTORY.json。componentの配置確認は受付許可や科学実行許可を変更しない。新規実信号・実compile・実Q3・本番探索は各0件、mandatory STOPを維持する。

## PBSのgateway/runtime

新SERVICEは `.rpe-central/deployment/romutail-pbs-current-95a74e448/SERVICE.json`。同folderのPBS_SETTINGS.jsonで、通常collectと割当終了後harvestに使う `gateway_python` を既存 `/home/AbeHiromu/projects/rpe-pbs-20261118/runtime/venv311rc1/bin/python` に明示した。新venvや依存の再導入は行っていない。科学workerも従来のPython3.11.0rc1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0を維持する。

PBS profileは、再帰的な `pbs` operational field以外が旧925配置と全一致する。[比較receipt](PBS_PROFILE_PRESERVATION.json)を参照。新PBS roleの受付前には既存manifest helperがSOURCE/runtimeを照合し、各assignmentでは科学REQUEST/ID/hash外のrequired_releaseと既存workerのSOURCE・入力guardを照合する。science REQUESTは変更しない。

本番設定の `task_seconds={}` は保全した。将来の実科学参加には、利用者が指定した対象・範囲と対応する採択済み工程別時間見積りが必要。単一Full PF検査の時間を全方式へ適用しない。人工検収の2CPU/4GB/4分などは専用人工設定であり、本番設定へ転用しない。

PBS925のtransport object storeは親履歴が不完全だったため、通常bundle fetchを採用せず、既存PBSのpartial-object/worktree配置方式を新fleet cacheだけに適用した。旧925とreadiness object storeは読取参照を維持し、旧SOURCE/run/serviceは編集していない。新Sourceのroot-tree不足1object（2,004B）を追加取得し、全SOURCE guard PASS後にのみ配置完了を通知した。

## 次版の登録・配置とserver1復帰

次版も固定SOURCEを先に決め、既存 `prepare_rpe_release_fleet.py` で公開対応表とhost layoutを新folderへ生成する。PBS hostはbase_profile/price_profileを保存PBS profileへ指定し、gateway_pythonを上記既存runtimeに明示する。既存SOURCE/runtime/inputを使い、不足するobject/入力だけ取得する。PBSのpartial Git配置は既存 `prepare_rpe_pbs_readiness.py --stage-commit` の方式を再利用し、通常の完全Git履歴を前提にしない。

配置後の確認入口は選択版の `scripts/manage_rpe_release_deployment.py --manifest <hostのmanifest> --layout <hostのlayout>`。初回のみ必要に応じて `--install` を指定する。PBSのparticipation componentは `optimizer,pbs_worker,pbs_gateway`、既存hostは `optimizer,worker`。通常clientも中央が返す選択版のhost layoutを使う。

必要な技術検収後、切替担当を一人にして既存 `PYTHONPATH=src python -m trottertracks.rpe_release_registry` で新IDを登録・採用する。登録済みentryを再採用するときも同じevidenceを指定し、`--expected-current` で並行切替を検出する。branch HEADの自動採用は行わない。

server1用packageは `.rpe-central/fleet/romutail-pbs-current-95a74e448/packages/server1.zip` に保存した。SOURCE root treeを含むcomplete code packと固定metadataを収録し、hashはdeployment receiptに記録した。復帰時は既存worker/科学view/runtimeの実在を先に確認し、不足分を専用配置へ補い、保存入力と人工参加・切断回収を検査してから既存受付方針に従う。server1の復帰を今回の完了条件にしない。

通常の開始・状態確認・停止・再開は[主HANDOFF](../HANDOFF.md)の共通入口を使う。次の科学実行は利用者の対象・範囲の明示後、§10.5の重複照合を通してから行う。
