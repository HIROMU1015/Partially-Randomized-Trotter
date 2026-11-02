# H8 B0 anchor run02：保存8軸と終了時STOP

利用者の「計算を行って」に基づき、固定4構成・8軸を一回実行した。8つのcompile workerはすべてexit 0で回収され、各waveの受入receiptとTRAIN6/HOLDOUT2の測定ファイルが保存された。ただし、終了時の保全確認中にobserverが`PROCESS_THREADS`を検知し、取得契約は**STOP**した。正常完了・B1への自動受入とは扱わない。

## 固定条件と測定値

H8直線鎖、隣接1Å、STO-3G、16 spin modes、Nα=Nβ=4、sector 4900。保存tol-only DFはrank28/tol1e-8/cutoff0、signed generation order、weighted Hermitian projectionを維持。保存DF/state/preparationを復元し、新SCF/DF/state・新signal・PR samplingは実行していない。T=.8、S2、q=1/8、δ=.8/.1、L_D=prefix p14/21。元constantを一回だけ含み、tail identityなし。symmetric-directional control、state preparationを除くnative full measured Hadamard wrapper、Qiskit1.3.0、rz/sx/x/cx、optimization1、seed17、backend/topologyなし。

以下は保存された実compiled RZ/軸の転記であり、モデル予測・shot総費用ではない。

| 用途 | 構成 | cosine RZ | sine RZ |
| --- | --- | ---: | ---: |
| TRAIN | B0 p14/q1 S2 | 53,448 | 53,448 |
| TRAIN | B0 p21/q1 S2 | 78,719 | 78,719 |
| TRAIN | B0 p14/q8 S2 | 416,440 | 416,440 |
| HOLDOUT | B0 p21/q8 S2 | 618,608 | 618,608 |

各caseの保存control receiptは1e-8未満。各axisのfingerprint before/after、profile/compiler、preparation bindingとtask_result bytes hashを結合した。これは保存済み局所検証であり、独立再現・CI・費用モデル精度の証明ではない。3点fit・1点holdoutの2%判定は未実施。

## 実行・停止の証拠

- 実行SOURCE: `db559a518214fd27fa96e2f1451911df05c4f4a8`。準備結果commit: `2b495b4c82742120b2caebe22666a89f92f7c099`。
- scientific provider: `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、265 files。SOURCE15、runtime34,273、入力8を固定。
- profile: `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`。
- compiler: `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。
- CPU初回/driver refresh両gateはPASS。6物理core、4 worker/thread1、2 wave、必要RAM176.25 GiBという条件を維持。
- observer wall: 4,845.376420209184秒（約80.76分）。記録されたpeak aggregate RSS: 40,196,489,216 bytes。これはSTOP直前までの観測値で、欠測を含む全期間の厳密上界とは呼ばない。
- snapshot1、saved preparation decode1、prefix view2、native evolution4、measured wrapper8、probe copy8、control probe16、compile8。reserved8/settled8、unknown reserved compile0。禁止されたfit/residual/reaccount/S4/B0I/SCF/DF/state/PR等は0。
- 8つのcompile workerはexit0/reap。observerのfirst stopは`RuntimeError:PROCESS_THREADS`。driverはTERMを受けて`DRIVER_STOP_SIGNAL_OR_DEADLINE`を保存し、その後exit−9、外側launcherの実wait exitは1。cleanup errorsなし・全owned process回収済み。
- tracebackはdriverの`r.protected(config)`内の`git status --porcelain=v1 -z`待ちを示す。従って8軸受入・測定保存・末尾`verify_files`通過後、保全確認を完了する前の停止と分かる。
- 違反したPID・thread数・実行ファイルは既存observerが保存していない。Git待ち中という事実だけでは、違反processがGitだったとは断定できない。`acquisition_final_gate.json`は存在せず、契約STOPをPASSへ書き換えていない。

## 読み取り専用の事後確認と保全

[公開evidence bundle](../../artifacts/resource_applicability/track_a_h8_b0_anchor_run02_stop/2026-10-16/README.md)にoriginal STOP、native wait、worker/reap、TRAIN/HOLDOUT、control/preparation、fresh CPUとSOURCE/input/profileの投影を置く。private raw/config/grantはコピー元のまま維持し、絶対pathを公開物から除いた。rawのSHA inventoryと公開byte inventoryを分ける。

主エージェントによる事後hash照合でSOURCE15/provider265/runtime34,273/input8を確認し、実行前baselineの旧保護678ファイル・117 worktreeのHEAD/statusと旧V2 untracked bytesを確認する。事後確認は科学再実行ではなく、失敗した実行契約の最終gateを代替しない。公開checkerも保存bytes/metadataのみを検証する。

## STOP後の扱い

自動retry・追加compile・自動refitを行わず、B1 fit・B2 S4 compile・C再会計へ進んでいない。旧B0の未結合16軸を混合せず、既存B0+I16軸を再compileしていない。V1/V2/H5/H7と他worktreeを保全する。

次に検討できるのは、監視の違反process証拠を残す改善と、管理用保全確認のthread動作の技術調査である。既存8軸を再利用可能と認定するか、修正後の処理範囲をどうするかはSTOP報告後に判断する。8軸の再compileを自動で要求せず、新計算・B1 fit等には別途明示承認を要する。公開・remote SHA確認後mandatory STOP。
