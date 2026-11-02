# H8 B2 recovery run01：科学起動前のCPU gateでSTOP

**2026-10-17 14:13:54 JST、初回fresh CPU gateでCPU6のraw busy 82%を検出し、契約上限5%を超えたためSTOP。科学driver/worker、native control、QPY、compile、fit、費用再会計はすべて0。CPUを変更しての再試行・自動retryは行っていない。**

利用者の「計算を開始して　計算の待ち時間に入ったらチャットは終了してよい」を、公開準備commit `bf56461fa9e0780d435bad5d30c2cf2103549041` の新scopeへの実行承認として受領。保存B0の2軸受入と、B0+Iの残り2軸だけを取得する新しい有効grant1件を別private領域に発行し、一回dispatchした。準備用無効grant・過去のgrant/ledger/markerは変更していない。

実行SOURCE：`77ce12d8c935702dc78f294b5863b4904f9c6a2c`。scopeは `H8_B2_SAVED_B0_ACCEPTANCE_B0I_TWO_AXES_PARALLEL_PREPARATION_V1`。停止報告用branch：`track-a-h8-b2-recovery-run01-stop-20261017`。計算起動元とは別worktreeで文書・公開projectionだけを追加した。

## 固定対象と実際の到達点

予定した科学条件はH8直線鎖/隣接1Å/STO-3G/16 modes、Nα=Nβ=4/sector4900、保存tol-only DF rank28/tol1e-8/cutoff0、signed generation order、weighted Hermitian projection、T=.8、p21/q8/δ=.1/S4。負時間中段を含む非mergeの3 S2段、symmetric-directional control、state preparationを除くfull measured native Hadamard wrapper。保存B0はp15/q1/δ=.8/S4の2軸で、元B2 run01の部分PASS結果として保全する。

B1の凍結model8420b9f2…、B0+I予測各軸1,852,640 RZ、compiler1239304c…/完全profile622ce4a8…、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/backend・topologyなしを変更していない。今回新しいRZ測定値・モデル誤差・PR費用評価はない。

| 項目 | 今回の状態 |
| --- | --- |
| 保存入力・SOURCE/provider/runtime | 31 source/265 provider/34,273 runtime/16 inputのbyte照合PASS |
| metadata段階の保存B0照合 | PASS。grantの受入承認はtrueだが、driverの受入receiptは未生成 |
| CPU提案 | 固定3窓・各1秒。自動追加窓なし |
| 固定CPU role | driver0/observer3/preparation worker4,5/witness6 |
| 初回fresh CPU gate | CPU6のみBUSY、82/100 ticks=82%。他4 coreは0/3/1/0% |
| fresh RAM/disk/PSI/OOM | CPU gateで先に停止し、このrunの後続fresh判定へ未到達 |
| fresh gate receipt | 未発行 |
| 科学driver・worker・control・QPY・compile | 全て0 |
| 科学output/control/one-shot lock | 全て未作成 |
| dispatcher/witness試行域 | 一回消費済み。再使用しない |

3窓CPU提案をfresh gateの代用にせず、起動直前に固定CPUとSMT siblingを含むraw counterを別に測定した。自身のCPU使用を引かない契約を維持した。CPU6がwitness割当coreであることは記録できるが、違反窓のprocess別CPU消費は保存していない。負荷の原因をwitness・launcher・他jobのいずれかに断定しない。

parallel準備2 worker/compile1の実装は今回も固定したが、科学driver前のSTOPなので、実H8でその並列処理やメモリ改善効果を検証した結果ではない。既存34人工tests/QPY比較の証拠を参照し、追加の科学計算・検証compileを実施していない。

## 終了証拠と保全

dispatcherの実native exitは0で、これは独立witnessへdispatchできたことを示す値。科学計算の成功とは扱わない。witnessがlauncherの実native exit1をwait完了し、`native_exit.json`へ保存した。原B2 run01で欠測した外側終了receiptを今回の値で補完せず、異なるrunの証拠として分離する。

witnessとlauncherの保存PID/start/parent/uidを終了後に2回照合し、同一identityは不在。科学driver/workerは未起動。CPU counters・launcher STOP・native wait・stderr・非返却dispatch/witness/admission markerを原本のまま保存し、有効grant/configはhashだけを公開した。local絶対pathは公開projectionで伏せる。

終了監査で旧688 grant/ledger/carry/marker等、125既存worktreeのHEAD/status、原B2 run01の45 raw/config/grant/receipt files、今回config/有効grantのbytesを照合一致。起動前と停止後にprovider/runtime/inputを再確認した。元B0 run02 STOP、B2 run01 MEMORY_PRESSURE/STOPと未settle予約1、B1/V1/V2 CLOSED_NO_GO/H5/H7/M1-Sを変更していない。

契約のfresh不合格STOP・自動retry禁止に従い、CPU再割当・CPU再提案・追加compile・refit・Phase Cを実施しない。今回科学予約は0であり、原run01の未完了予約を返却していない。新たな起動には別の明示指示と未使用試行域が必要。

[公開STOP bundle/checker](../../artifacts/resource_applicability/track_a_h8_b2_recovery_run01_stop/2026-10-17/README.md)をcommit・pushし、remote SHAを確認後mandatory STOPする。checkerは保存bytes/metadataだけを検査し、CPU samplingや科学実行を行わない。
