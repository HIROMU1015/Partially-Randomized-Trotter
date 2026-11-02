# H8 B2：停止記録の修正と並列実行準備

**修正・並列化・人工検証とSOURCE sealを完了した。実H8 compile、保存signal再解析、fit、再会計、科学有効grantは0。B0の保存2軸を新契約で受け入れ、残るB0+Iの2軸だけを取得する未承認経路である。別途実行承認とfresh gateを待ち、公開後mandatory STOPする。**

branch：`track-a-h8-phase-b2-recovery-parallel-readiness-20261017`。実行SOURCE：`77ce12d8c935702dc78f294b5863b4904f9c6a2c`。起点はrun01停止報告 `d5f290248147d0127d73afd35a53e2b10ff835df`。修正元の実行SOURCE244c45e7…、原結果・grant・ledger・markerは変更していない。

## 受入範囲と未完了部分

H8直線鎖、隣接1Å、STO-3G、16 modes、Nα=Nβ=4/sector4900、保存tol-only DF rank28/tol1e-8/cutoff0、signed generation order、weighted Hermitian projection。T=.8、今回新規対象はq8/δ=.1/L_D=p21/S4。保存B0対照はq1/δ=.8/p15/S4。S4は負時間中段を含む非mergeの3 S2段。full measured native Hadamard wrapper、state preparation費用除外、symmetric-directional controlを維持する。

| 対象 | 原結果 | 今回の準備範囲 |
| --- | --- | --- |
| B0 p15/q1 S4 cosine/sine | 各168,353 RZ、各誤差0.2153638062%、native exit0/reap/SHA確認 | 新grantの明示受入がある場合だけ保存2軸を再利用。再compileしない |
| B0+I p21/q8 S4 cosine | control PASS、compile中断、完了結果なし | 一回の新規取得対象 |
| B0+I p21/q8 S4 sine | 未起動 | cosine PASS後に一回取得する対象 |

B1正本83a3c4a6…の凍結model8420b9f2…を維持し、B0+I予測は各軸1,852,640 RZのまま。新fit・H6係数移送・モデル再推定なし。`abs(predicted-measured)/measured < 1/50` を各軸独立に判定し、2%以上・technical anomalyで即STOP、sineは先行cosineのPASS後にだけ起動する。軸平均・自動retry/refit・追加候補で救済しない。

B0の保存2軸は原B2全体がSTOPしたrunの部分結果であり、原契約を遡及PASSにしない。新しいscopeと `accept_saved_B0_2_axes=true` を含む明示承認が必要。今回のdraftはfalse、approved/runtimeもfalseである。原B2未settle予約1は返却・書換えせず、新試行の結果と混合しない。

## 並列化とメモリ保持の変更

| 段階 | 実装した同時数 | 動作 |
| --- | --- | --- |
| 保存preparation復元・S4回路構築 | 1 | 保存操作列とp21 viewを使用。新SCF/DF/state/scalar生成なし |
| left/right control | 2 worker | 別processで同じ固定evolutionへ作用し、小さい誤差・overlapだけを回収 |
| cosine/sineのX/Y control・fingerprint・QPY保存 | 2 worker | 各workerは担当軸だけを保持。独立workerのnative exit0/reapと結果SHAを確認 |
| 親の大回路解放 | 1 | 戻すのはQPY path/hash・fingerprint・誤差のみ。GC/利用可能なglibc trimとRSS前後を記録 |
| QPY読込・compile・モデル判定 | 1 workerずつ | 必要な1軸だけ読込。元のcanonical fingerprint一致をcompile前後で要求し、PASS後に次軸 |

prepare workerは同時最大2、compile workerは最大1で、両stageを重ねない。driverから直接forkする計算workerに限定し、AI subagentは使用しない。既存の親ownership・PDEATHSIG・native wait4・全回収・1thread制限を再利用する。

QPYはprivate stageに各最大1 GiB、2軸だけを保存する。保存bytes SHA・サイズ・path境界を確認し、read時のcanonical fingerprintが検証時の値と一致しなければcompileしない。compilerに渡す回路・control・scalar・gate順・測定範囲を変更するための再構成ではない。QPYを置くscratchもstage 8 GiB監視の対象で、公開結果の256 MiB上限とは区別する。

この変更でcompile中の親processがcosine/sineの大回路を同時保持する構成を除いた。一方、準備stageでは2 workerが動くため、実H8のpeak RSSや全体速度、PSI停止の解消はまだ未測定。2 workerだから全体が2倍速いとは主張しない。

## STOP記録と終了receiptの修正

PSI/OOM/available監視の失敗時は、cleanup前にcurrent/baseline raw snapshot、判定reason、対象group、full avg10、適用thresholdを保存する。fresh gateのCPU記録とresource記録は別ファイルにし、同名上書きで失敗証拠を失わない。終了時のresource snapshotも保存する。host full avg10>=1%、cgroup full avg10>0、OOM増加、resource欠測のSTOP条件は緩めていない。

新dispatcherは有効grantを検証してから、CLIとは別sessionの小さいwitness processを起動する。witnessがlauncherの実native exitをwaitし、reported exitとは別のreceiptへfile/dir fsync付きで保存する。launcherはlive witnessのparent identity/affinityを要求し、監視中もwitnessのidentity・thread/AS/RSS/CPUを確認する。CLI終了後のwait継続とexit7/被SIGKILLの−9保存は人工processで確認した。原run01の欠測exit値を推定で補完していない。

compile件数は実際の `axis_*/*/compile_entered.json` を数える。旧wave_* globを使用しない。予約とsettle、API突入と完了を別に保持し、STOP時も未完了予約を返却しない。

## 新しい資源案と起動条件

| 条件 | 新scopeの案 |
| --- | --- |
| CPU/process | 別5物理core：driver1・worker最大2・observer1・witness1、各thread1 |
| compile並列数 | 1。2 worker並列化はcontrol/fingerprint/QPYの準備stage |
| AS/RSS | driver64/32 GiB、worker各64/32 GiB、observer512/256 MiB、witness128/64 MiB |
| aggregate RSS＋reserve | 96.3125＋16 GiB、fresh実効空きRAM112.3125 GiB以上 |
| CPU/wall | 旧24 CPU時間/12時間、observer1800 CPU秒、witness120 CPU秒 |
| disk/inode | fresh16 GiB/40,000以上、stage8 GiB、QPY各1 GiBまで |
| observer | 1秒、gap5秒超STOP、raw CPU busyはSMTを含め5%以内 |

旧3core/80.25 GiBのgrantを流用しない。SOURCE/input/provider/runtime/profile・元結果受入・5物理core/quota/RAM/disk/inode/PSI/OOM/保全のfresh gateを通過した場合だけ、新scopeで一回起動できる。draftのCPUは空、command null、有効grant0、fresh gate未発行。今回の修正指示は新規compile承認として扱っていない。

B0+Iのscalarは保存p21 full-Fock Cをoriginal constantへ一回だけ加える。保存未回転full-H reference、経験的allowance factor1/10、厳密bias/PR母平均保証の未確立事項を維持する。compilerはQiskit1.3.0/rz,sx,x,cx/opt1/seed17/backend・topologyなし、compiler1239304c…/完全profile622ce4a8…を固定した。

## 技術検証と保全

固定Python3.11.0rc1で新15＋既存19＝34 unittest methods PASS。新scope/保存B0受入/無効grant拒否、2%境界とcosine不合格時のsine未起動、2 worker同時生存と成功・失敗の全回収、実observerでのPSI snapshot保存後cleanup、CLIの科学import前gate、QPY容量/tamper/path拒否、別session witnessのnative waitを確認した。

2 qubit人工fixtureではleft/rightとcosine/sineの各2 worker・計4workerのnative exit0/reapを確認し、QPY保存前後のcanonical fingerprint、旧経路と新経路のRZ/CX/depth/size/各depth・compiler hashを一致確認した。人工compileは旧/新の両軸計4、最大control誤差約3.71e-15。実H8 arrayの復元・新compileは0。local検証であり、独立レビュー・CI・外部再現とは呼ばない。

31 SOURCE files、provider265、runtime34,273、入力16件をGit/disk/bytesでsealした。旧687 grant/ledger/carry/marker等、124旧worktree HEAD/status、原B2の45 raw/config/grant/receipt filesを照合一致。旧B0 run02とB2 run01のSTOP、B1結果、V1/V2 CLOSED_NO_GO/H5/H7/M1-S・他worktreeを保全した。

次の承認対象は保存B0の2軸受入と、残るB0+I p21/q8 S4の最大2軸一回取得だけ。成功・失敗によらずmandatory STOPし、Phase C再会計は別途承認。今回、新H8科学計算・数値再解析・compile・fit・再会計・有効grantは0。

[公開bundle/checker](../../artifacts/resource_applicability/track_a_h8_phase_b2_recovery_readiness/2026-10-17/README.md)をcommit・pushし、remote SHA確認後mandatory STOPする。
