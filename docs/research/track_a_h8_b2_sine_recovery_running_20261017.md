# H8 B2：cosine実測PASS・sine保存回路からの限定再開

**2026-10-17T16:48:38.918552+09:00時点、sineの保存QPY読込・fingerprint確認を継続中。起動前2回のfresh gateをPASSし、136.0秒の複数観測で同一workerのCPU時間増加と監視継続を確認した。sineの実compile完了値・モデル判定・B2最終完了はまだない。**

## 保存cosine実測と元STOP

H8直線鎖/隣接1Å/STO-3G/16 modes、Nα=Nβ=4/sector4900、保存tol-only DF28/tol1e-8/cutoff0、signed generation order・weighted Hermitian projection、T=.8、B0+I p21/q8/δ=.1/S4のfull measured native Hadamard wrapper。負時間中段を含む非merge 3 S2、symmetric-directional control、保存scalar/full-H reference、経験的allowance factor1/10、state preparation除外を維持する。

SOURCE2e604e4a…のattempt02で、2026-10-17 16:13:42 JSTにcosineの実測1,852,640 RZ、固定予測1,852,640、各軸相対誤差0%、native exit0/reap/fingerprint前後一致を保存した。compile wallは770.213秒（API入口以後のfingerprint確認等を含む）。これは保存実測値であり、再fit結果ではない。

同runは16:15:03 JST、sineのQPY読込中・compile API未突入でMEMORY_PRESSUREによりSTOP。上位 `/sys/fs/cgroup/user.slice` のfull avg10=0.18%が契約の>0に該当した。本人user.sliceとsessionは0%、host full avg10も0%、OOM/OOM_kill増加なし、実効空きRAMは約447.3 GiB。上位の圧力を発生させたprocessは特定していない。元runのpeak aggregate RSSは約41.15 GiBで、RAM上限超過とは区別する。全owned回収、実native launcher exit1をwitnessが保存した。原STOP・sine未settle予約1・さらに以前の未settle予約1を保持する。

## 技術修正と今回の範囲

利用者の「計算が途中で中断された場合は修正して計算を再び開始してほしい」を継続する技術回復の承認として記録した。過去のgrant/ledger/marker/STOPを書き換えず、新しいSOURCE、grant、未使用実行域でsineの1軸だけを取得する。保存B0 2軸とcosine実測を再compileせず、sine回路生成・4 control・fingerprint生成・QPY dumpを繰り返さない。

byte/SHAで固定した保存control/native receiptと、sine QPYの46,793,071 bytes/SHA `8d5a0600…`、canonical fingerprint `d4064743…`を結合した。読込時にサイズ/SHA/pathを再確認し、compiler呼出前後のcanonical fingerprint一致も必須とする。保存受入は元runの正常完了を意味しない。

実装SOURCE：`e3ecc03a4d02503ccdf20afd9d750cb49d8ca6ff`。前段のsine専用SOURCEは `aaf3d909f20f20354016966358ce754dbf5b7aff`。新scopeは `H8_B2_VALIDATED_SINE_ONLY_STRICT_EVIDENCE_CONCURRENT_OBSERVATION_V1`。code branchは `track-a-h8-b2-concurrent-preservation-recovery-20261017`、この稼働報告branchは `track-a-h8-b2-sine-recovery-running-20261017`。

B1正本83a3c4a6…/model8420b9f2…、予測各軸1,852,640 RZ、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/backend・topologyなし、compiler1239304c…/profile622ce4a8…を固定。`abs(predicted-measured)/measured < 1/50` の科学条件と2%以上STOPは不変。新fit/signal/SCF/DF/state/sampling/再会計は行わない。残る独立compileは1つなのでworker1、4物理coreはdriver/worker/observer/witness。新実行のaggregate RSS64.3125＋reserve16 GiBをfreshに要求する。過去のCPU/worker/RAM上限を将来の恒常的な上限にはしない。

## 並行worktreeの保全判定

別の `h8-noncancellation-readiness-20261017` が並行してcommit・資料作成を進めたため、全worktree HEAD/statusの凍結条件が科学起動前の準備を拒否した。immutable証拠が変わったこととは区別する。同worktreeは今回の科学input/provider/SOURCEではない。

技術修正で、明示したこの1worktreeだけを読取専用の更新観測とした。baseline/current HEAD・status hashと変更有無をreceiptに残し、「不変」とは報告しない。SOURCE/provider worktreeを観測扱いへ除外すること、別名のworktreeを無断除外すること、baseline行を欠落・重複させることは拒否する。他の133 worktreeと699 immutable files、33 SOURCE/265 provider/34,273 runtime/31 inputの厳密照合を維持する。こちらから別worktreeへ書き込む処理・共有OS/scheduler/cgroup変更はない。

前段sine試行域attempt01/02/03/05はmetadata準備・保全段階のSTOPで新科学0。有効grantを一回dispatchしたattempt04はCPU57/58のraw busy83/33%で初回fresh STOP・新科学0。raw5%・自身の使用量を引かない判定を維持した。attempt04では管理シェルのCPU0固定をdispatchにも適用してしまい、最初のgrant検証が拒否された。dispatch marker消費前だったことを確認し、管理コマンドの固定を外して同grantを一度dispatchした。失敗記録を上書きせず保存している。

21人工test methods（起動6・保存spool受入/限定scope9・並行保全6）PASS。保存cosine/native/model/profile、4 control、fingerprint/QPY tamper、2%STOP、観測対象制限、frozen worktreeと既存grant変更時のSTOPを確認した。前段SOURCEでは2 qubit保存QPYを実providerで人工compile1回し、実native終了・hash不変・意図したモデル不一致STOPを確認した。人工検証のfixture欠測1件を修正したログも保存する。独立レビュー・CI・外部再現とは呼ばない。

## 現在の稼働と次の作業

今回のattempt01はdriver2/observer4/compile worker5/witness7で、全fresh gateをPASS。同一workerのCPU時間が増加し、snapshot時RSSは約4.48 GiB、観測最大監視間隔4.916秒。保存cosine・検証済みsine受入を記録済み。sineのRZ実測値や完了時刻をこのsnapshotから推定していない。

稼働中監査で233過去immutable files、config/有効grant、SOURCE/input/provider/runtimeを照合した。V1・V2 CLOSED_NO_GO・H5/H7/M1-S・元B0 STOPを保持する。公開物は有効grantのhashとpath-redacted receiptのみ。

Codexの次の作業は、sineのモデル判定と実native終了・保全receiptを確認すること。技術中断時は今回の承認範囲で修正・別試行を記録して再開する。モデル誤差2%以上は科学STOP。B2終了時にmandatory STOPし、Phase Cにはユーザーの別承認が必要。自動コード修正daemonは導入していない。

[保存snapshot/checker](../../artifacts/resource_applicability/track_a_h8_b2_sine_recovery_running/2026-10-17/README.md)。commit・push後remote SHAを確認し、計算待ちへ移る。
