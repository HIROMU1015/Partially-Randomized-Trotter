# H8 B2：CPU起動修正と稼働確認

**2026-10-17T14:40:26.878362+09:00時点、attempt02は科学処理を継続中。起動前2回のfresh gateをPASSし、194.8秒間の複数観測で同一driverのCPU使用増加、約1秒間隔の監視、STOPなしを確認した。現在はS4回路構築中で、新規compile結果・RZ実測値・モデル判定はまだない。これは完了報告ではない。**

実行SOURCE：`2e604e4a3a04eee5090e03d9bb83818d7afd3c87`。コード修正branch：`track-a-h8-b2-cpu-handshake-recovery-20261017`。この報告branchは `track-a-h8-b2-cpu-handshake-running-20261017`。計算中のSOURCE worktreeはcleanのまま固定し、公開資料を別worktreeへ追加した。

## 修正と実行承認

利用者の「修正を行って計算を再び開始してほしい。計算が安定して進み始めるところまで見てほしい。計算が途中で中断された場合は修正して計算を再び開始してほしい。」を、今回の修正実行と技術中断後の新試行への承認として記録した。過去の一回grantを再使用せず、各attemptは新規config/grant/marker/実行域を使用する。各attempt内部のretryはfalse、技術修正・新試行の根拠を別に残す。モデル不一致の科学的STOP、追加候補禁止、Phase C未承認は維持する。

旧SOURCE77ce12d8…では、launcherがSOURCE/runtime照合を終えるまで選択5 CPU上で実行可能だった。新SOURCEではobserverをexec前から管理CPUへ固定し、driverの照合も科学開始までは同じ管理CPU上で行う。witnessのready handshake後に、blocking waitpidでnative終了を待つ。CPU判定と同一の固定1秒窓でobserver/witness/driverのPID・start identity・affinity・user/system tickを保存する。自身のCPU使用をraw値から減算せず、raw5%・SMTを含む全選択coreの判定を維持した。旧CPU6の82%負荷主体は保存不足のため断定しない。

固定Python3.11.0rc1で新6＋既存15＋既存19＝40人工unittest methodsをPASS。exec直後のaffinity、blocking wait中のCPU使用、signal時の子回収とnative exit、PID再利用/欠測、raw5%境界を確認した。科学入力・H8回路・compileは人工検証で使用していない。独立review・CI・外部再現の証拠ではない。

## 試行と現在の状態

| 試行 | 到達点・判定 |
| --- | --- |
| 過去のB2 run01 | MEMORY_PRESSURE/STOP・保存B0の部分PASS・未settle予約1を変更しない |
| 過去のrecovery run01 | CPU6 raw82%で科学driver前STOPを変更しない |
| 今回attempt01 | 初回CPU gate PASS。driver照合後の再確認でCPU55が8/101 ticks=約7.92%となりSTOP。管理3 processの同窓CPU増分は0。control用driverは起動したが科学予約・回路・compileは0。全owned回収、実native launcher exit1を保存 |
| 今回attempt02 | 別の未使用域・新grant・固定3提案窓で新しいCPU割当。initial/driver refresh双方PASS、科学処理開始。driver58/observer54/準備worker46,42/witness41 |

attempt01の負荷主体も断定しない。今回はユーザーが技術中断後の再開を明示承認しているため、attempt01のSTOPを保持したうえでattempt02へ進めた。閾値を緩めたり、同じ一回grantを再使用したりしていない。稼働確認の管理コマンドは選択core外の自分のCPU0上で実行した。共有OS・scheduler・他ユーザーの環境やprocessは変更していない。

attempt02の初回fresh実効空きRAMは約465.1 GiB。保存済みSOURCE31/provider265/runtime34,273/input16を固定し、compiler/profileと保全・CPU/quota/RAM/disk/inode/PSI/OOMのgateをPASSした。snapshotまでの最大監視間隔は4.768秒（契約5秒以内）、観測peak aggregate RSSは4.149 GiB。同一driverのCPU時間は観測区間で194.81秒増加した。将来の完了・peak RAM・所要時間を保証する値ではない。

## 維持する科学範囲と並列化

H8直線鎖/隣接1Å/STO-3G/16 modes、Nα=Nβ=4、sector4900、保存tol-only DF rank28/tol1e-8/cutoff0、signed generation order・weighted Hermitian projection、T=.8。新対象はB0+I p21/q8/δ=.1/S4 cosine/sineの最大2軸だけ。負時間中段を含む非merge 3 S2段、symmetric-directional control、保存p21 scalar/full-H reference、経験的allowance factor1/10を維持する。新SCF/DF/state/signal/sampling/fitは行わない。

B1正本83a3c4a6…/凍結model8420b9f2…、各軸予測1,852,640 RZ、compiler1239304c…/完全profile622ce4a8…、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/backend・topologyなしを固定。`abs(predicted-measured)/measured < 1/50` を各軸独立に判定し、2%以上なら科学STOP。先行cosineのPASS後にsineをcompileし、再fitや追加候補へ進まない。保存B0 p15/q1/δ=.8/S4の2軸は別grantで受入し再compileしない。元のSTOPをPASSへ変更しない。

controlのleft/right、cosine/sineの検証・fingerprint/QPY保存は各2 worker。QPYを用いて親の大回路を解放し、compileは1軸ずつ行う。今回のsnapshotは回路構築段階なので、実H8での並列stageの完了・高速化率をまだ主張しない。5core/2準備worker/1compile worker等はこの実行契約の条件であり、今後の新作業に適用する恒常的なCodex資源上限ではない。

## 保全と次の作業

稼働中監査で692既存保護files、127既存worktreeのHEAD/status、過去の停止run・今回attempt01を含む117 immutable files、config/有効grant、provider/runtime/inputを照合一致。V1、V2 CLOSED_NO_GO、H5/H7/M1-S、元B0 STOPを保持する。有効grantの内容やlocal絶対pathを公開せず、hashとredacted projectionだけを公開する。

Codexの次の作業は、残る2軸の実測・各軸判定と終了receiptを確認し、技術中断時は今回の承認の範囲で原因・修正・新試行を記録して再開すること。daemonによる自動コード修正は導入していない。科学的なモデル不一致時はSTOPし、完了後は成功・失敗によらずPhase B2のmandatory STOPを維持する。Phase Cの費用再会計はユーザーの別承認が必要。

[公開snapshotと保存checker](../../artifacts/resource_applicability/track_a_h8_b2_cpu_handshake_running/2026-10-17/README.md)。コード修正と稼働snapshotをcommit・pushし、remote SHAを確認する。snapshotを最終実測結果として扱わない。
