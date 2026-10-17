# H8 B2：PSI監視契約の技術修正とsine限定再開

**2026-10-17T17:22:24.738031+09:00時点、新試行は保存済みsine回路の読込・fingerprint確認を継続中。fresh gateを2回PASSし、157.1秒間の複数観測で同一workerのCPU増加・監視継続を確認した。sineの実測RZ・B2最終判定はまだない。**

実行SOURCE `f58da9590597f42ff7260c22d771734096a706bb`、code branch `track-a-h8-b2-sine-psi-observation-20261017`。この報告は計算中のSOURCEと別worktree/branch `track-a-h8-b2-sine-psi-running-20261017` へ保存する。旧SOURCE、grant、ledger、markerとSTOPは不変。

## 旧試行の停止と今回変更した条件

直前のSOURCE e3ecc03a…のsine専用attempt01は17:01:25 JSTにcompile APIへ入り、17:01:35 JSTにMEMORY_PRESSUREでSTOPした。本人user/session cgroupのfull avg10は0.36%、hostは0.18%、OOM増加なし、実効空きRAMは約443.1 GiB、peak aggregate RSSは約25.40 GiB。未完了sineのRZは不明で、モデル不一致ではない。全owned回収・witness実native launcher exit1を保存した。元runのSTOPと未settle予約を返却・改変しない。

現行の共通AGENTS.md（2026-10-17資源利用方針）と、利用者の「計算が途中で中断された場合は修正して計算を再び開始してほしい」に基づく技術修正として、新SOURCE/新grantでPSIを診断記録へ変更した。従来のhost full avg10>=1%・各cgroup full avg10>0を新試行のSTOP gateには使用しない。旧実行の判定を遡及変更したものではない。

| 新試行の条件 | 処理 |
| --- | --- |
| host/user/session/上位sliceのPSI | 1秒ごとにraw snapshot、full avg10、累積stall値を保存。非ゼロ単独ではSTOPしない |
| OOM・OOM_kill・group killの増加 | 即STOP |
| 実効空きRAM・cgroup headroom不足 | 即STOP。fresh 80.3125 GiB、実行中reserve16 GiB等の明示契約を維持 |
| 観測の欠測・非finite・cgroup集合変化 | 即STOP |
| owned processのAS/RSS/CPU・native終了・lineage・監視gap・disk | 従来の明示契約による監視とSTOPを維持 |
| 各軸モデル誤差2%以上 | 科学的STOP。再fit・追加候補・不一致のretryはしない |

OS、cgroup quota、scheduler、共有ライブラリ、他ユーザーのprocessは変更していない。PSI修正は自分のプロジェクトのobserverだけに適用した。今回の4 core/1 compile worker/RAM条件はこの試行の契約であり、将来の新作業へ恒常的な資源上限として移送しない。

## 保存値の再利用と固定する科学条件

H8直線鎖/隣接1Å/STO-3G/16 modes、Nα=Nβ=4/sector4900、保存tol-only DF28/tol1e-8/cutoff0、signed generation order・weighted Hermitian projection、T=.8、B0+I p21/q8/δ=.1/S4。負時間中段を含む非merge 3 S2、symmetric-directional control、保存scalar/full-H reference、経験的allowance factor1/10、state preparation除外のfull measured native Hadamard wrapperを維持する。

B0の保存2軸と、SOURCE2e604e4a…で16:13:42 JSTに完了したcosineを再利用する。cosineは実測1,852,640 RZ、予測1,852,640、誤差0%、native exit0/reap/fingerprint前後一致。これはモデル推定ではなく保存実測。保存sine QPY 46,793,071 bytes/SHA8d5a0600…/fingerprint d4064743…と元4 controlのPASSをbyte結合し、残るsineだけをcompileする。

B1正本83a3c4a6…/model8420b9f2…、各軸予測1,852,640 RZ、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/backend・topologyなし、compiler1239304c…/profile622ce4a8…は不変。`abs(predicted-measured)/measured < 1/50` を維持し、新SCF/DF/state/signal/control再計算/fit/費用再会計は0。sineのQPY load1・compile1・判定1に限定する。

## 人工検証と実行証拠

[observer実装](../../scripts/h8_phase_b2_recovery_observer.py)・[PSI方針tests](../../tests/test_h8_b2_psi_observation.py)。

固定Python3.11.0rc1で人工test methods26件PASS（起動6・保存spool受入9・並行保全6・PSI方針5）。PSIが非ゼロでもrawを保存して継続する実owned supervisor、OOM/空きRAM不足/不正観測のSTOP、signal/native wait、保存値改竄拒否、1軸限定と2%不一致STOPを確認した。実H8入力を使う新しい数値解析・compileは人工検証で実行していない。これは自身のlocal検証であり、独立review・CI・外部再現ではない。

新しいscope/grant/実行域で初回CPU/resource gateとdriver照合後のfresh gateをPASS。現在のroleはdriver1/observer2/worker3/witness4、snapshot時aggregate RSS約4.95 GiB、観測最大gap4.908秒。現在のstageは保存回路の読込・canonical fingerprint確認であり、compile完了の主張ではない。

33 SOURCE/265 provider/34,273 runtime/31 inputを固定し、286過去immutable files、702保護files、135 frozen worktreeとconfig/有効grantを稼働中に照合した。指定した1並行worktreeは読取専用の更新観測として分離し、「不変」とは報告しない。V1・V2 CLOSED_NO_GO・H5/H7/M1-S、元B0/過去B2のSTOPを保持する。公開物はgrantのhashとpath-redacted projectionのみ。

Codexの次の作業はsineの実測・各軸判定・native終了・保全を確認すること。技術中断時は原因と新試行を今回の承認範囲で記録して修正・再開する。自動コード修正daemonは導入していない。B2終了時mandatory STOP、Phase Cや非相殺費用stageは別承認。全4軸PASSはまだ確定していない。

[公開snapshot/checker](../../artifacts/resource_applicability/track_a_h8_b2_sine_psi_running/2026-10-17/README.md)。計算中のSOURCE worktreeを変更せず、報告をcommit・pushしてremote SHAを確認する。
