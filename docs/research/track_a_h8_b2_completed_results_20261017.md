# H8 Phase B2：登録S4全4軸の実測検証完了

**2026-10-17T17:46:48.422662+09:00、残るB0+I p21/q8 S4 sineが正常終了した。実測1,852,640 RZ、固定モデルに対する誤差0%。新しい受入契約で結合済みの保存3軸と合わせ、登録4軸が各々2%未満でPASSした。B2はmandatory STOP。Phase Cの費用再会計は未実施・未承認である。**

## 実測と予測の区別

| 対象 | 軸 | B1固定予測 RZ/軸 | 実測 RZ/軸 | 相対誤差 | 判定 |
| --- | --- | ---: | ---: | ---: | --- |
| B0 p15/q1 S4 | cosine | 1175933/7 ≈ 167,990.4286 | 168,353 | 0.2153638062% | PASS |
| B0 p15/q1 S4 | sine | 1175933/7 ≈ 167,990.4286 | 168,353 | 0.2153638062% | PASS |
| B0+I p21/q8 S4 | cosine | 1,852,640 | 1,852,640 | 0% | PASS |
| B0+I p21/q8 S4 | sine | 1,852,640 | 1,852,640 | 0% | PASS |

判定式は `abs(predicted-measured)/measured < 1/50`。B0のexact errorは `2538/1178471`、B0+Iは両軸`0`。実測欄をモデル推定値で埋めていない。B1の係数は再fitしていない。

## 科学条件と来歴

対象はH8直線鎖・隣接距離1Å・STO-3G・16 modes、Nα=Nβ=4/sector4900、保存tol-only DF28/tol1e-8/cutoff0、signed generation orderとweighted Hermitian projection、T=.8。prefixはB0の15とB0+Iの21、qは1と8、δは.8と.1の登録点だけ。S4は負時間中段を含む非merge 3 S2のYoshida合成、symmetric-directional control、state preparation費用を除くnative full measured wrapper。

B0は元scalarを一回、B0+Iは元scalarと保存full-Fock tail identity scalarを一回。保存unrotated full-H referenceと経験的数値allowance factor1/10を維持した。新SCF/DF/state/signal/control/PR sampling・新候補探索は行っていない。

B1正本commit `83a3c4a6688e66576f85471bf41a45e29cbfb705`、固定model SHA `8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0`。Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/topologyなし。compiler SHA `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、profile SHA `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12` は4軸で同一。

| 実測の出所 | 実行SOURCE | 扱い |
| --- | --- | --- |
| 元B2 run01のB0 cosine/sine | `244c45e752fa703e286d40ece9ad641430ea9c6e` | native exit0・reap・固定モデル判定・fingerprintに結合し、新契約で保存2軸を受入 |
| CPU-handshake recovery attempt02のB0+I cosine | `2e604e4a3a04eee5090e03d9bb83818d7afd3c87` | 同様に保存1軸を受入。元attempt02全体のMEMORY_PRESSURE STOPは保持 |
| PSI-observation attempt01のB0+I sine | `f58da9590597f42ff7260c22d771734096a706bb` | 保存検証済みQPYを一回loadし、compile一回・モデル判定一回で完了 |

今回はsineのcontrol・回路構築を再計算せず、保存QPY 46,793,071 bytes/SHA `8d5a0600ac30526ed7585c090ad277a480a490cf854162123acf9abc92ff548d` を使用した。canonical fingerprint `d406474366df2214e7e79ce793affedceaf1f0fcea542b2ae50c293473ffa5f1` はcompile前後で一致した。全4軸は保存された各軸のnative exit0/reap・control・SOURCE証拠と結合され、未完了の中断軸は実測として使っていない。

## 終了・資源・保全

最新runはdriver・observer/launcherともnative exit0。外側witnessが実native waitを保存し、workerを含むowned identitiesの消滅を終了後監査でも確認した。launcher wall 1632.031秒（約27.20分）、compile API以降の処理（後段fingerprintを含む）は700.198秒、peak aggregate RSS 26.252 GiB。OOM各counterの増分0、cleanup errorなし。fresh CPU/resourceを2回PASS、最終resource/保全/B2 gateもPASS。

33 SOURCE・265 provider・34,273 runtime・31入力を終了後にもbyte照合した。702保護files、135 frozen worktree、286過去immutable filesは不変。指定した非相殺診断の1worktreeは読取専用観測であり、実行契約上「不変」を要求していない。計算中の封印済みSOURCE/config/grantを後から変更せず、V1・V2 CLOSED_NO_GO・H5/H7/M1-Sと既存grant/ledger/markerを保全した。今回の報告は従来の報告専用branch `track-a-h8-b2-sine-psi-running-20261017` に追記する。

前の試行で発生したCPU・保全・MEMORY_PRESSURE STOPと未settle予約はそのまま保持した。利用者の技術修正・再開承認に基づく新SOURCE/新grantで保存済み正常終了軸を受け入れた結果であり、元runを遡及的にPASSに変更したものではない。繰り返した中断分を隠して「元の一回で成功した」とは扱わない。

並行作業による不要な中断を減らすため、最新SOURCEはPSIを観測記録に変更し、OOM/実効headroom・整合性保護を維持した。利用者の追加指示を共通AGENTSに記録したが、実行中のSOURCE/grantへの後付け変更は0。OS/cgroup quota/scheduler/他ユーザーのprocessは変更していない。今回のRAM/4 core/1 compile workerは当runの契約であり、将来の計算の恒常的上限ではない。

## 検証範囲と次の判断

新SOURCEの実行前人工26 test methodsのPASSは[稼働準備報告](track_a_h8_b2_sine_psi_running_20261017.md)に記録した。今回は保存終了receipt・input/source/runtime・過去証拠の読取専用監査と公開物checkerだけを実行し、科学計算・fit・compileを再実行していない。これらはCodex自身のlocal検証であり、独立review・CI・外部再現ではない。

登録4軸でS4費用モデルの誤差基準を通過した。未実測の全prefix/q格子への誤差保証、厳密bias保証、PR母平均保証は得られていない。5精度の最良決定論候補、PR総RZ削減率、非相殺shot/費用比較は今回算出していない。

- **ユーザー**：Phase C（5精度の保存signal/shotと固定・検証済み費用モデルによる再会計）を実行するか別途判断する。B2完了はCの実行承認ではない。
- **Codex**：本報告・公開可能なreceiptをcommit・publishし、remote SHAを照合してmandatory STOPする。次Phaseへ自動移行しない。
- **Claude Code**：非相殺診断の実行は同作業の別SOURCE/契約と明示承認に従う。ここでのB2証拠公開だけでは実行承認にならない。

[全4軸のmachine-readable結合・保存checker](../../artifacts/resource_applicability/track_a_h8_b2_completed/2026-10-17/README.md)。有効grant、入力NPZ、QPY、private runtimeそのものは公開していない。
