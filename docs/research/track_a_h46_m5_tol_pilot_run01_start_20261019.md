# H4/H6 m5_best：tol-only technical pilot run01 起動記録

2026-10-19 13:07 JST時点。**起動済み・安定稼働、完了は未確認。** ユーザーの「technical pilotの一回実行」への明示承認を新4grantへ結合し、13:03:26 JSTに固定SOURCEで一回起動した。計算はバックグラウンドで継続する。これは最終結果報告ではない。

科学SOURCE：`c7d53f04c3be131078ee91e8bef3c78ca434ed3e`。準備資料commit：`54f3e8da374b2ad60755dd6bbc6ae3b8150c6502`。branch：`track-a-h46-m5-tol-native-readiness-20261019`。準備・固定planの説明は[readiness報告](track_a_h46_m5_tol_native_readiness_20261019.md)、起動証拠は[summary](../../artifacts/resource_applicability/track_a_h46_m5_tol_pilot_run01_start/2026-10-19/summary.json)と[起動gate](../../artifacts/resource_applicability/track_a_h46_m5_tol_pilot_run01_start/2026-10-19/fresh_gate.json)。

## 承認・実行範囲

対象は直鎖H4/H6、隣接1 Å、STO-3G、保存tol-only1e-8入力、T=.8、P=0。H4は保存DF10/p3、H6は保存DF19/p5。DF rankは保存された観測値であり、rank指定の新生成ではない。H4 legacy DF12を混ぜず、全SCF/MO/DF/state規則の系間同一性は未認定のまま。

|契約|公式と構成|最大compile軸|PR本数|full-H reference|
|---|---|---:|---:|---:|
|H4_TOL_PILOT_DET|S2/Yoshida4/m5 × full/B0/B0I、q2|18|0|1|
|H4_TOL_PILOT_PR|S2/Yoshida4/m5、q2/K4、signed piece当たりr2|12|6|共有|
|H6_TOL_PILOT_DET|S2/Yoshida4/m5 × full/B0/B0I、q2|18|0|1|
|H6_TOL_PILOT_PR|S2/Yoshida4/m5、q2/K4、signed piece当たりr2|12|6|共有|

合計30 evolution task、最大60軸、PR12 trajectory、reference2 job。m5係数版 `PF_M5_BEST_20D64C2_BINARY64_V1`、11 signed S2 piece、symmetric-directional/full Hadamard wrapper、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17を維持。scalar一回・元finite-RTE law・full-H referenceの位相無調整を維持する。

主比較へのm5追加採用AD03、後続124 signal候補、単価fit、5ε会計、独立確認、ランキングは承認されていない。shot allowance/headroom/厳密bias certificateは採択しない。このpilotを独立確認標本にpoolしない。

## fresh gateと保全

4契約のSOURCE/input/provider/runtime/seedと固定plan SHA、専用の未使用run rootを照合した。H4各309/H6各310 fileのsealが一致し、使われるSOURCE16 fileもGit objectと一致した。暫定120seedと既知除外4483の封印・導出が一致。未知外部jobの全seedを網羅したという主張はしない。

起動時の有効affinity64、idle CPU約52.55相当、MemAvailable約463 GiBから最大30 workerを割り当てた。恣意的な歴史的worker capは適用していない。own processの内側BLAS/transpilerを1threadにしてnested並列を避け、独立群・trajectoryをspawn poolで並行する。referenceは各系一回だけ共有し、cosine判定後sine、代表sample0 PASS後に同群sample1を投入する。

元proposal02のDRAFT/INVALID grantとplan/seed/sealは上書きしていない。今回の承認receipt・有効grant・configは別の`authorization_run01`へ保存し、4markerは一回分として消費済み。実際に使うsealed fileは保全し、他worktreeのHEAD/statusを凍結せず、他chatへ書込・signalを行っていない。共有OS/package/cgroup設定の変更もない。

## 安定稼働snapshotと終了条件

13:07 JSTのsnapshotでは、H4/H6のreferenceは両方PASS、21/30 task完了、失敗laneは0。残りには未投入の依存sampleもあり、pending件数だけを全残task数とは扱わない。完了taskの再実行はしていない。[観測snapshot](../../artifacts/resource_applicability/track_a_h46_m5_tol_pilot_run01_start/2026-10-19/observation_snapshot.json)は終了証拠ではない。

finiteかつ1e-9未満のnative/occupation/Hadamard/compile前後検証とSOURCE照合を維持する。sample0代表の正しさ検証とsample1費用取得を区別する。実OOM・継続不能な不足・own process異常・科学的不一致・identity破損は該当scopeをSTOP。単なる上位PSI/CPU競合では科学workerをkillしない。自動retry・追加compile・再fitはない。

launcherは終了時にlane別と全体のcompletion witnessを保存し、worker exit/reapを確認してmandatory STOPする。外部wait4 processもnative終了code・回収・RSS・wallを保存する。**現在はnative exit/全worker回収/全60軸PASSを確認した段階ではない。** 全Phaseの完了後も次Phaseを自動起動しない。

## 成果物と次の担当

公開可能な起動bundleは承認とgrantのprojection、fresh gate、reference receipt、静的観測snapshot、保全gate、inventory。入力NPZ/seed値/有効configのprivate path・event原本は含めない。技術snapshotの構造・hash照合は作者自身の確認で、独立reviewerによる検証ではない。公開pushは行わない。

Codex：計算終了後、native exit・全worker回収・SOURCE/input/ledger/60軸の整合を確認し、保存された結果を転記してlocal commitする。これはまだ未実施。科学的な再解析・新compileはこの終了監査に含めない。

ユーザー：最終pilot報告を確認した後、後続Core準備・実行の範囲とm5主比較採用AD03を別途判断する。実行中のpilotに追加操作は必要ない。
