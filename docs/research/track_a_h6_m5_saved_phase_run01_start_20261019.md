# H6 full m5_best：保存回路4再生の一回解析起動

2026-10-19。**承認を受けて15:04:07 JSTに一回起動。計算中で、完了・最終診断は未確認。** この文書は起動とsnapshotの記録であり、成功結果ではない。

SOURCE：`99e8723d402245a9ad923e7dbc0800f47bc575b6`。準備commit：`3484fc233733e66b4056616d1e04926162a00354`。新しい起動・結果資料branch：`track-a-h6-m5-saved-phase-run01-20261019`。

実行plan SHA256：`b2b2a818dd318a9a981cacae3eeb59617501992e9d6e60ff45d1c151dc288254`。[準備契約](track_a_h6_m5_saved_phase_readiness_20261019.md)、[起動bundle](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_run01_start/2026-10-19/summary.json)。

## 1. 承認と固定範囲

ユーザーの「進めて大丈夫」は、選択された「固定4再生の一回解析を承認」に対応する明示承認として記録した。今回の通常再現atol1e-12を採用する。原runのraw基準1e-9未満、旧STOP、旧58受理軸、未検証費用を変更しない。

対象は保存H6直鎖1 Å/STO-3G/tol-only1e-8 DF19、T=.8、P=0、full m5_best/p19/q2、symmetric-directional/full Hadamard cosineの1点。保存wrapper/compiledと同じ保存lowを使い、通常/拡張各2回の計4再生だけを行う。

新compile・sampling・SCF/DF/state/full-H reference生成、signal探索、shot/費用再会計、fit、ranking、sine、m5主比較採用AD03・後続Coreは承認範囲に含めない。DRAFT/INVALIDは上書きせず、新しい承認receipt・config・grantをprivate authorization_run01へ作成し、一回消費した。

封印済みplanのDRAFT表示は準備時点の履歴として保持する。このrunの承認状態は新承認receiptへ結合し、原planやSOURCEを後から書き換えていない。正式な共通プロトコル変更でもない。

## 2. 起動gateと進行

SOURCE・12入力role・runtime1831fileのseal、旧STOP/native exit/reap、未使用output/supervisor、684保護fileの不変を起動前に確認した。起動直前とlauncher自身のfresh resource gateもPASS。

契約は最大4 worker。今回は観測idle CPU約1.34 core相当から**1 worker**が選ばれ、4 jobを順に投入する。affinityは256論理CPU、実効空きRAMは約491.7 GBで、RAM不足による制限ではない。既存のCPU利用状況に基づく固定admission規則を実行中に緩めていない。shared OS/権限/cgroup/他ユーザーprocessの変更は0。

以下は初期snapshotの時点で通過したもの。

- 拡張精度：binary64 mantissa53 bit、longdouble64 bitで実際の追加精度を確認。
- 保存診断の数値12項目と重複lowを再現。原raw差1.504043707445665e-9を再現し、旧FAILを保持。
- 2 QPY：13 qubit/ancilla12、layout None、最後の測定だけ、数値parameterと許可gate classを確認。
- WRAPPER_ORDINARYは保存beforeとの再現にPASS（保存ベクトル差0）。最新snapshotではCOMPILED_ORDINARYが再生中で、CPU時間増加を確認。STOPは未観測。

[時点付きsnapshot](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_run01_start/2026-10-19/summary.json)と実runの状態は区別する。snapshot後の完了をこの文書から推測しない。4再生結果の比較・拡張精度での差・最終exit/reap・最終保全は未監査。

## 3. 継続とSTOP

SOURCE/入力/runtimeの不一致、未知gate、非finite、通常再現失敗、own異常はSTOP。自動retryや追加jobはない。host/上位cgroupの一時PSI・CPU競合は観測だけとし、他worktreeを凍結しない。

外部supervisorはown launcherをwait4してnative exitを保存する。launcherは全own workerを回収し、結果SHAとcompletion_witnessを保存してmandatory STOPする。プロセスはdetachedで動いており、チャット終了後も同じ一回実行を継続する。

次の担当は**Codex**：終了後に結果・4 job/ledger・SOURCE/input SHA・native exitとworker回収・旧結果保全を監査し、公開可能な結果を報告してmandatory STOP。ユーザーへの追加承認依頼はこの固定実行には不要。後続Phaseや新compileには別承認が必要。公開pushは行わない。
