# 現行N3の状態と、承認後だけ使う停止手順

2026-11-06 JST。**本taskは読み取り専用の状態確認を行い、現行jobへsignalを送信せず、停止・設定変更・再起動をしていない。** 状態は取得時点の観測であり、後刻まで固定された状態ではない。

## 実行を区別する

| 実行 | 固定SOURCE/結果 | 状態 |
|---|---|---|
| 旧N3 | SOURCE03928ce55bc5a1bfd070aa9f0a6bf8a990a0ff25、結果d97e916526f431280e687815d3e970bbf03a077e | 中断保全・STOP、397 bit/397価格、原因UNKNOWN |
| Bの最初のlauncher | SOURCE5f2eac41cbc324643d663b8110b1cf127aed54df、結果ade6a25fa554903bb477e0f32142c154d01fad1d | import失敗、科学開始0、trigger撤去・STOP |
| 新B・manager_run02 | 科学SOURCE **ee49b53fd6f9a17d715edb1df3aac8cbf5731d7a**、途中公開38defac3bff9eac7585e61d49098aec004e2016d | **RUNNING**、今回停止していない |

新Bの契約は[EXECUTION.md](../o2_a1_n3_B_retry_20261106/EXECUTION.md)、campaign ID=A1-N3-B-H4-m0-q1-20261106-run01、seed20261107。
保存H4/1.00 Å/STO-3G/legacy DF12/8system qubits/全4電子sector/L_D3、m0/q1/δ=.02/r1/K2/atol0、actual Q/selected full、N2 ground、X/Y、認証済exact engine。
現行4方式のε=1/8、α=1/20、旧quotaを、本taskから変更していない。新752-shot契約案を現行途中へ適用していない。

実worktree：
`/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/o2-a1-n3-B-retry-20261106`

science directory：`artifacts/o2_a1_new_target/2026-11-06/n3_campaign_B_run01`
manager directory：`artifacts/o2_a1_new_target/2026-11-06/n3_campaign_B_manager_run02`

読み取り専用観測は[artifact](../../../artifacts/o2_a1_n3_scope_redesign/2026-11-06/)のCURRENT_OBSERVATION_INITIAL/FINAL、OWN_PROCESSES、EVIDENCE_ORIGINSを参照する。
親・cron dispatch・application manager・outer recorderについてPID/start ticks/boot/PGID/SIDとUIDを照合した。parent logは進捗を更新しており、STOPなし・worker125が存在した。
SOURCE/SETUP/INVOCATIONと旧契約、四つの固定参照03928/d97e/857350e/a53cを実際に取得してSHA256を記録した。
A1 X/Yは各320 train完了・rank4・range guard通過。凍結係数はA1_X_FROZEN/A1_Y_FROZENの保存exact値を参照する。再fit・新分散評価はしていない。

globalなmemory.currentや共有cron cgroupのpeakをjob単独RAMと呼ばない。monitorのworker RSS和は共有pageを重複計上し、max VmRSSの瞬間観測をlifetime peakと呼ばない。実際の親VmHWMはOWN_PROCESSESに別記した。
現行jobが独立に取得を続けるため、観測間でbit/価格/call数が増えても本設計taskの新科学callではない。

## 正規の停止経路

既存managed.Runtime.signalsは親のSIGTERMを捕捉し、STOP_REQUESTEDを耐久保存して共有STOP eventを立てる。
B parentのexcept BaseException→finalizeはpool回収より前にSTOPを保存し、engineは次のguardで停止、新規submitを拒否、既存bit→価格順序を保持する。
worker回収後にsummary/reconcile/ATTEMPTS/CAMPAIGN_LEDGER/STOPを保存し、managerは親raw exitとsemantic STOPを別々に記録する。正常な操作停止でも親は非0終了になり得るので、その値だけでOOM/事故と分類しない。

**指定stageの末尾で自動停止する入口は現行driverにない。** 親TERMは現在のstageを最後まで完了させる約束ではなく、取得済データを保全する協調停止である。
既に保存bitがあって価格がないtrial、開始したがbitがないtrialが残り得る。欠測はUNKNOWNとし、停止後に勝手に再測定・再compile・不足価格補完を行わない。

## 必要な承認

今回のユーザー指示§2は「実行中のジョブを勝手に停止・変更・再起動しない」、末尾は「今回は現行ジョブの停止…を許可しません」と明示している。
したがって、下の一回の停止・既存finalizerによる回収/保全を行うには**現行新Bの停止への別途明示許可**が必要。本書公開は停止許可ではない。
過去の意図しない中断の修復許可を、今回は明示不許可の停止/変更/再実行へ流用しない。新しいpilot実行許可も、この停止許可と区別する。

## 承認後にだけ実行する具体的手順（今回は未実行）

1. 現在のSOURCE/identity/STOP有無を再取得。自然終了済みならsignalを送らず、その結果を保全する。
2. 記録された**親processだけ**をpidfdで参照し、identity・UID・SOURCEを再照合する。pidfdはPID再利用との競合を避けるために使う。[Python pidfd_open](https://docs.python.org/3.12/library/os.html#os.pidfd_open)、[pidfd_send_signal](https://docs.python.org/3.12/library/signal.html#signal.pidfd_send_signal)。関数の存在はこのhostで確認したが、現行jobへ使用していない。
3. 操作理由をexclusive journalへfsyncし、親だけへSIGTERMを一回送る。操作journalはprovenanceであり、承認を証明する新しいreceipt/grantではない。
4. manager/親/workerの終了・STOP_REQUESTED/STOP・bit/価格/欠測を読み取り監視。既存finalizerが終わるまで待ち、停止中に旧データを移動・上書きしない。
5. 旧SOURCE/全journal/STOP/cache/ledger/実現費用を新規保全先へ保存し、内容を照合して公開する。強制killや同jobの自動再起動はしない。回収不能ならUNKNOWNを保持して報告する。

以下は**将来承認後の操作例**であり、今回の作業では実行していない。

```python
# Run with the fixed venv and this live worktree's src on PYTHONPATH.
from pathlib import Path
from datetime import datetime, timezone
import json, os, signal
from trottertracks.algorithm_codesign import a1_n3_lifecycle_observer as obs
from trottertracks.algorithm_codesign.a1_n3_managed_runtime import durable_new

root = Path('/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/o2-a1-n3-B-retry-20261106')
out = root / 'artifacts/o2_a1_new_target/2026-11-06/n3_campaign_B_run01'
expected = json.loads((out/'PARENT_IDENTITY.json').read_text())
if (out/'STOP.json').exists():
    raise RuntimeError('already stopped/completed; observe and preserve, do not signal')
if json.loads((out/'SOURCE.json').read_text())['source_commit'] != 'ee49b53fd6f9a17d715edb1df3aac8cbf5731d7a':
    raise RuntimeError('SOURCE mismatch; do not signal')
fd = os.pidfd_open(expected['pid'], 0)
try:
    current = obs.process_identity(expected['pid'])
    if any(current.get(k) != expected.get(k) for k in ('pid','start_ticks','boot_id','pgid','sid')):
        raise RuntimeError('identity changed; do not signal')
    status = Path('/proc', str(expected['pid']), 'status').read_text()
    uid = int(next(line.split()[1] for line in status.splitlines() if line.startswith('Uid:')))
    if uid != os.getuid():
        raise PermissionError('refuse another owner')
    durable_new(out/'OPERATOR_STOP_INTENT.json', {
        'reason': 'USER_AUTHORIZED_SCOPE_REDESIGN_STOP',
        'utc': datetime.now(timezone.utc).isoformat(),
        'target_identity': expected, 'restart': False,
    })
    signal.pidfd_send_signal(fd, signal.SIGTERM)
finally:
    os.close(fd)
```

pidfd/API権限やidentity照合に失敗した場合はsignalなしで終了し、広いprocess名へのpkill、group kill、sudoや共有service停止へ切り替えない。
このpidfd送信例は実jobでは未実施。既存parentのTERM/STOP/bit保全経路はqualified managerの人工13件・integration6ケースの既存結果を参照し、今回それを実job停止試験へ拡張しない。

## 本taskのmandatory STOP

新契約草案・技術helper/人工test・status/停止案の公開照合が終わったら本設計taskをSTOPする。
**これは現行新Bのprocess停止を意味しない。** 停止を許可するユーザー指示がない限り、jobは元の契約で動作を続ける。


### 最終読み取り観測（2026-11-06T14:43:36.240148+00:00 UTC）

RUNNING、live worker=125。保存bit=896、価格=896。これらは非原子的な途中観測で、完了数・全費用ではない。
最新parent進捗はCURRENT_OBSERVATION_FINAL.jsonのparent_tailを参照。新pilotへの流用なし。停止/変更/signal送信は0、現行科学processは継続中。
