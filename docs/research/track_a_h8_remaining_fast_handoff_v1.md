# H8 heavy完了後・残7ケース高速化引継ぎ

2026-10-13 JST。利用者はS4/q8を現runで完了させ、残7だけ高速化版へ引き継ぐ方針を指示した。旧pending continuationを既存STOP_PIPELINEで退役し、heavy本体へ停止要求を出さず専用worktreeで接続を実装した。

SOURCE `507b69243d68c590212928201f1528b22b6a8282`、branch `track-a-h8-remaining-fast-handoff-20261013`、起点REVIEWf8c50394。旧SOURCE5c207fb8の245runtimeと高速化SOURCEf7c46135を保全、新runtime closure253＋新test1。input d4c43461/rank28/sector4900・profile4e32bb9b・compiler1239304c/opt1/seed17/basis rz,sx,x,cxは維持する。

## 引継ぎ経路

prepareはsource/input/profile・旧signal/1164probe・退役controller・元task/seedをtemplateへ固定する。seed生成に新SOURCEを使わずOLD_SOURCE5c207fb8の登録[0,1,2,4,5,6,7]を保持する。armは旧heavy両軸・native exit/reap・完全result・control・counter・sourceを照合し、旧coordinator/worker退出後だけfinal manifest/authを生成する。旧grant/output/lockを再利用しない。STOP/partial/missing証拠を拒否し、新grantでrunが一回起動する。

旧2compile＋新18compile＝20、trajectory4・occurrence15・logical control40を保持する。signal再計算0、failed/unknown carry返却/reset0。CPU worker20/21/24・co22/23、worker AS32GiB/RSS24GiB・内部thread1、aggregate76GiB/admission96GiBと既存5秒fail-closed監視・fresh quota/FS/inodeを維持する。未測定ケースを8workerへ昇格しない。

新runは`h8_fast_cost_port_v1`のsingle-pass hashとcontrol branch再利用へ接続。logical probeを維持し、実E作用2/小X,Y suffix2として別記録する。旧heavyと新7のSOURCE・validation method来歴を区別し、同じquantum circuit/compiler/seed条件の20metricを集約する。actual compiler/settings/circuit条件を高速化のため変更しない。

40人工metadata/mock検査PASS：original seed・18call・親未完了/STOP/退出待ち、所有/退役controller、wrong source/seed/compiler/metric/counter/paired axis拒否、二重dispatch/使用済output拒否、plan/auth/source/carry binding、combined20の算術、guard失敗のcount telemetry。新SCI/NPZ読込/実transpile0、旧44検査を繰返さない。新workerは同じinstruction guard失敗時にtracebackが残る間だけcompiled.dataの長さをscalar記録し、今回の旧runの未記録値は補完しない。

## 実際の親STOPと現在状態

準備中に旧heavyがworker exit1、理由`TRANSPILED_INSTRUCTION_CAP`で終了した。transpileはcosine1回に到達・終了したが、posttranspile500万instructionsを超え、metric保存前にSTOP。正確な超過個数は旧sourceが保存しておらずunknown/null。保存済みwrapper0/20・RZ/CX/depth/sizeもnull、費用0として扱わない。初回control4は合格済み（max error1.8552e-13）。

旧case wall7007.756秒（1時間56分47.8秒）、workerHWM7.421757GiB/VmPeak8.081406GiB、RSS24/AS32上限内。transpile224.849秒、controlとhashがwallの大半を占める。[stage記録](../../artifacts/resource_applicability/track_a_h8_remaining_fast_handoff_v1/2026-10-13/handoff_status.json)はinclusiveなのでnested hashを重複加算しない。全worker wait/reap・cleanup error0、coordinator/workerは退出確認。coordinator正確exit code/終了時刻は未記録/null。旧continuationのUSER_STOPはheavyのfailure原因と区別する。

親成功条件が不成立なのでtechnical plan未seal・新grant/arm/run未発行・新compile/trajectory0。本格H8なし。heavyの結果を成功扱いしたり、残7を無断先行起動したり、旧heavyを自動retryしない。

## 次の判断案（未承認）

高速化は同一回路/費用を維持するため、posttranspile instruction上限のSTOPを解消するには別のresource guard判断が必要。候補は500万→1000万instructions（正確な出力count未記録なので十分性未確認）、失敗compile1を保持して総compile20→21、再検証control4を保持して総control40→44。trajectory4、CPU/RSS/AS/compiler/science条件は維持する。新SOURCEで同じS4/q8を再実行し、完了後に残7を接続する方針への変更は別承認が必要。失敗分の返却/resetを行わず、科学回路を縮小してguardへ合わせない。

これは承認案だけで、sourceの500万/20/40上限や実行grantを変更していない。実出力count・資源・費用・full campaign時間の推定を欠測から保証しない。まずguard/追加callの方針を確定する。本格resource-mapは未承認で、起動しない。

[資料入口](../../artifacts/resource_applicability/track_a_h8_remaining_fast_handoff_v1/2026-10-13/README.md)。共有環境・既存venv・H4/他job/GPUに変更・query/useなし。軽量source/tests/資料だけ公開する。
