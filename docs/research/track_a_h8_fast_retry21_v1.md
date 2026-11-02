# H8高速化版S4/q8一度再実行・成功後残7

2026-10-13 JST。利用者は提示したposttranspile instruction上限1000万・累積compile21・control probe44と、高速化版S4/q8再実行→成功後残7の方針へ「続けて大丈夫です」と承認した。この承認だけを新plan/authへ結合する。本格H8 resource-map・次科学stage・automatic retryは含まない。

SOURCE `9bfea8bde44cbb2ea9187476ddd496066a8853b3`、branch `track-a-h8-fast-retry21-20261013`、base REVIEWadbd6766。旧245runtime不変、新closure257＋新test1、prior fast portとmetadata workerを再利用しcurrent科学条件・回路・sampler・compilerを変更しない。SOURCE1d064749はCPU継承のsetup STOPのみでscientific worker/lock/call0、9bfea8bdへ修正して新exclusive output/auth/controlを固定した。旧setup証拠を保全しcallを返却/resetしない。

## 固定・変更・会計

linear H8/1Å/STO-3G/tol-only rank28/16 modes/Na=Nb4/sector4900/T.8・input d4c43461・保存state・profile4e32bb9b・compiler1239304c（Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/topologyなし）を維持する。original SOURCE5c207fb8が生成したtask/seedを同じ値で使い、新SOURCEからseedを生成し直さない。完了済み8signal/1164probeの再計算0。

planの差はtranspiled_instructions500万→1000万、compile20→21、control_probe40→44だけ。input/CPU/RSS/AS・thread1・5秒fail-closed・output/FS/inode/quota・OOM/pressure政策は同じ。変更3項目は新grantへ明示し「runtime_authorization=true」はこの限定scopeだけ、本格H8はfalse。

旧失敗caseのactual compile1/control4を保存し、新heavy2/4＋残7compile18/control36＝新成功compile20/control40、累積compile21/control44。trajectory4/occurrence15、過去H6/H4/入力/state/未知matvec予約を保持、返却/reset0。失敗caseのmetricやexact postcountを0/推定で補完しない。1000万で十分かは未測定で、超過/新STOP時は再度停止する。

## 起動と順次接続

run ID `heavy02`、SOURCE9bfea8bd、coordinator PID410647・CPU31、worker PID410910・CPU30。worker AS32GiB/RSS24GiB、aggregate28GiB/admission32GiB、coordinator softAS8/hardAS32GiB・RSS2GiB＋margin2。private pipeline PID410386がcoordinatorをPopen所有しnative wait/reap/exitを記録する。完了・両軸metric・control method・source/input/compiler・native全reap・coordinator退出を確認した場合だけ、残7をCPU20/21/24/co22/23・最大3worker・aggregate76/admission96GiBで起動する。重い/未測定ケースを8workerへ変更しない。

科学runnerは新manifest/grantのbyte SHAとsource closureを検査し、fresh SOURCE/profile/input/carry・CPU・memory/PSI/OOM・FS/inode/quota・exclusive output/one-shotを通過して起動。Python -I -P -B・internal thread1・Qiskit num_processes1、cache/MPL/Numba/tempはhomeのみ。親pipeline CPU31制限を子coordinatorへ継承させず、子exec前にそのscopeのapproved assigned_cpusを復元し、coordinator自身がfresh後に自分のCPUへpinする。他jobのaffinityや共有設定へ変更を加えない。

legacy workerはnamespaceコピーで新plan/auth validatorだけを置換する。disk上の旧module/globalsは不変、既存limits/ResourceTrace/fast cost port/own-run cleanupを保持する。controlはactual evolution二branchのX/Y再利用、hashはcanonical v2 one-pass、logical4/physical2を別記録する。compileは各axis別、20metricは科学的母平均・正式精度認定ではない。

23metadata/mock検査PASS（8.60秒）：変更3capsだけ、original seed/resource保持、失敗actual1/control4を消さない、新成功heavy gate、cap/CPU/profile/grant/task変更拒否、20wrapperと累積21/44、namespace clone不変、heavy再失敗時remaining非dispatch、child CPU復元。既存44/40科学fixture検査を繰返さず、新real input/sampling/transpile test0。起動後の実Qiskit compileは今回承認された本scopeの科学実行として別記録する。

## 入口と停止

private `/home/AbeHiromu/projects/h6-handoff-evidence/20261013/h8-fast-retry21-v1/`：`auth-heavy02`、`heavy02/resource_observations.jsonl`、`control02/heavy_coordinator.log`、`control02/heavy_native_exit.json`、`control02/remaining_auth`、`remaining02`、`control02/terminal.json`。heavy成功だけがremaining起動条件。stop/failure時は終了しautomatic retry0・source/upper勝手変更0・本格H8進行0。

開始観測でworker claim・CPU30・thread1・SOURCEとfirstSTOPnullを確認した。完了や実速度倍率はまだ主張しない。旧コンパイラとのbit完全同一性・population mean/tail bound・全H8map性能は未検証のまま。

[一つの資料入口](../../artifacts/resource_applicability/track_a_h8_fast_retry21_v1/2026-10-13/README.md)。source/testsと軽量plan/auth/profile/launch資料のみGitへ固定し、NPZ・実runtime/cache・credentialsはcommitしない。共有/既存venv/H4/他job変更0、GPU query/use0。
