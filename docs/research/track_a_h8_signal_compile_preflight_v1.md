# H8 signal/reference・16-mode回路・8ケースcompile事前検証

2026-10-13 JST。利用者の「この部分を行って」を根拠に、保存済み入力のsignal/reference・native circuit/control検証と8ケース20compileを実行する。本格H8比較契約の採用、92候補探索、独立32確認の開始は含まない。

## 固定条件

linear H8/1.00 Å/STO-3G/16 modes/Na=Nb=4/sector4900、tol-only DF1e-8、rank28、全fragment・生成順・cutoff0を保持する。snapshotは1,299,688 bytes、SHA256 `d4c4346127e97c241439ae1c4ae010962b0456b19f461cfd3d9b06e7d4a50955`、生成SOURCE `2ff3fc9f6e1091cbd9915507acfa74112c7d3055`。SCF/DF/stateの再生成0。T=.8、元の有限RTE law、normalization、paired X/Y、shot会計、Qiskit1.3.0、rz/sx/x/cx・opt1・seed17・topologyなしを維持する。

本段階のSOURCE `5c207fb874083022e5d48d69c879d0bcb40f894c`。旧base `0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41`の全runtime238件は不変、新closure245件。Python環境は既存H6専用venvを変更せず使用し、採用profile SHA `4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75`を維持する。旧host compiler output bit同一性は未検証。

## 実装と数値検証

`h8_sparse_reference_v1.py`で独立fermionic ladderからoccupation CSRを作る。primitiveの厳密cross-spin zero・alpha/beta equalityを確認してからG²を作用する。HamiltonianやtailはCSRの二回作用を使い、4,900平方のdense行列や16-qubit full unitaryを作らない。referenceは`expm_multiply`の一回時間作用・二半分時間作用と保存stateのRayleigh residualを照合する。production Numba matvecとの3probe比較も行う。

primary deterministic actionは各スピンの70次元occupation行列の固有系とtensor作用を使う。interleaved modeからalpha/beta別順序へのfermionic signを明示し、定数とnegative Yoshida時間を保持する。prep、primitive、time phaseを同じ入力内で再利用する。登録388時間点×保存state/先頭/末尾3probe=1164primitive検査で、実native16-mode回路・spin作用・独立CSR作用を照合する。

Hornerによるcorrected/raw平均と、独立forward Taylor・exact-tail作用を8候補で比較する。stateを再正規化しない。H6 dense400 expm/eigh証拠をH8のPASSへ置き換えず、reference方式の差を明記する。数値allowanceは従来係数で算出した経験値であり、厳密誤差証明や母平均上界ではない。

## ケースと費用予算

| 対象 | q | r | K | primary compile | ordinary追加 |
|---|---:|---:|---:|---:|---:|
| full rank28 S2 | 1 | — | — | 2 | 0 |
| full rank28 S4 | 1 | — | — | 2 | 2 |
| full rank28 S2 | 8 | — | — | 2 | 0 |
| full rank28 S4 | 8 | — | — | 2 | 0 |
| prefix7 PR S2 | 1 | 4 | 2 | 2 | 0 |
| prefix7 PR S2 | 2 | 4 | 2 | 2 | 2 |
| prefix7 PR S2 | 4 | 4 | 4 | 2 | 0 |
| prefix7 PR S2 | 8 | 2 | 2 | 2 | 0 |

合計8group、20compile、4trajectory、15 occurrence、40control probe。precision適格性によらずfeasibility情報を取得し、1 PR trajectoryを母平均推定に使わない。namespace `H8_EIGHT_CASE_FEASIBILITY_V1_20261013`、seedはSOURCE・入力SHA・cell・replicaに結合しworker数・順序に依存しない。H6/H4のeventやcacheを流用しない。

H6 known1716compile/842trajectory＋予約20/10、H4補完532/256、H8 SCF1/DF1/state成功61を保持する。停止stateのexact matvecはunknown、予約10000を保持する。old production H4予算は別lineageのまま。返却/reset0。新20/4/15は本scopeだけの枠で、本格campaignの累積上限を採用したものではない。未完了分・途中停止時の消費を0にしない。

## 資源・監視・起動

最初にsignal単独、次にheavy S4/q8単独、完了/cleanup後に残7ケースを実行する。worker AS32GiB/RSS24GiB、coordinator softAS8/hardAS32GiB/RSS2GiB、margin2GiB。heavy最初はworker CPU30/co31、job admission32GiB/aggregate28GiB。残ケースの最大3workerはCPU20/21/24、co22/23、aggregate76GiB/admission96GiB。8heavy同時起動は認可しない。実測を基に本格計算のworker数を提案する。

全内部thread1・Qiskit num_processes1・Python -I -P -B。own-run affinityのみ。独立coordinator監視interval.2秒、staleness/観測5秒以内、host PSI<5%・全cgroup PSI0・新OOM0を維持する。fresh host availableは152.25GiB＋job admission、effectiveはjob admission以上、running host168.25GiB/effective16GiB。欠測・所有不一致・上限超過はfail-closed、自分の未reap process groupだけcleanupし、first STOPを保持する。

output2GiB、worker単位32MiB、log2MiB、監視journal32MiB。private cache/temp/Matplotlib/Numbaはhome内に置く。fresh filesystem available10GiB・inode100000・quota既知、CPU affinity、SOURCE・入力・profile、未使用output・one-shot lockを直前に確認する。共有環境・既存venv・他job・旧H4を変更しない。GPU query/use0。

実行用prepare/runnerは`scripts/resource_applicability/{prepare,run}_track_a_h8_signal_preflight_v1.py`。manifest/grant SHAを必須にし、各scopeの新grantとexclusive outputを分ける。REMAININGは完了signalとheavy terminalにbindingする。各scope完了/STOPで終了し、retryや本格H8自動進行を行わない。現在の[資料入口](../../artifacts/resource_applicability/track_a_h8_signal_compile_preflight_v1/2026-10-13/README.md)に進行と実測を保存する。

人工15検査PASS：complex spin factorizationと独立occupation全column、negative/zero time、CSR/expm・Horner/forward、非finite・sector escape・spin mixing・exact tiny coefficient、人工native16-mode、費用予算・seed・weighted admission。人工testsに実input/trajectory/Qiskit transpileは使わない。実H8結果は別に報告する。
