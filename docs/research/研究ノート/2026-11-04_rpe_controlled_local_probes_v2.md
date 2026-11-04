# 2026-11-04 Controlled RTE・Hadamard v2：12局所probe整合

run01の固定ID不在を受け、ユーザーの修正・再実行指示内で新SOURCE
`7279f700a58921209b851c8d9c363d38ff2f1d98`を固定し、run02へ1回実行した。
保存218 componentのID/sign/basisでc0=df-fragment-3:Z0Z2（負）、
c1=df-fragment-4:Z0Z1（正・異basis）を固定し、defaultの実payload membership検査を追加した。
有利な数値signalを探索してpairを選んでいない。人工67 PASS後に科学実行した。

linear H4/1Å/STO-3G/保存legacy DF12、Partial L_D3/δ=.02/K2、B physical-state ray/sector70、
extract_identity_phase/cutoff1e-12。6r×q_probe1/all0・q_probe2/stressの12本、24軸がatol1e-10内。
96-bit occupation-energy/signed-component参照とuncontrolled/raw/optimized control両branch、Re/+Im wrapperを照合。
最大axis残差上端約4.03e-15、最大vector残差上端約2.66e-15、wall約15.25秒、native exit0。
36 evolution build、48 control branch action、24 wrapper/ideal meanを固定call内で実施した。
sampling/compile/長q回路/成功率・費用再解析0、v2内retry0。

[報告・8原本](../rpe_controlled_hadamard_local_probes_v2_execution_20261104/REPORT.md)に結果と限界を保存した。
run01停止・旧SOURCE/結果/条件付き判定は変更しない。
今回のPASSは指定pair・trajectory・q_probe1/2の局所一致であり、保存全18qや任意component/trajectoryの保証ではない。
次は別指示後の全18categorical SOURCE対応・長q phase転送の静的整理を提案。
全体UNDETERMINED/false、U_exec=nullを維持し、公開照合後mandatory STOP。
