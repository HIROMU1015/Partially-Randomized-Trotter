# 2026-11-03 RPE候補A：preparation検査で1回の試行を停止

ユーザーの実行許可後、固定SOURCE `09237b48ea22ad10d860e67e4a2c1465aad65e41`を1回使用した。
linear H4/1Å/STO-3G/legacy DF12、8qubit・4電子sector70、Bの保存physical state、
Partial L_D3、δ=.02、r1/K2、q1/q8、cosine/sine各510/454の計画である。

[実行報告](../rpe_fresh_iid_measurement_a_execution_20261103/REPORT.md)と
[原RESULT](../../../artifacts/rpe_fresh_iid_measurement_a/2026-11-03/run01/RESULT.json)を新規保存した。
DF preparation後のhash/component検査で`ValueError`、exit1。
`INTERRUPTED_OR_INVALID_MANDATORY_STOP`、科学statusは`UNDETERMINED`。
sampling/state-action/仮想shot・統計診断には未到達であり、保存meanとの不一致を意味しない。

静的調査では、保存signalの係数閾値1e-12に対し、A runnerが引数省略により既定値0.0を使う不整合を確認した。
準備実装の引数指定漏れで、人工18 testsはこの実H4入力条件の一致を覆っていなかった。
実行時のhash/component数は保存されていないため、値を推測して埋めたり、再計算で補ったりしない。

判断：一回実行・中断STOPの固定契約を保持し、出力を残して停止する。
旧A/B/V1–R1、M06-F/R2/R3、H8その他の研究は変更しない。
次段案はCodexによる閾値の明示継承・引数受渡しテストを含む修正版SOURCEの準備と、
その新SOURCEに対するユーザーの別の1回実行許可。自動retryは行わない。

## 追記：修正版run02が完了

後続指示により修正版の実行と、意図しない技術中断時の修正・再実行が許可された。
SOURCE `d0b1da1b04c266a1f7abf8e43966ca6e289a8695`を固定し、同じ物理条件・seed recipe・N/α/β/ε・診断規則で
run02を1回実行した。今回は中断がなく、追加retryは0。

[完了報告](../rpe_fresh_iid_measurement_a_run02_20261103/REPORT.md)。
coefficient_atol=1e-12により保存preparation hash/218 componentを再現した。
1,928 trajectory・条件付きsignal・仮想±1 shot、8,284 event位置を32 workerで生成し、約4.318秒・exit0。
4軸・12比較すべてNO_DISCREPANCY_DETECTEDだが、半幅約.156/.165の有限標本診断であり
同等性・全law・全18round成功率は認定せず、科学statusはUNDETERMINEDのまま。

判断：これは計画した正常完了時の判定であり、技術中断として再実行する理由にはしない。
結果を公開し、追加samplingなしでSTOP。先行run01とB/V1–R1、他研究を保全する。
次段案はユーザー/GPTが必要な許容mean差と対象roundを定め、Codexが対応する検証計画を準備すること。
