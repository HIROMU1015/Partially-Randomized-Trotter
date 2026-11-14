# 2026-11-14 D：Pauli LD1/m0 B0の一attemptとSTOP

ユーザー指定SOURCE62ae1bd5のcallerを、元C c628 canonical d905d219/RMSE:62a9…、
T.1/q1/R3/K2、184 Pauli、normal1 path/index0/seed14788159701244282917、XY最大2 compileで一度起動した。
B新Z1＋T1/Q3 configurationへは置換していない。metadata preflightと親8・内部子5testはPASS。

実行用新worktreeがsparse checkout patternを継承し、tracked
references/h46_m5_tol_pilot_inputs_v1.jsonを展開していなかった。
私の準備でそのimport依存を見落とし、native_contextのrestore importでFileNotFoundError。
実入力restore/prepare/sample/event/build/compileは0、child1/parent2で固定callerがSTOP。
SOURCE・guard・seed・scalarは変更せず、retryや欠測fileの展開後の再起動もしていない。

raw RESULT/SOURCE/traceback/time、preflight/環境/技術test、private job.jsonのhashを保全。
wall6.72秒・maxRSS604088 KiBは起動失敗の観測で、回路性能ではない。
価格C/metrics/fingerprint/sidecarと実H4 s_R/scalar/operator gateは未取得。
この失敗をPauli/DF物理接続の否定や科学的NO-GOにせず、mean/N/GもUNKNOWNを維持する。

[REPORT](../d_pauli_ld1_m0_pilot_results_20261114/REPORT.md)／[次回条件](../d_pauli_ld1_m0_pilot_results_20261114/NEXT_STEPS.md)。
完全tracked依存の展開と数値restore直前のimport-only確認、新出力/SOURCE・一attemptは別途判断。
結果commit/push・remote照合後mandatory STOP。
