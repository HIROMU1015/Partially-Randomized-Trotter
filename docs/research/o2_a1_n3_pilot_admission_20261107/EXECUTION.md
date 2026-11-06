# 752-shot pilot：CPU判定の改善とattempt02

2026-11-07 JST。ユーザーの「すすめられるところまで進めてよい、承認は取らなくてよい」を、直前の資源STOPからの技術修正と同じ752-shot予備検証の一回再投入に適用する。旧STOPを解除・削除しない。科学的なrank/range失敗の救済、測定追加、別条件・本検証への拡大には使わない。

保存linear H4/1.00 Å/STO-3G/legacy DF12、8 system qubits/全4電子sector、L_D3/m0/q1/δ.02/r1/K2/atol0、X/Y、N1 actual Q/selected full policy、N2保証ground、認証済みexact Fraction engine、3-control/全5正層を維持する。
元[採択済み実行仕様](../o2_a1_n3_pilot_science_20261107/EXECUTION.md)のtrain160/probe80/独立評価512、32domain/64子stream、path64word、範囲D_CV≤4、配分floor4/108 largest remainder、Qiskit1.3.0/rz,sx,x,cx/opt2/seed17、探索的診断/精度保証なしを全て維持する。

## 今回の技術変更

旧attempt01はSOURCE a139d7d、結果8d9241cに保存され、trial・word・bit・価格は全て0。したがって消費された測定からの選択・再学習はない。campaign `A1-N3-PILOT-H4-m0-q1-20261106-v1`、seed **20261108**と全stream順を維持し、新attempt02のfresh output `artifacts/o2_a1_new_target/2026-11-07/n3_pilot_752_run02`、manager `n3_pilot_752_manager_run02`、独立cacheを使う。旧結果・cacheは読み込まない。科学callは一回の実試行・合計752以内、scientific gate失敗時はSTOP。

- 新`a1_n3_resource_admission.py`でown affinity内の各CPUの/proc/statを1秒間隔で二窓観測。8区分のtickを用い、guest項を二重加算しない。idleのみを空きとし、iowaitは空きへ加算しない。各CPUのidle比率の和をfloorした利用可能数を求め、二窓の小さい方をworker数にする。人工worker上限なし。
- before/after/delta、affinity、空きコア相当の厳密分数、判定worker数と時刻を耐久保存。CPU変動や負差分を黙って補正しない。
- 空き0なら最大30分、15秒ごとに再観測する有限待機。これは実入力・stream初期化・測定より前の待機であり、科学retryではない。timeout/STOPで原本保全して終了する。モデル読込・RNGより先に資源待機へ移した。
- 既存activationへadmissionとattempt02/output/branch/docを接続。PilotDriverへattempt番号のmetadataのみ渡す。kernel/測定/統計式/fit/配分/RZを変更しない。
- 既存qualified consume-once cron→outer→manager→親、Restart=no、own nice+10 worker、STOP→drain→reconcile/bit先行保存を再利用。共有設定・他processへ操作しない。CPUの将来空きを永久保証するものではない。

科学入力前に人工テスト、SOURCE/profile/入力固定、GitHubのremote SHA・本文/blob/tree照合を完了する。一回起動後は完了または最初の科学STOPで保全・公開してmandatory STOP。無条件の自動retryは実装しない。
