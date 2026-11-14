# 並列化・高速化手法の再利用索引

既存手法の入口。性能対策の着手時に読み、対象・SOURCE・科学条件・実行経路が合う部分を再利用する。
詳細と証拠の正本はリンク先のcode・test・REPORTに置く。有用な手法や注意点が増えた時に短く追記する。
本書は新たな科学実行や稼働SOURCEの変更を許可するものではない。

## Full-H D7で整備した手法

実装の基準は `f8d648ab99c4c1eb84723dacc1452b8e3f525482`、checkpoint入力/controllerは
`887c3b9d04d15be96637b6634265424e7dd50339`。wrapper内部の高速化適用漏れ修正は
`dfab06ee0782e5946bdf303cef6ce78e4ae79c25`。元科学価格のproducer SOURCEは置き換えない。

| 手法・対象 | 流用する実装 | 適用条件・注意点 | 確認範囲・根拠 |
|---|---|---|---|
| 独立pathをprocess並列にする | [path_worker / child](../scripts/run_full_h_kappa2_p2_accelerated.py) | pathごとのseed・出力・provider状態を分離し、結果をindex順へ戻す。親receiptなどの依存関係を先に満たす | 人工barrierで別PIDの同時実行を確認。[検証記録](research/d_full_h_p2_acceleration_20261115/TRANSFER_TESTS.txt) |
| X/Y wrapperを個別に並列compileする | [ParallelAxesCompiler](../scripts/full_h_p2_acceleration.py) | 同じcompiler/profileで各軸一回。先行生成したYと実際のYのfingerprintを照合する。representative circuitはQPYで戻し、元のcompiled-vector gateを維持する | 人工同時実行・4-qubit toy＋ancillaの実Qiskit/QPY/gate PASS。D7 path1は取得済みだが、全jobの速度倍率比較は未測定。[REPORT](research/d_full_h_p2_acceleration_20261115/REPORT.md) |
| 同一controlled rotationの構築を再利用する | [memoized_controlled_rotations](../scripts/full_h_p2_acceleration.py) | support数とbinary64角度の完全なkeyでprototypeをcacheし、copyしてappendする。命令順・global/control phaseを保持する | toyのcanonical fingerprint・全unitary一致、構築回数削減を確認。toy primitiveの時間比をH4全体へ外挿しない。[TESTS](research/d_full_h_p2_acceleration_20261115/TESTS.txt) |
| 同じcanonical bytesを速くfingerprintする | [use_fast_fingerprint](../scripts/h46_b3_fast_fingerprint.py)、[fast_legacy_builds / resume_saved_path](../scripts/full_h_p2_acceleration.py) | nested context・module alias・復元を確認する。wrapper内部のmutation検査など、呼出し先が行う再計算もcontext内に入れる | checkpoint wrapperがcontext外だと旧writerへ戻る不具合を人工再現。dfab06ee0で修正し、旧writer禁止を含む9 test PASS。修正の実H4適用・性能は未確認。[回帰修正](research/d_full_h_p2_acceleration_20261115/RELIABILITY_FIX.md) |
| 保存済みnative結果から残検査を継続する | [resume_saved_path](../scripts/full_h_p2_acceleration.py)、[checkpoint controller](../scripts/continue_full_h_p2_checkpoint.py) | 元event/compile/QPY・SOURCE/physical/cohort/index/seed・gateを照合し、sampling/compileを再実行しない。固定契約・停止/継続条件を守る。controllerのPIDや入力を他jobへそのまま転用しない | 人工でsampling/compile 0と元event/fingerprint/phase一致を確認。実G36223 path0の最終検査は未完了。[固定入力・移管記録](research/d_full_h_p2_acceleration_20261115/REPORT.md) |

## 以前からある関連例

[PR-2 S2並列実行層](pr2_s2_parallel_execution.md)は、cell単位のspawn pool、phase barrier、
完了順に依存しない集約、persistent compiled-cost cache、段階別logを既存コードへ接続している。
toyでserial/parallel/cache再利用の一致を確認した記録がある。PR-2 V4/S2専用の科学条件・
実行契約を持つため、D7へそのまま実行しない。記録内のworker数・上限を新規作業へ恒常的に継承しない。

## 追記する内容

対象と削減した処理、code/commit、流用できる条件、検証記録、実測した入力規模・時間/資源、
未確認範囲や失敗例を数行で残す。既存entryの改善ならそのentryを更新する。
計測していない速度倍率を埋めたり、記録のためだけに新しい科学計算を行ったりしない。
