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
| 同じcanonical bytesを速くfingerprintする | [use_fast_fingerprint](../scripts/h46_b3_fast_fingerprint.py)、[fast_legacy_builds / resume_saved_path](../scripts/full_h_p2_acceleration.py) | nested context・module alias・復元を確認する。wrapper内部のmutation検査など、呼出し先が行う再計算もcontext内に入れる | checkpoint wrapperがcontext外だと旧writerへ戻る不具合を人工再現。dfab06ee0で修正し、旧writer禁止を含む9 test PASS。SOURCE8cで実G36223の元fingerprint照合と最終検査完了を確認。全job速度倍率は未測定。[回帰修正](research/d_full_h_p2_acceleration_20261115/RELIABILITY_FIX.md) |
| 保存済みnative結果から残検査を継続する | [resume_saved_path](../scripts/full_h_p2_acceleration.py)、[checkpoint controller](../scripts/continue_full_h_p2_checkpoint.py) | 元event/compile/QPY・SOURCE/physical/cohort/index/seed・gateを照合し、sampling/compileを再実行しない。固定契約・停止/継続条件を守る。controllerのPIDや入力を他jobへそのまま転用しない | 人工でsampling/compile 0と元event/fingerprint/phase一致を確認。実G36223 path0の最終検査はSOURCE8cでsampling/compile0で完了。[固定入力・移管記録](research/d_full_h_p2_acceleration_20261115/REPORT.md) |

完了済みpathを再処理しない継続は [complete_full_h_p2_checkpoint.py](../scripts/complete_full_h_p2_checkpoint.py) を参照する。
SOURCE `0dc1eeaee3d9d6465f18edc33d4e17ed312bd6f4` は保存native path0と完了path1を組み合わせ、
sampler/compilerを禁止して残検査だけを行う。11人工test PASS。段階別PROGRESS.jsonlと120秒ごとの
worker内STACKS.txtを記録する。STACKSの `Timeout` は定期dumpの表示であり、設定された実行打切りではない。
ただし0dcの実workerは2回目dump付近で異常終了し原因未特定。11人工test PASSだけで長時間実行の
安定性を認めず、現段階ではこの継続経路を実対象で完了確認済みとして流用しない。子終了code/signalの
保存と、timer発火を含む観測自体の検証が必要。科学retryはしていない。
[固定入力と移管報告](research/d_full_h_p2_completion_20261115/REPORT.md)に適用条件を残す。

0dcの監視異常は同じPython 3.11.0rc1＋人工FastWriter反復で再現した：定期frame dumpありは
SIGSEGV、なしは完了。旧実workerのsignalは記録欠落で不明。修正はworker内の非同期dumpを除去し、
親processで段階・資源・終了code/signalを記録する `observe_child`。計算workerへthreadを追加せず、
fast fingerprintのsingle-thread条件を保持する。旧入力と18 compileを再利用し、科学条件は変えない。
[再現証拠・修正・適用範囲](research/d_full_h_p2_completion_20261115/RECOVERY.md)。
SOURCE8cce54064で実G36223の残検査が正常終了し、wall21:39.15・peakRSS9,530,444 KiB。結果f98cb9638の公開後4軸bindingもPASS。旧0dcの不安定経路とは区別し、全18軸のC引渡し後mandatory STOP。全jobの速度倍率は未測定。

## 以前からある関連例

D7独立holdoutの入口は [run_full_h_d7_holdout.py](../scripts/run_full_h_d7_holdout.py)。既存path/X/Y並列・
controlled cache・fast writerをそのまま使い、任意callbackで実経路のstageと子終了code/signalを記録する。
親observerがpath別logを読み、計算workerに非同期dump/threadを追加しない。
[SOURCE準備・検証](research/d_full_h_d7_holdout_20261115/SOURCE_PREPARATION.md)。実対象はm5 wall22:05.61・m6 wall1:26:15で完了。native状態vector検査に逐次処理が残る。

追加の公開高速化は [publish_full_h_d7_holdout.py](../scripts/publish_full_h_d7_holdout.py) の完了cell先行処理、
導入済みpigzによる並列gzip、2 cellのprocess並列公開照合。元bytes/hashと復号/part順を保持し、4 exact比較の式は変更しない。
稼働中科学SOURCEの変更・再起動なしで後処理を重ねる。[適用条件・人工3件](research/d_full_h_d7_holdout_20261115/PUBLICATION_ACCELERATION.md)。
実測はm5 raw約586MBの圧縮2.096秒/復号・hash等を含むarchive2.970秒、m6 raw約1.17GBは4.024秒/5.738秒。公開cell照合は別PIDで7.761/13.239秒。全体速度倍率は未測定であり、人工bytesの時間を科学処理へ外挿しない。

[PR-2 S2並列実行層](pr2_s2_parallel_execution.md)は、cell単位のspawn pool、phase barrier、
完了順に依存しない集約、persistent compiled-cost cache、段階別logを既存コードへ接続している。
toyでserial/parallel/cache再利用の一致を確認した記録がある。PR-2 V4/S2専用の科学条件・
実行契約を持つため、D7へそのまま実行しない。記録内のworker数・上限を新規作業へ恒常的に継承しない。

## 追記する内容

対象と削減した処理、code/commit、流用できる条件、検証記録、実測した入力規模・時間/資源、
未確認範囲や失敗例を数行で残す。既存entryの改善ならそのentryを更新する。
計測していない速度倍率を埋めたり、記録のためだけに新しい科学計算を行ったりしない。

## 小さいnormal pathへ既存gateを再利用する際の注意

m0 q1/R9の固定normal pathでは、X/Y並列・既存cache/fast writerを再利用して7.84秒で2 compileしたが、中央rotationがすべて1-qubitでZZ置換実績が0だった。元7数値gateはPASSでも既存のcoverage条件 `replaced_controlled_zz > 0` が不成立となり、単価はnullで保留した。gateを満たすための再抽選・強制事象を行わず元SOURCE/証拠を保全する。性能策のreuseと取得受入を区別する。[m0結果・STOP](research/d_full_h_m0_reference_20261115/REPORT.md)。
