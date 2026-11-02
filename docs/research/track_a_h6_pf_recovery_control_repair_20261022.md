# H6中断後の実行制御修正と未承認回復仕様

2026-10-22 JST。**起動・監督処理の修正、人工13件、一時user serviceでの実動作検証がPASS。科学計算・新grant・pushは0。** 修正SOURCEは `96c1c60cf`。専用branch `pf-common-h6-metadata-execution-20261022`でローカル保存する。

[前回の中断監査](track_a_h6_pf_cost_interruption_20261022.md)にある完成187候補、残22候補、追加内部callの必要量、旧grant/ledger/marker/STOPを変更していない。今回の依頼は技術修正として実施し、追加科学予算への承認とは扱っていない。H6全域と3系588候補の最終比較は引き続きUNRESOLVED。

## 起動・監督処理の変更

`scripts/run_detached_local_job.py`は自分の一時user serviceへ処理を分離する。`Restart=no`を明示し、同じlaunch rootとworker entryの再使用を禁止した。`KillMode=control-group`はこの一時service内だけに適用し、共有serviceや他ユーザーの設定を変更しない。元CPU affinityと明示されたBLAS環境を保持し、CPU/RAM/workerの上限は追加していない。

子processの起動失敗、非0終了、supervisorへのSIGTERM/SIGINTをSTOPへ保存する。signal受信時はこのsupervisorが新規作成したsessionだけへ終了を通知し、直接の子をwaitしてから記録する。5秒の終了待ち後も生存する自分の子groupだけを強制終了する。これは後始末の猶予であり、科学計算の時間上限ではない。他jobを名前で検索して停止する処理はない。

正常終了・異常終了のいずれでも最後のheartbeatを`running=false`へ更新する。`status`はPIDとkernel start timeを照合し、process不在やPID再使用を成功と扱わない。外部コマンドのexit0と科学stageの成功は分け、科学的成功には元の終了receiptが必要である。

SIGKILLやホスト消失時にはSTOPを書けない可能性が残る。その場合もprocess不在と成功receipt欠落を報告し、自動再開しない。今回の制御修正は元のCOST中断原因を特定したものではなく、実分子計算の完走もまだ検証していない。

## 未承認22候補回復仕様

`scripts/prepare_h6_pf_cost_recovery.py`は保存済みJSON・台帳・SHAを読み取り、新しい出力rootへ仕様を封印する。科学的な配列復元、Hamiltonian/stateの生成、回路作成、compile、signal/shot評価、費用順位の計算は行わない。

対象は元の未完22候補だけで、候補置換・q16・重複を拒否する。完成187pairのarchive、元SOURCE/input、旧SIGNAL/QUALIFY終了receipt、旧COST非返却台帳、旧STOPを含む6,504照合項目（重複する役割を除く6,488ファイル）を封印した。元の科学条件・元native手法・scalar/control/compiler政策を継承し、旧partial軸は価格として採用しない。

新取得案はnative evolution/occupation/low/high各22、signed S2 piece1,520、wrapper/build/compile/各作用44軸。消費済みを含む累積上限超過はpiece1,229、native各処理7、wrapper build4。既存の[予算差分](track_a_h6_pf_cost_interruption_20261022.md)と完全一致し、予算の再配分や消費の返却は行っていない。

仕様は `execution_authorized=false`、`additional_budget_approved=false`、`new_grants_created=0`、`execution_command=null`。これらを活性化した値へ変えると、この準備経路の検査は拒否する。COST成功前のACCOUNT開始も認めない。ACCOUNTの1,380 endpoint/20比較、累積台帳監査、H2/H4の参照結合、local commit後mandatory STOPまでの依存条件を記録した。

これは実行可能な承認済みpackではない。旧runのSIGNAL/QUALIFYを参照し、複数SOURCE由来の完成価格を接続する再取得driverと監査readerは、元のfull-run driverをそのままresumeして代用できない。追加予算の明示承認後に、新しい実行SOURCEと承認済みpackへ結合する必要がある。今回、その活性化や実分子driverの起動は行っていない。

## 検証と保全

| 検証 | 結果 |
|---|---|
| 新人工：契約の不正活性化・候補置換・予算改変・旧証拠改変の拒否 | 4 test methods PASS |
| 新人工：成功・非0終了・spawn失敗・request改変・PID再使用・SIGTERM/非干渉 | 6 test methods PASS |
| 既存：非返却予算・shell不使用・launch失敗保存 | 3 test methods PASS |
| 自分の一時user service：起動元終了後に3秒人工タスクが完了 | PASS、exit0/reap、heartbeat false |
| user serviceの実設定 | Restart=no、KillMode=control-group |
| 元入力・完成archive・control等の封印 | 6,504項目／6,488ファイル一致、実行無効 |

人工SIGTERM試験では、同時起動した別の人工childが停止されないことを確認した。service実動作試験は元のlauncher processの終了を再現するもので、実際のチャット切断やサーバー再起動を行う試験ではない。いずれもローカル検証であり、CI・外部レビュー・独立研究再現の証拠とは表現しない。

[検証summary](../../artifacts/resource_applicability/track_a_h6_pf_recovery_control_repair/2026-10-22/validation.json)、[実動作記録](../../artifacts/resource_applicability/track_a_h6_pf_recovery_control_repair/2026-10-22/detached_artificial_validation.json)、[実行無効の契約projection](../../artifacts/resource_applicability/track_a_h6_pf_recovery_control_repair/2026-10-22/recovery_contract_UNAPPROVED.json.gz)、[旧成果の保全](../../artifacts/resource_applicability/track_a_h6_pf_recovery_control_repair/2026-10-22/preservation_checks.json)。

制御コードは既存の科学数値コードを変更せずに追加した。旧SOURCE、raw結果、全grant/ledger/marker/STOPを保持し、保存科学結果・PR cohort・標本平均/分散/SEは不変。ローカルcommit後mandatory STOP。追加科学計算は予算承認待ち。
