# B → A/C/D：全13round・K1・C−SのV2接続

2026-11-13。[REPORT](REPORT.md)／[S3判断表](S3_DECISIONS.md)。branchは`codex/rpe-decoder-statistics-20261113`を再利用。正本canonical ID=`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。本資料を含む公開commitを固定する。

## Aの受入

A169d3316の全13 packet、元S0/m1、補助bba150834のPHYSICAL_INPUTSを受入済み。新10round validatorは`rpe_diagnostic_intake.validate_completion_packet`。旧S0/m1 validatorは無変更。全nominalは64,60,56,52,47,43,38,34,30,25,21,17,12、十分N/boundはnull。m3/m4のobserved norm missing・SOURCE_GATE_LIMIT_NOT_OBSERVED、全allowanceのempirical資格を保持した。

`rpe_s3_intake.accept_a`はfull identity/law/physical/request/sourceと、物理packet→元resultのselector先のcopy一致を検査する。`prepare_statistics`は既存`mean_from_a/prepare`へcopyして渡すだけ。model signalやnormalization/shotの数値式をBで再評価しない。

[STATISTICS_INPUTS](metadata01/STATISTICS_INPUTS.json)はS3に渡せる配列形式だが、科学実行条件が未決定のため**実行packetではない**。`input_complete=true`、`physical_qualification_complete=false`、`scientific_execution_ready=false`、`S3_executed=false`。truth proxy採用・count caller・IID joint law・未解決loss・MSE precision/confidence・loss support/反復数は未採択。S3/実Gは起動しない。

## CのK1受入

`rpe_s3_intake.accept_k1(config,k0,k1)`がC83d219ccのnative JSONを既存`accept_structural_packet`へ接続する。source19b95d8、K0 PASS、preparation、全identity/law/q/R/axis、4セル、数値区間、元quantity `FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT`を保全。Bの内部構造envelope quantityは従来の`MEAN_STRUCTURAL_RZ_COUNT_NOT_COMPILED_C`で、元quantityも残す。

m0/X/Y=1743.6682557187337、m6/X/Y=1764659.275564426はcopyのみ。compiled_price_records=[]、mean_compiled_C=null、qualified_for_compiled_price=false。Cの条件付き数値enclosure/確率契約・kernel SOURCEは検証を新たに証明したものとして扱わない。Cの他11round計算はC担当。

## Dの残差接続

`compiled_residual.py`とtestはDaf6f27dからbyte一致で取込み。既存DのAPIをそのまま使う。

```python
from trottertracks import rpe_s3_intake as B_intake

# accepted C structural row + supplied D residual record (or None)
out = B_intake.residual_price(config, structural_row, residual_record,
                             cohort=cohort, source=record_source)
price_record = out['price_record']
```

残差recordはD既存HANDOFF_B形式：`quantity=FULL_WRAPPER_MEAN_C_MINUS_S`、`target=C_MINUS_S`、`mode`は1つ、full wrapper、m/axis/full identity/law/profile/physical ID、SOURCE/uncertainty、mean_D（未取得はnull）。finite cohortならrole/n_paths/expectationを明示する。これは本番資格を与えるAPIではなく、既存B.assessは未資格sumを拒否する。

値ありpaired directには、Dの同path取得証拠の参照を`paired_paths`へ付ける。1path/axisごと`index/event_digest/S_event_digest/C_event_digest`、`candidate_id/axis/compiler`、`structural_source/compiled_source`、`forced_conditional=false`を持たせる。SOURCEへの参照だけを検査し、Bで残差平均や分散を作らない。X/Yを別の標本数として二重計上せず、forced rare・役割をpoolしない。

`E[S]`と`E[C−S]`のlawが違う/未宣言ならpriceはnull。lawが一致した人工例でもUNQUALIFIED_MODEL。複数の残差model、raw compiled cluster加算、内部局所量のfull price化は拒否する。残差未取得ではNoneを渡し、mode/n_paths/残差0を捏造しない。今回の実compiled CとGはnullのまま。

Full PF保存136候補の73/0/63と旧56/17/63はB170047の受入結果をそのまま参照する。新RZ計数・receipt再実行・menu登録はない。

## 再現・次の作業

```sh
python -B scripts/prepare_rpe_s3_intake.py --out /tmp/rpe-s3-fresh
python -B tests/check_rpe_s3_intake.py
```

CLIは固定Git metadata受入だけ、科学実行optionはない。出力は新規directory専用。実signal/shot統計/N/C/Gは起動しない。[TEST_INTAKE](TEST_INTAKE.json)／[TEST_STATISTICS](TEST_STATISTICS.json)／[TEST_V2_REGRESSION](TEST_V2_REGRESSION.json)を参照。

**ユーザー**は[S3判断表](S3_DECISIONS.md)の条件・B科学SOURCE・finite scopeを判断/明示許可する。**B**はその許可範囲だけ統計実行。**A**はglobal物理誤差契約と必要証拠、**C/D**はkernel拡張・C−S/compiled law平均/不確かさを担当する。科学実行・監査改善R1〜R6・実Gへ自動進行せず、公開照合後**mandatory STOP**。
