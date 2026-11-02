# C1：H6共通CA・RPE shots・条件付き総資源の限定実行結果（2026-10-26）

**4候補すべてが`NUMERICAL_PILOT_CA_PASS`となり、同じsではXDFの条件付き総RZが小さかった。** CDF／XDFの算定shotsは全13 round・X/Yで一致した。CDFの総RZはXDFを基準にs1で1.978607%、s4で1.962935%多い。

科学callは119／119、COMPLETE119、correctness FAIL／inflight 0、retry／resume 0。SOURCE・契約・既存数値条件は変更せず、保存値限定の評価を1回実施した。これは**資格Bの誤差、ideal IID Born、own-DF ground oracle、状態準備費用除外、serial nativeモデルに条件付けた数値pilot**である。厳密CA、実device成功、状態準備込み・無条件の総資源優位ではない。

## 固定条件・SOURCE・来歴

直鎖H6・1.00 Å・STO-3G、Nα=Nβ=3／sector400、修正codecを用いた保存fit最終CDF rank14／保存selected XDF rank14。通常number表現、元g補正κ・scalar・元frame順、全15因子ordered symmetric S2、s=1主対象／s=4対照、δ=0.1、m=0..12を保持した。PR splitは非適用。各DF ground oracleを仮定し、system状態準備費用を除外する。

CA=0.00159 Ha、DF reserve=CA/8=0.00019875 Ha、β_total=π/6、β_stat=π/12、候補ごとのfailure上限0.01、各round・X/Yのα_axis=0.01/26を維持。4候補同時や事後選択に対する追加保証は付けない。

固定P0はQiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation degree1、qubits initially zero=falseの保存metrics。literal/sharedは前段の同target・ordered gate／parameter／qubit／declared phaseのexact IR一致に基づき共用した。4候補の結果に両routeのalias来歴を保持し、8回の重複評価はしない。

- 科学SOURCE：`55b3d86817308a8d676221982aae13e1a6e3bed4`、tree `6b8cad5a683fdcef5fb4ba9d02d7f846dc0ea8a6`。
- [固定119-call契約](research_c_c1_h6_ca_resources_contract_unapproved_20261026.json)、scope `C1_H6_SAVED_CA_RPE_SERIAL_RESOURCES_V1`。
- 契約SHA256：`2a0af39e3a795eda74eb9ebac7b60b91a818f6f0c92135179a7b62e90a608504`。
- SOURCE manifest SHA256：`e9b6b4b106b3a8abd8d7973bae8df8c6320949e9045085802074faf100d64678`。
- Branch：`codex/research-c-c1-h6-ca-resources-execution-20261026`。
- Output：`artifacts/research_c_c1_h6_ca_resources/2026-10-26/run01`。

ユーザーが119-call契約の実行へ進むことをチャットで許可したため実行した。契約の歴史的な`unapproved`名／metadataは変更せず、承認証明・原文保存・新grantを追加していない。

## gateと数値CA評価

SOURCE54件・保存input47件・契約／manifest／依存環境のpreflight／postflightはPASS。保存ground／alias、rank14のDF予算と元guard、4組のexact-IR alias、native metrics／measurement scope、PF／合成／combined／phaseのidentityもPASSした。

既存`assess_round`、Hoeffding shots、`assess_fixed_round`を再利用し、全52 roundで整数N・非負の統計margin・位相十分条件がPASS。PFと合成誤差を各一度だけ渡したsolverのraw signal errorは、保存combined値と全行で一致した。DF誤差＋guardはHaで、既存`assess_energy`へ一度だけ加えた。

| 候補 | 保存DF norm＋guard（Ha） | 数値energy誤差評価（Ha） | 候補failure上限 | 判定 |
|---|---:|---:|---:|---|
| CDF s1 | 0.000108030585139252 | 0.001386347908377286 | 0.01 | NUMERICAL_PILOT_CA_PASS |
| CDF s4 | 0.000108030585139252 | 0.001386347908377286 | 0.01 | NUMERICAL_PILOT_CA_PASS |
| XDF s1 | 0.000103612307630943 | 0.001381929630868977 | 0.01 | NUMERICAL_PILOT_CA_PASS |
| XDF s4 | 0.000103612307630943 | 0.001381929630868977 | 0.01 | NUMERICAL_PILOT_CA_PASS |

すべてCA=0.00159 Ha以内。DF値は保存結果の再利用で、新規operator normを測った値ではない。energy評価も実測energy誤差ではなく、登録したWeyl＋最終RPE区間の数値十分条件である。

同じarmでs1/s4のenergy評価値が同じなのは、全round条件を満たしたうえで同じβ_totalと最終horizon M12を使う既存energy式による。PF／合成誤差の違いはshots算定へ入り、別々に評価される。βやDF reserveを結果に合わせて変更していない。

## 新たに算定したshots

以下は**採用済みHoeffding十分式による各軸の算定N**。同じsではCDF／XDFが全roundで同じNとなり、X/Yにも同じ登録Nを使う。H2/H4の保存shotsの転用ではない。実際の量子sampling／測定shotsは0。

| m | s1：CDF＝XDF、X＝Y | s4：CDF＝XDF、X＝Y |
|---:|---:|---:|
| 0 | 512 | 511 |
| 1 | 512 | 511 |
| 2 | 512 | 511 |
| 3 | 512 | 511 |
| 4 | 513 | 512 |
| 5 | 513 | 512 |
| 6 | 514 | 512 |
| 7 | 516 | 512 |
| 8 | 520 | 512 |
| 9 | 528 | 512 |
| 10 | 544 | 513 |
| 11 | 580 | 515 |
| 12 | 664 | 519 |
| **13round×X/Y合計** | **13,880** | **13,326** |

CDFの保存合成上界が小さいことは、今回の整数Nを減らす結果にはならなかった。これは当該固定条件内の算定結果であり、別条件でもshotsが一致するとの一般化はしない。overflow／underflow／整数容量によるUNRESOLVEDや、事後N調整は0。

## 条件付き総RZと副指標

既存`compose_serial_cost`でn=s·2^m回の保存kernel＋head＋axis tailを構成し、各axisのNで加重した。headを数え、tailの測定depth／sizeを重ねて加算していない。予定executions＝予定measurements＝上記shots合計となる。

| 候補 | 算定shots合計 | 条件付き総RZ count | 同s比較 |
|---|---:|---:|---|
| CDF s1（主） | 13,880 | 520,491,751,108 | XDFより1.978607%多い |
| XDF s1（主） | 13,880 | 510,393,076,228 | 登録比較で小さい |
| CDF s4（対照） | 13,326 | 1,761,144,891,569 | XDFより1.962935%多い |
| XDF s4（対照） | 13,326 | 1,727,240,281,697 | 登録比較で小さい |

割合の分母はXDF。同s比較2件の既存判定は`XDF_LOWER_CONDITIONAL_INTERROGATION_COST`。s間の最適化や候補追加はしていない。

| 候補 | 総RZ depth | 総CX count | 総CX depth | 総total depth | 総circuit size |
|---|---:|---:|---:|---:|---:|
| CDF s1 | 231,299,698,084 | 306,659,760,960 | 261,705,659,712 | 507,573,487,852 | 994,309,619,692 |
| XDF s1 | 226,940,270,116 | 301,300,464,192 | 257,626,195,008 | 498,694,652,908 | 968,772,970,540 |
| CDF s4 | 782,479,191,641 | 1,037,643,479,376 | 885,512,615,120 | 1,717,292,262,491 | 3,364,338,959,227 |
| XDF s4 | 767,996,983,033 | 1,019,642,229,424 | 871,842,492,976 | 1,687,651,106,555 | 3,278,460,815,659 |

depthはserial fenceを持つscheduled stageのshots加重和であり、巨大な一体回路をglobal compileした最小depthではない。追加のinter-kernel／Gaussian融合や、未観測のliteral/shared削減を仮定していない。

費用intervalがsingletonなのは、登録した決定論serialモデルで各回路の費用が固定されるためである。これは誤差enclosure、4候補のRPE成功同時保証、実hardwareの精度／費用保証とは異なる。全値は既存binary64集計の整数精度範囲内で、clipや代替aggregatorは0。

このH6条件ではCDFの条件付き総RZ削減を観測しなかった。CDF一般のNO-GO、他rank／分子条件での優劣、無条件の総資源優位へ拡張しない。H2/H4の既存結果は変更しない。

## call・資源・原本の保全

| 処理 | COMPLETE logical units |
|---|---:|
| saved bundle load | 1 |
| evidence attachment | 4 |
| round accuracy／shots＋整数margin | 52 |
| energy assessment | 4 |
| round serial cost（X/Y pairで1unit） | 52 |
| cost aggregation | 4 |
| cost comparison | 2 |
| **合計** | **119** |

全attemptとCOMPLETE terminalが一対一で対応し、FAIL／inflight0。前段3,335／上限6,535枠は流用せず、新規119-call契約を1回だけ実行した。新fit／optimizer／rank探索、DF／PF／合成誤差再計算、800基底state-action、新build／compile、量子samplingはすべて0。retry／resume0。

実際のaffinity32 CPU・空き資源／pressureを確認し、独立4候補を既存resource planで並列化した。CPU／RAM／worker／wallの恣意的上限は追加していない。runner wall1.4798秒、process CPU9.7761秒、peak RSS352,108 KiB。whole-job `/usr/bin/time`は1.82秒、peak RSS357,228 KiB、swaps0、exit0。既存eval venvをread-onlyで再利用し、task-local cacheのみを使った。共有設定・他ユーザーjobの操作は0。

postflightはSOURCE54件、保存input47件、契約／manifestと旧protected証拠872件のhash一致を確認。[raw19ファイル・SOURCE・call journal・progress・STOP](../../artifacts/research_c_c1_h6_ca_resources/2026-10-26/run01/)、[4候補総計CSV](../../artifacts/research_c_c1_h6_ca_resources/2026-10-26/candidate_totals.csv)、[104 axis別結果CSV](../../artifacts/research_c_c1_h6_ca_resources/2026-10-26/round_axis_results.csv)、[要約・postflight](../../artifacts/research_c_c1_h6_ca_resources/2026-10-26/)を保存した。旧FAIL・627-call履歴・3,335-call全基底結果・grant／ledger／marker／STOPは変更していない。

[技術準備時の人工／回帰102 PASS](research_c_c1_h6_ca_resources_preparation_20261026.md)のSOURCEを無変更使用した。今回、科学runを検査目的で再実行していない。CSV／要約作成は保存済みJSONの整形と保全確認であり、solver／energy／cost evaluatorやoperator処理の再起動ではない。

実行履歴：

```sh
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v \
  /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python \
  scripts/run_research_c_c1_h6_ca_resources.py \
  --contract docs/research/research_c_c1_h6_ca_resources_contract_unapproved_20261026.json \
  --execute --output artifacts/research_c_c1_h6_ca_resources/2026-10-26/run01
```

run01は消費済みであり、このコマンドを再起動しない。

## 資格と完了状態

4候補の共通CA・shots・条件付き総資源の**数値pilot評価は完了**した。合成誤差は完全800入力基底による資格Bだが、DF／PF／合成／reference／丸めを囲った資格Cではない。`rigorous_enclosure`、厳密`CA_achieved`、`guaranteed_actual_shots`、状態準備費用はnull、保証付き総資源優位はfalseを維持する。

本契約内の追加科学作業はない。**ユーザー／研究C**が、この固定H6結果を踏まえて研究のレビュー・次の方針を判断する。**Codex**は別途の具体的依頼と必要な独立の科学実行／詳細レビュー許可後にのみ対応する。結果を見たrank探索・追加fit・候補拡張・別条件や厳密enclosureの計算へ自動進行しない。

結果をcommit／pushしてremote SHA・SOURCE・保存input・成果物を照合した後、**mandatory STOP**。
