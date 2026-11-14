# Full PF／Truncated p7：保存4件・新規32件へのStage 0改訂

**現在の状態は[保存4件binding完了の追補](ADMISSION_UPDATE.md)を参照。** Cのp7受入・新版CELL_PLAN・共通指示の取得は解消し、SOURCE22ccf3972へ接続済み。policy/許容差は未採択で科学実行STOP。以下は初版0501096bd時点の記録を保全したもの。

2026-11-14、Codex D。**技術改訂と人工検証を完了。Stage 1は開始不可、mandatory STOP。**
新規選択はCALIBRATION 20 wrapper＋HOLDOUT 12 wrapperへ変更した。
ただしCのp7保存receipt受入と新版CELL_PLANは、確認した公開branchでは未発行である。
現時点で正式受入済みはFull PFの保存2件だけで、p7保存2件は受入待ちとして予約除外する。
32件設計をC受入済みの実行契約へ昇格させていない。実H4 prepare/sampling/build/compile、
新価格取得、保存値の再集計、実fit、holdout判定、N/G、Pauli B0再実行はすべて0。

## SOURCEと旧成果の保全

- 改訂実装SOURCE：`2b9983590a642e67fda794521baac7e5eaf0496f`。
- 既存branch `codex/d-price-records-c-bridge-20261114`を再利用。開始時HEADは旧Stage 0公開
  `c848bbefa30c23a9ee69170482fb02accae31601`、作業treeはcleanだった。
- 旧実装`1fd47847b3300395f1b0c0de1868ed7b83905497`、さらに前の`62ae1bd5`、旧34件計画、
  [旧Stage 0資料](../d_pf_price_stage_split_20261114/REPORT.md)とSTOPを履歴・原pathに保持した。
- CのFull PF受入はbundle `3368805575d459d948e3afe7b1e54b4814f64129`、公開
  `2fa88065318f0f2ab13937ef610828200b35874c`を固定blobとして再利用する。
  Cのbranch/worktree・guard `c628e9e8`・price modelを変更・mergeしていない。
- C branch `codex/m5-m4-receipt-admission-20261114`のremoteは、開始時・途中・証拠記録時とも
  `2fa88065318f0f2ab13937ef610828200b35874c`だった。
  未公開の受入を成功とも不成立とも推定しない。詳細は[SOURCE_STATUS](SOURCE_STATUS.json)。

## CELL_PLANとcoverage

**C新版CELL_PLANのSOURCEは未取得。** [CELL_PLAN_PROPOSED](CELL_PLAN_PROPOSED.json)は
上記D改訂SOURCEに固定した技術提案であり、C発行の正本ではない。
元C CELL_PLAN（3368805575）の36行から、p7 M5/m4 X/Yのactionだけを
`NEW_COMPILE`→`SAVED_RECEIPT_FOR_C_ACCEPTANCE`へ変更した。
configuration、candidate/physical ID、T/q/実幅、role、features、model_groupを含む他fieldは同一。
signed PF weights・保持block・fragment順・scalar quantity・compiler/control/wrapperは、
同じconfigurationおよび不変の科学関数に束縛される。
[SOURCE_AND_PLAN_CHECKS](SOURCE_AND_PLAN_CHECKS.json)に2行の差分と19関数のAST同一確認を保存した。
[FINAL_CHECKS](FINAL_CHECKS.json)に旧成果保全・36行coverage・科学出力先未作成の最終確認を保存した。

| route | 保存較正wrapper | 新規CALIBRATION | 新規HOLDOUT | 全価格 |
|---|---:|---:|---:|---:|
| Full PF p10 | 2（C受入済み） | 10 | 6 | 18 |
| Truncated p7 B0I | 2（C受入待ち） | 10 | 6 | 18 |
| 合計 | 4（うち2待ち） | 20 | 12 | 36 |

両route共通で、S2は幅16h/7・q=(7,14,28)、Y4は16h/3・q=(3,6,12)、
M5は16h・q=(1,2,4)、roundはm=(4,5,6)。h=0.1、H4/1.00 Å/STO-3G/保存tol-only DF10、P=0。
S2/Y4のm4/m5とM5のm5を新規較正、m6をholdoutとする。
X/Yは別々の測定付きfull wrapper・別modelである。

- Full M5/m4/q1：physical `0cd1527c80f7b9841a83`、Cの保存各85960 RZをそのまま参照する。
- p7 M5/m4/q1：physical `13d1cd4869951cdf1e2c`。旧原表155529e20に元candidate/cohort/trial locatorが存在する。
  指示書の各60950 RZ・scalar 2.293146549384713をDで新生成・再会計・受入していない。
  正式record/sidecar/evidenceはCの固定成果を待つ。

[PREFLIGHT_METADATA](PREFLIGHT_METADATA.json)は2本の既存callerのmetadata出力であり、
新しいtrialや実C_INPUTSではない。実m5価格は未取得なので、実データの12モデルは存在しない。

## 必要最小限の実装差分

既存`select`、`execute_stage`、`run_cells`を継続使用する。新backend/candidate生成器は追加していない。

1. `plan_rows`は旧C表から2 actionだけを変更する。将来の`P7_CELL_PLAN`固定blobが提案表と
   全fieldで一致しなければSTOP。テンプレートに同一cellが重複する場合も拒否する。
2. `NEW_WRAPPERS`とp7 callerのscopeを10/6・合計16へ更新した。保存2 cell・計4 wrapperは新jobへ入らない。
3. `saved_calibration`をroute別に共通化した。Cの元入力recordをcopyし、physical/configuration ID、
   axis、SOURCE、kind、profile、sidecar/cohort、固定evidence commitとの対応を照合する。
   Fullの旧参照先は不変。p7参照は`P7_ADMISSION=None`のままで、正式bindingを偽装しない。
4. `admission_gate`はC新版表とp7受入SOURCEが未束縛なら、両routeともoutput/childより前にSTOPする。
   現在のmetadata preflightは成功しても、このscience gateは通らない。
5. `duplicate_preflight`をscience childより前に追加した。保存表・catalog・元trialの正確なpath・
   当routeの既存stage receipt・関連する自ユーザーの実行中caller/jobを読取照合する。
   追加保存証拠または未解決のjob重複は再利用候補としてSTOPし、自動compile・自動受入しない。

`checked_trial/full_trial`、native/provider checks、wrapper、scalar処理、price_bindings、fit_inputs、
`run_cells`、凍結model gate、既存bindは不変。
CALIBRATIONの各groupはm4/m5の2点、M5は両routeとも保存m4＋新規m5となる。
方式2×公式3×軸2の12 groupを維持し、m6・別formula/幅/route/axisを混合しない。
HOLDOUTは引き続き公開済みCALIBRATIONとCの凍結modelを要求し、既存出力を上書きしない。

## 重複照合と運用指示の取得状況

[DUPLICATE_PREFLIGHT](DUPLICATE_PREFLIGHT.json)では、新規対象32 wrapperについて4 stageすべて
`NO_DUPLICATE_IN_CHECKED_SOURCES`だった。これは確認した保存原表155529e20、catalog e7dca5f7、
既知元trial path、当routeのstage結果、関連running jobの範囲に限る。
全サーバー・未提示private archive・未知の別callerを網羅した証明ではない。
H/state/snapshot・split/保持block/scalar・PF時間・ID・native/wrapper/lowering/compiler・X/Y・SOURCE/cohortの
詳細一致は、固定C context、科学SOURCE gate、Cの正式receipt受入により維持し、単なる価格一致で共有しない。
追加証拠が見つかれば、そのcellを取得せずCへ戻す。

指定の`research_project_instructions_unified_2026-11-14.md`は、既知の共通ディレクトリ・対象treeでは
取得できず、保存先をユーザーへ照会した。今回は指示本文§7に再掲された§10.5の照合条件を実装した。
未読の共通文書全体への適合を認定していない。実行前に正本を取得し、残る差分を確認する必要がある。
重複snapshotは将来の実行直前にも更新する。今回のsnapshotを永久な実行許可として使わない。

## 人工・回帰検証

[TESTS](TESTS.txt)を参照。作者自身の技術確認であり、独立レビューではない。

- Stage parent 17件PASS：36行/2 action差分、20＋12、両routeの保存除外、全group m4/m5、
  C未公開時の無science STOP、保存bindingの不整合、重複cell、追加保存trial/表・running job、
  stage namespace、二重実行/bind拒否、凍結model/holdout gate、m6混入・group混合拒否。
- Stage childの人工回路3件PASS：既存trialとscalar、並列worker、XY fingerprint、失敗保存・retryなし。
- Full PF parent 6件PASS（内部の人工回路2件もPASS）。
- Truncated parent 3件PASS（内部の人工回路5件もPASS）。
- C guard 14件PASS：架空record/不整合の受理・拒否。初回はPYTHONPATH未指定によるimport失敗だけで、
  正規`PYTHONPATH=src:tests`で再実施した。科学実行のretryではない。

p7の正例は**人工の受入interface fixture**で、保存p7実価格を使ったfit/受入ではない。
人工C_INPUTSでは両route各6 groupにm4/m5が入り、M5には旧SOURCEの保存m4と別SOURCEの新規m5が入る。
架空値でのみ凍結modelの形を検証した。実係数・予測・holdout判定は作成していない。

## Stage 1の判定と次の担当

**開始不可。** 必要条件は[HANDOFF](HANDOFF.md)／[NEXT_EXECUTION](NEXT_EXECUTION.md)に整理した。

- **Codex C**：p7保存receiptを正式受入し、新版CELL_PLAN・原cohort/SOURCE保持のrecord・sidecar・proof・guard結果を公開する。
- **ユーザー**：共通運用指示の保存先を提示し、比較policy・holdout許容差/分母を正式採択する。
- **Codex D**：上記取得後に固定blobを接続・照合し、既存最大34件承認内の32件について開始可能状態を報告する。
  今回は技術成果の公開でSTOPし、科学実行へ進まない。Cの実fit/freeze/holdout判定は別途科学実行承認が必要である。

旧source/input/raw/価格/grant/ledger/marker/KEEP/STOPを変更せず、共有設定・他担当process/fileに干渉していない。
