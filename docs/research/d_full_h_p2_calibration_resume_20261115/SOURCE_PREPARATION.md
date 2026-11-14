# D7 CALIBRATION再開：SOURCE・入力固定

旧SOURCE `6069605d50270111d1510478b08e11b8e5a18d86` と旧STOP `df0e11fd1` は保全する。今回のユーザー指示は、以下2点の限定SOURCE変更と、既採択D7の較正5 cell・最大18実compileの再開を許可している。修正確認後の再承認は不要。

## 限定変更

1. `parent_receipts` はG566の完了RESULTをlocator・内容hash・実行SOURCE・physical row・cohort/index・seedで照合する。paired分岐は既存 `h46_same_event.decode_event` とrequest dataclassを使い、保存event列から直接復元する。build前にlaw・準備hash・scalar・時間・順序・位相を検査し、normal samplerを呼ばない。
2. normal pathの取得経路を維持し、compile callbackで実際の個別compile recordとその内容hash・時間・RSSを捕捉する。較正の `PRICES.json` は各path/軸の直接価格、`C_INPUTS.json` は既存 `PRICE_CALIBRATION_RECORD_V1` の有限cohort平均、`IMPLEMENTATION_EVIDENCE.json` は両者をpath checksと実行SOURCEへ結ぶ。sidecarのpublication commitは結果公開後に束縛する。

P-1 caller、sampler、native、wrapper、compiler、lowering実装、A checkerは変更していない。H4の試運転は行わない。

## 人工検査

- [metadata-tests.txt](metadata-tests.txt)：11 PASS。全較正5 cellのX/Y証拠→元compile record/価格/receipt対応、公開locator結合、親SOURCE不整合、旧stage/domain/rank制約。
- [toy-tests.txt](toy-tests.txt)：2 PASS。既存toy/fake compilerを再利用。親2列をJSON往復後に直接連結、samplerとsample_eventsの呼出し0、order-2位相・scalar/順序一致、子X/Yのfake compile 2回。scalar/phase/seed破損はbuild前に拒否。
- 主agent自身によるlocal技術確認であり、独立レビュー・CI・実H4性能の証拠ではない。

## 固定入力と実行

[FROZEN_JOBS.json](FROZEN_JOBS.json) は `a43050c78:PROPOSALS.json` のD7 planを採択状態にしたものと、5つのcell jobを含む。q/R/T、physical ID、law、cohort名の `PROPOSED_V1` 部分も含め全seedを維持する。A checkerはrank3。HOLDOUTは全設計のmetadataとして残るが、今回のjobには含まない。

取得順序はG566→P2R566→Q8R566→G4528→G36223。一回のcaller呼出しで一つのcellを取得し、失敗時は次を投入しない。親receiptはG566完了後の生成物であり、P2R566のjobには事前固定した親locator/selectorを使い、取得したbytesのhashだけを補う。固定SOURCE commitを全cellで維持し、結果取得中にHEAD・SOURCEを変更しない。CPU/RAM/workerの独自上限は設けない。逐次投入はcell失敗時の即STOPと親子依存のためである。

`--execute-approved-job` を使い、各cellのconsoleと `/usr/bin/time -v` 記録も保存する。出力は新しい `run01/<cell>/`、privateは各jobに明記した専用領域。前回の重複照合を引き継いだ差分確認は [DELTA_PREFLIGHT.json](DELTA_PREFLIGHT.json)。受入可能な再利用価格0、予定18実compile。旧価格の再集計や換算はしていない。

## Cへの接続と終了境界

`bind_calibration_publication` は結果公開後、SOURCE付きRESULTから各recordを照合し、C `c628e9e8` の既存 `evidence` readerで公開実装証拠が読めることを確認する。これはD1による技術検査であり、C担当が較正値を受入済みであることやモデル資格を意味しない。canonical round専用guardは呼ばず、架空のround identityを付けない。

全較正の引渡し、または重大gate失敗でmandatory STOP。Cのfit/係数凍結、m5/m6 HOLDOUT、外挿、law E[C]、N/Gへは進まない。有限cohortのconfidence・rare寄与・sampler/ideal距離・長時間資格も未解決のまま。ロードマップ本体は編集しない。
