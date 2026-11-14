# p7保存receiptと新版セル計画の固定bundle

固定旧原表`155529e20`のp7 B0I M5/m4 `13d1cd4869951cdf1e2c`、X/Y各60,950 RZをcopyで受け入れるための技術証拠。scalarは保存metadataによるc＋tail identity＝2.293146549384713。新しい価格の平均・fit・予測・compileは行わない。

`PRICES.json`・`SIDECARS_UNBOUND.json`はD62の元record emitterとCの既存受入helperを再利用。原cohort EXPLORATION、carrier/trial/軸別cache receiptのSOURCE/hashを保持する。原trialはbyte copy、原CSVの該当2行は文字列のまま保存。各軸の保存QPYについてscalar/measurement suffixとfingerprintを確認し、同じゲート数を根拠に別回路を採用していない。QPYのdeserialize/hashは保存回路の技術検査で、実回路の生成・compile・価格再計数ではない。

固定bundleの`IMPLEMENTATION_EVIDENCE.json`へsidecarを結合した後、無変更のC628 guardを通す。UNBOUND版のplaceholderは消費不可。正式な比較policy・観測値基準2%は未採択。

`CELL_PLAN_V2.json`は旧C `2fa88065:…/CELL_PLAN.json`の2行のactionだけを保存再利用へ変更したもの。旧版は変更しない。全36行、保存4＋新規較正20＋holdout12。候補・幅・q・model_group・roleは不変。

PR-DFの52軸coverageはC `3ae2e605`の固定metadataを再利用し、同条件P1のSOURCE参照を追加した。C1の別domainは別catalog。S/Cの値・係数・新しい平均・Gを生成していない。最終REPORT/HANDOFFで資格と不足事項を整理する。
