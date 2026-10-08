# 2026-10-09 Track B：RA-D0 v4設計案の受領

GPT側から、構造保存変数・階層counts配分・丸め余裕付き内側LPを使うv4数値層の設計案が共有された。
[設計案全文](../../tracks/algorithm_codesign/ra_d0_v4_numerical_design_20261008.md)を元bytesで保存し、
[受領と次の統合監査](../../tracks/algorithm_codesign/ra_d0_v4_design_intake_20261009.md)に担当・GO/STOP・未完了項目を整理した。

設計の目的は元certificateを満たす候補を構成し、B2/B3の中心比較へ到達できるかを判断することである。
primaryは引き続き元B2 outer lowerに対するB3 certified upperのstrict改善。
inner infeasibleを元クラスのinfeasibleにせず、T0.2単点PASSをB3優位に拡張しない。
次の候補は一件の統合数学・数値監査であり、単点診断T0.3/T0.4の追加ではない。

今回は設計記録のみ。独立監査、実装、backend導入、test、登録最適化、新authorizationは行っていない。
GPT自己検算240例は未独立再現。v3/T0/T0.1/T0.2の既存分類と消費済みmarkerは保持する。
基点はT0.2 `d3a7cbb239487ddedf44699378f6c182c1fe5993`。
資料公開後STOP。新しい科学実行は別のsource review・contract・authorizationを要する。
