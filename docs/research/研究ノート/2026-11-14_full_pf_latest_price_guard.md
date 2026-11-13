# 2026-11-14：最新Full PFとS1比較guard

A71f04のexact shortlistだけを旧coverage builderへ接続した。S2 16h/7・Y4 16h/3・M5 16hの39round/78軸、保存direct4round/8軸、他35roundはcatalog外。旧d7c09の.2/.4/.8の成果と計画は保全した。早期actual幅と定常要求幅を混同しない。

D3d39のSOURCE根拠を使い、PriceProvider/I.compare/I.selectへnative/lowering比較guardを追加。新RTE scenarioへ旧Full-H価格を混ぜず、PF/TruncatedはRTE非該当。固定Git sidecarとproducer SOURCEを照合し、未知を0価格にしない。新人工14＋回帰37＝51 PASS。Bの直接cost/selectは別取込みが必要で、CがB branchを修正済みとはしない。

[REPORT](../full_pf_latest_price_guard_20261114/REPORT.md) / [HANDOFF](../full_pf_latest_price_guard_20261114/HANDOFF.md)。実科学計算・保存価格算術・fit/予測/compile/N/G0。比較policyは未採択、最新幅の有限案は別承認。公開後mandatory STOP。
