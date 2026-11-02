# H4 canonical encoding speedup

[変更と検証](../../../../docs/research/track_a_h4_encoding_speedup_20261010.md)、[SOURCE closure80](source_freeze_v1.json)、[限定test計画](test_plan_v1.json)、[12件結果・人工timing](limited_tests_v1.json)、[新SOURCE準備binding](preparation_binding_v1.json)、[変更箇所のみの独立review](independent_delta_review_v1.json)、[commit対象](commit_inventory_v1.json)。

SOURCE `3beb3870369eeb8ef76b4ead965c886eba7d16e9`。実装はbounded numeric matrix canonical text reuse。旧/new bytes・digest一致、1803-instruction人工encoding約2.29倍、production全体は未測定。run10は継続し変更しない。新SOURCEは未launch/未seal、将来のSOURCE seed/binding再結合が必要。

独立担当の変更箇所だけのreviewはPASS、具体的P1/P2なし。追加tests/compile/入力読込/本番操作0。cacheはdriverの単一producer用でありthread間共有を想定しない。
