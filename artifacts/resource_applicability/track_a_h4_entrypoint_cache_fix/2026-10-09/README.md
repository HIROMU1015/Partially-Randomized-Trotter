# H4 entrypoint disk cache修正audit

入口：[run03結果・原因・修正・次binding](../../../../docs/research/track_a_h4_entrypoint_cache_fix_20261009.md)。
新SOURCE `cd162e9305143d81c28908b9732f8eef12cd6b89`、branch `track-a-h4-entrypoint-cache-fix-20261009`。

- [46 SOURCE closure・旧new blob/hash](source_freeze_v1.json)
- [private disk cache停止profile v2](library_cache_profile_v2.json)
- [限定8件・6 plugin群metadata一致](limited_tests_v1.json)
- [STOP/carry22と未認可の次binding](preparation_binding_v1.json)
- [独立技術review](independent_entrypoint_cache_review_v1.json)

run03は一度実行してSTOP・全14退出。sourcefixはmetadata-only検査、追加compile/本体なし。
現17GiB/74805とcarry22/12956511264B/6004.111340102032s保持。次21GiB/74806はproposalのみ。
次のplan/auth/reviewは未binding、global flagsfalse/allowed[]/未seal/commandnull。scientific successは未検証。
