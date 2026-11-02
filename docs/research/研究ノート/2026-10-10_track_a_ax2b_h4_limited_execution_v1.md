# 2026-10-10 Track A：H4限定一回実行のcoverage interface STOP

利用者の継続指示を直前の固定8 cell・3000秒/AS8GiB・別認可・一回実行・公開後STOPへ適用した。
[結果・監査](../track_a_ax2b_h4_limited_execution_v1.md)へ記録する。science source61091c2・manifest c56ebc4不変、別認可2f4f536を結果前にcommit/push。
worker/parentともH4_LIMITED_STOP。worker理由ACTUAL_COVERAGE_CHANGED、correctness0/8、reference/primitive/control/sampling/compile counter0。
原wall2.045328374952078秒、8 file/126,556 bytes。snapshot load1/native準備8はsource制御フローによる推論で独立counterなし。
保存JSONのlistとruntime schedule tupleを直接比較するinterface差を静的確認。925型差、canonical JSON digest一致。ただし実runの全actual boundsは未保存で、全値一致とは断定しない。
seal synthetic35件にはserialized scheduleのround-trip境界検査が欠けていた。原source/manifest/認可/STOPを修正せず、再実行しない。
次の候補は新versionでのserialization比較保全修正・合成round-trip検査・新固定。その実装と新科学起動は今回未実施。
一回grantは消費済み。N/Gnull、u未認定、UNDETERMINED、H6_NOT_AUTHORIZED、DRAFT_NOT_AUTHORIZATION、mandatory STOP。
