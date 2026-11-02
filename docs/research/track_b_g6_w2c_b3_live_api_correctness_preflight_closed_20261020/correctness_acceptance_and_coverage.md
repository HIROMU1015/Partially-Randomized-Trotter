# 判定・coverage候補

SOURCE/input/runtime/precision、許可trial、予定trial完了、Evidence完全性、exact rational/event invariant、A/B prefix、ZERO/reject/accepted分類、B preflight/angle/requery、F01、STOP/UNKNOWN、trace/receipt/caller、call/bytesを別項目で判定する。
技術失敗・UNKNOWN・欠測はtechnical STOP。exact event不一致はscientific mismatch STOPとして別分類する。

旧Full-N driverはcorrectnessでat least one accepted per cellを要求する。新候補は16 calls固定でacceptedがないcellがあればINCONCLUSIVE_COVERAGEとし、追加sampling/retryを実行しない。
全予定完了かつ不一致なし・必要coverageがあればCANDIDATE_CORRECTNESS_PASS。
いずれもユーザー/GPTによる基準採択待ち。未出現のZERO/coin reject等をPASSとはしない。
追加検証の必要性・scopeは別途採択・別grant・別実行承認が必要。人工coverageを実科学coverageへ転用しない。

**未採択・実科学未実行。mandatory STOP。次の担当：ユーザー・GPT。**
