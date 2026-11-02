# Global CPU/wall・保存会計

単一result上限68,719,476,736B、log上限268,435,456B。公開n256はresult **15,502,271B**、log **43,190,850B**でreserved=observed=physical、unknown0。logの内訳にはtrace42,846,844B、control/stateが入り、trace自己課金・RAM/state二重課金・state更新trafficの混同を検査した。条件別traceのS/P1とshared finalizationを一度ずつglobalへ足す。

別freshローカルn256はresult15,491,499B、log43,194,710Bで同じ会計が成立。runtime/タイミング等の文書表現差もある人工値で、過去rawを書き換えて一致させていない。FAKE trusted DBとcaller inputsのharness領域はruntime result/logと別、観測文書slotはresultへ含む。

Global CPU=caller parent+全worker。public parent8.123531273s+worker25.152102000s=33.275633273s、wall20.433934483s。pair/IPC/observer、coarse/fineはouterの内訳・代替partitionであり再加算しない。条件windowとshared未配賦を分離。parentの個別I/O exclusiveやobserver単独費用、caller観測自身の保存費用はnull。失敗workerのgeneric generator/auditは元coarse partitionに保持し、型別内訳を既知にしない。unknownを0にしない。

S→P1固定順、OS cache、process起動、I/O/memory帯域、他jobの交絡があり、両条件fusedなのでfused単独効果は識別不能。各条件1反復からCPU母平均CIを得ない。量子confidenceを転用しない。実G7の保存量・CPU上界やspeedupへ外挿しない。Full-Nのper-trial保存適用は未確立で、集約等は別契約。

全caseは[call](global_call_accounting.json)、[bytes](global_byte_accounting.json)、[cost](global_cost_accounting.json)。不完全・失敗は比較CSVへ入れない。memory streamingは3,961,273B対42,846,844B traceの別process HWM測定で確認し、全trial ID集合を保持しない。計算資源のRSS上限を実行へ課したものではない。
