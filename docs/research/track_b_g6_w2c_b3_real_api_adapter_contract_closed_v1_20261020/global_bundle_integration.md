# Global Bundle閉鎖統合

既存Port Bundle bridgeを新namespaceへ最小派生し、contract/adapterのimportだけを差し替えた。scheduler・spawn worker・ControlledEndpoint・pair/F01・fused codec・CachedObserver・Global Admission・Storage V2・STOP/post-closeの本体は再利用。bridge_delta.patch/storage_consumer_delta.patchに差分を記録する。

人工fixture→new exact Codec/Result Adapter→global worker→Pair/F01→Storage V2→global receipt→post-close callerが実経路。evidence.adapted.codecに原document/digestとdriver投影が到達する。n256後半は従来どおりfull documentsを構成/検証/hashしてhash-onlyで保持する。未保存full documentを取得済みとは扱わない。

consumerは通常経路でproducer/bridge/contract/factory/fixture/authority/scienceをimportしない。固定storage検証のversioned schema/seedだけを変更し、人工SHA系列を別実装で再構成して、event/Evidence/angle/digestをworker PID・global offset・ENTER/COMPLETE/state・receipt/callerへjoinする。共通codec/result validatorを共有する同一作者の技術自己検証であり、別人の独立受入ではない。

single claim/coordinator、single global call/result/log予算、S→P1/6-phase、retry0、resume禁止、STOP後ENTER禁止、UNKNOWN非返却、trusted prefixを維持。manifest内のadapter名だけでは成功にしない。
