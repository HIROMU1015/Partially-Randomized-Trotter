# 静的API inventory

固定SOURCE：[78c571005fe766c61b32dc867e006531099ff608](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/78c571005fe766c61b32dc867e006531099ff608)。直接基点 `d24cccbecdf0928fe977080dd7b28c591df0089e`。ローカル独立受入 `6bffb5b5819918698b522270ff19b255766d3072` は固定参照のみでmerge/rebaseしない。

今回の範囲は静的読取・AST・hash照合と人工技術検証。実科学入口閉鎖、科学的採択なし、mandatory STOP。サーバーCodex主担当一名による自己検証であり、別reviewerの独立受入ではない。

8 moduleの201 class/function/nested entryをASTだけで採取した。scientific_api_inventory.jsonにpath、SHA256、行、signature、docstring、return/raise構文、個別SOURCE segment SHAを保存。返却構文は動的schema推論ではない。意味論・型・副作用・失敗境界は同JSONのsemanticsとapi_binding_manifest_candidate.jsonに記録した。

| Arm | A | B |
|---|---|---|
|ordinary|make_generator → CanonicalGenerator.sample|既存LegacyEndpointのcanonical_snapshot/AuditParity/event_witness、event(group_index,raw_word,child)再照会、enclose_atan|
|partial_return_tail|同上・partial groups＋ordinary tail|同上|
|closed_P3_tail|同上・closed P3 groups＋ordinary tail|同上|
|full_return|FullReturnGenerator.sample|F3 prepare → GeneratorSession.sample/angle_for_event|

F3.prepareは正の入力のcanonical3方式をCANONICAL_PREFLIGHT_NOT_IMPLEMENTEDで拒否する。存在しないcanonical F3 facadeを仮定しない。sigmaはprovider/angle contextでありG7 constructorの引数ではない。X/Yは実行identity/seed domainで、G7の測定axis引数ではない。

G7はevent dictまたはFullのNoneを返す。Full Noneはraw nonreduced/coin rejectをtrace/packet境界で区別し、例外をzeroへ変換しない。word/raw_word/circuit_time_wordはtuple[int]、ratio/coefficient/proposal/weightはFraction、child/phaseはint。weight×proposal=coefficient。finite rational approximationと理想のirrational normalizerを区別する。

F3はEvidence(document,digest)のEVENT/ZERO/IDENTITYとcontext/traceを返す。F3Failureはcategory/reason/stage/input/detailsとSTOPを持ちsession terminal。固定正入力ではIDENTITY経路を実行しない。F3 hex $rat/$tuple codecとdriver exact codecは別documentで、bytes互換を仮定しない。

Fullはorder→raw labels→reduced判定→local packet→acceptance→accepted時child。packetはReturnKernel.digital_parent→parent→word_series/convolution/_mass、root/envelope/child law/weight。sampleは同packet再照会を避け、B angleはevent→packetを再照会してfingerprintを照合する。setupのfirst_passage/returnsを保持し、成長するpacket cacheはない。共有状態とbits消費、session失敗、副作用のsource file readを区別する。providerはIR記述でありnative実行は範囲外。
