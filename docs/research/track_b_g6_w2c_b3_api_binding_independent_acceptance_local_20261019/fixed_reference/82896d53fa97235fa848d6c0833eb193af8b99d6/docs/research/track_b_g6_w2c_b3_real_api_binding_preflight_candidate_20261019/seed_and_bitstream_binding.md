# Seed・bits・identity

固定SOURCE：[78c571005fe766c61b32dc867e006531099ff608](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/78c571005fe766c61b32dc867e006531099ff608)。直接基点 `d24cccbecdf0928fe977080dd7b28c591df0089e`。ローカル独立受入 `6bffb5b5819918698b522270ff19b255766d3072` は固定参照のみでmerge/rebaseしない。

今回の範囲は静的読取・AST・hash照合と人工技術検証。実科学入口閉鎖、科学的採択なし、mandatory STOP。サーバーCodex主担当一名による自己検証であり、別reviewerの独立受入ではない。

Bundle、SOURCE release、科学input、scope/claim、run、mode、arm/layer/axis、phase/pass/trial、seed domainを分離。global trial IDにはrun/modeとlocal trial IDを含める。mode/layer/phase/passは科学seed domainへ追加しない。同arm/axisのS/P1、A/B、baseline/instrumented、correctness/profile対応prefixをそろえ、armとX/Yは分離する。

科学候補namespace PROPOSED_B3_LIMITED_G7_PILOT_20261019_V1 は文字列参照のみ。実bitsは生成しない。新人工namespaceはARTIFICIAL_API_BINDING_V1/arm/axisのみ。SHA256(seed UTF8 || counter big-endian uint64)、各bits(160)で256-bit blockの上位160を返し下位96を捨てる。consumed_bitsとcounter、width、value_hex、discardをtrace化。独立したoffline実装が人工bitsだけを再照合する。

人工canonicalは2 draws/attempt、人工FullはZERO_R1、ZERO_C2、EVENT3 draws。これはG7の可変degree/conditional/rejection lawを模倣した同値証明ではない。実G7/F3 consumption equivalence、fresh IID、科学path coverageはpending。失敗prefixは正常比較から除外。retry0/resume false、source/hash/domain/consume mismatch拒否を試験した。
