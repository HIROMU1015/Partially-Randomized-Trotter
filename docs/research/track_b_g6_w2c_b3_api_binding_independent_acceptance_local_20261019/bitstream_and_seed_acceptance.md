# Seed・消費trace・trial identity

作業基点 `6bffb5b5819918698b522270ff19b255766d3072`。API候補公開HEAD `82896d53fa97235fa848d6c0833eb193af8b99d6`、人工測定SOURCE `78c571005fe766c61b32dc867e006531099ff608`。科学SOURCE `76ca3c4089b201d55da0fede08e1f77733ac38a5` はbytes/ASTだけを読む。これはローカル技術受入であり、科学的採択・本番実行許可ではない。

固定科学参照P5_general_order：p=(1/5,3/10,1/2)、x=5/7、m=5、sigma=+1。値は契約参照として保持し、科学列を生成していない。

人工domainは `ARTIFICIAL_API_BINDING_V1/<arm>/<axis>`。SHA256(domain UTF8 || counter uint64 big endian)の上位160 bit、下位96 discard。counter/width/value/discard/totalをexactに再現する。canonicalは2 draws、Full ZERO_R1/ZERO_C2/EVENT3 draws。これはG7のdegree依存・可変消費lawと同じものではない。

|identity要素|global trial ID|人工/将来科学seed domain|
|---|---|---|
|run|含む|含めない|
|mode S/P1|含む|含めない|
|arm|含む|含む|
|layer A/B|含む|含めない|
|axis X/Y|含む|含む|
|phase correctness/profile|含む|含めない|
|pass baseline/instrumented|含む|含めない|
|trial index|含む|counter系列の対応位置。文字列domainには含めない|

同arm/axisのS/P1、A/B、phase/passは対応prefix一致。armとaxisは8 domainへ分離。run/modeを含むglobal IDは衝突しない。local IDは固定6 fieldでtrial indexを連続比較し、duplicateを拒否する。

独立checkerはdraw数だけでなくSHA値・counter順序・eventへの結合を比較する。boolをintとして受け入れず、width/counter/total/child等の改変を拒否。profile hash-only文書も人工再構成値と照合。新規40 component negatives、保存sidecar mutations、公開22 mutationsを別記する。

実科学namespaceは固定文書中の文字列参照のみ。実G7/F3の消費同値、IID、実path coverage、scientific outcomeは全て `NOT_ESTABLISHED`。人工一致をその証明に昇格しない。
