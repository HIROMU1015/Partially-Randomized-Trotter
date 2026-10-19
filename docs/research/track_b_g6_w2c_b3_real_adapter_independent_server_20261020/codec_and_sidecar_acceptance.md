# Codecとsidecar

独立exact algebraとliteral golden値を使用し、候補validatorをoracleへimportせず49 check PASS。
None、bool、signed/large int（4096 bit）、既約Fraction、tuple/list、nested map、heterogeneous hashable key、unicodeの受理subsetで原document/digest/sourceとdriver投影が一致。
Bool/int、非既約・不正denominator、signed/noncanonical hex、duplicate/equal key、reserved/unknown tag、missing/extra sidecar、wrong digest/source、unsupported float、投影key collisionを拒否した。

Driver投影はtuple/listをarray、keyをstrへ写すので単独ではlosslessではない。
F3原document/digestを含むsidecarを保持した場合に限り元の型・値を復元できる。
旧F3が寛容にdecodeできる非canonical文書全域との互換性は主張しない。
DriverDocumentより前のtuple/key型も復元不能。実live G7/F3 documentとの互換性は未確認。
詳細：[独立Codec結果](independent_codec_verification.json)、[入力](probe_inputs.json)、[subject出力](probe_outputs.json)。
