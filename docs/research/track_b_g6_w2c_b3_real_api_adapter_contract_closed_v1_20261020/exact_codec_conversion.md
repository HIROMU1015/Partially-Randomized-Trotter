# Exact Codecと署名対象

codec.pyは独立したexact encoder/strict decoderを持つ。F3 SOURCE仕様のhex `$int/$rat/$tuple/$map`をcanonical JSONにし、SHA256(document UTF8)を原digestとして保持する。Fractionは既約/正denominator、boolはintと区別し、tuple/listは区別する。signed integer/rationalを保持。mixed mapはpacked keyのjson.dumps(sort_keys=True)順、通常string mapはcanonical JSONのkey順を使用。

driver projectionはhex rat/int、tuple/listをarrayへ、map keyをstrへ写す。これは単独ではlosslessでない。新sidecarは原F3 document/digest/source identity、driver document/digest、original fingerprintをすべて保存する。復元時は両投影を再構成して厳密比較する。元の型と値のround-tripは原sidecarを通した範囲で成立する。

str-key collision、duplicate JSON/map key、unknown/reserved `$` tag、不正rational、非canonical hex、missing/extra field、不正digest/source、unsupported typeはtyped rejection。旧F3が読み得る非canonical文書の全域まで受け入れたとは主張しない。将来decoder拡張も別versionで扱う。

人工literal golden bytes、nested map/tuple、large signed integers、exact Fraction、wrong digestと新negative/mutationsで確認した。実G7/F3からのlive Evidence互換性はNOT_ESTABLISHED。保存科学値の再解析はしていない。

DriverDocumentも原driver bytes/digestを厳密decode/再encodeし、AuditParity Sealを別sidecarで保持する。driver正規化以前のtuple/key型は復元不能として明示する。
