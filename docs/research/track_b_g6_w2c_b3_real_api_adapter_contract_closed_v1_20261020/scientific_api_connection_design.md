# 静的API接続設計

8 module/201 entryのSOURCE bytes・AST inventoryとstatic call sitesを保存した。実moduleはloadしない。

|方式|Layer A|Layer B|
|---|---|---|
|Ordinary|make_generator→CanonicalGenerator.sample|LegacyEndpoint相当のcanonical_snapshot/AuditParity/event_witness、event(group_index,raw_word,child)再照会、enclose_atan|
|Partial Return + tail|同CanonicalGenerator|同canonical監査|
|Closed P3 + tail|同CanonicalGenerator|同canonical監査|
|Full Return|make_generator→FullReturnGenerator.sample|F3.prepare→GeneratorSession.sample/angle_for_event|

Axisはconstructor引数ではなくtrial identity・seed/downstreamの軸。sigmaもG7 constructorへ追加しない。正入力canonicalに対するF3.prepareのCANONICAL_PREFLIGHT_NOT_IMPLEMENTEDを保持。IDENTITYはx=0の別分岐で固定正入力の代用にしない。

SOURCE上で確認：実名signature、None/packet境界、Evidence codec、sample/requery経路、failure STOP。静的に推定：document observation契約と将来wrapperの引渡し。実行待ち：実型・law・可変draw・actual evidence/requery互換。未実装：real loader/factory、canonical F3 facade、real authority。

新binding.pyは名前をdataとして保持する。result.pyはexact Documentだけを受け付け、foreign read callable/science objectを拒否する。人工backendと将来接続interfaceを区別する。
