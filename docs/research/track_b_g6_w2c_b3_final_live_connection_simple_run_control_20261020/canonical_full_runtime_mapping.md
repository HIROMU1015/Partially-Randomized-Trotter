# Canonical/Fullの経路

|方式|Layer A|Layer B sample/audit|B angle/requery|
|---|---|---|---|
|ordinary|G7.make_generator → CanonicalGenerator.sample|versioned LegacyEndpoint / canonical_snapshot / AuditParity / event_witness|event(group_index,raw_word,child) → enclose_atan、Canonical context|
|partial_return_tail|同上|同上|同上|
|closed_P3_tail|同上|同上|同上|
|full_return|G7.make_generator → FullReturnGenerator.sample|F3.prepare → GeneratorSession.sample / F3 preflight / AuditParity|angle_for_event(original Evidence) → event(word,child) → packet|

Canonical正入力をF3 facadeへ流さず、CANONICAL_PREFLIGHT_NOT_IMPLEMENTEDは保持する。
Canonical angleはAuditParity内で元のgroup/raw/childによるrequeryを1回だけ行い、元のcontext/EvidenceとSealを保存する。
Full Bのangle requeryはF3内部の1回をobserverで捕捉する。別の診断queryやsamplingを追加しない。
EVENT/ZERO_R/ZERO_C/IDENTITY/query/typed failureは混同しない。今回の固定正入力sampling計画はEVENT/ZERO2種だけ。
full angle enclosureの数学的正しさ・実Canonical held state・実F3 typed objectsの互換性は今回未検証。


**実科学未実行、実行未承認。mandatory STOP。次の担当：ユーザー・GPT。**
