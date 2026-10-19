# Canonical／Full runtime境界

|対象|construction/sample|B監査・angle/query|
|---|---|---|
|Canonical3方式 A|make_generator(p,x,m,arm,**precision) / CanonicalGenerator.sample(bits)|ratio保持|
|Canonical3方式 B|同じG7生成、LegacyEndpointと同じowned observer構造|canonical_snapshot(g,_law)、AuditParity、event_witness、event(group_index,raw_word,child)、enclose_atan|
|Full A|make_generator / FullReturnGenerator.sample(bits)|raw→packet iff reduced→acceptance→child|
|Full B|prepare(p,x,m,**precision_and_context) / GeneratorSession.sample(bits,*,trial_id)|preflight、原Evidence object→angle_for_event→event(word,child)→packet|

Canonical正入力のF3 prepareはCANONICAL_PREFLIGHT_NOT_IMPLEMENTEDのまま。Full facadeへ統一しない。
precisionはroot_bits/probability_bits/eta/rho、Full prepareにはsigma/angle_tau/angle_root_bits/angle_termsも渡す。
Angle contextはCanonical=input/policy/preflightとdriver event identity、Full=F3 input/policy/source/preflight/sigma＋原record.digest＋parent/child。
group/raw/childとFull packetのcall観測で再照会を照合し、追加drawや追加requeryを行わない。角度intervalの人工整合はatan保証ではない。
live import・runtime型・数学的妥当性はNOT_ESTABLISHED。

**LIVE_RELEASE_CANDIDATE_UNAUTHORIZED。実科学未実行。mandatory STOP。次の担当：ユーザー・GPT。**
