# Canonical／Fullの静的接続

|対象|生成・sampling|B監査・angle/requery|
|---|---|---|
|ordinary / partial_return_tail / closed_P3_tail A|make_generator -> CanonicalGenerator.sample(bits)|Aはratioのみ|
|Canonical B|LegacyEndpoint(modules,arm,layer,angle,source_check,*,permit=None)|canonical_snapshot(g,_law)、AuditParity、event_witness、event(group_index,word,child)、enclose_atan|
|Full A|make_generator -> FullReturnGenerator.sample(bits)|Aはratioのみ|
|Full B|prepare(p,x,m,**precision_and_context) -> GeneratorSession.sample(bits=None,*,trial_id)|sealed preflight、AuditParity、angle_for_event(record)内部でevent(word,child)->packet再照会|

Canonical正入力をF3.prepareへ渡すとCANONICAL_PREFLIGHT_NOT_IMPLEMENTED。解除せずCanonical専用経路を保持する。
実引数はlive_api_binding_candidate.jsonのAST抽出一覧。F3 make_requestにはeta/rho/root_bits/probability_bits/angle_tau/angle_root_bits/angle_termsが必要。
driverのtau/K_phi/Jをangle_tau/angle_root_bits/angle_termsへ変換する。

Fullはorder draw→raw labels→reduce比較。非reducedはpacketなしZERO_R、reducedはpacket1回→acceptance draw、rejectはZERO_C、acceptedのみchild draw。
原rawをreducedへ再割当てしない。eventはword/raw_word/child/ratio/phase/coefficient/proposal/weight/provider_calls/helper/circuit_time_wordを保持。
Canonicalはgroup indexとraw_wordが必須で、reduced wordだけでは再照会できない。FullのB angleはoriginal Evidenceを渡し、内容fingerprint・context照合とpacket再照会を原APIが行う。
実runtime型と実法則の互換性はNOT_ESTABLISHED。

**未採択・実科学未実行。mandatory STOP。次の担当：ユーザー・GPT。**
