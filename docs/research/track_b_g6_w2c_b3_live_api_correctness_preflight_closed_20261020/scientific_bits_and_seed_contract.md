# Seed／bit consumption契約

候補namespace PROPOSED_B3_CORRECTNESS_ONLY_20261020_V1は未採択文字列としてのみ保存する。
実行する人工namespaceは既存ARTIFICIAL_REAL_ADAPTER_V1。旧人工fixture再利用であり、実科学namespaceは消費しない。
SHA256(UTF8(seed)||uint64 big-endian counter)、必要blockを連結、high bitsを返し、未使用bitsをdiscardする静的仕様を確認した。
width/counter前後/value/discard/消費前後/failure positionを保持。正整数width、counter overflow、型不正はterminal STOP。

Modeはglobal trial IDに含めseedから除外。A/B、S/P1、phase/passは同じarm/axis prefix、arm・X/Yは分離する。
release/input/scope-claim/run/session/runtimeは別binding。対応trialでのみprefix比較し、IID_claim=false。
Fullではdegree+1/degree+2/degree+3等のdraw数となり、Canonicalもgroup modeに依存する。人工固定draw数は実規則の採択ではない。
実variable draws、conditional child law、reject、消費、IID条件はNOT_ESTABLISHED。

**未採択・実科学未実行。mandatory STOP。次の担当：ユーザー・GPT。**
