# 2026-10-09 Track B RA-D0 v4 exact backend pilot

基点`beb82427d202f479cc2ba954480d73a51941e322`から独立branch
`track-b-ra-d0-v4-exact-backend-pilot-20261009`で、利用者が認可したisolated synthetic-only backend pilotを準備した。
SoPlex 7.0.0、private GMP/Boostを使用し、static library buildは完了。
harness compile中にRSS guardが発火し、`V4_EXACT_BACKEND_RESOURCE_INCONCLUSIVE`として停止した。

監視コードはsupervisorのtreeに含まれるcompiler treeを二重計上していた。
発火indicatorをtrue peak RSSと呼ばず、実memory超過、solver/certificate failure、RA-RTEの研究価値を結論しない。
cap後はretry禁止を適用し、修正・再build・solver呼出しを行わなかった。
synthetic/registered LP=0、build/solver retry=0、production/science/synthesis=0。
旧135＋数学監査18 protected pathsは不変、過去分類とSTOPを保持した。

[技術report](../../tracks/algorithm_codesign/ra_d0_v4_exact_backend_pilot_20261009.md)と
[GPT handoff](../../tracks/algorithm_codesign/ra_d0_v4_exact_backend_gpt_handoff_20261009.md)をcommit・pushする。
既存のprotected正本文書を上書きせず、新規証拠はこのdated noteとhandoffから案内する。
backend採用は保留。追加pilotの必要性・範囲はGPT側で判断する。
資料公開後mandatory STOP。新authorization、registered solve、production sourceへ進まない。
