# 2026-10-07：Track B RA-D0 source review v2

利用者の修正指示に従い、`0ddf67756516e08f85fed1b987459a5e862676b7`から独立branch
`track-b-ra-d0-source-review-v2-20261007`を作り、登録table最適化前の実行契約を実装した。
[source report](../../tracks/algorithm_codesign/ra_d0_source_review_v2_20261007.md)と
[GPT handoff](../../tracks/algorithm_codesign/ra_d0_gpt_handoff_20261007.md)を現在の入口とする。

B0_savedを包含の始点から分離し、B1/B2に固定τ membershipを加えた。
B2の数値classを含むouter lowerとB3 dyadic upperだけをprimary比較にする。
Cartesian budget recipeは採用せず、最大12の完全resource vectorsから同vector capsを組む。
P1/P2を独立batchとして全点freeze後にだけB3を呼ぶ。anchor-firstとcoverage-only LOCALを固定した。
main/auxiliary/resource/source/authorization guardとfailure STOPも実装した。

旧35＋追加45＝80 focused tests PASS。旧R1/v1 artifacts・sign controls・Track A・旧STOPは保持した。
registered最適化、minimum/budget値/witness、新synthesis/science/NPZ/GPUは0。
authorizationは作成していない。`READY_FOR_SEPARATE_RA_D0_ONE_SHOT_REVIEW`でGPT最終reviewへ戻す。
過去のv1未確定記録を上書きせず、本改訂のmachine manifestへ証拠を追加する。
runtime内完了、registered certificate取得、研究新規性は保証しない。mandatory STOP。
