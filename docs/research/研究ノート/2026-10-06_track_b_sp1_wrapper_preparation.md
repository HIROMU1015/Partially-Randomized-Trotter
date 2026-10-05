# 2026-10-06 Track B：SP-1契約案と共通会計の技術準備

受領GPT reviewは `PROCEED_TO_WRAPPER_ACCUMULATION_AND_PLACEMENT_PILOT`。
研究全体を再設計せず、primitiveからwrapper累積／D-R placementへ進む方針を具体化する。
具体template／ε・α／shot・資源上限／materialityはreview原文で未固定なので、採択済みとせず契約案にする。

SP-0.5結果 `e57c1fdd28589422e9c973e34e53f6c725b921e9` を基点に
別branch/worktree `track-b-sp1-wrapper-preparation-20261006` を作った。今回原文一件だけを明示保存した。
A/root・旧result/source/authorization/marker・BF/BM closuresを保持する。

Sparse Probabilistic Synthesis v2の本文では、積momentのみならずwhole-circuit T費用／crossoverも既知と確認。
累積反転それ自体を新規性候補へ格上げしない。研究判断はGPT側へ返す。
同一Pauli反復をNONEで融合せず数えるbaselineの問題も記録し、交互Pauliの明示案をreview対象にした。

新しいexact Fraction kernelは小さい人工populationのみを扱う。
独立branch/outcome列挙でouter相関・control積weight・bias・shot cap・初期化を照合する17 testsがlocal passed。
analytic phase／inverse／wrong shared randomness／fusion controlsも含むが、actual wrapper adapterの合格ではない。
SP-0.5保存値の再採点、SP-1 domainのscience sweep、合成器呼出し、trajectory、Hamiltonian、NPZ、GPU、full suiteは0。

[契約案／GPT packet](../../tracks/algorithm_codesign/sp1_wrapper_preregistration_proposal_v1.md)、
[manifest](../../../artifacts/track_b_sp1_wrapper_preparation/2026-10-06/preparation_manifest_v1.json)。
common fusion・数値interval/channel adapter・source-bound review・別authorizationが未完了、RUN_READY=false。
必要資料をcommit/pushし、STOPしてGPTの具体契約案reviewへ戻す。science run・次stageは自動認可しない。
