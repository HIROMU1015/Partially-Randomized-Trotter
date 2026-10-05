# 2026-10-06 Track B：BM-0.5後GPT reviewを設計仕様へ反映

利用者が `track_b_post_bm05_research_redesign_20261005` と添付GPT reviewを返却した。
研究方針の採否はGPT側、Codexは文献対応・仕様化・保存情報の所在確認を担当する。

review判断を採録し、B-Fの現限定仮説とB-M現adapterのnew-method路線を閉じる。
これは研究B全体／multirate／確率的合成のno-goではない。BM applicationは保留、旧BM-1は実行しない。
新候補は固定DF/PF/finite-RTE full Hadamard wrapperのrandomization placementと合成・測定costの設計。

[一つの統合review仕様案](../../tracks/algorithm_codesign/synthesis_placement_design_review_20261006.md)に
四mask、ancilla込みchannel、signed phase、有限bias、weight二次モーメント、range、共通Bernstein shot規則、
actual T/T†と全wrapper cost、最大四template／二catalogue案、未固定budgetを収録した。
既知PAI／PR Appendix E §3／TE-PAI／resource-optimal ISとの関係を本文で確認する。
同じscore／splitであるだけで停止する旧基準を新RQへ一般化しない。

基点はBM-0.5 commit `d55de044b8e956ba6292209a94bb081014dfdae2`。
独立branch/worktreeは `track-b-synthesis-placement-design-20261006`。
原文二件だけをbytes保持でcopyし、参照元path・SHA256・root別branch identityを
[設計JSON](../../../artifacts/track_b_synthesis_placement_design/2026-10-06/design_contract_v1.json)に記録した。
Aは `4c23453c541700c6a41ba71fc5ec9323b53858d6` のgit text/JSONへ固定して静的確認した。
primitive/event angleを生成する能力はあるが、確認した保存schema／cost payloadにgate別angle列はない。
全repoの不存在とは言わず、SQLite／runtime／circuitを開かず、aggregate RZをT／Γ²へ換算しない。

共有science実装・旧source/result/artifact/status・A worktree・root入力は変更していない。
NPZ resolve/stat/hash/load、Hamiltonian生成、science signal、trajectory sampling、
circuit build/compile/synthesis、GPU、testsはいずれも0。
今回確認するのは原文copyの一致、設計JSONの構文、追加リンク、公開path境界である。
原文二件のbytes／SHA256、追加local／固定commitリンク18件、参照git blob identity9件を照合した。
新追記より前の文書本文が保持されることと、RUN_READY／実行認可がfalseであることも確認した。

合成器、具体input、完全resource budget、数値guard、materialityは未固定。
この公開は実装／pilot実行のauthorizationではない。必要資料のcommit/push後、STOPして利用者/GPT reviewへ戻す。
