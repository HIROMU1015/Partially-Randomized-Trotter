# Track B R0.5 GPT handoff — 2026-10-06

**Scoped classification: `METHOD_DELTA_CANDIDATE`; gate: `CONDITIONAL-R1`。**
R1の必要性をGPTへ返す。R1そのもの、synthesis、compile、科学実行は認可せず実施していない。

## 固定基点と読み順

Base: `672d6bc667eaa7b9ca4979b012f1530499d701b8`（R0 result/review）。
独立branch: `track-b-rte-reallocation-r05-novelty-audit-20261006`。
Worktree: `/home/abe/Project/prt-worktrees/track-b-rte-reallocation-r05-novelty-audit-20261006`。
Containing publication commitのfull SHAはcommit/push後の利用者への報告に従う。

1. [Equivalence/novelty audit](rte_reallocation_r05_equivalence_novelty_audit_v1.md)：Z1–Z5/C1–C5とclaim-level分類。
2. [Primary-source locator table](rte_reallocation_r05_primary_source_locator_table_v1.md)：target・情報・atom・取得・sampling。
3. [Symbolic comparison readout](rte_reallocation_r05_symbolic_comparison_readout_v1.md)：固定9条件とsupport。
4. [Machine comparison](../../../artifacts/track_b_rte_reallocation_r05/2026-10-06/comparison_v1.json)、
   [exact artifact](../../../artifacts/track_b_rte_reallocation_r05/2026-10-06/exact_symbolic_comparison_v1.json)、
   [provenance manifest](../../../artifacts/track_b_rte_reallocation_r05/2026-10-06/audit_manifest_v1.json)。

## Reviewで重く見る点

- Aのrestricted theorem/closed formは、読んだPTSC/CTSの直接parameter choice/corollaryとは一致しない。
  ただしEuler、identity redistribution、common angleという枠組みは既知で、広いnovelty claimは不可。
- 同じI0のordinaryとゼロ次PTSCからlogical ensemble/norm差が残る。
  AのI0生成はO(m)で、ゼロ次PTSCに対する漸近取得cost改善を主張しない。
- CTS free式E-1+sqrt(1+O²)は、一般I0で実行可能なCTSの値ではない。
  明示したI1 per-word rephasing/paddingなら同式を実現できるが、literal collected値とも別。
- 固定Pauli fixtureではcollected CTSのnormalizationがAより小さい。
  AをPauli域の最良手法とせず、I0/DF想定とPauli collection域を分ける。
- Markov layeringはfull collectionの負担を減らすが、Pauli情報や同じμの保証まで取り除かない。
- Aのodd atomはCTSのsigned Pauli/単一Pauli rotationに直接一致しない。
  native実装可能性/controlled-Q oracleを無料とせず、同じ前提で比較する必要がある。

## GPTへ戻す判断

1. 指定corpus内で残る狭いclass theoremとlogical generator差が、理論noteまたはR1の情報価値を持つか。
   世界初やcitation network全体の優先性は本監査では確立していない。
2. R1を検討するなら、[既存の条件付き提案](rte_reallocation_r1_proposal_for_review_v1.md)を今回の
   oracle/CTS所見に照らし、必要scope・strong baseline・claimと結果前contractを採用するか。
   この旧proposalの78 call案やthreshold未固定事項を自動で承認しない。
3. Pauli collectionが可能な域と一般involution域で、どの問いを論文対象へ残すか。

実装前blocking/pendingはnative rotation/controlled access、odd builder、有限precision/bias、
IS/range/zero-cost処理、取得/native cost、primary materiality、独立source/authorization。
研究判断はGPT側。Codexはこれらを自動採用・追加実行しない。

## 保持した証拠と停止

旧R0の証明・比較artifact・checker、SP/BF/BM/BSのresult/marker/authorization、Track Aは変更しない。
R0.5の固定technical checkerのみ一回。9 norm条件、18 sign比較、mean/phase/support照合PASS。
raw input一資料のみidentity付きsnapshot。旧科学結果の再採点・再分類・再実行なし。
新科学/sampling/synthesis/compile/分子/DF/NPZ/GPU=0、共通API変更0。

`RUN_READY=false`、`science_execution_authorized=false`、`R1_execution_authorized=false`。
必要資料を担当branchへcommit/pushし、remote SHA照合後mandatory STOP。
