# Track B R0: finite-mean RTE reallocation technical review packet

2026-10-06 JST. Status: `R0_TECHNICAL_AUDIT_COMPLETE_RESEARCH_DECISION_PENDING_GPT`.

候補Aの有限平均保存、一般奇数次数での非負達成構成、限定class内の
normalization最適性は独立導出で確認した。固定free-word/Fraction検査も通過した。
一方、Taylor係数再配分・共通角という広い原理は既知であり、Aの具体family・
達成式・限定最適性の優先性と実装資源の有用性は未解決である。
本packetは研究GO、新規性成立、algorithm採択、R1実行を判定しない。

## 作業根拠・分離・読み順

利用者が提示した[新設計入力](inputs/track_b_all_evidence_algorithm_redesign_20261006_user_input.md)
§13の作業指示を実施した。[GPT本文](inputs/track_b_all_evidence_algorithm_redesign_20261006_gpt_review.txt)
もbyte-exactに保存し、[input identity](../../../artifacts/track_b_rte_reallocation_r0/2026-10-06/input_identity_v1.json)
へ元path、SHA256、bytes、改行を記録した。rootの未commit入力自体は変更しない。

- Branch: `track-b-rte-reallocation-r0-audit-20261006`。
- Worktree: `/home/abe/Project/prt-worktrees/track-b-rte-reallocation-r0-audit-20261006`。
- Base: `5a4ae817ec8d833bb2929c0c0a85e2d4d3064e7d`（BS-0.5公開commit）。
- Track A別系列: `4c23453c541700c6a41ba71fc5ec9323b53858d6`の
  `docs/research/track_a_post_pm2_claim_evidence_map.md`を参照identityのみ記録。
  Aの数値を本R0で再採点・転用・独立再現していない。

GPTへの読み順は本packet → [独立証明](rte_reallocation_r0_independent_proof_v1.md)
→ [claim単位の一次文献表](rte_reallocation_r0_prior_art_v1.md)
→ [条件付きR1提案](rte_reallocation_r1_proposal_for_review_v1.md)。
監査詳細は[manifest](../../../artifacts/track_b_rte_reallocation_r0/2026-10-06/audit_manifest_v1.json)。
入力文書は提案であり、既存科学source/authorizationを置き換えない。

## 主張ごとの判定

| 主張 | 判定 | 根拠とscope |
|---|---|---|
| A: 同じ有限P_(2d+1)の平均保存 | PROVED | 係数matching、独立involution抽出、terminal b_m=0。非可換性を仮定せず証明 |
| A: 任意奇数次数・x>0の非負達成式 | PROVED | 二つのprefix加重平均とtanh(x)の関係から全次数を証明。有限fixtureの外挿ではない |
| A: B≥sqrt(E_d²+O_d²)と達成 | PROVED | nonnegative adjacent-degree classのEuclidean下界。d≥1なら通常pairより厳密に小さい |
| A: K2閉形式、small-x、eta補間 | PROVED | 安定な有理式、差x^4/9+O(x^6)、線形mean・凸norm上界 |
| A: 負時間、odd ±i、complement角 | PROVED | 全体phaseと追加Qを保持するoperator恒等式。controlled位相をsystem global phaseとして捨てない |
| A: 既存DF controlled wrapperでodd event動作 | UNRESOLVED | 現行RTEEventはoddを拒否。既存テストの合格を転用しない |
| A: 任意LCU全体で最適 | COUNTEREXAMPLE | 単一involution x=1で別LCUのB²=17/18、限定classは65/18。提案の限定定理自体は反証しない |
| A: ordinary端点がWan/PRのpairing | KNOWN_EQUIVALENT | sign/cutoff/独立抽出規約を合わせた有限first mean。各branch/channel同一とはしない |
| A: 再配分・共通角という広い原理 | KNOWN_EQUIVALENT | PTSC/CTSにも直接の関連構成がある |
| A: 具体family/達成式/限定定理の新規性 | UNRESOLVED | focused本文照合で優先性未確定。汎用LCUへ書けることだけで棄却もしない |
| B: K2 return式とconditional probability | PROVED | identity吸収、ordered i≠j分布。s2=1/zero supportを分離 |
| B: identity/Taylor cancellation原理 | KNOWN_EQUIVALENT | Zhao–Yuan §4.2の直接の既知機構。有限bit sampler/実取得costは未検証 |
| C: P3 contractivityとmean誤差伝播 | PROVED | abs(x)≤sqrt(3)の範囲のみ。canonical weight second momentは減らさない |
| C: continuous precision KKT | PROVED / KNOWN_EQUIVALENT | 標準最適化の内点式。precision上下限・actual gate countは別問題 |
| A/B/C: 実資源改善、DF適用・新held-out | UNRESOLVED | 本R0はそれらを評価しない |

`PROVED`は明記した仮定内の数学、`KNOWN_EQUIVALENT`は表の原理/特殊点に限定する。
世界初、論文成立、実装性能の判定ではない。

## 実施した検証と限界

[事前fixture protocol](../../../artifacts/track_b_rte_reallocation_r0/2026-10-06/symbolic_fixture_protocol_v1.json)
を記録後、[独立checker](../../../scripts/tracks/algorithm_codesign/check_rte_reallocation_symbolic.py)
を一回実行した。`Fraction`による自由wordの隣接Q_i²=I縮約のみを許し、
commutationやPauli closureは与えていない。

- A: degree 1/3/5/7、x={1/8,1,8}、
  sigma=±1、eta={0,1/2,1}とx=0 endpoint、計80 mean fixture。
- B: 三つの確率列（通常、集中、単一support）×同じ三x、9 fixture。両signと条件確率をexact照合。
- 位相欠落、terminal漏れ、非可換branch順序変更、controlled位相欠落の4 mutationを拒否。
  任意LCU最適性という除外overclaimへのcounterexampleも保存した。
- 一回のlocal technical run: 約2.705秒、peak RSS 15,616 KiB、CPU上限15秒・AS256 MiB・wall30秒内。

[保存結果](../../../artifacts/track_b_rte_reallocation_r0/2026-10-06/exact_symbolic_checks_v1.json)
にfixtureごとのmean identity、checker SHA、Python identity、runtimeを保存した。
一般奇数次数の証明は別途独立導出であり、80 fixtureだけで証明したと扱わない。
branch順序のmutationは平均一致だけでは検知できないため、word identityを別検査した。
有限bit確率生成、行列/solver評価、controlled/native実装・合成guardは未検証。
local focused technical evidenceであり、immutable CI・外部再現ではない。

## 既存結論との境界

B-Fの限定negative、B-Mのcompact-BCH同値、FR、P-D/R3、BS-0.5の個別STOPは維持する。
SP-0.5/SP-1のraw結果・one-shot marker・authorization・contract、およびSP-1の23 critical pathを
hash照合し、変更しない。保存値の再分類や元run再実行は行わない。
M1/M2はTrack Aの既知/development/local evidenceであり、Bの新held-outにしない。
Aのruntime/cache・PM結果を新しいB成果として数えない。

今回のtechnical checker作成と記号検査は、科学実装やpilotとは区別して記録する。
科学run=0、sampling=0、synthesis/compile=0、行列/solver=0、分子/Hamiltonian/NPZ/GPU操作=0。
Track A/rootへのwrite=0、共通API変更=0、full test suite=0。

## 次のreviewへ返す内容と停止

Aの数学は今回の反証を通過した。ただしPTSC/CTSとの具体構成・情報access・取得手順の
差分を精査する必要があり、広い再配分原理を新規性にしない。
少数のR1を検討するなら、[一つの条件付き提案](rte_reallocation_r1_proposal_for_review_v1.md)
に示したtarget、call cap、guardと予算から、GPTが必要性・strong baseline・claimを決める。
未固定accuracy/threshold/IS policy等は別contractへ結果前固定する。現在は実行可能契約ではない。

`RUN_READY=false`、`science_execution_authorized=false`、次stage自動認可なし。
本packetと必要資料を担当branchへcommit/push後、mandatory STOPし、
研究方針・RQ・新規性・論文着地点・追加検証範囲の判断をGPTへ戻す。
