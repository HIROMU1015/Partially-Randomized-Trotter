# R1.5 GPT handoff：保存値帰属から次の数学設計判断へ

2026-10-06 JST。**POSTHOC_ATTRIBUTION_DESIGN_INPUT**。
input R1 commit=`24bfeb84a4ce87b56985d174dd98d1d5e1702a2b`、
原result SHA256=`f726ad70cb2643533f0d037b518cde1b702724adb4e6571fea25e26e4bfdd61e`、
analysis script SHA256=`b26c3bdea5a42988c0534d45bd092d0cc28e50535e08dc75b4692ed601799911`。
**no science rerun、no synthesis rerun、no new candidates**。原result/marker/source/authorizationは不変。
R1はresource map complete awaiting GPT reviewのまま、R1.5は独立validationではない。

## 読む資料

1. [帰属報告とQ1–Q5](r1p5_saved_value_attribution_v1.md)。
2. [summary](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/attribution_summary_v1.json)：
   全front、二つのfocal条件のexact factorization、全controlled precision curves、質的回答。
3. [Pareto JSON](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/precision_envelope_pareto_v1.json)／
   [CSV](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/precision_envelope_pareto_v1.csv)。
4. [factorization](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/factorization_table_v1.json)、
   [angle/T-count表](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/synthesis_angle_table_v1.json)、
   [確率加重寄与](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/angle_usage_contributions_v1.json)。
5. [verification](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/verification_v1.json)、
   [evidence/provenance manifest](../../../artifacts/track_b_r1p5_saved_attribution/2026-10-06/evidence_manifest_v1.json)。
6. [固定R1 result報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/24bfeb84a4ce87b56985d174dd98d1d5e1702a2b/docs/tracks/algorithm_codesign/r1_one_shot_result_validation_20261006.md)、
   [固定R0 class theorem](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/672d6bc667eaa7b9ca4979b012f1530499d701b8/docs/tracks/algorithm_codesign/rte_reallocation_r0_independent_proof_v1.md)、
   [固定R0.5 novelty境界](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/61dd534567fda5c7348fdc688814089eb26a3561/docs/tracks/algorithm_codesign/rte_reallocation_r05_equivalence_novelty_audit_v1.md)。

R1.5の固定commit SHA/URLはCodexのpush後報告を使用する。
大きいexact JSONのGitHub表示制限時は報告・summary・CSVから入り、raw fileで正確な値を読む。

## Q1–Q5

| 問い | 保存値が支持する限定回答 |
|---|---|
| Q1：gainの帰属 | 複数要因。x=1/4,1e-3のA/ordinary G_T=0.9002996 shots比×0.8844632 native T比。x=1/8,1e-4はshots比1.0012158と悪化し、native T比0.9072597で補う。N内部のcausal percentageは分解しない |
| Q2：precision envelope | Aはx=1/8の1e-4、x=1/4の1e-3/1e-4で残る。A:1e-6は両xでPTSC_K0:1e-6にdominateされる。二符号は同座標controlで独立replicationではない |
| Q3：単一precision依存か | 一つの登録precisionだけには限定されないが、gainは固定pygridsynthのangle/precision列に依存して見える。ρ half-angleはx=1/4,1e-3で36 T対xの42 T、x=1/8,1e-4で46 T対52 T。優位は全precisionへ移送されない |
| Q4：RA-RTE設計への情報価値 | `SUPPORTS_RA_RTE_DESIGN`。representationとprecisionを共同に考える設計入力という限定診断。採択・新規性・science承認ではない |
| Q5：保存値で未決 | 未登録parameterのcost/bias/feasibility、fair resource objective、precision rule、強い対照・新規性、multi-block会計、実access/DF scaleにはGPT側の新しい数学設計が必要 |

primaryは2-qubit finite P₃ task、distinct-basis controlled、x={1/8,1/4}、σ=±1、登録三precision。
Gは共通finite-confidence sufficient-shot forecast。1Q分解には平均2.5 gates/shotの準備・読出しを含める。
x=1/4のA:1e-4残留はG_1Qによるもので、PTSC-K0:1e-4のG_T/G_CXは低い。
Pareto pointであることをmaterialな総資源改善やalgorithm winnerとは呼ばない。

## GPT側で判断する範囲

RA-RTEの設計自由度を検討する必要性・最小feasible class・数学contract・強い対照・論文着地点を判断する。
Codexは新しいeta/precision/angleを生成しておらず、resource最適化の実行もしていない。
known Euler/common-angle frameworkとrestricted normalization theoremの境界を保ち、
cost-awareであるという名称だけで新規性を確定しない。

R1.5はR1の科学分類を上書きしない。BF/BM/SP/BS/Track Aの結果とSTOPを保持した。
**mandatory STOP。R2、eta最適化、DF接続、追加synthesis/scienceは未認可。**
