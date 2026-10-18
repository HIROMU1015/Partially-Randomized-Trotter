# Full対Ordinary・Partial・Closed P3のbreak-even条件

[入口](README.md) / [古典の項の定義](classical_total_cost_comparison.md) / [量子law・IR](quantum_event_cost_comparison.md) / [取得計画](missing_cost_evidence_and_execution_plan.md)。本資料は既存費用式の通常の代数的整理であり、新budget・非自明定理・具体入力の評価ではない。

## 1. Nの受渡しと比較前提

a∈{O,P,3}、F=Full。Nは[W2b][W2B]の各axis認証十分全trial数を使う。ceil前の値Tを用いてNを連続化したり、実moment・未知normalizer・digital受理率でNを下げたりしない。同じSOURCE/input/K/H/η/ρ/外向きroot・angle/provider error・target/state/readout・ε/confidenceの比較であることが前提である。

W2bより同一固定条件でN_F≤N_O。P/3とはmoment・range・ceilに依存する。N_F=N_aもあり、strict trial改善は仮定しない。元のG7 16 axes（2 target×4方式×X/Y）のfamilywise保証を別集合へ拡張しない。[P_U反証][PU]を再調査せず、理想normalizerの順序とNや費用の順序を同一視しない。

## 2. CL：古典費用の必要十分条件

同じ計算モデル・単位・実装層で、S_aはsetup/budget/preflight/cold取得の実呼出回数を含み、μ_aは全trial平均online費用、Ω_aはμに含めない保存/最終処理費用とする。Δ_a=S_F−S_a+Ω_F−Ω_aと定義すると

\[
\boxed{C_{cl,F}\le C_{cl,a}
\iff\Delta_a+N_F\mu_F-N_a\mu_a\le0.}\tag{CL1}
\]

従って、trial削減で回収できる費用として書けば

\[
\boxed{(N_a-N_F)\mu_a
\ge N_F(\mu_F-\mu_a)+\Delta_a.}\tag{CL2}
\]

左辺は削減trial分、右辺は残るN_F trialの追加処理とsetup/保存差である。N_F>0で

\[
\boxed{\mu_F\le\frac{N_a\mu_a-\Delta_a}{N_F}.}\tag{CL3}
\]

μ_a>0ならr=N_F/N_a、t=μ_F/μ_a、δ=Δ_a/(N_a μ_a)としてrt+δ≤1。0除算を避け、μ_a=0はCL1で扱う。右辺が負、μ_F≥0なら成立不能である。これは未知量を取得した後の判定式であり、数値を仮定して順位を確定していない。

- strict改善はCL1の`<0`、tieは`=0`、対照strict優位は`>0`。
- N_F=N_aならN_F(μ_F−μ_a)+Δ_a≤0が必要。Nが同じというだけではFullの追加処理を払えない。
- N_F<N_aでも、coin rejectまでのpacket/law/sortやF3 hash/angle再query費用が大きければ逆転し得る。
- P/3よりN_F>N_aの場合も、μ_Fやsetupが十分小さければ成立は論理上可能。Nだけで費用優劣を決めない。
- 複数axisではCL1のNμをΣ_j N_a,j μ_a,jへ置換する。setup共有回数を勝手に1や全Nと決定しない。

**取得状態**：S/μ/ΩにはF2の標準bit上界とSOURCE経路がある。CPU秒の認証期待値、全N schedule/output、両方式同等F3入口の費用はない。従って現時点の一般CPU比較は`CONDITIONAL_BREAK_EVEN`（条件は特定済み）かつ`INSUFFICIENT_CERTIFIED_INFORMATION`（費用値の順位は未決）である。F3 Full対未実装canonicalを完成実装同士の比較として扱う場合は`NOT_COMPARABLE_UNDER_CURRENT_CONTRACT`。

## 3. Q：量子の各費用座標

πは全trial分母のdigital event質量、c_d(e)は同一の価格規則、w_a,d=Σ_e π_a,e c_a,d(e)。各座標dについて

\[
\boxed{C_{q,F,d}\le C_{q,a,d}
\iff N_Fw_{F,d}\le N_aw_{a,d}.}\tag{Q1}
\]

z_a>0ならν_a,d=w_a,d/z_aなので

\[
\boxed{\frac{N_F}{N_a}z_F\nu_{F,d}\le z_a\nu_{a,d}.}\tag{Q2}
\]

canonical aではz_a=1。z_F>0かつν_a,d>0なら

\[
\boxed{\frac{\nu_{F,d}}{\nu_{a,d}}
\le\frac{N_a}{N_F z_F}.}\tag{Q3}
\]

これはaccepted一回あたり許容できる価格増の上限を表す。z/νの未知値は記号であり、新たに評価していない。z_F=0やzero価格はQ1へ戻る。均一価格を使えるのは認証された座標のみ。general native価格はlabel・word・control役割・child/angle・誤差/合成policyに依存する。

strictはQ1の`<`、tieは`=`。N_F≤N_OだけではQ1を満たさない。Ordinaryとの同一proposal＋child非依存の構造価格という追加条件を使うと、[量子資料§3](quantum_event_cost_comparison.md)の非負和から論理的accepted数・total CQ・helper RZ/H・skeleton CX・outer Zの期待総量の非劣を認証できる。これはnative T/CX/depth全体への結論ではない。P/3にはその包含関係を適用しない。

## 4. AFF：取得済みevent価格がaffineな場合

同じ固定literal provider会計において、未知共通provider価格v_k≥0を分離し、

\[
C_{q,a,d}=t_{a,d}+\sum_k A_{a,d,k}v_{d,k},\quad
\Delta C_d=\Delta t_d+\sum_k\Delta A_{d,k}v_{d,k}
\]

と書ける場合は、**Δt≤0かつ全ΔA≤0**が非負価格全域で非劣となる十分条件である。有限価格でΔt<0ならstrict、又は負のΔAに対応する価格が正など、総差がstrictになる条件を別に確認する。係数が正負混在なら、取得済み/認証価格領域V上でΔt+ΣΔA v≤0となる領域を比較する。任意の都合のよい価格を選んで普遍優位としない。

[G7既存報告][G7R]は、登録P5ケースのT_RZ切片とlabel別CQ/preparation係数について、この型の**既存のstrict結論**を保存している。その引用範囲では`CERTIFIED_FULL_BETTER`（conditional expected T affine model）という歴史的判定がある。今回新たに係数を再集計・再ランキングしていない。これはbit/CPU追加費用を含まず、物理CQ実装・state準備・完成backendを認定しない。異なるforward/control/adjoint価格や非加算最適化を導入するなら、その元の価格モデルを超えるので追加証拠を要する。

## 5. ENC：上界比較と実費比較

同一費用量Cについて認証区間[C^−,C^+]がある場合の**十分条件**を用いる。

| 分類 | 認証条件 | 主張できる内容 |
|---|---|---|
| `CERTIFIED_FULL_BETTER` | C_F^+<C_a^− | 当該座標のstrict順位 |
| `CERTIFIED_FULL_NONWORSE` | C_F^+≤C_a^−、又はSOURCE構造から直接C_F≤C_a | 当該座標の非劣。strictは別 |
| `CERTIFIED_TIE` | exact共通値・直接等式、又は両区間が同一点 | 等しい。区間重複だけではtieでない |
| `CERTIFIED_COMPARATOR_BETTER` | C_a^+<C_F^− | 対照strict優位 |
| `CONDITIONAL_BREAK_EVEN` | CL/Q/AFF条件を記号で特定、必要な費用値が未取得 | 成立条件だけ確定 |
| `INSUFFICIENT_CERTIFIED_INFORMATION` | 有効な上下界不足、重複区間、未知価格 | 順位未決。重複は同等の証拠ではない |
| `NOT_COMPARABLE_UNDER_CURRENT_CONTRACT` | 精度/target/policy/層/単位が不一致 | 共通契約なしで順位を出さない |

**C_F^+<C_a^+だけでは実費C_F<C_aを証明しない。** 両方とも上界なら「保証上限の順位」とだけ表記する。F2のO式は未知定数を含む計算モデル上界であり、有限のCPU費用区間に変換していない。下界が0しかない場合、正のFull上界と比較側上界から優位は出せない。相関する費用項を独立に最大化して得た上界も、上界としては使えるがtightな実費とはしない。

## 6. 対照ごとの現時点の判定

| 比較・座標 | 現在言えること | 残る条件 |
|---|---|---|
| F対O：認証N | W2bから非劣を継承 | SOURCE/input/K等の一致 |
| F対O：accepted論理実行、total CQ、固定helper/skeleton、outer Z | 共通digital proposal・child非依存価格の範囲で期待総量非劣 | 登録query-model・正常law・zero前skip。実測数/native価格でない |
| 全arm：明示outer/helper workspace2 | このIR座標はtie | provider追加領域を除く |
| F対O：実古典CPU/bit実費、total label価格/native | CL/Qの条件式まで | S/μ/Ω、label/angle価格等 |
| F対P/3：一般N/量子/古典 | W2b＋CL/Qの条件付き比較 | Nの具体条件、実digital期待価格、古典追加費用 |
| 既存G7 P5結果のliteral期待T | 過去報告の限定strict結論を引用可能 | 同じ固定入力/価格モデルのみ。新一般順位ではない |
| F3監査Full対canonical完成実装 | 現在直接比較不能 | canonical非identity入口が未実装、費用も未知 |
| P5Closed/G9対G7 | 現契約では比較不能 | bias/confidence/log/axes/root/zero等の共通化が必要。今回は作らない |

## 7. Paretoと研究判断

同じ対象・精度・費用モデルで、共通vectorの全座標が非劣、少なくとも一座標strictならそのvectorについてPareto改善と呼べる。logical CQの改善と古典CPUの増加、期待数の改善とhard capの増加、時間とmemoryの交換があればトレードオフである。未知座標を0にしてPareto優位を完成させない。CPU秒/bit/memory/T/CX/depthを無換算に足さず、未採択のscalar価格・materiality thresholdを作らない。

今回閉じたのはCL/Q/AFF/ENCという**条件**と、SOURCEで直接分かる限定IR関係である。一般の実費順位・科学的GO・最適性を認定していない。B/C取得が必要かは目的と既存限定証拠で決まる。[計画](missing_cost_evidence_and_execution_plan.md)を判断資料としてユーザーへ返し、**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[G7R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md
[PU]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/850ac89e1a6ea955def805e5462b8694ec0840b4/docs/research/track_b_g6_pu_analytic_proof_20261018/proof_or_counterexample_certificate.md
[W2B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed26d1c85cf7ad80625f7abbc662b70abd690a97/docs/research/track_b_g6_w2b_certified_trial_budget_20261018/README.md
