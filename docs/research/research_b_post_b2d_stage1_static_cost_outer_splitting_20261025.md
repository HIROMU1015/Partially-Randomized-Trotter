# 研究B：B2-D Stage 1後の静的費用比較とouter D/R splitting

2026-10-25 JST。**静的SOURCE調査・descriptor数え上げ・次期検証案。科学的採択、同精度native費用優位、GO／NO-GOは判定しない。**

現在の短いD runでは、A4の17-S2 kernelによる節約より、各D runの20＋20-S2 processor端点費が大きい。Stage 1でA2／A3に対する独自のfull signal利得も確認されていないため、直ちにA4のnative合成へ進む根拠は弱い。次の技術的優先候補は、outer Yoshida4と、条件付きで正係数の弱摂動向けSABA2とする。後者を一般四次法とは扱わない。

## 1. 引継ぎ、範囲、SOURCE

起点は結果公開commit **`51b1387996b1b53feb550efa7cb98aadcf5f9189`**。独立worktree、branch `codex/research-b-b2d-static-cost-outer-20261025` で本書のみ追加する。
完走した科学SOURCEは **`758a05ed6aac58a5945af5a5330c792a842b174d`**、保存後のCLI表示修正は **`5bef9c9cbcd7cfd595ec82874531ad3c7982d014`**。後者や本書の版で科学再実行したとは扱わない。

以下は[既存Stage 1報告][STAGE1]の記載をそのまま引き継ぐ。保存CSV／rawの新しい集計・ランキング・差分解析は行っていない。

- Synthetic 2-mode、one-body＋3 DF因子、N=1／mode0 occupation、L_D=2 weight-ranked prefix。分子geometry／AO basis／分子DF rank探索は非該当。
- δ=±0.09、q=2,3、nested n_D=1,2、r_RTE=2／K=2、complex128、correctness atol=1e-12、位相診断無効。36行、941 DAG dispatch、448 exponential key。
- 入力・非可換性、独立A4数学参照4件、published列規約4件はPASS。その後に全比較を完了した。
- A4はA0／A1よりfull coherent signal誤差が小さい。A2との差は小さく、A3対A4の差は丸め誤差水準で優劣INCONCLUSIVE。高次方式のfull operator誤差は約5.1–7.6×10⁻⁷に集中した。
- Outer D/R splittingが制約になっている可能性があるが、一般の支配項と確定したわけではない。Native費用、processor端点費の実価格、shot数、全RPE総資源は未評価。

新規Hamiltonian／operator／signal評価、回路生成・compile、native費用計測、係数探索、sampling law評価、Stage 2は行わない。科学SOURCE、旧契約、raw結果、grant／ledger／marker／STOPを変更しない。研究CのC1最優先・他研究との非干渉を維持する。

## 2. 読み取った実装資産と再利用の限界

PRのSOURCE参照は上記起点版、別PF repositoryは固定commit **`20d64c2fea1ed2d2188777df4f3d962ee94df486`**。別PFの次の4ファイルはGitHubから取得して静的に読んだ。

| 資産 | 確認した役割／今回の利用 | 再利用時の限界 |
|---|---|---|
| [PF product_formula.py][PF] | `yoshida_4th_list`、published Y8m10b、YP8m8 kernel／processor、`_get_s2_sequence`。既知係数と明示列を利用 | Legacy Moralesとpublished列を混同しない。係数の再探索・丸め修正なし |
| [PF pf_decomposition.py][DECOMP] | `symmetric_s2_sequence`、`inverse_s2_sequence`、`iter_s2_sequence_steps`。隣接同termだけの融合 | 非隣接blockの可換化やRTE境界越しの融合を許さない |
| [PF processed_cost.py][PFCOST] | `ProcessedCostComponents.total` はkernel回数と明示processor pair数による加算モデル | H-chainのPauli rotation／RZ-layer表であり、PRのnative DF価格ではない。価格関数は実行していない |
| [published Morales H2 validator][VALIDATOR] | Grouped-Pauli H2で既存列を直接appendし `gate @ unitary` で作用させる参照 | Stage 1は数学的 `P K^n P†` のappend逆順を独立照合済み。旧validatorのoperatorや数値費用をPRへ移送しない。今回は実行なし |
| [b2d_adapter/adapter.py][ADAPTER] | `DRun`、`actual_adjoint`、`make_b2d_plan`、`fuse_adjacent_d_runs`。演算子を作らないdescriptor | `fuse_boundaries=True` は同一context／τの隣接runだけ。Tailは無条件の障壁 |
| [b2d_evaluator/plans.py][PLANS] | A2／A3の既知係数展開、A4の既存DRun、τ=δ/(2n_D)、同方向の前後D半step | 固定5-arm／共通δのplanであり、任意のouter stage列を既に実装したものではない |
| [df_partial_s2.py][DF] | DF preparation、ordered block、basis往復、個別signed指数、scalar／identity phase | 既存requestはpartial-S2用。任意outer列には最小接続が必要で、full D／Rのexact指数は供給しない |
| [rte.py][RTE] | `make_rte_config`、既存event law、`finite_rte_operator_moments`、`compose_finite_rte_occurrences` | 後者はheterogeneous tailのみのordered積。D interleavingや新outer lawは実装していない |
| [df_partial_s2_repeated.py][REPEATED]、[df_rte_qiskit.py][DFRTE] | 共通stepのq反復、signed time、event phase、controlled diagonal処理 | 共通設定のattenuationを不均一stageへそのまま流用しない。新wrapperの生成・検証は今回なし |

関連testsのassert対象も静的に参照した。`tests/test_rte.py` のheterogeneous occurrenceごとのnormalization、`tests/test_df_partial_s2*.py` のsigned time／scalar／control、既存B2-Dの列順・scope・STOP testsは再利用候補だが、新outer法の科学的正当性を示すものではない。今回はこれらの科学testsを実行していない。

## 3. A2・A3・A4の静的費用

### 3.1 数え方

`m` はordered D block数。Stage 1ではone-body＋選択DF因子2個なので **m=3**。`n=n_D`、`q` は共通macro stepの反復数で、全時間は `T=qδ`。
1個のD-S2 primitiveを `S_D,2(s)` と記す。m blockに展開すると、append順は

```text
D0(s/2), D1(s/2), …, D[m-1](s), …, D1(s/2), D0(s/2)
```

で、literal block指数数は `2m−1`。連続するS個のS2では、隣接同一blockの時間だけを足すと `(2m−2)S+1`。固定列で余分な零時間対消去がないことをdescriptorで確認した。ここで数えるblockはGaussian basis変更を含む一つの指数の記述単位であり、native gateではない。

| D半時間 h=δ/2 の方式 | KernelのS2数／反復 | Processor P | Actual inverse P† | 反復数 | S2 word総数 | Literal block指数 | 隣接融合後（m=3） |
|---|---:|---:|---:|---:|---:|---:|---:|
| A2 Yoshida4 | 3 | 0 | 0 | n | 3n | 15n | 12n+1 |
| A3 published Y8m10b | 21 | 0 | 0 | n | 21n | 105n | 84n+1 |
| A4 published YP8m8 | 17 | 20 S2 | 20 S2 | n | 17n+40 | 85n+200 | 68n+161 |

A4は既存の `G_D(h;n)=P_D(τ) K_D(τ)^n P_D(τ)†`、τ=h/nを維持する。数学的P列は `Q(τ)Q(−τ)` で、一般に対称列ではない。Appendは右端から作用し、P†は**採用した同じ実現列の逆順・符号反転**で作る。新しい負時間sampleや独立に合成した近似Pをactual inverseの代用にしない。[固定PF SOURCE][PF]、[既存adapter][ADAPTER]

正のτに対し、kernelの負S2係数数はA2が1/3、A3が6/21、A4が5/17。PとP†は各10/20。これは時間符号の構造であり費用倍率ではない。δ<0では実時間符号も反転する。現在のD内部の負stageは、RTE tailの時間δを変更しない。

### 3.2 Macro境界融合とq+1の端点費

共通τを持つ隣接D runなら

\[
(P K^n P^\dagger)(P K^n P^\dagger)=P K^{2n}P^\dagger
\]

とできる。**P(τ)を保ってkernel回数を足す**のであり、P(2τ)へ取り替えない。
q回の `D_half — R — D_half` にはq個のR障壁があり、macro境界のDだけを融合するとD runは **q+1個**、kernel反復総数は **2qn**、A4 processor pairは **q+1組**。RをまたいだP†P相殺、全executionを一組のPで挟む置換、別shot間の相殺はできない。[`make_b2d_plan`／`structural_cost`][ADAPTER]

融合後のS2 word総数Sとblock指数数Eは、`b=2m−2` として次になる。

| 方式 | S | E（一般m） | E（Stage 1のm=3） |
|---|---:|---:|---:|
| A2 | 6qn | b·6qn+(q+1) | 24qn+q+1 |
| A3 | 42qn | b·42qn+(q+1) | 168qn+q+1 |
| A4 | 34qn+40(q+1) | b·[34qn+40(q+1)]+(q+1) | 136qn+161(q+1) |

このm=3列ではblock別に `D0: S+(q+1), D1: 2S, D2: S`。Blockごとの費用が違っても内訳を保持できる。同一blockの隣接指数以外はまとめていない。

以下は1 executionの **literal展開 → 上記合法融合後** のD block指数数。Tail・scalar・入力準備／測定は含めない。

| n_D | q | A2 | A3 | A4 | A4 processor pair |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 60 → 51 | 420 → 339 | 1140 → 755 | 3 |
| 1 | 3 | 90 → 76 | 630 → 508 | 1710 → 1052 | 4 |
| 2 | 2 | 120 → 99 | 840 → 675 | 1480 → 1027 | 3 |
| 2 | 3 | 180 → 148 | 1260 → 1012 | 2220 → 1460 | 4 |

**確認方法。** 既存 `make_arm_plan`／`make_b2d_plan` のmetadataだけを、架空block labelとopaque tail payloadで展開した。`Fraction` の符号付き時間について、同一block・control・contextの隣接項だけを加え、零時間は除去、tailでは融合を切った。A4の全列からの相殺と、既存 `fuse_adjacent_d_runs` 後の展開数を上の式と照合した。n=1,2の表と長い列の境界算術も確認したが、時間発展や新しい科学gridの評価はしていない。標準ライブラリのみを使い、科学数値／回路moduleのimportを禁止した一時的な数え上げであり、新しい評価器・test suite・成果物群は追加しない。

Stage 1 evaluatorはD half等の数値nodeを共有cacheで評価する。**941 dispatch／448 exponential keyは36行全体の数値評価scopeであり、上の1 executionの指数数や、生成済みnative回路の最適化実績ではない。** 上表は将来合法的に融合する構造を数えたものである。

### 3.3 端点費の償却条件と判断

同じn・同じ共通τで比べる静的条件は次の通り。

- 孤立したD半run：`17n+40 < 21n` は **n>10**。n=10で同数。
- q回を合法融合したfull列：A4<A3は **n>5(1+1/q)**。q=2なら整数n≥8、q=3ならn≥7。qを大きくしても有限qでn=5は届かず、n≥6が必要。
- Processorを全体一組と誤算すると `n>5/q` になるが、R障壁を無視した条件なので採用しない。

これはS2 word／block指数の償却点であり、**同精度の交点ではない**。Stage 1のn=1,2ではA4はA3より各D blockの出現数が多く、A2はさらに短い。既存報告の「A2でもfull signalが近い」と合わせると、この短runでA4だけのnative費用測定を優先する情報価値は低い。一方、本当にD内部精度が必要で長い反復を要する別途選定条件なら、17対21のkernel差には調査価値が残る。端点を償却するためだけに不要なnを増やす提案はしない。

| 問い | 静的に回答できる範囲／不足 |
|---|---|
| A4はA3と近い精度で安くなり得るか | 長いrunの構造的可能性は残る。同精度で必要なn、controlled DFの実価格は未取得 |
| 端点費が節約を消すか | 今回n=1,2のD指数数では消す。Native合成・角度簡約後も同じ倍率になるとは未認定 |
| A2が実用上有利か | 短い列と既存の小さなfull signal差は優先検討理由。精度要求・shotを揃えた総費用優位は未確認 |
| A4へ追加投資すべきか | Outer誤差の切り分けを先にする。D精度律速・長run・端点負担を満たす用途が得られた時に限定費用検証を再判断 |

### 3.4 Native価格へ移す際に必要な内訳

`processed_cost.py` のaffineなkernel／processor分離は再利用できるが、Pauli rotation数・H-chain RZ-layer価格は流用しない。必要なのは各DF blockのbasis往復、one-body／DF-squared diagonal、controlled diagonal、角度合成精度とsigned time、隣接basis簡約、RTE event長・分布、入力準備／測定の価格である。小さい角度や逆角の合成価格は、単なるblock数だけでは決まらない。[DF builder][DF]、[RTE builder][DFRTE]

個別blockはbasis inverse→diagonal→basisで実装され、controlは既存のdiagonal-only規約を再利用できる。同一basisの隣接指数融合と、異なるDF basis間の最適化は区別する。Scalar／抽出identityの `exp(−icT)` は全物理時間に一度、controlledではancillaの相対位相として保持する。RTEのevent phaseも消さない。系だけで同じglobal phaseだから無料とすることはできない。

将来は同じphysical誤差目標・T・状態・confidenceで必要n／outer分割を揃え、(i) D kernel、(ii) runごとのPとactual inverse、(iii) tail event、(iv) control／scalar／preparation／measurement、(v) normalizationで変わる推定rangeとshotを分けて測る。Compiled depthはgate数と別に記録する。今回はその価格、shot数、誤差当たり費用を計算していない。

## 4. Outer改善に共通する平均・位相の条件

現在の理想outer参照は `J2(h)=U_D(h/2) U_R(h) U_D(h/2)`。`U_D`／`U_R` はそれぞれ**和全体のexact指数を表す参照記号**であり、現PRの量子primitiveではない。

有限RTE occurrence jのcorrected meanをA_j、normalizationをN_jとすると、raw meanは `A_j/N_j`。独立tail drawと決定論Dを交互に作用させる列では

\[
M_{raw}=D_s(A_s/N_s)D_{s-1}\cdots D_1(A_1/N_1)D_0,
\qquad M_{corr}=(\prod_j N_j)M_{raw}.
\]

これは既存mean定義と独立性からの記帳規則であり、新しい誤差保証ではない。Signalは同じphysical ψについて `exp(−icT)⟨ψ|M|ψ⟩`。Signed stageごとのconfig・Taylor bias・N_jを保存し、共通step用の `N^q` を不均一列へ仮定しない。[RTE mean／occurrence合成][RTE]

有限Taylorのcorrected meanは一般にnonunitaryで、`A_R(−h)` を `A_R(h)^{-1}` とみなせない。実現event列のadjointは、負時間のfresh drawと異なる。新outer randomization ξでNも変わる場合、補正対象は条件付きの `E_ξ[N(ξ) E(W|ξ)]` であり、平均Nと平均raw signalの積へ置き換えない。相関drawではさらに独立積への分解を使えない。

System channelだけの誤差保証は、controlled coherent signalのfirst moment／relative phaseを決めない。各実現のliteral operator、scalar、event phase、control-zero branchを保つ必要がある。位相をfitして消したoverlapで改善認定しない。

## 5. 既知outer候補の比較

### 5.1 B1：高次compositionと正係数のalternative

**Yoshida4。** 既存係数を再利用する最小の外側高次候補は

\[
J_4(\delta)=J_2(a\delta)J_2(b\delta)J_2(a\delta),\quad
a=\frac{1}{2-2^{1/3}},\quad b=-\frac{2^{1/3}}{2-2^{1/3}}.
\]

理想exact D/Rなら既知四次composition。係数は現A2のgeneratorと同じで、新しい探索を要しない。[固定PF実装][PF]、[Moralesのcomposition整理][MORALES]
各macroでR stageは1→3個、中央は負時間、Rの絶対時間総和は `|δ|(2|a|+|b|)>|δ|`。RTE signed configを各stageに作る必要があり、tail bias・normalization・event費が外側利得を相殺し得る。負時間を絶対値へ置換してはならない。

現finite RTEと内部D近似を代入した実装に四次保証が自動継承されるわけではない。最初は短いA2をD近似に使い、A3を内部精度対照とする方が単純である。A4を含めるなら一般に `P_D(−τ)≠P_D(τ)` なので、`G_D(−h)=G_D(h)†` を確認なしに対称baseの根拠にしない。異なるstage幅のprocessorも相殺できず、現共通τのq+1則をそのまま外挿しない。

**Published高次PF。** Y8m10b／YP8m8を理想二分割S2の外側へ適用する列自体は既知SOURCEから作れる。ただし21または17-S2 kernel、YPではfull D/Rを含む20＋20-S2 processorになり、tail呼出し・負stage・normalizationが増える。内側のfinite-RTE baseはexactなself-adjoint S2ではない。既存のB2-F保留理由を解消せずにfull YPを優先候補へ格上げしない。[B2限定設計 §5][DESIGN]

**SABA2。** 別の既知選択肢は、`H=D+εB` の弱摂動を想定した

\[
U_D(c_1\delta)U_R(\delta/2)U_D(c_2\delta)U_R(\delta/2)U_D(c_1\delta),
\quad c_1=(1-1/\sqrt3)/2,\quad c_2=1/\sqrt3.
\]

係数はLaskar–Robutel §4 Eq.(14)–(17)、Table Iの既知SABA2（先行McLachlan法にも言及）。全係数が正で、R stageは2個。理想Lie積では一重Bの主要項を消し、modified Hamiltonianの残差は `O(εδ⁴+ε²δ²)` を狙う。**一般四次法ではない**。[原論文][SABA]
Quantum PRでのoperator／signal保証は未取得であり、tail weight rankingだけでは小さいεや対応commutator比を保証しない。D近似・finite RTEのbiasも残る。既存のsigned D blockとtailのaccessで列を表せ、exact mixed和oracleを新設する必要はない。Rの絶対時間総和を増やさずに係数変更の価値を問える点がYoshida4との比較理由だが、2回に分けたN_j積やsampling費が有利とは未認定。既存PRにSABA2専用実装を確認しておらず、将来はこの5-stageの最小接続が必要となる。

### 5.2 B2：interaction-picture／THRIFTと過去のclosure

Interaction pictureでは `H_I(s)=exp(isD) R exp(−isD)` と時間順序を扱う。Truncated-Dyson型のknown algorithmは、効率的D evolutionと時間依存Hamiltonianのencoding／アクセスを前提にする。現DF block列とfinite RTEだけから、そのアクセス・SELECT/PREPや時間順序処理が無料で得られるわけではない。[Low–Wiebeのinput modelとinteraction picture][IP]

THRIFTは `D+εΣB_l` に対し、D指数に加えて **`exp[−is(D+εB_l)]`** というmixed sumのアクセスを必要とする。個別DF指数とRTE tailだけでは満たさない。小規模dense exact行列は古典参照としては使えても量子oracleの価格にはならない。[THRIFT Thm.1／Eq.(8)][THRIFT]

過去の[固定closure][CLOSURE]（commit `7af83266f96b714bb3fa2f1ef5e4bc89faf37db3`）と、その [`qc_mixed_access_math_v1.py`][MIXED] をGitHubから必要箇所取得した。対象はH4 linear 1 Å／STO-3G、canonical RHF、全16 DF因子を保持、N=4／8 spin modes、T∈{1,4} atomic time（PR／L_D非該当）で、今回のStage 1 fixtureとは異なる。Registered occupation／bitwise variable-angle arithmetic templateには `NO_POSITIVE_BUDGET_IN_REGISTERED_MODEL`、系列は `CLOSED_CURRENT_TEMPLATE_FEASIBILITY_SERIES`。これは**当該templateのclosure**でありTHRIFT一般の不可能性ではない。新しい具体的な低費用accessなしに、旧oracleを無料として再提案・再実行しない。

同SOURCEの `stages` にある既存の分解・Yoshida composition記述は参考にできるが、実装アクセスの不足を埋めない。THRIFT付録のsmall-perturbation最適化もexact mixed accessや負stageの課題と区別する。

### 5.3 B3／B4：randomized splittingとfull processor

Randomized term orderingは既知の改善方向だが、既知保証は指定されたexact elementary指数とrandom lawに対するもの。現PRのTaylor次数・component samplingと別のouter乱数を導入するなら、独立性／共有乱数、corrected first moment、controlled phaseと重みを新たに定義する必要がある。[Childs–Ostrander–Su][RANDOM]

B4 energy-scale separationのstandard accessでもexactな和全体D／R指数、mixedではさらにD+εB_l指数が必要。論文にはchannelだけでなく平均unitaryの評価もあるため「channel保証しかない」とは扱わないが、内部D近似・finite RTEへ置換した保証は未取得。同一高次step内の共有乱数をfresh drawへ変えることもできない。[B4原論文][B4]、[既存限定調査 §2–3][SURVEY]

既存B2-Fも、現確率kernelに合うcorrectorの保証・和アクセスが不足する。固定Pでの相似変換はkernel固有値を変えない。Pを付けるだけでouter誤差が消える、あるいはfull列のprocessorが無料になるという再提案はしない。CPFの既存表示不整合を推測訂正して実装仕様にすることも避ける。[B2限定設計 §5／§9][DESIGN]

### 5.4 比較表

| 候補 | 狙い／既存研究との差 | 現access・SOURCE再利用 | 正当性の残件 | 費用と最小検証負担 |
|---|---|---|---|---|
| Outer Yoshida4 | 既知高次composition。PR固有の問いはfinite RTEでfull signal利得が残るか | 既存係数、D descriptor、signed tailを再利用。新exact oracle不要 | 理想四次と実meanを分離。内部誤差・負時間N_j・literal phase | R stage3倍、負stage。Ideal参照とfinite mean、refined-S2対照の限定比較 |
| 正係数SABA2 | 既知弱摂動split。PRではRが弱い条件と費用利得が未確認 | 同じD／tail access、既知5-stage係数。新mixed oracle不要 | 弱摂動の実効性、残る二重R項、D／RTE近似bias。一般四次でない | R stage2倍。Yoshidaと同じ参照系で係数変更の効果を切り分ける |
| Outer published Y8／YP8 | 既知高次／processedのfull列への適用 | 列generatorは再利用可能 | Finite baseの非unitary性・非self-adjoint性、full processorの保証不足 | 多数のtail・負stage・full端点費。上2候補より先に進める理由は弱い |
| Interaction-picture／THRIFT | D優勢を利用する既知法 | Split metadata、旧math記述まで。新time-dependent／mixed access不足 | Oracle／encodingの実現と近似誤差を含む保証 | 新access・controlled費の立証が先。旧closureを再開しない |
| Random ordering／B4 | Ensembleで誤差項を減らす既知法 | 個別event／phase資産は利用可能、既存sampling lawそのものではない | Exact access、乱数構造、weighted first momentの正当化 | Sampler・finite-bit・shot費が追加。現時点で新科学評価を優先しない |
| B2-F full corrector | Full operatorの補正を狙う既知方向 | 位相／actual inverseの基盤のみ | Corrector選択、和access、inexact kernel保証が未解決 | Full processor合成とtail反復費。不足条件を維持して保留 |

## 6. 次に判断する候補は2件まで

**第一候補：outer Yoshida4を既知baselineとして限定検証する案。** Stage 1のfull誤差残差をouter側から動かせるかを、既存係数と最小SOURCE変更で問える。既知法の単純適用自体を新規性とはしない。D内部のA4をさらに高精度化するより、tail bias／normalizationとouter利得の競合を直接切り分ける判断価値が高い。

**第二候補：正係数SABA2の条件付き比較案。** 負R stageと絶対時間増加を避ける既知alternativeとして、weak-residual型の係数変更に価値があるかを問う。Stage 1のL_D rankingだけではweak-residual条件は未確認なので、正式採択・改善保証を先に置かない。候補数を増やすための新係数探索はしない。

THRIFT／IP／B4／full YP／B2-Fは現在のaccess・保証不足を理由に優先外とする。一般的NO-GOや性能劣位の認定ではない。A4 native合成への追加投資は、outer診断後にD精度律速／長runの用途が残るかで判断する。

## 7. 次の最小作業案と別途の実行許可

以下は**提案**であり今回実装・実行していない。研究Bのユーザーが候補と条件を採択し、科学実行を明示許可してから進める。承認証明書・packet・receipt・grantは要求しない。

1. **研究Bのユーザー：** Yoshida4を先行しSABA2を同一限定比較に含めるかを判断する。初期のモデル・sector・state・L_D・総時間はStage 1と同じものを候補とする。新しい分子系や追加長時間窓を自動で選ばない。
2. **Codex（採択後の技術作業）：** 既存planとRTE configを最小拡張し、stageごとのD／R時間、normalization、実現順序、phaseを表す。専用frameworkは作らず、opaque stubで列順・符号・N_j積・tail障壁・fresh output／失敗保存／STOPを確認する。旧科学SOURCE・run・STOPは保持する。
3. **研究Bのユーザー：** 新SOURCE、対象候補、T、outer反復・refined-S2対照、内部D精度、r_RTE／K、数値精度・tolerance、call scopeを具体化した限定科学契約の実行を許可する。新しいstage幅を扱うため、旧941／448のscopeや旧Stage 1実行許可は流用しない。
4. **Codex（その許可後）：** 同じfixture・入力identity・非可換性を先に検査し、classical exact-D/R列を診断参照として、理想outer／D近似入りideal-tail／finite corrected mean／raw meanを分離する。A2を短い内部D baseline、A3を内部誤差対照とし、同じTの元outer-S2とrefined-S2を残す。A4追加、高精度、追加grid、native compileは自動追加しない。
5. **Codex（同じ限定実行内）：** Physical operator誤差と同一ψのX/Y・半径・複素signal直接誤差、tail bias、N_j積、stage／block構造を保存する。単一時刻の改善を収束次数の確認とは呼ばない。差がroundoffで識別不能ならINCONCLUSIVE、correctness不一致なら比較を始めずSTOP。完了・失敗後はmandatory STOPし、shot／総資源やStage 2へ移行しない。

次に得たい情報は、outer変更でfull誤差が動くか、その利得がfinite RTEとnormalizationを入れて残るか、refined-S2や短いA2に対して追加stage費を正当化するか、である。新しい費用計測やshot推定は、その限定結果を見て研究Bが別途判断する。

## 8. 今回の確認と停止

実施したのは、固定資料・SOURCEの読取り、既知係数／descriptorの静的数え上げ、文書の整合確認のみ。人工テストPASSを科学的性能証拠にしたり、過去結果を再ランキングしたりしていない。実装SOURCE、実行環境、共通指示、他研究の成果物は変更しない。
本書だけをcommit・pushし、remote commitと本文一致を確認した時点でmandatory STOPする。通常の研究B作業として統括への報告は行わない。

## 参照資料

一次文献は、[Morales et al.：高次／processed PF][MORALES]、[Laskar–Robutel：弱摂動向けSABA][SABA]、[Bosse et al.：THRIFT][THRIFT]、[Low–Wiebe：interaction-picture simulation][IP]、[Childs–Ostrander–Su：randomized PF][RANDOM]、[Kim–García-Pintos：energy-scale separation][B4]。実装・過去結果は本文中の固定commitリンクを参照する。

[STAGE1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51b1387996b1b53feb550efa7cb98aadcf5f9189/docs/research/b2d_stage1_execution_20261025.md
[PF]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/product_formula.py
[DECOMP]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/pf_decomposition.py
[PFCOST]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/processed_cost.py
[VALIDATOR]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/review_response/validate_morales_y8m10b_h2.py
[ADAPTER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51b1387996b1b53feb550efa7cb98aadcf5f9189/src/trottertracks/b2d_adapter/adapter.py
[PLANS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51b1387996b1b53feb550efa7cb98aadcf5f9189/src/trottertracks/b2d_evaluator/plans.py
[DF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51b1387996b1b53feb550efa7cb98aadcf5f9189/src/trotterlib/df_partial_s2.py
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51b1387996b1b53feb550efa7cb98aadcf5f9189/src/trotterlib/rte.py
[REPEATED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51b1387996b1b53feb550efa7cb98aadcf5f9189/src/trotterlib/df_partial_s2_repeated.py
[DFRTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51b1387996b1b53feb550efa7cb98aadcf5f9189/src/trotterlib/df_rte_qiskit.py
[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51b1387996b1b53feb550efa7cb98aadcf5f9189/docs/research/research_b_b2_limited_design_20261024.md
[SURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51b1387996b1b53feb550efa7cb98aadcf5f9189/docs/research/research_b_b4_b2_pr_connection_static_survey_20261024.md
[CLOSURE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_mixed_access_feasibility_v1_closure_20261013.md
[MIXED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/src/trottertracks/representation_exploration/qc_mixed_access_math_v1.py
[MORALES]: https://arxiv.org/html/2210.15817v3
[SABA]: https://arxiv.org/pdf/astro-ph/0005074
[THRIFT]: https://arxiv.org/html/2403.08729v3
[IP]: https://arxiv.org/html/1805.00675v2
[RANDOM]: https://arxiv.org/html/1805.08385v2
[B4]: https://arxiv.org/html/2608.07720v1
