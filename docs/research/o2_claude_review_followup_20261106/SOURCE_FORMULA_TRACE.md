# 高次PF × finite RTE：SOURCEと数式の対応

2026-11-06 JST。静的読解と既知の式の代数的照合。Hamiltonian生成、状態作用、normalizationの数値評価、保存値の比・fit・費用再集計、sampling、回路build/compileは行っていない。

## 1. 結論と適用domain

| 確認対象 | 状態 | 意味 |
|---|---|---|
| paired-Taylorの定義、signed time、Bの積による合成 | `SOURCE_CONSISTENT` | 下記の固定SOURCEで直接確認 |
| 各S2 stage中央に1 occurrence、総event数 q R_outer | `SOURCE_CONSISTENT` | signal/nativeの両経路で確認 |
| Claudeの共通τと二乗則 | `CONSISTENT_WITH_ASSUMPTIONS` | 小τ、K≥2、連続的な絶対時間比例配分で成立。実装は最低1 step付き整数配分 |
| B2/B3の費用Gを二乗則だけで説明すること | `PARTIALLY_SUPPORTED` | normalization機構は対応。Gの因果的説明と定量的一致は未検証 |
| 未取得private依存閉包・全歴史cohortの独立再監査 | `UNRESOLVED` | 今回の確認対象外。公開provenanceと取得SOURCEの範囲を超えない |

対象はlinear H4/H6、隣接1.0 Å、STO-3G、保存tol-only threshold 1e-8のDF10/DF19。rankは保存観測で、生成時の固定rank引数ではない。H6の保存weighted-Hermitian completionを維持。B2はone-body＋保存fragment順prefixを保持し残りをRTE、B3はL_D=0でone-bodyを保持し全二体fragmentをRTE。T=.8/1.2/1.6/2、δ=T/q、K=2/4/6の取得域。B2基本prefixはH4 {1,2,3,5,7,9}、H6 {3,4,5,7,11,15}。q/Rの基本格子と条件付き拡張は各REPORTを保持する。旧H4 DF12、旧H6 DF11、PauliやH12へ数値を移送しない。

## 2. carrierと科学実行SOURCE

結果carrierは `155529e20a8fac39e00786c40b57830b71203c4f`。実行roleは[scientific_provenance.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/scientific_provenance.json)から追った。

| role | 完全SHA | 対応 |
|---|---|---|
| provider | `c31ef424175c02379eafebe8fc6c7548c1d9aabe` | rte.py、df_partial_s2.py、rte_compiled_cost.py、df_rte_qiskit.py |
| signal | `e417ffc9f52b13c18f5493193ce510d858664259` | h46_long_time_signal.py、h46_new2_core.py、h46_m5_plan.py |
| native | `27ebabdd2943a4de787ec8b976c38e06fda5a497` | h46_long_time_native.py |
| extension signal | `2069360497a7849b63cb102e02ca1ca412462f52` | h46_long_time_extension.py |
| extension native | `f1e95576a8a924face4213b86172455f4f993d7c` | 同上の取得実行role |
| continuation | `ff323f1b09f96c0ce01955728832ff6b0f4c230f` | K6 probe、pipeline、report |

上のsignal/native/providerの列挙ファイルとextension kernelはcarrier版とbyte一致を確認した。provenanceの保存hashに対し、carrierのpipeline/reportは一致しないが、指定continuation SOURCEの同名ファイルは一致する。reportの差には再利用価格・固定δ価格・保存副指標の読込、mean/trial binding検査等の追加がある。従って最終費用集計の再現には最終report版と入力の対応を別に保持する必要がある。今回確認したnormalization kernelは一致するが、初期reportのhashだけで最終会計の全処理を同一と認定しない。旧reportを実行して差を消していない。SOURCE_INDEXに各固定blobを記録する。

B3は[provenance](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/scientific_provenance.json)のsignal `e47696e86c2f98f52f390ed084bb1bfeff4bd197`、native `638e0a989b4471a075549ccb58617b1030e6748f`、confirmation `ae3d301e2318a3f9f1331c047c68d310771b6849`を区別する。取得済み各版のh46_b3.pyは、元長時間signal/nativeを利用する構造。後続recovery・高速化roleの全private実行byte閉包までは再監査していない。

## 3. finite RTEの定義

[provider rte.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c31ef424175c02379eafebe8fc6c7548c1d9aabe/src/trotterlib/rte.py#L1100) `_paired_order_weight` L1100–1114、`finite_rte_distribution` L1460–1491が正本。

\[
a_n(\tau)=\frac{|\tau|^n}{n!}\sqrt{1+\frac{\tau^2}{(n+1)^2}},\quad
\beta_K(\tau)=\sum_{n=0,2,\ldots,K}a_n(\tau),\quad p_n=a_n/\beta_K.
\]

Kは非負の偶数event次数の上限。paired numeratorの多項式は次数K+1までである（[h46_new2_core.py L48–58](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e417ffc9f52b13c18f5493193ce510d858664259/scripts/h46_new2_core.py#L48)、rte.py L1854–1873）。Kを通常の「Taylor多項式最高次数」と読み替えない。

`exact_finite_distribution`がβ_K。`paper_upper_bound=exp(τ²)`（rte.py L1117–1124）は別fieldであり、実際のBに代入する値ではない。ここでのexactは有限分布の定義に対するもの。浮動小数点評価の厳密包含を意味しない。

\[
\ln\beta_0(\tau)=\tfrac12\tau^2+O(\tau^4),\qquad
\ln\beta_K(\tau)=\tau^2+O(\tau^4)\quad(K\ge2,\ K\ \mathrm{even}).
\]

Claudeの展開はこの定義に整合する。K0の1/2係数をK2/4/6へ適用しない。別law・別normalizer・qDRIFTの固有値依存減衰にも移送しない。今回、この展開の数値テストや実τでの残差評価はしていない。

λ_Rは保存DF tailを基底回転した非identity Z/ZZ involutionの係数絶対値和。DFのfragment重みそのものの和やHamiltonianの作用素normではない。`matrix_only_tail_coefficients` L131–146はidentityを別集約する（[実行SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e417ffc9f52b13c18f5493193ce510d858664259/scripts/h46_new2_core.py#L131)）。抽出identityとHamiltonian constantはscalarへ一度だけ戻す。

## 4. stage係数、挿入位置、時刻

[h46_m5_plan.py L13–55](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e417ffc9f52b13c18f5493193ce510d858664259/src/trotterlib/h46_m5_plan.py#L13)と[保存m5定義](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/references/h46_m5_best_formula_v1.json)を照合。

| 公式 | S2 stage係数列 | 符号と正規化 |
|---|---|---|
| S2 (`2nd`) | (1) | 正、和1 |
| 標準Yoshida4 (`4th`) | (a, −u/(2−u), a), u=2^(1/3), a=1/(2−u) | 中央負、和1。5段Suzuki再帰とは異なる |
| `4th(m5_best)` | reversed(tail) + (1−2 sum(tail)) + tail | 11段。tailは正、中央負、和1の構成 |

m5のtail literalは `0.11225428783206336, 0.14181961643121604, 0.1211920365713068, 0.13781643533753404, 0.12547397927357543`。中央はbinary64の `1 - 2 * sum(tail)` を維持。由来は別PF repositoryの `20d64c2fea1ed2d2188777df4f3d962ee94df486`、`actual_circuit_optimized_4th_m5_list`。名称m5だけで他論文の最適化公式へ同定しない。正式な厳密代数的4次certificateを今回追加していない。

各outerに各係数を1回ずつ適用する。各stageは、保持one-body/prefixの半時間sweep、中央RTE occurrence、逆順半時間sweep。RTEは中央に1 occurrenceで、両half sweepへ2回入る構成ではない。`evolve_batch` L262–288とnative `compose` L38–67で一致する。

\[
\delta=T/q,\quad t_i=w_i\delta,\quad \tau_i=\lambda_R t_i/r_i,\quad
\sum_i t_i=\delta,\quad T_{abs}=|T|\sum_i|w_i|.
\]

実時間はsigned sum、normalizationは絶対時間に依存する。plan内の歴史的`SYSTEMS`にあるH4 rank12は、weights/scheduleを再利用する長時間研究のrank指定ではない。長時間SOURCE自身のRANK=10/19と保存bindingを優先する。

負stageでもsigned step_timeをsamplerへ渡す（[df_partial_s2.py L712–762](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c31ef424175c02379eafebe8fc6c7548c1d9aabe/src/trotterlib/df_partial_s2.py#L712)）。order重みは|τ|で同じ、rotationはatan(τ/(n+1))、phaseは(−1)^(n/2)（rte.py L1594–1658）。負時刻を正に置換してbiasを計算する実装ではない。同じseedの正時間trajectoryの単純な逆回路であることまでは意味しない。nativeのactual-adjointによるuncomputeと、signed-time lawは別に確認すべきもの。

## 5. ABS_TIME整数配分とevent会計

`ABS_TIME_PER_OUTER`は[plan.schedule L47–55](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e417ffc9f52b13c18f5493193ce510d858664259/src/trotterlib/h46_m5_plan.py#L47)に存在。allocationの本体は[h46_new2_core.py L16–28](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e417ffc9f52b13c18f5493193ce510d858664259/scripts/h46_new2_core.py#L16)。

1. R_outer≥stage数Sを要求。不足なら拒否し、暗黙にeventを増やさない。
2. まず各stageへ1 step。
3. 残りR_outer−Sを|w_i|/A、A=Σ|w_i|に配り、floorを取る。
4. 余りを小数部分の降順へ配る。同率はstage index順。
5. Σr_i=R_outer、min r_i≥1を検査する。

したがって実装は一般に `r_i=R_outer |w_i|/A` と等しくない。小さいRや多段m5で最低stepの影響が残る。左右対称の係数列でもtie-break後の整数配分が完全な回文になるとは限らない。

native `sample_requests` L70–87は各stageでlen(events)=r_iを検査する。q outerを通じてoccurrenceはqS回、`RTEEvent` objectはn_ev=qΣr_i=qR_outer個。B3の`s`もこのqR_outer（[h46_b3.py L72–76](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/638e0a989b4471a075549ccb58617b1030e6748f/scripts/h46_b3.py#L72)）。これは量子shot N、古典価格標本n、order-n event内のn+1個のcomponent選択、compiled gate数のいずれとも異なる。n_evを「全乱数draw数」や「RZ数」と呼ばない。

## 6. Bの正確な合成と近似

[実行signal.evolve_batch L247–311](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e417ffc9f52b13c18f5493193ce510d858664259/scripts/h46_long_time_signal.py#L247)は

\[
\ln B=q\sum_i r_i\ln\beta_K\!\left(\lambda_Rw_i\delta/r_i\right)
\]

を集約する。native.cost_trial L244–246も実occurrenceのevent数×log βを加算する。stage境界に追加normalizerはなく、deterministic half sweep、scalar、Hadamard wrapperに追加のBもない。境界はcompiled Cへ影響し得るため、Cについての加法性は意味しない。

小τで、a_K=1（K≥2）、a_0=1/2とおくと、固定finite Kについて

\[
\ln B=a_Kq\lambda_R^2\delta^2\sum_i\frac{w_i^2}{r_i}
+O\!\left(q\lambda_R^4\delta^4\sum_i\frac{w_i^4}{r_i^3}\right).
\]

連続配分r_i=R_outer |w_i|/Aに限って

\[
\ln B\approx a_K\frac{\lambda_R^2T^2 A^2}{n_{ev}}.
\]

Cauchy–SchwarzによりΣw_i²/r_i≥A²/R_outer、等号条件が絶対時間比例配分。これはleading termの配分関係であり、有限Kの正確なlog BやGの厳密下界を新しく認定するものではない。実装の整数配分には一般式を使う。Claudeの「全stageで共通τ」は連続近似であり、SOURCEの厳密な等式ではない。

K2/4/6はleading termを共有しても、有限τ補正、cutoff bias、event次数law、compiled価格は異なる。小τの確認は最大|τ_i|と近似残差により行う必要があり、保存値を新しく評価していない本調査では全候補が漸近域内と認定しない。

corrected finite numeratorは多項式積、raw meanはcorrected/B。finite-RTE biasはcorrected−exact-tail PF、outer PF biasはexact-tail PF−full-H reference（signal L295–310）。Bで割る減衰と、finite Kによるbiasは別物。shot式はheadroomとlog_B marginも使い、BだけからNは確定しない。

## 7. B2/B3とMC2の支持範囲

B3の`setup` L30–38はprefixを0へ変え、`signal_job` L113–114は同じevolve_batchを使う。nativeも元sample_requests/compose/cost_trialを再利用する。よって同じ配分・正規化構造をB2/B3双方へ適用できる。ただしλ_R、保持費用、PF bias、候補格子が変わる。

| 根拠の種類 | 今回言えること |
|---|---|
| SOURCEの直接事実 | signed stage、中央RTE、整数r_i、Bの積、one-body保持が一致 |
| 条件付き数理 | 固定λ_R/T/event予算でA²がleading normalizationへ現れる |
| 既存の保存傾向 | B3 REPORTは取得域で高次nomineeがS2より高費用、長時間B2 REPORTは条件付きで高次の低費用を報告 |
| 整合的な解釈 | 大きいtailのB3で絶対時間penaltyが重要になり得る |
| 未確認 | normalization残差の定量評価、整数配分寄与の大きさ、G変化の一意な原因、全系・全公式への一般化 |

上の保存傾向は[長時間REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md)と[B3 REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md)の引用範囲。新しい勝敗・比を作っていない。q、prefix、R/K、headroom、N、single-shot C、候補選抜が同時に変わる。最良Gの比をそのままA²の検証としない。Claudeの既存1点log_B比も再計算・追認していない。

技術的整合性はMC2の新規性を保証しない。直接先行は[PRIOR_ART_MATRIX](PRIOR_ART_MATRIX.md)、許可が必要な保存値検証は[MINIMAL_NEXT_VALIDATION](MINIMAL_NEXT_VALIDATION.md)。
