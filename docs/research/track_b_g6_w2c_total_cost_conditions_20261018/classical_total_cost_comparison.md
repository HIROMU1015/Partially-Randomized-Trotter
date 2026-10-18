# 古典全trial費用：標準bitモデルとF3監査層

[入口](README.md) / [break-even](break_even_conditions.md) / [不足証拠](missing_cost_evidence_and_execution_plan.md)。ここでの量は記号であり、実入力への数値代入・CPU再集計をしていない。

## 1. 二層・単位・呼出回数

**層A**は固定G7 generatorと[F2][F2]の標準bitモデル。4方式を同じ算術・精度で会計する。**層B**は[F3][F3]監査付き実装。Full非identityは実装済みだがcanonical非identityは未対応で、完成済み入口同士の対称比較にはならない。未実装canonical監査費用を0と置かない。

bit operation、random-bit数、実CPU秒、peak memory、output bytesは別座標。random-bit供給費用はbitモデルの一部として明記できるが、その実供給器の秒数をbit演算へ無換算に加算しない。以下のO式は**条件付き上界**で、正確な期待実費や測定CPU秒ではない。未知定数を持つO上界同士では順位を認証できない。

対象j（target/axis/policy identityを含む）ごとに、実行scheduleが必要とするsetup/budget/preflight/取得回数をkとして

\[
S_a= k_{s,a}C_{setup,a}+k_{b,a}C_{budget,a}
      +k_{p,a}C_{preflight,a}+C_{acquisition,a},\qquad
C_{cl,a}=S_a+\sum_j N_{a,j}\mu_{a,j}+\Omega_a.
\]

μは全trialを分母としたonline期待費用。Ωはμへ既に入れた処理を除く保存・最終処理費用。1axisなら和はN_a μ_a。setup/budgetを何回呼ぶか、X/Yで共有するかは未実装の全N実行層のscheduleで固定する必要がある。constructorを再利用可能であることは、全N runnerで実際に一度だけ呼ぶ証拠ではない。arm間・axis間で共有するcold取得も、両側に同じ機会・課金規則を与える。

## 2. F2の課金をそのまま使用する

Lはpの長さ、n_o=(m+1)/2、ℓ_n=1+ceil(log2(n+1))、H_p/H_x/Eは入力と精度の整数対記述長、Hはdyadic確率bits、KはK_rootとする。F2で固定した

\[
\begin{aligned}
Q(b)&=O((b+1)^3),& R(b,K)&=O((b+K+1)^3),\\
D(n,g)&=O(n\ell_n Q(g+H+E+\ell_n)),&
S(n)&=O(nQ(H+\ell_n)+H),\\
b_c&=1+m(H_p+H_x+\ell_m),\\
w_e&=b_c+K+(m+2)H+E+\ell_L+\ell_m,\\
w_p&=b_c+K+H+E+\ell_L+\ell_m,\\
w_b&=b_c+K+(m+2)E+\ell_m
\end{aligned}
\]

を使う。Qは正確有理算術/GCD、Rはscaled整数除算/isqrt/外向きroot、Dは分布構築とsorting、Sは毎callのlaw検査・累積走査と**一回分のH-bit供給**である。実bit供給器ならSの+Hを+T_bits(H)に置換し、二重加算しない。SHA診断をfresh-IIDの実証としない。

| 段階 | Full | Ordinary / Partial / Closed P3 |
|---|---|---|
| frozen setup | O((Lm²+m²+L)Q(b_c+E+ℓ_L+ℓ_m))+3n_o R(b_c,K)+D(n_o,K+ℓ_m)+D(L,H_p) | O((L+m²)Q(b_c+E+ℓ_L+ℓ_m))+gR(b_c,K)+D(g,K+ℓ_g)+D(L,H_p)、g≤n_o+L+1 |
| 内訳 | Green、U用DP、empty parent、order/B_o/Uの各root。自動memoなし | Green不要。group係数・root、group/label law。mode固有係数は必要 |
| budget別call | O((1+ℓ_m)Q(w_b))+C_log、固定96項log | 同じ標準モデル。各方式のB/momentを使う。任意confidence APIではない |
| F1式の検査モデル | O((L+m+1)Q(w_e))、与えられたH/Kの検査 | 対称に必要な入口仕様は別。F3実装と同一としない |

3n_oはfrozen Full constructorのroot呼出数であり、F3 P5による再計算を含まない。C_logも一度と自動決定せずk_bで課金する。H/Kの探索を無料に含めない。

## 3. FullのR/C/A：coin rejectでもpacket費用を払う

[G][G] `FullReturnGenerator.sample` と[K][K] `digital_parent` に対応させる。lはraw word長、cはchild support数。以下はF2の既存課金である。

\[
\begin{aligned}
C_{raw}(l)&=S(n_o)+lS(L)+O(l\ell_L),\\
C_{packet}&=O((m^3+Lm^2+L^2)Q(b_c+E+\ell_L+\ell_m))
              +2R(b_c,K)+D(L,b_c)+O(Q(w_p)),\\
C_{coin}&=O(H+Q(w_p)),\\
C_{event}&=O((m+L+1)Q(w_e)+m^2(\ell_L+\ell_m)^2).
\end{aligned}
\]

| 経路 | 実際の処理prefix | 支払う費用 |
|---|---|---|
| R：raw nonreduced zero | order、raw labels、各selector検査/走査、reduce判定、zero record | C_raw(l)＋zero保存 |
| C：acceptance coin reject | raw-validまで＋local parent/全child係数、root2回、child dyadic law/sort、Parent.s再集計、coin | C_raw(l)+C_packet+C_coin＋zero保存 |
| A：accepted | Cのcoin判定まで＋child draw、π/c̃/weight、reduce/Counter/event metadata | C_raw(l)+C_packet+C_coin+S(c)+C_event＋必要なangle/監査/保存 |

`Parent.child_law`は各childでsを再sumする。F2のL²項を保持し、coin後だけの処理に移さない。Full.sampleはaccepted後のlocal query二回目を避けるが、有理weightと_eventの再計算は残る。**angle取得で旧eventを再取得するF3の別経路**はこのsample本体の省略とは別である。

R/C/Aをdigital law上の排反経路として、失敗を含まない登録モデルでは

\[
\mu_F=\mathbb E[C_{raw}(l)]
+\mathbb E[\mathbf1_{rawvalid}(C_{packet}+C_{coin})]
+\mathbb E[\mathbf1_A(S(c)+C_{event}+C_{angle,used})]
+\mu_{audit/record}.
\]

R/Cのrecordはμ_audit/record又はΩのどちらかで必ず課金する。二重に数えない。C_angle,usedは利用時のみ追加する。最大prefixを全trialへ課すF2上界は使用できるが、期待実費とは呼ばない。raw-valid率、coin率、全trial digital受理率は異なる記号であり、ここで評価していない。

## 4. Canonicalの実際のmode費用

`CanonicalGenerator.sample/event` の静的呼出回数を示す。law構築にはexact比率・id作成・D/sorting、selectorにはS、eventには正確ideal/proposal/weight・word還元・Counter・metadataを課す。表は**sampleからeventまで一回**であり、外部からeventをもう一度取得する場合は追加する。

| 方式・mode | law構築（setup後の一trial） | H-bit draw数 | eventで残る処理 |
|---|---|---|---|
| Ordinary全group / P・3のiid tail | conditional追加なし | 1 group＋l raw labels＋1 child | word全長のideal/proposal積、weight、reduce、metadata |
| Partial `root_iid` | conditional追加なし | group＋child＝2 | root係数とlabel proposal、weight |
| Partial `distinct` | **first_distribution 2回＋conditional 2回** | group＋first＋second＋child＝4 | sampleで作ったfirst/conditionalをeventで再構築し確率取得 |
| Closed P3 `root_collected` | **root-child _law 2回** | group＋child＝2 | sample/eventで各1回再構築 |
| Closed P3 `fixed_first` | **conditional 4回** | group＋word-second＋child＝3 | sampleのword/child各1回、eventのword/child各1回。firstはgroupで固定 |

各selectorのsupport長はgroup g、label L、conditional L−1など実supportを使う。lawが存在するgroupだけ課金し、退化入力で存在しないmodeを追加しない。canonicalのgroup rootsはsetupで保持され、onlineで毎trial rootを再計算する実装ではない。raw wordが還元されても、canonicalではeventを返す。**raw/coin rejectionを追加して費用や受理率を変えない。** 正常モデルではcanonicalの全trial digital受理率は1である。

従ってμ_aは上のmode別処理費用を実digital group/label確率で重み付けた期待値となる。modeの確率を新たに計算せず記号のままにする。cache済みconditionalへ書き換えた仮想版を既存SOURCEの費用に使わない。将来の合法なcache/streamingは4方式に同条件で認め、別SOURCEと取得費用が必要である。

## 5. AngleとF3追加費用

層A frozen generatorはratioを返す。F2の既存angle手順を**使う場合**の課金は

\[
C_{angle}=R(b_c,K_\varphi)+O(J\ell_JQ(b_\theta)),\quad
b_\theta=O(J(b_c+K_\varphi+\ell_J)+E).
\]

これはF2からの引用で、J/K_φ/τの新選択ではない。数学的angle enclosureと物理RZ合成・strict error certificateは別物である。

層Bでは実装済み[F3 SOURCE][F3G]を次のように追加する。

| 呼出箇所 | 必須費用・SOURCE上の事実 |
|---|---|
| `prepare` | Exact型/入力/規格化、P0〜P4、旧Full constructor、P5でorder/label lawを再構築・照合。P5のorder roots n_o回は旧3n_oとは別。source_identityで旧3＋新3SOURCEを読む/hash、policy/input/evidence作成 |
| session `_guard` | 各公開session処理でfingerprint、SOURCE/cert/hash照合、kernel/law全状態のcanonical serialization/hash。無料constant checkとしない |
| `sample` | guard、TraceBitsによるselector再検証/trace（追加乱数drawはしない）、event invariant、証拠serialize/hash。R/Cの正当zeroとtyped failureを区別 |
| `angle_for_event` | 再guard・証拠照合、旧generator.event→packetの**再query**、invariant、atan enclosure（root・有限J項端点）、context/evidence保存 |

従って層Bのμ_Fにはsample側guard/trace/evidenceを全trialへ、angle側再guard/再packet等を実際にangle取得するaccepted経路へ加える。setup側追加はS_Fへ加える。SOURCE/hashのcache削除・証拠削除を施した仮想高速Fullを「現実装」としない。canonical側同等入口の費用S_a^audit、μ_a^audit、Ω_a^auditは**未実装・未知**であり、0ではない。

## 6. Memory・output・失敗

F2 Fullの標準generator peakは

\[
O\bigl(Lm b_c+(L+n_o)(b_c+K+H+E+\ell_L+\ell_m)+w_e+m\ell_L\bigr)\quad\text{bits}.
\]

canonicalにはGreen stateがなく、group/label/conditionalの同時live状態を別に数える。ここで新しいtight memory定理を作らず、個別peak順位は未認証とする。F3は状態serializeの一時コピー、証拠・trace、hash、angle/context、cache/output bufferingが追加される。Python実bytesはF2モデルのbit領域と同じではない。

全N分のoutputを保持する場合のS_saved(N)、保存時間C_output(N)、cache取得/evictionを別に記録する。peak memoryは一般にN倍ではなく、実際の同時live状態・buffer・保持方針で決まる。未実装streamingを導入済みとして容量を消さない。

Nominal全N費用式は、失敗runを成功runだけに条件付けて平均する意味ではない。typed failure時はsetup＋処理済みprefix＋失敗証拠保存を課金し、量子prefix消費も別に残す。失敗をzero trial・無費用・dropoutとせず、自動retryしない。未完了prefixに全N confidenceを付与しない。

**結論**：同一層・単位なら[break-even](break_even_conditions.md)の代数条件が使える。F2上界と過去短traceだけでは実CPU順位を認定できない。F3対称入口・全N schedule・outputを含む費用取得は別途承認が必要。mandatory STOP。

<!-- FIXED_REFERENCES -->

[F2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/full_bit_complexity_accounting.md
[F3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/docs/research/track_b_g6_f3_preflight_angle_20261018/implementation_contract.md
[F3G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/src/trottertracks/algorithm_codesign/g6_f3_generator.py
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
