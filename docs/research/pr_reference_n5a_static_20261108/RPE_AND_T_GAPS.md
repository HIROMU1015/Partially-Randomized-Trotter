# N5-A：RPE成功とT-countに残る証拠

2026-11-08 JST。[REPORT](REPORT.md)、[round条件](ROUND_QUALIFICATION_MATRIX.md)。数式は必要な接続の記号的整理のみ。新しい数値を代入・評価していない。

## 1. RPEを閉じる条件

Round mの物理RZ各位置にstrict certificateがあり、同じnative単位に対して

`epsilon_syn_(m,x) <= sum_i epsilon_(m,x,i) + epsilon_shell_(m,x)`

をspectral embeddingとtelescopingで組む必要がある。期待多重度、signed D価格補正、共通frameの価格取消し、B′ event価格の非負性は、この物理誤差和の省略理由ではない。

X/Yの複素信号には両軸を覆う上界を使う。N4-Cでは既存CT-1のmax-axis一様条件を再利用し、

`epsilon_pair_m = max_x epsilon_syn_(m,x)`

`d_syn_m <= 2sqrt(2) epsilon_pair_m`

`d_total_m = d_old_m + d_syn_m`

を1回だけ適用した。Compiler／vacuum／π／SOURCE arithmeticは既存d_old側、量子化／基本guard／RZ emitterは各word certificate側、初期outer設定／segment joinはshell側に帰属させる。同じ項を別名で再加算しない。Raw信号単位で合算してから半径・位相・統計予算へ変換する。

| 証拠 | m7 | 他round |
|---|---|---|
| Role wordとL域・10^-11 policy | N3/N4で資格済み、C条件付き | 域とprecisionを分けた再利用照合が必要 |
| Worst-case RZ数／d_old | CT-0保存値 | 全round保存値あり。今回は引用のみ |
| Shell S<=1のSOURCE根拠 | N4-Cで確認 | 同じSOURCE/Cなら記号的移送可能。数値diskへの適用未実行 |
| 新合算disk→radius/phase→保存N | N4-Cの199/axis PASS | 未実行。旧native PASS／追加精度capが存在してもCT PASSとはしない |
| 失敗率・最終energy | 保存条件に依存するm7の十分条件 | 全18round CTの新しい成功保証は未資格 |

精度を変える場合はC0 priority／固定価格との整合も必要。新しいcapを自動採択したり、保存shot／alphaを増減してFAILを回避したりしない。十分条件不成立とアルゴリズム自体の失敗を区別する。元Hの表現・保存数値行列モデル等に残る条件も今回の静的監査で消去していない。

## 2. 価格構造の移送

[CT-5B](../pr_reference_ct5b_20261107/CONDITIONAL_T_ADVANTAGE.md)の一般差分式を先に保持する。Cと同じ採用policyが成立し、frame／product／wrapperの価格項を本当に落とせる場合に限り、round別には例えば

`DeltaT_(m,x)(omega) = K_(m,x) + sum_(events) d_(r_m,order) + E_B,m(omega) - v_Pm(phi_m(omega))`

という形へ移せる。中央rotationはq_m*r_m個のevent、B′ event phaseはoccurrence内に集約されるSOURCEであり、r1のq個の式をそのまま使わない。K、d、vacuum price capはいずれもroundとpolicyへの依存を残す。

| 項 | 再利用する記号的根拠 | 追加で必要な証拠 |
|---|---|---|
| D価格 | SOURCE固定、qに応じた既存境界構成、C下のliteral template | Round別multiplicity×採用word価格の保証。m7のKを流用しない |
| Frame | Paired trajectoryのbasis pair multisetと共通word policy | 新precision／域での全word資格。取消しは1-shot価格差の話 |
| 中央 | 同じrなら共通のhalf/full literal | r変更ごとの実word・T価格。d_minの符号も未評価 |
| Event phase | 適合wordがあれば価格は非負 | 全必要角のword資格。省略してよいのは差の下界だけ |
| Product／wrapper／CSWAP | 保存policyで確認されたtemplateと価格 | Precision/policy変更後も旧T=0や固定価格が保たれるか再資格 |
| Vacuum | 既存のL域・6072 cap | 新しい域／precisionでの有限実行と価格保証。2 finite trajectoryの最大を全support capにしない |

m7では保存 **K=160222、d0=106、d2=114、B_P=6072<173790** を引用でき、完成policy等を条件としたpointwise十分条件が閉じている。新たな差分・改善率は計算していない。m0〜6はr1でもq／scalar価格が異なり、r変更後は中央word価格も変わる。旧anchorを数値代入する許可にはならない。

Precisionを変えると、ideal Clifford由来のbinary angleでも旧T=0 wordのguardが新しい要求に適合するとは限らない。旧「T=0だからproduct残差を落とせる」という議論も、元のword・strict要求・C条件に限定する。新規合成が常に必要と決めつけず、先に保存record／実guard／emitterを照合する。

## 3. Pointwise・期待・shot重み付き総T

Pointwise T優越が成立すれば、同じlaw・同じroundの期待1-shot価格にも移せる。しかし全18roundでは

`T_total,a = sum_(m,x) N_(m,x),a * E_[Omega_m][T_(m,x),a]`

が目的量となる。ここでlawのbits／r、必要Nと全価格資格を対応させる必要がある。新しい期待値・総Tを今回は計算しない。

**保存m17はB′626／Reference630 shots/axis。** Shot係数が等しくないと、例えば

`N_B E[T_B] - N_R E[T_R] = N_B E[T_B-T_R] + (N_B-N_R) E[T_R]`

であり、1-shotで共通frameが取消せても、右辺第2項の絶対価格は自動的には消えない。Frameを含むReference絶対価格の安全な上界等が必要になる場合がある。単に各roundの1-shot符号を示すだけで総T優越を認定しない。CTでの保存Nの十分性がまだ未資格なroundでは、その価格重みの採用も条件付きである。

期待差に必要なvacuum/order等の価格は同じcanonical lawに従う結合変数。独立性やrare/nonrareの無加重平均を仮定せず、必要な期待／結合上界を別途資格する。Pointwise十分条件、一様vacuum cap、期待vacuum capのいずれを採用するかはGPT／ユーザーの判断へ返す。

追加8 qubit、native CX/RZ/depth、T-depth、FT資源は別指標。今回のT議論へ無根拠に足し合わせない。統括兼Dのnative研究を再実行しない。
