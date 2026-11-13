# DECISION_PACKET_V0：条件付き費用・感度の設計

2026-11-13 JST。**数値部分はユーザーの有限再解析承認待ち。新しい数値結果はない。** [SOURCE_SCOPE](SOURCE_SCOPE.json)のcanonical/候補/SOURCEを使用し、下の記号式を実測結果と混同しない。[実行範囲](NEXT_OPTIMIZATION_PLAN.md#1-承認を求める有限保存再解析b-prelim-v0)を承認後にのみ埋める。

## 0. 証拠水準と未実行欄

| 表示 | 意味 |
|---|---|
| SAVED_OBSERVATION | 元SOURCEで取得済みの値・診断・statusをそのまま参照。新しい資格は付かない |
| CONDITIONAL_DERIVATION | 指定したnominal/構造量/実装scenarioを条件とする算術。十分Nや正式Gではない |
| HYPOTHETICAL_SENSITIVITY | 未知N比やcompiled補正を自由記号で置く式。実測・採択値ではない |
| UNQUALIFIED / UNKNOWN | 不足証拠・law/数値/長時間適用の未資格。0へ補完しない |

| 数値出力 | 今回の値 | 承認後の上限 |
|---|---|---|
| 構造proxyのround寄与/全体/数値enclosure | null | 13round・26軸、固定nominal1配分 |
| tail寄与率 | null | 開始round k∈{6,9,10,12}の4つ |
| 構造量の時間成長診断 | null | 各軸13個のS/T²、隣接12比。fit・外挿なし |
| PR/PF損益分岐係数 | null | 保存価格のある8候補対×2軸＝16係数 |
| D pilotのcohort内C/S・D/S比 | null | m0/m6×X/Y、各2比。既存summaryだけ |
| proxy感度の係数 | null | 13round、上の4tail group、16局所閾値 |
| 十分N、正式G、winner | null | この再解析後もnull |

## 1. 全roundの条件付き構造費用

j=(m,a)、保存nominalをn⁰_j、Cの構造central/enclosureをs_j,[ℓ_j,u_j]とする。各軸を別recordで扱い、X=Yは今回保存された一致の範囲だけで用いる。

\[
H_m=\sum_a n^0_{m,a}s_{m,a},\quad
H=\sum_m H_m,\quad w_m=H_m/H,\quad
W_k=\sum_{m\ge k}H_m/H.
\]

**Hは条件付きの構造費用指標**である。nominalは必要Nでなく、sはE[C]でない。n⁰にB²や価格cohortのpath数を再乗算しない。Hやその部分和は正式Gの上下界でもない。

数値enclosureは非負の整数n⁰で端点を重み付けする。tail集合Aと補集合Āに対し部分和の端点をL_A,U_A,L_Ā,U_Āとすれば、正の分母での寄与率boxは

\[
{L_A\over L_A+U_{\bar A}}\le W_A\le
{U_A\over U_A+L_{\bar A}}.
\]

これは元Cの数値契約に条件付く囲いで、confidence intervalやcompiler誤差でない。round/wire間の独立性を仮定しない。分母0/欠測・非有限ならunknown、再正規化しない。

S/T²と隣接S比は**固定L_D1/K2/共通κ schedule内**の記述だけ。q/Rのceil・R255等、wrapper/決定論head/terminalがあるためT²を厳密モデルにしない。他κ/L_D/Kやcompiled Cへ延長せず、係数fitもしない。

## 2. 構造寄与が正式Gへ移る条件

真の適格Nと価格がある場合だけτ_j=N_j/n⁰_j、ρ_j=E[C_j]/s_jと書ける（n⁰_j,s_j>0）。

\[
G_{PR}=\sum_j n^0_js_j\tau_j\rho_j.
\]

τ/ρを1へ固定して正式Gを作らない。Hでtailが大きくても、τ/ρが未知なら実Gの支配率は未確定。仮にtailのτ/ρとそれ以外の比がそれぞれ共通で、tail/非tailの積倍率の比をzと置く場合だけ、

\[
W_{tail}^{G}= {zW_{tail}^{H}\over zW_{tail}^{H}+1-W_{tail}^{H}}.
\]

zは記号のままにし、任意の数値gridや「N比≈1」を代入しない。これは未知の重みが結論を変える仕組みの説明であり、配分探索ではない。

同じ条件付きproxyにおけるgroup一括価格倍率のlog感度はgroupのW。したがって承認後に得るWは「その固定proxyなら、どの価格誤差が積和に効くか」の係数になる。これは**Nの独立round最適化の根拠ではない**。N変更のRMSE成立は元decoderで全roundを結合して別評価する必要がある。

適格Nと誤差b_jが後に得られた場合の`|ΔG|≤ΣN_j b_j`は再利用する。今回はbも必要Nも作らない。価格区間の同時boxとN集合があれば、その範囲でだけ比較できる。未知の数値/rare/law/systematicを0にしない。

## 3. PRとFull PFの局所損益分岐

同じtask・round・軸にあるFull PFの保存直接価格をf_j、PRの構造量をs_jとし、必要N比r_j=N_PR,j/N_PF,j、compiled/構造比ρ_j=C_PR,j/s_jを未知記号とする。

\[
N_{PR,j}C_{PR,j}<N_{PF,j}f_j
\iff r_j\rho_j < \theta_j,\qquad \theta_j=f_j/s_j.
\]

承認後に計算するのは保存f/sの**閾値θだけ**。これはcompiled価格同士の比でも、実測N比でも、PRの勝利でもない。価格profileに旧PFと新PRの違いがある場合は、その別実装scenarioを明示し、同一loweringの比較と表示しない。

| Full PF公式・要求幅 | m3の直接価格 | m4の直接価格 | 計画での扱い |
|---|---|---|---|
| S2/.1 | あり | あり | 両roundのθ_X/Y |
| S2/.4 | あり | 欠測 | m3のみ、m4 θ=null |
| 標準Y4/.1 | あり | 欠測 | 同上 |
| 標準Y4/.4 | あり | あり | 両roundのθ_X/Y |
| M5/.1 | あり | 欠測 | m3のみ、m4 θ=null |
| M5/.4 | あり | 欠測 | 同上 |

候補の具体的IDはSOURCE_SCOPEの`full_pf_lanes`。q/formula/幅をPRへ一致させることを公平性条件にしない。各方式の精度達成が別に必要。m3/m4の局所不等式を合算して全RPEの結論にするには、欠測roundを含む全N/Cが要る。

全体式は`Σ_j N_PF,j (r_j ρ_j s_j−f_j)`の符号だが、必要Nも長時間fも未知なので今回は数値化しない。長q PF価格を`a_f+b_f q+e_f(q,T,scalar,profile)`と表すのは未検証の感度記号で、a/bの新fit・推定やe=0の採択をしない。保存小qの同一直線性だけからm12へ価格を補完しない。

## 4. D pilotのcompiled対構造差

既存RESULTのfresh summary内にある同cohortのbar C_Q、bar S_Q、bar D_Qをそのまま使い、承認後に各軸のbar C_Q/bar S_Q、bar D_Q/bar S_Qを計算する。m0/m6それぞれfresh4 pathの**development標本比**であり、比の期待値、全law平均比、信頼区間ではない。replay/P1をpoolしない。元の標本分散・XY共分散は転記可能だが、新CI/variance推定・outlier選別をしない。

ideal構造E_P[S]とsampler残差E_Q[D]の関係は

\[
E_P[C]=E_P[S]+E_Q[D]+\{E_P[D]-E_Q[D]\}.
\]

中括弧は未知であり、0と置かない。pilotのbar Dを全13roundへ比例移送したり、CのE_P[S]へ無条件に足してcompiled平均を作らない。Sはcompiled Cの厳密な下界/上界でもない。実lowering差、相殺、rare/tailでDの符号もサイズも変わり得る。

## 5. signal・減衰・誤差から言えること

承認後の一覧は、PR13・Full PF m3/m4の12・新m6の4セルの**既存**raw/corrected/logB/局所bias/radius/allowance/statusをSOURCE付きで表示するだけ。欠測はnull。別経路の原値から新phase/radius/energy biasを再構成せず、abs bias/phase比による候補ランキングもしない。

raw減衰があるとshotが増え得るが、B²や局所半径から必要N比を決定できない。誤branch/alias/未解決loss・target誤差を含む全decoderのriskが制約である。局所phase/T、S3の点RMSE、Track Aの有限T errorを全RPE RMSEの代用にしない。

screenのnorm/unitarity gate失敗には失敗残差が保存されていない。失敗値を復元する科学計算はこの再解析案に含めず、Aの技術調査と必要な別承認へ返す。未開始のM5/B0I/Full-Hを0 biasや敗者として処理しない。

## 6. 予備分析が支える判断と支えない判断

承認後、Hのtail寄与が集中していれば、**そのproxyを条件に**長時間価格domain・shot/correction感度を次の重点課題として提案できる。θは、どの程度のN比/compiled補正なら短roundの費用差が逆転するかを示す。pilot比は旧Sと実compilerの差を無視する仮定の妥当性を局所的に検討する材料になる。

これらはL_D/K/κの新候補を生成せず、正式採択や最適方式を返さない。全体N・長時間C・数値/target/law資格が不足する限り、正式Gとwinnerはnull。数値を見て追加格子・再fit・新H4 runへ自動拡張しない。
