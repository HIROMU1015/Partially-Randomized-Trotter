# CT-5B：全support T差に必要な条件

2026-11-07 JST、固定q128/r1・strict epsilon1e-11。[構造条件C](STRUCTURAL_CERTIFICATES.md)、[有限key/未知価格](FINITE_KEY_AND_UNKNOWN_PRICE.md)。以下は記号的な恒等式・十分条件である。新しい価格和、極値、閾値、確率、期待値を数値評価していない。

## 1. PolicyとΩ

Ωは保存canonical independent finite-bit lawのsupport。m7の保存bitsは73であり、浮動確率による別lawやrare/nonrareの無加重平均へ変えない。Pは、到達する各literal positive binary keyに共通の実wordを選び、負角にはactual adjointを返すpolicyとする。既知keyはCT-4Bのe=-11採用recordを固定し、W/strict guard/joint phaseを保持する。

現在のPはpartial cacheに留まる。以下の実費等式・期待値は、**すべての必要keyでPが有限に返る場合の条件式**であり、その完了を証明したものではない。undefinedな価格を0にしない。最小wordの存在価格やLP下界を実行可能なbaseline価格と扱わない。

## 2. 一般の差分式

CT-5Aの記号を継承する。

\[
\begin{aligned}
\Delta T_x(\omega)&=K_x+\delta F(\omega)+R_\Delta(\omega)
+Q_\Delta(\omega)+E_B(\omega)-V(\omega)+A_T(\omega,x),\\
K_x&=D_B-D_R-C_{sw}+S_B+W_B(x)-W_R(x).
\end{aligned}
\]

A_Tはliteral role投影と実native価格の対応が未資格な場合の会計差であり、新しい残差回路/normalization方式ではない。D/frameのSOURCEが固定でも、Cの成立を確認せずA_TやdeltaFを0にしない。

今回の静的導出により、Cと共通record固定を条件として、Dは固定、frameのkey差は0、product価格差とwrapper差は0となる。product-Z消去後のhistogramが異なっても、そのT=0 templateの差だけならA_TのT加重寄与は0である。よって

\[
\boxed{\Delta T_x(\omega)
=K+\sum_{i=1}^{q}d_{o_i}+E_B(\omega)-v_P(\phi_\omega)}
\]

\[
K=D_B^\star-D_R^\star-C_{sw}^\star+S_B^\star,
\qquad d_o=2t_P(h_o)-t_P(f_o).
\]

D_B^star/D_R^starはCT-4BのD未融合＋signed境界補正の保存anchor、C_sw^star/S_B^starも保存価格。Kの数値は新しく計算しない。h_o/f_oは保存のhalf/full positive key。oは0/2で、d_oの数値や最小値も評価しない。Cが崩れる場合は一般式へ戻し、削った項を未知項として戻す。

E_B=sum_i e_(k_i)はB′の物理event位相価格で、wordがある場合は非負。未知のevent位相b2価格に上界がなくても、**差の下界**に対してはE_Bを落とせる。これはB′の未取得wordを実行可能とする資格ではない。

## 3. Pointwise優越

同一trajectoryの依存関係を保持した十分条件は

\[
\sup_{\omega\in\Omega}
\left[v_P(\phi_\omega)-\sum_i d_{o_i}-E_B(\omega)\right]<K.
\]

これは全trajectoryのReference Tが小さいための直接の差分条件。vacuumとorder/product符号/basis runは同じωから生成されるため、独立な価格変数として勝手に最大化・最小化しない。式のsupを今回計算していない。

一様vacuum capを別途資格する場合の、より粗い十分条件は

\[
v_P(\phi_\omega)\le V_{\max}\quad(\forall\omega),\qquad
V_{\max}<K+q d_{\min},\qquad
d_{\min}=\min_{o\in\{0,2\}}d_o.
\]

order集合のsupersetを使い、E_B≥0を落とした条件である。各最悪値を独立に置いた保守性があり、必要条件ではない。d_minや右辺を新しく数値評価したり、CT-4Bの2 vacuum価格の最大値をV_maxとしたりしない。

Cの一部を未資格のまま使う場合は、例えばdeltaF/Q_delta/A_Tの安全な下界L_F/L_Q/L_Aと固定項の下界K_Lを別に資格し、`K_L+L_F+L_Q+L_A+q d_min−V_max>0`を十分条件にできる。これらの下界を根拠なく0と置かない。

## 4. 期待優越

期待優越に、全trajectoryのpointwise優越は必須ではない。全必要wordが取得でき、有限な期待実費が定義される条件の下で

\[
\mathbb E[\Delta T_x]
=K+q\bigl[(1-p_2)d_0+p_2d_2\bigr]
+\mathbb E[E_B]-\mathbb E[v_P(\phi_\omega)].
\]

p2は保存canonical73-bitのorder2確率を表す記号であり、新しく確率を算出していない。[既存distributions](../../../src/trottertracks/pr_reference_bprime_accounting.py)はfloor-CDFの整数区間幅を有理確率に変換し、V3の保存multiplicityへ接続する。今回は同関数やexpected_partsを実行しない。

価格差の期待を足し引きするために、vacuumとorderの独立性は要らない。中央の次数別一回価格と保存lawを対応させ、期待の線形性を使う。E[v]の評価や条件付き境界を作る場合は、同じlaw/同じωとの整合が必要である。frameの共通価格が消えるため、期待frame価格の新規数値集計も差の式には不要となる。

E_B≥0を使い、十分なvacuum期待上界U_Vが

\[
\mathbb E[v_P(\phi_\omega)]\le U_V
<K+q[(1-p_2)d_0+p_2d_2]
\]

を満たせば期待T優越の十分条件となる。これは一様V_maxを要求するより弱い選択肢。さらに、結合変数 `v_P(phi)−sum_i d_(o_i)−E_B` の期待上界を直接資格する方法もあり、一様vacuum capだけが唯一の経路ではない。今回U_Vや結合期待の数値は未取得。

## 5. 差に不要な価格と、実行に必要なword

| 項目 | T差の安全な下界 | 両armの実行/誤差資格 |
|---|---|---|
| 共通frameの未知価格 | key係数差0なら数値不要 | 全到達frame keyのstrict wordが必要 |
| B′ b2 event位相価格 | 非負性で下界から落とせる | 到達するならwordが必要。又は非到達を証明 |
| D/CSWAP/scalar | 固定anchorを使う | C・同じ採用record・strict保証が必要 |
| Ref vacuum価格 | 一様/期待/結合差のいずれかの十分境界が必要 | 全到達vacuum角で取得・停止・strict保証が必要 |

従って、全real-angleの高価なSK fallbackを最初から作る必要性は確定しない。一方で、差が有利そうだというだけで未取得wordの実行を省略することもできない。

## 6. RPE成功・総資源との独立性

frame価格の取消しやE_Bの下界での省略は、**演算子誤差の取消しではない**。両armのすべての物理位置についてstrict局所guardを満たし、同じjoint/reset/測定でtelescopingし、旧raw diskへ一度だけ合算する必要がある。D融合のsigned価格補正を負の誤差上界として使わない。

CT-0 m7のepsilon1e-11十分条件、保存199 shot/axis、alpha1/720、phase cap .4は維持する。共通key欠損や全vacuum policy未資格が残る現状で、CT-4Bのfinite PASSをcanonical CT RPE成功へ移さない。成功保証が得られてもT優越は別問題。

固定199という正のshot係数は、同じround/axisの期待T差の符号を保存するが、今回はその積も全18round会計も計算しない。追加8qubit、native CX/depth/RZ、T-depth/FT資源をTへ合算しない。

記号的条件の整理を完了し、**mandatory STOP**。数値閾値・極値・期待値の評価は別途承認scopeとする。
