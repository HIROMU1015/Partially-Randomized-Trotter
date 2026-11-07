# N3-A：有限辞書による角度被覆とword構成

2026-11-08 JST。[REPORT](REPORT.md)の固定条件、基準`0a6ba582d246b3b724d42c8b56475c16ebef1dd9`。以下はCodexによる**記号的な構成・十分条件の導出**。b、基本角度表、word、precision候補を選択／生成していない。

## 1. 保存角度域とSOURCE依存性

[CT-0 PHASE_SUPPORT.json](../../../artifacts/pr_reference_ct0/2026-11-07/PHASE_SUPPORT.json)のm7/q128/r1行は、実binary vacuum角について`absolute_actual_binary_angle_upper=82.49247654978845`を保存している。これをLとし、保存binary64値を厳密有理数として読む。上端の到達は主張しない。

[pilot.vacuum_phase](../../../src/trottertracks/pr_reference_interference_pilot.py)はlaneの保存global_phaseから実命令順にRZ/RZZ角の半分を減算する。[transfer.assemble](../../../src/trottertracks/pr_reference_transfer_qualification.py)の既存F/G分割・D融合・basis run・event/productの順序を保ち、最後の`fl(phase_F+phase_G)`を`reference_core`のancilla Pに与える。compiler AのP→RZはこのliteral角を保つ。新controlを後置しない。

[transfer_error.phase_bound](../../../src/trottertracks/pr_reference_transfer_error.py)と[CT-0の既存説明](../pr_reference_ct0_20261107/PHASE_PRECISION_FEASIBILITY.md)に従い、round-to-nearest binary64、保存の有限SOURCE、演算順、remainder経路と`float(2*pi)`対真の周期の差、旧overflow/underflow条件を引き継ぐ。lane毎の逐次減算とQiskit global_phaseのremainderを解析的なbasis相殺や単一countへ置換しない。旧phase生成diskは既に会計済みである。

本方式はこの**生成後の実binary64 RZ target**を近似する。SOURCEのvacuum生成アルゴリズムを変更せず、旧角域を再計算していない。有界性／有限性そのものは、word取得・停止・T上界の証明ではない。

## 2. 正確なgrid index

有限binary64入力thetaを`as_integer_ratio()`相当の整数p/d（d>0）で厳密に読む。NaN/Infは対象外、+0/−0は同じ恒等target。**未採択の**整数b>=0についてh=2^-bとし、p*2^b/dを最近接整数へties-to-evenで丸めたnを定める。符号を分け、非負整数の商・余りで`2*remainder`とdを比較すれば、floatでgrid座標を作らず、tieを正確に処理できる。

\[
\widetilde\theta=n2^{-b},\qquad
|\theta-\widetilde\theta|\le2^{-b-1},\qquad n(-\theta)=-n(\theta).
\]

theta/hをfloatで計算してからroundした結果へ、この保証を無条件に付けない。端点でも同じ整数丸めを使う。丸めによって|tilde theta|がLを少し越える場合も、次のN_bが覆う。

\[
N_b=\left\lfloor L2^b+\tfrac12\right\rfloor,\qquad |n|\le N_b.
\]

N_b>=1なら、`2^(J_b+b) <= N_b < 2^(J_b+b+1)`となる整数J_bを定め、

\[
\mathcal A_b=\{2^j:-b\le j\le J_b\},\qquad M_b=J_b+b+1.
\]

N_b=0なら全grid indexは0、辞書は空でよく、量子化条件だけが必要となる。これは退化ケースの定義で、今回の精度でそのケースを採用できるという主張ではない。N_b/J_b/M_bを今回数値評価していない。実装時もJ_bは整数bit lengthで求め、浮動logの境界へ依存しない。

既存acquire／guardが`float(theta)`を使うため、基本角の直接再利用には**-1074 <= j <= 1023**かつ2^jが非ゼロの有限binary64として正確に表現されることを要求する。これはbinary64形式の制約であり、bの採択ではない。範囲外の有理基本角を0/Infへ変換して合成することは禁止。必要なら別のrational target interfaceが要るため、判断と別scopeへ戻す。

## 3. 基本word使用数と順序

|n|の通常の非負2進展開を使い、全体符号sigmaを付ける。

\[
\widetilde\theta=\sum_{j=-b}^{J_b}s_j2^j,\quad
s_j=\sigma u_j,\quad u_j\in\{0,1\}.
\]

各指数を高々1回用い、使用基本word数はM_b以下。任意のbalanced/signed-digit法を自動採用しない。carryで上位桁が増える表現や基本角を反復する表現には、その桁域・多重度を別に証明する必要がある。

理想回転には`RZ(a)RZ(c)=RZ(a+c)`がstrictに成り立ち、位相を捨てる恒等式ではない。正角では、選ばれた基本wordを**固定した指数順**でwritten matrix productとして連結する。理想RZは可換でも、近似word同士は一般には可換でない。生成順はpolicyとword hashに含める。

負角は正の|theta|に対して得た**連結word全体のactual adjoint**を返す。これはchunk順も各chunk内の文字順も反転する。同じ指数順のまま各wordだけをadjointにする方法を「全体のactual adjoint」と呼ばない。zeroは空の恒等wordを使う数学的定義とし、新しいzero recordを今回は生成しない。

## 4. Phaseと既存emitter

[exact_word_matrix](../../../src/trottertracks/track_c_h4_synthesis.py)はWを含むexact cyclotomic行列を使う。[lower_native](../../../src/trottertracks/track_c2_b_verify.py)はwritten productを逆順に時系列emitし、Wをglobal_phaseへ反映し、負角には`.inverse()`を使う。[CT-1](../../../src/trottertracks/pr_reference_ct1.py)にはH/T/t/S/X/Wの実adjointとemitter照合がある。これらの規約は再利用可能だが、新しい長い連結wordでの実装検証を済ませたわけではない。

W=exp(i*pi/4)IはT metricで0でもstrict operatorの一部。exactなW指数は整数mod8で管理可能だが、これは`theta mod 2*pi`を許すものではない。`RZ(theta+2*pi)=-RZ(theta)`なので、角度keyの無条件2pi同一視はしない。4pi縮約も新たに実装しない。既存SOURCEのremainderは変更しない。

既存emitterのQiskit global_phaseは浮動加算である。数学的なW台帳と実emitterのphase表現の一致、必要なら追加の誤差境界を[STRICT_ERROR_AND_TERMINATION](STRICT_ERROR_AND_TERMINATION.md)の条件に含める。phase fitや合成後の新controlで埋めない。

## 5. Cacheと別表現

同じliteral正角で既存CT-4B採用recordがある場合は、そのwordを優先して固定する候補を検討する。負角はその実adjoint。cache missだけを辞書へ送る**入力keyに依存する共通policy**なら、既知D／CSWAP／scalar／中央価格を維持できる。未検証・可変cacheを単に`acquire`へ渡す方式では足りない。policy全体のcapは[T_PRICE_FEASIBILITY](T_PRICE_FEASIBILITY.md)に従う。

基本辞書の内部recordは`(基本literal角, 内部精度)`で識別し、最終native targetのepsilon_Gに対する既存優先表と区別する。内部でより厳しい基本wordが必要でも、それを同じ角の既存native採用wordへ自動上書きしない。比較の共通policyは、両armの同じnative target／最終精度への選択を意味する。

別の理論的選択肢として、binary64の全指数域まで基本辞書を広げ、入力をそのまま2進展開すれば量子化誤差を0にできる。ただし辞書と積の長さ・基本word精度の負担が増え、現在の価格優越を改善する根拠はない。高radix/window法も辞書数と積長のtradeoffを追加する。今回は比較探索・採択をしない。

RZの1パラメータ加法性を使う本候補は、CT-3Aで検討した全SU(2) net／commutator再帰をそのまま実装する必要はない。一方、単なる有限成功角の最近傍lookupを全域保証へ格上げする方法でもない。これは既知の角度分解とstrict近似の組合せであり、新規性の判定ではない。
