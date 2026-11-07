# N3-A：実採用wordのT上界とN1への条件付き接続

2026-11-08 JST。[構成](FINITE_DICTIONARY_CONSTRUCTION.md)、[誤差／停止性](STRICT_ERROR_AND_TERMINATION.md)。以下は記号式とN1保存値の引用。新T価格、辞書価格和／最大値、期待値、閾値、改善率を計算していない。

## 1. 辞書経路の実価格

固定した基本wordのT+T†文字数をt_jとする。単純連結をそのままemitし、新たなfusion／再compileを加えず、各基本wordを高々1回使う本構成では

\[
T_{dict}(\theta)=\sum_j u_j(\theta)t_j
\le B_{dict}:=\sum_{j=-b}^{J_b}t_j.
\]

負角を連結word全体のactual adjointにすればT合計は同じ。Tdaggerはその内数であり、再加算しない。基本wordが0Tでもそのstrict誤差とWは消えない。局所近似wordを物理出現ごとに実行する価格であり、cache共有を量子gateの共有とは数えない。

この和は基本wordの**実採用価格**がすべて取得・資格済みの場合の上界。存在する別の短いword、最小T、漸近3log／4log則、LP下界を代入しない。20,000文字という既存acquireの拒否条件も、未取得全基本wordや再利用cacheの取得完了／T capを保証しない。長さ制限へ引っ掛かるならtotal性を失う可能性があり、条件変更を無断採択しない。

## 2. 既存word優先policyのcap

CT-5B/N1の固定D・CSWAP・scalar・中央価格を維持するには、既存CT-4Bの採用record表C0を同じliteral key／wordで優先する候補が適合する。C0は正のabsolute-angle keyの表で、signed targetはその絶対値のkeyを照合する。tableはimmutableかつ資格済みで、負角は同じwordのadjointとする。任意のcache hitを無検証で採用する現acquireの挙動とは区別する。

この複合policyでは、vacuum targetがC0にある場合は辞書を使わない。従って**B_dictだけをpolicy全体のcapとしてはいけない**。

\[
B_P=\max\{B_{dict},B_{C0}\},\qquad
B_{C0}\ge\max_{\theta\in C0}T_{C0}(\theta)
\]

という資格済みの上界を使えば安全である。ここでC0は非負の実word価格の表であり、signed融合補正を混ぜない。より狭く`C0 intersect vacuum support`だけのcapを使うには、その重なりを覆う別の証拠が要る。今回はどちらの最大値も計算しない。

既存wordを辞書の安いwordへ勝手に置換すれば、固定K／d0／d2やframe取消しに使う共通policyの前提が変わり得る。その場合は保存N1閾値を無条件で再利用せず、別途の判断・scopeへ戻す。役割ごとに同一keyの異なるwordを使う方法も、今回の共通policyの採択ではない。

## 3. Pointwise十分条件

保存[N1 RESULTS](../../../artifacts/pr_reference_n1_numeric/2026-11-08/RESULTS.json)のV_thresholdは**173790**。同じ条件Cと固定価格を保持し、全必要wordの取得／strict資格があるなら

\[
v_P(\phi_\omega)\le B_P\quad(\forall\omega),\qquad
B_P<173790
\]

はpointwise優越の保守的十分条件を満たす。辞書だけの経路なら、全基本t_jの和が閾値よりstrictに小さいことが十分となる。現時点ではt_j、B_dict、B_C0、B_Pが未資格／未評価なので、比較PASS／改善率は出さない。

この粗い和には同時に出現しないbitや非到達grid indexも含まれ得る。B_Pが閾値を超えても、実際のsupが超えることやReference劣位を意味しない。より狭い到達集合／結合差を使うなら、追加証拠を定義して止まり、価格を下げるための探索を自動開始しない。

## 4. 期待十分条件

保存期待閾値は

\[
U_* = \frac{1602930468830332421754729}{9223372036854775808}.
\]

一様cap B_PがU_*未満なら期待優越にも十分。ただし期待優越にはpointwise優越は必要ではない。一様capの粗さを避ける場合、canonical73-bit lawを固定SOURCEのvacuum recurrenceと厳密丸めへpush-forwardした確率を使う。

正負両方を含むabsolute-angle cache hit key kの確率をpi_k、cache missで|n|のj桁を使う事象の確率をa_jとすると、固定した複合policyについて

\[
\mathbb E[v_P]=\sum_{k\in C0}\pi_k t_k+\sum_j a_j t_j.
\]

各確率の安全な上界があれば、非負価格との積和でUVを構成できる。bit事象、order、product符号、vacuum角の独立性は不要であり、仮定もしない。pi_k／a_jは同じtrajectoryから生成されるjoint lawと整合させる必要がある。bitが各1/2、gridが一様、vacuumが連続一様という仮定を勝手に置かない。p2だけからvacuum bit確率は決まらない。

必要なのは実policyの確率質量を覆う証拠と、UV<U_*の厳密比較。保存rare/nonrare各1 trajectoryの無加重平均、有限最大値、未観測eventの切捨てではない。確率／期待価格／角分布を今回生成・計算していない。有理期待値に整数Tのceil/gcd格子を流用しない。

## 5. 未取得証拠

| 項目 | 現在の状態 |
|---|---|
| b、J_b、基本集合、内部誤差配分 | 記号条件のみ、数値採択なし |
| 完成辞書・全基本word／guard／実価格 | 未生成、未取得 |
| B_dictとcache優先を含むB_P | 記号上界のみ、数値評価なし |
| canonical vacuum価格のUV | 未取得、分布評価なし |
| N1閾値 | 保存値の引用のみ、再計算なし |
| 全support T優越／CT RPE成功 | UNQUALIFIED |

総資源へ接続できるのは、同じ誤差・成功条件と全roleの取得資格が揃った後である。Reference vacuumは1shotに1個だが、それだけで他roleの誤差や費用を省略しない。今回は199-shot積、全18round総T、T-depth、FT資源を計算していない。
