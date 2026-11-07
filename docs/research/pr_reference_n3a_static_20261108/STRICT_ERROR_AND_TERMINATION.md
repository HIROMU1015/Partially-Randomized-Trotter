# N3-A：strict誤差・有限precision・停止性

2026-11-08 JST。[構成定義](FINITE_DICTIONARY_CONSTRUCTION.md)、[REPORT](REPORT.md)。以下は記号的導出で、guard・合成・数値試験を実行していない。epsilon_G=10^-11は**最終の1 native RZ**に対する既存許容値として固定する。内部b・eta_j・余裕は未採択。

## 1. Normと量子化

delta=theta-tilde thetaとすると、2×2対角行列の各差の絶対値は同じなので

\[
\|R_Z(\theta)-R_Z(\widetilde\theta)\|_2
=2|\sin(\delta/4)|\le|\delta|/2\le2^{-b-2},
\]

\[
\|R_Z(\theta)-R_Z(\widetilde\theta)\|_F
=\sqrt2\,2|\sin(\delta/4)|\le\sqrt2\,2^{-b-2}.
\]

これはstrict行列差であり、位相を最適化した距離ではない。最近接grid誤差には、binary64 targetを正確な有理数として読み、整数で丸める構成条件が必要。

## 2. 基本wordからの合成certificate

各基本word W_jはexact unitaryであり、Wを含む元の基本target RZ(2^j)について保存外向き**Frobenius**上界g_jを持つとする。負符号のactual adjointも同じ上界を持つ。

Unitaryによる左右乗算はFrobenius／spectral normを変えない。有限積の差を1因子ずつ展開しtriangle inequalityを使えば、近似wordが互いに非可換でも

\[
\|W_{dict}(\theta)-R_Z(\theta)\|_F
\le \sqrt2\,2^{-b-2}+\sum_j |s_j|g_j.
\]

従って、実emitterがexact wordと一致する条件の下で

\[
\sqrt2\,2^{-b-2}+\sum_{j=-b}^{J_b}g_j
\le\epsilon_G-\mu,\qquad 0<\mu<\epsilon_G
\tag{F}
\]

は全域strict Frobenius許容の、余裕を持つ十分条件。spectral誤差はFrobenius以下なので既存whole会計に使える。g_jは既にFrobenius上界であり、**和をさらにsqrt2倍しない**。一方、文献等のspectral保証eta_jだけから組むなら、既存guard規約へは`sqrt2*(2^(-b-2)+sum eta_j)`を使う。

例えば未採択の0<lambda<1を用い、量子化側をlambda*(epsilon_G-mu)、基本word和を(1-lambda)*(epsilon_G-mu)以下とする分割が可能。量子化条件は

\[
2^{-b}\le2\sqrt2\,\lambda(\epsilon_G-\mu).
\]

M_b>0で一様基本許容を使うなら`g_j <= (1-lambda)*(epsilon_G-mu)/M_b`は十分。各基本wordを単にepsilon_Gで合成するだけでは、積全体が同じepsilon_Gを満たす保証にならない。これらは許容値の条件式であり、b／lambda／mu／内部precisionを数値選択したものではない。

有限算術でFを確認する場合は、c>=sqrt2を証明できる有理数cで量子化項を上から囲い、保存有理上界g_jと厳密加算できる。floatのlogや近似sqrt値で境界PASSを決めない。古典的生成／emitterの追加Frobenius誤差e_emitがあるなら、Fの左辺へ**一度だけ**加え、同じ最終epsilon_G内に収める。exact Clifford+T命令としてwordと完全同一ならe_emit=0を代数的に資格できるが、Qiskitの浮動global_phase表現で自動的に0とは置かない。

## 3. 既存guardとの整合と限界

[strict_guard](../../../src/trottertracks/track_c_h4_synthesis.py)はbinary targetの整数比からexpを区間評価し、W付きexact_word_matrixとの差のFrobenius上端をFractionで返す。`t`をexact行列評価のためTの7乗へ展開する処理は、物理T†の価格を7Tへ課金する規則ではない。実emitterはT†を1物理出現として扱う。

完成辞書の基本recordがこのguardで資格され、入力丸め／連結／phase規則が証明されれば、Fは**合成によるcertificate**となる。全vacuum targetごとに新たなpygridsynth／guardを呼ぶ必要はない。ただし現`acquire`・`validate_record`はこのrouteのrecordを扱う実装ではない。既存の`guard_method=exact cyclotomic ...`を未評価の連結wordへそのまま付けない。基本certificateとFによる証明を明示する新しい最小adapterが必要になる。

既存acquireの高精度guard再試行は固定1回であり、全wordに対してその桁数で判定できるとは証明されていない。数学的誤差が許容値よりstrictに小さく、正しい区間評価が精度と共に収束するなら、有限精度でその余裕を識別できる。しかし、その性質を現実装の全入力・固定retryへ無条件に移さない。基本wordごとの実資格済み上界を保存する経路と、未完成辞書の将来生成を区別する。

## 4. 辞書完成後の停止性：条件付きの構成的証明

有限b/J_b、全基本recordと有限文字列、決定的な採用規則が固定・資格済みなら、次はすべて有限処理である。

1. binary64の整数比読取、有限整数の最近接丸め。
2. 高々M_b桁のbit分解と、有限表からのlookup。
3. 高々M_b個の有限word連結、必要なら全体のactual adjoint。
4. exact W台帳、有限和の誤差certificate、実word由来価格情報の返却。

文字列長は正側で基本word長の和以下。負側も既存有限alphabetのinverse書換え／回路inverseにより有限長である。T↔T†で実T合計は保存される。整数／メモリ操作が通常どおり完了する計算モデルを前提とし、online処理には合成探索・無限retry・backend待ちを入れない。

欠損keyがあればその表は「完成辞書」ではない。例外を返すことは有限失敗ではあるが、全入力で**wordを返す**total policyとは異なる。miss時に現pygridsynthを無制限に呼ぶ方式は、この停止証明の外。

## 5. 辞書生成時の停止性：別の命題

RZはdet=1なのでstrict近似の数学的対象に含まれる。[Selinger v2 Remark26](https://arxiv.org/pdf/1212.6253v2)のdet条件を限定照合した。正の許容誤差を持つ有限個の基本targetについてwordが存在することは、固定pygridsynthが必ずそれを返すことや、短いT価格を保証することとは異なる。

[固定2.0.0 gridsynth.py](https://github.com/quantum-programming/pygridsynth/blob/65f6dd06538675d63a239009574dea5c636668b1/pygridsynth/gridsynth.py#L245)には、strict探索で分母指数を増やす外側ループがある。[config.py](https://github.com/quantum-programming/pygridsynth/blob/65f6dd06538675d63a239009574dea5c636668b1/pygridsynth/config.py)の内側試行／timeout設定は、全基本keyでの取得上限ではない。CT-2／CT-3Aの未資格を維持する。

数学的には、strict近似可能性と、各有限wordのexact行列・計算可能なtargetに対する**収束する保証付き区間評価**があれば、有限wordと評価精度を公平に全探索し、許容のstrict内側にあるwordを見つける理論的手順を考えられる。1つの境界wordの判定を永久に待たず、各有限段階の処理が有限であることが条件。存在する内点wordは有限段階で検出でき、有限個の基本targetの辞書も数学的には完成する。

これはCodexの計算可能性に関する条件付き導出で、現pygridsynthの停止定理でも、既存mpmath guardに対する全域収束証明でもない。全探索を実装／実行せず、実用時間やN1以下のT capも与えない。T数を先に制限した探索には、その制限内のword存在を別に保証する必要がある。

実務上の最小経路は、別途固定・承認した有限基本集合を取得し、**全recordが実際に資格済みとなった後でだけ**完成辞書としてfreezeすること。その完成後なら、各vacuum targetでbackendを呼ぶ問題を切り離せる。未取得keyがある段階の実装資格はUNQUALIFIEDであり、有限集合というだけで生成完了を宣言しない。今回その取得を開始していない。

## 6. RPEへの限定接続

最終compositeが1 native RZに対してepsilon_Gを満たせば、CT-0の局所spectral上界・全空間telescopingへの入力になり得る。量子化と基本誤差はその**1 RZの内部合算**であり、さらに別raw diskとして二重に足さない。旧vacuum生成誤差も既存diskへ再加算しない。

全support成功には、他の全RZのword、actual inverse、reset／observable、compiler diskと合成diskの同単位合算、位相・半径・shot条件の証拠が引き続き必要。Fの記号式や有限cacheのPASSだけで保存199 shot/axisの全support CT RPE成功を資格しない。今回RPE再会計は実施していない。
