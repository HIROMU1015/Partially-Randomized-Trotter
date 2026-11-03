# Law/Taylor：代数、SOURCE、保存binary64の三層

2026-11-03 JST。静的導出・SOURCE照合と人工検証だけを行った。独立詳細科学レビューや実H4再解析ではない。
対象条件・固定commit・保存入力の可用性は[INPUT_AVAILABILITY.md](INPUT_AVAILABILITY.md)。

## Layer 1：整合入力で成立する有限Kの恒等式

signed Hermitian involutionをP_i（符号込み、P_i²=I）、確率をp_i≥0、Σp_i=1、A=Σp_i P_iとする。
偶数K、n∈{0,2,…,K}について

    x_n = τ/(n+1),   θ_n = atan(x_n),
    w_n = τ^n/n! sqrt(1+x_n²),   b = Σ_even n≤K w_n,   π_n = w_n/b.

独立なcomponent抽選の回転をproductの左に作用させると、one-event平均は

    E = Σ_n π_n (−1)^(n/2) (cosθ_n I − i sinθ_n A) A^n.

`w_n cosθ_n=τ^n/n!`、`w_n sinθ_n=τ^(n+1)/(n+1)!`なので、偶数/奇数次数を組にすると

    E = [Σ_{k=0}^{K+1} (−iτ A)^k/k!] / b = T_{K+1}(A)/b.

この導出では個々のP_iの可換性を使わない。独立抽選の積の期待値がAの積になることを使う。
rotationの線形表示にはP_i²=IとHermitian性が必要で、一般のDF fragmentそのものをinvolutionとしてはいけない。
負τも偶数nの重みは非負、sin側に符号が入り同じ式が成立する。

独立なr eventのtailはE^r=T_{K+1}(A)^r/b^r。
Fを先に、Rを最後に作用させる1stepはS=R E^r F、独立なq stepの平均はψ†S^qψ。
同じtailやtrajectoryを反復再利用する場合には、この独立抽選の平均式を無条件に適用できない。

論文のAppendix A(A18–A29)のpaired RTEと対応する。
論文の回転はexp(+iφP)でφ=−atan(x)、SOURCEはexp(−iθP)でθ=+atan(x)なので符号は整合する。
論文のr回合成後のBと、この実装のsingle-event b（保存名B）は区別する。
有限Kの式は上の有限多項式への厳密な恒等式であり、指数関数そのものへの等式ではない。
[原論文v2・Appendix A](https://arxiv.org/pdf/2503.05647v2)

## Layer 2：固定SOURCEで実際に行う演算

| 対象 | 関数・SOURCEの処理 | 整合条件との関係 |
|---|---|---|
| 成分係数・cutoff | df_rte_tail.exact_df_diagonal_coefficients、extract_df_diagonal_tail。dot/sum/乗算、abs(c)>cutoff、identity抽出 | 関数名exactはsymbolic展開の意味。binary64丸めの不存在を意味しない |
| probability / coefficient平均 | p_i=abs(c_i)/λ（float）、samplerへp_i/math.fsum(p)。m17.grouped_averagesの他方はc_i/λ | 理想実数演算では一致。独立した割算・sum・basis和では丸めが異なる |
| τ | research_direction_full_scope._finite_inputsとrte.make_rte_config | 初期tolerance用はλ*δ/r、返すconfigはstep_time=δ/r後にλ*step_time。同じ実数式だが演算順は別 |
| finite support | rte.finite_rte_distribution | 偶数0…K。K2の参照はdegree3まで |
| weight | rte._paired_order_weight | exp(n log(absτ)−lgamma(n+1))*hypot(1,absτ/(n+1))。τ=0は分岐 |
| b / π | finite_rte_distribution | b=float(math.fsum(weights))、π=weight/b。丸め済みπをもう一度数学的に正規化した値とは限らない |
| θ / phase | rte._make_event、m17.literal_law_mean | math.atan(τ/(n+1))、phase=(−1)^(n//2)。回転成分のsignはP_i側 |
| event作用順 | rteのevent定義、rpe_hadamard_failure_validation._trajectory_signal | product列の後にrotation、各stepはF→r event→R |
| Taylor参照 | rte.finite_taylor_operator、finite_rte_operator_moments | term=term@((−iτ/k)A)、degree1…K+1。tail=T^r/exp(r log b) |
| F/Rとidentity | 候補A v2 numeric_half_operators | constant+extracted identityのscalarをFへ一回。ここで追加補正しない |
| 測定軸 | fresh_iid_adapter.conditional_probability | cosine=Re、sine=+Im。仮想shotはclip許容・p=(1+mean)/2・別PCG streamで比較 |

上記はcarrier4591b8d…と元m17/候補A/B-1a固定SOURCEを直接確認した対応表。
RTEFiniteDistributionの検査はcanonical weight等をfloatで再評価してiscloseで照合する。
validatorの通過は実数の恒等式や確率のexact unit-sumを証明しない。今回はこのconstructorをH4へ呼んでいない。
pairing、有限support、回転符号、r/qの位置、normalization一回に、静的に確定した数式変更は見つからなかった。
これはSOURCE全体の正しさ、実数演算モデルとの数値同一性、真の平均の誤差保証を認定したものではない。

## samplerへの接続に残る差

NumPy1.26.4 Generator.choice（replace=True、p指定）は、渡されたpからcumsumを作り、
その最終値でCDFを除算してからuniform drawをsearchsorted(side=right)で分類する。
受け取るpのunit-sum検査と、抽選に使うCDFの正規化は別処理である。
元literal_law_meanは保存πを直接和に使うので、浮動小数点でのCDFを含む実装lawとの完全な同一性は未確認。
[固定版NumPy SOURCE・choice](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_generator.pyx#L776-L784)

PCG64のdouble変換はuint64の上位53 bitを2^53で割る形式。
理想的に独立一様な53-bit wordを仮定した場合、生成済みCDF境界C_jに対する累積bucket massは

    G_j = ceil(2^53 C_j)/2^53,
    p_j^grid = G_j − G_{j−1},   G_0=0, G_J=1

（境界番号は1…J）。丸めたCDFをexact dyadicとして読んだ式で、今回H4へ代入していない。
原確率からのCDFとgrid質量を区別できるが、固定seedのPCG64列の統計的独立性をこの式で証明するものではない。
[PCG64 callback](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_pcg64.pyx#L31-L38)、
[53-bit変換](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_common.pxd#L59-L60)

## Layer 3：保存値を固定した場合

元m17はA_prob/A_coeff、π64/τ64/B64等を保存し、B-1aはθ64等を新SOURCE replayとして保存した。
B-1aは新θ64を固定してHP cos/sin、保存B64からHP log/expを計算した。
これはHP atanや整合π/Bの新生成ではなく、Layer 1の恒等式の前提を再強制していない。
二つの保存Aの差やπ/θ/Bの丸めを残してHPで計算しても、恒等式から数値一致は導けない。
またbasisの数値matrixが厳密unitaryであるとも仮定できない。

既存結果の引用：H4/1Å/STO-3G/保存DF12/L_D3/単一δ=.02/m17 q131072/r32 K2で、
元float gap約6.2594e-10、Task A保存factor gap約6.2431e-10、B-1a保存A gap約3.2584e-10。
差の残存だけで数式の定義差と断定しない。HP収束を実装lawへの誤差上界と認定しない。
元のINCONSISTENT/u_num=null/RPE_tolerance_certified=false/科学UNDETERMINEDを維持する。

## 今回の人工確認

[test_static_identities.py](test_static_identities.py)は実H4を読み込まず、Fractionの有限次数係数、
独立な非可換signed成分の平均積、normalization、telescoping、人工CDF/gridとstrict予算判定を確認する。
数式上の関係と計画の判定規則を検査する技術テストであり、H4の係数・CDF・norm・成功確率の検証ではない。
