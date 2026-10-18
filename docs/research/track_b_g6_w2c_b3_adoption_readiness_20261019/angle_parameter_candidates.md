# Angleパラメータの具体候補（未採択）

## 1. 既存保証と実装条件

固定F3 `make_request` はτ>0 rad、正整数Kφ/Jを必須にする。科学samplingのK_root=256とは別のprecisionである。
`enclose_atan` は0≤r≤2、σ±1、外向きroot、半角z、各端点J項のalternating atanと次項remainderを使用する。
`phi_interval`がatan(r)を囲み、midpoint代表値のradius≤τのときだけ返す。J/Kφの自動選択・探索・retryはない。
Full `angle_for_event` はguard/context/元event照合とpacket再queryを行い、失敗でsession terminal STOP。
canonical側も原event再queryと共通auditを保つ。これらを省略したcache版は作っていない。

元のfinite-bit audit §6は**厳密な**半角引数が2/3未満であることとalternating remainderを述べる。
丸めた上端まで無条件に2/3未満だと扱わず、以下の初等的な余裕を入れる。
これは既存lemmaの通常の上界代入であり、新しい非自明な数学定理／科学誤差予算の提案ではない。
根拠は[固定F3 SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/src/trottertracks/algorithm_codesign/g6_f3_precision.py)と
[既存finite-bit監査§6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md)。

## 2. 保存eventを使わない一様な十分条件

δ=2^(-Kφ)。元root_intervalは√(1+r²)を幅≤δのdyadic区間[s_lo,s_hi]で囲む。
F3のroot条件s_lo≥1とr≤2から、z_hi−z_lo=r(s_hi−s_lo)/((1+s_lo)(1+s_hi))≤δ/2。
既存lemmaの厳密z<2/3に対し、Kφ≥3ならz_hi<2/3+1/16=35/48<3/4。
各端点のatan remainderは E≤(3/4)^(2J+1)/(2J+1)。atanの既知のLipschitz定数1と
phi=2 atan(z)を使うと、SOURCEが返すintervalのradiusは

`B(Kφ,J) = 2^(-Kφ-1) + 2*(3/4)^(2J+1)/(2J+1)`

以下となる。従ってB≤τは既存F3受入条件の保守的な十分条件。
F3 API自体はKφ≥1/J≥1を許容するが、この一様な十分評価はKφ≥3を使う。
この十分上界がτを超えた場合も、個々のSOURCE計算が不可能だとは確定しない。未確立と不適用を区別する。
r=0はSOURCEのidentity angleでradius=0。r>0ではatan(r)∈[phi_lo,phi_hi]、
|midpoint−atan(r)|≤B≤τ、区間幅≤2τ。ここで実G7のeventや角度を評価していない。

## 3. 具体候補

| ID | τ（代表値の絶対誤差、rad） | Kφ（root grid bits） | J（各端点の級数項数） | 上の十分上界 |
|---|---:|---:|---:|---|
| A6（R1の提案） | 10^-6 | 32 | 32 | B<2^-20<10^-6 |
| A9 | 10^-9 | 48 | 64 | B<2^-37<10^-9 |
| A12 | 10^-12 | 64 | 96 | B<2^-53<10^-12 |

上界のdyadic比較は、(3/4)^4<1/2を既存のremainder式へ代入すれば確認できる。
例えばA6ではtail≤2^-15/65<2^-21、root部分2^-33<2^-21、合計<2^-20。
A9のtail<2^-38とroot2^-49の合計<2^-37、A12のtail<2^-54とroot2^-65の合計<2^-53。
機械確認はこの固定パラメータだけの有理比較であり、G7のp/x/event/SOURCEや保存結果を数値実行していない。
これらは最適／最小のKφ,Jではない。人工fixtureのτ=1/16,Kφ=8,J=4を採択値に流用していない。

R1にA6を提案する理由は、今回目的が物理角度の改善ではなく既存F3の数学的取得費用の比較であり、
準備した3組の中でJ/Kφが小さく、既存一様domain条件を満たすためである。物理primitive errorとτが
同じ10^-6という数値であっても、別の座標であり、誤差allocationを意味しない。数値は未採択。

## 4. Operand・計算・evidenceへの影響

rの有理numerator/denominatorのbit長をb_rとする。rootの整数格子はKφ+O(1) bit、
root前のscaled整数除算にはO(b_r+Kφ) bitの中間値も現れ、半角有理値はO(b_r+Kφ)、最大冪は2J+1。端点の有理値／remainderの記述長は
O(J*(b_r+Kφ+log J))というSOURCE算術からの粗いモデルで扱う。実CPU秒への換算ではない。
各enclosureは1 root、2端点×(J項＋次項)を処理する。A6/A9/A12では、それぞれ66/130/194の
冪に由来する項評価がある（Fraction内部GCDや個々のbit演算を数えた値ではない）。
Kφを大きくするとoperandが、Jを大きくすると級数と証拠rationalのbit幅が増える。
F3は全J個の中間項をevidenceへ並べず、端点partial/remainder等の固定個数の有理値を保存する。
保存行数がJ倍になるわけではないが、1証拠のbytes・serialization/hashが増え得る。
Fullの元packet/event再query、canonical条件law再構築、全SOURCE/state guardは別途そのまま課金する。

## 5. 全eventへの適用と未確認事項

Fullは既存short-step条件／guardのr≤2x/(l+1)≤2を使う。
canonicalのiid/tail r=x/(l+1)≤1、fixed-first r=x(1-p_j)/3≤1/3。
root群はa0=1−χx²/2≥1/2、0<b0≤x≤1よりb0/a0≤2。
Closed rootでは0≤μ3≤χ≤1より0≤2χ−μ3≤2、従ってb0≥x−x³/3>0。
これは入力値を代入しない静的domain対応で、既存lawを変更する新保証ではない。

したがって既存SOURCEがそのpostcondition通り動く条件の下で、3候補は全accepted eventの数学的radiusを
一様に十分にする。実G7のP0–P5、actual law/context、具体event/phase/weight、A/B trace、
output、実resourceの成立は**NOT_ESTABLISHED**で、初回correctnessから確認する。
実装不一致、r domain外、root/angle precision failure、保存失敗ならSTOP。J/Kφを増やして再試行しない。

## 6. 物理RZとは切り離す

scopeはユーザー指定の`MATHEMATICAL_ENCLOSURE_ONLY_NO_PHYSICAL_CHANGE`。
Layer Aはratioのみ、Bはaccepted毎の数学的enclosure。代表値を物理RZに渡さず、既存RZ strict errorや
sequenceを再計算しない。既存ε_axis、primitive error、confidence、Nは不変。
native保証、母期待CPUのconfidence、coherent精度の新certificateは追加しない。
具体τ/Kφ/Jと費用workloadの採択はユーザーが行う。mandatory STOP。
