# M1-S V1：算術・coverage・resource設計

2026-10-17 JST、V0。以下は将来SOURCEの設計根拠であり、数値correctnessを実証した記録ではない。新規の実入力／模擬入力評価0。

## 保存信号層と新算術層の分離

旧pilotの保存visibilityと総raw-H phase bias、および対応する半径のcoverageを条件付きで受け入れるところから始める。旧IEEE binary64、BLAS、trig/log、underflow等の未検証仮定は解消しない。exact rationalを新段に使っても、旧pilotや実機のcertificateにはならない。

旧JSONはbinary64のround-trip decimalである。将来SOURCEはPython JSON floatを`as_integer_ratio()`で正確な二進有理数に戻す。decimal綴りをそのまま有理数とする別targetにはしない。科学parameterのγ/αは契約内の整数・有限小数・分数**文字列**をexactに解釈する。指数式やevalを使わず、2^-24等を勝手に新しい科学許容値へ流用しない。

## 新算術kernelの数学的設計

Python任意長整数と`Fraction`を用い、libm、BLAS、旧shot helperをimportしない。精度pは契約で採択する共通値で、今回null。pを増やして閾値を跨ぐ探索は実装しない。

1. `pi=16 atan(1/5)-4 atan(1/239)`。atanの交代級数は0<x<1で項が減少する。第n部分和と次の符号付き項を加えた値の間に真値が入る。停止条件は次項の絶対値≤2^-p。区間差で係数16/-4を伝播する。
2. sinは`0<x<pi/2<2`でTaylor交代級数。隣接項比`x²/((2n+2)(2n+3))<1`が最初から成立する。部分和と次項付き部分和を両端にする。次項≤2^-p×部分和の絶対値で停止する。小さなxを絶対許容値で0へ潰さない。
3. `log x`（x=2/α_axis>1）をexactな2の冪で`x=2^k y, 1≤y<2`に分解する。`z=(y-1)/(y+1)`は[0,1/3]。

\[
 L_n=2\sum_{i=0}^{n-1}\frac{z^{2i+1}}{2i+1},\qquad
 0\le\log y-L_n\le\frac{2z^{2n+1}}{(2n+1)(1-z^2)}.
\]

同じ式でlog2を囲み、`log x=k log2+log y`を伝播する。採択pから決めた停止条件で、必要なseries長を逐次決定するだけであり、saved結果との一致やshot利得による調整をしない。

4. π区間から`b+=min(pi,abs(b_hat)+r_b)`の区間を作る。qの下端が正ならsinの単調性で`sin(q)`を囲む。A^-はexact rational。正の分母の区間除算でS1式を囲み、上端のexact ceilingを採用する。これにより、式の真のceilingと厳密に一致することを要さず、保守的な十分設計を構成する。下端／上端のceilingも保存し、整数境界で不確かさを隠さない。
5. `b+`自体は真biasの下界ではない。qの区間は採用した上界から作る設計余裕の区間である。`q<=0`はそのrouteの不成立だけを意味する。

この解析的説明は実装の形式検証ではない。V0ではkernelを一度も呼ばない。今後の数値correctness検証を行う場合も別の実施承認の範囲を明示する。

## 統計・位相幾何

独立な±1変数のmeanについてHoeffdingの二側境界は`Pr(|Xbar-E X|>t)≤2exp(-N t²/2)`。X/Yの両座標がt以内なら複素誤差は√2t以下。振幅A≥A^->0、誤差disk半径d=A^-sin(q)、0<q<pi/2なら、原点からdiskへの接線幾何により円周位相差はq以下。総raw参照への距離は円周triangle inequalityで`b+ + q≤gamma`。principal実数差のunwrap保証ではない。

これは[Hoeffding (1963), DOI](https://doi.org/10.1080/01621459.1963.10500830)のbounded-sum不等式を±1に特殊化した、既存Packet S1と受領レビューの式である。今回publisher本文は403で取得できず、新しい全文確認を行ったとはしない。新たな科学レビュー／統計policy採択は行っていない。

## 保証範囲と公平性

72行全てで同じγ・α scope・軸配分・精度p・ceiling規則を使う。family policyでは72×2軸のunion bound、row policyでは各行の2軸のunion boundに限定する。旧保存誤差のcoverageはfailure probability αへ勝手に割り当てない。

物理meanと保存理想meanの誤差上界`d_impl`は未取得。今回SOURCEは0を代入して実機shot保証へ拡張しない。Task Aの理想signal仮説だけを表示する。最適shot、native費用、energy、科学GOは計算しない。

G9で理論的なforecast整数差が現れても、旧絶対contrastが不分解という証拠を保持する。visibilityだけや大きな相対比で科学的有意性を主張しない。fresh IID以外のsampling policyは閉じる。

## Resource見積りと将来の実行準備

入力は454422 bytesのJSON1個、state/vector/matrix/Pauliを読み込まない。保存行108・予測対象72・対応比較48が固定。p共通のπ/log区間を一度だけ計算して全行に共用し、各method行でsinの両端を囲む。行間の並列化は小さなscalar処理には不要と判断し、CPU1 worker/GPUなし、各thread1を候補とする。新benchmarkは行っていない。

最大2 atan＋2 log級数＋144 sin級数、各4096項。power-of-two range reductionとscalar組立も100万bound-step capへ含める。有理数の分子・分母65536 bits、shot整数4096 bits、p=8..256の実装範囲。上限超過は自動緩和せずtechnical STOP。exact rationalの分母成長による停止の可能性がある。少ない行数だけから実際の速度やメモリ適合を保証しない。

候補はaggregate memory2 GiB、input1 MiB、result/log各16 MiB。RLIMIT_AS/FSIZEを将来runで適用し、SOURCEはscienceを実行する子processを作らない。stdoutは最終status、failure tracebackは16 KiB以内。resultはatomic checkpointし、旧結果を書き換えない。wall/CPU time capは設けない。OS kill/OOMでは最後の成功checkpointとexclusive消費記録を残し、retryしない。

SOURCE Sを固定した後に候補Cへ全file SHA256とSを記録する。正式契約・採択C、さらに別途source-bound one-shot承認Aが必要。今回のguard testsは未採択候補の拒否を確認するだけで、未来の認可済み数値経路のcorrectness/runtimeを検証したものではない。

**科学計算0、forecast0、数値correctness0。V0技術準備後mandatory STOP。**
