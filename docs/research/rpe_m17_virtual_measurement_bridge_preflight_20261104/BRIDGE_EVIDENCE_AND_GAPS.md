# 測定平均bridgeの既存証拠と不足

2026-11-04 JST。数学的整理・静的SOURCE対応・人工入力のみ。
対象物理条件とSOURCEは[VIRTUAL_MEASUREMENT_SOURCE_MAP.md](VIRTUAL_MEASUREMENT_SOURCE_MAP.md)。
新H4のnorm・誤差・平均・残余予算を評価していない。未評価bridgeを0にしない。

## 3つの確率モデルと定義域

| Model | 定義 | 現状 |
|---|---|---|
| I：ν_cat | latest SOURCE_REPLAY_GENERATED成分/basis、各抽選の独立一様53-bit word、固定CDF、ideal未compile gate、保存ψのnormalized ray。builder π64 phaseも含む | 既存包含で局所十分条件を確認済み |
| II：ν_virtual,IID,a | Iと同じIID categorical wordに加え、独立一様53-bit measurement wordを使用。固定したfloat64 state-action、範囲検査、clip、p計算、strict Bernoulliを使う±1測定の期待値 | m17数値contextと丸めkernel条件の固定、誤差上界、全supportでの完了が未確立 |
| III：固定seed PCG64 | namespace/q/axis/shot/roleでseedを固定した具体的な決定的列・有限shot出力 | 再現性検査の対象。数学的IID分布平均と同一視しない |

Model IIの範囲検査はpartial algorithmを作る。正の確率を持つtrajectoryで例外が起き得るなら、
そのままでは無条件の±1変数もν_virtual,IIDも定義できない。
失敗shotへの値の割当、失敗条件付き分布、retryによる置換は別モデルであり、無言で採用しない。
したがって**平均誤差上界だけでなく、全supportで既存検査を通過する根拠が必要**。
平均的な誤差が小さいことやrun02の有限標本での正常完了は、この根拠を与えない。

Model IIIは固定すると確率変数ではない。seed自体へ分布を置けば新しい有限ensembleを定義できるが、
IID uniform53 word列と等しいとは限らない。seed衝突なし・別role・経験的PRNG試験はその証明にならない。
今回PCG64の完全なIID性を証明することも、新しい経験的検査も目標としない。

## Bridgeごとの証拠表

下表の「影響」は保存N/α/εの局所十分条件への影響。実測値や旧診断差を上界と読み替えない。

| Bridge / 比較対象 | 既存SOURCE・証拠 / 静的に確認した点 | 追加評価と上界の前提 | 条件への影響 / 状態 |
|---|---|---|---|
| State：raw ψ†Tψ と φ†Tφ | 旧_trajectory_signalはraw vdot、latest ball_adapter.expectationはψ†ψで除算。latest RESULTにnorm²区間保存 | n²=ψ†ψの保存区間を使い、exact unitary Tに対する差≤\|n²−1\|。float dot/actionはOperator項へ分離 | state項を1回加算。`REQUIRES_NUMERICAL_BOUND`（保存区間はあるがbridge未算出） |
| Operator：ideal logical T とfloat action | F→r event→R、products→rotation、signed P、event phaseの順を確認。latest ideal basis/halfの包含、元m17 F/R64がある | P_i64構成、idealとの差の包含、trig出力誤差、matrix-vector/scale/subtract/vdotの一様丸めbound。kernel・rounding・subnormal条件が必要 | q/rで増幅。旧Acb powered radiusはFP trajectory誤差ではない。`REQUIRES_NUMERICAL_BOUND` |
| Phase：ν_cat と ν_logical | ideal builderのπ64加算と旧exact signed phaseが異なる。latest RESULTにphase transfer上端あり | 同じgate/θ/符号/word lawでcoupleすること。numeric halfの合算scalar差はOperator項 | generic cat接続にはphase transferが必要。common-reference法なら再加算不要。`SUPPORTED_WITH_FIXED_ASSUMPTIONS` |
| Probability：clipped t とfl確率p | `p=fl((1+t)/2)`、有限・受理後t∈[-1,1] | binary64 round-to-nearest-even、通常の基本演算で\|2p−1−t\|≤u、u=2^-53。下記の解析導出 | 一shotの周辺meanへ≤u。`SUPPORTED_WITH_FIXED_ASSUMPTIONS` |
| Clipping / domain：ideal c とΠ(raw) | magnitude検査→座標検査→clip。run02の調整0は有限標本の記録 | c∈[-1,1]ならprojectionは非拡大。別途全supportの検査完了を保証する必要 | clipping誤差をaction誤差へ二重加算しない。projection性は`SUPPORTED_WITH_FIXED_ASSUMPTIONS`、完了は`UNRESOLVED` |
| Bernoulli：p と独立一様53-bit strict閾値 | `random<p`で+1。NumPyの53-bit変換とsource比較順を確認 | measurement wordをtrajectoryと独立一様とする。m_grid(p)=2ceil(2^53p)/2^53−1 | 0≤m_grid−(2p−1)<2u。`SUPPORTED_WITH_FIXED_ASSUMPTIONS`。固定PCG64には転用しない |
| PRNG：Model III と Model II | role別master、q/axis/shot seed、shot毎新Generator、有限key検査。旧run02は1,928 shotの診断 | 数学的IIDを追加仮定するか、別の妥当性の根拠・モデルが必要。衝突検査では不足 | HoeffdingのIID前提の適用scopeを限定。`REQUIRES_PROBABILISTIC_ASSUMPTION` |
| SOURCE identity：latest export と過去payload / 新m17 context | latestは新replayであり、元payloadではない。旧A contextはr1、m0/m3のみ | 保存F/R64とlatest local matrixによる新m17 contextを明示固定。過去payloadの同一性は欠落入力を補わない限り未保証 | 新定義のModel IIを評価できても歴史的run02の昇格にはならない。`UNRESOLVED`（旧payload）、`SUPPORTED_WITH_FIXED_ASSUMPTIONS`（新contextの定義案） |
| compiled/noise/実state preparation | latest scope外、今回も未評価 | 別研究scopeと許可が必要 | 今回のModel II検証では`NOT_APPLICABLE`。全体保証へのbridgeはnull |

## 確率計算と有限gridの解析上界

受理されたbinary64座標rawをclipしたtに対し、u=2^-53とする。
binary64 round-to-nearest-evenではs=fl(1+t)の絶対誤差はu以下。
t≥−1で、最小の正の1+tは2^-53なので、この範囲のs/2はunderflowせずexact。
従ってp_fl=s/2に対して

    |(2 p_fl−1)−t| ≤ u.

M=2^53、kが{0,…,M−1}上の独立一様整数のとき、strictなk/M<pの個数はceil(Mp)。
よって

    m_grid(p) = 2 ceil(Mp)/M − 1,
    0 ≤ m_grid(p) − (2p−1) < 2/M,
    |m_grid(p_fl) − t| ≤ 3u.

p=0/1もこの式で扱える。これは**受理された1個の条件付き測定に対するデータ非依存の解析bound**。
q×r個のeventごとにBernoulli測定するSOURCEではないので、この3uをq×r倍しない。
SOURCEの53-bit変換は[NumPy1.26.4](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_common.pxd#L59-L60)を参照。
FP演算modeの前提や、入力が受理される条件をこの式で証明したことにはならない。

c∈[-1,1]に対して|Π(raw)−c|≤|raw−c|。
従ってraw signalの誤差を直接boundすれば、別の「clipping量」を足す必要はない。
ただしabs(signal)とraw座標の検査を通ることは別途必要。平均誤差boundから全support完了は導けない。

## 長いstate-actionをsamplingせずboundする案

丸めを含む作用は非線形で、`E[fl(T_ω v)] = fl(E[T_ω]v)`とは限らない。
旧mean_adapterの平均operatorをq乗するだけではfloat trajectory平均の証明にならない。
利用できるのは、状態に一様な**誤差majorant**を先に作り、そのmajorantをlawで平均する方法。

ideal unitary A_jに対し、全ての到達し得るvについて

    ||fl(Ahat_j v) − A_j v|| ≤ t_j ||v||,  β_j=1+t_j ≥ 1

を示せれば、ideal raw作用とのstate差は||ψ||(∏β_j−1)以下。
最後のvdot丸めをd||ψ||||vhat||でboundし、n²=ψ†ψとすると、normalized logical signalとの差は

    E |z_fp−z_logical| ≤ |n²−1| + n²[(1+d) B−1] + A_underflow,
    B = E ∏β_j.

underflow等の加法誤差が必要ならA_underflowへ外向きに伝播し、無言で0にしない。
共通のcomponent増幅β_Pとorder nごとのrotation複合処理β_rot,nを使う場合、

    G_n = β_rot,n β_P^n,
    B = [β_F β_R (Σ_n p_n^grid G_n)^r]^q.

β_rot,nはcos/sinの出力差、rotation matvec、scale/subtract、event.phaseまで含む。
n productsだけを数えたりrotationを1回の理想matvecとしたりしない。
このfactorizationはIID orderと状態一様majorantを使うもので、FP programの線形性を仮定していない。
component依存のより鋭いboundは、必要な場合のみ次期SOURCEで事前固定する。

matrix準備丸めは、完成したP64/F64/R64をexact dyadicとし、ideal包含との差をnormでboundできる。
runtimeのcomplex128 @/vdotは、実際のBLAS kernelの演算・加算順・FMA・丸め・subnormal/overflow条件が必要。
例えばrealificationと|A|によるγ_k boundを使うなら、**kの根拠となるkernel条件を先に確定**する。
NumPy使用や人工テストPASSだけから任意のγ係数を保証値としない。ここは未実装・未確立。
2つの保存θに対する実際のmath.cos/sin出力は、将来一度取得しArbで比較すれば、libm全域の正確丸めを仮定せずに済む。

完了保証にはBの平均値ではなく、全supportを覆う積majorantやさらに鋭いsignal範囲が必要。
`abs(z)`自体の浮動小数点評価と実際のbinary64閾値も含めて両検査を検証する。
このuniform boundが不十分なら、平均budgetに余裕があってもModel IIの無条件保証は未成立。

## 保存十分条件への接続とphaseの扱い

各軸で一般に

    |ν_virtual,IID,a − μ0,a| ≤ U_cat,a + η_virtual←cat,a

が必要。ν_logicalを共通参照とすれば

    η_virtual←cat,a ≤ η_action/raw,a + 3u + η_cat←logical,

と書けるが、U_catには既にcat←logicalのphase transferが入っている。
より直接には、保存logical interval I_L,aから将来新たに外向きに

    U_L,a = sup_{x∈I_L,a} |x−μ0,a|,
    U_virtual,IID,a ≤ U_L,a + η_action/raw,a + 3u

を構成する。これなら同じphase transferを三角不等式で往復する保守性を避けられる。
**U_catからphase上端を単純に引いてU_Lを作ってはいけない**。
保存logical intervalの利用・新U_Lの算出は新科学再解析であり、今回は未実行。
過去のν_cat/U_cat/phase boundを変更したり、その判定を取り消したりしない。

完了・数値・law identityの全てが確立したscopeでのみ、保存sの包含と

    upper(U_virtual,IID,a + s_a) < lower(ε_a)

を両軸でstrictに比較する。境界一致・区間不確定・bridge欠落は認定不可。
上界が大きくて不成立でもRPE失敗の証明ではない。今回は`U_exec=null`のまま。
Model IIで成立してもModel III、全18round、最終energy・総費用の保証にはならない。

既存`CONDITIONAL_IDEAL_CATEGORICAL_SUFFICIENT_CONDITION_CONFIRMED`、
全体`RPE_tolerance_certified=false` / `scientific_status=UNDETERMINED`、
旧`numeric_diagnostic_status=INCONSISTENT` / `u_num_bound=null`を保持する。
