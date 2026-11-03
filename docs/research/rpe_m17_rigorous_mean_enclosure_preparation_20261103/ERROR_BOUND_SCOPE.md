# 包含する平均と、残るbridge

2026-11-03 JST。技術準備のみ。実H4の再生成・CDF・包含平均・予算再評価は未実行。

## 評価の両端

| 記号 | 今回の定義・扱い |
|---|---|
| μ_ideal | 厳密な定義式に整合するfinite-RTE平均。今回この入力を新生成するB-2は行わない |
| μ_64 | 保存A/π/τ/B等をexact binary64数として解釈した形式的平均。保存値生成前の誤差を復元しない |
| μ0 | V2の保存complex model平均。元m17固定RESULTのfixed_literalsから同じbinary64へround-tripして使用 |
| ν_logical | 新SOURCE replayのideal basis/gate、保存θ64、exact signed involution・event phaseと、下記53-bit IID lawの平均 |
| ν_cat | ν_logicalに、固定uncompiled builderのπ64加算によるphase差も含めたideal controlled/Hadamard測定の周辺平均 |
| ν_exec | 固定PCG stream、float仮想測定、compiled回路、noise等、採用する実行形態の平均。ν_catからの未評価bridgeを0としない |

第一の包含は**新SOURCE replayを明示したν_cat**が対象。過去の218成分の原payloadを取得したものではない。
保存H/DF因子から再生成する係数・basisの生成誤差そのものを「厳密な分子理論値への誤差」として保証しない。
生成後のbinary64パラメータと確率処理が、今回指定するideal測定対象の入力定義になる。
preparation/tail hashの一致は必須のprovenance検査だが、過去payload・実PRNG・compiled回路の同一性保証ではない。

## Categorical lawを正確に定義する

componentはSOURCE生成p_i64をmath.fsumで足し、float p_i64/sumをchoiceへ渡す。
orderは元m17の保存π64を渡す。両方でNumPy1.26.4の`p.cumsum(); cdf /= cdf[-1]`を再現する。
このCDFは新計算で得るbinary64境界として保存し、exact dyadicとして扱う。
独立一様な整数k∈{0,…,2^53−1}を仮定したとき、

    G_j = ceil(2^53 C_j)/2^53,  p_j^grid=G_j−G_{j−1}, G_0=0

で、searchsorted(side=right)のbucket massをexact rationalで取得する。
末端C_J=1、単調性・非負質量・和1を検査する。RNG、seed、標本頻度は使用しない。
指定ν_catはこのgrid lawを直接使うので、連続categorical→gridの差を未評価のまま加えていない。
保存πをそのまま正規化済みの厳密確率と扱ったμ_64とは異なる対象である。
[NumPy1.26.4 choice](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_generator.pyx#L776-L784)、
[53-bit変換](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_common.pxd#L59-L60)

独立な抽選ごとに新eventを用いるという仮定で、component平均A=Σp_i^grid P_i、

    E=Σ_n p_n^grid (−1)^(n/2)(cosθ_n I−i sinθ_n A) A^n,
    S=(R E^r)F,  ν_logical=φ†S^qφ

を評価する。component相互の可換性は仮定しない。CDFの確率にattenuationが入っているため、Bによる追加除算はしない。
θ64はB-1aが新生成して保存した値を固定し、HP atanや整合π/Bの再生成はしない。

## Ideal gate・sector・stateの意味

basis gateは標準Qiskit RZ、XXPlusYYに限り、生成されたbinary64パラメータをexact realとして解析式で包含する。
SOURCEのlocal matrix/hashもexportするが、そのfloat matrixをexact unitaryと仮定してはいない。
これらのgateは解析式からexactに粒子数を保存する。全粒子数sectorのbit集合とwire範囲を検査し、
sectorへ直接作用させる。数値leakage=0.0を根拠にしない。未対応gateではSTOPする。
signed Z/ZZもsector不変で、U D U†はHermitian involution。対角halfもexact unitaryである。

stateはBの保存complex128ベクトルの**正規化したray**φ=ψ/√(ψ†ψ)を測定入力として定義する。
生の保存ベクトル・fingerprintを変更せず、期待値をψ†Sψ/(ψ†ψ)として外向きに評価し、norm²区間を記録する。
これは無言のデータ再正規化ではなく、ideal量子測定の状態定義である。
過去のfloat仮想信号は生のψによるvdotを用いており、その差や実state preparationをν_execへ接続するbridgeは未評価。
正確な分子基底状態への距離をこの処理で保証しない。

F/Rは固定SOURCEのprimitive生成を再利用する。U†→diagonal→U、Fはblock順、Rは逆順。
constant/identityは元controlled builderの`−δ constant`と`−δ identity`の独立したbinary64角度を各一回適用する。
旧numeric_half_operatorsの合算scalarをそのまま理想回路の厳密値とは扱わない。

### RTEのπ64 phaseを省略しない

df_rte_qiskit._buildはeventの負phaseと負product成分の符号をmath.piとして加算し、
controlled sequence末尾のPhaseGateにまとめる。論理的な−1とexp(iπ64)は厳密には異なる。
K2、identity抽出後、r eventのsequenceにつきπ64を足す回数は最大3r。
u=2^-53、k=3r、γ_k=ku/(1−ku)とすると、exactなkπとの角度差は保守的に

    d_phase,tail ≤ k |π64−π| + γ_k k |π64|,
    η_phase ≤ q d_phase,tail

で覆える。最初のkより少ない回数もこの式が覆う。正の通常値の加算でunderflow/overflowがなく、
0の加算は結果を変えず、両側のtrajectoryはunitaryなので、scalar phase差→期待値差は角度差で上から評価できる。
この外向き上端をν_logicalのRe/Im区間の双方へ加え、ν_catを包含する。
実行時に別のbuilder/summation規約へ切り替えない。
Qiskitのcontrol・basis cancellationはideal操作の数学的意味で扱い、その内部合成やcompiled/noise精度は別bridge。

## 大きなqの誤差伝播

各演算子を`(C,e,N)`で保持し、||A−C||₂≤e、||A||₂≤Nとする。
Cはexact dyadicのAcb行列、e/Nは外向きの上端。
矩形ballの中心への移動で生じる各成分の半径をFrobenius上端へ集約し、元のeへ加える。
積では

    ||AB−CD||₂ ≤ e_A N_B + (N_A+e_A)e_B

に、AcbでC Dを計算したときの新しい丸め半径を加える。
和、確率重み、adjointでも誤差半径を保持する。N=1は上記のideal unitary/確率混合の構造に基づく箇所だけ。
任意の保存float行列へN=1を転用しない。r/qはbinary powerで、使用しない最後のsquareは行わない。
φの両側期待値へのtransferは||φ||=1からe以下となり、最終Re/Imのballへ加える。
midpointのみを真値と見なす段階はない。通常のAcb計算とradius伝播は共に包含計算で、96/160 bit収束とは異なる。
[Arbの包含・区間比較](https://python-flint.readthedocs.io/en/latest/general.html#inexact-numbers-and-numerical-evaluation)

## RPE十分条件とUNKNOWN

各軸の包含I_aから外向きにU_cat,a=sup_{x∈I_a}|x−μ0,a|を得る。
μ0は保存binary64へround-tripし、α/εは元m17のDecimal残余評価と同じ保存decimal文字列のexact rationalとして読む。
固定N/αからs_a=sqrt(2 log(2/α_a)/N_a)も包含し、

    U_cat,a+s_a<ε_a   （両軸）

を確実に判定できた場合のみ、**指定IID・ideal categorical scope**の十分条件を確認したと報告する。
ε境界はstrict。不確かな区間比較はUNDETERMINED、正当な下端で不可能な場合は「この十分条件を認定できない」でありRPE失敗の証明ではない。
ν_execにはU_exec≤U_cat+η_exec←catが必要。以下は今回はnullのままである。

- 固定PCG64列→IID一様モデルの妥当性（数学的な誤差0とはしない）。
- float state-action、clipping、Bernoulli比較の周辺平均差。
- 実際のcontrolled-gate内部合成、compiled回路、backend noise。
- 実state preparation、および元の未保存成分payloadとの完全なidentity。

したがってcategorical条件が確認できても、全体のRPE_tolerance_certified=false、scientific_status=UNDETERMINEDを保持する。
旧numeric_diagnostic_status=INCONSISTENTと旧u_num_bound=nullを遡及変更しない。
PF/RTE/statの予算やN/α/εを変更せず、評価済みの構成誤差を二重に加えない。
全18round、branch/半径・物理state前提、energy成功率、最終費用をこの局所結果で保証しない。
