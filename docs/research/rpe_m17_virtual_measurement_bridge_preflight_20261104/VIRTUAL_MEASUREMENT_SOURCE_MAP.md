# Ideal categoricalから仮想測定へのSOURCE対応

2026-11-04 JST。静的調査・人工検証のみ。新しい実H4の作用・CDF・包含・sampling・RNG replayは行っていない。
結論と次段は[NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md)、保証の不足は[BRIDGE_EVIDENCE_AND_GAPS.md](BRIDGE_EVIDENCE_AND_GAPS.md)を参照。

## 固定対象と参照の役割

linear H4、隣接1.00 Å、STO-3G、保存legacy DF12、Bの保存physical ground-state vector、
8 qubit・4電子sector dimension70、Partial L_D=3、単一δ=.02、m17/q131072/r32/K2。
extract_identity_phase、coefficient_atol=1e-12、各軸N411、α=.016957089147912167、ε=.15244325851620388。
保存PF/RTE/stat予算・β・位相規約を変更しない。

| 資料 | 元SOURCE / 科学実行SOURCE | 結果公開commitと用途 |
|---|---|---|
| 最新包含 | `18fffce48e33136b91eb6ac5a6f9a5f77a08d36f` | `34840ce9bf81d985b32764d9bc9d77645ed52e59`：[REPORT](../rpe_m17_rigorous_mean_enclosure_execution_20261103/REPORT.md)、原RESULT、export、CDF、operator |
| 候補A run02 | `d0b1da1b04c266a1f7abf8e43966ca6e289a8695` | `80616e6450a63c77d5e76276739a2322111adda4`：[REPORT](../rpe_fresh_iid_measurement_a_run02_20261103/REPORT.md)。m0/q1・m3/q8、r1のみ |
| legacy実装carrier | `4591b8d75a2a6a55c74a51225f0c61648023c8ee` | sampler、ASTで抽出する作用関数、controlled builderの出所。実行許可ではない |
| 候補B | 同報告に記録されたSOURCE | `e22d7718c47358ff650dbc04e8e72f6ab1c8bf41`：[固定request・state](../rpe_fixed_request_measurement_b_20261103/REPORT.md) |
| 元m17 | `128e9a5bf2630cd0333ec5f69b81b25def78eb79` | `cd749066904d3f29b28b03c19ab69370372b157f`：[REPORT](../rpe_m17_mean_validation_execution_20261103/REPORT.md)、保存F/R64 |
| Task A | `4479a1c9cf150250254e2c2650f70fb43841fed1` | `a5304864ec5e1412a36c31574bcb95fb13f80202`：[REPORT](../rpe_m17_numerical_precision_execution_20261103/REPORT.md) |
| B-1a | `9d27f931d6a572210eb206569806f96a46c2d094` | `f766a1f7fc0cd9131a21d0594da242865fe1cb0d`：[REPORT](../rpe_m17_b1_construction_precision_execution_20261103/REPORT.md)、θ64の由来 |
| B-1b/B-2調査 | 科学実行なし | `5632d8b39bed09b56fcb5120034466636e95364e`：[入力調査](../rpe_m17_b1b_b2_assessment_20261103/INPUT_AVAILABILITY.md) |
| RPE V2 | 同資料のSOURCEを参照 | `78c49d0b5f1d71e63b5959e78813237923bbc648`：[保存round行](../pr_rpe_v2_branch_success_20261103/V2_ROUND_RESULTS.csv)。μ0/N/α/β/εの転記元 |

最新包含の[ERROR_BOUND_SCOPE](../rpe_m17_rigorous_mean_enclosure_preparation_20261103/ERROR_BOUND_SCOPE.md)、
[SCIENTIFIC_EXECUTION_PLAN](../rpe_m17_rigorous_mean_enclosure_preparation_20261103/SCIENTIFIC_EXECUTION_PLAN.md)、
[IMPLEMENTATION_REPORT](../rpe_m17_rigorous_mean_enclosure_preparation_20261103/IMPLEMENTATION_REPORT.md)と3 adapterを確認した。
これらの準備時の「未実行」と、その後の結果公開を混同しない。

## 各shotの処理

基準はd0b1da1…の[fresh_iid_adapter.py](../rpe_fresh_iid_measurement_a_preparation_20261103/fresh_iid_adapter.py)、
[run_fresh_iid_a_v2.py](../rpe_fresh_iid_measurement_a_preparation_20261103/run_fresh_iid_a_v2.py)。
`legacy_functions()`はcarrierの3関数のFunctionDefを変更せずAST抽出する。
`legacy_rte()`はcarrierとbyte一致した[rte.py](../../../src/trotterlib/rte.py)を読む。

| 段階 | 実装で行う処理 | 包含結果との対応・差 |
|---|---|---|
| Request | `fixed_axis_inputs()`はm0/q1/N510、m3/q8/N454、r1/K2の両軸を固定 | **m17仮想測定を実行したSOURCEではない**。汎用AxisInputや作用関数がq/rを受け取れることとは別 |
| 成分確率 | `_validate_components`で和をmath.fsumし、p_i/sumをbinary64で作ってchoiceへ渡す | 最新exportの成分順・p64から同じ操作を再現したCDF/grid law。入力identityが必要 |
| order抽選 | `Generator(PCG64(seed)).choice(len(orders), p=order_probabilities)` | m17では保存order vector/τ/θを固定する必要。run02のr1から流用しない |
| component抽選 | 各eventで`choice(len(components), size=order+1, p=component_probabilities)`。復元抽出 | index0がrotation、残りがproducts。重複IDは正常な抽選結果で、event再利用ではない |
| event | θ=math.atan(τ/(order+1))、phase=(−1)^(order/2)。K2のorderは0,2 | 包含は保存されたθ64をexact realとして扱う。式から再生成するB-2は別問題 |
| trajectory | `one_shot`がq×r個を新生成し、長さを確認 | IIDモデルでは各order/component wordを独立に仮定。固定PCG列の証明にはならない |
| state-action | raw ψにF、r events、Rを作用、q回反復しnp.vdot(ψ,value) | 包含はideal gate・normalized ray。float丸め、raw norm、half位相の差が残る |
| 測定 | 下記の検査→clip→確率→別streamのstrict閾値→±1 | 最新ν_catはideal測定の平均。float仮想測定へのbridgeは未評価 |

NumPy1.26.4のchoiceは`cdf=p.cumsum(); cdf/=cdf[-1]`の後、
`self.random`と`searchsorted(side='right')`を使う。
独立な一様53-bit wordを仮定すれば、CDF境界C_jに対する累積質量はceil(2^53 C_j)/2^53となる。
最新の`categorical_laws.json`はこのlawであり、event_probability metadataの積や連続確率そのものではない。
[NumPy choice SOURCE](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_generator.pyx#L776-L784)、
[doubleへの53-bit変換](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_common.pxd#L59-L60)。

## 作用・sector・位相・state

carrierの[_apply_event_to_state / _trajectory_signal](../../../src/trotterlib/rpe_hadamard_failure_validation.py)は次の順序。

```text
value = complex128(raw_saved_psi)
q回:
    value = forward64 @ value
    r event:
        product_component_idsの順に value = P64[id] @ value
        value = cos(theta64)*value - 1j*sin(theta64)*(P64[rotation] @ value)
        value = event.phase * value
    value = reverse64 @ value
z = complex(np.vdot(raw_saved_psi, value))
```

np.vdotは第一引数を共役する。[NumPy vdot](https://numpy.org/doc/1.26/reference/generated/numpy.vdot.html)。
ψ†ψによる除算や状態の再正規化はこの関数にはない。
ideal入力φ=ψ/√(ψ†ψ)とは、exact作用だけでもψ†Tψ=(ψ†ψ)φ†Tφという差がある。
保存stateのnorm検査通過を差0と扱わない。raw bytes/fingerprintは保全する。

旧`restricted_basis`はQiskitの数値local Operatorをfull 2^n空間でcomposeし、
leakageを数値検査後、sector permutationに従ってsliceする。
P64は`(basis * signed_Z_or_ZZ_diagonal) @ basis.conj().T`。
signed componentは係数のsignを含み、eventの(−1)^(order/2)とは別。
leakage許容差の通過やfloat U†U≈Iを、厳密unitarity/sector不変性としない。
最新ideal basisはRZ/XXPlusYYパラメータの解析式から厳密なsector不変性を使っている。

旧halfはdiagonal primitiveのfloat phase累積・np.exp・数値basisの積。
Fはscalar Iからblock順に左作用、Rは逆block順に左作用する。
scalarは`np.exp(-.02j*(constant_coefficient+extracted_identity_coefficient))`。
ideal側の独立したfl(−δ constant)、fl(−δ identity)角をexact realとして適用する規約とは、
合算・乗算・expの丸めを含めて比較が必要。追加scalar補正・B除算はしない。

また、controlled builderは負event phaseや負product符号をπ64として加算しPhaseGateへ集約する。
旧float作用はsigned P64とevent.phaseの乗算。latest ν_catはπ64加算transferを含み、
補助ν_logicalはexact ±1 phaseを使う。この差を無言で落とさない。

## 範囲検査からBernoulliまで

実際の順序は次の通り。定数は`MATRIX_ATOL=1e-10`、`MEAN_ROUNDOFF_ATOL=1e-12`。

1. signalをcomplex化。Re/Im非有限、または`abs(signal)>1+MATRIX_ATOL`なら例外。
2. axisを検査し、cosineはRe、sineは**+Im**をfloatで取る。
3. rawが`[-1-MEAN_ROUNDOFF_ATOL, 1+MEAN_ROUNDOFF_ATOL]`外なら例外。
4. `used=min(1., max(-1., raw))`、変更有無を記録。
5. `p_plus=(1+used)/2`をfloatで計算。
6. 別roleのPCG64からfloat(rng.random())を1個取得。
7. **`random < p_plus`なら+1、それ以外は−1**。

両許容差は入力受理のallowanceであり、全trajectoryの数値誤差上界ではない。
clippingの前にSTOPし得るため、「clipすれば常に±1測定が定義できる」とはいえない。
例外shotを捨てたり置換したりする処理はなく、worker例外は残りjobをcancelして停止する。
run02のclipping0件はその保存有限標本の事実だけで、m17全supportの保証ではない。

## Seedとrandom state

role∈{trajectory,measurement}ごとにSHA256(namespace:role)先頭8 byteを63-bit masterへ変換。
carrierの`_seed_for_shot`がSHA256(master:q:axis:shot)先頭8 byteを63-bit seedへ変換する。
mはdyadic scheduleのqで間接的に区別されるが、**m/r/Kは明示的なkeyに含まれない**。
別のr/K研究までstreamが独立だとは主張しない。既存recipeは変更しない。

各shotのtrajectory samplerとmeasurementはそれぞれ新しいGenerator/PCG64 instance。
workerの_CONTEXTは固定数値payloadを共有するが、shot間の可変RNG stateを持たない。
seedはrequestで決まり、workerの完了順では変わらない。
`check_stream_keys`は対象有限key集合の重複/seed衝突検査であり、独立一様分布の証明ではない。
同じbuild・環境・呼出列でのstream再現性というNumPyの説明もIID保証とは別である。
[NumPy compatibility](https://numpy.org/doc/1.26/reference/random/compatibility.html)。

## 保存されている入力と不足

latest exportは218成分の順/符号/p64、11 basisのdescriptor・param・**local binary64 matrix**、
4 blockのprimitive、ordered sector、raw state、step phase anglesを保存する。
ZIPは7つの包含operatorのexact center/global radius/norm上端を保存する。
**個々のP_i64や完成した数値basis U64は保存していない**。run02のin-memory payload原本も未保存。
SOURCE replayのpreparation/tail hash一致は、過去の未保存行列の完全identityを示さない。

exportにはconstantとextracted identityの元係数が別々には残らず、丸め後のstep phase anglesがある。
これを逆算して旧scalarの元係数が復元できたとは扱わない。
次期案は、保存m17 operators.npzのF/R64を明示的に採用し、latest local matrixから旧数値compositionでP_i64を一度構成する。
これは**既存作用関数を使う新たに固定したm17数値context**であり、run02の歴史的payload同一性の主張ではない。
具体的SOURCEのfreezeとkernel条件の確定は次期技術準備に残す。

原本4件のbytes/hashと固定commitを照合した記録は[VALIDATION.txt](VALIDATION.txt)。
元RESULT・raw・SOURCE・grant/ledger/marker/STOPおよび既存科学statusは変更していない。
