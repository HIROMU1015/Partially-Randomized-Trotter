# O2-A2：全attemptの測定・scale・費用会計

2026-11-08 JST。対象は[今回具体化した2-unitary LCU](A2_EXECUTABLE_ACCESS.md)。数式は理想gateモデルの既知恒等式に対する記号的会計で、実H4のvariance/N/価格を評価していない。

## 1. Outcome、成功率、分母

`d²=||Ψ_f−Ψ_c||²`、`S=1+ζ²d²`、`β=ζ+√(1+ζ²)`、`p=S/β²`。全attempt outcomeは、成功時β²z、失敗時0。rangeは

\[
 |T|\le R=\beta^2\lambda_+,\quad
 E[T|\omega]=\Delta_\omega=f_f-f_c.
\]

success状態を正規化して測るzの平均はΔ/Sであり、そのままΔとしない。成功だけで平均するならSの取得/補正と全attempt数が別途必要。本adapterはp/Sが未知でも全attempt zero-fillにより理想不偏性を保つ。成功確率を推定する追加測定や比推定は使わない。

common exact real weight wがあればT,Rをw,|w|倍する。これはpath条件付き差分の接口であり、path依存wを用いる階層ではbase/coarse側のweighted marginalも別途対応させる必要がある。今回の階層候補はraw weight=1に限定する。異なるfine/coarse weightなら追加項/測定が必要で、今回の接口の範囲外。状態・真signal・E₀・gap・normを古典decoderへ渡さない。

## 2. 全attemptの二次・四次moment

`O_X²=O_Y²=I` とΨ_f/Ψ_cのnorm1により、2×2の積を整理すると

\[
 E[T^2|\omega]=\beta^2\langle\chi|\widehat O_b^2|\chi\rangle
              =\beta^2(d^2+\zeta^{-2}),
\]

\[
 \operatorname{Var}(T|\omega)=\beta^2(d^2+\zeta^{-2})-\Delta_\omega^2.
\]

これは当該LCUの直接の会計確認で、新しい一般Multilevel定理ではない。全pathでは `E[T²]=β²(E[d²]+ζ⁻²)` と全分散を使う。path平均の分散を実outcomeの分散に置き換えない。

四次momentも成功率を戻すと `E[T⁴|ω]=β⁶⟨χ|Ô_b⁴|χ⟩`。常に `E[T⁴]≤R²E[T²]` が使える。実PRの二次/四次momentは取得しておらず、qDRIFTのlevel依存評価を移送しない。

**同じfine/coarseでもnoise床が残る。** d=0、Δ=0の場合にp=1/β²だが、`E[T²]=β²/ζ²=(1+√(1+ζ⁻²))²>4`。小norm observableだけを見たshot削減は、LCU scale/成功率の補正後には成立しない。

## 3. 強いbaselineとの記号的比較

| 方式 | 条件付き実outcomeとvariance | 量子アクセス・費用上の注意 |
|---|---|---|
| fine直接 | ±1、variance `1−f_f²` | 通常controlled U_f、一回。必要なら既存層別化を使える |
| 独立fine/coarse差分 | Y_f−Y_c、variance `2−f_f²−f_c²` | 2つの通常Hadamard測定。base/coarseも有料 |
| metadata CV/層別化 | 既存の条件付き測定noiseを残す | train/配分/取得費を含める。旧SCV pilotを再解析しない |
| 単純coherent分岐 | `( |0⟩Ψ_f+|1⟩Ψ_c )/√2`、observable `2Z⊗O_b`。M2=4、variance `4−Δ²` | selectorを追加した明示fine/coarse SELECT。両controlled gate列を価格へ含める |
| 今回のscaled LCU | 上記M2、range R | 分岐に加えてherald/PREP、b操作、observable基底回転が必要 |

今回LCUは **同じconditional meanのcoherent分岐に対してM2が大きい**。β²/ζ²>4かつd²≥0だからである。従って当該LCUに、測定noise削減を理由とする追加利益はない。これは任意のdilation/encodingに関するno-goではない。gate最適化・native価格の数値比較はしていないため、全compiled資源の一般的な大小も断定しない。

n=0の同一eventをmetadataで省略できる場合、その同じ省略は全対照にも認める。階層の稀な差分だけによる利益と、A2の量子測定結合の利益を混同しない。coarse K0 full operationは決定論D/basis/wrapperを含み、無償のreferenceでもない。

## 4. Attempt費用とbase費用

回路が実際に適用するのは、fineとcoarseそれぞれ一つの **二重制御system gate列**。selector branchの確率でそれらのgate費を平均して片方だけ課金しない。両列は同じunitary回路内に置く。

adapterの記号的な1-attempt価格は、元state準備を研究範囲として除外した上で

\[
 C_{\rm LCU}=C_{C^2U_f}+C_{C^2U_c}+C_{H_a}
   +C_{\rm PREP}+C_{\rm PREP^\dagger}+C_{CRY_b}+C_{\rm readout}.
\]

新しい追加処理を状態準備除外に含めない。logical ancillaは旧aに加えb,hの2個、計n+3 qubits。native loweringで必要になる補助workspaceと合成精度・RZ/T/CX/depthはUNKNOWN。古典path生成、metadata写像、記録、scale/weight計算、cacheも別費用。

本adapterはreadoutを全attemptへ適用する。別途feed-forwardを実装した場合だけ `C_prefix+p C_success_readout` と書ける。失敗でもprefixは有料。成功試行のみ繰り返す方式なら全attempt数を数える必要があり、自動retryを導入しない。

全体費用は `N₀ E[C₀]+Σ_l N_l E[C_l]`。base/coarse推定と各levelの独立標本が必要で、同じcoarse測定を独立性の検討なしに重複利用しない。必要Nと単価・古典取得費のすべてを揃えて初めて資源改善を主張できる。

## 5. 有限confidenceへの条件付き接口

独立な全attempt outcome、既知のvariance上界V_lとabsolute range R_l、残余幅s_l>0、失敗配分α_lが資格化されれば、既存Bernstein型の十分条件

\[
 N_l\ge\left\lceil\log(2/\alpha_l)
    (2V_l/s_l^2+4R_l/(3s_l))\right\rceil
\]

へ接続できる。baseと差分の誤差幅/数値biasを座標誤差へ合成し、全αを管理する。ここで数値ε/α/ζ/Nを採択・計算していない。既存の弱いrange-only対照だけを使って成功扱いにしない。

MLMCのleading variance/cost目安 `Σ√(V_l E[C_l])` だけで、range/bias/base/有限Nを含む十分費用を保証しない。今回LCUには前節のM2負担が既にあり、実H4で追加費用比較する積極的根拠を残せなかった。

## 6. 有限精度・gridの未取得項

binary64のPREP/RY/制御化、λとβ²のdecoder、測定基底、入力state、追加制御gate合成、三bit joint readout/gridの誤差は別項目。旧N2のstate保証は元stateの参照として保持するが、旧single-control engineの総誤差をこの測定へ移送しない。

例えば成功Kraus A=K/βと実装Âがともにcontraction、`||Â−A||≤η_A`、observable誤差η_O、classical scale g=β²の誤差η_gを別途保証できれば、通常の積の誤差評価から座標biasを

\[
 \eta_g\|\widetilde O\|+\beta^2(2\|\widehat O\|\eta_A+\eta_O)
\]

等で会計できる。入力state誤差、outcome丸め、joint grid誤差もさらに必要。ideal joint lawとのtotal variationがη_TVならbounded outcomeの項は `2R η_TV`。これらのηは今回 **UNKNOWN** で、0を代入しない。旧53-bit一測定の条件付き項を三bit readoutへそのまま流用しない。

## 7. 結論

物理アクセスは明示できたが、**この単純なscaled LCUは測定noise削減の機構を実現しない**。新しい非自明なdilation理論やoracleで補わず、資源改善の科学候補としては保留する。別の取得構造や有利なscaleの明確な根拠がないままH4のBorn/compile/shotへ進める理由はない。O2全体の正式終了・一般NO-GOは採択していない。
