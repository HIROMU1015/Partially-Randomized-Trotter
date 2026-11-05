# Track D：Reference方式のPRへの適用条件

2026-11-06 JST。既知gadgetとSOURCEの条件照合。新回路設計・非自明な新定理の証明・数値検証はしていない。[baseline](CONTROL_BASELINE_AUDIT.md)と [信号地図](SIGNAL_PRESERVATION_MAP.md)を併読する。

## 1. 原著の回路と適用範囲

[SE-QPE v2](https://arxiv.org/abs/2604.14921v2) §II Fig.1/Eq.2は、順にcontrol・state lane・reference laneを持ち、U=ABに対して分枝演算が

\[
|0\rangle\!\langle0|\otimes(A^\dagger\otimes B)
+e^{i\theta}|1\rangle\!\langle1|\otimes(B\otimes A^\dagger)
\]

となるgadgetである。本書のθはradianで、原著のP(θ)の記号は位相をcycle単位で記す。回路は二つのregister間の **CSWAP cascadeを前後の2層** 持つ。片側一層だけのSWAP testではない。

原著Eqs.3–6はAとBの共通固有基底を十分条件として、連続gadgetでも固有成分の確率を維持しcanonical/iterative QPEのphase-register分布を保存する。Eq.7のbalanced splitは同じbase unitaryの整数powerを半分ずつ両laneへ配る。独立sampleの二つのPR halfは一般にこの条件を満たさない。

原著§IIIはDF basis changeを既に非制御とした対照を使う。比較はtime-evolution部分が主で、QFT・状態準備を除外する。FeMoco価格やethyleneの実機demonstrationをPR/RPE full interrogationの検証と見なさない。原著のgate表はdense diagonal、all-to-all等の条件に依存し、[Simon–Love](https://arxiv.org/abs/2511.13855v1)の対称controlled最適化まで適用した対照の実証ではない。

## 2. 今回の一shot照合

実際のgadgetの分枝は、入力ψ⊗rについて

\[
|a\rangle=A^\dagger|\psi\rangle\otimes B|r\rangle,
\qquad |b\rangle=B|\psi\rangle\otimes A^\dagger|r\rangle.
\]

従って、ユーザー指定の内積恒等式は原著Eq.2の順序に対応する：

\[
\langle a|b\rangle
=\langle\psi|AB|\psi\rangle\langle r|(AB)^\dagger|r\rangle.
\]

**そのshotのfull path** Vω=AωBωが `Vω r=exp(i χω) r` を満たすなら、θ=χωのancilla位相補正により、内積は直接制御の `〈ψ|Vω|ψ〉` に一致する。これは既知gadgetへ内積恒等式を代入した条件整理であり、新しいQPE一般定理ではない。

ここではψがA/Bの固有状態である必要も、A/Bが可換である必要もない。ψ=元Hの厳密基底状態でも、有限PF/random pathの固有状態とは限らないが、一shotのこの恒等式には支障がない。逆に、この一shotの対応から、状態を保持して反復するcoherent QPE全体の分布一致は導けない。

分割は、同じ保存ordered pathの前後をそのままA/Bに対応させる必要がある。A†は **全gateの逆順・実inverse・scalar phaseを含むactual adjoint**。AとBの順序交換、別のrandom halfの生成、時間角だけの反転は代替にならない。分割なしのA=I等も形式上許されるが、balanced depth利益の証拠にはならない。

## 3. pathごとの判定

以下はidealな演算子の構造条件。新しいreference/stateを生成して検証したものではない。

| pathの種類 | reference保存・位相取得の条件 | 現時点の分類 |
|---|---|---|
| 物理的DF由来 `W Z_p W†`, `W Z_p Z_q W†` | Wが粒子数保存Gaussianなら真空を位相倍で保持。Z/ZZの真空固有値は+1。sandwichのW/W†位相は相殺 | 条件を満たすpath族では成立を構造から確認。DFというinterface名だけでは不十分 |
| 同じcomponentのrotation/product | 符号付きPauliの真空固有値は係数符号で既知。rotationは既存eventのsigned angle、productはsign phaseで追跡 | finite-RTEのproduct→rotation順、order phaseを保持すれば取得可能。具体的分子実装のReference correctnessは未実施 |
| identity/scalarを含むpath | identity productの符号、identity rotation、定数scalarを全てχωへ含める | 保存可能。無制御global phaseが消えるからenergy offsetも無料で消せる、とはしない |
| deterministic one-body/DF squared PF | 各complete blockが真空を保持。normal-ordered number blockの真空作用と、抽出scalar/identityの会計を対応させる | 同じ物理Gaussian条件下で成立。元HのEvacを有限random pathのχωの代用にしない |
| random DF RTE、複数event | 各factorが同じ真空rayを保持するので、event符号・product符号・rotation位相を順に合成可能 | 異basis・非可換でもこのray条件は保つ。全lawの具体的回路実証は未取得 |
| deterministicとrandomを結合したordered path | 全factorが上記条件を満たし、scalarを一度だけ会計 | whole interrogationのreference候補。cutの相殺損失・分割点は費用未確認 |
| 一般Pauli path | X/Yを含む単一Pauliや非自明rotationは通常真空を別のbitstringへ移す | 真空方式は一般には不成立。特殊角度・全wordの相殺・別の共通referenceの存在は個別未確認 |
| mixed DF/Pauli、任意custom basis registry | 各実factorのreference保持を別に示す必要 | 未確認。Hamiltonian全体の粒子数保存からは導かない |

[df_rte_tail.py](../../../src/trotterlib/df_rte_tail.py) はI/Z/ZZを生成するが、[builder tests](../../../tests/test_df_rte_qiskit_builder.py)にはRYをbasisとする人工例もある。この例はbuilderの一般性を検査するもので、粒子数保存Gaussianの証明ではない。`DFBasisRegistry`のhash一致はreference保存の資格ではない。

[物理Gaussian producer](../../../src/trotterlib/df_trotter/ops.py)のcost経路は正方のorbital unitaryからnumber-conserving Bogoliubov/Givensを作る。support completionを使っても、同じnumber-conserving Fock liftとして実装されることが条件。一般のpairingを含むGaussianやtaper後のsector-only encodingへ無条件に移せない。真空を表せないencodingで追加registerだけ増やしても、この方式の条件は満たさない。

## 4. 利用可能なphase情報

DFの真空保持componentに限れば、既存eventは `event.phase`、productの `coefficient_sign`、rotationの `unsigned_rotation_angle` を持ち、真空上の作用はこれらの既知scalarから読める。後者は名前に反して係数符号を掛けた角度を返す。orderによる符号も落とさない。複数stepではconstant/extracted identityが既に別会計か、tail内かを元requestに従って区別する。

正規化B、attenuation、sampling probabilityはunitaryのreference位相ではない。位相補正のためにそれらを変更しない。有限random pathはexact exp(−itH)と異なるので、物理Hamiltonianの真空energyだけで全shotを補正する根拠にはならない。

単独Gaussianの実gate分解には真空phaseがあり得る。[既存人工接続記録](../track_a_native_df_qiskit_connection_validation_20261016.md)はそのphaseを明示している。complete sandwichの相殺、splitで露出したbasis、compilerが返すglobal phase、元pathのscalarを区別する。未知E₀・gap・exact biasやcontrolled state preparationは要らないが、既知χωの計算・bit精度・ancilla rotation合成は必要である。

## 5. registerの再利用

一shot内積に必要なのはfull Vωのreference保存である。Aω/Bωそれぞれがrを保持することは、fresh-state一shotでは必須でない。しかしその場合、出力reference laneは元rに戻るとは限らず、coherent reuse・途中resetは別問題になる。

completeな真空保持factorの境界で分ければA/Bも真空rayを保持し、reference laneの理想的な再使用条件を説明しやすい。それでもsystem laneの共通back-actionが消えるとは限らない。RPEでは各shotで元Hの厳密基底状態とreferenceを用意する前提を保ち、前shotのsystemをそのまま使わない。

必要qubitsはstate n、reference n、測定ancilla 1。cat fan-outを使えば更にworkspaceが必要。初期真空の用意と状態準備除外を、CSWAP、reset、維持、routing、phase correctionやcontrolled state-preparationが無料という仮定へ拡張しない。
