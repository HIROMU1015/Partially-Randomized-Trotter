# A：SOURCE→meanと18roundの保証範囲

基準`3258a89f225d9530fba38e9d2343bc0165dd6d7d`、SERVER 2、2026-11-06。
条件・保全・終了は[共通REPORT](REPORT.md)を参照。

**保存HamiltonianのHermitian行列表現、供給される厳密ground、理想native gate、fresh independent bitsというモデルでは、全36条件の十分条件が成立した。元Hとその保存行列表現の厳密な包含まで無条件に認定したわけではない。**
[採用SUMMARY][summary]の`CLOSED_FIXED_MATRIX_IID_IDEAL_GATE_MODEL`を、このモデル範囲で読む。
元H参照の表現誤差の新しい包含は未取得なので、その外側の保証は[共通status][joint]でUNQUALIFIEDとした。
既存H4入力照合PASSを引き継ぎ、照合・SCF・DF fit・ground solveは再実行していない。

## 1. 保存SOURCEから生成されるoperator

保存された13個のD blockと、tailの218 componentを再利用した。
source basisはRZとXXPlusYYだけで、数学的にparticle数を保存する。
登録gateの解析式を192-bit区間で評価してbinary64中心と係数誤差を取り、70次元number sector内で局所row更新を行った。
ここで70次元を使うのは数保存する**SOURCE**だけであり、native中間gateを射影してleakageを消す処理ではない。
native誤差には前段の全joint-spaceに有効なcompiler A上界を使う。

各basis Bの評価誤差e_Bを、局所係数誤差とcomplex row更新の前進誤差から伝播した。
Dのdiagonal energyは保存eps/eta/lambdaの正確なbinary64入力から区間で作る。
`G=B diag(E) B^dagger`と保存block Hamiltonianの差を、matrix積の丸めとe_Bを含めて評価した。
生成されたdiagonal primitiveのphaseについて、各占有列で

`phi = p.global_phase - sum(theta_Z Z/2) - sum(theta_ZZ ZZ/2)`

と`-t E`との差も直接区間評価した。scalarをprojectiveに消していない。
Y4の正負stage、両half、実inverseを同じSOURCE規約で扱った。
最大のD generator差上界は`5.916718431625832e-11`。

tailは使用される7 basisへまとめ、保存component確率を正確なrationalとして正規化した。
各signed involution `P_i=B_i Z/ZZ B_i^dagger`と`A=sum p_i P_i`について、source Aと保存tail/lambdaの差上界は`5.1717094511696015e-12`。
identity componentは0。有限supportをtrajectoryとして全列挙していない。
[SOURCE generator結果][generators]。

### finite-RTE meanとnormalization

K2の1 eventは最大2個の積と1個の回転である。
componentを独立に引くため、回転と積の期待値を因数分解でき、meanは

`p0 (cos(a0) I - i sin(a0) A) - p2 A^2 (cos(a2) I - i sin(a2) A)`

となる。各P_iがHermitian involutionであることが根拠で、Pauli成分どうしの可換性は仮定しない。
canonicalな正確な角と重みでは、これは`P3(-i tau A)/B`に一致する。
`B=sqrt(1+tau^2)+(tau^2/2)sqrt(1+(tau/3)^2)`。
保存floatのorder確率をrational正規化した値とcanonical重みのL1差、保存atan角との差を区間で会計した。
Taylor residualには`exp(abs(tau)) abs(tau)^4/(24 B)`を使う。
raw meanとscaled ideal factorはいずれもcontractionなので、局所残差をr・qへ加算できる。
SOURCE basis/angle生成とfinite-RTEを同じ項にまとめる一方、前段のevent pi累積誤差とcompiler誤差は別に保持した。

量子shotのproposalは既存`integer_cdf`・`proposal_budget`を再利用した。
入力float確率を正確なrationalとして正規化し、floor-CDFのL1誤差をdraw数へ加算する。
最長S2 roundは88 bits/draw、最大16,777,216 drawのモデルで、raw proposal disk上界は約`3.343e-17`。
これはfresh independent bitsのモデルに対する境界であり、BのNumPy条件付き価格samplerと同じ実装だとは扱わない。
quantum shot自体は実行していない。

## 2. PF・数値diskの評価を置き換えた範囲

前段の経験的`numerical_disk`を包含証明へ格上げせず、新しい明示上界を求めた。
保存Hermitian blockと保存eigenpairを使い、新規対角化は行わない。

- `delta=||V^dagger V-I||`、`R=HV-VD`を丸め込みで評価する。
- polar unitary Qに対して`||V-Q|| <= delta/(1+sqrt(1-delta))`を使う。
- `||H-QDQ^dagger||`をR、delta、polar距離から押さえ、unitary exponentialのDuhamel boundで伝播する。
- exp位相は区間で評価し、matrix積・反復二乗・state expectationの丸めを加える。

complex matrix積の前進誤差モデルは、n項の積に対する保守形`gamma_(8n+4) ||A||_F ||B||_F`。
通常のIEEE binary64 round-to-nearest・通常のcomplex arithmetic・overflow等なしを前提とする。
normと係数を区間で外向きに評価する。これは実行binaryの形式検証や第三者CI証拠ではない。
[spectral certificate][spectral]。

理想PF kernelはrに依存せず、同じideal-tail evolutionを使うため、その反復二乗をcacheした。
rが変わる有限RTEのmean・normalizer・生成誤差は各recipeで別に評価した。
q1の観測差を無条件にq倍する処理ではない。

## 3. 保存groundとalias

保存physical Hamiltonian行列とそのHermitian表現を数値参照として使用した。
保存stateを正確に正規化したvectorと、その計算中心の差を保持した。
Rayleigh値・residualを丸め込みで評価し、保存gapの数値をそのまま証明に使わず、cut位置の**検証付きLDL inertia**を調べた。
LDLの残差と逆行列の検証から、congruence後の摂動がDの各符号を変えないことを確認した。
これはground eigensolveの再実行ではない。
cut `-2.050072341074671`未満の固有値が正確に1個と分かるため、Templeの境界からground energyとstate近似誤差を押さえた。

| 項目 | 結果 |
|---|---:|
| ground energy区間 Ha | [-2.166387448635427, -2.1663874486340977] |
| 保存stateとgroundの距離上界 | 1.4551928902900854e-11 |
| inertia perturbation上界 | 4.3589e-9 |
| Dの固有値絶対値の下界 | 0.12412789 |
| Gershgorinによる全sector spectrum絶対値上界 Ha | 3.0552911673129084 |

`delta0 ||H||+.4 < pi`が成立する。
したがって、この固定行列モデルではgroundだけでなく当該sector全spectrumについて、初期phase window `[-pi,pi)`の端点から`.4 rad`のmarginを確保できる。
decoderへE0を渡してaliasを選ぶ方法ではない。
[GROUND_ALIAS.json][ground]。

## 4. 全18roundとshot

ideal PF signalの位相をground energy区間へ照合し、数値評価diskを含むPF位相上界を求めた。
半径にも同じ不確かさを使い、raw attenuationを一度だけ反映した。
SOURCE/finite-RTE、compiler、event phase、finite-bit proposalをraw diskとして加え、残る正の位相幅を統計へ渡す。
旧PF/input/numericalの位相幅を再度足して二重計上することはしていない。
各方式の18round・36軸のalphaは厳密な`1/720`、和は`.05`。
18roundの分解能と`3 beta<pi`は既存条件を維持する。

| 最長round m17 | S2 LD3 | Full Y4 LD12 |
|---|---:|---:|
| PF数値評価disk上界 | 1.8158312e-5 | 1.4347754e-4 |
| PF位相上界 rad（数値幅を含む） | 0.014006604 | 0.000146681 |
| SOURCE生成＋finite-RTE operator上界 | 2.9002037e-7 | 1.2725611e-6 |
| 新raw半径下界 | 0.572687235 | 0.999851023 |
| 残る統計位相下界 rad | 0.385986831 | 0.399847816 |
| 各軸十分shot数 | 626 | 193 |
| 18round両軸合計 | 8,598 | 6,914 |

全roundで正半径・正統計予算を確認した。[全36行TSV][table]、[JSON][rounds]。
Full Y4の保存6,912 shotは今回の保守上界の十分数6,914とは区別する。
2 shot差は実際の失敗を観測した結果ではなく、保証に使う数値幅を広げた結果である。

## 5. 技術検査・未資格・STOP

新しいSOURCE basis評価器を2つの保存basisについてQiskitの256次元SOURCE operatorへ照合した。sector matrix差は約`2.0--2.3e-15`で、評価器の上界より小さい。
exact rationalで計算した小さいcomplex積がBLAS前進誤差幅に入ること、既知の対角行列でinertia符号が一致することも確認した。[技術検査][checks]。
既存44ケースと旧H4 q1 full-wrapper suiteは再実行していない。

初回開発実装は、正規化等の微小な積の項を式へ残す補正が必要となった。
[初回archive][initial]にSOURCE・結果・STOPを保存した。
補正後は別`verified/`領域で同条件を評価し、そちらだけを採用する。新しい物理条件・価格標本の追加ではない。

**成立したのは固定保存行列モデルの十分条件。**
保存行列と元Hの連続入力・基底変換・行列組立ての表現誤差について、旧入力照合PASSを新たな厳密包含へ格上げしていない。
この外部対応を仮定しない無条件の元H保証はUNQUALIFIEDのまま。
その対応を保証する必要がある場合、次は保存入力の表現誤差だけを対象に範囲を定める。SCF/DF/groundを再生成する必要があるという結論ではない。
価格結果から成功保証を逆に認定していない。
**Aは完了しmandatory STOP。** 次の実行はユーザーの別途許可後のみ。

[summary]: ../../../artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/SUMMARY.json
[generators]: ../../../artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/SOURCE_GENERATORS.json
[spectral]: ../../../artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/SPECTRAL_CERTIFICATES.json
[ground]: ../../../artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/GROUND_ALIAS.json
[table]: ../../../artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/ROUND_TABLE.tsv
[rounds]: ../../../artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/ROUNDS.json
[checks]: ../../../artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/TECHNICAL_CHECKS.json
[initial]: ../../../artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/initial_attempt.tar.xz
[joint]: ../../../artifacts/track_c_guarantee_cn_price/2026-11-06/JOINT_STATUS.json
