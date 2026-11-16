# H4 PR-Pauli P0：LD7・8・9のgate超過の切り分け

対象は保存Linear H4、1.00 Å、STO-3G、tol-only DF10、full256とsector(2,2)/36。
B2の固定carrier `9fa9525b3` のLD4/6/7/8/9を照合した。**LD7は非scalar側の不一致、
LD8・9はscalar差と非scalar側の不一致が併存する。** collectorより前の内部係数除去が
原因候補として具体化したが、失われた実係数と演算段階を完全に特定したわけではない。
元の1e-9 gate、辞書、STOP、SOURCEは変更していない。実分子配列load・P0再実行・
signal/reference/normalizer/RPE/compile/価格計算は0。

保存値の転記とJW→辞書identityの照合は[SAVED_EVIDENCE.json](SAVED_EVIDENCE.json)、
必要な最小追加検証は[NEXT_CHECKS.md](NEXT_CHECKS.md)に記載する。

## 1. 保存した絶対残差と判定

全行の`Pauli_tail_full_fro`の閾値は**絶対Frobenius norm 1e-9**。
full Hamiltonianの大きさで割る相対gateではない。次の数値はすべて元resultの転記。

|LD|Pauli_tail_full_fro|保存scalarで置換した対照の残差|sector残差|実行された判定|
|---:|---:|---:|---:|---|
|4|3.519293440081184e-16|3.500616571332133e-16|1.7047496129172293e-16|PASS|
|6|2.734427000071977e-16|2.666743647141904e-16|1.2118136565895058e-16|PASS|
|7|1.1129429626958517e-7|1.1129429626958517e-7|4.8191843932886624e-8|full gateでSTOP|
|8|9.56360360613471e-7|8.941583155721405e-7|4.704421010698378e-7|full gateでSTOP|
|9|1.3255266620668722e-6|1.0327468214877722e-6|6.266760641082842e-7|full gateでSTOP|

比較量は、DF tailを`F`、辞書非identity行列を`A`、JW identity係数を`a`、保存DF scalarを`s_R`として、
`||A+aI-F||_F`と`||A+s_R I-F||_F`である。sector側は前者を保存sectorへ射影して比較する。
全identity metricsはgateループの前に計算・保存されるため、停止runにも対照残差がある。
ただしLD7〜9で正式に実行された最後のgateはfullの1件であり、後続gateを追加判定してはいない。

`full_H_split_fro`はLD4/6/7/8/9で順に5.035231992788507e-15、4.675171583561525e-15、
3.829328891943401e-15、3.73794318539818e-15、0。head/tailのHermiticity・合計sector保存も
すべて直前までPASSしている。保存DFのhead/tail分割そのものの不整合を示す記録ではない。

## 2. 比較演算子の規模：保存済み量と欠測を区別

|LD|保存DF descriptor λ_R|辞書の非identity λ_P|保存DF scalar s_R|JW identity a|保存非identity項数|
|---:|---:|---:|---:|---:|---:|
|4|0.08716784725627223|0.1156975426596128|0.015599379901014688|0.015599379901014686|184|
|6|0.022078920369741746|0.026294457645548053|0.004183531176913642|0.004183531176913646|184|
|7|0.0003129539143720037|0.00042858697343968894|4.5302064712666724e-5|4.530206471266667e-5|184|
|8|8.326693194807275e-5|0.0001374074786820495|1.222662819047054e-5|1.2205423639252299e-5|101|
|9|1.5605698403540042e-7|1.098752230101463e-8|5.193351122457893e-8|0|1|

**比較対象DF tailのfull256 Frobenius norm自体は、result/FIXED_INPUTS/descriptorに保存されていない。**
`tail.hermiticity_fro`等は対称性の残差で、`||F||_F`ではない。λ_Rは保存DF descriptor、λ_Pは
既に生成されたPauli辞書の係数L1で、どちらもこの比較対象のFrobenius normとして代用しない。
λ_Pの大きさだけで真のDF tailの小ささや相対誤差を認定しない。対応するoperator NPZは保存されて
いるが、今回その配列を読み込んだ新しいnorm計算はしていない。従って相対残差の数値は未取得。

保存係数の在庫照合では、最小絶対非identity係数はLD4=1.2376839037083983e-5、
LD6=2.212751878839852e-7、LD7=3.640668108958911e-8、LD8/9=1.098752230101463e-8。
全LDで、**保存された**非zero係数に1e-8未満はない。このことは途中で小さい寄与が消えなかった
証明ではない。特にLD7は項数184を保っていても、同じlabelへの小さい追加寄与が欠ける可能性がある。

## 3. 同じ実行経路と係数選択

全5分割は同じH/state/snapshot binding、P0 `fd111f989b9cd2f052d7189e4b7b6fab10aa1a9f`、
stage1拡張`2e1949a0b66dd8fe086ac7cd3104eb89b4553e38`を使用。
拡張は固定P0のprefixループだけを(2,LD)へ変更する。
`g_matrices[LD:]`と`full.matrices[LD+1:]`は、後者のmatrix0がone-bodyなので整合している。
別LDのdescriptorや辞書で代用している経路ではない。

|段階|固定SOURCEと処理|確認できた係数の扱い|
|---|---|---|
|保存入力|保存weighted-Hermitian completion、全DF fragment保持|loaderのcoefficient_cutoff=0。これは下記の記号演算の閾値とは別|
|one-body記号式|M1 `5e4f0048...:src/trotterlib/o1_m1_matched_split.py`のnested `one_body`|`matrix!=0`を列挙するが、FermionOperatorへの集積は`+=`|
|square/重み/集積|P0 `square_factor_dictionary`：`a*a`、weight乗算、`residual +=`|使用版OpenFermionの乗算にはこの小係数削除はない。集積にはある|
|JW|同関数の`jordan_wigner(residual)`|Fermion termごとの変換をQubitOperatorの`+=`で集積|
|保存JW|`LD*_JW_COEFFICIENTS.json`|ここが最初の保存された記号係数。上記各集積前後の係数は未保存|
|collector|`75fb0ba...:src/trotterlib/pauli_signal_adapter.py:66–105`|追加cutoff=0、imaginary許容1e-12、exact zeroだけ除去。全5分割でJWと辞書の係数・identityが完全一致|
|Pauli行列|同`:153–158`|Qiskit SparsePauliOpで構成しbit reverse。人工例で7.5e-9のI/Zも保持|

使用版OpenFermion 1.6.1の`SymbolicOperator._issmall`/`__iadd__`
（`ops/operators/symbolic_operator.py:67–83,421–438`）は、**各部分和**について`abs(c)<EQ_TOLERANCE`
なら係数を削除する。`EQ_TOLERANCE=1e-8`であり、明示的な`compress()`がなくても働く。
従って「collectorのcutoff=0」は、Fermion/JW構築の全過程が無除去であるという意味ではない。
DFのtol-only=1e-8、OpenFermionの係数閾値=1e-8、P0のoperator gate=1e-9は別の条件である。

LD4/6にはruntime=OpenFermion1.6.1とcoefficient_policyが保存されている。
LD7〜9はそれらを記録する処理より前にSTOPしたため、runtime/policyのfieldが欠測している。
現在の同一専用venvのSOURCEとPASS runの記録は上記機構を支持するが、STOP runの実runtimeを
独立に証明するfieldはない。policy欠測を別policyの証拠、control欠測をcontrol FAILとは扱わない。
LD2 controlの正式bitwise照合はLD4/6ではPASS、LD7〜9では未到達。

## 4. scalarとLD別の結論

`c`はfull_H_split比較で一回だけ加えるHamiltonian定数であり、tail比較の両辺から除外されている。
JW identity `a`と保存DF scalar `s_R`は別fieldとして維持する。恒等項を非identity λ_Pへ含めたり、
P0残差を消すためにscalarを書き換える処理は行っていない。

### LD7

- 保存scalar差は−5.421010862427522e-20。`s_R`で置換しても残差は同じ1.1129429626958517e-7。
  **scalar差だけでは説明できず、非identity表現側の不一致が残る。**
- JW185項→非identity辞書184項は全係数同一。最初に保存結果で確定できる数値不一致は、
  Pauli再構成とDF tailのfull gate。rawJW以前のどの小さい寄与が欠けたかは未確定。
- 内部`+=`の小係数除去は具体的な有力候補。全項数がLD4/6と同じだから無除去とは言えない。
  実LD7の中間係数traceなしに原因確定とはしない。

### LD8

- post-JW identityが保存DF scalarより2.1204551218240576e-8小さい。
  **collectorより前の保存JWでscalar成分の差を既に特定できる。**
- 非identity項は101。scalarを戻した既存対照でも8.941583155721405e-7残るので、
  identity欠落だけの問題ではない。rawJW→辞書で新たな除去は確認されない。
- 中間寄与の閾値除去はscalarと非scalar両方を説明し得るが、失われたlabel/係数の数量的帰属は未取得。
  LD7とLD9の原因を代用せず、LD8の残存factorで検証する必要がある。

### LD9

- post-JWの時点でidentity項はなく、残るのは`Z2`係数−1.098752230101463e-8の1項のみ。
  保存DF scalarは5.193351122457893e-8で、**collectorに入る前の段階で明確なidentity不一致**。
- 保存scalar対照にも1.0327468214877722e-6残る。scalarを書き戻すだけでは元gateを満たさない。
- 最後のDF factor1個の記号展開が最小の切り分け対象。現記録だけではone_body/weighted集積/JW集積の
  どれがどれだけ寄与したかは確定できない。LD9に限って原因を特定してもLD7/8の帰属は別に残る。

LD4・6は同じ経路・閾値で約1e-16のtail一致。入力分割・scalar/collectorの共通仕様の対照にはなるが、
より小さいtailの内部集積で閾値除去が起こらない保証ではない。

## 5. 人工検証と残る説明

[人工6テスト](../../../tests/check_pauli_p0_symbolic_tolerance.py)は固定P0/M1関数・実OpenFermionを
再利用し、one_body `+=`、weighted residual `+=`、JW `+=`それぞれで小さい寄与の消失を再現。
2modeの`w*(n0+n1)^2`では、解析的に非zeroでidentityも1e-8を超えるのに、固定P0経路が空のJW辞書を
返す例を確認した。ライブラリのglobal toleranceは変更せず、明示compressを禁止した状態で再現する。
collector/行列builderは小さい供給係数を保持した。

これは**実LD7/8/9の原因を確定する新科学証拠ではなく、同じコードが持つ機構の人工再現**。
残る説明は、(i)one-body集積の除去、(ii)weighted Fermion集積の除去、(iii)JW部分和の除去、
(iv)その組合せと順序依存、(v)残る実係数化/CAR–JW対応・数値差、(vi)停止runの未記録runtimeの差である。
固定入力binding、正常なDF split、係数をそのまま保持するcollectorは後段の誤用の説明を狭めるが、
中間traceなしに(i)〜(v)を分離はできず、(vi)は新診断のruntime記録も必要。gateを緩和して判定を上書きする理由にはならない。

保存JSON照合4、人工機構6、既存P0人工回帰5の計15テストを実施。詳細は[TESTS.txt](TESTS.txt)。
primary agentは保存資料照合・報告、sub-agentはコード追跡・人工6件を担当し、primaryも全15件を実行した。
独立科学レビューではない。production moduleの変更なし。

## 6. SOURCEと次の引渡し

一次記録は`9fa9525b3:docs/research/b2_pauli_split_p0_20261116/LD{4,6,7,8,9}/`の
`result.json`、`FIXED_INPUTS.json`、`LD*_JW_COEFFICIENTS.json`、`LD*_DICTIONARY.json`。
元carrierはLD4 `c35495565`、LD6 `f3f47f7f9`、LD7 `10baee31e`、LD8/9 `9fa9525b3`。
scientific_sourceは順に`6e4b6bd6a`、`c35495565`、`f3f47f7f9`、`10baee31e`、`10baee31e`。
全て元記録のSTOPと資格を保持する。

B2へ：LD7〜9を使用可能な辞書/configuration/信号へ昇格させない。次の計算が必要なら
[有限追加検証](NEXT_CHECKS.md)の承認範囲を別途確定する。今回P0のretryや修正辞書の採択はしていない。
研究全体への影響は、未準備LDの障害を「内部閾値除去の具体的候補と、未記録の中間証拠」へ絞ったこと。
新しい最適候補、RPE精度、価格、総費用の結論はない。
