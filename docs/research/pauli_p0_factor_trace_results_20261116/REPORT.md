# H4 LD7・8・9：保存演算子再解析と実factor係数追跡結果

**この固定H4入力でのP0 gate超過は、OpenFermion `+=`の途中微小係数除去で説明できる。**
LD7のgate超過を支配するのはJW集積。LD8・9はweighted Fermion集積とJW集積の両方が寄与する。
最初の経路差は全LDともone-body構築だが、その微小差だけではgate超過を説明できない。
元のP0判定は全3分割ともFAIL/STOPのまま、本番辞書の採用はしていない。

対象：Linear H4/1.00 Å/STO-3G、保存tol-only DF10、保存weighted-Hermitian completion、
同H/state/snapshot、full Fock 256次元。P0の演算子診断でありRPE/N/価格の評価ではない。
承認範囲は[SCIENCE_SCOPE.md](SCIENCE_SCOPE.md)。旧調査は`49a6d7ade`、
科学SOURCEは**`655e74212`**（全SHAは[run01/EXECUTION.json](run01/EXECUTION.json)）。
元P0 carrier `9fa9525b3`と別のfresh `run01/`で1回実行した。

## 1. 実行範囲と照合

§10.5照合では元LD7/8/9のoperator、raw JW、dictionary、FIXED_INPUTS、STOP結果を受入。
関連最新成果`133265772`等はLD4/6 intake準備であり、同じ係数追跡の完了成果はなかった。
LD4/6の追加実計算は今回行わない。元P0・global tolerance・gateは変更しない。

- 保存operator再解析3件を先に実施。保存JWからの再構成は全3件で保存Aとbitwise一致。
- 保存factorを元save-only loaderで1回復元。新SCF/DF fit/state生成は0。
- LD9（残存factor index9の1個）→LD8（8,9）→LD7（7,8,9）の順に各1 trace。
- tracked defaultのJWは、全分割で元raw JWの全項・挿入順・係数と完全一致。
- runtimeはPython3.11.0rc1、NumPy1.26.4、SciPy1.14.1、OpenFermion1.6.1。
  EQ_TOLERANCEは開始時・終了時とも1e-8。局所subclassとJW関数の局所globalsを使い、libraryのglobalを変更しない。
- 全段階正常終了、process exit 0、実行本体5.38秒。P1/mean/reference/normalizer/sampling/compile/Q3/N/Gは0。

実行順・所要時間は[PROGRESS.jsonl](run01/PROGRESS.jsonl)、親processの終了は
[PROCESS_EXIT.json](PROCESS_EXIT.json)、元SOURCEと入力照合は[PREFLIGHT.json](run01/PREFLIGHT.json)。

## 2. true DF tailの大きさ・絶対/相対残差

DF tail行列を`F`、保存Pauli非identity行列を`A`、JW identityを`a`とし、
誤差は`E=A+aI-F`、相対残差は`||E||_F/||F||_F`。λ_Rやfull Hのnormで代用していない。
元のgateは**絶対Frobenius norm 1e-9**であり、この診断でも同じ値を保持する。

|LD|保存DF tailのFrobenius norm|元経路の絶対残差|相対残差|途中除去なしの残差|
|---:|---:|---:|---:|---:|
|7|1.0145358112553073e-3|1.1129429626958517e-7|1.0969972181847215e-4|3.5573812289955387e-19|
|8|3.2228795926790614e-4|9.56360360613471e-7|2.9674095265175074e-3|7.321528670104556e-20|
|9|1.3138255011214958e-6|1.3255266620668722e-6|1.0089061758470879|4.371044642790625e-22|

元絶対残差は再解析で3件とも保存値と数値差0。途中除去なし経路は全て元のgate内で保存DF tailと一致した。
これは診断経路のoperator一致であり、元辞書やRPE精度の再資格ではない。
元のpost-JWはLD7=185項、LD8=102項、LD9=1項（identityを含む数え方）。
途中除去なしの診断出力には、それぞれ481/527/549項がある。微小項も保持するため、これをそのまま
新しいsampling辞書へ採用したという意味ではない。

## 3. identity・非identity誤差の分離

`e_I=Tr(E)/256`、`E_0=E-e_I I`として分離する。
identity誤差normは`16|e_I|`、非identity側はtraceless残差`||E_0||_F`。
これらはHilbert–Schmidt直交し、二乗normが合成する。単にscalar差を全誤差と呼ばない。

|LD|identity誤差係数 e_I|identity誤差norm|traceless誤差norm|
|---:|---:|---:|---:|
|7|−4.060376626077645e-20|6.496602601724232e-19|1.1129429626958517e-7|
|8|−2.120455121823801e-8|3.3927281949180814e-7|8.941583155721406e-7|
|9|−5.193351122457894e-8|8.30936179593263e-7|1.0327468214877722e-6|

保存scalar `s_R`への置換対照も元値を再現。LD7/8/9で残差は1.1129429626958517e-7、
8.941583155721405e-7、1.0327468214877722e-6で、scalarだけの修正ではどれもgateを満たさない。
Hamiltonian定数cはtail比較の双方に含めず、保存scalarの書換えもしていない。

## 4. 最初の差と、gate超過を生んだ段階

defaultは実OpenFermionの元`__iadd__`を呼ぶ。shadowは同じ入力・factor順・積・weightを使い、
局所的な加算で非zeroの微小部分和を除去しない。ソートや一括再配列は行わず、各経路の辞書挿入順を保つ。
途中除去の有無によるkeyの残存・再出現順の差は、その処理の結果として記録する。

### 最初の差：one-body構築

添字は保存8 modeの0-based。`a†_p a_q`の係数が、FermionOperatorへ加算された直後に消えた。

|LD|最初のfactor|最初に消えた項|加算直前の係数|元経路の加算後|
|---:|---:|---|---:|---:|
|9|9|a†_0 a_2|−3.341281621031456e-14|0|
|8|8|a†_0 a_0|−5.462047332320947e-14|0|
|7|7|a†_0 a_2|−4.726947909899728e-14|0|

積とweight乗算では新しい微小係数削除はなく、one-body時点の差が伝播する。
各factorのweighted段階の係数差から、CAR monomialのoperator norm≤1を使って
`16 Σ_f,t |c_default−c_shadow|`を評価すると、one-body差由来のFrobenius規模は
LD9≤1.1288e-17、LD8≤4.1408e-15、LD7≤1.3607e-14。
これは記録されたbinary64係数からの三角不等式の診断値で、丸めを外向きに囲った新しい厳密保証ではない。
**最初に差が出た場所と、1e-9 gateを超えた主因は区別する。**

### weighted Fermion加算

LD9のfactor9で、`a†_0 a_0 a†_0 a_0`のweight込み係数`+9.668557200631008e-9`は
積・weight乗算後には存在するが、`residual +=`で0へ消える。元部分和は0。
LD8でもfactor9の同じ項が同じ条件で消える。

LD7では、この項にはfactor7から`1.9478414357013157e-5`が既に存在し、factor9加算後は
`1.9488082914213788e-5`として残る。同じfactor9を含んでも、prefixが変わると途中除去の結果が変わる。

### JW加算

- LD9：Fermion項`a†_0 a_0 a†_0 a_4`からのPauli `Y0 Z1 Z2 Z3 Y4`の
  `+2.7792146103540544e-9`が0からの加算で消える。
  identityにも`−4.618902521896947e-9`の部分寄与が0から加算された直後に消える。
- LD8：JW集積でも非identityの微小部分和が消える。identityの削除イベントはなく、
  後述のbridgeでもscalar差が残るため、identity不一致は主に前段Fermion側に由来する。
- LD7：`Y0 Z1 Z2 Z3 Y4`の部分和`−2.3841798725517375e-7`へ
  `+2.433365467301491e-7`が加わり、`+4.918559474975347e-9`になったところで消える。
  加算前後の寄与はいずれも小さいとは限らず、途中の相殺結果に閾値が適用されている。

全消失イベントは各`run01/LD*/COEFFICIENT_TRACE.json.gz`に、演算順・factor・Fermion source term・
消失label・前値・加算係数・除去前部分和・float hexを保存した。

|LD|one-body消失イベント|weighted residual消失イベント|JW消失イベント|
|---:|---:|---:|---:|
|9|16|128|745|
|8|32|128|745|
|7|48|0|86|

これは非zero部分和の除去イベント数で、固有label数・最終欠落項数ではない。
極小の相殺丸め項も含むため、全イベントが有意な誤差へ同じ割合で寄与したとは扱わない。

## 5. 残差をどこまで説明できたか

default matrixをD、shadowをS、**defaultのFermion residualをno-filter JWへ渡すbridge**をBとする。

`D−F = (D−B) + (B−S) + (S−F)`

このvector恒等式でJW側、前段Fermion側、残る差を分ける。以下のnormを無条件に足し合わせない。

|LD|前段Fermion側 ‖B−S‖_F|JW側 ‖D−B‖_F|未説明の残差 ‖S−F‖_F|
|---:|---:|---:|---:|
|7|1.0680267142981713e-16|1.1129429626957834e-7|3.5573812289955387e-19|
|8|6.391798989003216e-7|8.722577282339722e-7|7.321528670104556e-20|
|9|6.391798989003106e-7|1.1267711871167262e-6|4.371044642790625e-22|

vector closure残差はLD7=0、LD8=1.65e-24、LD9=5.14e-23。
未説明残差/元絶対残差は順に3.20e-12、7.66e-14、3.30e-16。
元のgate超過に対応するoperator差を、途中除去による差として固定gate内で説明できた。
途中除去なし経路が別Hamiltonianを推定したのではなく、同じ保存DF tailと一致することも確認した。

### LD別の結論

- **LD7**：最初の差はone-bodyの約1e-14係数除去。しかし前段Fermion側の最終operator差は約1e-16。
  JWだけをno-filterにしたbridgeもtailとの残差1.0680e-16で一致する。
  このgate超過の主因はJW部分和の1e-8除去であり、scalarや分割添字の誤りではない。
- **LD8**：factor9のweighted Fermion加算で128回の消失があり、JW加算にも有意な非identity誤差が加わる。
  JWだけをno-filterにしてもbridge残差6.3918e-7でgate外。
  scalar差は前段Fermion側、traceless差には前段とJW側が寄与する。両段の途中除去を扱う必要がある。
- **LD9**：factor9のweighted集積とJW集積の両方で寄与が消え、identityも失われる。
  元表現の誤差はtrue tail normの約1.009倍で、単なる絶対値の小ささの問題ではない。
  両段を保持したshadowは約4.37e-22で一致し、元の1項だけの結果を内部除去で説明できた。

## 6. 最小修正案と資格の境界

[MINIMAL_FIX.md](MINIMAL_FIX.md)に、本番採用とは分けて局所修正案を示す。
LD7だけならJW集積の変更で今回のgate内に入るが、LD8/9にはFermion residual集積も必要。
全体の再発防止にはP0のone-body→Fermion→JWの構築経路に限って非zero微小寄与を保持し、
全体集積前にlibraryの近似ゼロ判定を掛けない方式を提案する。
global toleranceを変えたりgateを緩めたりしない。

今回のshadowは診断用で、本番辞書・canonical ID・λ_P・scheduleの採用/更新をしていない。
signal、価格較正、Q3へ接続せず、元P0のFAILとSTOPを保全した。

## 7. 技術検証と再利用

新しい人工テスト12件PASS：係数trace6、operator分解4、実callerのmock成功/部分保存2。
caller検査でbasis fixtureとpath演算順を修正し、実科学実行前に再検証済み。
P0/M1/save-only loader/Pauli collector/JW実装を再利用し、新しい物理モデルやsamplerは作らない。
traceでは局所subclassで加算を観測し、元JW bytecodeを局所globalsで再利用。
保存operatorとfactor復元を使い回し、全event列や不要な再構築は行わない。
実行順は原因追跡のためLD9→8→7を維持。CPU/RAM/workerの独自上限は設定していない。

primary agentがoperator再解析・caller・実H4診断・報告を担当し、sub-agentがtrace coreと人工6件を担当。
primaryも全12件を実行した。独立科学レビューではない。
元P0・入力・結果・KEEP・grant・ledger・marker・STOPを変更せず、公開照合後mandatory STOP。

研究進行への影響：LD7/8/9のP0障害の原因を固定入力で特定した。
次の依存関係は局所修正方針の承認→新SOURCEでのP0再資格であり、信号/価格/Q3はさらに後続。
全RPE RMSE・費用順位・最適候補について新しい主張はない。
