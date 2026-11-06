# O1 M2：実H4 operator correctness

2026-11-06 JST。**総合状態：PARTIAL / NOT QUALIFIED。積分・CARの接続は数値的一致、既存Pauli bridgeは完全残差を保存できず不適合。** 固定adapter `471b8587535e46732d4ebf0f31e4900ea14b9346` を無変更で使う。旧実験のretryではなく、新しい限定科学検証である。PF、時間発展、RTE law/sampling、state生成・固有値solve、回路build/compile/synthesis、shot/費用、新fit/rank探索は対象外であり、実行していない。重要科学レビューや新しい定理証明は行っていない。

結論を読む入口は[assessment.json][assessment]。元の一律entry規則を保存した[result.json][raw]の`status=PASS`を総合correctnessのPASSと解釈しない。

## 実行前に固定した方法・許容値

保存update005のbase＋全8frameをcoreとし、`R=Hphysical−Hcore`を扱う。`load_saved_pair`は既知hashの元h/g/scalarとJ/Uを明示的にdecodeする。core生成SOURCE `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4` と今回の検証SOURCEは区別する。新検証SOURCEは固定adapter commitと、新driver `scripts/validate_o1_m2_h4_operator.py` の実行時SHA256の組である。

独立referenceには既存`pr_dictionary_validation.reference.car_annihilator`の占有bit/CAR符号を再利用し、CSRのbilinear/quartic単項式をcacheする。元のnormal-ordered積分からH/core/Rを直接構成し、J/Uのdensity-product全coreと比較する。dense256行列の積を何度も繰り返さない。Pauli bridge出力はQiskitの行列化を使わず、labelのbit actionから独立に行列化する。full-Fock256と`Nalpha=Nbeta=2`の36次元部分行列を比較し、途中sector射影は行わない。

結果を見る前に、既存`c1_stage1.gamma` / `operation_lengths`とguard8による工程スケール方式を採用した。

\[
\tau=8\gamma(L)\max(1,S_{\rm physical}+S_{\rm core,density}
+S_{\rm core,integrals}+S_R),\quad
\gamma(L)=\frac{L2^{-53}}{1-L2^{-53}},\quad L=33288.
\]

Lは`max(operation_lengths(4,256,8))`。積分packetのscaleは`|c|+2 sum|h|+2 sum|g|`、density coreのscaleは`|c|+2 sum|kappa|+2 sum_tij |J_tij| s_ti s_tj`、`s_ti=(sum_p |U_tpi|)^2`で固定した。これはCAR単項式の絶対係数和に基づくoperand scaleである。通常積分coreとraw density coreの両方を比較するため両scaleを入れる。実τはoperator差を計算する前に出力する。全entry判定は絶対差、`rtol=0`。

これはbinary64の工程に対する数値検査許容値であり、rigorous rounding enclosure、energy保証、微小係数の完全保存証明ではない。前段synthetic testの`1e-14`を実H4へ移送しない。不一致を見てτを変更しない。OpenFermionの`EQ_TOLERANCE`は無変更で、実際に落ちるfermion係数の数・L1・operator影響と全mapper差を別に記録する。

## 結果

H4直鎖・間隔1.0Å・STO-3G、4空間軌道、8 spin modes、保存CDF rank8の全frameを検査した。LD2 tailは使っていない。実行はoperator-onlyの1回、経過3.130052秒。元h/g/scalarと保存J/Uをロードし、HcoreとRを新規構成した。stateはロード・生成していない。

事前式から得た実許容値は **`tau=8.011293947336095e-9`、rtol=0**。operand scaleはphysical `48.13275195368546`、density core `174.7004629621234`、normal-ordered core `48.132751998822975`、R `6.930386640756261e-8`。

| operator比較 | full256 最大entry差 | sector36 最大entry差 |
|---|---:|---:|
| raw density全core vs normal-ordered core | 6.661338147750939e-15 | 2.220446049250313e-15 |
| raw density全core＋R vs 元H CAR | **1.0658141036401503e-14** | **2.1094237467877974e-15** |
| normal-ordered core＋R vs 元H CAR | 3.9968028886505635e-15 | 1.2212453270876722e-15 |
| physical Pauli vs 元H CAR | 6.217248937900877e-15 | 2.3314683517128287e-15 |
| core Pauli vs core CAR | **2.2651976998076384e-9** | **2.2651976998076384e-9** |
| R Pauli vs R CAR | **2.2651962304775996e-9** | **2.2651962304775996e-9** |
| Pauli core＋R vs 元H CAR | **1.1337143446255027e-9** | **1.1337143446255027e-9** |

全体の最大差はcore Pauliの`2.2651976998076384e-9`。積分段の符号・一体補正とfull coreのCAR和は丸めスケールで整合した。一方、Pauli residualは**nonidentity 0項、identity 0**となった。R CARの最大entry `2.2651962304775996e-9` が丸ごと失われている。

### 残差符号・一体補正・scalarとsector

原gから独立な要素和で計算したκとadapter κの最大差は`1.232595164407831e-32`。R_hと`0.5 sum_r R_g[prrq]`の差は`1.0408340855860843e-16`、`R_g−(g_original−g_fit)`は`0`。R_h自体は最大`8.000222706527893e-11`、R_g自体は最大`7.998633243176693e-10`であり、ゼロ一体項として扱っていない。旧保存field `g_fit−g_original`とは逆符号である。

核scalarはphysical/coreとも`2.29310124732`、Rは`0`。電子identityは核scalarと区別する。R CARの`trace(R)/256`は **`4.6249802598818945e-11`** だが、Pauli bridgeでは0になった。元H/coreのtotal identityはそれぞれ`-0.3314778134168109` / `-0.33147781346306027`であり、これらには既知核scalarも含まれる。電子寄与だけはそこから`2.29310124732`を引く。[assessment.json][assessment]にtotalとelectronicの両ラベルを分けて保存した。元rawの`bookkeeping.electronic_identity`はphysical/coreについてtotal identityを格納しているため、この説明を優先する。

spin順は`2p=alpha, 2p+1=beta`、occupation bitのLSBがmode0、Qiskit Pauli labelの右端がqubit0。full-Fockから`Nalpha=Nbeta=2`の36indicesを選択した。CAR physical/core/Rとdensity coreのsector外への最大entryはすべて0。これは完全なoperatorの検査であり、Pauli eventや回路途中でsector36へ射影してよいという主張ではない。

### 微小項の消失原因

使用runtimeのOpenFermion1.8.1では`config.EQ_TOLERANCE=1e-8`。既存mapperの`get_fermion_operator(InteractionOperator)`は各項を`SymbolicOperator.__iadd__`で加え、絶対値がこの値未満の項を削除する。当adapter・driverはこの定数をpatchしていない。

| packet | 非zero fermion単項式 | 内部閾値未満／実際に消失 | その係数L1 | 消失分CARの最大entry |
|---|---:|---:|---:|---:|
| physical | 1057 | 528 | 6.610392591521696e-14 | 1.7294761782189885e-15 |
| core | 1057 | 528 | 4.4102191974080635e-8 | 2.2651976998076384e-9 |
| R | 1056 | **1056（全項）** | **6.930386640756261e-8** | **2.2651962304775996e-9** |

実際の消失項を独立CARで足し、その分を元CARから除いたoperatorとmapperを比較すると、coreは最大`5.329070518200751e-15`、Rは`4.1359030627651384e-25`だった。したがって主要差は原κやnormal-order代数ではなく、既存fermion変換での微小項削除に帰属する。残る差にはJW側の内部cutoffとbinary64丸めが含まれ、それらを厳密演算として完全分離したとは主張しない。

Rを小さいという理由でゼロへ置き換える設計は、今回の完全残差条件とは異なる。元Hとのmodel誤差・新しい予算配分・性能上の正当化は評価していない。この残差の大きさや一体補正の実測を費用改善の証拠として使わない。

### 一律entry PASSと総合不適合を分ける

事前の総H operand scaleによるτはRそのものより大きいため、元driverの一律entry判定は全項でPASSとなった。誤った符号のnegative control `Hcore−R−Hphysical`も最大`4.5303924371146826e-9`でτ未満となり、この一律基準だけでは完全残差を検出できないことが分かった。

τを結果後に下げたり緩めたりして再判定していない。元driver・元[result.json][raw]をそのまま保全し、[assessment.json][assessment]に**工程entry規則PASS／完全残差Pauli bridge FAIL／総合PARTIAL_NOT_QUALIFIED**と明記した。1056非zero項の全消失とidentity消失を、総Hスケール内という理由で科学correctnessのPASSへ昇格させない。

## SOURCE・input・runtimeと再現入口

- 新検証SOURCE：adapter commit `471b8587535e46732d4ebf0f31e4900ea14b9346` ＋ [driver][driver] SHA256 `a3f24b51f06198cb540f155560bd0504d399d9792e7431983e8070009050d512`。
- adapterは無変更。実行前に固定commitのadapter bytesと照合した。原κ／tensor contraction／general-K／Pauli helperとbit-CARは固定commitの実SOURCEを再利用した。
- 保存core生成SOURCE：`980b514c81eb415fe053f558d7d3c0b8cfb5b9b4`。これは新検証実行SOURCEではない。
- 元入力の結果commit：`aced1647b81e8d28091058cdc7b6e3e1ee54896a`。原pathはcore `artifacts/research_c_pr_integrated/2026-11-02/run01/update005_candidate.npz`、元H `artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz`。
- 新directoryの[core copy][core-copy]と[input copy][input-copy]は、上記commitにある既存Git blobの同一bytesコピー。SHA256はそれぞれ`938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e` / `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。既存原本・旧SOURCE/raw/grant/marker/STOPは不変更。
- raw result SHA256：`7de721b3c6ee69e027933cd1093f0195f515b040c39eef780ddb26fd096dca35`。
- runtime：Python3.12.3、NumPy2.5.4、SciPy1.18.1、Qiskit2.6.2、OpenFermion1.8.1。前段の専用`/tmp/track-a-o1-m2-runtime-20261106`を再利用し、共有packageを変更していない。float64/complex128、GPUA100-gleap02。runtime版対照は追加しない。

実H4実行前に、小人工1空間軌道の疎CARを独立な占有対角値と、2qubit全16Pauli labelのbit actionをI/X/Y/Z Kronecker積と照合してPASSした。実H4の判定値は上記の事前式を用い、人工test値を移送していない。

実行command（固定driverはfresh出力のみ。既存resultを上書きしない）：

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-a-o1-m2-h4-operator-correctness-20261106
PYTHONDONTWRITEBYTECODE=1 \
XDG_CACHE_HOME=/tmp/track-a-o1-m2-h4-cache-20261106 \
MPLCONFIGDIR=/tmp/track-a-o1-m2-h4-cache-20261106/matplotlib \
/tmp/track-a-o1-m2-runtime-20261106/bin/python scripts/validate_o1_m2_h4_operator.py
```

## 未解決事項・次の最小段階・STOP

**固定471b adapterの積分/CAR側は検査範囲で整合し、既存Pauli bridgeの完全残差保存は未解決。** adapter修正や別runtimeでの再実行は開始していない。原rawと固定SOURCEを維持したまま公開する。

次の担当：**ユーザー**が、内部cutoffを避ける最小接続修正と人工検査、その新SOURCEによる同じH4 operator-only再検証へ進むか判断する。許可された場合は**Codex**が、係数の完全収集・一体/scalar・spin/JW規約を維持する最小修正を行い、残差自身を区別できる比較条件を結果を見る前に定めて検証する。既存adapterの過去版は上書きしない。新fitや大きいframeworkは不要。

PF、RTE law・mean/半径、元H state検査、回路、shot/総費用へ自動進行しない。今回のSOURCE・結果・input copy・報告を専用branchへ公開し、全変更fileのremote再取得一致とcleanを確認して**mandatory STOP**する。

[driver]: ../../../scripts/validate_o1_m2_h4_operator.py
[raw]: ../../../artifacts/track_a_o1_m2_h4_operator_correctness/2026-11-06/result.json
[assessment]: ../../../artifacts/track_a_o1_m2_h4_operator_correctness/2026-11-06/assessment.json
[core-copy]: ../../../artifacts/track_a_o1_m2_h4_operator_correctness/2026-11-06/inputs/update005_candidate.npz
[input-copy]: ../../../artifacts/track_a_o1_m2_h4_operator_correctness/2026-11-06/inputs/input.npz
