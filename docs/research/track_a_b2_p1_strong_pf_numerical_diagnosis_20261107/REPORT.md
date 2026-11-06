# B2 P1強PF：5 armの独立数値不一致の技術調査

2026-11-07、SERVER 2 GPUA100-gleap02。

**基本stepの非unitarity・構成誤差が長い累乗で増幅する機構を人工例で確認した。ただし、実H4の5停止点の原因別寄与・主因は未確定。数値アルゴリズムの修復は行わない。**
既存SOURCEの数式・順序に相違は見つからず、人工6診断と既存人工18テストはPASSした。
実H4の新規operator構築・信号計算・保存信号の再解析・再資格判定は実行していない。
**3 PASS / 5 STOPを変更せず、公開照合後mandatory STOP。**

## 固定資料と調査対象

- 実行SOURCE：`a7940b58c6cb939d77e9f8e44d214b3787b7e101`。
- 8 arm結果：[`2583ce4957ed44e914ce6571e3e676830439dfa7`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/2583ce4957ed44e914ce6571e3e676830439dfa7)、[元REPORT](../track_a_b2_p1_strong_pf_20261107/REPORT.md)、`artifacts/track_a_b2_p1_strong_pf/2026-11-07/run01/result.json`。
- 当該result JSONのSHA256：`c5e21203833185c68b5d003925333342f64bb2e738154bf58e17e0d5c4f69103`。
- P1：`7bc952772e451f27c1654dab6803192f2b204c9f`、B2信号：`d7a55ac181de71cd2b2c49644810373cafd0a450`、B2 shot：`0d5ca357ccfd8fcc850bef800c5bcadfde2a153e`、既存強PF接続：`66a41ad9219f12377e7d325e272c1a90d66ab7e5`。これらの科学処理は起動していない。

以下は**保存済み診断fieldの引用**で、新しいH4誤差算出・fit・再ランキングではない。
step差・unitarity defectは旧SOURCEの`max(abs(entry))`で、spectral/operator normではない。

| arm | 最初のSTOP m | 保存step独立差 | 保存native defect | STOP点の複素差 | 同じ点の半径差 | 同じ点の位相差 [rad] |
|---|---:|---:|---:|---:|---:|---:|
| Full S2 s4 | 17 | 2.83143e-14 | 5.30687e-14 | 1.15257e-9 | 1.15257e-9 | 1.32189e-12 |
| Full S2 s8 | 16 | 5.14086e-14 | 1.01807e-13 | 1.07431e-9 | 1.07431e-9 | 2.87192e-12 |
| Yoshida4 s2 | 17 | 3.87493e-14 | 7.61613e-14 | 1.67252e-9 | 1.67205e-9 | 3.95452e-11 |
| Yoshida4 s4 | 16 | 8.43844e-14 | 1.52434e-13 | 1.36834e-9 | 1.36830e-9 | 1.04776e-11 |
| Yoshida4 s8 | 15 | 1.57893e-13 | 3.10085e-13 | 1.52926e-9 | 1.52926e-9 | 5.76122e-13 |

固定独立閾値1e-9は複素差・半径差・円周位相差に適用される。旧失格は複素差/半径差で、PF位相cap=.06 rad超過ではない。
保存fieldに**CAR側step defect、step行列自体、累乗の高精度参照、各構成段階の丸め誤差分解はない**。
したがって保存診断だけで「matrix_powerが主因」「native構成だけが主因」と確定できない。小さい位相差も演算子構成ミスの完全否定にはならない。

## 数理的対応と数値経路の違い

| 確認項目 | SOURCEで確認した対応・限界 |
|---|---|
| schedule | `b2_rpe_connection.strong_baseline_schedule`は既存`ordered_schedule`を使用。Full S2はone-body→全signed factor→逆順の対称列。Yoshidaは`_get_w_list('4th')`の`(w[1],w[0],w[1])`、正・負・正の順で、基本δ/substepsに適用。 |
| block | `prepare_xdf_basis(...,SPIN)`とCAR側は同じone-bodyとλ_l[dΓ(g_l)]²を使用。近似した別tail、別rank、PF次数を使わない。 |
| basis transition | `StrongBuilder.sequence(relative_fusion=True)`と数値adapterは、初期Γ(V_0)†、各diagonal、遷移Γ(V_next†V_previous)、終端Γ(V_last)を時系列に適用。CAR側`_ordered_exponentials`も左から新しいfactorを掛ける。 |
| identity/scalar | 既存diagonal primitiveはRZ/RZZとglobal phaseを合わせてn、n_i n_jを再現。factor内identityを保持し、核scalarは基本δに一度。B2 tail identityの別加算なし。結果へのglobal phase fitなし。 |
| sector/order | exact interleaved spin-lift判定、Qiskit bit-orderと既存CAR/Fock helperを共用。旧保存generator leakage=0、basis leakageも検査済み。今回は実H4のsector作用を再計算していない。 |
| native step | NumPyの数値固有ベクトル、binary64 primitive/位相計算、`fock_lift`のminor determinant、relative行列積、chronological積。`native.runs`/`combined_diagonal`の隣接共有とfsumは既に使われている。 |
| CAR step | 元のCAR generatorからSciPy `expm`を作り、同じscheduleで行列積。こちらも丸め誤差を持つので、数値的なexact unitaryとは仮定できない。 |
| 長round | 両stepはcomplex128で`np.linalg.matrix_power(step,2**m)`へ渡る。保存stateは同一で、経路間の正規化・位相合わせはない。 |

厳密にunitaryなVについてΓ(V_next†V_previous)=Γ(V_next)†Γ(V_previous)なので、隣接基底は代数的に消去でき、両経路は同じPFを表す。
この等価性は**厳密算術の目標**についての確認であり、二つのbinary64 stepが同一の数値行列、または厳密unitaryだという主張ではない。

特に、`fock_lift(V)`は渡された行列のminorをそのまま評価する。数値Vがわずかに非unitaryなら、その外積表現も非unitaryになり得る。
一方、既存`precision.gaussian`はLAPACK GivensとRZ/XXPlusYYの角度列を生成する。理想gateの積はunitaryであり、裸のΓ(近似V)と数値的に同一とは限らない。
**裸の外積表現とemitterの数値的差は区別すべき候補原因**で、今回実H4のgate列をbuildして確認したものではない。

NumPy1.26.4の実装を読取確認した。正整数の`matrix_power`は二乗反復・行列積を使い、入力を正規化しない。Q=2^mではm回の二乗であり、Q個の実回路を展開する処理ではない。[NumPy公式仕様](https://numpy.org/doc/1.26/reference/generated/numpy.linalg.matrix_power.html)

一般にA^Q−B^Q=Σ_{k=0}^{Q−1} A^{Q−1−k}(A−B)B^k。unitary近傍でも、基本stepの差は概ねQ倍の尺度に増幅し得る。
二乗中の丸め誤差も後続の二乗で増幅するため、演算回数がm回であることから誤差がm倍に留まるとは言えない。
これは一般的機構の説明で、保存されたentrywise defectからH4の停止点を予測・認証したものではない。

## 人工診断と原因の資格

新規コードは人工診断scriptのみ：[diagnose_b2_p1_strong_pf_numerics.py](../../../scripts/diagnose_b2_p1_strong_pf_numerics.py)。
[機械可読の人工結果](../../../artifacts/track_a_b2_p1_strong_pf_numerical_diagnosis/2026-11-07/run01/synthetic_results.json)に全数値とscript SHA256を保存した。
実分子入力・保存H4数値を受け取る引数はない。private/public入力loader、H4 runner、回路constructor、shot APIを禁止して実行した。
既存の`native_mean_step`、`fock_lift`、`_ordered_exponentials`、`matrix_power`を再利用した。
mpmath1.3.0の80桁演算は人工行列だけに使用し、共有環境へのinstallや設定変更はしていない。

| 人工診断 | 結果・意味 |
|---|---|
| 厳密表現可能なunitary | 2×2の90度回転をQ=131072で累乗して誤差0。長い累乗だけで必ず失敗するわけではない。 |
| 既知の半径defect | η=2^-46を与えた(1+η)Uの誤差は予測`(1+η)^Q−1`に一致。Q=131072で約1.86265e-9、同じ格納行列の高精度累乗との差は約1.73e-18。この人工例では入力defectが支配。 |
| 構成精度と累乗精度の分離 | 同じbinary64角度.3を使う2×2回転で、格納行列の高精度累乗と理想回転との差は約5.99e-12、binary64累乗の追加誤差は約2.01e-12。構成から80桁で計算すると約3.92e-77。誤差項は相殺もあり、絶対値を単純加算して原因比率にしない。 |
| Schur負例 | full Tの累乗は元行列を保つ。固有値の位相だけを残す方式は、半径defectまたは非対角成分を落とし、約1.86e-9の作用を隠す人工例を確認。実結果にこの操作を適用していない。 |
| zero-time Gaussian負例 | 人工orbitalを(1+η)Iとすると、時間0でも裸の外積表現によるstepに約5.68e-14のidentity差が生じ、Q累乗後約7.45e-9。厳密Iの対照は誤差0。実H4の基底がこの形だという推定ではない。 |
| 演算順序の負例 | signed時間・identityを含む人工CAR例で、誤った順序の差は80桁でも約.00749残る。同順序のbinary64/高精度差は約8.95e-17。高精度化は意味の誤りを直さない。 |

追加人工6診断PASS。既存`tests/test_b2_p1_strong_pf_validation.py`も**18/18 PASS（2.90秒）**。
固定8 armのS2/Yoshida schedule、負stage、signed factor、scalar、数値作用、予算非再配分、初回失格STOPを人工入力で確認した。
219件や実H4 scienceの再実行ではない。人工の精度改善を実H4のPASSへ転用しない。

確定したのは、SOURCEの代数的対応と、人工例で分離できた誤差増幅機構である。
**H4の実際の原因は、固有基底/外積/diagonal/CAR expm/基本step積/累乗のどこにどれだけあるか未確定**。
保存値の半径方向の不一致は構成・累乗丸め仮説と整合するが、その仮説の確証ではない。実装上の構成差も未排除。

## 修正の判断と方法比較

**本体の数値経路・式・閾値は修正しない。** `src/`と既存テスト・runnerは親commitと同一bytesを維持する。
不足診断を飛ばして修復法を採用する根拠がなく、今回の差分は人工診断script、人工結果、REPORTと直接の索引のみ。

| 方法 | 適合性・今回の判断 |
|---|---|
| 不要な基底積・積回数の削減 | 既存の隣接fusion/cacheは維持。共通frameやsubstep単位での再構成は候補だが、構成誤差を減らすかは別検証が必要。未実装。 |
| 厳密unitaryな表現 | 既存Givens/gate意味との接続は候補。近似Vの裸のΓをそのまま厳密unitaryと扱わない。実H4 emitterの検証は今回範囲外。 |
| Schur/固有位相 | SchurのA=ZTZ†でfull Tを保つなら元行列のdefectも残る。位相だけの累乗への置換は自動採用不可。[SciPy公式仕様](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.linalg.schur.html) |
| polar・再正規化 | 元stepを変更する。閾値通過を目的に採用しない。今回未使用。 |
| 高精度累乗のみ | 基本step構成の既存誤差は消えない。まず原因分離の参照として使うのが適切。 |
| 両経路の構成から高精度化 | 意味を保つ候補。既存`_ordered_exponentials(...,arithmetic=...)`を再利用可能。ただしnative側`complex`/float cast、eigh、minor/diagonal処理の精度も揃える必要があり、一行のdtype変更だけでは足りない。根拠なく一式を作り直さない。 |

将来高精度化する場合も、同じ固定factorのbinary64値と既存Fraction時間・Yoshida係数を入力とし、新しいDF fitや係数の再定義をしない。
数理的PF、丸め済み基本step、その入力を近似してemitしたgate列を区別し、どの対象との差を計算したか記録する。

## 次の限定実H4診断案（未承認・未実行）

原因を実H4について確定し、修復法を選ぶには追加の科学実行承認が必要。
最初は各停止armの**m=0、停止直前、停止点**だけを対象とする計15点を提案する。

| arm | 診断対象m |
|---|---|
| Full S2 s4 | 0,16,17 |
| Full S2 s8 | 0,15,16 |
| Yoshida4 s2 | 0,16,17 |
| Yoshida4 s4 | 0,15,16 |
| Yoshida4 s8 | 0,14,15 |

既存SOURCE `a7940b...`の入力/構成を基準として、同じ保存DF12・元H state・sector・scalar・spin/bit-orderを使う。
H_phys target、DF rank12、δ=.02、18round枠、P1配分/allowance、独立1e-9は変更しない。既存z_phys/z_DF参照を再生成しない。

1. 基本δ operatorを両経路で一度ずつ再構成し、**両方の**unitarity defect、基底・遷移・diagonalの残差を保存。旧経路の同定を優先する。
2. 上記15点に限り、binary64累乗と、**同じ丸め済みstepを入力した**高精度累乗を比較。これは累乗内丸め誤差を分離する診断で、operatorの修正・正規化ではない。独立高精度照合にも既存1e-9を維持。
3. 高精度累乗でも両step間の差が残る場合、基本step構成の誤差または構成差として、次に必要な精度修正を絞る。今回の人工例だけからnative単独の修正を選ばない。

この案は原因分離まで。5 armの救済や全18round再資格判定は含めず、3旧PASS armも再実行しない。
修正後の5 arm全18round評価が必要になった場合は、固定新SOURCE・同じP1と入力・対象範囲を示して改めて承認を得る。旧3 PASS/5 STOPを上書きしない。
入力不一致・参照不成立・数値異常なら診断をSTOPし、位相fit・polar・spin projection・条件変更で救済しない。

**担当**：ユーザーが上記15点の原因診断を承認するか判断し、Codexが明示承認後に実行する。今回は実行していない。
技術的な原因分離にGPT詳細科学レビューは必須ではなく、開始していない。branch・shot・価格・総費用・成功保証にも進まない。

**mandatory STOP。実H4の未承認科学計算なし。**
