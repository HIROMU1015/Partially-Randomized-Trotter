# B2 P1強PF：実H4・5 arm / 15点の数値原因診断

2026-11-07、SERVER 2 GPUA100-gleap02。

**15/15点で診断成立。5つの元STOP点すべてで、高精度累乗後にも基本step間の差が1e-9を超えて残った。**
追加のbinary64累乗処理差より基本step間の差が大きく、累乗処理だけの高精度化では解消しない。
両stepに非unitarityを観測したが、どちらが厳密なPFに近いか、基本step構成内のどの操作が主因かは**INCONCLUSIVE**。
既存3 PASS / 5 STOPを維持。修正・再適格化・branch/shot/価格へ進まず**mandatory STOP**。

## 固定SOURCE・入力・実行範囲

- 診断SOURCE：[`b81734e177b3161f287aea908720eac5c9394a00`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/b81734e177b3161f287aea908720eac5c9394a00)。人工12件PASS後、commit/tree/bytesを照合して実行した。
- 元の科学SOURCE：`a7940b58c6cb939d77e9f8e44d214b3787b7e101`、8 arm結果：`2583ce4957ed44e914ce6571e3e676830439dfa7`、直前人工診断：`e217d145bd99c2b34302017390ecb57987c06b90`。
- P1：`7bc952772e451f27c1654dab6803192f2b204c9f`、B2信号：`d7a55ac181de71cd2b2c49644810373cafd0a450`、B2 shot：`0d5ca357ccfd8fcc850bef800c5bcadfde2a153e`、既存接続：`66a41ad9219f12377e7d325e272c1a90d66ab7e5`。参照の読取照合だけでB2計算は再実行しない。
- [結果JSON](../../../artifacts/track_a_b2_p1_strong_pf_h4_diagnosis/2026-11-07/run01/result.json)、[15点CSV](../../../artifacts/track_a_b2_p1_strong_pf_h4_diagnosis/2026-11-07/run01/rounds.csv)。HP信号・差は120桁のdecimal文字列で保存し、評価前にbinary64へ丸めない。表だけ短い桁で表示。

対象はFull S2 s4: m=0,16,17、Full S2 s8: 0,15,16、Yoshida4 s2: 0,16,17、s4: 0,15,16、s8: 0,14,15の15点のみ。
Full S2 s1/s2・Yoshida4 s1の3 PASS armは再計算していない。中間二乗は累乗の内部処理で、未指定roundの信号・適格性は算出していない。

linear H4、1.00 Å、STO-3G、cutoff前H_phys target、保存DF12、Nα=Nβ=2/sector36、保存元H基底stateを固定。state preparationは除外。
δ=.02、Q=2^m、T=Qδ、Full S2/標準Yoshida4、各既存substeps、全signed factors・one-body・核scalar・factor identity・spin/JW/bit-orderを維持。
P1のε_E=.00015936001019904 Ha、95%目標、β=.4、区間[-128,128] Ha/shift0、exp(-itH)、model/PF/RTE/stat/implementation/reserve=.02/.06/.04/.24/.02/.02、α=.05/36、u=1e-8と独立1e-9も維持した。今回はこれらの達成保証を判定しない。

元H/state/DFの固定hashは既存`load_private_inputs`で照合し、JSONに記録。
H_phys S0 hashは`bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2`、
state S0 hashは`1a2df11b16743e68fb27a3f4b39e78e145897cb49a8cc931b080e44e7400c1db`、
DF Hamiltonian hashは`de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。
private state NPZ SHA256=`deb561367e216fd4c62d013c10a4adbbef5da8c3e8139c39170d71f80530a80c`、DF NPZ SHA256=`3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。
DF snapshotのstateは使用せず、元stateをDF/PF固有stateと仮定していない。private原本のhashは実行後も一致。
元H/DF演算子の再構成は入力hash照合の既存処理だけで、z_phys/z_DFの時間発展は行っていない。

## 再利用と診断方法

新規差分は[診断adapter](../../../src/trotterlib/b2_p1_strong_pf_h4_diagnosis.py)、[runner](../../../scripts/run_b2_p1_strong_pf_h4_diagnosis.py)、[人工テスト](../../../tests/test_b2_p1_strong_pf_h4_diagnosis.py)のみ。元のnative/CAR科学実装を変更していない。

1. `strong_baseline_schedule`、spin-aware `prepare_xdf_basis`、`native_mean_step`、`_ordered_exponentials`を元SOURCEと同じ入力・順序で呼び、各armの基本stepを一度ずつ構成。SOURCEの保存hashを全て照合。
2. 元SOURCEの基本step独立差・native defect・leakage、指定15点の両binary64信号のreal/imag/radius/phase、独立差が**保存値と完全一致**。同じ閾値内というだけでなく、保存scalarの一致を確認した。
3. 直前人工診断の`mp_matrix`を再利用し、stepとstateのbinary64成分をその値のままmpmathへ変換。基本stepを再構成・再定義せず、80桁と120桁で既存行列整数累乗（**2）を反復。共通二乗を再利用し、指定点だけで期待値を計算。
4. full operatorのentrywise差、信号の複素差・半径差・円周位相差を80/120桁で照合。全て既存1e-9以内。最大operator差は約1.40610e-77、最大信号複素差は約1.52798e-77。これは数値的な精度安定性で、厳密rounding enclosureではない。
5. N/C両経路のbinary64累乗処理差、HP間の基本step差を複素数として分解。stateへの行列作用・内積の丸めもD_powerに入るため、補助項でmatrix-powerだけの差とcontraction差を分離した。STOP点のcontraction差は約2.63e-17〜1.86e-16で、支配項ではない。

人工12件PASS（3.10秒）：exact binary変換、80/120桁照合、非normal人工行列、projectionしない動作、複素分解、元保存値の小さな改変でも拒否する動作、8 armのsigned diagonal/基底診断を確認。既存219件や3 PASS armを再実行していない。

## 基本stepの診断A

全て**max(abs(entry))**。spectral normや全長回路誤差上界に読み替えない。

| arm | N−Cの最大entry差 | N†N−Iの最大entry | C†C−Iの最大entry |
|---|---:|---:|---:|
| Full_S2_s4 | 2.83143e-14 | 5.30687e-14 | 5.55112e-15 |
| Full_S2_s8 | 5.14086e-14 | 1.01807e-13 | 9.32587e-15 |
| Yoshida4_s2 | 3.87493e-14 | 7.61613e-14 | 6.43929e-15 |
| Yoshida4_s4 | 8.43844e-14 | 1.52434e-13 | 1.64313e-14 |
| Yoshida4_s8 | 1.57893e-13 | 3.10085e-13 | 2.19824e-14 |

native側だけでなくCAR側にもdefectを観測した。値の大小だけでどちらが数理的な厳密PFに近いかは判定できない。
C†(N−C)のHermitian成分は約2.83e-14〜1.58e-13、anti-Hermitian成分は約1.51e-15〜4.67e-15。
これはlinearizedな振幅/位相診断で、厳密なphase boundやerror attributionではない。

共通基底・transition・diagonalの残差最大値：

| 診断 | 最大entry残差 |
|---|---:|
| orbital orthogonality | 8.88178e-16 |
| 元one-body/Gのorbital再構成 | 8.32667e-16 |
| sector Fock基底のunitarity | 3.77476e-15 |
| diagonal generatorとCAR generator | 5.32907e-15 |
| relative orbitalのunitarity | 1.66533e-15 |
| sector relative基底のunitarity | 5.21805e-15 |
| Γ(V_next†V_prev)とΓ(V_next)†Γ(V_prev) | 5.55112e-16 |
| diagonal primitiveと直接occupation位相 | 1.14439e-16 |
| generator/basis/transitionのsector leakage | 0 |

既存scheduleとの完全一致により正負Yoshida stage・時系列順序を確認。核scalarは基本δに一度、factor identityは既存primitive内に保持。
現証拠で大きな符号・順序・bit-order誤りは見つからない。一方、数値固有基底・minor・相対基底積・diagonal・CAR expm・chronological積のどの丸めがstep差を作るかは、この診断だけでは分離できない。
裸のΓ(数値V)、数理的PF、理想gate列の意味は同一の数値対象と扱わず、gate列は今回buildしていない。

## 元STOP点の分解B/C

DN=z_N64−z_NHP、DC=z_C64−z_CHP、DS=z_NHP−z_CHP、P=DN−DCとして、**D_total=DN+DS−DC=DS+P**。
各点でこの複素恒等式を確認し、120桁計算での記録残差は0。各項のreal/imag・半径・位相・比較はJSONに保持する。
絶対値の単純和や百分率を原因寄与率としていない。cosはDSとPの方向cosineで、負は相殺、正は増幅を示す。

| arm | 元STOP m | 元と再現した\|D_total\| | \|DS\|（HPでも残る差） | \|DN\| | \|DC\| | \|P\| | cos(DS,P) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full_S2_s4 | 17 | 1.152574613e-09 | 1.150381532e-09 | 1.44746e-12 | 3.2291e-12 | 2.4256e-12 | 0.903948 |
| Full_S2_s8 | 16 | 1.074312714e-09 | 1.07508923e-09 | 2.09914e-13 | 9.61609e-13 | 1.08446e-12 | -0.716287 |
| Yoshida4_s2 | 17 | 1.672520694e-09 | 1.678786681e-09 | 1.84881e-12 | 4.42351e-12 | 6.26648e-12 | -0.999921 |
| Yoshida4_s4 | 16 | 1.368342063e-09 | 1.36874575e-09 | 4.6924e-13 | 3.44877e-13 | 7.8671e-13 | -0.513345 |
| Yoshida4_s8 | 15 | 1.529256805e-09 | 1.528525586e-09 | 4.19954e-13 | 3.93741e-13 | 7.44702e-13 | 0.981887 |

**5点全てで|DS|>1e-9、|P|<1e-9**。基本step間の差が高精度累乗後にも閾値を超えて残る。
Full S2 s4とYoshida4 s8ではPが増幅方向、Full S2 s8とYoshida4 s2/s4では相殺方向に働いた。
このため元STOPを「binary64累乗だけの丸め誤差」として扱えず、HP累乗だけでの救済にも根拠がない。
**主要な残差の所在は基本step間に絞れたが、その差に含まれるnative/CARそれぞれの構成誤差・構成差の内訳はINCONCLUSIVE**。

全15点の要約：

| arm | m | Q | \|DS\| | \|P\| | \|D_total\| |
|---|---:|---:|---:|---:|---:|
| Full_S2_s4 | 0 | 1 | 8.7767212e-15 | 1.0964513e-16 | 8.6675451e-15 |
| Full_S2_s4 | 16 | 65536 | 5.7519077e-10 | 1.2128243e-12 | 5.7628733e-10 |
| Full_S2_s4 | 17 | 131072 | 1.1503815e-09 | 2.425599e-12 | 1.1525746e-09 |
| Full_S2_s8 | 0 | 1 | 1.6404563e-14 | 6.7851169e-17 | 1.6336844e-14 |
| Full_S2_s8 | 15 | 32768 | 5.3754462e-10 | 5.4221611e-13 | 5.3715634e-10 |
| Full_S2_s8 | 16 | 65536 | 1.0750892e-09 | 1.0844567e-12 | 1.0743127e-09 |
| Yoshida4_s2 | 0 | 1 | 1.2808126e-14 | 3.8176572e-17 | 1.2770008e-14 |
| Yoshida4_s2 | 16 | 65536 | 8.3939334e-10 | 3.1334073e-12 | 8.3626018e-10 |
| Yoshida4_s2 | 17 | 131072 | 1.6787867e-09 | 6.2664816e-12 | 1.6725207e-09 |
| Yoshida4_s4 | 0 | 1 | 2.0885403e-14 | 2.2256442e-16 | 2.1107545e-14 |
| Yoshida4_s4 | 15 | 32768 | 6.8437288e-10 | 3.9349324e-13 | 6.8417084e-10 |
| Yoshida4_s4 | 16 | 65536 | 1.3687457e-09 | 7.8671015e-13 | 1.3683421e-09 |
| Yoshida4_s8 | 0 | 1 | 4.6646899e-14 | 8.4620018e-17 | 4.6563666e-14 |
| Yoshida4_s8 | 14 | 16384 | 7.6426279e-10 | 3.7222005e-13 | 7.6462827e-10 |
| Yoshida4_s8 | 15 | 32768 | 1.5285256e-09 | 7.4470177e-13 | 1.5292568e-09 |

機械可読classificationは事前固定の1e-9尺度で記録した。元STOP点は全て`STEP_DIFFERENCE_REMAINS_ABOVE_FIXED_THRESHOLD`。
それ以外の10点は両項が閾値以内なので`INCONCLUSIVE`とした。これは超過原因分類を割り当てない意味で、分解や精度参照の不成立ではない。

## 改善候補と次の必要範囲

今回は**アルゴリズムの修正を行わない**。結果・判定・P1条件の事後変更もない。

- 最小の修正候補は、同じfactor/時間/scheduleのまま基本stepの構成精度を改善するadapter。両経路を対象に、orbital eigensystem、Fock minor/relative積、diagonal位相、CAR expm/積のどこで差が増えるかを人工例で確認する。累乗だけの高精度化を修復案にはしない。
- 既存`_ordered_exponentials(...,arithmetic=...)`等を優先し、原入力のbinary64値・既存Fraction時間とYoshida係数を保持する。単に丸め済み基底を高精度へcastしても、その基底の既存defectは消えない。必要な場合に限り、同じHermitian factorからorbital対角化を含む構成精度を上げる。新DF・state solve・PF式の開発ではない。
- 実装を行う段階では人工系で両構成の高精度参照、signed stage/scalar/identity、sectorとbit-order、HP変換、長時間の丸めを検証。強制polar、信号正規化、固有値の絶対値置換、phase fitで閾値を通過させない。
- 修正後の実H4適格性を主張するには、固定新SOURCE・同一入力/P1で停止5 armの全18round（最大90点、各arm最初の失格でSTOP）の新規科学実行が必要。今回の15点診断を全18round PASSへ転用しない。既存3 PASSは旧SOURCEの証拠として維持し、新SOURCEの共通処理を変えた場合に必要な回帰範囲は別途固定する。

**担当**：ユーザーが基本step精度修正・人工検証の実施範囲を判断し、Codexが承認後に対応する。実H4再適格化はさらに対象SOURCE・範囲を明示した実行承認後。
今回、修正・再資格判定・strong PF branch/shot・実回路compile・価格/総費用・GPT詳細科学レビューには進んでいない。

実行は約46.91秒。Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1、mpmath1.3.0。実行時CPU affinity64、利用可能RAM約429GiB。
5つの独立armに対応する5 worker processを使用し、CPU/RAM/BLASの恣意的制限・共有設定変更はしていない。
private state/HDF5/provenanceの原本は読取のみで、operator/state/積分の数値配列を公開しない。

**公開確認後mandatory STOP。旧3 PASS／5 STOPを保持し、後続科学計算へ自動進行しない。**
