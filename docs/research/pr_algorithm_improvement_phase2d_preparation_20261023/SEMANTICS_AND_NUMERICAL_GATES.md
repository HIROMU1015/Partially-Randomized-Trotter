# 意味論・有限bit・数値gate候補

全閾値は**未採択の実行前固定案**。結果は未取得。数値PASSと一般証明、有限signalの統計保証とphysical bias、文書verifierのPASSを区別する。

## 1. 保持するoperator

`R=sum a_i Q_i`、Q_iはfull register上のHermitian involution、`P_i=sign(a_i) Q_i`。時間順products i1,...,inの後にchild jを適用すると、

`U_event=(-1)^(n/2) exp(-i theta_n P_j) P_in ... P_i1`。

`w_n=|tau|^n/n! sqrt(1+(tau/(n+1))^2)`、`B=sum_even<=K w_n`、`Pr(n)=w_n/B`、`theta_n=atan(tau/(n+1))`。componentは独立p_i。`event_probability=(w_n/B) prod p_i`、`event_coefficient=w_n prod p_i`。SOURCEは最初のsampled indexをrotation、その後をproductとして保存する。実適用はproducts→child。`unsigned_rotation_angle` accessorは符号を既に反映するのでsignを二重適用しない。

有限corrected meanの目標は一microstep `B E U=P_(K+1)(-ihR)`。全体は[fixture仕様](FIXTURE_AND_INPUT_IDENTITY_DRAFT.md)のM_K、補正係数 `B_tot=B^(rq)`。これはfirst momentでありchannel等価でもexp(-iTH)一致でもない。physical Hとfinite targetの一致は別gate。

control0とcontrol1の相対phaseを保持する。identity、負product、paired -1、外部scalar、DのI寄与をsystem global phaseとして捨てない。Qiskit `global_phase` を含むactual Operatorを用いる。control1だけをoverlapでphase合わせする検査は禁止。参照規約で決まるphase以外の任意最適化をしない。

各S/PがR全体を表すことと各Pauliのsector保存は別。全16次元でcomponentを適用し、全体Hとmeanの2粒子sector restrictionを比較する。中間projection／sector内に切ったPauliをunitaryとすることは禁止。basis/parityは非制御、中央と必要phaseは制御。adjointはactual inverse。

## 2. 閾値のSOURCE上の先例と限界

起点の `rte.py` はprobability検査atol1e-12、config整合rel1e-14/abs1e-15、toy involution検査validation_atol1e-10、coefficient_atol既定0を持つ。`tests/test_rte.py` の有限mean比較は1e-13級。`tests/test_df_rte_qiskit_builder.py` は小eventのatol1e-12級だが一部np.allcloseの既定rtolを残す。repeated builderのfusionテストも1e-12級。旧toy constructionのATOL2e-10も確認した。これらを新fixtureへの保証として転用しない。exact zero policyとoperator toleranceも同じ値にしない。

binary64 unit roundoffは約1.1e-16、system16／controlled32次元。新fixtureは四microsteps、複数basis、D/c、compile分解を含むため、単一eventより蓄積誤差の余地を取る。一方、signal allowance .025とは十分離す。下表はこの小系のengineering acceptanceで、gate数依存の厳密forward-error boundを証明したものではない。新しいgate数が想定外でも閾値を自動緩和しない。

| gate | 候補acceptance | 併記する残差 |
|---|---|---|
| exact係数／zero／sign／input identity | exact一致 | 浮動閾値で代替しない |
| H、Hermiticity、Q_i²=I | max-entry absolute <=1e-11 | Frobenius、spectral normも報告 |
| raw/optimized/adjoint controlled operator、control0/off-diagonal | max-entry <=1e-10 | control block別、unitarity |
| unitarity U†U-I | max-entry <=1e-10 | full32次元 |
| physical/catalog corrected mean | max-entry <=2e-10 | one-step／全体、独立reference差 |
| X/Y信号、未compile | complex absolute error <=2e-10 | Re/Im別、state norm |
| compiled operator／compiled X/Y | max-entry／complex absolute <=5e-10 | layout、QC.global_phase |
| p／event probabilityの各値・mass、丸めL1 | <=1e-12 | negative/NaN/Infは即失敗 |
| B／角度／coefficientのfinite-bit整合 | abs <=1e-12*max(1,reference magnitude) | 高精度値とdouble差、集約誤差 |
| phase角の追加診断 | wrapped angle <=1e-10 rad | magnitude>=1e-8時だけ。主判定はcomplex operator差 |
| literal recipe整数count | exact一致 | 各event・各reserve別 |
| DP vs直接列挙期待price | double abs <=1e-10*max(1,price) | 90/128digits集約差 <=1e-40*max(1,price) |
| 90/128digits独立reference整合 | max-entry／complex abs <=1e-60 | 最初から2評価を登録、増精度retry禁止 |

全array比較は明示rtol=0。normを計算する検査自体も将来の科学call内である。degree3 polynomialは独立factorial係数、Dやexact exponentialの参照は固定mpmath経路を予定する。mpmath1.3.0の[expm公式tagged SOURCE](https://raw.githubusercontent.com/mpmath/mpmath/1.3.0/mpmath/matrices/calculus.py)にTaylor/Padeがある。二つの高精度結果の近さは、厳密interval enclosureやbias certificateの代用ではない。

## 3. Selector区間とceil

lambdaはexact有理数から得る。sqrt/log/atan、B/N/Phi/Jの区間にはoutward-rounded scalar providerを固定し、SOURCE/envと一緒に承認前に検査する。候補はmpmath1.3.0の[interval context](https://raw.githubusercontent.com/mpmath/mpmath/1.3.0/mpmath/ctx_iv.py)。必要関数のenclosureとceil境界の扱いを実際に確認していないので、現在provider適格性は未確定。

N式の区間が整数境界を跨ぐ場合、未確定を丸めて隠さず `SHOT_COUNT_UNRESOLVED` で停止する。Jの区間が重なる場合は優位性未確定として現候補を保持する。ordinary二精度一致を「認証区間」と呼ばない。精度を結果に応じて増やさない。interval providerを完成させられない場合は価格認証／selectorを `UNPRICED_FALLBACK` とし、承認範囲の再判断が必要。

## 4. 有限bitと統計の資格

exact理想law、doubleで保存したlaw、actual circuit meanを区別し、丸め／角度／compilerのそれぞれを比較する。probability normalizationはSOURCE同様p/sum(p)を扱い、その差も記録する。最終的なquantum IID shotが使うlawはbindingされたfinite-bit lawであり、理想lawとの差を無視しない。F2のfinite-target allowanceの適格性には、これら実装差と残るbiasのbudget割当が必要。小さいmatrix residualだけで一般保証を宣言しない。

N policyは複素finite meanに対する候補 `eta_stat=.025, alpha=.05`、axis allowance eta_stat/sqrt(2)、axis failure alpha/2。
`N_X=N_Y=ceil(4 B_tot^2 / eta_stat^2 * log(4/alpha))`。
fresh IID quantum outcomes ±1、fresh canonical event draws、補正後range[-B_tot,B_tot]、固定state／time／axis／B、任意のtrajectory cherry-pickingなし、という条件を必要とする。今回もF2も量子measurement shotsは実抽選せず、Nを条件付き必要数として報告する。

有限targetの統計allowanceと `|z_K-z_exact|` のbias allowance候補 .025は別。もし真signalへの厳密な同精度主張が必要なら、採用済みbound／enclosureとfinite-bit・compile誤差の合成を承認前に定める。90/128digitsの小系bias診断は `NUMERICAL_DIAGNOSTIC_ONLY` であり厳密certificateではない。統計資格が欠ければ同精度比較は不適格。physical bias資格が欠ける場合はfinite-target条件付き比較までに制限し、「真signalの同精度資源改善」と表示しない。

どのgateも未実行。失敗、未登録変更、NaN/Inf、不明phase／layoutはSTOP。成功・失敗を問わずF1後とF2後に各mandatory STOP。
