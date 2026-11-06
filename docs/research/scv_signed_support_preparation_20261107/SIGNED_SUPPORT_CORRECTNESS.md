# Artificial correctnessと限界

2026-11-07 JST / SERVER1。**新規24＋既存82＝106件PASS、local人工検証。実H4の改善証拠ではない。**

[最終log](ARTIFICIAL_TESTS.log)、[機械可読結果](TEST_RESULTS.json)、[SOURCE/環境](SOURCE.json)、[新テスト](../../../tests/tracks/algorithm_codesign/test_scv_signed_support.py)。

## 人工fixture

旧 `test_scv_scalar.metadata/enumerated` を再利用した、発明したdyadic lawのみ。orderは0/2に質量3/4,1/4。label a,b,c,zeroの質量は1/4,1/4,1/2,0で、2^53 gridに厳密に乗る。3-qubitの架空catalogでf(a)=−1/2,f(b)=1,f(c)=−1,f(zero)=1/2とした。identity空supportや全label同一featureの別fixtureも用意した。

人工の正path全列挙で5層のν・m₂・Σを比較し、すべてFractionの厳密一致。代表的にν0=−1/2、ν1=−1/3、ν2,3,4=−3/8。m₂0=1/4、m₂1=1。これらは**人工fixtureの値**であり、H4の値ではない。実装側は全path列挙せず、一位置和で計算する。

OLSへ渡す応答は、別々のdomainを持つ人工metadata・固定bit表だけ。真のBorn/path信号は入力しない。中心化の代数恒等式のテストだけで任意の解析的toy response表を使い、これをOLSへ渡していない。fixture内の件数・係数はテスト用で、将来の科学条件を採択するものではない。

## 新規24テストの確認対象

| 対象 | 確認内容 |
|---|---|
| metadata/feature | 同じlabelのsign/support、n0/L0・n2/L2、明示rotation不一致拒否、正負/半整数/identity、絶対値・角度の不使用 |
| identity | LAW/catalog hash、H/preparation、完全label対応・順序、zero-grid保持、不正support/sign/qubit、入力の防御的コピー・変更検出 |
| moments | 全5層の全列挙とのν/m₂/Σ一致、h0定数・h1非定数、h2–4の自由rotation、zero-grid寄与0、sampling/path列挙なし |
| centering | 全5層でE_Q[u−ν_h|h]=0、任意固定slopeに対する平均の保持、h1をglobal平均で代用しない |
| fit/freeze | 既存OLS一回・rank2、定数catalog/定数trainのrank不足、finite slope、binary64→Fraction固定、refit/clipなし、他4層λ=0 |
| range | λの正/負/0とfeature全端点の包含、width=2+2|λ|、負featureの補正、不正box/mean/nonfinite拒否、旧indicator model拒否、誤ったenclosureでfail-closed |
| fresh paired | train/eval domain分離、axis/scope/weight/cell/全5quota検査、同じrecordで両平均・variance・covariance・差のvarianceが手計算と一致、評価中の再fitなし |
| 費用/STOP/境界 | 人工価格は物理一回、train込み、cache hit課金、重複拒否、既存STOP後の入口拒否、実loader/RNG/engine/Born/buildに依存しないpure adapter |

### Indicator rangeを流用できない具体例

λ=1,ν=−1/3,B=1,u=−1ならZ=5/3。誤ってu∈[0,1]を使うと上端2/3で包含に失敗する。新adapterは[−1,1]から正しい上端5/3を得て、[0,1]指定と旧indicatorモデルの混入を拒否する。これは数学的rangeのcorrectnessであり、許容guard値の採択ではない。

### 不偏性と独立性の意味

trainingを条件としてλが固定され、fresh評価が元のQ(·|h)に従い、ν_hがそのexact平均なら

\[
E[Z_h\mid training]=E[B_h\mid h]-\lambda_h(E[u_h\mid h]-\nu_h)=E[B_h\mid h].
\]

したがってΣ_h p_h E[Z_h|training]は元の有限target。最適λとの一致は要求しない。人工テストはexact centeringとdomain拒否・frozen計算を確認した。将来の実PRNG独立性や実際の測定分布を今回立証したわけではない。

## 再利用した82テスト

| module | 件数 |
|---|---:|
| test_scv_scalar | 17 |
| test_scv224_pilot | 17 |
| test_a1_control_variates | 15 |
| test_a1_n3_connection | 22 |
| test_a1_n3_campaign | 11 |

conditional取得、旧CV、scope/stream、full-price/cache、mock bit先行保存、journal/reconcile、STOPを回帰確認した。既存テストの小規模人工回路・toy engine検査を含み、実H4 circuit/stateは投入していない。科学worker・managerは起動していない。

## 実行記録と再現

初回24件中7件がrank確認のobject dtypeでERRORになった。[初回log](ARTIFICIAL_TESTS_INITIAL.log)を保全。Fraction配列をNumPyへ渡す箇所を既存OLSと同様の明示float配列へ修正した。moment・feature・frozen range演算はexactのまま。[修正後24件log](ARTIFICIAL_TESTS_FIXED.log)と最終106件logを保全した。

```bash
cd /tmp/o2-scv-signed-support-preparation-20261107
MPLCONFIGDIR=/tmp/scv-signed-artificial-cache-20261107/mpl \
NUMBA_CACHE_DIR=/tmp/scv-signed-artificial-cache-20261107/numba \
XDG_CACHE_HOME=/tmp/scv-signed-artificial-cache-20261107/xdg \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:tests/tracks/algorithm_codesign \
/home/AbeHiromu/projects/partially-randomized-trotter-wp11/.venv-server-opt2/bin/python -B -m unittest \
  test_scv_signed_support test_scv_scalar test_scv224_pilot \
  test_a1_control_variates test_a1_n3_connection test_a1_n3_campaign -v
```

Python3.12.3、NumPy1.26.4、Qiskit1.3.0の既存venvをread-only再利用。cacheは本作業の/tmpへ分離。CPU/RAM/thread上限の追加なし。最終suiteは106件、3.846秒でPASS。localでの結果をCI・独立再現とは呼ばない。

## 実科学でUNKNOWNの事項

実H4でのν/Σ・非定数性、rank/係数・range幅、X/Y相関、残差分散、学習費回収、総RZ改善、十分shot数、期待費用保証は未検証。旧224のbit・係数を再解析していない。新しいshot数・seed・guardは未設定。人工PASSは科学的GO/NO-GOではなく、実H4への実行許可も含まない。
