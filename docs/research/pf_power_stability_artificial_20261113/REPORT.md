# PF macro-step/powerの数値安定性：人工入力による技術調査

2026-11-13 JST。**人工36次元行列の192条件で、元spectrumの再構成gateを通過してもmacro-step/powerのunitarity gateが失敗する経路を再現した。直接expmは今回の48条件すべてで元の1e−9 gateを通過した。実H4のrun02の原因を確定した結果ではなく、production backendは変更していない。**

[全測定値](RESULT.json)／[集計](SUMMARY.json)／[人工実行ログ](ARTIFICIAL_RUN.log)／[新8テスト](TESTS.txt)／[旧9回帰](REGRESSION.txt)／[HANDOFF](HANDOFF.md)／[次の有限科学案](NEXT_SCIENCE_PLAN.md)。基点`699178305cec241b83220f6c307c65e0539a7856`の既存A branchを継続した。実Hamiltonian・保存state・保存signal/数値結果の新解析は0。

## SOURCE確認と再利用

- `e417ffc9f52b13c18f5493193ce510d858664259:scripts/h46_new2_core.py:spectrum`を既存`pr_rpe_power_mean.kernels`からそのまま使用。Hermiticityと`V diag(λ) V†`の再構成誤差を検査するが、`V†V−I`は検査していない。
- 固定科学SOURCE`256caad3414a80de02fb68eca89c13c957ed6354`の`h4_four_method_screen_mean.deterministic_step`を人工ctxで再利用。S2・標準Y4・M5 bestのweightsは既存`h46_m5_plan.weights`を使用し、forward/reverse block順・負時間とmatrix_powerを保持した。
- 既存`algorithm_codesign/numerics.py:Spectrum`のconstructorへ**同じ供給eigensystem**を渡し、Gramと再構成の条件付きguardを観測した。別の対角化に置換していない。同classのaction/finite-RTEや旧上限、Hamiltonian誤差契約を今回へ採択していない。
- `c_j_u_stable`等の既存安定性処理も確認したが、最適化gradientの問題であり今回の指数関数へ流用しない。Bのdecoder/統計器、C/Dの価格側は変更していない。

実行環境はNumPy1.26.4/SciPy1.14.1。SciPy 1.14.1の標準Hermitian問題は既定driverがEVRで、EVD・EVも明示指定できる。[SciPy公式仕様](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.linalg.eigh.html)。同仕様はEVDが常に最も正確と保証していない。Hermiticityはlibrary任せにせず、元SOURCEの検査を保全した。

## 人工実験の定義

物理分子ではない36×36 complex Hermitian行列を11個用意。QRによる人工basisから生成し、生成時だけHermitian対称化した。regularは固有値linspace[−1,1]、clustered_zeroは零24・1e−12が6・1が6、block jの倍率は(j+1)/55。固定seedは20261113/20261114。人工stateも別の乱数から生成・初期正規化した。実H4の固有値・DF係数・stateを参考に調整していない。

2 fixture × S2/Y4/M5 × Δ.1/.4 × q1/64/1024/4096 × 4経路＝192条件。経路は元archived spectrum（EVR）、明示EVD、明示EV、直接SciPy expm。expmも同じ22×piece個のblock適用順を用い、共通半stepをcacheした。積公式のexact full-Hとの誤差ではなく、同じPF recipeの数値実装同士を比較している。

全経路でq乗は同じ`np.linalg.matrix_power`。power後のstate normとoperator unitarityの**閾値は元の1e−9のまま**。projection、Gram-Schmidtによる固有vector修正、極分解、伝播後の再正規化、閾値緩和、失敗後の自動fallbackはしていない。代替経路は事前に宣言した対照である。

## 測定結果

| 数値経路 | gate失敗/48条件 | 最大state norm誤差 | 最大power unitarity誤差 | expm powerとの差の最大Frobenius norm |
|---|---:|---:|---:|---:|
| 元archived EVR | 7/48 | 5.21616084e−11 | 3.72406942e−9 | 1.86136677e−9 |
| 明示EVD | 4/48 | 1.38598688e−10 | 1.79021590e−9 | 8.95914578e−10 |
| 明示EV | 1/48 | 4.92126340e−11 | 1.18466847e−9 | 5.94014086e−10 |
| 直接expm | 0/48 | 1.21014310e−12 | 3.29568474e−11 | 自己比較0 |

元EVRの失敗はq1024に2件、q4096に5件。q1/q64は全経路で通過した。ordinaryな2 fixtureではstate norm gateは全192条件で通過し、失敗は全36次元operatorのunitarity gateに現れた。これは検査を削除する根拠ではない。元EVRの最大spectral再構成誤差は4.33973370e−15、最大Gram defectは4.92155896e−14で、各spectrumの旧gateは通過していた。

192条件は約.688秒で完了した人工観測で、実H4の所要時間や普遍的なbackend順位ではない。EVD/EVにも失敗が残るため、単なるdriver切替を修正済みと呼ばない。直接expmを、同じ物理recipeの**次の限定検証の優先候補**とする。実H4への採用は未実施・未資格。

## 確認できた機構と限界

既存のspectral factorは`V diag(exp(−itλ)) V†`なので、t=0では`V V†`になる。λ=0の部分空間では`V diag(λ) V†`の再構成だけではGram defectを検出できない。人工対照としてA=0、V=Iにoff-diagonal 3e−12を手動注入した。これはLAPACKの実出力ではなく、旧gateの観測不足を示す対照である。

この対照は再構成誤差・固有方程式residualとも0、Gram defect4.24264069e−12。S2のq1ではunitarity1.69705627e−11で通過し、q4096では6.95114250e−8で失敗した。M5もq1では通過、q4096では7.64625675e−7で失敗した。同じpower/signed順だけで誤差が増幅する十分な機構を確認したが、run02の6失敗がすべてこれに起因したとは断定しない。

実H4の失敗残差は未保存のまま。本調査で再計算して補っていない。丸め、basisの直交誤差、macro-stepの積、powerの寄与は実H4ではまだ分離できない。局所phase・energy RMSE・十分Nやwinnerへの接続も未評価。

## 最小限の実装・検証

`pf_power_stability.py`を独立したdiagnostic helperとして追加した。供給matrixとspecの観測、明示driver、同順expm、`power_diagnostics`が対象。最後の関数は失敗値を返してからcallerがSTOPできるため、元run02のように例外だけで残差を失わずに済む。failed結果を修復したりPASSへ変更したりしない。既存screen・normalizer・sampling law・SOURCEを変更する差分はない。

`check_pf_power_stability_artificial.py`は人工fixture専用で、分子入力optionを持たない。新8テストと旧screen9テスト、計17技術test PASS。手動注入対照、失敗値保持、signed順、driverの明示、既存STOP/rawのbyte保全を検証した。人工PASSは実H4の精度達成や科学的原因の確定ではない。

次は**ユーザー**が[代表1セルの二経路診断](NEXT_SCIENCE_PLAN.md)を承認するかを判断し、**Codex A**が承認後だけ新しい診断SOURCEで実行する。**B**は数値安定性の未解決と元4局所recordを保持する。今回の技術成果をcommit/pushしremote SHA/tree/主要成果物を照合した後、mandatory STOP。実H4・残16セル・全13roundへ自動進行しない。
