# O1 M1：q64・r4/r8の限定減衰検証

2026-11-06、SERVER 2（GPUA100-gleap02）。**固定4点が事前判定を通過し、r増加による減衰軽減を観測した。科学statusは`LIMITED_SIGNAL_PROBE_COMPLETE`、`mandatory_stop=true`。** 1回実行・exit 0で終了し、追加条件・再実行へ進んでいない。shot数・価格・全RPE費用・エネルギー精度／成功確率の保証は評価していない。

## 固定SOURCE・入力・条件

実行SOURCEは[準備commit `8b58cff3679c10aee0dd950874e5e344d0844b7a`][PREP]。既存[限定runner](../../../scripts/run_o1_m1_attenuation_validation.py)、matched split/resource bridge、Gaussian/native builder、独立CAR/Fock・paired-polynomial validatorを変更せず再利用した。runner SHA256は`1b99dae1fc65cbc9e97390aaa6577b149a7413403d0209f307f743cf5fb3bf84`。使用する固定COREはrunner内で準備baseのbyteと照合し、実行commit・SOURCE hash・runtime・入力provenanceを[result](../../../artifacts/track_a_o1_m1_attenuation_science_20261106/run01/result.json)へ保存した。

- linear H4、1 Å、STO-3G、保存DF rank10、H4-Aの共通`T_eff/V_square`、8 system qubits、faithful tail identity、全signed factor・one-body・核scalar。
- 元Hの保存`state`を厳密基底状態と仮定し、状態準備費用を除外。`DF_state`へ置換せず、新state solve・DF fit・rank探索なし。
- `delta=.05`、q=64（T=3.2）、K=2、r=4→8。各rでP/DF両辞書を同時比較する4点のみ。最初の不適格で後続rを停止する固定SOURCE規則を維持した。
- PF/RTE/stat/model/実装/reserveの位相配分は`.06/.03/.15/.03/.01/.02`、`beta_RPE=.30`のまま。stat枠からshot数を導出していない。
- 入力commit `4591b8d75a2a6a55c74a51225f0c61648023c8ee`、path `artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。既存の同byteコピーをhash照合して読み、入力を変更していない。
- Python 3.12、NumPy 1.26.4、SciPy 1.14.1、OpenFermion 1.6.1、Qiskit 1.3.0、qiskit-nature 0.7.2の既存専用環境とGaussian依存をread-onlyで再利用した。任意のCPU/RAM/thread capは追加せず、64 CPU affinity内の既存BLAS設定を利用。rの逐次順はfirst-FAIL契約による。共有設定・他ユーザーのprocess／fileは変更していない。

## normalization・attenuation・実半径

新4点は以下の保存fieldの転記である。`normalization`はq*r全体の積、Aはそのattenuation、観測半径はraw複素信号の絶対値、下界は事前C0 allowanceを含む。r2行は[C0固定報告][C0]からの引用だけで、再実行・保存値再集計はしていない。

| r | 辞書 | normalization | attenuation A | raw観測半径 | 観測半径下界 |
|---|---|---:|---:|---:|---:|
| 2（旧C0） | P | 5.7451518508e10 | 1.740598031e-11 | — | 1.739342668e-11 |
| 2（旧C0） | DF | 3.1414227069e10 | 3.183271063e-11 | — | 3.180975208e-11 |
| 4 | P | 724149.6371145074 | 1.380930057473568e-6 | 1.3808361104046116e-6 | 1.3803483095535042e-6 |
| 4 | DF | 507375.7977914745 | 1.9709257011328482e-6 | 1.9707916156360853e-6 | 1.9700954042463196e-6 |
| 8 | P | 999.2431391646896 | 0.0010007574341075218 | 0.0010007489125511842 | 0.0010000423544962408 |
| 8 | DF | 829.4258410624586 | 0.001205653297127858 | 0.0012056430308613822 | 0.001204791811555364 |

**この限定条件では減衰軽減は成立した。** ただしr8でも観測半径は約1e-3であり、単位半径とは扱わない。r増に伴うevent数・一shot回路価格増は今回測定しておらず、C0の総費用負結果を覆したとは言えない。辞書のnormalization差と、同一物理有限meanの一致も別に扱う。

## PF/RTE複素信号と誤差の区別

両rに共通のPF信号は`0.7965902387066686 + 0.6045196273162331 i`、元H exact信号は`0.7965347265512389 + 0.6045927798096367 i`。PFの元DF exact信号に対するwrapped phase差は`9.183075036387692e-5` rad、PF半径は`0.9999999941066523`で、r変更では外側PFを変えていない。

| r | 辞書 | normalization補正後finite-RTE複素信号 |
|---|---|---|
| 4 | P | 0.7965377452435536 + 0.6044762704679544 i |
| 4 | DF | 0.7965377452435523 + 0.6044762704679567 i |
| 8 | P | 0.7965835662625109 + 0.6045143437666273 i |
| 8 | DF | 0.7965835662625077 + 0.6045143437666298 i |

各辞書のraw複素信号、corrected radius・phase、独立nominal mean、PF／元H／保存DFとのoperator差は[signal_r4](../../../artifacts/track_a_o1_m1_attenuation_science_20261106/run01/signal_r4.json)と[signal_r8](../../../artifacts/track_a_o1_m1_attenuation_science_20261106/run01/signal_r8.json)へ全文保存した。

| r | 観測RTE信号差 vs PF | 観測RTE phase差 vs PF [rad] | allowance込みRTE phase bound [rad] | allowance込みPF phase bound [rad] |
|---|---:|---:|---:|---:|
| 4 | 6.808362478187162e-5 | 2.8045041868764285e-6 | 0.000303574721763975 | 0.00032731382818756885 |
| 8 | 8.51101683516282e-6 | 1.7520208729308182e-7 | 0.0004793048280740036 | 0.0005626225571335285 |

観測RTE phase／複素信号誤差はr8で小さくなった。一方、事前program-length allowanceはrとともに増すため、保守的phase boundはr8の方が大きい。これは誤差改善を否定する値でも、結果を見て許容値を緩めたものでもない。modelと実装のphase boundはいずれもr4で約`0.0002354831`、r8で約`0.0004707918`で固定予算内だった。これらはbinary64/program-scale診断であり、厳密なenclosureや包括的成功保証ではない。

## controlled implementation・事前判定・STOP

各rで既存validatorの257 checksがPASS。one-body half、Gaussian基底・選定Pauli/DF event、scalarと両halfを含む**代表complete controlled step**の相対位相を`diag(I,U)`と照合した。phase alignmentはしていない。代表complete stepのmax-entry差はP/DFでr4 `6.866350197783356e-16 / 7.9407902683204855e-16`、r8 `7.021666937153402e-16 / 6.206335383118183e-16`。q64の辞書meanと独立nominal meanの実装差max-entryはr4 P/DF `4.9389315466122184e-14 / 5.204483501406398e-14`、r8 `7.957007996290967e-14 / 7.960926175286229e-14`である。

PF/RTE近似誤差と上の実装差は別fieldである。例えばfull operatorのfinite-RTE vs PF max-entry差はr4 `0.013749014018757283`、r8 `0.0017294999555100034`で、実装不一致の約1e-14と混同しない。**q64全trajectoryの長回路operatorを完全列挙・認証した結果ではない。**

判定式は[準備報告][PREP]のまま、operator tolerance `4.272926672916701e-10`、basis tolerance `6.548361852765083e-11`、implementation tolerance r4 `4.5987373317266e-7`／r8 `9.192988090446636e-7`。各rの新payloadからetaを作り、radius allowance r4 `0.00011772767569220095`／r8 `0.00023534049511543387`と複素allowance 2etaを適用した。人工atol・旧r2の数値allowanceの転用、閾値の事後変更はない。正半径・asin domain・PF/RTE/model/実装の個別配分と総配分式を全4点が通過した。

[result SHA256](../../../artifacts/track_a_o1_m1_attenuation_science_20261106/run01/result.json)は`f8cea470ea34ba8e4687f94d3268726a466ebd70239b18aa97d319134ac8e8a0`。実行[log](../../../artifacts/track_a_o1_m1_attenuation_science_20261106/run01/execution.log)を同byte保存し、科学process終了・input／SOURCE hash不変更を確認した。新テストや追加科学runを結果後に実行していない。

**mandatory STOP。** shot計算、価格MC、transpile/compile、sampling、全round費用、格子拡張・新fit・rank探索・旧実験retryはしていない。C0の総費用負結果、旧FAIL・SOURCE・raw・grant・marker・STOPと`not_certified`の留保を維持する。UWCや新規性の判断もない。

次の担当：**ユーザー**が、今回の減衰軽減を踏まえて次の検証を行うかと対象条件を判断する。全RPE費用へ進むには、全roundのr_m/K_m・誤差／失敗配分・強DF-PF対照の適格step・価格の不確かさ規則を先に固定し、必要なshot・full no-prep価格・全round科学実行範囲を明示承認する必要がある。**Codex**は本限定成果の公開照合後に停止し、それらへ自動進行しない。

[PREP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b58cff3679c10aee0dd950874e5e344d0844b7a/docs/research/track_a_o1_m1_attenuation_preparation_20261106/REPORT.md
[C0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/docs/research/track_a_o1_m1_c0_parallel_science_20261106/REPORT.md
