# H4 DF10：長時間PFの二経路数値診断

2026-11-13 JST。**F2-COARSE/m12の二経路を各1回完了した。元spectral経路はpower後のoperator unitarity誤差1.0071230339680433e−9で固定1e−9 gateに失敗し、直接expmは5.899874925253737e−12で通過した。complex meanの経路差は4.313865617047318e−11。閾値・物理recipeは変更していない。**

この代表1点では、spectral basisの有限精度に由来するblock/macro-stepの非unitarityがpowerで蓄積する経路が観測され、直接expmが有効だった。旧run02の他5失敗、M5/B0I/Full-H RTE、全RPEの精度を検証したものではない。[全原結果](run01/result.json)／[要約](SUMMARY.json)／[B HANDOFF](HANDOFF.md)／[次の再開案](RESTART_PROPOSAL.md)を参照。

## 科学条件・SOURCE・実行範囲

Linear H4/1.00 Å/STO-3G、保存tol-only DF10・weighted-Hermitian completion、sector(2,2)/36、保存H/state/snapshot。Full PFはone-body＋全10fragment、S2/prefix10、T409.6、Δ.4、qPF1024。候補IDは`4f0aca817edccc7fbe17`。q_RPE=4096とは別。fragment順、scalar一回会計、元prior/decoder/予算・P0、数値閾値1e−9を保持した。

| 用途 | 固定SOURCE |
|---|---|
| recipe/元screen・loader | `256caad3414a80de02fb68eca89c13c957ed6354` |
| 人工調査・diagnostic helper | `dd24dd329949c255d5e37b90ff8993fa8df7159b` |
| 二経路の起動・結果保存入口 | `ade6438f8babcb1671294246ec9756558d255289` |
| 元spectral kernel | `e417ffc9f52b13c18f5493193ce510d858664259:scripts/h46_new2_core.py:spectrum` |
| reference carrier | `bba150834ca30bdb186fbfd58c55a8d51177b855`の`PHYSICAL_INPUTS.json:rounds[12].diagnostics.reference_pair`、元診断出典は[REFERENCE.json](run01/REFERENCE.json)に保存 |

元二資料のコードを変更せず、最小の起動入口を人工テストしてcommitへ固定した。今回のSOURCE同定と承認証明は別で、新grant等は発行していない。[SCOPE](run01/SCOPE.json)は物理候補・binding/profile・各依存SOURCEを保持する。binding内の旧T=.8はreference/profile用の履歴fieldで、今回の発展時間はcandidate.T=409.6。

既存loaderで保存snapshotを1回復元、11blockの元spectrumを1回ずつ評価。direct経路は同blockのhalf-step expm11個をcacheし、22適用の同じS2 macro-stepを構成した。両経路は同じmatrix_power(q1024)を1回ずつ使用し、保存constantをtotal Tで各1回だけ適用。新reference・finite-RTE normalizerは0、complex meanは合計2。他セル、新SCF/DF/state/ground計算、sampling、回路compile、N/RMSE/C/Gは0。

2026-11-13 16:39:46〜16:39:48 JST、約2.60秒。NumPy1.26.4/SciPy1.14.1/complex128、`src/`＋`scripts/`のimport経路を使用。[EXECUTION](run01/EXECUTION.json)と[ログ](EXECUTION.log)を保全。36次元の二経路で共有入力・cacheを使い、追加のworker制限や共有環境の変更はしていない。

## 取得した二経路の値

| 観測 | 元spectral | 直接expm |
|---|---:|---:|
| macro-step unitarity Frobenius誤差 | 2.055427880094077e−12 | 7.81494053477498e−15 |
| macro-step後のstate norm | 1.0000000000000409 | .9999999999999989 |
| power後のoperator unitarity Frobenius誤差 | **1.0071230339680433e−9** | **5.899874925253737e−12** |
| power後のstate norm | 1.0000000000419915 | .9999999999988605 |
| power後state norm誤差（scalar前） | 4.199174341579237e−11 | 1.1396439347777232e−12 |
| state norm gate ≤1e−9 | 通過 | 通過 |
| operator unitarity gate ≤1e−9 | **失敗** | **通過** |
| complex meanの実部 | .8565499611146139 | .8565499610772614 |
| complex meanの虚部 | .5160206749185224 | .5160206748969415 |

元経路のgate失敗を診断値として保存した後、計画済みdirect経路も実施した。自動fallbackや旧結果の救済ではない。失敗した元meanは比較診断専用であり、元screenへ有資格recordとして戻さない。raw=corrected、PFのB1/logB0/attenuation1を両経路で保持し、再正規化・投影はしていない。

| 経路間の差 | 値 |
|---|---:|
| macro-step operator差 Frobenius | 1.0276927492497617e−12 |
| power operator差 Frobenius | 5.036635696065883e−10 |
| scalar適用後のstate-vector差 | 4.314317586473168e−11 |
| complex mean差の絶対値 | 4.313865617047318e−11 |

任意の新しい許容差を採択した比較ではない。両経路の値・差をそのまま保存した。旧1e−9 thresholdを緩めず、元operator gate失敗の意味も変更していない。

## 原因について確認できた範囲

全11blockは元spectrumのHermiticity/再構成gateを通過した。新たに同じeigensystemを観測した最大値は、再構成5.196086146441816e−13、固有方程式residual2.465242351000783e−14、Gram defect3.507540452292337e−13。各blockは[SPECTRAL](run01/SPECTRAL.json)、half-stepの比較は[BLOCK_FACTORS](run01/BLOCK_FACTORS.json)に保存した。direct expm自体には固有ベクトルがないので、その直交性・spectral再構成はnull/非該当とし、元specの観測をdirectの独自対角化結果とは呼ばない。

特にblock8（one-bodyをblock0とする）では、再構成誤差7.534508391379821e−19が小さい一方、Gram defect3.507540452292337e−13、spectral half-stepのunitarity誤差7.015020372937031e−13が観測された。同じblockのdirect expmではunitarity7.691850746060767e−16だった。

`V exp(−itΛ) V†`ではGram defectが伝播へ入るが、`V Λ V†`の再構成では小さな固有値に重み付けされる。今回のblock/macro-step/power観測は、再構成gateだけでは見えにくいbasis直交誤差と行列積の丸めが、反復後のoperator gateへ現れる機構を支持する。同じ物理recipeで直接expmを用いるとmacro-step・powerの非unitarityが減少し、固定gateを通過した。

これは代表1点の数値実装差の診断であり、各寄与の厳密分離や唯一原因の証明ではない。SciPy一般の不具合、元PF公式の不成立、旧run02の全失敗の原因確定とは呼ばない。原子・分子の条件やPF公式を変えずに、今回の数値gate失敗を回避できる経路を確認したという範囲に限る。

## referenceと物理的なsignal誤差は別

保存referenceは`.1470900528467786 + .989123104742252i`。独立pairの元1e−10 gateを確認したが、新referenceは取得していない。direct meanの保存referenceからのcomplex biasは`.7094599082304829 − .4731024298453105i`で、元経路も近い。**数値unitarityの改善は、この粗いS2長時間候補の物理signal biasを小さくしたことを意味しない。**

局所complex biasをTで割ったenergy bias、全RPE RMSE、十分Nへ変換していない。state/ground、actual law、prior/decoder等の科学的資格も変更せず、N/global RMSE/winnerはnull。直接expmの局所gate通過を、候補の精度達成や資源最小性の認定に使わない。

## 再開についての提案

**残りの限定screenは、別途承認後に再開することを推奨する。** PF/B0Iのmacro-stepには直接expm＋既存matrix_powerを優先し、signed weights・適用順・scalar・1e−9 gateはそのまま維持する。これは数値実装方式の推奨であり、coarse S2等の科学的候補採択ではない。M5/B0Iは人工照合済みでも、実H4の今回1点で全面保証されたとは扱わない。

元run02の4セルと今回direct1点をSOURCE・資格の差を保持して再利用する場合、未完は**15セル＝Full PF7＋B0I2＋Full-H RTE6**。今回direct点の共通screen packet/allowanceへのbridgeは未評価で、保存値を無断で再診断しない。full-H RTEはPFではないため、finite-K polynomial/normalizer/raw/corrected lawをdirect PF指数関数へ置換せず、元の独立した経路を維持する。[再開案](RESTART_PROPOSAL.md)に有限scopeとSTOP条件を記す。

今回の新4人工＋旧8回帰＝12技術testはPASS。6項目のSOURCE・新record・旧raw保全確認も実施した。元run01/run02、固定SOURCE/input/raw/結果/KEEP/STOP、B/C/Dの作業は保全。commit/push・remote SHA/tree/主要成果物確認後、mandatory STOP。残15セルや全20/13roundへの実計算は開始しない。
