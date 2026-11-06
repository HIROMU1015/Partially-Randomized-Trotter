# Track A B2 P1：全18roundの条件付きshot会計

2026-11-07 JST / SERVER 2（GPUA100-gleap02）/ Codex主エージェント。

**全18roundのshot算術を実行した。X軸3,902,892、Y軸3,902,892、合計7,805,784 shots。全点適格で`ALL18_CONDITIONAL_SUFFICIENT_SHOTS`。** m17は7,688,270 shots、総数の約98.4945%を占める。P1のengineering allowance、理想fresh IID・bounded outcomesを前提とする十分shot数で、`not_certified`を維持する。価格・総回路費用は算出していない。**結果公開後mandatory STOP。**

## 固定SOURCE・入力・条件

実行SOURCE **[`05015ae5ceeacc747f88d848cce8c3558a4881da`][SOURCE]** を人工テスト後・実行前にcommit公開し、remote/local treeと4変更ファイルのbyte一致を確認して1回実行した。科学実行時間.207秒。

- [`result.json`](../../../artifacts/track_a_b2_p1_shots/2026-11-07/run01/result.json)：全18roundのPF半径下限・corrected disk・B/A・raw半径下限・N_X/N_Y・総和・寄与・条件付きstatus、SOURCE・入力blob/SHA256・来歴・留保。
- [`rounds.csv`](../../../artifacts/track_a_b2_p1_shots/2026-11-07/run01/rounds.csv)：round別shot数と会計入力、原信号SOURCE、結果commit、exactな総shot寄与比。
- [実行前PLAN](PLAN.md)。旧SOURCE・保存信号・branch結果・STOP・private入力は不変更。

| 固定成果 | 使用したもの・今回の扱い |
|---|---|
| 全18round信号 | `d7a55ac181de71cd2b2c49644810373cafd0a450`の`artifacts/track_a_b2_p1_remaining_rounds/2026-11-07/run01/all_18_rounds.json`および同runのresult.json |
| m0/17原結果 | `ea2c0915ba2f7e11a6dac58dc01cf1af3c0d0ff5`の`artifacts/track_a_b2_p1_two_round/2026-11-07/run01/result.json`。原実行SOURCE `5ef55f923e73e61bada383d0cfe98283e7984e9d`を保持 |
| m1..16原SOURCE | `071cb00955379b0bf4da149bd9793d4125968cad`。endpointとは別fieldで来歴を保持 |
| branch/energy PASS | [`d424bb8…`][BRANCH]の保存result.json。全18round・同じ入力/条件、参照した信号のcommit/path/blob/hashとSOURCEを照合。decoder・energy確認は再実行していない |
| P1正本 | [`7bc9527…`][P1]。H_phys、Linear H4/1.00 Å/STO-3G、DF12、generation LD3、partial-S2/r4/K2、元Hの保存基底state、Nα=Nβ=2/sector36、prep費用除外 |

P1のε_E=.00015936001019904 Ha、95%目標、β_RPE=.4、δ=.02、m0..17/Q=2^m、prior[−128,128] Ha・shift0を維持した。位相capはmodel .02/PF .06/RTE .04/stat **.24**/implementation .02/reserve .02 rad、各軸α=.05/36。u_z/u_radius/u_normalization=1e-8、u_phase=1e-8、独立照合1e-9を変更・再配分していない。

元H/state/DF/sector/orderの固定identityは次と一致した。今回確認したのは**公開保存結果と原SOURCE間の結合**で、private配列を読み直した科学検証ではない。

| 対象 | 固定identity |
|---|---|
| raw H_phys / S0 array hash | `bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2` |
| 元H full state / S0 array hash | `1a2df11b16743e68fb27a3f4b39e78e145897cb49a8cc931b080e44e7400c1db` |
| DF Hamiltonian hash | `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424` |
| private state file / HDF5 SHA256 | `deb561367e216fd4c62d013c10a4adbbef5da8c3e8139c39170d71f80530a80c` / `a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958`（保存metadataの照合のみ） |

interleaved α/β、Qiskit LSB、OF sector basis hash、原signalの全field・m/Q/Tと条件付きPASSを照合した。DF/PF固有stateへの置換はない。保存branch結果が参照した信号hashも今回の入力と一致した。

## 既存会計の再利用と資格確認

新規は保存JSON adapter、runner、人工test、PLANの4ファイルだけで、既存科学コードは不変更。既存`validate_saved_bundle`による原結果照合の後、**保存5信号のscalarから`B2SignalRecord.bounds`を同じuで照合し、保存boundsと全field一致**を確認した。新しい信号生成・PF/RTE simulationではない。

保存λ_R=.5832704246339983、τ=.0029163521231699912、b_event=1.0000085051026735とK2/r4を既存`RTEConfig`/canonical finite distributionに結合した。tail hashは保存済みの参照で、回路preparation・samplingは行わない。各roundで既存`account_radius_round`を**実際に呼び**、その`radius_phase_diagnostics`→`finite_rte_signal_validation._hoeffding_shots`を使用した。

既存式は、

\[
\rho_{\rm raw,lb}=A(\rho_{\rm PF,lb}-\epsilon_{\rm corrected}),\qquad
\eta_{m,b}=\rho_{\rm raw,lb}\frac{\sin(\beta_{\rm stat})}{\sqrt2},\qquad
N_{m,b}=\left\lceil\frac{2}{\eta_{m,b}^2}\log\frac{2}{\alpha_{m,b}}\right\rceil .
\]

式・ceil→Python intの処理は再実装していない。会計に渡すのは**attenuation前のPF半径下限とcorrected disk**であり、既に減衰したraw半径をPF半径として渡していない。会計が返したA・raw半径下限・RTE位相条件は保存値と完全一致した。正半径、各capとreserveを含む総枠も確認した。

APIの必須`cost_metric='rz_count'`は互換用の識別子に限り、**cost=None**で価格処理は呼ばない。価格/総回路費用はnullで保持し、価格がないことを費用0としない。model上界や実装誤差の厳密証明を新たに認定せず、明示された条件付き会計として実行した。

**人工8件PASS（2.87秒）。** 小さい人工boundsによる実API接続・Aの一回適用・価格なし、非正半径/予算超過拒否、重複/欠測/合計不整合/float整数/α変更の拒否を確認した。`10^40`を超える架空整数でもJSON/CSVと独立Decimal和が一致。219件・信号simulation・branch検証は再実行していない。

## 全18roundのshot結果

X/Yは同じ統計位相予算・軸別α・半径による等方的な十分条件を使うため、各roundで同数となる。raw平均の実部と虚部が同じという意味ではない。shot数は丸め表示せず整数で示す。完全な会計入力と正確な寄与分数はJSON/CSVに保存。

| m | Q | T(Ha^-1) | raw半径下限 | N_X | N_Y | round合計 | 総shot寄与(%) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | .02 | .999965960313 | 515 | 515 | 1030 | .01319534 |
| 1 | 2 | .04 | .999931941783 | 515 | 515 | 1030 | .01319534 |
| 2 | 4 | .08 | .999863908195 | 515 | 515 | 1030 | .01319534 |
| 3 | 8 | .16 | .999727854903 | 516 | 516 | 1032 | .01322097 |
| 4 | 16 | .32 | .999455803853 | 516 | 516 | 1032 | .01322097 |
| 5 | 32 | .64 | .998911923814 | 516 | 516 | 1032 | .01322097 |
| 6 | 64 | 1.28 | .997825051454 | 518 | 518 | 1036 | .01327221 |
| 7 | 128 | 2.56 | .995654853300 | 520 | 520 | 1040 | .01332345 |
| 8 | 256 | 5.12 | .991328606921 | 524 | 524 | 1048 | .01342594 |
| 9 | 512 | 10.24 | .982732426684 | 534 | 534 | 1068 | .01368216 |
| 10 | 1024 | 20.48 | .965763041939 | 552 | 552 | 1104 | .01414336 |
| 11 | 2048 | 40.96 | .932698271949 | 592 | 592 | 1184 | .01516824 |
| 12 | 4096 | 81.92 | .869926084143 | 681 | 681 | 1362 | .01744860 |
| 13 | 8192 | 163.84 | .756771407009 | 899 | 899 | 1798 | .02303420 |
| 14 | 16384 | 327.68 | .572702973939 | 1570 | 1570 | 3140 | .04022658 |
| 15 | 32768 | 655.36 | .327988702984 | 4786 | 4786 | 9572 | .12262702 |
| 16 | 65536 | 1310.72 | .107576591453 | 44488 | 44488 | 88976 | 1.13987269 |
| 17 | 131072 | 2621.44 | .0115727232601 | **3844135** | **3844135** | **7688270** | **98.49452662** |
| 合計 | | | | **3902892** | **3902892** | **7805784** | 100 |

全18roundのPF半径下限と誤差diskは既存の値を保持し、最終点でもρ_PF,lb=.999999989619209、corrected disk=7.463994425252446e-8。これにA=.0115727242440354を一度だけ適用したraw下限=.011572723260113877をshot式に用いた。

初期roundでは各軸515 shots程度だが、長roundでAが低下し、式の`N∝ρ_raw,lb^−2`に従って急増する。m17のB=86.41007760254907に対応して必要shot数が支配的になる。これは**固定r/K/δと固定予算の観測**であり、最適性・強PFに対する費用優位/劣位・r/K変更の根拠を認定したものではない。1 shotの価格は未取得である。

## 集計・失敗確率・留保

重複・欠測のないm0..17を要求し、各roundでN_X+N_Y=N_round、全round和=全X和+全Y和を整数で確認した。JSON/CSV保存後にも独立に再読込し、CSVの数値文字列からDecimalでX/Yを逆順を含め加算して、**3,902,892＋3,902,892＝7,805,784**と一致した。計数をfloatへ変換せず、寄与もexactな有理数を保存している。

P1指定のα=.05/36は数学的に1/720で、36軸の配分和は厳密に**1/20=.05**。既存float APIへの値は.001388888888888889のまま保持し、binary64のfsumも.05だった。統計予算.24、他成分cap、reserve、uを変更していない。

| 資格・保証 | 今回の状態 |
|---|---|
| 保存信号/bounds＋P1による条件付き十分shot数・合計 | **全18round PASS** |
| 理想fresh IID canonical trajectory・各軸±1 bounded outcomes | 会計の仮定。今回sampling/測定で検証したものではない |
| 実機での95%成功 | **NOT_CERTIFIED** |
| 数値/実装誤差の厳密保証、全support/長回路correctness | **未認証**。前段のengineering allowance・代表native証拠の範囲を維持 |
| 強Full S2/Yoshida4の適格性・価格・費用優位 | **未評価** |
| 価格・`ΣN×E[C]`・総RPE回路費用 | **未算出** |

会計で得たのは上記仮定の下での十分shot数で、真の最小shot数を証明したものではない。古典価格MCの標本数・CIはここでの量子測定失敗確率とは別であり、今回は一切計算していない。

## 次の担当・必要な承認と終了

**ユーザー**：次に同条件の強PF信号適格性確認、またはB2のfull no-prep価格評価のどちらへ進むか判断し、必要な固定SOURCE・有限対象・round・compiler・費用範囲・価格不確かさの評価範囲を別途明示承認する。**Codex**：その承認まで停止し、今回のshot数を既存M1価格や単回費用へ無条件に掛けない。

今回、新H/state/DF、PF/RTE simulation、branch再解析、private入力読込、新sampling、回路build/compile、CX/RZ/depth価格、価格MC、baseline・総費用・最適化、詳細科学レビューは0件。Python3.11.0rc1/NumPy1.26.4、GPU不使用。開始時の空きRAM約281 GiB・load28.58/30.96/23.21、観測affinity64、BLAS thread変数未設定。任意の資源cap・共有設定変更・他プロセス操作はしていない。

SOURCE・公開JSON/CSV・REPORTを専用branchへcommit公開し、remote SHA/tree/全変更内容一致を確認して**mandatory STOP**。旧SOURCE/raw/result/grant/marker/STOP、M1終了提案・M2保留、private state/HDF5を保全し、次の科学計算へ自動進行しない。

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/05015ae5ceeacc747f88d848cce8c3558a4881da
[BRANCH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d424bb8bec1aaf1224a33513a06e64801749a69f/docs/research/track_a_b2_p1_branch_20261107/REPORT.md
[P1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7bc952772e451f27c1654dab6803192f2b204c9f/docs/research/track_a_b2_signal_conditions_20261106/REPORT.md
