# 既存証拠と、まだ判断を変え得る点

本書の数値は固定報告の引用で、新しい集計・モデルfit・ランキングではない。既存のraw event、NPZ、QPY、価格配列を読み直して数値処理していない。独立CI再現とも呼ばない。

## Domainを固定した証拠表

Track Dの実数値を引用する行はすべて、Linear H4／隣接1.00 Å／STO-3G／Nalpha=Nbeta=2／元H／number-CDF8／元frame順L_D2／ordered partial S2/s1/r32/K0／h=delta=.1、元H厳密ground仮定・状態準備費除外。P0はQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approx1／initially_zero=false／topology-free／宣言位相保持。q/cutと証拠scopeは行ごとに示す。RZはT、depthは実機時間ではない。

| 証拠段階 | 固定資料・既存到達点 | 未保証・今回の判断との関係 |
|---|---|---|
| 理想測定 | [B6b / 7ffe8b28](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7ffe8b281b76cef3a6fd06f43413904d6766d0ec/docs/research/track_d_b6b_bc_integration_20261108/MEASUREMENT_EQUIVALENCE.md)：任意入力のcoherence、whole-rayとhalf-rayの条件を区別。人工/回帰24件と人工P0 16 wrapper postcheck | literal全support・全roundのcompiled誤差、actual joint-lawは別。方式選択で必要なpath条件の根拠 |
| q1実H4有限標本 | [Stage A固定報告](../track_d_h4_paired32_reference_20261106/RESULT.md)：32保存path、raw32・wrapper post128。CX/depthは全32本で減、RZは全32本で増。event16後cut | 一資源の勝敗を総合優位にしない。単純なCX selectorが改善余地を持つ証拠ではない |
| q2〜32の実H4価格 | [C7-4 / 64072ba8](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64072ba81bb70db1df1b34e94adb08e78926f24b/docs/research/track_d_c7_4_model_stability_20261107/MODEL_REPORT.md)：q2/4/8/16/32、n16/16/32/16/8、計88 whole-path/352 price records。q/2 complete-step cut、strongの融合を保持 | 全law母平均、H4以外、長q、モデル設計後の独立holdoutではない。X/Yを独立nへ倍増しない |
| 局所構造・非加法性 | [C7-2固定版](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b1e7cc5b6270e6e62ab57b7c889f37484ce7d7b7/docs/research/track_d_c7_2_cx_structure_20261107/REPORT.md)：人工54 P0、raw98/post54/observable等36と回帰12。shell、CX界面、A/A†に加法性反例 | 人工2 mode、geometry/basis/CDF rank非該当。孤立CSWAP価格やlane価格の和をfull価格にしない。全P0 bridge未取得 |
| qモデル | C7-4のCX差は全88本で208−224q。RZ/sizeは全q affineで小さい探索残差、depthはbasis/lane-runが候補 | 有限観測であり全path公式でない。新係数や閾値はここで算定しない。compiled DAG endpointはcompile後oracle |
| 条件付き13-round会計 | [8828b298](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8828b298975c88a2c0d1fd57ff6ed905233840f5/docs/research/track_d_bc_total_20261108/REPORT.md)：m0..12/q1..4096、保存十分N合計14,738、固定A/F/S。主AのCX11.1671%、RZ1.1200%、累積total-depth54.6549%減、幅9→17との既存点推定 | 約99.4〜99.5%がq≥64 proxy。全体CI/SD/covarianceは不足につきnull。actual成功保証・最終的な総資源優位ではない。新selectorの効果をこの割合へ加算しない |
| q64数値修正 | [a708a130](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a708a13077b0616a6047bbca2ba36f21ae58755b/docs/research/track_d_c8_1_q64_20261108/NUMERICAL_REPAIR_RESULT.md)：cut32、同sample00の4保存P0に対しsimulation fusion無効で4 postcheck PASS、旧FAIL/STOP保持 | 旧閾値component1e−10、observable2e−10は不変。任意qの丸め保証や速度改善ではない。最終paired/chiと残7の完了をこの報告から認定しない |
| q64別担当の限定再開 | [b1e7cc5b PLAN](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b1e7cc5b6270e6e62ab57b7c889f37484ce7d7b7/docs/research/track_d_c8_1_q64_20261108/COMPLETION_PLAN.md)：保存8 path、先頭の最終paired/chi後に残7へ進む有限scope | 指示時点で別担当が進行中。本調査は固定PLANだけ参照し、現在の進捗・完了は照会/認定しない。SOURCE/output/process/STOP無操作 |

13-round会計の固定条件はenergy .00159 Ha、failure .01、beta_total=pi/6、beta_stat=pi/12、alpha_axis=.01/26、prior[-10,10]を含む。これを新しい最適化条件として今回採択し直したわけではない。

## O2から移管する到達点

| 固定commit | 到達点 | Track Dへ自動移植できない点 |
|---|---|---|
| [0edc0fa5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0edc0fa5ad4656fdd492ca6c96ade4f77cadf412/docs/research/o2_vacuum_reference_cswap_20261108/REPORT.md) | 人工1〜3 system qubits、14 correctness/関連PASS、128 compiled比較。非可換splitの単回分布保存、coherent QPE反例。長いchanging-basisで有利、短/強融合で不利 | 実H4ではない。diagonal_only対照の人工資源効果を、directional strong対照へ移さない |
| [781e155d](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/781e155d0907a82c962e41ab8e89f985735bc49b/docs/research/o2_vacuum_reference_h4_preparation_20261108/REPORT.md) | 人工2-mode DF配列、2 fragment/L_D1/delta=.02、geometry/basisなし。追加18 tests、opt2の人工24行。state-free復元と全入力POVMの丸め込み方式 | 想定保存H4はlegacy DF12/L_D3/m0/q1/.02/r1/K2。未採択契約・実H4未実行。Track Dのnumber-CDF8/L_D2/r32/K0/P0とは別 |

O2の未採択2-path実H4契約を今回採択・実行しない。既知SE gadgetの再実装も不要。phase ledgerと誤差評価の考え方を既存Track Dの条件へ接続する技術課題としてのみ残す。

## 既存証拠から言えること・まだ言えないこと

候補Aのround-only切替には、RZの短q不利/長q有利という既存報告の動機がある。ただしCXは保存範囲の全pathでReference側が有利で、pathに応じて勝敗が頻繁に入れ替わる根拠はない。depthも保存平均や既存報告はReference側を支持する。よって単にpath依存selectorを追加すれば大きく得になるとは推定しない。

C7-4のfeature効果は「価格を説明しやすい」ことの探索証拠であり、「方式を安く選び分けられる」ことの証拠ではない。絶対価格がrunに依存してもpaired差の符号が同じならselectorに価値はない。保存88 pathすべてを新モデル設計に使用済みで、再利用を独立検証としない。

まだ判断を変え得るのは、事前取得可能な特徴で資源目的に対する勝敗領域が分かれるか、単純なround-only/常時Referenceより利益が残るか、選択/学習/compile償却費と17-qubit制約を含めても得になるか、cut移動やhead-tailの合法な接口融合に既存strongを超える余地があるかである。この調査ではいずれも数値を算出していない。
