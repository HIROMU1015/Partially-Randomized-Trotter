# O1 M1：r=8限定full no-prep CX価格pilot

2026-11-06、SERVER 2（GPUA100-gleap02）。**固定16 trajectory・32本のX/Y価格測定を正常終了した。両候補の標本平均は必要単価を大きく上回った。追加の大規模価格MCへ進まず、この固定r8の2候補は研究上終了することを推奨する。** これは小標本に基づく研究判断の提案で、確率法全体の厳密下界・数学的不可能性・全RPE総費用の判定ではない。**mandatory STOP。**

## 固定条件・再利用

- linear H4、1 Å、STO-3G、保存DF rank10、H4-A・共通 `T_eff/V_square`、faithful tail identity、全signed factor・核scalar・ordinary-controlled相対位相を保持。元Hの保存 `state` を厳密基底状態と仮定し、状態準備費用を除外する。
- q=64、r=8、K=2、delta=.05。Pauli PRとDF PRを各8 trajectoryだけ測定。PCG64 seedはPauli **2026110612**、DF **2026110613**。各trajectoryはfreshなq*r=512 event、同じevent列のX/Yを別々にbuild・compileした。重複drawの除去、追加sampling、自動retryはない。
- Qiskit **1.3.0**、basis `rz/sx/x/cx`、optimization **1**、transpiler seed **7**。backend／target／coupling mapの指定なし、all-to-all。最終whole circuitのCX countを測定し、旧価格やproxyは使っていない。
- 科学SOURCE：[`1b64e46c7305e34c2a16aba9ac2e01256153838d`][SOURCE]。下界調査commit `2263000e203f8b9b075f7296a9e33f34e8ea24db` から、薄い[runner][SOURCE]と[人工テスト][TEST]だけを追加した。既存 `prepare_matched_resources`、`configure_law`、`ParallelMatchedNoPrepCostProvider`、Hadamard wrapper、compile pipelineは変更していない。runnerのsubclassは既存providerの入力event列と順序付きraw価格を保存するだけで、抽出やcompile処理を再実装していない。
- 入力：commit `4591b8d75a2a6a55c74a51225f0c61648023c8ee` の `artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`。SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。固定Git blobから専用一時copyを作り、既存snapshot loaderを使用した。
- 元state SHA256 `d4432409eb6b1d4f9f9f86a3b57664d9e3c84a4fe3fe9de400d64b7ba60270af` を照合。stateはcohortのbindingに使い、新しいstate solve・signal評価はしていない。
- [保存shot・必要単価結果][SHOTS]とのcontext／preparation hash、canonical lawのlambda_R・normalization、compiler hash、full no-prep scopeが一致した。保存信号configのtail hashラベルを価格lawとして代用していない。[信号検証][SIGNAL]の既存correctness証拠を再利用し、新しいoperator／state-action検証は実施していない。従来の証拠範囲は選択primitive・controlled 1 stepと独立q64 meanであり、全q64 trajectoryのoperator認証へ昇格しない。

## 測定結果

X=cosine、Y=sine。以下はcompile後の実CX countで、trajectory番号は各seedのstream順（0起点）。

| trajectory | Pauli X | Pauli Y | DF X | DF Y |
|---:|---:|---:|---:|---:|
| 0 | 13847 | 13847 | 23351 | 23351 |
| 1 | 13586 | 13586 | 25678 | 25678 |
| 2 | 14511 | 14511 | 25034 | 25034 |
| 3 | 13541 | 13541 | 23473 | 23473 |
| 4 | 14557 | 14557 | 24778 | 24778 |
| 5 | 13973 | 13973 | 23266 | 23266 |
| 6 | 14708 | 14708 | 25393 | 25393 |
| 7 | 15520 | 15520 | 23533 | 23533 |

| 候補 | X標本平均 | X sample SE | Y標本平均 | Y sample SE | shot重み付き標本平均 | 必要な期待CX単価（未満） | 診断 |
|---|---:|---:|---:|---:|---:|---:|---|
| Pauli PR | 14280.375 | 237.56081458228758 | 14280.375 | 237.56081458228758 | **14280.375** | 4.335396744008651 | 標本推定値は条件を満たさない |
| DF PR | 24313.25 | 356.10350371365746 | 24313.25 | 356.10350371365746 | **24313.25** | 6.292396396272071 | 標本推定値は条件を満たさない |

sample SEは不偏標本分散を用いた `s/sqrt(8)`。X/Yが同価格だったのはこの16 trajectoryでの観測であり、全supportの同一性を主張しない。

保存shot数はPauli `N_X=N_Y=1,133,345,886`、DF `N_X=N_Y=780,863,721` をそのまま使用した。新shot計算はない。各paired標本に `(N_X C_X+N_Y C_Y)/(N_X+N_Y)` を適用し、その平均を求めた。paired価格のsample SEはPauli **237.56081458228752**、DF **356.10350371365746**。X/Yを独立標本と見なしてSEを過小評価していない。末尾のSE桁差は既存online accumulatorと直接計算の丸め差である。raw平均・SEとproviderの集計一致も確認した。

比較には保存された厳密な必要単価Fraction（Pauli `818917344/188890981`、DF `1637834688/260287907`）を使用し、表示丸め値で境界判定していない。C0の強DF-S2・4 substeps・既存relative fusion／spin policyの保存全7round費用 **B=9,827,008,128 CX** は[既存結果][SHOTS]を引用し、再集計・再compileしていない。

## 判断と限界

観測した全trajectoryの価格も標本平均も必要単価を大きく上回った。減衰軽減が確認されたr8でも、今回のfull no-prep価格との交換は診断的に不利である。固定2候補の総費用改善を追って大規模価格MC・全round計算へ進む材料は得られなかった。

ただし、8標本の平均は確率法全体の厳密な期待価格ではなく、**sample SEは信頼区間ではない**。未観測trajectoryの確率質量・価格範囲に基づく保証は作っていない。以前の下界調査の **UNDETERMINED** は覆さず、数学的不可能性を認定しない。今回は全7round総費用・成功確率の新規検証もしていない。C0の負結果と `not_certified`、価格CIなし、成功保証の未認証という留保を維持する。新しい科学レビュー・研究方針の正式採択は行っていない。

**次の担当・STOP条件：** ユーザーが、この固定r8の2候補を研究上終了する提案を採用するか判断する。終了するなら追加価格評価は不要で、Codexの次の計算はない。なお続行を選ぶ場合でも、対象・追加標本数・seed・compiler・停止規則・不確かさの目的を別途固定し、新しいsampling／build／compile／価格集計について明示承認が必要。今回は追加計算を提案条件の確定前に開始しない。

## 保存物・テスト・終了

- [result.json][RAW]：固定SOURCEのhash、入力・compiler・runtime・seed・law、保存shot／閾値、raw価格・集計・判定・実行記録。
- [Pauli raw価格][PPRICE]／[DF raw価格][DFPRICE]：各trajectory・各軸のCXその他既存metric。
- [Pauli event列][PDRAW]／[DF event列][DFDRAW]：指定streamから抽出したorder・rotation・product成分。既存builderはcanonical eventのphase・probability等も照合する。再samplingして照合したものではない。
- [compile進捗ログ][PROGRESS]：完了した32本のfingerprint・CX・PID等。[実行ログ][LOG]：単回科学実行、**exit0、elapsed約263.65秒**。
- 人工検査：新paired価格集計テスト2件と既存並列providerテスト14件、**計16 PASS（54.40秒）**。人工testのみで保存H4入力は未使用。実価格測定をtestで再実行していない。

利用可能CPUは64。両armを並行開始し、各armは指定32本に含まれる最初の1軸をメモリprobeとした後、残り15軸を15 workerで実行した。初回peak RSSはPauli 1,343,612 KiB、DF 4,443,756 KiB。worker数は既存providerがCPUと実測メモリから選び、恒常的な小規模capは追加していない。worker内の数値threadは各1とし、外側の並列処理との過剰な多重化を避けた。共有環境・他ユーザー・共有packageは変更せず、cacheは専用一時directoryを使用した。

実行後も固定SOURCE hashの不変、各arm8列・各列512 event、32本の完了価格を確認した。追加標本・retry・r4価格・新signal／shot計算・baseline再compile・全RPE総費用・新fit／rank探索・M2作業はない。旧SOURCE・raw・result・grant・marker・STOPは保全した。専用branch `codex/track-a-o1-m1-r8-price-pilot-20261106` へ結果を公開し、remote SHA・tree・REPORT本文を照合して終了する。**mandatory STOP。**

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1b64e46c7305e34c2a16aba9ac2e01256153838d/scripts/run_o1_m1_r8_price_pilot.py
[TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1b64e46c7305e34c2a16aba9ac2e01256153838d/tests/test_o1_m1_r8_price_pilot.py
[SHOTS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7e9587fb7398ebdab156d83abc869d82e5f42c9/docs/research/track_a_o1_m1_cost_feasibility_science_20261106/REPORT.md
[SIGNAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185104f64a4594355a3813d04d9e50bd4abdb6b5/docs/research/track_a_o1_m1_attenuation_science_20261106/REPORT.md
[RAW]: ../../../artifacts/track_a_o1_m1_r8_price_pilot_20261106/run01/result.json
[PPRICE]: ../../../artifacts/track_a_o1_m1_r8_price_pilot_20261106/run01/P_raw_prices.json
[DFPRICE]: ../../../artifacts/track_a_o1_m1_r8_price_pilot_20261106/run01/DF_raw_prices.json
[PDRAW]: ../../../artifacts/track_a_o1_m1_r8_price_pilot_20261106/run01/P_draws.json
[DFDRAW]: ../../../artifacts/track_a_o1_m1_r8_price_pilot_20261106/run01/DF_draws.json
[PROGRESS]: ../../../artifacts/track_a_o1_m1_r8_price_pilot_20261106/run01/compile_progress.jsonl
[LOG]: ../../../artifacts/track_a_o1_m1_r8_price_pilot_20261106/execution.log
