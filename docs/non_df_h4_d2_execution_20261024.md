# H4 D2：固定Pauli bodyのcontrol方式比較・one-shot完了

2026-10-24 JST。**D1と同じ保存bodyでcontrol方式を変更した2/2回路の計測が完了した。Pauli-PFは255500→15028、Pauli-PR index0は9652→1145 RZ。** 同一bodyのcontrol実装＋固定compiler応答に大きい費用差があることを確認した。表現だけの因果効果・母平均・N/G・精度certificateではない。

SOURCE `74515d8eb4f073c40fc783681a3775dbca89e9b1`、新grant `6a035101fc0c3fb26f79799f4f8a8535b14367fc46f933a6f0a4804b50118982` を一度だけ使用した。validation2/build2/compile2、generic対照再生成・再compile0。engine＋6taskは全exit0・native wait/reap済み、自分の残存process0。旧953file/code48/input91とrun76fileのSHAを照合した。**mandatory STOP。D3へは進まない。**

[全結果](../artifacts/non_df_h4_d2_execution/2026-10-24/result_summary.json)、[10指標比較CSV](../artifacts/non_df_h4_d2_execution/2026-10-24/comparison_table.csv)、[保存・終了検証](../artifacts/non_df_h4_d2_execution/2026-10-24/verification.json)。

## 固定条件・承認・来歴

Linear H4、隣接1.00 Å、STO-3G、system8＋ancilla8、S2/T=.8/q8/r4/K4/cosine。Pauli184非identity項・PR prefix27。D1と同じPF singleton／PR trajectory_index0の保存bodyを使用し、角度・law・event・分割・wire・命令順を保持した。比較背景のDFはtol-only1e-8/rank10・PR prefix1＋one-bodyで、今回は再構築していない。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし。状態準備を除くfull Hadamard wrapper。

ユーザーの「次の作業に進んで大丈夫」を、直前の固定2回路・最大2compileへの実行承認として[保存](../artifacts/non_df_h4_d2_execution/2026-10-24/approval.json)した。準備commit `57b65ce2d2c707409b2c06367d4bb1107b8ab66b` と無効契約をGitHubから実取得し、契約blob `2044288d90bd9e0178f94fd3a09f8028c7d45fdc` のlocal一致を確認した。

元H4科学SOURCEは `0c329b30a81accb35c212be8defa66fee0dd0f12`、保存body生成SOURCEは `48d4d27a05a529e05b11599f5f3c671302826617`、generic対照を再現したD1 SOURCEは `9843b803878db6c3aee7ab5e7c6bb4818572c944`。今回の新D2 SOURCEは上記74515d8eで、旧sourceへ変更していない。専用branchは `codex/non-df-h4-d2-execution-20261024`。

| 対象 | 固定trajectory ID |
|---|---|
| pauli_jw_PF index0 | `d761627d56a4454e644358fef8907b8a7db28c4feeb44130224c20e4e50aae4d` |
| pauli_jw_PR index0 | `93ef5e4e28b499472c8ba9329f6cb139f63c5de071279783b3cca3319793fc97` |

## SOURCE接続と実測

[controller](../src/trotterlib/h4_d2_control.py)、[runner](../scripts/run_h4_d2.py)、[人工tests](../tests/test_h4_d2_control.py)。D1のwrapper/compiler、SOURCE-B token照合、durable save、locked nonrefundable CallBudget、resource observer、native subreaper witnessを流用した。候補module自体は準備commitから変更せず、元SOURCE・grant・STOPへ変更していない。

Stage Aで両bodyのSHA・明示operand・SOURCE規則のtoken・保存集約phaseを照合した。PF2944／PR464命令が対応し、QPY自己整合性だけをsemantic根拠にしない。Stage Bで両controlled inputと全instruction mappingを保存し、Stage Cで各1回compileした。各stageの2taskは並列で、最初のstage gate失敗で後続を起動しない。今回はすべて成立した。D1のgeneric対照価格を引用し、対照再compileは0。

| 対象 | 指標 | D1 generic | D2 central | central−generic |
|---|---|---:|---:|---:|
| pauli_jw_PF | RZ | 255500 | 15028 | -240472 |
| pauli_jw_PF | CX | 164096 | 26768 | -137328 |
| pauli_jw_PF | depth | 358563 | 35694 | -322869 |
| pauli_jw_PF | rz_depth | 160288 | 9227 | -151061 |
| pauli_jw_PF | cx_depth | 150752 | 25200 | -125552 |
| pauli_jw_PR | RZ | 9652 | 1145 | -8507 |
| pauli_jw_PR | CX | 6872 | 1856 | -5016 |
| pauli_jw_PR | depth | 12973 | 2035 | -10938 |
| pauli_jw_PR | rz_depth | 5918 | 664 | -5254 |
| pauli_jw_PR | cx_depth | 6494 | 1357 | -5137 |

比はSOURCEが生成したCSV/JSONに既約分数で保存し、分母をD1 genericの同じ指標と明示した。符号のclippingや寄与率fitはない。これらは新D2実装の計測値で、旧結果の価格を書き換えていない。

新inputはgenericと異なるprogramなので、旧hashへの一致を求めず、新inputの保存SHA・recursive hashとcompiled input hashの一致を確認した。Pauli-PR historical hash FAIL／原因UNKNOWNは保持した。full H4 Operator/state/signalは計算せず、SOURCE局所恒等式＋保存operand対応＋人工検査の範囲でcontrolを転送した。[位相対応と対象外identity経路](non_df_h4_d2_preparation_20261024.md)の制限も維持する。集約phaseをancillaへ一度だけ移し、event phaseを重複加算しない。

固定same-bodyのPauli PF/PR双方でRZ/CX/depthが小さくなった。従来の費用差を解釈する際にcontrol実装とcompiler応答を分けて扱う必要がある、という限定された証拠である。DFの保持内容・event粒度・basisや非加法性をこの差だけで説明せず、256本平均・母平均・N/Gへ外挿しない。個々のcompiler passの因果寄与も特定していない。固定wrapperにはcallbackがなく、pass_traceは未取得のままとした。

## 人工検査・保存監査で確認した表現差

新authority/pipeline人工10件と候補12件、計22件PASS。主agent自身の検査で、独立レビューではない。grant未承認/STOP/消費済み/改変拒否、予算非返還、stage失敗でcompile0、両target gate、人工body接続とphase・order・wire、input改変時のcompile拒否を確認した。

保存検証の初回に、PRのactual/expected tokenの**raw JSON SHAが異なる**ことを検出した。新SOURCEが実際に要求したのはPythonの数値token一致で、そちらは成立済みだった。[静的監査](../artifacts/non_df_h4_d2_execution/2026-10-24/token_serialization_audit.json)により、保存mappingからactual SHAを、保存SOURCE operandからexpected SHAをそれぞれ再現し、32箇所の `float operator.real=1.0` と `int coefficient_sign=1` の表現差を確認した。数値tokenは一致し、PF側はraw SHAも一致する。

[初回検証不合格](../artifacts/non_df_h4_d2_execution/2026-10-24/verification_attempt01.json)を残し、監査側の条件を元SOURCEの数値一致へ合わせた。両raw SHA・元validation・sealed source・結果は変更していない。これは元Pauli-PR historical input hash FAILとは別のmetadata表現差で、その旧原因を解明したという主張ではない。追加QPY読込・build・compile・science retryは0。

## 終了監査・原本保全・公開

[新seal](../artifacts/non_df_h4_d2_execution/2026-10-24/seal.json)はcode48/input91と既存runtime6999fileのmanifest、今回の承認、旧保全mapへ結合した。SOURCEを先にcommitしてからgrantを発行・消費した。

| 科学call | 上限 | 消費 |
|---|---:|---:|
| 保存body SOURCE検証 | 2 | 2 |
| central controlled full build | 2 | 2 |
| 費用top-level compile | 2 | 2 |
| generic full build/対照compile | 0 | 0 |
| sampling・H4 state/signal・N/G | 0 | 0 |

[ledger](../artifacts/non_df_h4_d2_execution/2026-10-24/run01/call_ledger.json)、[CONSUMED](../artifacts/non_df_h4_d2_execution/2026-10-24/run01/CONSUMED.json)、[最終STOP](../artifacts/non_df_h4_d2_execution/2026-10-24/run01/STOP.json)。engine exit0・6taskも全exit0/native wait/reap。自分のPID/start_ticksだけを再観測し残存0、他プロセスを走査・停止していない。run出力76fileと旧953file/code48/input91のSHAは一致した。

CPU affinity64と実空きRAM/storageを観測し、独立2taskを並列実行した。CPU/RAMの恣意的な低い上限、共有設定・他ユーザー・他worktreeへの変更は0。旧2つのD1 grantは再利用・再発行していない。

新規SCF/FCI/DF・状態/signal・sampling/RNG replay・shot/G・正式certificate受理・D3は0。元H4 native UNKNOWN、Pauli-PR historical hash FAIL／原因UNKNOWN、旧20行N/G null、C/L COMBINATION_BLOCKED、旧compiler MISSING_EVIDENCEは保全した。新native PASSは**今回D2だけ**の終了証拠である。

[commit/push記録](../artifacts/non_df_h4_d2_execution/2026-10-24/publication_status.json)。SOURCE74515d8e以降の科学codeは変更せず、結果・終了監査・報告を追加commitする。

**mandatory STOP。次はユーザーがD1/D2結果を踏まえて、残るbasis/compiler要因のD3準備が必要か判断する。Codexは明示指示後だけ次の準備へ進む。追加科学計算やレビューへ自動移行しない。**
