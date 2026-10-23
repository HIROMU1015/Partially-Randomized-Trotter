# H4 PR/PF 4方式：保存費用による探索的総費用比較 run01

2026-10-24 JST。**承認されたH4 one-shotを正常終了。4方式とも適格、Full・PRともYoshida4が低費用で、H2で見られた次数変更による費用方向の逆転はこのH4固定4候補では観測しなかった。4点の点推定最小はPR Yoshida4。**

これは有限探索標本・保存reference精度を条件とするempirical比較であり、母平均順位、一般的なPR優位、正式な共通native certificateや最終科学採択ではない。原因帰属はnull。GPT科学レビューは開始していない。

## H4の保存出力

| 方法 | 費用trajectory数 | C平均/RZ・軸 | analytic N合計 | G合計/RZ | paired G標準誤差 |
|---|---:|---:|---:|---:|---:|
| Full S2 | 1 | 62571 | 220075222 | 13770326715762.0 | — |
| Full Yoshida4 | 1 | 93799 | 144812741 | 13583290293059.0 | — |
| PR S2 | 48 | 51251 | 220194970 | 11285212407470.0 | 80607553015.50136 |
| PR Yoshida4 | 33 | 49547.666666666664 | 154678685 | 7663967924818.334 | 58964969222.79669 |

表は元出力fieldの転記で、cosine/sineのC平均は各方式で等しい。軸別C/SE/N/Gは[CSV](four_method_saved_output_table.csv)、u/headroom/normalization・元統計は[comparison.json](comparison.json)。Fullのn=1は決定論的な1組の単価で、統計SEを0としていない。PRは元48/33 trajectoryの有限標本、paired SEは同じtrajectoryのcosine/sineの共分散を保持した既存実装による。母平均保証や全誤差源を含む信頼区間ではなく、新しい名目区間を導入していない。

元rankingは `full_Y4_cheaper=true`、`PR_Y4_cheaper=true`、`reversal_candidate=false`、`opposite_order_directions=false`。保存4点の順序は PR Yoshida4、PR S2、Full Yoshida4、Full S2。4方式最良と4次化の方向逆転を別々の問いとして記録した。結果に応じた候補変更・再選定はない。

PR Yoshida4の主Gは7663967924818.334、既存resource_rowのG_pointは7663967924818.333。これは準備時に明記したmultiply-then-divideとmean-then-multiplyの演算順に対応する保存値であり、どちらかへ書き換えていない。新しい比・削減率・区間・追加rankingは生成していない。

## 固定条件・H2との対応

Linear H4、隣接1.0 Å、STO-3G、tol-only DF1e-8/rank10、T=.8、ε_signal=.0005、α_axis=.025、P=0、system8/ancilla8、usual signed bias、symmetric-directional control、状態準備を除くfull Hadamard wrapperのcompiled RZ/軸。Fullはprefix10のS2/q8 `H4_FULL_10_8_2nd` とYoshida4/q4 `H4_FULL_10_4_4th`。PRはprefix3/K4/ABS_TIME_PER_OUTER32のS2/q8 `M5_MIN_72b305020499dd68` n48とYoshida4/q4 `M5_MIN_ab57ad5bd40dc583` n33。m5_best、DF12、独立確認cohortを混ぜていない。

Full単価62571/93799 RZ・保存PR81 trajectory/162軸と、既存H4のsignal/bias/reference/log_B/allowanceを再利用した。H4 u=1e-9と候補固有の保存log_B_marginを継承し、empirical_allowance自体は再計算しなかった。8軸のanalytic shot整数はすべて元の保存H4値と完全一致した。保存referenceのFull/PR末尾差、raw/signalのlog_B末尾差は承認済みの条件として保持し、新しいreference計算・bias補正や0誤差置換はない。[準備報告と固定参照](../../track_a_h4_four_method_exploratory_ready/2026-10-24/REPORT.md)。

H2と並べた[保存field表](H2_H4_saved_output_table.csv)および[結果対応](H2_H4_result_correspondence.json)は、H2 commit `9e5ea6b235d7beecc02651b1393b561948e85789` の完了出力と今回出力の引用のみ。H2ではFull S2低費用・PR Yoshida4低費用で逆転候補、H4では両方Yoshida4低費用で逆転なし。H2はrank3、Full/PR prefix3/1、q4/2、PR n33/34であり、H4のrank10、prefix10/3、q8/4、n48/33とは別の固定構成である。allowance・referenceの入力も異なる。分子間の標本pooling、H2再会計・再ランキング、一般的な系サイズ依存や因果説明は行っていない。

## 承認・SOURCE・実行と監査

ユーザーの「できるだけ並列化して計算を開始して」を、直前に提示した固定H4 one-shot契約の実行承認として受領した。[承認receipt](authorization_receipt.json)。新H4専用の未使用grantを1件だけ発行して開始時に消費し、元H2/H4 grant・ledger・marker・STOPは変更も再利用もしていない。新authority原本と内部絶対pathはprivate領域に保持する。

- 契約：`H4_FOUR_METHOD_ARCHIVED_COST_EXPLORATORY_V1`。
- SOURCE：`4cbbc1b6b49bd64e765ccc596ec5016a1a4d5503`。
- 契約canonical SHA256：`f2fe519082fd0994de01660ad14810d3f2d2ef7bbd64c470f78eb3183931a408`。
- input manifest file SHA256：`465d16b6f00c606673dac547bba732334933a1ec15583e616709a575988a1d6a`。
- 元比較出力SHA256：`83348730804d9c3a392ee44c2fffd0b856b0bec7048bd7673a4a320f99f12452`。

[process receipt](process_receipt.json)：native exit0、親wait完了・child回収、wall 1.177413 秒。**試行1、失敗0、中断0、再試行0、SOURCE変更0。** 保存値による小規模算術のみで、重い回路処理はない。

[exact ledger](execution_ledger_receipt.json)：bootstrap1/bundle_load1/allowance4（保存値取得）/accounting4/u_aware_shots4/axis_headroom8/cost_statistics4/resource_row4/metric_summary8/ranking1、合計39 event。empirical_allowance/setup/prepare/primitive/signal/reference_recompute/sampling/build/lowering/compile/transpile/simulateは全て0。結果を得るための会計・統計・順位kernelの再呼出しもない。

[元completion](native_completion.json)・[元STOP](native_STOP.json)・[出力転記監査](output_quote_audit.json)で、9個の元result file SHA、39 ledger、契約、SOURCE、消費markerとの対応を検査した。[最終監査](final_verification.json)はPASS。同run_root/grantの再実行はone-shot gateが拒否する。

## 並列化と保全

独立したSHA照合を利用可能CPU affinity数のthreadで並列化した。[資源観測](resources.json)。数値本体は約1.18秒の算術であり、封印済みSOURCEの逐次ledger・共有instrumentationの復元を保つ単一processで実行した。これは恣意的なCPU/RAM上限ではなく、固定SOURCE/契約の実行順序を守る選択である。新たな並列版科学SOURCEの作成や未承認benchmark、共有環境・他ユーザーprocessの変更はない。

[保全監査](preservation_audit.json)は今回使うSOURCE/input/runtime、旧原receipt/authority、H4準備成果物、既存H2完了結果・ledger等の既知1141 fileを前後SHA/bytesで照合し、変更0。無関係なworktreeの通常進行を停止・拒否条件にしていない。論文主表・Figure2・採択済み共通PF資料は変更しない。

数値SOURCE不変更のため、準備時の人工11 test methodと封印runtime人工smokeを再利用した。今回は実承認会計と出力/ledger/保存shot/保全の検査を追加した。作成者Codex自身の技術検査で、独立科学レビューではない。

## 次の担当とSTOP

ユーザー：このH2/H4の条件付き探索結果と限界を確認し、GPT科学レビューを開始するか判断する。Codex：今回の結果・固定資料・監査・commit/pushまでを扱い、追加科学計算・再選定・詳細レビュー・最終evidence freezeを自動開始しない。GPTレビュー用の固定資料入口は本報告、comparison、両系CSV、結果対応、準備の条件対応であり、レビュー自体は未開始。

旧strict COMPAREはBLOCKEDのまま保全し、探索結果を正式採択結果と混同しない。**mandatory STOP。**
