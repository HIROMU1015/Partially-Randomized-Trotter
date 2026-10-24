# H4 D2：固定Pauli bodyのcontrol方式比較

2026-10-24 JST。ユーザーの「次の作業に進んで大丈夫」を、直前のD2固定2回路・最大2compileへの実行承認として保存した。準備commit `57b65ce2d2c707409b2c06367d4bb1107b8ab66b` と無効契約をGitHubから取得し、契約blob `2044288d90bd9e0178f94fd3a09f8028c7d45fdc` のlocal一致を確認した。

現在はproduction接続・人工検査完了、科学実行前。SOURCE固定・新grant発行後に一度だけ実行し、結果・終了監査を追記する。

## 固定条件とscope

Linear H4、隣接1.00 Å、STO-3G、system8＋ancilla8、S2/T=.8/q8/r4/K4/cosine。Pauli184非identity項、PR prefix27。D1と同じPF singleton／PR trajectory_index0の保存bodyを使用し、角度・law・event・分割・wire・命令順を保持する。比較背景のDFはtol-only1e-8/rank10・PR prefix1＋one-bodyで、今回は再構築しない。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし。状態準備を除くfull Hadamard wrapper。

既存generic対照は[D1](non_df_h4_d1_source_execution_20261024.md)のRZ255500／9652を引用し、測り直さない。D2は同じbodyをcentral-rotation controlへ変換する実装＋compiler応答の比較で、表現だけの効果・母平均・N/G・精度certificateではない。

## 実装・承認・監査

[controller](../src/trotterlib/h4_d2_control.py)、[runner](../scripts/run_h4_d2.py)、[人工tests](../tests/test_h4_d2_control.py)、[実行証拠](../artifacts/non_df_h4_d2_execution/2026-10-24/README.md)。D1のwrapper/compiler、SHAとSOURCE-B token照合、durable save、locked nonrefundable CallBudget、resource observer、native subreaper witnessを流用した。旧source・grant・STOPへ変更しない。

Stage Aで両bodyのSHA・明示operand・既存SOURCE規則のtoken・保存集約phaseを照合する。Stage Bで両controlled inputとmappingを保存し、Stage Cで各1回compileする。共通gate後の各2taskは並列。新inputはgenericと異なるprogramなので、旧hashとの一致を求めず、新input自身の完全性とSOURCE semantic対応を別に検査する。Pauli-PR historical hash FAIL/原因UNKNOWNは保持する。

集約phaseをancillaへ一度だけ転送し、event phaseを重複加算しない。[候補の位相対応・適用外identity経路](non_df_h4_d2_preparation_20261024.md)は準備報告を参照。新H4 Operator/state/signalやsamplingは呼ばない。新規SCF/FCI/DF、N/G、正式certificate、D3、retry/resumeはscope外である。

新authority/pipeline人工10件と候補12件がPASS。主agent自身の検査で、独立レビューではない。新SOURCE・使用input/runtimeをsealし、今回の明示承認を新grantへ結合する。使用済みD1 grantは流用しない。native wait/reapは自分のengine/taskだけを対象にし、旧監視終了の欠落を埋めない。CPU/RAMの恣意的な上限を置かず、共有設定・他ユーザーには変更しない。

実行後は必ずmandatory STOPし、次の作業は保存結果を見てユーザーが判断する。
