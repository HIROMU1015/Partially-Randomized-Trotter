# E/T v5：別one-shot実行authorization

利用者の今回の明示指示（原文）：

> 次の作業に進んで大丈夫

直前にCodexは、技術準備・164 focused PASS・GitHub取得検証を完了し、
新sourceへの別明示one-shot実行承認が次の境界であると報告した。
この応答を、その提示済み固定契約の一回実行への承認として記録する。

- source S5：`d0a12bae817ce75c5876a06c2db00397914ca9b0`。公開確認用P5を実行sourceにしない。
- contract v5 SHA256：`586b2ca43a49e6e8adb539ac364b0b6d76cbf02fe827981b9d9ee95334ac00fa`。
- source manifest SHA256：`a5267a3f8121f50a01447299265fa5a64a9b00d6a47867fed1e9ebbe46a7903e`。
- 実行branch：`track-b-df-correction-et-v5-one-shot-execution-20261012`。
- このcommitはS5のauthorization-only直接子。変更pathはauthorization JSONと本receiptだけ。
- E-r3/r6保存入力、uncontrolled/MSB-controlled、native最大22、T最大64keys。
- CD6taskは保存済みsubstage参照のみ。全科学条件・誤差・runtime・資源上限を維持。
- fresh resultは`artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v2/`。
- runs=1、retries=0、成功・失敗を問わずmandatory STOP。
- 旧v4および以前の結果・source・contract・authorization・consumed markerを保護。
- G10/RA-D0再開、分子/NPZ、GPU、追加fixture/precision/合成keysは認可しない。

A5をpushし、remote SHA・direct parent・clean・source/contract/manifest・新namespace未存在を
照合してから固定runnerを一回だけ呼ぶ。終了後は保存値と外側processのstatus/provenanceを監査し、
必要結果・GPT引き継ぎ資料をcommit/pushする。研究方針・科学的優劣の判断はGPTへ戻す。

承認記録UTC：`2026-10-12T11:56:51.504832+00:00`。
