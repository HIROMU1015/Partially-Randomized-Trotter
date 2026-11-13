# 2026-11-13：S3最終科学実行計画案

B ab12/bf515、A bba150834/169d3316の固定証拠を使い、既存SCIENTIFIC_PLAN/S3_DECISIONSを最終計画へ具体化した。過去の計画を改稿せず、新資料で全13nominal coverageとR3選択を反映。実装・科学SOURCE・入力は無変更。

推奨はproxy条件付きの固定13round/nominal1配分、主loss UNDEFINEDによる条件付きRMSE・未解決率。全試行の特定RMSEを定義するには追加loss/出力契約が必要で、任意の[0,U]感度envelopeと区別した。U4096/η=.01U/各CI97.5%・joint95%と零観測CP解像度10^−4なら、汎用設計算術からn43,819を提案。粗いpilot精度であり化学精度の確認には使わない。未解決率0観測も真の確率0ではない。

全科学条件は未採択。Aの経験的allowance・未知law/target誤差を0にせず、mean再解析やL2変換はしない。新実S3/N/C/G0、十分N/global bound null。コード変更・新科学テストなし、静的SOURCE/metadata/文書・設計式のみ確認。[最終計画](../rpe_s3_final_plan_20261113/SCIENTIFIC_PLAN.md)／[HANDOFF](../rpe_s3_final_plan_20261113/HANDOFF.md)。次はユーザーの条件採択・有限scope明示許可後にBが実行。計画公開照合後mandatory STOP。
