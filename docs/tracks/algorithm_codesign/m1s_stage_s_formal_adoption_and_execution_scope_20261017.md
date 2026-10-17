# M1-S Stage S 正式採択・今回の計算開始承認の固定

2026-10-17 JST。候補公開commit `0dd660ece1f2e342d2666d37d4b4536adb5e8ca7`、固定SOURCE S `7a338c4eba778c4ce1f189cfa8c8f050afb5df08`。

ユーザーの今回の発言は「その方針でよいので、計算開始まで進めて」。直前のStage S候補・次工程を了承して計算開始まで明示したため、固定候補の正式採択と、別artifactでS/Cへ結合するone-shot実施承認を受領したと解釈する。原文と解釈を分け、別々の二発言があったとは記録しない。採択記録だけで止める従来の中間手順は今回の開始指示で進めるが、最終mandatory STOP、failure時も一回消費、retry禁止は維持する。

## 採択する条件

linear H4 1.00 Å/STO-3G/8 modes/保存DF rank12、G3/G6/G9、δ1/8・1/16 Ha^-1、R2・4、T1・4 Ha^-1、保存state・元norm²。既存全48対照、元108方式行inventoryを保全する。G3の16、G6 T4の8だけを事後的主対象とし、G6 T1不成立8、G9絶対差未解像16はcarryとして報告する。

保存raw-H理想Task A、γ→π/3^-の開端点screen、連続比はsupremum、整数はfloor(S(cap))+1、両方式の整数が解像した場合だけwitness/no-witnessを判定する。no-witnessを全γ領域の不可能性としない。PER_METHOD_ROW、α_row=1/20、fresh-IID/Hoeffding/X/Y等配分、p128＋8 guard bits、refinement0を採択する。条件付きcoverageはSAVED_CONDITIONAL_RADII_IDEAL_MEAN_ONLY／CONDITIONAL_ENGINEERING_NOT_CERTIFIED。platform、oracle固有state、物理IID、RPE成功、native総資源を認定しない。

CPU/RAM/GPU/worker/CPU-time/wall-time人工capなし、exact-rational CPU backend、独立cellと利用可能affinityによる並列化、入力/結果/log各16MiB・既定call/instruction上限は候補と同一。共有設定・他ユーザーのプロセスは変更しない。

## 契約と実施承認を分ける

正式machine契約は`artifacts/track_b_m1s_stage_s_formal_adoption/2026-10-17/v1/execution_contract_adopted.json`。元候補から変えたfieldはstatus、adopted、coverage_model_adopted、selected_scienceだけで、SOURCE・入力・数値条件・resourceは不変。contract_grants_execution=false、契約単独でのscientific_execution_authorized/run_ready=falseを保持する。これは今回のユーザー実施承認がないという意味ではなく、契約自身から実行許可を発行しないという意味である。

この採択Cを先にcommitし、そのcommitとhashへ新しい専用execution authorization Aを結合する。Aを公開・照合してから固定runnerを一回起動し、結果または失敗prefixを保全する。新しいSOURCE修正、精度緩和、retry、Stage C/Task B拡張は承認されていない。

[採択文脈](../../../artifacts/track_b_m1s_stage_s_formal_adoption/2026-10-17/v1/user_adoption_and_execution_context.json)、[採択検査](../../../artifacts/track_b_m1s_stage_s_formal_adoption/2026-10-17/v1/adoption_validation.json)。既存候補・旧成果・消費済markerを保全する。実行終了後はmandatory STOPし、結果に基づく科学判断をユーザーへ返す。
