# M1-S：保存γ境界とπ/3の限定照合契約候補

2026-10-17。状態は`CANDIDATE_NOT_ADOPTED_NOT_EXECUTABLE`。ユーザーの「こんな感じで進める」と提示された方針に基づく**技術準備だけ**であり、提示文に明記された「今回の質問自体は採択・実行承認とは扱っていない」を保持する。限定decision gate、科学条件、実験契約は未採択、数値照合は未承認。過去のV1修正・完了承認は完了済みV1までのもので、新gate実行へ転用しない。

## 目的と対象

保存V1のG3/G6境界γ*が、検討中の保守的nearest-consistent-branch RPE decoderの一round角誤差上限候補π/3 radより厳密に小さいかを、保存区間だけで判別する。decoderの正式採択・実装ではない。全RPEに普遍的な必要条件として扱わない。

対象はlinear H4 1.00 Å、STO-3G、8 modes、保存DF rank12、G3/G6/G9、δ=1/8,1/16 Ha^-1、R=2,4、T=1,4 Ha^-1の既存cell。元の108方式行・36zero controlは再解析しない。48対照のうちG3/G6の32対照を全件照合対象とし、G9の16対照は未解像性を保持して報告する。結果に応じたcell選別・grid変更・γ最適化・ランキングを行わない。

## 固定入力と来歴

- 基点・保存結果固定commit：`b34572770e7c2105e16da003af050d789f490662`。
- 唯一の科学数値入力：`artifacts/track_b_m1s_saved_signal_gamma_v2_result/2026-10-17/completion_attempt_02/result.json`。
- 入力bytes：146764、SHA256：`bb2d09e6963fdc8b8871170ba28e98c36f4e353f452e50870103a7e97bf2e55d`。
- 元V1 SOURCE S2：`677dd6957e0e1913b70ca5f999458835ae8a2e23`、C2：`2cb5160d9051a6568b8acc0cfb29fc169bdfca09`、A2：`161ee8c3dc82c7fa19e0e1cfb79d71d3c236a03c`。これらは新実行承認として使わない。
- [最新GPTレビュー原本](../../research/m1s_gamma_v1_completed_results_scientific_review_20261017.md)を同一bytesで保存。SHA256：`18492941acc4fc7fe7fd46b6f002c21f324e52a661ec13bc0a024c5e86ad7303`。
- 共通運用指示2026-10-16版SHA256：`8323be3f65856adf0f4d095d8e68546db3326822023c20e101ba5f596605ebfa`。

準備段階で科学入力はbytes/hash照合のみ。JSONの科学数値decode、π/3への数値代入、境界区間の再集計を行わない。

## 判定規則候補と数学的条件

保存された閉区間`[L,U]`がγ*を、`[p_lower,p_upper]`がπを覆うことを条件として採用する。根の共通正margin領域・方向が元V1の`SHARED_LOWER`→`SHARED_HIGHER`、`ENCLOSED_INTERIOR`であることを将来実行時に確認する。逆向きや不正な入力を、このgateの不成立へ読み替えない。

| 分類 | 計算規則 | 保証する内容（保存coverageを条件とする） |
|---|---|---|
| SATISFIED（成立） | `3*U < p_lower` | γ*≤U<π/3。shared有利側とγ<π/3の開領域が交差する。 |
| NOT_SATISFIED（不成立） | `3*L >= p_upper` | γ*≥L≥π/3。今回の直接tradeoffに対し、その開領域への交差を認めない。 |
| UNRESOLVED（未解像） | 上記のどちらでもない | 保存区間だけでは厳密不等式を区分できない。 |
| G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED | 数値閾値照合をしない | `ABSOLUTE_CONTRAST_UNRESOLVED_NO_RELATIVE_RESCUE`を維持。 |
| TECHNICAL_STOP | 入力identity・schema・orientation・区間整合・資源・保存処理の失敗 | 科学的な不成立の証拠ではない。到達情報を保存してSTOP。 |

π/3の区間は`[p_lower/3,p_upper/3]`の正確な有理数スケーリングだけで表示する。SOURCEの実判定は交差積の厳密な整数・有理数比較で行う。追加の丸め許容値、float近似、libm、π/atan再計算、区間細分・精度増加を使わない。上側等号は厳密成立を保証しないため、それだけでは成立にしない。下側で`3*L >= p_upper`が確認できれば厳密不等式の否定を保証する。区間が接触・重複する場合、別の否定条件を満たさなければ未解像のまま残す。

この推論は**入力区間が対象量を覆うという既存の条件付きモデル**に依存する。有理数比較自体が正確でも、元signal、保存状態、π/境界SOURCE、IEEE/BLAS/trig等を数値certificateへ格上げしない。証拠区分は`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`。初回の新SOURCEの数値correctness・実入力runtimeは未検証。

## 結果の意味と対象外

成立は、保存された保守的な連続十分shot設計の有利領域が、指定した角誤差上限候補より下にも存在するという限定診断である。実際のstate contamination、synthesis、SPAM、roundごとの追加bias等を差し引いた実用γ予算ではない。raw-H平均信号位相は単一energyの位相とは限らず、保存T=1,4だけでは全dyadic roundを覆わない。

strictな整数shot削減・native費用・状態準備費用・多round RPE成功・総資源改善・研究GOは一切認定しない。全32対照の成否や「一つでも成立」の集計から研究の継続／停止を自動決定しない。G9を相対指標によって救済しない。

不成立でも、この保守的decoder routeの限定条件についての診断であり、M1-S全体の一般NO-GOではない。研究の継続・縮小・理論成果としての整理は結果取得後のユーザー／GPT判断へ返す。今回は新しいGPT詳細レビューを開始していない。

## Resource・実行guard候補

CPU1 worker、GPUなし、memory2GiB、input1MiB、result/log各16MiB、最大48対照、fraction field1024、整数／有理数65536 bits、文字列表現25000字まで。これらは数値的許容誤差ではなく資源上限候補であり、正式採択済みではない。ソースと固定入力bytesから有限の処理inventoryを拘束する。起動時はinteger-string APIが存在する場合だけ設定し、旧API不在障害を持ち込まない。

**今回新規に用意する契約は、ユーザーの現行performance policyに従いCPU／wall-time capなしを候補とする。** 過去V1のCPU160秒／wall320秒契約は変更しない。明示中断・memory／output上限・入力固定・最大件数・bit／文字上限・failure checkpointは保持する。実入力でのruntimeや数値correctnessを保証したとはしない。任意の有限時間上限へ変更するなら、採択前に契約とSOURCEの整合を再固定する。

`decision_gate_adopted=false`、`adopted=false`、`coverage_model_adopted=false`、`contract_grants_execution=false`を維持する。将来、ユーザーによるgate／契約／条件付きcoverageの採択後にも、専用の明示execution authorizationが必要である。guardは採択フラグ、固定SOURCE・契約・入力・interpreter、authority-only子commit、clean tree、未使用専用outputを照合する。旧V1のauthorization、grant、markerを流用しない。

各将来試行はexclusive markerを一度だけ作成し、技術失敗でも消費する。部分結果と失敗情報をatomic checkpointで保全し、自動retry・閾値緩和・別方式への切替を行わずmandatory STOP。今回、新しいauthorization／grant／marker／実結果は作らない。

## 成果物・検査と未決事項

実装：`src/trottertracks/algorithm_codesign/m1s_gamma_pi3_v1/`、runner：`scripts/tracks/algorithm_codesign/run_m1s_gamma_pi3_v1.py`、静的checker：`scripts/tracks/algorithm_codesign/check_m1s_gamma_pi3_preparation_static.py`。machine候補・schema・検査・保全・evidence index・publication receiptは`artifacts/track_b_m1s_gamma_pi3_preparation/2026-10-17/v1/`。

今回の検査はAST構文、metadata/schema、guard閉鎖、入力hash、SOURCE／契約identity、旧成果保全に限る。実データ・toy数値correctness testは行わず、比較関数を呼ばない。未評価のSOURCEを実行検証済みと扱わない。

残る承認は、(1) π/3という限定decision gateの採択、(2) このmachine契約・条件付きcoverage・resource候補の採択、(3) 固定SOURCE／正式契約を指定する別途one-shot数値照合実施承認。通常は(1)(2)を同時に採択してよいが、(3)を推定しない。採択記録固定後は改めてSTOPする。技術的な未解決障害は準備段階の検査結果に従い記録し、実入力数値correctness／runtimeが未検証という制約を残す。
