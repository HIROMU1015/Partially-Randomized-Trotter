# M1-S 次期shot・cost検証契約候補（未採択）

2026-10-17 JST。基点 `81d6103ec89d485cb7336c3a90bd95dad66992aa`。
`DRAFT_WITH_OPEN_SCIENTIFIC_CHOICES / DESIGN_ONLY / NOT_EXECUTABLE / MANDATORY_STOP`。

## 1. 契約の位置付け

[Decision Packet](m1s_shot_cost_feasibility_decision_packet_20261017.md)のP1–P4、S1–S3、C1–C3を記号的な検証候補としてまとめる。旧V1やπ/3の採択契約を更新せず、今回のmachine候補は別pathとする。`adopted=false`、`coverage_model_adopted=false`、`contract_grants_execution=false`、`scientific_execution_authorized=false`、`run_ready=false`。過去の条件付きcoverageの採用と次期契約の採用は区別する。

Task A先行／Task B最終目標／raw無補正／保存全cell比較は既存方針として継承済み。一方、この次期段階のγ・confidence・算術・費用条件は未採択である。旧α=0.05、p=128、CPU160/wall320秒を無断で新契約の値にしない。

## 2. 固定inventoryと証拠の継承

linear H4 1.00 Å、STO-3G、8 modes、保存DF rank12、G3/G6/G9、δ1/8・1/16 Ha^-1、R2・4、T1・4 Ha^-1。同じ概念label数によるshared_pm対ordinary/independent_pmの48対照、72非zero方式行、36zeroを含む108方式行を継承する。新しいgeometry、time、split、方式を追加しない。

`preserved_gate_registry.json`はユーザー指示と固定報告の区分を全48識別子へ対応させたmetadataであり、境界・信号・数値ランキングは入れない。G3の16成立、G6 T4の8成立、G6 T1の8不成立、G9の16絶対差未解像をそのまま保持する。passing群を独立検証済みsubsetと呼ばない。

## 3. 科学的な採否が必要なfield

|ID／field|選ぶ必要のある内容|未決に残す理由|
|---|---|---|
|D1 phase_requirement_policy|共通のγ値または外部精度に基づく有限profile/domainと評価規則、rad単位|π/3はcapであり要求精度ではない。既知γ*を見て有利な点を選ばない|
|D2 confidence_policy|α値、PER_METHOD_ROW／同時family等の範囲と軸・round配分|旧per-row95%を全体保証に変えない|
|D3 phase_and_state_budget|Task A理想診断と物理実装／Task Bの境界、b_raw→*・実装/state/SPAM項の必要証拠|state normや残差ではtarget overlap・多round biasを認証できない|
|D4 coverage_acceptance_policy|保存および新算術の条件付きcoverageをどのclaimの根拠として採るか|platform認証や統計confidenceと混同しない。保存の未検証条件は残る|
|D5 cost_metric_and_scope|加算metric、期待値／確率上界、固定費・状態準備の扱い|無料state/HDや異単位の加算を避ける|
|D6 native_implementation_policy|native profile、compiler/seed/cache、制御・identity・HD・合成誤差配分、品質検証|既存DF/RTE costはM1-Sの価格ではない|
|D7 decision_and_stage_policy|S/Cの採択範囲、全48報告、eligibilityの事後性、必要な改善幅、STOP／未解像の扱い|科学GOは自動判定しない。stage間を自動承認しない|

fresh-IID・Hoeffding・X/Y等配分、S1の十分設計、S2/S3の整数比較を技術上の第一候補とする。採択前はこれらもnext-contractの提案であり、最適shot数の主張をしない。全未決fieldはmachine契約でnullにする。値の推測や旧defaultの挿入をしない。

## 4. 比較・判定候補

同じtask、同じγ・confidence・fresh-IIDの測定法で正の位相余裕・visibility下界が得られることを前提とする。保存raw総biasへモデル差を再加算しない。数値coverage、γ/domain不適合、情報不足、科学的な優位不成立を混同しない。

- `DOMAIN_INFEASIBLE`：選択した無補正設計でq≤0、visibility下界≤0等。不可能性一般やM1-S NO-GOではない。
- `MISSING_REQUIRED_EVIDENCE`：state／実装／cost等、選択scopeに必要な証拠がない。理想Task Aに不要な証拠を、そのtaskの必須条件としない。
- `INTEGER_SAVING_RESOLVED`／`INTEGER_TIE_RESOLVED`／`NO_INTEGER_SAVING_RESOLVED`：同じ十分設計で、enclosureに基づき確定した分類。
- `ARITHMETIC_UNRESOLVED`：整数境界・位相条件・費用区間を解像できない。閾値緩和・結果依存の精度増加はしない。
- `G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED`：旧未解像性を保持。形式的shot比で救済しない。
- `COST_BREAK_EVEN_SATISFIED`／`COST_BREAK_EVEN_NOT_ESTABLISHED`：選んだmetricと有効な費用情報の範囲内に限定。costの上界同士の順位だけで実費優位を確定しない。
- `TECHNICAL_STOP`：provenance、実行前提、resource、correctness等の技術失敗を記録して停止。automatic retryなし。

これは将来のschema要件候補であり、今回これらの新しい結果は発行しない。将来は元のπ3区分と新shot/cost分類を別列にし、誤って過去の区分を上書きしない。

## 5. SOURCE・保存・guard候補

今回のSOURCEは非数値の設計specとmetadata checkerのみ。数値engine・runnerは未実装。`future_source_identity=null`。closed guardは契約内容に関係なく例外で終了する。flagだけの変更で実行を開放できない。schemaは今回の未採択候補だけを受理し、将来の正式契約は別versionで作る。

将来の最小SOURCE案は、保存入力をidentity確認後に読むadapter、intervalのsin/logとinteger ceiling、同一confidence条件の比較、全grid ledger、source-bound契約・authorization gate、結果保全を分離する。cost adapterはM1-S固有wrapperの意味論と精度を検証してから別実装する。詳細は[技術引継ぎ](m1s_shot_cost_source_resource_and_handoff_20261017.md)。

将来結果の必須field候補：source/contract/input commit・SHA256、選択γ/profile、α/scope、coverage、A^-／b^+／qの区間、S／n／N区間、全method/contrast ID、旧π3区分、新整数分類、cost componentと除外項、resource実測、失敗・未評価理由。decimal表示だけを正本にせず、exact interval端点を保管する。現時点ではfield名の設計だけで値はない。

## 6. resource候補（全て未採択）

保存信号stage SはCPU1 worker・GPUなし・全thread1・aggregate memory2GiB、input16MiB、result/log各16MiB、既存108方式行inventory・48対照を超えない候補とする。保存finite-signal、V1、π3のfileサイズと小規模scalar処理に基づく候補であり、実H4 runtime・数値correctnessの実測ではない。算術はp=128 bitsの固定interval、integer bit cap65536を技術候補とする。旧V1の実装資産と分数上限を参照した開始候補であり、整数境界が解像する保証ではない。γ評価点数と演算call capはD1確定後に別SOURCEの静的経路から有限値を決める。これらの未作成SOURCEに依存する技術capsは科学的未決事項とは分ける。

新しい契約のCPU/wall time capはユーザーの現行policyどおり既定なし。旧契約のcapには触らない。native stage Cのworkers・memory・output・compile/call capはprofile・回路範囲が未定なのでnull。cache・seed・compiler threads・aggregate memoryは実行前に固定する。上限をnullにしたまま本番runnerを起動しない。

## 7. 次の順序・STOP

1. ユーザーがD1–D7と最小実施stageの科学条件を判断する。
2. Codexがその範囲の数値SOURCE・resource算定・schema・guardを準備し、固定最終契約候補を公開する。
3. ユーザーがその契約を正式採択する。採択だけでは計算しない。
4. ユーザーの別途source-bound科学実施承認後、一回の承認範囲を実施・保全してmandatory STOPする。

今回、正式採択記録・科学実施authorization・grant・markerを新設しない。旧v4/v5/独立one-shot/pilot/V1/π3のSOURCE・結果・契約・消費済markerを変更しない。**新規科学計算0。Mandatory STOP。**
