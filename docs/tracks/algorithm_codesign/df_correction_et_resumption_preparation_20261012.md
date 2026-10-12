# DF correction E/T resumption v4：受理改訂・技術準備

2026-10-12 JST。**READY_FOR_SEPARATE_E_T_ONE_SHOT_AUTHORIZATION / mandatory STOP**。
今回の利用者指示と[GPT科学判断原本](../../research/track_b_e_r3_identity_acceptance_and_resumption_review_20261012.md)に従い、E/T再開の技術準備だけを行った。登録E/Tの実行認可ではない。

## 来歴と改訂範囲

基点は公開済み診断commit `ef0b2419df2d25187ad14e865ae0a38d17a04628`。
専用branchは `track-b-df-correction-et-resumption-preparation-20261012`。
レビュー原本34,446 bytes／SHA256 `b3342726c5d353a122393faa15c685a7ea0314922a613e0a648d4eccdb8e76b7`と共通指示原本を選択してcommit `1901b4b018f05aedf94d0ca5e786f8d91e84e714`へ保存し、GitHubからbytes/SHA256一致で再取得した。
旧science source `41349b73157c7781e0b534cf0fe61e06bf439954`、旧authorization、旧TECHNICAL_INCONCLUSIVE、消費済みmarker、診断source/resultを変更しない。

新sourceの入口は [runner](../../../scripts/tracks/algorithm_codesign/run_df_correction_et_resumption_v4.py)、
[専用namespace](../../../src/trottertracks/algorithm_codesign/df_correction_et_resumption/)、
[機械契約](../../../artifacts/track_b_df_correction_et_resumption_preparation/2026-10-12/contract_v4.json)。
最終source SHAと公開確認HEADは `source_publication_receipt.json` を読む。公開確認commitを将来の実行source／authorization-only childと混同しない。

これは旧失敗と登録診断を見た後の結果前amendment。旧契約を無修正で完遂した実験とは呼ばない。

## 四層の受理

1. **入力・条件**：保存E-inputsのr3／r6、rank2×3を読込み、file bytes/SHA256と各frameの順序・shape・complex128 bytesを確認。seed7101107の再生成・draw追加はしない。signs(+,-,+)、nested commutator、u=1/128、uncontrolled／MSB controlled、compiler basis/seed/version/opt0を維持。
2. **演算子**：旧Ball組立、exact-frame Gram-Schmidtと独立spectral reference、strict-phase Operator guardをそのまま用いる。raw expmを理想targetに読み替えない。numeric2e-7、synthesis8e-7、合計1e-6を維持。Bのproxy・linearization・有限bit会計も旧式を用いる。
3. **実装・費用**：各nativeの完全なordered gate/wire/raw angle/global phase IRを保存し、RZ inventory・CX/1Q費用との整合とIR digestを確認。新native結果の全bytesとIR identityをT inventoryへ固定。freeze receipt→job→worker→sequence→最終集計のSHA bindingを継承し、native実装の取り違えも拒否する。
4. **表現来歴**：shared recursive payload/hashを名前も含め無加工保存。historical expectedは不変。不一致だけでは不受理とせず、None／payload例外／保存失敗はtechnical STOP。全名前削除、新expected、counter reset、shared hash policy変更は行わない。

A/Bの実装target、bias、weight、momentは方式別のまま。AとBの有限平均が同一と仮定しない。

## CD保存値と費用の境界

CDの6行は旧runの完了substageとして、原CDの1,118,571 bytes／SHA256 `98f62b87107681a931d0502f37b81eea9806c890ef278c64532b7d5d60dbd075`、chunk、旧marker/terminal/summaryを照合して参照する。`INCOMPLETE_RUN_DIAGNOSTIC_ONLY`と旧run分類は保持する。
CDを再計算・再最適化・再認証しない。新runner／workerにはCDやE recipeを実行するstageを設けていない。
CD native費用とprep/readout費用は記号変数のまま。別fixtureのE単価とCD補正頻度を掛けたPR総費用、全PR winner、新規性は判定しない。旧Phase E費用も新単価として流用しない。

## 固定実行計画と資源上限

将来認可される場合、r3／r6を各1 workerで並列評価（各worker内ではuncontrolled→controlled、代表B support-weight順序を維持）。両rankのnative完了後だけT inventoryをfreezeし、B→phase→登録A位置の順を維持して各batch最大2 workers。BLAS/OpenMP/MKL/Rayon1、QISKIT_PARALLEL=FALSE。mainで候補やcompilerを選び直さない。

| scope | 固定上限 |
| --- | --- |
| 全体 | wall2400s／CPU1800s／aggregate RSS3072MiB |
| native | wall600s／CPU450s／worker RSS1024MiB／aggregate2560MiB／compile22 |
| T | wall1200s／CPU900s／worker RSS512MiB／aggregate1536MiB／keys64 |
| T per key | wall30s／CPU20s、retry0 |
| output/log | 64MiB／4MiB、terminal reserve256KiB |
| review files | 512KiB以下、256KiB UTF-8 chunks、元bytes/SHA復元 |

wall capは今回GPT review §6.2で明示維持された旧上限であり、projectの将来新規契約に対するno-wall defaultを変更しない。
static timing/sourceから主要候補はr6のQSD/Operator・有理Pauli transform・T keys。CPU32とaffinity/memoryをread-only記録。GPU query/use0。既定の2 workers・threads1と既存cache／chunk helperを用い、capを増やさない。speedup測定や登録domainの完遂保証はない。

T単価は新nativeからのliteral keys、旧strict raw-radian adapterとpygridsynth2.0.0、dps100／up_to_phase=false、gate precision1/40000000000、既定seed/selection/no refillを維持する。
Aの部分Tは固定加法completion下界。全coverage・numeric・phaseが揃わなければ全TはT_INCONCLUSIVE。Bのcomplete上界には全label coverage・uniform numeric bound・RZ数×gate精度のsynthesis allowanceを要する。最短合成や他のwhole分解の下界と呼ばない。

## 技術検証と証拠

[focused receipt](../../../artifacts/track_b_df_correction_et_resumption_preparation/2026-10-12/focused_validation_receipt.json)：**141 PASS／FAIL0／skip0**（新53、旧数理/native/guard/storage/inventory回帰88）。固定.venv311、threads1、full suiteなし。

登録外1/2-mode・rank1・Pauli control、mockだけで、名前差によるhistorical mismatchの許容、payload/保存例外STOP、入力変更、符号、strict controlled位相、wire変更、native IR/count/angle/phase改ざん、T sequence／moment関連既存会計、source/manifest/authorization/marker/remoteの不整合、namespace reuse、partial/terminal保存失敗を検証。
誤ったraw expmをidentityへ置換するoff-domain controlは、独立spectral referenceのguardで拒否された。正しいsmall native pipelineも通過した。登録r3/r6の誤差marginやT coverageを保証するものではない。

初回new48 PASS→combined138 PASS/1 FAIL（mockによるPYTHONPATH漏れ）→テスト環境復元追加後139 PASS→独立target/terminal保存controls追加後141 PASS。全試行log/XMLを保持。PYTHONPATH問題のためにproductionや固定runtimeを修正していない。
[固定runtime照合](../../../artifacts/track_b_df_correction_et_resumption_preparation/2026-10-12/runtime_preflight.json)はnumeric／synthesis両方PASS。backend合成は呼んでいない。
[静的/source監査](../../../artifacts/track_b_df_correction_et_resumption_preparation/2026-10-12/preparation_audit.json)はsource差分、科学条件、保存値identity、保護対象、未認可拒否、fresh namespaceを確認する。

## one-shot保護・次の手順

新result namespaceは `artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v1/`。
準備時は存在しない。pending authorizationでruns0／science_execution_authorized=false。現在の指示は準備だけで、authorization-only commitは作らない。
将来の認可は凍結source Sの**唯一のparentがSである直接子A**から開始し、変更pathを新authorization JSON＋新receiptだけに限定する。明示指示の原文、source、contract SHA、manifest SHA、run1/retry0/STOPを記録しpush。HEAD/remote A、clean、fresh namespaceを確認してexclusive markerを一度消費する。
childはsource/contract/manifest/markerを確認し、各jobのexclusive started receiptで再起動を拒否。cap／payload／numeric／chunk／inventory／sequence failureはtechnical inconclusiveでSTOP。terminal保存失敗は別minimal receipt、さらに保存不能ならouter processがnonzero exitとmarkerを保存しprefixを受理しない。

GPT review §9/§11に従い、旧source-review要件をこのamendmentの科学判断＋Codex技術確認へ明示更新した。新commitだけを理由にGPT詳細reviewを反復しない。数理保証・phase・fixture・A/B比較・科学受理の本質変更が必要なら、その問題に限定してGPTへ戻す。

**mandatory STOP。登録E/T／CD再計算／identity再診断／新T合成／G10／RA-D0／分子／NPZ／trajectory／sampling／LP／GPUは0。**
次は別途のユーザー明示one-shot承認。実測A/B結果の科学的解釈・継続判断はGPT。
