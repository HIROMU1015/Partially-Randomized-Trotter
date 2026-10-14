# Track A D0：保存値リソース再会計one-shot結果

**全Phase Aの完全一致後、Phase B/Cを一度実行し正常終了した。** 旧precision 1,088件、旧費用map 2,659件が一致。5ε×factor1/10の1,360precision行と6metricの適格費用4,956行を保存した。新signal・trajectory・compile・SCF/DF/state生成・科学array読込・H8操作は全て0。母平均winnerや新しい科学的結論は採択していない。

利用者の明示指示により2026-10-15 JSTに実行した。branch／資料の`20261014`は利用者指定の識別子を維持している。主要入口は本書と[結果bundle](../../artifacts/resource_applicability/track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/README.md)。旧[準備資料](track_a_d0_saved_value_reanalysis_preflight_20261014.md)・原結果・科学レビュー・grant履歴は保全した。

## 1. 固定SOURCE・入力・実行認可

解析SOURCE **`fa75c7997e135da7b7a0c4e23949e47252bd7a2e`**、準備REVIEW **`b941565835c539fc414d3a75dcb8ca439561a4d5`**。originの準備branchはこのREVIEWに一致し、GitHub固定SOURCE／契約9fileと原本の公開入力7fileをbyte/SHA確認した。[固定契約](../../artifacts/resource_applicability/track_a_d0_saved_value_preflight/2026-10-14/d0_reanalysis_contract_v1.json)のSHA256は **`8d9e908322aaf115861108fa698b13a994a2f0a676e1a48f80a2dbf77fd4a311`**。

8scalar/CSV入力、合計 **4,984,136 bytes**。完全な入力SHAは[入力索引](../../artifacts/resource_applicability/track_a_d0_saved_value_preflight/2026-10-14/d0_input_inventory_v1.json)と結果provenanceに保持した。実行前・結果監査後ともbyte SHAが一致。H6 private保存scalarは既存receiptの1,382,011 bytes／`b9f6cb3dc1a168ddc409833c78a617a7ddd05a3a41e0665e9dec47346d516dfe`に一致し、公開原JSON・NPZ・runtime/cache/checkpointは追加commitしない。

|系列|候補／費用標本|固定科学／実行SOURCE・result|
|---|---|---|
|H4 native fixed16|full B1 8＋prefix3 B2 8。det n1、PR n32|SOURCE217c971dd537b1cc7fbc13742bc0f6c263ff7f45／result0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41|
|H6 exploration92|B0 12＋full B1 8＋B2 72。det n1、PR n8|science5a903aea2492c2d211c1e15e7737df0d50694c12、execution78a39051d8b6d41c33d592ecfe172f8824adfaeb／resultdf99d6ee28942df8d12aac3c4db7cd202df55d77|
|H6 confirmation union8|PR8各独立n32＋登録det20（費用あり16）|SOURCE87b6b9f46b7e8ba9abe132ee21478c114b6f46f7／result321cd8c3066194fe98741cbebbcdba1515d1deaa|

全系列linear 1.0Å/STO-3G/T=.8。H4はlegacy explicit DF12、α2β2/8mode/sector36。H6はtol1e−8/cutoff0/signed generation-weighted projectionのDF19、α3β3/12mode/sector400。H/state/snapshot hashesは契約/provenanceへ固定した。六距離H4 geometryは混ぜず、DF政策を統一したstrict size-scaling比較と扱わない。

symmetric directional full wrapper・structured native Gaussian lowering、保存compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/no topologyを維持。D0ではcompilerを実行していない。旧compilerのoutput bit完全同一性は未検証のまま。

今回の利用者指示を[専用authorization](../../artifacts/resource_applicability/track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/authorization_v1.json)に結合した。SOURCE・契約・入力・Python3.12.3/runtime file hash・output・資源・marker・STOPを固定し、旧grantを流用していない。解析SOURCEと契約はbyte不変。全系列Aを先行する[運用driver](../../artifacts/resource_applicability/track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/execute_fixed_d0_once.py)は、固定moduleのaccounting/weighted_metric/verify_original/analyze_scope/family_selection/fixed_crossing/lower_envelope APIを使用する。新shot/cost/selection/P規則は導入していない。

## 2. Phase A：旧値完全一致

|系列|完全precision_rows比較|既存weighted mapのN/G/paired SE比較|
|---|---:|---:|
|H4 fixed16|128 PASS|51 PASS（RZ）|
|H6 exploration92|736 PASS|2,072 PASS（RZ/CX/depth/size）|
|H6 union8＋det baseline|224 PASS|536 PASS（同4metric）|
|合計|**1,088 PASS**|**2,659 PASS**|

旧4ε `.05/.01/.005/.001`、factor1/10の完全precision_rowsをPython数値/dictのexact equalityで確認した。bias/u/log_Bからのheadroom、eligibility、axis N、N_total、log-only/integer guardを含む。許容差変更・SOURCE変更・再実行は0。全3系列PASSまでは新.0005・未集計metric・P計算へ進まなかった。[original equality](../../artifacts/resource_applicability/track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/original_equality_v1.json)が証拠。

これは固定保存値の再会計一致であり、旧環境／compiler全体の同一性、新signalの正確性や経験的uの証明ではない。

## 3. Phase B：ε=.0005の適格性・RZ点最良

以下はfactor1、P=0。費用は必要shot見積り×保存compiled countであり、hardware shotsの実測ではない。

|系列|適格B0/B1/B2|B2のRZ点最良|G_RZ(PR)|G_RZ(full PF)|PR/full PF点比|
|---|---|---|---:|---:|---:|
|H4 fixed16|未登録/4/2|p3/q8/r2/K2|5.3246451e12|1.2392575e13|0.429664（57.03%低い）|
|H6 exploration92|1/4/18|p5/q8/r2/K4|3.2696191e13|6.5737570e13|0.497374（50.26%低い）|
|H6 union8＋det|1/4/2|p5/q8/r2/K2|3.2627743e13|6.5737570e13|0.496333（50.37%低い）|

PF点最良はH4 p12/q4/global S4、H6 p19/q4/global S4。PRのpaired SEは順に1.2471297e10（n32）、6.2199847e10（n8）、6.0429890e10（独立n32）。SEは記述的な標本SEで、CI／母平均保証ではない。

旧ε=.001のfull-PF点比はH4 .510507、H6探索 .387811、確認 .387006。新精度でratioが必ず改善するという結論ではない。[全30scope/ε/factor行](../../artifacts/resource_applicability/track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/precision_summary_v1.csv)と[candidate switches](../../artifacts/resource_applicability/track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/candidate_switches_v1.json)に旧精度との候補切替を保存した。

factor10でも全5精度・6metricのfamily点最良IDと適格件数はfactor1と同じだった。費用値を同一と省略せず、両factorの結果を保持した。経験的u×10感度の安定性を数値certificateと扱わない。

## 4. 6指標の最良候補・比較

`G_m=Σ_axis N_axis mean(C_m,axis)`。旧H4はpaired integer totalのmean、旧H6はaxis mean×Nのsumという演算順序を維持し、exact rational診断値も保存した。RZ選抜候補の他metricと、metricごとの再選択を別fieldに記録した。

ε=.0005/factor1のPR点最良は次の通り。全familyのG/SE/NとRZ-selected比較は[540行のmetric表](../../artifacts/resource_applicability/track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/metric_selection_summary_v1.csv)。

|metric|H4 fixed16|H6探索92|H6確認union8|
|---|---|---|---|
|RZ|p3/q8/r2/K2|p5/q8/r2/K4|p5/q8/r2/K2|
|CX|同K2|同r2/K4|同r2/K2|
|RZ depth|p3/q8/r2/K4|同r2/K4|同r2/K2|
|CX depth|p3/q8/r2/K4|**p5/q8/r4/K2：旧union8外**|同r2/K2|
|Total depth|p3/q8/r2/K4|同r2/K4|同r2/K2|
|Size|p3/q8/r2/K2|同r2/K4|同r2/K2|

RZ点最良が他metricでも必ず最良ではない。H6探索のCX_depth選抜は旧確認集合に含まれず、探索8pairの点推定である。追加32確認を開始していない。[確認coverage](../../artifacts/resource_applicability/track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/confirmation_coverage_v1.json)へ両factorを保存。

例えばH6確認.0005ではG_RZ=3.26277e13、G_CX=1.40259e13、G_total-depth=1.24047e13、G_size=6.72112e13。単位が異なるためこの数値の大小で物理costを一つに換算しない。one-shot axis mean/max depthはcandidateのmetric内に別fieldで保持。shot-weighted depthはdepth-shotというserial会計量であり、hardware runtime／critical path／T-depthではない。

## 5. Phase C：仮想Pの固定pairとlower envelope

`G_j(P)=G_j(0)+N_total,j P`、P≥0。全180scope/ε/factor/metric曲線と420固定pairの交点・平行・同一直線・負domain除外を[preparation sensitivity](../../artifacts/resource_applicability/track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/preparation_sensitivity_v1.json)へ保存した。RZ/CX/各depth/size per shotの単位は別々。実state-prep回路を取得したことにはしない。

ε=.0005/factor1のRZ/shot感度：

|系列|P=0で選んだPR/full-PF固定pair交点|全候補lower envelopeの切替|
|---|---:|---|
|H4 fixed16|87,189.49|PR p3/q8/r2/K2 →P=87,189.49でfull S4 q4 →P=2,689,221.98でfull S4 q8|
|H6探索92|172,573.82|PR p5/q8/r2/K4 →P=58,354.47で**PR r4/K2** →P=179,714.39でfull S4 q4 →P=7,104,961.77でfull S4 q8|
|H6確認union8|172,936.15|PR p5/q8/r2/K2 →P=172,936.15でfull S4 q4 →P=7,104,961.77でfull S4 q8|

H6探索では固定pair交点と全面再選択のPF移行点が異なる。中間で選ばれるr4/K2は確認union8外なので、確認32pairで裏づけられたenvelopeと呼ばない。確認結果のenvelopeは取得済みunion8＋detに限定される。境界ties・exact rational交点を保存し、上表は表示丸めのみ。

## 6. Paired SE・欠測・選択後推定の限界

cos/sinを同一trajectory pairにまとめ、cell/phase/replica/seed/event/compilerを照合した。`SE=stdev(N_cos C_cos,i+N_sin C_sin,i)/√n`で、2n軸を独立標本と数えない。H4 PR n32、H6探索n8、H6確認n32を分離。det n1はSE=null。

計1,360precision行のうち費用適格826、不適格534。MISSING／NUMERICALLY_UNRESOLVEDは両factor/全5精度で0だった。ただしH6 B0 p5の4qはcompiled cost自体が未取得で、今回も不適格として費用計算しなかった。MISSING0を全92候補のcompiled cost取得済みと解釈しない。H4 native B0/B0+Iも未登録であり、費用0という対照ではない。

新.0005でsignalやtrajectoryは増えていない。探索8pairの選抜後点推定、確認union8内でも確認結果を見た点選抜、fixed H4 panelという範囲を維持する。未観測rare-event費用の補完、母平均tail bound、familywise confidence/winner、新独立確認は0。

## 7. 技術監査・資源・終了

[技術監査](../../artifacts/resource_applicability/track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/technical_audit_v1.json)は、固定input SHA／1,088旧precision／2,659旧costに加え、1,780整数shot軸、4,956metric point/exact rational/paired SE、420交点、1,872 envelope区間・vertex検査に合格した。結果監査は既存outputsの整合性照合であり、one-shot pipelineのretryではない。local技術監査で、独立科学review／CIではない。

|資源|実測／上限|
|---|---|
|Wall|2.223352秒／300秒（pipeline window。Python startupの一部は外側）|
|CPU|native process累積2.257601秒／240秒|
|peak RSS|56,774,656 bytes＝54.1445 MiB／256MiB|
|AS|hard512MiBをprocess限定で設定。超過STOPなし|
|6結果bytes|**7,682,915 bytes**。payloadとterminal reserve内|
|観測|584回、最大guard interval .390674秒|
|process/thread/CPU|1/1/同時1core、継承affinity不変。taskset0、子process0|

直前のeffective memory約501.55GB、host full PSI0、可視own ancestor pressure0。容量/inode/mount quota options gate合格。runtimeでhost/ancestor OOM増分なし。歴史的非zero OOM baselineを今回のOOMと補完しない。資源予約・shared設定・venv変更・他job操作0。

[native exit receipt](../../artifacts/resource_applicability/track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/native_exit_receipt_v1.json)：exit0、wait完了、first STOP null、own PID退出確認。正確なnative exit timestampは未記録/nullで、終了確認時刻と区別した。専用outputと`output.oneshot`はhomeの監査directoryに保持し、削除・再使用していない。

## 8. 既存結論との関係・未解決事項

旧4εをそのまま再現し、native限定の新.0005・metric・P感度を補った。H4 geometryやR09の科学結論、原稿、候補registry、科学SOURCEを改訂していない。P=0・経験的u・保存compiler・有限候補/標本に条件付けた点比較であり、普遍的PR優位／実state-prep込み削減／energyやchemical accuracyの達成ではない。

今回scopeの技術的未解決は検出されなかった。未取得native B0+I、policyを統一したsize比較、q>8、母平均rare tail/同時保証、実hardware state preparationは別の証拠不足として維持する。解析の数値一致からcompiler出力bit equivalenceを認定しない。

H8正式result `681ec23c93fdd1090d325908f4b155a79115afdf`、SOURCE c31ef424…は公開済みというprovenanceのみを維持し、D0のinputs/候補比較へ追加していない。H8 signal/trajectory/compile/reaccount/runtime/checkpoint/ledger変更0。

## 9. 新しい科学判断が必要な事項

次の科学担当には、.0005の正式評価domainへの採用と点比較の記述、RZ選抜／各metric再選択をどう論文へ表示するか、確認union8外のCX_depth/P選抜を探索としてどう扱うか、仮想Pの意味とnative政策差を明記する比較範囲を返す。今回GPT詳細科学reviewは開始していない。

これらを理由に追加確認・conditional sampling・native B0+I・H5/H7・T変更・H8 D0へ自動的に進まない。実行終了後の再試行は0。

## 10. 公開・次の担当・mandatory STOP

branch `track-a-d0-saved-value-reanalysis-result-20261014` に6curated JSON、summary/監査、専用authorizationと運用driver、研究ノート・軽量索引だけを明示stage/commitする。NPZ・元private signal JSON・raw runtime/checkpoint/cache/ledger・credential・内部SSH情報は含めない。manifestに各file bytes/SHA、終了後receiptにactual result commit/remote SHAを記録する。

non-force pushとremote/file byte照合を行い、認証失敗時はGit設定を変えず未公開状態と手動commandを報告する。次は利用者による結果確認と、必要な場合の科学review開始判断。**COMPLETE_MANDATORY_STOP**。source/contract/resource変更・retry・D1/D2/D3/H8拡張を開始しない。
