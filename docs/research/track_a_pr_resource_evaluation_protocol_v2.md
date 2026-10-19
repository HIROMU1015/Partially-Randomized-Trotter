# Track A PRリソース評価：共通実験プロトコル v2（正式採用）

> **2026-10-19 H2完了・正式受入の追記（ユーザー指示により適用済み）**：H2の現行状態は末尾の「10. H2独立確認P完了・正式受入」を優先する。以下§1–9に残るH2の「P未seal・未実行」「提案n未採用」「最終SOURCE/seed未固定」は、初回採択時snapshot `829d697a` の履歴である。H4/H6/H8の既存条件・判定・制約は変更しない。[現行manifest](../../artifacts/resource_applicability/track_a_protocol_v2_h2_update/2026-10-19/protocol_manifest.json)と[追記記録](../../artifacts/resource_applicability/track_a_protocol_v2_h2_update/2026-10-19/update_record.json)を参照。初回採択bundleはcommit `1f001cfc4e8e0531279a26613be29a52c7c0640c` 時点の不変snapshotとして読む。

2026-10-19（Asia/Tokyo）。**ADOPTED_V2_DOCUMENT_ONLY_NO_EXECUTION_AUTHORIZATION。ユーザーの指示により正式採用。**

文書基準の基点は[採用済みv1（9036d8a8）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9036d8a87ad1c38e83b9ff654b412dde7041e9d3/docs/research/track_a_pr_resource_evaluation_protocol_v1.md)。v1の本文・採択記録・例外表は変更しない。
採用元は[固定v2改訂案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0808b254aa02a94c7482f5c6e353b54532a54f8/docs/research/track_a_pr_resource_evaluation_protocol_v2_DRAFT.md)、commit `c0808b254aa02a94c7482f5c6e353b54532a54f8`。
文書実装の来歴はH4/H6/H8掲載用整備 `1923fce8f7461513d1bc47b5061e87cd70917e87`。v1と採用元草案は履歴として保全する。
[採用記録](../../artifacts/resource_applicability/track_a_protocol_v2_adopted/2026-10-19/adoption_record.json)と[正式版manifest](../../artifacts/resource_applicability/track_a_protocol_v2_adopted/2026-10-19/protocol_manifest.json)を参照。
本版は採用元の固定snapshotを文書基準とし、採用時に他研究の進行状態を再照会・変更するものではない。
本版は採用元で確認した公開SOURCE・固定結果・契約・監査を整理し、**H2をv1対象外からプロトコル対象へ追加する**。
既存の科学契約、保存条件、受入閾値、grant、結果、STOPへ遡及適用しない。本版の正式採用も個別科学実行の承認とは別である。

入口：[変更点一覧](track_a_pr_resource_protocol_v2_changes_20261019.md)、[証拠bundle](../../artifacts/resource_applicability/track_a_protocol_v2_draft/2026-10-19/README.md)、[実施状況](../../artifacts/resource_applicability/track_a_protocol_v2_draft/2026-10-19/stage_status.csv)、[入力条件](../../artifacts/resource_applicability/track_a_protocol_v2_draft/2026-10-19/system_conditions.json)。
GitHubの関連branch headを読取確認し、固定した67公開資料を実取得した。snapshotはH2最新 `829d697a`（当初の `7b202a1a` は履歴）、H4/H6/H8掲載資料 `1923fce8`、H8 P `e6eea538`、Mechanism `1ff6ac0b`等に結合する。
[branch観測](../../artifacts/resource_applicability/track_a_protocol_v2_draft/2026-10-19/branch_observation.json)は他チャットの進行を凍結する規則ではない。
SOURCEの規則、保存projection、公開監査receipt、報告書の申告、原private witnessの再監査は区別する。

> **履歴注記（冒頭の公開snapshot）**：上記のH2「最新829d697a」と公開資料一覧は初回採用時点の履歴である。H2の最新の完了・受入状態は[§10](#h2-current-status)を参照。

## 1. 対象・比較単位・状態

対象は保存DF Hamiltonian・保存状態における有限時間の複素signal推定資源。linear H2/H4/H6/H8、隣接距離1.0 Å、STO-3Gを整理する。
energy/QPE/RPE全round、実backend/noise、状態生成の物理的無料性、FT T資源、母平均保証の完成を意味しない。

比較キーはv1を継承し、分子・geometry・basis・入力policy版・Hamiltonian/state/snapshot hash・sector/bit順・DF政策/実rank・fragment順・PF係数版・L_D/q/r/K/R・時間/δ・bias/allowance/shot会計・control/preparation/compiler/runtime・cohort・PF集合版・結果commitを含む。
候補名、同じrank、同じcompiler optionsだけで同一実験へjoinしない。

`COMPLETE`は記録された固定scopeの完了、`NOT_EXECUTED`は公開記録で未実行と明示されたstage、`PENDING`は未固定/判断待ち、`NOT_VERIFIED`は証拠未照合を表す。
公開証拠がないだけの項目を未実行と断定しない。完了、正式受入、共通入力gate、掲載採否、母平均保証を別軸とする。
H2の追加は**正式採用されたプロトコルの対象拡張**であり、現在の論文主表へH2のPR比較を追加採用したものではない。

## 2. 現在の入力条件と相違

|系列|DF政策・実rank|物理sectorと状態生成|正規化・残差等の記録|証拠の限定|
|---|---|---|---|---|
|H2新tol-only|threshold1e-8、final_rankなし、返却rank3|4 qubits、Nα=Nβ=1、sector4、dense scipy.linalg.eigh|最大振幅の位相固定、norm gate1e-12、残差gate1e-10、gap gate1e-12、保存後再正規化なし|公開F generation_policy/SOURCEと完了監査。元NPZは今回decodeしていない。norm値自体は別field未保存|
|H4新tol-only|同policy、rank10|8 qubits、Nα=Nβ=2、sector36、dense eigh|正規化eigh vector＋位相固定、残差gate1e-10、保存後再正規化なし|保存F/FPをD190/A50/Pで再利用。H4 DF12を読み替えない|
|H6 tol-only|同policy、rank19|12 qubits、Nα=Nβ=3、sector400、記録上eigsh SA/HF初期vector|記録上k1/tol1e-12/ncv40/maxiter1000、一度のbinary64正規化、absolute1e-9/relative1e-10残差gate|具体設定は保存policy比較/Methodsの記録。全原provider/input witnessを今回再監査していない|
|H8 tol-only|同policy、rank28|16 qubits、Nα=Nβ=4、sector4900、記録上eigsh SA/HF初期vector|H6と同じ記録上のsolver・正規化/残差規則|公開designのbindingとSOURCEを確認。原profile・実run全closureの同一性は未確認|
|H4 legacy|**固定DF12、別入力系列**|旧保存state・旧D/P/探索系列|元契約のまま保全|Figure2AのH4等。新DF10の費用・状態・Pへ移送しない|

H2 Fにはcharge0/multiplicity1、RHF、SCF conv_tol1e-9/max_cycle50、Angstrom、symmetryなしが保存されている。
H4/H6/H8の公開比較記録には同じRHF/canonical MO helper、spin mappingとtwo-bodyの1/2規約、signed weights・generation順・cutoff0等の一致する設定がある。
H4/H6/H8のcharge/multiplicityをこの入力一覧で独立に抽出し直していない場合はnullと理由を残し、電子数から推測しない。
MO位相/順序、初期化、library/runtime、projectionの数値工程、state solver・縮退処理まで**全生成規則が同一とは認定しない**。
weighted Hermitian projectionは工学的検査で、interval certificateではない。異なるsolverや残差gateをv2で一律の新値へ変更しない。

旧H2の設定選択rank3・sector6と、新H2 tol-only rank3・spin sector4は別snapshot。rank値の一致を入力一致としない。
DF政策を揃えたことだけで分子サイズの単独因果効果、普遍的優位、漸近scaling、H10/H12への外挿を主張しない。

## 3. 時間・精度・回路・compiler・費用

> **履歴注記（§3・精度表）**：下表H2の「n80/12は未採用提案」「P最終SOURCE/seed未固定」は初回採用時点の履歴である。最新状態は[§10](#h2-current-status)：P完了、n=80/12採用、2条件ともREPRODUCED（有限標本）。

|項目|現在の主条件・継承規則|限定|
|---|---|---|
|時間|T=.8、δ=T/q。保存されたq scheduleを使う|未登録qへの補間や自動拡張なし|
|精度|D/Aの主5ε=.05/.01/.005/.001/.0005、独立Pの主対象は両端|H2 Pは両端の2候補を保存草案で固定、n80/12は未採用提案。P最終SOURCE/seed未固定|
|状態準備費用|P=0|状態生成が物理的に無料という主張ではない|
|会計|通常signed bias、allowance倍率1。保存の軸精度配分・shot会計を継承|経験的allowanceと厳密bias保証を区別。非相殺や倍率10の欠測を補完しない|
|回路|full cosine/sine Hadamard wrapper、symmetric_directional、元constant＋tail identity一回、reference不回転|ordinary_ancilla_branchは低I/高Uの意味論guardで、別lowering政策への変更ではない|
|compiler|Qiskit1.3.0、rz/sx/x/cx、opt1、seed_transpiler17、backend/coupling/layout/routing等の未指定を保存|runtime・provider・profile・fingerprintの同一性は別照合。H8原profile bytesと旧出力完全同一性は未確認|
|費用|両軸のcompiled RZ×保存analytic shotを合計。PRはpaired trajectory費用の保存mean/SE|実quantum shots、非Clifford数、FT T count、最終energy/RPE費用ではない|

v1のα_total=.05/α_axis=.025等は当該保存会計SOURCEへ結合した規則として保持する。古典的な費用標本数設計のfamily α/powerとは異なる。
保存値の再引用は許されるが、本改訂では新しいshot会計・mean/SE・差・比・CI・n・順位を計算しない。
`DIRECT_MEASURED` / `MODEL_ESTIMATE` / `MISSING` / `UNDETERMINED`を保持し、未知を不適格・費用0へ変えない。

## 4. 固定候補集合と到達点

[完全な登録候補の転記](../../artifacts/resource_applicability/track_a_protocol_v2_draft/2026-10-19/candidate_registrations.json)を参照。下表は登録規則の要約であり、新候補を生成・追加する契約ではない。

|系・現行系列|決定論的PF対照|PR探索|時間窓・比較範囲|
|---|---|---|---|
|H2 tol-only DF3|D70：B0/B0I p0/1/2、full B1 p3、q1/2/4/8/16、S2/S4|A66完了・新n8：p0/1/2、r4で全q・K2/4。r2/8はq1/4/16に限定|δ=.8/.4/.2/.1/.05。full-rank B0/B0I aliasを重複登録せず、空tail B2を除外。p0のideal同値でもnative分解を区別|
|H4 tol-only DF10|D190：B0/B0I p1..9、full B1 p10、q1/2/4/8/16、S2/S4|A50：p1/2/3・r4・K2/4・全q、およびcentral p2のr2/8・K2/4・全q|同δ窓。FP1候補n32＋新49候補n8を分離|
|H6 DF19|強化136：B0/B0I S2 p5..15、S4 p11..15、full p19、q1/2/4/8|疎な登録A16、p3/4/5・q/r/Kは登録表の個別組合せ|Dのδ=.8/.4/.2/.1。Aにはq16の個別点もある。全直積へ拡張しない|
|H8 DF28|強化184：B0/B0I S2 p7..21、S4 p15..21、full p28、q1/2/4/8|疎な登録A16、p6/7/8・q/r/Kは登録表の個別組合せ|Dのδ=.8/.4/.2/.1。A/Pにはq16・δ=.05の個別点もある|

H2 D70とH4 D190は保存済み登録集合の対照で、H6/H8の全域にはmodel推定も残る。独立Pで使った固定PFの直接実測を全域へ一般化しない。
同じ候補数・prefix・n・r/Kを全系へ強制しない。候補境界・非直積schedule・登録subsetを記録し、global optimumとは呼ばない。
m5_best等の新PF採用保留はv1 AD03のまま。本案で新しい高次PF/PR系列を追加採用しない。

### 4.1 H2：v2での追加対象

> **履歴注記（§4.1）**：本節の「P未実行・未seal」、未採用n/null/未承認の欄、および末尾の独立P再現性未判定は初回採用時点の履歴である。最新状態は[§10](#h2-current-status)：P完了、n=80/12採用、2条件ともREPRODUCED。母平均保証の未認定は現在も維持する。

- **F完了**：SOURCE `37b1cf72990885962d8449d8bd5ec7e303329350`、受入済みF commit `7ea6df2dbe34e64a34f468c343bb12922d3e2829`。新tol-only DF3・sector4入力を保存。
- **FP完了**：SOURCE `de72e5d4e48177734c0ddd79dffac14222410658`、完了commit `af724928689be3592b391fab3a14538bf4752b76`。PRは2候補それぞれ**native診断n1**で、Aや独立Pの統計標本ではない。
- **D完了**：結果snapshot `7b202a1a569c6839c4c7a5dfb18b6ff51ec3f3eb`、復旧SOURCE `cf99e4452df96291d7e9d37d76d2e91be5cbc413`。元de72 SOURCEの保存signalを受け入れ、固定追加56 compile軸でD_run02を完了。旧D_run01のSTOP・予約・未採用partialを保全し、過去runを遡及COMPLETEにしていない。
- 保存Dの最良PFは5精度順にB1_p3_q1_2nd、同、B1_p3_q2_2nd、B1_p3_q4_2nd、同。これは保存winnerの引用で、新順位計算ではない。
- **A66完了**：最終seal `a99b23ce`、科学SOURCEはde72のまま、完了commit `1d2f67ec285956ef4090ea26520d64b11e925f9c`。全66候補に適格endpointがあり、新n8、528 trajectory・1056 compiled軸を実行済みと公開完了報告/監査に記録。FPのn1は混合しない。
- **S_STATS完了**：SOURCE `e0b959e8a617f7934d05c3e249832ccc1c43909f`、完了 `829d697ae7905b3ea90a61da9d2d4da15d8b757a`。Aの保存n8 mean/SEとDの保存PF費用だけによる過去のone-shot標本数設計。24シナリオ・2方向・12感度条件を保存済み。今回再計算していない。
- **P未実行・未seal**：保存された2候補はε=.05のB2_p1_q1_2nd_r2_K4（L_D1/δ=.8、PF B1_p3_q1_2nd）とε=.0005のB2_p1_q4_2nd_r2_K4（L_D1/δ=.2、PF B1_p3_q4_2nd）。保存提案nは順に80/12。採用済みnではなく、P_n_final・P_SOURCE・P_seeds・P_grantはnull、P_science_authorized=false。

**時点履歴**：当初指示と初回GitHub確認 `7b202a1a` ではA未実行、S_STATS/P未sealだった。再確認時点（2026-10-19 10:13:46 UTC）に公開branchが829d697aへ進み、ユーザーが「最新829d697aを反映し、当初のA未完了は履歴として残す」を選択したため、現在欄を更新した。
[履歴projection](../../artifacts/resource_applicability/track_a_protocol_v2_draft/2026-10-19/H2_status_history.json)と[現行projection](../../artifacts/resource_applicability/track_a_protocol_v2_draft/2026-10-19/H2_latest_condition_projection.json)を分離し、旧dependency draftのnullや既存STOPを変更しない。

公開Aの点順位は探索n8の結果で、S_STATSはその保存統計による計画である。独立Pの再現性/母平均は未判定。A/S_STATS完了をH4等の独立Pと同じ証拠段階へ昇格させない。

### 4.2 H4/H6/H8：v1後の進捗

H4 tol-onlyはF/FP、D190（406f7057）、A50（85cb5b4d）、S_STATS（d728d088）、P（821cc0f6）が完了。旧DF12 D104/Pは独立系列として維持する。
H6はD136/A16の保存会計後、S_STATS（d229ab9f）、独立P（3cccfcb2）、採用（877e7c28）まで完了。
H8はD184/A16の保存比較と固定n48 P（e6eea538）が完了結果として採用されているが、PR詳細実行監査の公開不足は残る。
H8で独立したS_STATS完了stageを今回の資料から確認できないため、固定n48設計を別のS_STATS実行済みと書き換えない。
H6/H8の初期入力・native取得は既存系列の来歴として扱い、H2/H4と同名の新F/FP完了stageを推測で新設しない。公開資料から同名stageを特定できないことを、未実行の証明にもしない。

H6/H8 Aの取得と後続保存会計のSOURCEは分ける。H6取得は31ef62a1、H8取得は元31ef62a1の代表16軌道＋87f1f0d5の継続112軌道、保存新A会計は8601b921。
H8 D保存会計の修正SOURCE 8ff002ddと初回STOPも区別する。継続SOURCEの変更を新しい独立cohortと数えない。
過去のworker/thread/RAM記録は履歴であり、将来の恒常上限へ移さない。

## 5. 独立確認Pの現在の固定条件

全行linear 1 Å/STO-3G、T=.8/P=0、通常signed/倍率1、表記のDF入力・L_D/δに限定した独立固定cohort。固定PF対照は各掲載条件の直接実測。

|系・DF|ε|固定PR|L_D / δ|n|固定PF|保存方向判定|低い点費用|
|---|---|---|---|---|---|---|---|
|H4 DF10|.05|B2_p3_q1_2nd_r4_K2|3 / .8|1108|B0I_p5_q1_2nd|NOT_REPRODUCED|PF|
|H4 DF10|.0005|B2_p2_q8_2nd_r2_K4|2 / .1|78|B0_p7_q8_2nd|REPRODUCED|PR|
|H6 DF19|.05|B2_p4_q2_2nd_r4_K2|4 / .4|48|B0I_p7_q1_2nd|REPRODUCED|PF|
|H6 DF19|.0005|B2_p4_q8_2nd_r4_K2|4 / .1|104|B0I_p11_q4_4th|REPRODUCED|PR|
|H8 DF28|.05|B2_p8_q2_2nd_r4_K4|8 / .4|48|B0I_p11_q1_2nd|POINT_REPRODUCED|PR|
|H8 DF28|.0005|B2_p7_q16_2nd_r4_K4|7 / .05|48|B0I_p15_q4_4th|POINT_REPRODUCED|PR|

H4粗精度の非再現は正式採用した結果。H6粗精度は事前の**PF優位方向**を再現しており、REPRODUCEDをPR優位の同義語にしない。
H2はこの表に入らない。Pの2候補は保存草案で固定済みだが、最終n/seed/科学SOURCEは未固定で、H8 n48や他系の必要nを移さない。
> **履歴注記（§5・H2段落）**：直前のH2最終n/seed/科学SOURCE未固定は初回採用時点の履歴であり、最新状態は[§10](#h2-current-status)を参照（P完了・n=80/12採用・2条件ともREPRODUCED）。既存のH4/H6/H8主表6条件へH2を混ぜない扱いは維持する。

費用・SEのexact保存値とCEM対応は既存論文packetへ参照し、新しい比・差・CIを作らない。

## 6. sampling・seed・標本数設計

v1 P手順を継承する：候補・PF・期待方向・endpoint・保存shot・nを新sampling前に封印し、元の無条件finite-RTE law、cos/sin同一trajectory、旧master/derived seedの除外、新namespace、固定n完了後判定を維持する。
候補間CRNを無断導入しない。nはtrajectory数で、cos/sinを2nの独立標本にしない。探索・旧union・条件付きpilotとのpooling、候補再選択、早期成功停止、自動追加samplingは禁止。
seed不一致や衝突0だけで統計的独立性を証明したとはしない。

H2 Aはnamespace `H2_FP_D_A_V1_20261019_A` と原direct_master_to_step_seeds_v1で完了。保存activation receiptは528 master/3888 seed値と既知origin/geometry除外の衝突0、seed再抽選なしを報告する。将来の未登録campaignや統計的独立性は保証しない。将来Pの新seed/再照合は未seal。今回新seedを生成・衝突解析していない。
> **履歴注記（§6・seed段落）**：直前の「将来Pの新seed/再照合は未seal」は初回採用時点の履歴である。Pの固定SOURCE/seed・完了証拠を含む最新状態は[§10](#h2-current-status)を参照。新seedや新samplingを今回承認する注記ではない。

H4の保存FP n32、新A49候補n8、P n1108/78を分離する。H6/H8のA16探索n8、旧union n32、新Pも別cohort。

H4 S_STATSとH6 S_STATSは別候補・別設計規則で完了した過去結果。近似normal/保存SE/効果差や、H6のorder2可視性計画を母分散・実power・母平均保証へ昇格させない。
H2 S_STATSも完了済みの別設計で、family α=.05、2片側Bonferroni、power=.8、SD倍率1/2/4/8、相対検出差.01/.02/.05を保存契約に記録。推奨条件SD倍率4/差.02で提案n80/12、提案予算92 trajectory・184 compiled軸・256 RTE eventが保存されている。採用済みn、実power、母平均保証ではない。H2の2候補とも保存S0_ONLY=8で、law確率・rare可視性n・未観測費用はnullのまま保持する。
H8の固定n48を共通最低nとしない。H2 Pはユーザーによるn採否判断、最終seal、別途実行承認が残る。raw分散再集計、bootstrap、不足共分散、必要n数値計算を文書準備の承認から推定しない。
> **履歴注記（§6・標本数と残作業）**：直前2段落のH2「提案n」「採用済みnではない」「n採否判断・seal・実行承認が残る」は初回採用時点の履歴である。最新状態は[§10](#h2-current-status)：n=80/12採用、P完了、2条件ともREPRODUCED。実power・母平均保証は未認定のままである。


## 7. 希少event・公開監査・比較可能性

未観測S2/S4、多重event層の費用・分散・上界を0としない。K2のorder4 support外と、K4での未観測order4を区別する。
H2 FPのS0_ONLY診断、H4/H6の保存order2観測、H8の高次未観測を同じ不確かさとして数値合算しない。
有限標本点順位、保存paired SE、経験的bias allowance、model誤差の検査PASSは、母平均順位・全support tail・厳密bias certificateを保証しない。

H8では公開SOURCE上のdirectional/scalar/reference規則と、選択PFの公開six-axis receiptを確認できる。PF receiptのexit/reap/fingerprintはその決定論jobに限る。
後続PR確認の全grant消費・実seed除外・provider/runtime/input bytes・全worker exit/reap・最終byte照合は公開詳細receiptで再検証できていない。
PR完了は公開結果/報告の範囲で保持し、`REPORTED_LOCAL_PASS_NOT_REVERIFIED_FROM_PUBLIC_RECEIPTS`を消さない。原profile bytes・旧compiler output完全同一性も未確認。
公開designのunapproved状態は設計時点の記録で、後の実行grantの不存在/存在を証明しない。私有資料の不存在を断定せず、無断公開しない。

v1のCore/Extended/Guaranteeと技術/科学STOPの区別を継承する。既存閾値をv2で一律変更せず、非再現・PR不利をtechnical failureやretryの理由にしない。
v1 §3のD/A/P/M/T/Sの数式・演算順・適用範囲・null処理は当該SOURCEへ結合したまま継承する。Mの費用恒等式と説明因子は同じcohortの保存入力だけに適用し、単純な回路比とshot比の積で代用したり、別標本へ移送したりしない。
正常終了の保存receipt、報告文だけの終了、外部再現/immutable CIを区別し、Codex自身の構造チェックを別担当の独立科学レビューと呼ばない。

## 8. 論文manifest・Methods・CEMとの整合

現在の論文主表は**H4/H6/H8独立確認6条件**、Figure2Aはlegacy/探索、2BはH8独立n48。H2はv2プロトコル一覧に加えるだけで、主表・CEMへPR比較を新設しない。
MethodsのH2対象外という記述はこの論文scopeに対して有効。本版の対象拡張と矛盾しない。H2を論文へ追加する場合の主張・証拠段階・Methods/manifest変更は別のユーザー判断とする。

旧A82欠測注記は原図の描画時点の履歴。後続Mechanism `1ff6ac0b`で固定32比較/64軸の説明は完成した。原図34ファイル・数値は不変で、掲載captionは後続完成を補足する。
完成因子は元cohortに限定し、新H4/H6 PやH2へ移送しない。H8主表2条件とFigure2BのH8_CONFIRM_1/3は同じ証拠で、追加独立証拠と数えない。

[整合表](../../artifacts/resource_applicability/track_a_protocol_v2_draft/2026-10-19/paper_alignment.csv)は現行manifest・Methods・CEM14行・台帳316行・captionのpath/SHAと対応規則を示す。旧資料・採用元草案を書き換えず、本正式版を別versionとして保存する。
v2から`COMMON_CORE_READY`、母平均保証、最終evidence freeze、詳細科学レビューの開始を自動認定しない。

## 9. 承認境界・資源・停止

今回許可されたのは公開資料の読取、保存値/条件の転記、文書更新、構造検証、専用commitと承認範囲の公開まで。
科学module実行、SCF/DF/state/reference/signal、sampling、compile、科学的数値/統計/比/shot会計/順位/n計算は0。既存SOURCE・raw・grant・ledger・marker・STOPは保全する。

将来の科学stageは、それぞれの契約SOURCE・入力・compiler/runtime・seed・科学call予算・有効one-shot承認に別々に結合する。本版の正式採用で旧grantを再使用したり、新しい有効grantを発行したりしない。
技術修正・中断復旧の既存承認と過去STOPを保全し、v2で勝手にretry契約を変更しない。
重い計算は許可scope内で候補/trajectory間の並列化、cache、重複削減等を行い、恣意的CPU/RAM/worker上限を設けない。共有設定・他processを変更せず、一時PSIや無関係worktreeの更新だけで正常workerを止めない。
AI subagentへの委任禁止は維持する。

> **履歴注記（§9・H2担当の判断と実行境界）**：以下のH2「提案n採用を認定しない」「n採否判断・P seal・実行承認が必要」は初回採用時点の履歴であり、完了済みPへの新たな承認要求ではない。最新状態は[§10](#h2-current-status)（P完了、n=80/12採用、2条件ともREPRODUCED）。一般の新規科学stageに対する上記の別承認境界は維持する。

**ユーザー（採用済み）**：固定v2改訂案を正式な共通実験プロトコルとして採用。H2の対象追加、保存snapshot、明示された例外・未検証事項を含む文書の採用であり、未検証事項の解消・科学的保証やH2 Pの提案n採用を認定しない。
**ユーザー／H2担当Codex（別stage）**：H2 S_STATS提案n80/12の採否を判断し、指示後にP最終seed・SOURCE・予算をsealする。P科学実行はさらに別個の明示承認を必要とする。本改訂担当はH2作業を起動しない。
**Codex（現在）**：正式版・採用記録の構造照合、保存、承認範囲の公開確認後、mandatory STOP。科学計算、詳細科学レビュー、最終freezeへ自動進行しない。

<a id="h2-current-status"></a>

## 10. H2独立確認P完了・正式受入（2026-10-19追記）

ユーザーの「実験条件プロトコルｖ２の方に追記しておいて」という指示により、H2の完了・正式受入をv2本文へ反映した。既に採択済みのv2への状態追記であり、プロトコル全体の再採択や新規科学実行の承認ではない。H2は旧v1対象外の追加系列とし、v1と初回v2採択記録は変更しない。

### 10.1 入力・実験条件

linear H2、H座標(0,0,0)/(0,0,1) Å、STO-3G、charge0/multiplicity1、RHF、SCF conv_tol1e-9/max_cycle50、symmetryなし。DF threshold1e-8・final_rank指定なしのtol-only分解は実rank3。signed weights・生成順・cutoff0を保持する。4 qubits、Nα=Nβ=1のspin sector次元4。旧H2の設定選択rank3/sector6とは別入力である。

状態はdense scipy.linalg.eigh、最大sector振幅の位相固定、norm gate1e-12、残差gate1e-10、最低gap gate1e-12、保存後再正規化なし。小さい保存残差や完了PASSをground-state certificateとはしない。H4 dense sector36、H6/H8の記録上eigsh sector400/4900とのsolver・正規化・残差gateの相違は保持し、全生成手順の完全同一性は未認定。

T=.8、P=0、通常signed bias、allowance倍率1。D/Aの5精度はε=.05/.01/.005/.001/.0005。full cosine/sine Hadamard wrapper、symmetric_directional、original scalar＋tail identity一回、reference不回転を継承する。Qiskit1.3.0、rz/sx/x/cx、opt1、seed_transpiler17、backend/coupling/layout/routing未指定。費用は両軸compiled RZ×保存analytic shotの合計であり、実量子shot数、FT T資源、energy/RPE全round費用ではない。

Hamiltonian `d0bc260db6fc0a485fe518050075c18ccc0dff532253dcc719a52a61f41afa80`、state `080b7ad2a46f38ed51e2989eb399ceda4cae95fca688722959ee65463cf174bf`、snapshot `62c6481bd1f8b05ec3b399b3a5293873796737866409049d808e8fe432dcacaa`。compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。

### 10.2 全stageの完了と来歴

|stage|科学SOURCE|完了commit|状態|
|---|---|---|---|
|F|`37b1cf72990885962d8449d8bd5ec7e303329350`|`7ea6df2dbe34e64a34f468c343bb12922d3e2829`|COMPLETE_MANDATORY_STOP|
|FP|`de72e5d4e48177734c0ddd79dffac14222410658`|`af724928689be3592b391fab3a14538bf4752b76`|COMPLETE_MANDATORY_STOP|
|D|`cf99e4452df96291d7e9d37d76d2e91be5cbc413`|`7b202a1a569c6839c4c7a5dfb18b6ff51ec3f3eb`|COMPLETE_MANDATORY_STOP|
|A|`de72e5d4e48177734c0ddd79dffac14222410658`|`1d2f67ec285956ef4090ea26520d64b11e925f9c`|COMPLETE_MANDATORY_STOP|
|S_STATS|`e0b959e8a617f7934d05c3e249832ccc1c43909f`|`829d697ae7905b3ea90a61da9d2d4da15d8b757a`|COMPLETE_MANDATORY_STOP|
|P|`ce1f511c4f7429190adcee58cde86be8a87a2fb1`|`f2d1a86c431f835db700f348ce618d8604e7ee23`|COMPLETE_MANDATORY_STOP|

Dは受入済みD_run02のD70に限定する。旧D_run01の未確認STOP・partial費用・非返還予約を成功へ書き換えない。A66は各新n8、FP診断各n1とPの独立cohortを混合しない。S_STATSは保存summaryによる24 scenario・12感度・2方向の既存設計であり、今回再実行していない。

PはS_STATS提案n80/12の採用、最終seal `6eacd7f82afb1c6d3fa6b66987df8dcfa1ed8bcb`、別途明示承認を経て完了した。科学SOURCE `ce1f511c4f7429190adcee58cde86be8a87a2fb1`、固定92 trajectory・184 compiled軸・256 RTE event、元のfinite-RTE law、cos/sin paired trajectory、追加sampling・自動retry・候補再選択なし。native exit0・同一親wait4・92 worker全回収を保存終了監査で確認済み。

### 10.3 正式採用した有限標本の2条件

|ε|固定PR / 固定PF|L_D / δ|n|保存G_PR exact|保存paired SE|保存G_PF exact|点費用・方向判定|
|---|---|---|---|---|---|---|---|
|0.05|B2_p1_q1_2nd_r2_K4 / B1_p3_q1_2nd|1 / .8|80|384135011/80|15992.018778605378|5533984|PR低費用、REPRODUCED|
|0.0005|B2_p1_q4_2nd_r2_K4 / B1_p3_q4_2nd|1 / .2|12|998482376201/4|276760628.92856276|305695076994|PR低費用、REPRODUCED|

ユーザーが両条件を**有限標本の結果として正式採用**した。P結果は[f2d1a86c](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2d1a86c431f835db700f348ce618d8604e7ee23/artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_completed_run01/confirmation_result.json)、受入記録は[8fbe7d77](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fbe7d77556c2353be837dbd9ed0fddc02ee81c4/artifacts/resource_applicability/track_a_h2_results_integration/2026-10-19/adoption_record.json)、全stageのSOURCE・契約・結果・終了監査は[固定証拠一覧](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fbe7d77556c2353be837dbd9ed0fddc02ee81c4/artifacts/resource_applicability/track_a_h2_results_integration/2026-10-19/stage_evidence.json)。mean/SE/PF/n/statusをそのまま転記し、新しい比・削減率・CI・検定・標本数・順位を計算していない。

q1の保存層はS0_ONLY79/S2_SINGLE1、q4はS0_ONLY12。希少・未観測高次/多重eventの費用・分散、law確率はunknownを維持する。S2を1件観測したことや保存SEから、母平均優位・実power・全support保証を認定しない。経験的bias allowanceも厳密bias保証ではない。

### 10.4 論文scopeと保全

H4/H6/H8の既存主表6条件と原図34ファイルは不変。H2追加2条件との[別8条件比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fbe7d77556c2353be837dbd9ed0fddc02ee81c4/artifacts/resource_applicability/track_a_h2_results_integration/2026-10-19/H2468_comparison_separate.md)は研究資料であり、論文主表の差替えではない。H2本文／補遺への掲載とpaper manifest・CEM・台帳追加案の適用は、引き続きユーザーの別判断とする。

H4粗精度NOT_REPRODUCED、H6粗精度REPRODUCEDかつPF低費用、H8のREPORTED_LOCAL_PASS_NOT_REVERIFIED_FROM_PUBLIC_RECEIPTSを保持する。H8主表とFigure2B/Mechanism H8_CONFIRM_1/3を重複した独立証拠と数えず、既存Mechanism因子をH2へ移送しない。

今回の更新は文書追記・manifest整合・構造検証のみ。科学SOURCE・raw結果・grant・ledger・marker・STOPを変更せず、新規科学計算・統計解析・sampling・compileは0。詳細科学レビュー・最終evidence freezeへ進まない。**追記完了後はmandatory STOP**。
