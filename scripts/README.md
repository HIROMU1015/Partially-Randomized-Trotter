# scripts 索引

## 2026-10-27：PR-aware DF B2・64 native kernel校正完了

[全gate・費用6指標・不確かさ・技術再実行](../docs/research/research_c_pr_aware_df_stage_b2_execution_20261027.md)／[修正契約](../docs/research/research_c_pr_aware_df_stage_b2_compat_contract_20261027.json)。保存H4・1Å/STO-3G、CDF8 Baseline/Proposed、元順LD2、partial S2 δ.1/s4/h1/40/r16/K0、own-DF oracle/状態準備除外、共通precision/P0。Run01は429 unitsでQiskit Already borrowed技術STOP（数値FAIL0）を保全し、共有build/compile/adjoint mutex・元64seedのfresh run02が652/652 COMPLETE／FAIL0、累積1081、技術再起動1・run02 retry/resume0。64 full512/adjoint/X-Yと67 compileがPASS。RZ標本mean±SEは5198.3125±66.7146対4765.875±80.3845（今回約8.3188%低い）。母平均保証・support・CA/shots/総資源は未資格。SOURCE50/input22/旧628不変、公開照合後mandatory STOP。

修正entryは`run_research_c_pr_aware_df_stage_b2_compat.py`、moduleは`c_pr_aware_df_stage_b2_compat.py`、testは同名management test。旧B2 runner/sourceは不変。

## 2026-10-27：PR-aware DF B2・native費用校正の技術準備

[共通native・full512・人工72 PASS・費用資格と制限](../docs/research/research_c_pr_aware_df_stage_b2_preparation_20261027.md)／[未承認652-call契約](../docs/research/research_c_pr_aware_df_stage_b2_contract_draft_20261027.json)。新`c_pr_aware_df_stage_b2.py`／専用runner／人工test。32独立trajectory/arm、64 kernel＋3shell、67 P0 compile、全operator/adjoint/X-Y後にwhole-kernel metricsを記録する案。SOURCE47/input22のbyte/header preflight PASS、旧175ファイル不変。実H4 decode/sampling/build-compile/cost・新科学call0、CI/support/ideal費用/CA/総資源は未資格。公開照合後mandatory STOP。

## 2026-10-27：PR-aware DF Stage Bの接続・人工検証準備

[2 CDF arm／law／sector reference／共通native接続・人工56 PASS](../docs/research/research_c_pr_aware_df_stage_b_readiness_20261027.md)／[未承認B1-only141-call案](../docs/research/research_c_pr_aware_df_stage_b_contract_draft_20261027.json)。新`c_pr_aware_df_stage_b.py`／`stage_b_runner.py`／同名runner・management test。保存Stage A J/Uの新評価は0、hash/header/JSON preflightのみPASS、旧97ファイル不変。B1は2arm×13roundのPF/RTE/A/signalをsector36で検証し、build/compile/sampling/shots/costは0の案。B2は人工native部品まで、実行範囲は別固定。科学call0、公開照合後mandatory STOP。

## 2026-10-27：PR-aware DFのx0接続を技術修正

[単一qualified x0引渡し・人工61 tests＋29 subtests](../docs/research/research_c_pr_aware_df_x0_repair_20261027.md)／[新V2・未承認30-call契約](../docs/research/research_c_pr_aware_df_pilot_contract_x0_v2_unapproved_20261027.json)。既存corrected-codecへopt-in prepared初期値を接続し、fit内の再logを除去。旧default・目的／gradient／optimizer条件は維持。実分子decode・新科学call0、旧18-call STOPを保全、新2026-10-27/run01未作成。別明示許可までmandatory STOP。

## 研究C：PR-aware DFの初期設計・技術準備（依頼識別子20261026）

[設計・再利用・人工49 tests＋29 subtests](<../docs/research/research_c_pr_aware_df_optimization_design_20261026.md>)／[未承認Stage A・30-call案](<../docs/research/research_c_pr_aware_df_pilot_contract_draft_20261026.json>)。`c_pr_aware_df.py`／`c_pr_aware_df_pilot.py`、`run_research_c_pr_aware_df.py`、同名management tests。既存ffsim loss/gradient・修正codec・SectorCARへ固定tail mass代理目的を最小接続。実optimizerは人工mock nit0、両env byte preflight PASS。実分子decode・新科学call0、科学的有効性／新規性・Stage B/Cは未資格・未承認、mandatory STOP。

## C1：H4 XDF9費用screen・両sでstrict改善を除外（2026-10-26）

[全gate・metrics・shots・比較・制約](../docs/research/research_c_c1_h4_xdf9_cost_screen_execution_20261026.md)／[raw48・QPY4・83-call journal・STOP・CSV](../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/)。固定SOURCE7328ecc9a…無変更、83/83 COMPLETE、correctness FAIL/inflight/retry/resume0。直鎖H4・1Å/STO-3G、Nα=Nβ2、元順XDF9、全frame S2 s1/s4、δ.1/m0..12、own-ground/preparation excluded、同precision/P0/shared/fenced serial。kernel RZ15720/各s。楽観的総RZ s1=144805110560/s4=530107748679はいずれも保存XDF8/CDF8より大きく、固定十分処方・実装・数値資格内でstrict改善を除外。full operator/actual合成誤差/CA/保証付きshots/最終費用は未資格・未評価。旧結果保全、公開照合後mandatory STOP。

## C1：H4 XDF9 native／楽観的shots screenの技術準備（2026-10-26）

[最小接続・単調性／ceil・限界・full512後続案](../docs/research/research_c_c1_h4_xdf9_cost_screen_preparation_20261026.md)／[未承認83-call契約](../docs/research/research_c_c1_h4_xdf9_cost_screen_contract_unapproved_20261026.json)／[SOURCE・人工／回帰97 PASS・passive証拠](../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/technical/)。新entry `run_research_c_c1_h4_xdf9_cost_screen.py`／module `c1_h4_xdf9_cost_screen.py`／同名management test。既存PrecisionBuilder/P0/solver/serial/rational楽観集計を再利用、XDF9 S2 s1/s4の2 kernelとPF26から固定処方の楽観的RZ下限を接続。実H4のdecode/build-compile/shots/cost・科学call0。下限は固定処方／serial実装内の数値screen、full operator／strict enclosure／actual CA・shots・資源優位ではない。SOURCE56/input24 byte preflight PASS、旧証拠不変、別明示許可までmandatory STOP。

## C1：H4 XDF9・表現／ground-alias／PF26がPASS（2026-10-26）

[全gate・全26 PF値・科学call・資格の限界](../docs/research/research_c_c1_h4_xdf9_pf_execution_20261026.md)／[raw45・SOURCE・journal・STOP・CSV](../artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/)。固定SOURCE9a4cda44a…無変更、95/95 COMPLETE、correctness FAIL/inflight/retry/resume0。直鎖H4・1Å・STO-3G、Nα=Nβ2/sector36、保存OpenFermion元順prefix9、元g補正κ・核scalar・frame順、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-XDF9 oracle/状態準備除外。保存DF予算PASSを再利用、表現10因子・sum・own-ground/alias PASS、両sとも13/13 PASSのPOTENTIALLY_FEASIBLE。最大PF誤差s1 .06843041535127227/s4 .004277093283650178。新fit/DF norm/native/合成/CA/shots/総費用0、27 protected filesと原packet不変。旧約27%／約12%比較を保全し、公開照合後mandatory STOP。以下は過去の準備・結果の履歴。

## C1：H2／旧H4／新H4／H6統合と次期XDF9準備（2026-10-26）

[保存16候補・baseline差分・SOURCE来歴](../docs/research/research_c_c1_h2_h4_h6_integration_update_20261026.md)／[A:H4 XDF9・B:H6 CDF13の比較と未採択推奨](../docs/research/research_c_c1_next_minimal_experiment_selection_20261026.md)／[人工78 tests＋29 subtests・技術準備](../docs/research/research_c_c1_h4_xdf9_preparation_20261026.md)。H4約27%→約12%をrank単独の因果効果とせず、3系をサイズscaling実験と扱わない。新entry `run_research_c_c1_h4_xdf9_pf.py`／module `c1_h4_xdf9_pf.py`／同名management testは、既存CAR／一般K／PFを再利用する[未承認95-call契約](../docs/research/research_c_c1_h4_xdf9_pf_contract_unapproved_20261026.json)の準備。original input／SOURCE24／保存JSON3のbyte preflight PASS、実分子decode／新科学call0。旧証拠保全、推奨は未採択、公開照合後mandatory STOP。後続native／CA／shots／費用は別範囲。 [CSV・来歴](../artifacts/research_c_c1_integration_update/2026-10-26/)／[SOURCE・人工log・passive証拠](../artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/technical/)。

## C1：H4保存CA/RPE/serial資源の固定119-call実行完了

[結果・全gate・shots/energy/総費用](../docs/research/research_c_c1_h4_ca_resources_execution_20261026.md)／[raw19・52round/104軸CSV・SOURCE/calls/STOP](../artifacts/research_c_c1_h4_ca_resources/2026-10-26/)。固定SOURCE18045333…無変更、119/119 COMPLETE、FAIL/inflight/retry/resume0。直鎖H4・1Å・STO-3G、CDF8/保存元順XDF8、Nα=Nβ2/sector36、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-DF oracle/状態準備除外。4候補の共通CA=.00159 Ha十分条件が数値PASS。CDFの条件付き総RZ削減は同sでs1 11.853846%／s4 11.990819%。保存level B・ideal IID Born・固定serial native modelのpilotで、厳密CA・無条件優位ではない。新fit/DF-PF-合成誤差再計算/build-compile/量子shots抽選0、旧証拠保全・mandatory STOP。以下は実行前準備などの履歴。

## C1：H4保存証拠のCA/RPE/serial費用（科学未承認）

[既存再利用・人工/回帰88 PASS・資格の限界](../docs/research/research_c_c1_h4_ca_resources_preparation_20261026.md)／[未承認119-call契約](../docs/research/research_c_c1_h4_ca_resources_contract_unapproved_20261026.json)。新`c1_h4_ca_resources.py`／同名runner・testが、保存52 full512 profile・各DF norm＋guard・共通precision/P0 metricsから既存RPE/Hoeffding/energy/serial費用を接続。直鎖H4・1Å・STO-3G、CDF8/保存順XDF8、Nα=Nβ2/sector36、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-DF oracle/状態準備除外。科学call0、SOURCE24/input12のbyte preflight PASS。新shots/CA/総費用は未評価、厳密enclosure・無条件優位は未資格。旧証拠保全・別明示科学許可までmandatory STOP。

## C1：H4 CDF8/XDF8・固定305-call実行完了

[結果・全gate・7 metrics・52 profile](../docs/research/research_c_c1_h4_native_fair_execution_20261026.md)／[SOURCE・QPY40・call journal・STOP](../artifacts/research_c_c1_h4_native_fair/2026-10-26/run01/)。SOURCE862a97d…を無変更実行し305/305 COMPLETE、correctness FAIL/retry/resume0。直鎖H4・1Å・STO-3G、CDF8/保存元順XDF8、Nα=Nβ2/sector36、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-DF oracle/状態準備除外。同H6 Gaussian/P0・既存shared routeで局所18/kernel4/shell3/adjoint8/X-YがPASS、full512数値Frobeniusによる52/52位相条件内。単一h-step RZはCDF12,272対XDF13,944（約11.99%減）。丸めenclosure・共通CA・shots・総資源は未資格/未評価。旧SOURCE/raw/STOPを保全しmandatory STOP。以下の技術準備欄は実行前時点の記録。

## C1：H4 CDF8/XDF8・共通precision/full512（科学未承認）

`run_research_c_c1_h4_native_fair.py`はdefaultでbyte/version preflightのみ。別科学許可後に保存CDF8/XDF8、共通H6 Gaussian/P0、4 kernel+3 shell・局所18・actual adjoint・full512 Frobenius・52 profile・native metricsへ接続する。[技術報告・156 unique PASS・fixture分離条件](../docs/research/research_c_c1_h4_native_fair_preparation_20261026.md)／[未承認305-call契約](../docs/research/research_c_c1_h4_native_fair_contract_unapproved_20261026.json)。新fit/DF-PF誤差再計算/CA/shots/総費用なし、今回新科学0・mandatory STOP。

## C1：H4 XDF8・固定SOURCEの限定科学実行完了

`run_research_c_c1_h4_xdf8_pf.py`をSOURCE328f31007…のまま一度実行。[結果・全26 PFセル・保全](../docs/research/research_c_c1_h4_xdf8_pf_execution_20261026.md)。93 calls COMPLETE、表現／own-ground・alias PASS、s1/s4ともPOTENTIALLY_FEASIBLE、correctness FAIL/retry/resume0。2026-10-26/run01は消費済み、再起動しない。DF norm／CDF8/XDF10／native／CA／shots／総費用の新評価なし・mandatory STOP。

## C1：H4 XDF8の表現・own-ground/alias・PF（科学未承認）

`run_research_c_c1_h4_xdf8_pf.py`はdefaultでbyte／環境preflightのみ。別明示科学許可後だけ`--execute`で保存順prefix8の表現とown-ground／H4 aliasを確認し、S2 s1/s4×13round＝26 PFセルを評価する。[再利用・人工22＋回帰92 PASS・次段階設計](../docs/research/research_c_c1_h4_xdf8_requalification_preparation_20261026.md)／[未承認93-call契約](../docs/research/research_c_c1_h4_xdf8_pf_contract_unapproved_20261026.json)。保存DF予算を再利用し、新fit／DF norm／CDF8-XDF10再評価／native／CA／shots／費用0。現在新科学0・mandatory STOP。

## C1：H4保存XDF prefix・科学実行完了

`run_research_c_c1_h4_xdf_prefix_budget.py`をSOURCE f7bc730d1…のまま一度実行し、356 calls／9prefixを完了。[結果](../docs/research/research_c_c1_h4_xdf_prefix_budget_execution_20261026.md)。保存順first-passing8、rank8/9 PASS、correctness FAIL/retry/resume0。2026-10-26/run01は消費済みで再起動しない。PF／native／shots／費用なし・mandatory STOP。

## C1：H4 XDF prefix1〜9・DF予算の技術準備

`run_research_c_c1_h4_xdf_prefix_budget.py`はdefaultでbyte／環境preflightのみ。明示科学許可後の`--execute`で元順prefix9件と保存rank10 anchorを評価する。[SOURCE再利用・57関連PASS・旧full256回帰制約](../docs/research/research_c_c1_h4_xdf_prefix_budget_preparation_20261026.md)／[未承認356-call契約](../docs/research/research_c_c1_h4_xdf_prefix_budget_contract_unapproved_20261026.json)。新fit／physical生成／rank10 norm／PF／回路／shots／費用0、今回新科学0・mandatory STOP。

## C1：H6保存値のCA/RPE/serial資源・限定実行完了

`run_research_c_c1_h6_ca_resources.py`をSOURCE55b3d868…で一度実行。119/119 calls、4候補NUMERICAL_PILOT_CA_PASS・52round PASS。同sのshotsは同じで、総RZはXDFが小さい。[結果](../docs/research/research_c_c1_h6_ca_resources_execution_20261026.md)。新fit/誤差再計算/state-action/build/compile/量子sampling0。run01は消費済み、retry/resume0・mandatory STOP。

## C1：H6保存値のCA/RPE/serial総資源・未承認

`run_research_c_c1_h6_ca_resources.py`：defaultはbyte/environment preflightのみ。[技術報告・102人工/回帰PASS](../docs/research/research_c_c1_h6_ca_resources_preparation_20261026.md)／[119-call契約](../docs/research/research_c_c1_h6_ca_resources_contract_unapproved_20261026.json)。保存JSONの4候補×13roundから新shots/energy/serial費用を評価する科学entryは、別明示許可後だけ使用。NPZ/QPY数値decode・新build/compile・DF/PF/合成再計算なし、retry/resume0。

## C1：H6全基底上界の限定実行完了

`run_research_c_c1_h6_synthesis_bound.py`をSOURCE e36e056…で1回実行。G4×800列、3335 calls、8経路×13roundが登録位相条件PASS。[結果](../docs/research/research_c_c1_h6_synthesis_bound_execution_20261026.md)。資格B／Cを区別しCA/shots/cost0。rawとSTOPを保全、消費済みrun01は再起動しない。

## C1：H6合成誤差上界・未承認

`run_research_c_c1_h6_synthesis_bound.py`：defaultはbyte/environment preflightのみ。保存compiled QPYとPF kernelsから全sector基底のfull-output作用差をstreamingする。[導出・134人工/回帰PASS](../docs/research/research_c_c1_h6_synthesis_bound_preparation_20261025.md)／[6535-call契約](../docs/research/research_c_c1_h6_synthesis_bound_contract_unapproved_20261025.json)。実行には別途明示許可が必要。新build/compile・DF/PF再計算・CA/shots/costなし、retry/resume0。

## C1：H6 stable-phase SOURCEの限定実行完了

`run_research_c_c1_h6_native.py`をSOURCE3d7b56b448…で1回実行し、298/298 calls・correctness全PASS。[結果・費用表](../docs/research/research_c_c1_h6_native_stable_phase_execution_20261025.md)。QPY52・104 UNRESOLVED profileを保持。旧2 native FAILと5-call診断を保全、累積627、run内retry/resume0。旧runner再起動・CA/shots/costへの自動進行は禁止、mandatory STOP。

## C1：H6 P0 phase精度修正版

`run_research_c_c1_h6_native.py`を[stable-phase契約](../docs/research/research_c_c1_h6_native_stable_phase_contract_20261025.json)で使用。旧runを再起動しない。`diagnose_research_c_c1_h6_native_state_action.py`は保存1経路・5-call独立診断で完了。[修正報告](../docs/research/research_c_c1_h6_native_stable_phase_repair_20261025.md)。

## C1：H6 native精度修正版

`run_research_c_c1_h6_native.py`は[新precision契約](../docs/research/research_c_c1_h6_native_precision_contract_20261025.json)を受け、H6専用PrecisionBuilderを使う。旧契約のrun01は再起動しない。[修正・185人工PASS](../docs/research/research_c_c1_h6_native_precision_repair_20261025.md)。

## C1：H6 native限定実行のcorrectness FAIL

`run_research_c_c1_h6_native.py`をSOURCE bf431765…で1回実行。162/298 calls、局所30／shell3 PASS、最初のCDF_s1_shared raw/referenceが固定閾値を超過。[部分結果](../docs/research/research_c_c1_h6_native_execution_20261025.md)。8 kernel bundle FAIL、adjoint/X-Y/kernel費用/104 profile未到達、QPY52・STOP保全。retry/resume0、旧runner再起動禁止・mandatory STOP。下記の未承認表示は準備時点の履歴。

## C1：H6 native finite-probe／kernel費用（科学未承認）

`run_research_c_c1_h6_native.py`：defaultはpassive preflight。30 local factors→3 shells→4候補×literal/sharedの8 kernel routes、actual adjoint/X-Y/P0 metricsへ接続。[報告](../docs/research/research_c_c1_h6_native_preparation_20261025.md)／[未承認298-call契約](../docs/research/research_c_c1_h6_native_contract_unapproved_20261025.json)。新fit/PF再計算/shots/総費用dispatchなし。

## C1：保存H6 rank14のsector PF実行結果

`run_research_c_c1_h6_saved_pf.py`をSOURCE87ea924ae…で1回実行。72/72 units、4候補・52roundが登録十分条件PASS。[報告](../docs/research/research_c_c1_h6_saved_pf_execution_20261025.md)。fit/optimizer/retry/resume0、旧runは再起動せずfresh run01を保全、native/shots/costへ進まずmandatory STOP。

## C1：保存H6 rank14のsector PF（科学未承認）

`run_research_c_c1_h6_saved_pf.py`：defaultはpassive preflight。保存CDF14／XDF14、一般K同値性とground/aliasの全PASS後、決定論S2 s1/s4×13roundを評価する専用entry。[技術報告](../docs/research/research_c_c1_h6_saved_pf_preparation_20261025.md)／[未承認72-call契約](../docs/research/research_c_c1_h6_saved_pf_contract_unapproved_20261025.json)。fit・rank探索・回路・shots・費用dispatchなし。

## C1：H6 rank14修正codec runnerの限定実行結果

`run_research_c_c1_h6_corrected_fit.py`をSOURCE33432e786…で1回実行し20/20 units・optimizer成功／DF予算PASS。[結果](../docs/research/research_c_c1_h6_corrected_fit_execution_20261025.md)。fresh run01とSTOPを保全、retry/resume0。新fit・後続評価は自動実行しない。

## C1：H6修正codecのrank14 1 fit（科学未承認）

`run_research_c_c1_h6_corrected_fit.py`：defaultはpassive byte preflight。保存XDF14→修正codec→sector400初期gate→最大1 ffsim fit→終了sector／予算の専用経路。[技術報告](../docs/research/research_c_c1_h6_corrected_fit_preparation_20261025.md)／[未承認20-call契約](../docs/research/research_c_c1_h6_corrected_fit_contract_unapproved_20261025.json)。新科学は別途許可後のみ。

## C1：修正codecの保存H6 round-trip

`run_research_c_c1_h6_codec_roundtrip.py`は既定byte preflightのみ。将来、保存rank14/18の初期J/Uを再分解せず、修正codec→4sector状態→同値性・DF予算保持だけを最大20callsで確認する。[未承認契約](../docs/research/research_c_c1_h6_codec_roundtrip_contract_unapproved_20261025.json)／[人工結果](../docs/research/research_c_c1_h6_codec_roundtrip_preparation_20261025.md)。科学許可後のみ実行し、fit／PF／native／shots／費用は対象外。

## C1：保存H6初期化診断

`run_research_c_c1_h6_initialization.py`は保存H6のrank14／18、変換前／直後／保存CDF終了後だけを評価する専用entry。別fit/eval環境、既存ffsim codec・SectorCAR・rank_assessmentを再利用し、optimizer／PF／nativeを呼ばない。[固定35-call条件](../docs/research/research_c_c1_h6_initialization_contract_20261025.json)／[実行結果](../docs/research/research_c_c1_h6_initialization_execution_20261025.md)。今回の許可は消化済み、fresh再実行には別の範囲・許可が必要。

`run_research_c_c1_h2_h6.py --molecule 2|6 --contract <contract>` は[H2・H6準備](../docs/research/research_c_c1_h2_h6_preparation_20261025.md)。既定はmetadata／byte preflightだけ。新科学条件を採択して別途許可後だけ `--execute --fit-python <isolated-fit-python> --output <fresh>` を使う。H2上限196 unitsはrank→native→数値CA/shots/cost、H6上限129はsector400 DF/PFまで。1コマンド1分子、retry/resume0、H6 nativeや次分子へ自動進行しない。[未承認案](../docs/research/research_c_c1_h2_h6_contract_unapproved_20261025.json)。人工14＋既存166 tests・29 subtests PASS、今回実科学0・STOP。

`run_research_c_c1_stage3_deterministic_accuracy.py` は[決定論4候補の精度準備](../docs/research/research_c_c1_stage3_deterministic_accuracy_preparation_20261025.md)。既定はbyte preflight。別科学許可後にだけ `--execute --output <fresh>` で保存PF・DF・shots・費用、s1保存Stage2残差／s4局所reference2／共通shellを接続する。[未承認契約](../docs/research/research_c_c1_stage3_deterministic_accuracy_contract_unapproved_20261025.json)最大110 units、fit／optimizer／retry／resume／新build・compile／shot-count solving0。厳密CA未資格、数値pilot PASSだけ費用へ接続。人工12＋独立process回帰140 PASS／29 subtests、実H4科学call0・STOP。

`run_research_c_c1_stage3_cost_compat.py` は[固定Qiskitのcompile競合修正](../docs/research/research_c_c1_stage3_cost_compat_repair_20261025.md)。旧P0を変更せずcompile callbackだけ排他化、他candidate workは旧並列計画。既定はbyte preflight、今回の修正再実行許可に従って`--execute --output <fresh>`で同8候補cost-onlyを実行する。[936-call契約](../docs/research/research_c_c1_stage3_cost_compat_contract_20261025.json)。旧run/STOPは再利用せず、人工100 PASS。

`run_research_c_c1_stage3_cost_only.py --mode cost-only` は[8候補native費用準備](../docs/research/research_c_c1_stage3_cost_preparation_20261025.md)。既定はbyte preflightのみ。別明示科学許可後だけ`--execute --output <fresh>`でD s1/s4・PR s4/r16/K0/K2各armの費用を測り、保存13round X/Y楽観Nで条件付き総RZ等を積算する。[未承認契約](../docs/research/research_c_c1_stage3_cost_contract_unapproved_20261025.json)最大936 units、PR各32校正、fit／retry／resume0。PF/RTE／shots再算定なし。support未資格は点推定のみ、CA／実shots保証へ昇格しない。人工94 PASS、今回実H4科学call0・STOP。

`run_research_c_c1_stage3_shots_screen.py` は[楽観的shotsスクリーニング](../docs/research/research_c_c1_stage3_shots_preparation_20261025.md)。既定はbyte preflightのみ。別明示許可後だけ`--execute --output <fresh>`で保存20候補×13roundのPF/RTE/Aから仮定付きX/Y算定値・合計・減衰診断を作る。[未承認契約](../docs/research/research_c_c1_stage3_shots_contract_unapproved_20261025.json)は最大282 logical units、fit／retry／resume0。未知誤差は実際の0と認定せず、CA・回路・総費用・資源順位なし。人工81 PASS、今回再解析0・STOP。

`run_research_c_c1_stage3_feasibility.py --mode feasibility-only` は[C1限定数値経路](../docs/research/research_c_c1_stage3_feasibility_preparation_20261025.md)。既定はbyte preflightだけ。別明示科学許可後だけ`--execute --output <fresh>`で260セルのPF/RTE誤差・減衰と20候補の十分条件分類を保存する。full modeへのdispatchなし、CA達成／shots／費用／順位は評価しない。[639-call未承認契約](../docs/research/research_c_c1_stage3_feasibility_contract_unapproved_20261025.json)、SOURCE・人工結果 `artifacts/research_c_c1_stage3_feasibility_technical/2026-10-25/`。今回科学call0、STOP。


`run_research_c_c1_stage3_integrated.py` は[C1 Stage3統合runner](../docs/research/research_c_c1_stage3_integrated_preparation_20261025.md)。既定はbyte preflightのみ、別途チャットでの科学許可後だけ`--execute --output <fresh>`を使う。保存入力からevidence生成・shots／費用比較までの一体経路で、旧packet評価SOURCEは無変更。[未承認契約](../docs/research/research_c_c1_stage3_integrated_contract_unapproved_20261025.json)は固定20候補／13round・最大5,637 logical units・fit／retry／resume0。人工19＋既存65 tests／29 subtests PASS、今回実H4科学call0、全law合成誤差未資格・mandatory STOP。


`run_research_c_c1_stage3.py` は[C1 Stage3技術準備](../docs/research/research_c_c1_stage3_preparation_20261025.md)のSOURCE／保存入力byte確認を既定とする。科学条件を採択して別途チャットで許可した後だけ`--evaluate-evidence <packet> --output <fresh.json>`で精度／shot／費用を評価する。分子evidence生成は既存loader／native／RTEと新しい最小接続関数を組み合わせる。[未承認契約](../docs/research/research_c_c1_stage3_contract_unapproved_20261025.json)、人工test `tests/management/test_research_c_c1_stage3.py`、証拠 `artifacts/research_c_c1_stage3_technical/2026-10-25/`。今回科学call0・mandatory STOP。


`run_research_c_c1_stage2_native.py` の[C1 Stage2限定実行](../docs/research/research_c_c1_stage2_native_execution_20261025.md)は採択されたT1/10・S2・P0 profileで全60主検査＋unitarity52件PASS。SOURCE／runner無変更、call243/243、fit／retry／resume0、68 QPY・12 caseのnative metric記録。原本 `artifacts/research_c_c1_stage2_science/2026-10-25/`。Stage3未承認・mandatory STOP。

`run_research_c_c1_stage2_native.py` は[C1 Stage2接続](../docs/research/research_c_c1_stage2_native_preparation_20261025.md)のbyte preflightを既定とする。実行前にチャットで条件採択・別明示科学許可を受け、`--execute --output <fresh>` を使用する。approval記録や証明照合は不要。処理 `trottertracks.c1_stage2_native/validation/execution`、test `tests/management/test_research_c_c1_stage2_native.py`、SOURCE／52 pins／人工証拠 `artifacts/research_c_c1_stage2_technical/2026-10-25/`、[未承認契約](../docs/research/research_c_c1_stage2_native_contract_unapproved_20261025.json)。人工80件＋29 subtests PASS、今回実回路build/compile・科学call0、mandatory STOP。

`run_research_c_c1_stage1_eval_only.py` の[保存fit評価専用科学実行](../docs/research/research_c_c1_stage1_eval_only_execution_20261025.md)は、公開SOURCE `bec41ac7dc7e034bd664817acfdcfed858ddadd0` を無変更で使用し37 gate PASS。旧fit来歴を独立に保持、新fit0・call141/141・retry/resume0。結果 `artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/`、直前の人工19件PASS。Stage2適格性は別判断、旧FAIL／STOP保全・mandatory STOP。以下の準備段階の未承認表示は履歴。

`run_research_c_c1_stage1_eval_only.py` は[C1評価専用入口](../docs/research/research_c_c1_stage1_eval_only_preparation_20261025.md)。既定はbyte/JSON preflight、実行は新scopeの別科学承認が必要。`--worker-fit`／fit Python引数はなく、旧one-fit runnerを再起動しない。処理 `trottertracks.c1_stage1_eval_only`、人工test `test_research_c_c1_stage1_eval_only.py`、SOURCE／証拠 `artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/`、[未承認契約](../docs/research/research_c_c1_stage1_eval_only_contract_unapproved_20261025.json)。人工82件＋29 subtests PASS、新fit0／最大141、H4未評価・mandatory STOP。

[C1 raw frame修正](../docs/research/research_c_c1_frame_repair_20261025.md)の人工監査は `artifacts/research_c_c1_frame_repair/2026-10-25/artificial_parallel_audit.py` と `preloaded_backend_audit.py`。新test `tests/management/test_research_c_c1_frame_repair.py`、修正 `trottertracks.c1_stage1.raw_frame`、SOURCE／証拠 `artifacts/research_c_c1_frame_repair/2026-10-25/`。人工63件＋29 subtests PASS、分子科学処理0。`run_research_c_c1_stage1.py` はone-fit入口のままで、保存exportの評価専用runnerではない。別準備・科学承認待ち、mandatory STOP。

`run_research_c_c1_stage1.py` の[互換性修正版Stage 1科学実行](../docs/research/research_c_c1_stage1_compat_execution_20261025.md)は新approvalで1 fit収束後、frame0 raw CAR／K同値性FAILによりSTOP。科学call107/147、retry/resume0、証拠 `artifacts/research_c_c1_stage1_compat_science/2026-10-25/`。SOURCEと既存runnerは変更せず、物理sector norm未到達、追加実行・Stage2/3なし。

`run_research_c_c1_stage1.py` は[ffsim／SciPy互換性修正版](../docs/research/research_c_c1_stage1_compatibility_20261025.md)でvenv Python symlinkを保持し、`trottertracks.c1_ffsim_compat` 経由の初期化へ接続する。人工tests `test_research_c_c1_ffsim_compat.py` を含む59件＋29 subtests PASS、証拠・SOURCE／lock `artifacts/research_c_c1_stage1_compatibility/2026-10-25/`。[新未承認契約](../docs/research/research_c_c1_stage1_compatibility_contract_unapproved_20261025.json)は新scope、fresh outputへの別明示承認が必要。科学実行0、mandatory STOP。

`run_research_c_c1_stage1.py` の[限定科学実行・技術的STOP](../docs/research/research_c_c1_stage1_execution_20261025.md)：固定SOURCE `de76aee72`、1 fitはSciPy1.14.1 batch logm非互換でoptimizer前に停止。call2/147、科学retry/resume0。artifact `artifacts/research_c_c1_stage1_science/2026-10-25/`。SOURCEを修正せず証拠保存・mandatory STOP。

`run_research_c_c1_stage1.py` は[C1 Stage 1技術準備](../docs/research/research_c_c1_stage1_technical_preparation_20261025.md)の限定入口。既定は数値decodeなしpreflight、科学runは別の明示承認が必要。処理 `trottertracks.c1_stage1*`、人工test `tests/management/test_research_c_c1_stage1*.py`、SOURCE/lock/証拠 `artifacts/research_c_c1_stage1_technical/2026-10-24/`。今回は人工検証のみ、mandatory STOP。

`scripts/` はコマンドラインから実行する入口を置く。研究ロジックの本体は原則として
`src/trotterlib/` にあり、runnerは条件の読込み、呼出し、成果物保存を担当する。

最新の研究段階は[`../docs/research/研究概要・現状.md`](../docs/research/研究概要・現状.md)、
各runnerと成果物の対応は[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)を
参照する。下記の「現行」は、最終科学結論を意味せず、現在の近似検証で使う経路を表す。

## 現行の検証runner

### PF誤差係数

| runner | 用途 |
|---|---|
| `run_pf_delta_validation.py` | 小規模系でPF係数の定義・delta依存・参照値を比較 |
| `run_pf_c_system_size_validation.py` | H-chainの系サイズ方向とstate-action経路を検証 |
| `run_df_ground_state.py` | DF Hamiltonianの基底状態snapshotを作成 |
| `benchmark_df_ground_state.py` | 基底状態計算経路の計測 |
| `validate_pauli_partial_cgs_all_prefixes.py` | Pauli partial回路係数のprefix検査 |

### finite RTE

| runner | 用途 |
|---|---|
| `run_finite_rte_signal_validation.py` | 演算子誤差、状態上の複素期待値、半径、位相上界を検証 |

### コンパイル後回路コスト

| runner | 用途 |
|---|---|
| `run_random_circuit_cost_validation.py` | イベント単純加算と一体コンパイルを比較 |
| `run_rte_boundary_cost_validation.py` | 2・3イベントの境界補正を検証 |
| `run_rte_boundary_cost_replication.py` | 境界分類を別乱数seedで再検証 |
| `run_hierarchical_cost_validation.py` | 局所補正、長さ、制御付き反復をまとめて検証 |
| `run_rte_order_stratified_cost_validation.py` | 低確率Taylor次数を条件付き抽出して検証 |
| `run_rte_connected_cluster_cost_validation.py` | 1--3イベント境界補正の運用推定を検証 |
| `run_rte_connected_cluster_transfer_validation.py` | $L_D$・短時間幅が異なる条件へ移した場合を検証 |
| `run_rte_connected_cluster_holdout_supplement.py` | 独立検証の標本を補充 |
| `run_rte_connected_cluster_k4_calibration.py` | 不通過条件の4イベント係数を較正 |
| `run_rte_connected_cluster_k4_extrapolation_diagnostic.py` | 4イベント補正の外挿診断 |
| `run_rte_paired_k4_l8_validation.py` | 同一イベント列で4イベント補正の構造残差を検証 |
| `run_controlled_repetition_holdout.py` | 制御付き反復回路の反復数依存を検証 |
| `run_rte_cost_angle_invariance_validation.py` | 回転角をまたぐmetric cache再利用可否を検証 |
| `run_h5_system_size_cost_validation.py` | H5の対応あり構造検証 |
| `run_h5_independent_cost_validation_batch.py` | H5の独立係数較正・未使用回路検証 |

### RPE信号・測定回数・1 shotコストへの接続

| runner | 用途 |
|---|---|
| `run_rpe_round_cost_connection_validation.py` | 短いRPE段で信号半径、shot数、回路costを接続 |
| `run_rpe_hadamard_failure_validation.py` | 仮想Hadamard測定と失敗確率を検証 |
| `run_rpe_hadamard_proxy_resource_validation.py` | $q=8$の1 shot cost proxyと資源集計を検証 |
| `run_rpe_allocation_sensitivity_validation.py` | 位相誤差・失敗確率配分の感度を比較 |
| `run_rpe_four_round_accounting_validation.py` | $q=1,2,4,8$の限定4段を集計 |
| `run_rpe_four_round_phase_validation.py` | $q=8$物理信号と4段の分枝付き位相復元を検証 |
| `run_rpe_target_round_horizon_validation.py` | 目標精度から必要round数を決め、固定設定の長$q$可否を行列診断 |
| `run_rpe_delta_round_schedule_validation.py` | 実行済み$\delta$をPFでscreeningし、round別$(r_m,K_m)$ scheduleを行列検証 |
| `run_rpe_delta_compiled_cost_validation.py` | 検証済み局所境界係数をround scheduleへ接続し、中央RTEブロックのcompiled-cost proxyを比較 |

## 長時間・複数条件のbatch runner

次のrunnerは、上記の検証を複数jobで実行・再開するためのもの。単独の数値だけで結論を
判断せず、対応する検証文書とmanifestを読む。

- `run_parallel_validation_batch.py`：共有サーバー向けtask manifest生成、dry-run、bounded実行、resume、status
- `run_rte_cost_data_batch.py`
- `run_rte_cost_followup_batch.py`
- `run_df_anchor_refinement_pipeline.py`
- `run_df_ld_window_cgs.py`
- `run_df_screening_anchor_cgs_h3_h14.py`
- `run_h5_df_cgs_gpu_slopes.py`
- `run_h9_ld5_df_cgs_ground_cache_slope_check.py`

## 診断・変換・可視化

| script | 用途 |
|---|---|
| `check_validation_manifest.py` | manifestのschemaと参照パスを検査 |
| `check_partial_randomized_diagnostics.py` | partial-randomized計算の診断値を確認 |
| `analyze_df_ld_direction_trend.py` | $L_D$方向の傾向を解析 |
| `compare_df_error_budget_rules.py` | 誤差予算規則を比較 |
| `compare_df_error_budget_rules_from_summaries.py` | 保存済み要約から誤差予算規則を比較 |
| `cache_grouping_step_costs.py` | grouping step costをキャッシュ |
| `export_df_cgs_cost_table.py` | CGS cost表を書き出す |
| `plot_df_optimized_costs.py` | DF最適化costを可視化 |
| `plot_df_partial_grouping_cost_comparison.py` | partial grouping間のcostを比較 |
| `profile_h5_8th_cgs.py` | 旧高次PF経路のH5 profiling |

## 旧経路・現在の結論に直接使わないrunner

次は履歴・比較・再現のため残している。出力を現行結果として使う前に
[`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)を確認する。

- `run_partial_randomized_pf.py`
- `run_df_screening_cost_minimization.py`
- `run_grouped_uwc_pf_qpe.py`
- `run_grouped_uwc_theta_sweep.py`
- `run_uwc_grouped_alpha_h2_h6.py`

特に旧DF screeningの入力表は基底状態不一致のため失効しており、UWC数値は参照raw artifactが
未登録である。runnerが存在することと、研究結果として利用可能であることを混同しない。

## runnerを追加するとき

新しい検証runnerは、可能なら次の同名セットを用意する。

```text
src/trotterlib/<validation_name>.py
scripts/run_<validation_name>.py
tests/test_<validation_name>.py
docs/<validation_name>.md
artifacts/<validation_name>/
```

結果を研究上の証拠へ加える場合は、`VALIDATION_STATUS.md`、研究概要、研究ノート、
`artifacts/validation_manifest.json`も更新する。


## PR混在辞書 F1保存入力経路（準備のみ）

`run_pr_dictionary_f1_saved.py` は旧 `run_pr_dictionary_f1.py` の認可gateを再利用する将来F1専用入口。新規の明示承認・固定packetが必要で、F1-PREでは実行しない。SOURCE、管理test、planned artifactは[準備記録](../docs/research/pr_algorithm_improvement_phase2d_f1_preparation_20261023/README.md)を参照。


## PR混在辞書F1の実行結果（2026-10-24）

`run_pr_dictionary_f1_saved.py` の承認済みone-shotは最初のcompile requestで技術的停止。SOURCE修正や再実行は行っていない。[実行結果](../docs/research/pr_algorithm_improvement_phase2d_f1_execution_20261024/README.md)と `artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24/` に完了／未完了、残差、provenanceを記録した。


## PR混在辞書F1：修正版の登録検証完了（2026-10-24）

[新結果とSTOP](../docs/research/pr_algorithm_improvement_phase2d_f1_rerun_20261024/README.md)。既存saved runner／adapter／reference／pricingを再利用し、SOURCE `8192530` と保存B入力で16familyを全件完了。全4armの登録正しさ検査とcatalog価格-DP照合はPASS、未資格intervalはfallbackのみ。認証score／shot／資源優位性は未評価。artifactは `artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24-r1/`。旧停止記録は保全し、F2へ進まない。


## PR混在辞書 F2実行前準備（2026-10-24）

[準備・SOURCE再利用・case・統計資格・契約・STOP](../docs/research/pr_algorithm_improvement_phase2d_f2_preparation_20261024/README.md)。入口 `scripts/run_pr_dictionary_f2_saved.py`、SOURCE `trottertracks/pr_dictionary_validation/f2_*.py`、管理test `tests/management/test_pr_dictionary_f2_preparation.py`。planned outputは `artifacts/pr_algorithm_improvement_phase2d_f2/<execution-date>/<fresh-run-id>/`。finite-target主比較を採択し、mock/管理確認のみ完了。統計条件とinterval資格は未解決で、科学実行・新grantは0件。


## PR混在辞書 F2統計資格確認（2026-10-24）

[資格確認と実行前STOP](../docs/research/pr_algorithm_improvement_phase2d_f2_qualification_20261024/README.md)。既存interval／log／ceilの技術test `tests/management/test_pr_dictionary_f2_scalar_intervals.py` は5件PASS。科学SOURCEはF2-PREのまま、共通finite targetに対する有限ビット誤差予算γの保証と統計条件の採択が未解決で、F2科学実行・新grantは0件。機械可読記録は同資料の `qualification.json`。入口は既存 `scripts/run_pr_dictionary_f2_saved.py` を維持する。


## PR混在辞書 F2限定資源計測（2026-10-24）

[限定契約・再利用SOURCE・停止条件](../docs/research/pr_algorithm_improvement_phase2d_f2_resources_20261024/README.md)。入口 `scripts/run_pr_dictionary_f2_resources_saved.py`、接続 `trottertracks/pr_dictionary_validation/f2_resources.py`、管理test `tests/management/test_pr_dictionary_f2_resources.py`。出力は `artifacts/pr_algorithm_improvement_phase2d_f2_resources/`。旧統計資格付き契約を維持し、認証Nを生成しない。

限定F2の[実行結果・部分資源・STOP](../docs/research/pr_algorithm_improvement_phase2d_f2_resources_20261024/EXECUTION_REPORT.md)：raw全件PASS、compiled operator3件FAIL、mean／4arm期待資源は未完了。SOURCE `9b1e103` と一回の実行原本を保全し、retryなし。


## 限定F2・位相保持修正と新規実行（2026-10-24）

[修正・実行契約・停止条件](../docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/README.md)。既存runnerとf2_resources／f2_executionを再利用し、gates.compiled_wrapperの公式definition展開を修正。回帰testは `tests/management/test_pr_dictionary_compile_phase.py`、planned artifactは `artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/`。

[修正版F2の全件結果・資源表・STOP](../docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md)：SOURCE `9468c70`、raw／compiled各13,676・reference27,356・mean4arm・weighted集計を完了。1-shotの指標順はPP<SP<PS<SS、認証Nは対象外。旧失敗は保持し、追加実験へ進まない。

`run_pr_dictionary_f2_attribution_saved.py` は[保存F2寄与分析](../docs/research/pr_algorithm_improvement_phase2d_f2_attribution_20261024.md)の入口。既存CSV・law・B atomまたは公開派生featuresだけを使い、回路・lawを生成しない。処理は `trottertracks.pr_dictionary_validation.f2_attribution`、人工testは `tests/management/test_pr_dictionary_f2_attribution.py`、artifactは `artifacts/pr_algorithm_improvement_phase2d_f2_attribution/2026-10-24/`。

## Rectangular Givens PS48の実行前準備（2026-10-24）

`run_pr_rectangular_givens_saved.py` は[候補Aの限定実装](../docs/research/pr_rectangular_givens_preparation_20261024.md)の入口。`--check-only` は保存hash・case identityだけを確認する。`--execute` は別のPS48科学承認・新one-shotを要求する。処理は `trottertracks.pr_dictionary_validation.rectangular*`、mock testは `tests/management/test_pr_rectangular_givens_preparation.py`、planと技術証拠は `artifacts/pr_rectangular_givens_preparation/2026-10-24/`。今回は科学実行なし。

PS48の[科学実行結果](../docs/research/pr_rectangular_givens_ps48_execution_20261024.md)は固定runner `run_pr_rectangular_givens_saved.py` で一回取得した。SOURCE変更・retryなし。保存結果だけの集計utilityは `artifacts/pr_rectangular_givens/2026-10-24/summarize_saved_ps48.py`、CSV／summaryも同所。これは新build／compileや全F2への展開の許可を意味しない。


## 研究C・C4：Q13限定科学検証（2026-10-24）

[実行契約・履歴・結果](../docs/research/research_c_rustiq_execution_20261024.md)。SOURCEは `src/trottertracks/pr_dictionary_validation/rustiq_adapter.py` と `rustiq_execution.py`、入口は `scripts/run_research_c_rustiq_limited.py`、管理testは `tests/management/test_research_c_rustiq_*.py`、artifactは `artifacts/research_c_rustiq_limited/2026-10-24/`。保存F1の4case・B0/B1/R1だけを対象とし、旧SOURCE・grant・STOPを保持する。


## 研究C・C4：B2角度結合baseline限定科学比較（2026-10-24）

[結果・固定SOURCE・契約・証拠](../docs/research/research_c_angle_fusion_b2_execution_20261024.md)。既存 `rustiq_adapter.py`／`rustiq_execution.py` と `scripts/run_research_c_rustiq_limited.py --scope C4_B2_SAVED_COMPARISON_V1` を使用。testは `tests/management/test_research_c_rustiq_*.py`、artifactは `artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/`。全登録gateがPASSし、全8wrapper・全8compiled指標でB2=保存R1。単純融合がこのfixtureの改善を再現した。旧route再実行・新reference・Rustiq/R2・retryは0。独立研究を区切るかはユーザー／研究C／統括の次判断であり未採択、mandatory STOP。

## H4強baseline：basis／relative Gaussian（2026-10-26）

`run_research_c_c1_h4_strong_baseline.py` はデフォルトでbyte preflightだけを行う。[技術報告](../docs/research/research_c_c1_h4_strong_baseline_preparation_20261026.md)／[未承認227-call案](../docs/research/research_c_c1_h4_strong_baseline_science_contract_draft_20261026.json)。別途許可された場合だけ`--execute --output <fresh run01>`でB1 XDF spin／B2 CDF・XDF relativeのs1三routeを評価する。旧one-shot runnerは再起動しない。科学実行は未実施。

## 2026-10-26：H4強baseline s1科学実行完了

[227-call結果・full512検証・5比較](../docs/research/research_c_c1_h4_strong_baseline_execution_20261026.md)。9 local／3 kernel／6 adjoint／9 complex X/Y gate PASS。B1 XDF RZ11821、B2 CDF11062/XDF11307、保存B0は再buildなし。単一kernelの数値pilotで、B3・s4・CA/shots/総費用は未評価。旧結果・SOURCE・STOPを保全し、公開照合後mandatory STOP。

## H4 B3：spin＋relative（2026-10-26）

`run_research_c_c1_h4_b3.py` は[B3技術報告](../docs/research/research_c_c1_h4_b3_preparation_20261026.md)の専用entry。デフォルトはbyte preflightだけで、別明示許可後のみ`--execute --output <fresh run01>`を用いる。[未承認65-call契約案](../docs/research/research_c_c1_h4_b3_science_contract_draft_20261026.json)はB3 XDF8／s1/h.1の1 kernel・1 compileだけ。旧B0/B1/B2は再実行せず、保存値との3比較を準備。今回科学実行なし。

## 2026-10-26：H4 B3 s1・65-call限定実行完了

[全結果・full512 gate・6指標・3比較](../docs/research/research_c_c1_h4_b3_execution_20261026.md)。固定SOURCE b40fce5c…無変更、65/65 COMPLETE、correctness FAIL/retry/resume0。H4直鎖1Å/STO-3G、元順XDF8/保存CDF8、Nα=Nβ2、全frame S2 s1/h.1、PRなし、own-ground/状態準備除外、共通precision/P0。新B3 spin＋relative一件のみbuild/compileし、raw/P0/actual adjoint/X-Yは全512入力・出力で数値PASS。RZ10885/CX6892/totaldepth11973は保存B2 CDF11062/7008/12127より小さく、他3指標も小さい。単一kernel比較で、同CA総順位・13round・shots・費用・厳密enclosureは未評価／未資格。旧raw・SOURCE・STOP保全、mandatory STOP。上の準備・未実行表記は履歴。

## 2026-10-26：H4 B2 CDF／B3 XDFのCA・RPE・総資源の技術準備

[再利用・接続・人工76 PASS・資格の限界](../docs/research/research_c_c1_h4_strong_baseline_ca_resources_preparation_20261026.md)／[未承認86-call契約](../docs/research/research_c_c1_h4_strong_baseline_ca_resources_contract_draft_20261026.json)。新module `c1_h4_strong_ca_resources.py`／同名runner・testsが保存full512 segment・norm guard・shell・元PF13を既存propagate/RPE/energy/serial集計へ接続する。H4直鎖1Å/STO-3G、CDF8/XDF8、Nα=Nβ2、全frame S2 s1のみ・δ.1/m0..12、PRなし、own-ground/状態準備除外、共通precision/P0/CA.00159。回路／Operator再生成不要、実科学call0。旧SOURCE/raw/STOP・旧約12%総資源とB3単一kernel結果を保全し、新総順位・厳密enclosureは未評価／未資格。公開照合後mandatory STOP。

## 2026-10-26：H4強baseline CA・実行前binding修正

[原因・人工78 PASS・修正契約v2・実科学call0](../docs/research/research_c_c1_h4_strong_ca_binding_repair_20261026.md)。旧86-call契約の保存contract参照が、同一bytesのdocs pathとartifact pathの文字列差でSTOPするため、canonical参照だけを修正した。科学SOURCE31件・数値条件・call上限86・旧契約/raw/STOP不変。run01未作成・未消費、科学実行は未開始。修正契約での別明示許可までmandatory STOP。

## 2026-10-26：H4強baselineの共通CA・serial総資源比較完了

[全結果・26round・shots・総費用・資格の限界](../docs/research/research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md)。固定SOURCE1aeedd961…／binding-v2無変更、86/86 COMPLETE、correctness FAIL/retry/resume0。H4直鎖1Å/STO-3G、B2 CDF8/B3 XDF8、Nα=Nβ2、全frame S2 s1のみ・δ.1/m0..12、PRなし、own-ground/ideal IID Born/状態準備除外、共通precision/P0/CA.00159。両候補のCA数値十分条件PASS、shots13606/13600、条件付き総RZ102056444569/100267421030でXDFが1.75297%少ない。level B数値pilotの限定比較で、厳密enclosure・無条件／一般の優位は未資格。旧結果保全、公開照合後mandatory STOP。上の準備・未実行表示は履歴。
