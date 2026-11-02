# H8限定feasibility：8ケース・20回路の完了結果

2026-10-13 JST。SOURCE `9bfea8bde44cbb2ea9187476ddd496066a8853b3`。既存の保存JSON・native terminal・SHAだけを集計し、新しいtrajectory/compile/signal/入力生成は実行していない。

**8/8ケース、20/20 full-wrapper compileが正常完了。** 最初のheavyは単独worker、残7は実測最大3workerで実行。heavy coordinator/remaining coordinatorのnative exitは両方0、全8worker exit0/wait/reap済み、cleanup error0・own-run残存0。自動retry・本格H8 resource-mapは実行していない。

linear H8/1Å/STO-3G/tol-only DF1e-8・cutoff0・signed generation order、actual rank28/16modes/Na=Nb4/sector4900/T=.8。PF full prefix28、PR prefix7、delta=T/q。保存state/input SHA d4c43461、compiler1239304c/Qiskit1.3.0・basis rz/sx/x/cx・opt1・seed17・topologyなし。

## 実測費用

以下はprimary symmetric_directionalの**1軸・1 full-wrapper当たり**。全8ケースでcosine/sineの6指標は一致したため1行にまとめる。paired合計RZ/CXは2倍だが、shot数を掛けた精度一致総費用ではない。状態準備は含まない。

| 構成 | RZ | CX | RZ depth | CX depth | total depth | size | worker最大RSS GiB | primary最終metric時点 分 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1_p28_q1_2nd | 104,229 | 46,312 | 11,356 | 18,225 | 34,535 | 217,188 | 0.789 | 5.12 |
| B1_p28_q1_4th | 309,505 | 138,110 | 33,815 | 54,549 | 103,044 | 645,542 | 2.042 | 15.05 |
| B1_p28_q8_2nd | 816,456 | 355,408 | 87,623 | 137,304 | 264,050 | 1,696,487 | 2.587 | 38.64 |
| B1_p28_q8_4th | 2,456,952 | 1,082,464 | 265,767 | 422,792 | 805,714 | 5,115,015 | 6.815 | 118.00 |
| B2_p7_q1_2nd_r4_K2 | 34,484 | 13,932 | 3,626 | 5,032 | 10,396 | 70,567 | 0.525 | 1.42 |
| B2_p7_q2_2nd_r4_K2 | 63,142 | 26,262 | 6,638 | 9,747 | 19,408 | 129,614 | 0.718 | 2.29 |
| B2_p7_q4_2nd_r4_K4 | 128,619 | 53,178 | 13,517 | 19,619 | 39,388 | 264,344 | 0.715 | 4.32 |
| B2_p7_q8_2nd_r2_K2 | 241,298 | 101,906 | 25,486 | 38,326 | 75,263 | 496,345 | 0.905 | 7.75 |

RSSはworker全体の最終HWMで、ordinary anchorを含む2ケースではprimary専用peakとは異なる。時刻列は診断clockのprimary最終metric終了点で、exit/terminal overheadは含まない。各wrapperのclassical_wall_secondsを合算してcampaign時間としない。

ordinary anchor（各値1軸）：PF S4/q1はRZ656,018/CX361,136/depth568,313/size1,258,563、PR p7/q2/r4/K2はRZ121,056/CX63,542/depth97,094/size232,080。same-anchor primaryのRZは309,505/63,142。これはcontrol policyの局所対照で、PR対PFの精度一致優位を表すものではない。

## Heavy S4/q8と時間・監視

heavy監視wall 7126.273秒（約1時間59分）、remaining7の監視wall 3012.395秒（約50分）。順次合計10138.668秒（約2時間49分）。開始/終了の観測時刻はnative exit receiptに保存、exact exit timestampはnullのまま。

heavyのworker peak RSS6.815GiB・VmPeak7.476GiB、aggregate peak6.844GiB。AS32/RSS24GiBの承認内。compile後size5,115,015は旧500万guardを超えるが、承認1000万guardの51.15%。旧failed exact sizeの欠測を、この新runの値で補完しない。

heavyの非重複stage内訳：numeric fingerprint4256.015秒（監視wallの59.72%）、control validation2139.720秒（30.03%）、実transpile両軸463.536秒（6.50%）、metric計測161.949秒（2.27%）。build/wrapper_fingerprintはhashを内部に含むので二重加算しない。追加の高速化候補はhashの正当な再利用とcontrol作用の削減であり、現データから旧方式との全体倍率は認定できない。

remaining7 aggregate peak 5.111GiB。最大3workerの実行を確認したが、8worker実動作の性能を測定した結果ではない。5秒監視内のstaleness/観測所要時間・pressure・OOM deltaはcompletion auditへ集約。first STOPは両runともnull。

## 整合性と限界

257 source・5 inputのbyte/streaming SHA・signal receipt・固定compiler/profile JSONを照合。NPZ科学arrayは読まない。実行前のnative318 subset検証を再利用し、実行後のvenv全inventory監査はしていない。既存venv/共有設定/H4/他jobへ変更する操作は行っていない。旧host compilerのoutput bit完全一致は未検証。

今回のcontrol40 logical probesの最大誤差は1.85519e-13。実E作用はcontrolあたり2branch、X/Y suffixを再利用する方式。過去の8signal/1164 primitive probes・人工15/44/40/23 testsは再実行していない。

PRは各構成1 trajectory、RTE event draws合計44はすべてTaylor次数0。会計上のoccurrence15はq sampling callsで、q*rのevent draw数とは別。希少次数費用の母平均・厳密tail upperは未認定。今回のper-wrapperの小ささを、精度一致・normalization/shot込み総費用の削減率へ換算しない。92候補のmap・独立32確認・正式精度認定・ground-state certificateは今回のscope外。

旧failed actual1compile/control4は保全、新成功20/40と合算して累積21/44。trajectory4・occurrence15、返却/reset0。setup01は科学worker/lock/call0。過去のunknown actual/終了code/timestampはnullから補完しない。

成功run3領域の保存容量は5.846MiB、block割当7.480MiB、file493件。NPZや旧setup/公開キャッシュ/他lineageはこの値に含めない。

## 本格計算の技術予算案（未承認）

GPT採択済みprimaryのrank28/p7,14,21/q1,2,4,8/r1,2,4/K2,4/fullPF S2,S4は維持。92 signal、探索最大596 group、独立確認最大256 group、core1708 compile/832 trajectory。optional tail40/80は別枠、transfer副評価も別枠。primitive/reference/control/normalizationの最終call上限はfull-map実装とcoverage固定後に算出する。

preflightの20/4を将来再利用できる可能性はあるが、full-map seed/namespace/selection bindingが未固定のため現予算の節約予約は0。H6 known+予約1736/852、H4補完532/256、H8累積21/4を含む既知lineage合計2289/1112。新coreを足すと3997 compile/1944 trajectory、optional tail込み4077/1984の提案。別の既存H4 production・未知歴史actualは含めていない。旧grantやcapを変更した値ではない。

提案はnormal最大8（RSS9/AS32GiB）、未測定p14/p21・rareをheavy/unknown（RSS24/AS32GiB）、aggregate76GiB/job admission96GiB/内部thread1。測定最大RSS×1.25=8.519GiBは9GiB枠内だが、未観測回路をnormalへ一括認定しない。1heavy+5normal=73GiB、3heavy=76GiB、8normal=76GiB。actual許可CPUは空、CPU20/21/24–29・co22/23はfresh監査前の案。

条件付きwall model：pに比例・測定q anchor・r/Kの増加なし・zero-order realization・PF q2/q4線形外挿と仮定すると、cost部分は91.2–184.4 worker-hours。8worker/負荷係数1.5–3なら約17–69時間、3workerなら約46–184時間。**未実測p14/p21・希少次数・92 signal/reference・I/Oを含む完走保証ではない。**

output32GiB/storage admission64GiB/inode100000案を維持し、optional tail含む892group×32MiB=27.875GiBにjournal1/index checkpoint1/scratch cache2GiBを明示配分する。block allocation・directory inodes・同時一時file・quotaをfresh監査、monitor logはrotation/hash-sealed bounded segmentsをfull runnerへ実装する。今回の数MiBを将来容量の確定値にしない。

残る技術事項はB0/p14/p21/full-map runnerと92候補のsignal/reference/allowance、bounded journal/complete-only resume、上限全会計・seed reuse mapping・fresh resources。残る研究判断はoptional tailをdiagnosticとするかestimatorへ入れるか等。runtime_authorization=false/approved=false/allowed_cpus=[]/absolute_launch_command=null、本格実行grantは未発行。

## 資料

- [20 measured rows JSON](../../artifacts/resource_applicability/track_a_h8_fast_retry21_v1/2026-10-13/measured_wrapper_rows_v1.json)
- [20 measured rows CSV](../../artifacts/resource_applicability/track_a_h8_fast_retry21_v1/2026-10-13/measured_wrapper_rows_v1.csv)
- [保存metadata監査・原証拠SHA](../../artifacts/resource_applicability/track_a_h8_fast_retry21_v1/2026-10-13/completion_audit_v1.json)
- [本格資源見積もりJSON・未承認](../../artifacts/resource_applicability/track_a_h8_fast_retry21_v1/2026-10-13/resource_map_technical_estimate_v1.json)
- [一つの資料入口](../../artifacts/resource_applicability/track_a_h8_fast_retry21_v1/2026-10-13/README.md)
