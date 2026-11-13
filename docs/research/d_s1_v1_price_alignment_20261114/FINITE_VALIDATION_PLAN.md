# S1/V1の有限追加案（未承認・未実行）

**取得順を自動実行する計画ではない。** C d7c09のFull PF案と同じセルから必要分を選ぶ。Full-H案も独立した選択肢であり、旧T2/m12のretry・未使用枠は使わない。Cのfit/予測、Dの実preparation/build/compile、BのN/Gにはそれぞれ範囲を明示した科学承認が必要。

A後続d48d873の追加幅は未実行計画。C/Bが元S2/.2を必要な比較familyへ残す場合だけP-S2を第一候補とする。別actual幅が選ばれたら本表をそのまま走らせず、Aのexact recipeと既存receiptを使い必要scopeを選び直す。旧/新幅の全gridや両menuの合算は採択しない。

## P：A/CのFull PF shortlistに合わせる

同保存H4/1Å/STO-3G/tol-only DF10、元H/state/snapshot/sector、one-body＋fragment0..9/元順序、h=.1/M12、P0、元deterministic native/control/scalar/wrapper/compilerを使う。RTE loweringは非該当で、PF nativeそのものは変更しない。

| 選択枠 | formula / actual幅 / round | T / qPF / physical ID | 役割と最大compile |
|---|---|---|---|
| **P-S2：最初の候補** | S2 / .2 / m5 | 3.2 / 16 / `987ba8031fd5b790cf59` | 保存m3 q4・m4 q8をCが受入・fit/freezeした後のholdout入力、X/Y最大2 |
| P-M5-CAL：別選択 | M5 best / .8 / m4 | 1.6 / 2 / `8cf3ab565283bebfbbba` | 保存m3 q1に対する追加校正入力、X/Y最大2 |
| P-Y4：別選択 | 標準Y4 / .4 / m5 | 3.2 / 8 / `0de9142bd399c82598c8` | 保存m3 q2・m4 q4のfit/freeze後、X/Y最大2 |
| P-M5：別選択 | M5 best / .8 / m5 | 3.2 / 4 / `aca80e6e14487cde159e` | 保存m3＋新m4のfit/freeze後、X/Y最大2 |

完全なconfiguration/canonical/physical IDは [REPRESENTATIVE_CELLS.json](REPRESENTATIVE_CELLS.json) にA固定packetからcopyした。新しいbuilderや幅の丸めで再生成していない。formulaのsigned weights・piece順・`T*w/q`の演算順は元 `h46_m5_plan`/composeを使う。

各軸の決定論回路1つをcompileし、trajectory samplingは**0**、index0、seed_transpiler=17。cohortは科学的乱数母集団ではなく既存costの出力を区別する識別子として、それぞれ `D_S1V1_P_S2_V1`、`D_S1V1_P_M5_CAL_V1`、`D_S1V1_P_Y4_V1`、`D_S1V1_P_M5_V1`。RNG seedは存在しない。決定論価格にn1 cohort標準誤差を付けない。

P-S2だけなら最大2 wrapper。全P枠を別途採用した場合はC既存案と同じ4round/8 wrapperであり、**C案8件にD案2件を重ねて10件にしない**。別lowering/nativeを選ぶ場合は保存校正の控除を自動適用せず、C案の変更をユーザーへ戻す。

保存5roundの価格値受入・校正fit・係数freeze・holdout予測/比較は**Cの科学scope**。今回は値を読み出していない。Dだけを承認するなら直接価格/receiptの取得までで、Cのfit/分析を含めたことにはならない。holdout値を係数決定へ混入しない。必要精度/採用threshold不明なら独立な価格点を得てもmodelを正式採択しない。

## F-H：Full-Hを新ZZ比較へ残す場合の最小技術pilot

現C signal SOURCE367dc98の `FULL_H_K2_QDYADIC_R448/PLAN.json` にある**m0のphysical request**を選択する。

- configuration ID：`cfa6aee3260de664de9acd444a628faae388c459a2eaf68a786c707573029d1c`
- physical ID：`FULL_H_8bc19ca6d8f181dac602`。T=.1、grouping q1、R448、K2、S2正piece1、448 event位置。qPFを1と再ラベルしない。
- full one-body＋全non-scalar DFをrandomize、head空、元Full-H preparation/tail/component/basis順。PRのprefix0やB3（one-body deterministic）へ置換しない。
- fresh normal **1 path/index0**、cohort **`D_S1_FULL_H_NEW_ZZ_M0_V1`**、outer0/piece0のseed **`16511236847028451318`**。
- seedは元Full-H9b83 `scripts/h46_full_h_pure_rte.py:trial_seed`（`DF_FULL_H_LIGHT_20261109` family）から取得した。PRの`H46_LONG_TIME_20261025` seed関数やT2.0専用familyを使わない。元seed関数の再利用は旧科学grant/cohortの再利用ではない。
- 新ZZ `CONTROLLED_ZZ_CX_CRZ_CX_V1` を実際のprovider moduleへ有効化し、X/Yを個別に取得、最大**2 compile**。旧Full-H価格との差の数値計算やmean/CIは含めない。

これは同candidateの新lowering path証拠であり、現Full-H lawの適格平均ではない。旧lightのT.8/K4等やT2.0の保存価格をこのm0の対照価格に使わない。旧保存eventが揃う場合の同path新旧replayは別の有限scopeを設計できるが、今回private原本の可用性を確認していないため実行案へ追加しない。

## 既存pipelineへの接続と科学call範囲

production/native基点はD01e068（新ZZはその前から存在）、provider c31、Full-H9b83、carrier155529/同compose-costのaf284。Full PF receipt/e7dとCの同native provenanceを保持する。科学実行用の最小caller・実行SOURCE完全SHAは**未確定**。選択範囲を受けて実loaderを人工入力で確認し、新commitで固定してから、その完全SOURCE/input/出力/call範囲を明示承認する。旧科学SOURCEの関数を黙って現HEADの同名関数へ差し替えない。

P-S2の予定呼出しは既存saved loader/native context（prefix10）1回、元compose/cost_trial index0を1回、evolution1・measured wrapper2、新compile最大2、normal sampling0。F-Hは既存Full-H `context('H4')` 1回（内部のbase prepare(ham,0)・Full-H追加preparation各1、元保存参照/occupation再構成/spectrum/独立oracle検査を含む）、元normal sample1 path、compose/cost_trial index0を1回、evolution1・wrapper2、新compile最大2。

両方ともindex0の元代表検査を保持し、low/high control・独立occupation trajectory・両wrapper Hadamard・compiled vectorを含める。cache miss時のstatevector呼出しはcost_trial内で最大6、normal pathのoracle作用照合1。新reference生成・全signal batch・N/費用report・law/rareの新数値解析は行わない。旧Full-H `main/run_cost`は固定n9やendpoint/費用会計まで起動するため、新pilotのlauncherとして実行しない。

新ZZの切替はFull-H setupで使う実provider moduleが確定した後、composeを含む範囲だけに適用し例外時も戻す。関数内でbuilderをcall-time importする元経路を使う。PR専用 `candidate_price_adapter.from_b` はprefix0/Full-Hを拒否する設計なので、guardを緩めず、既存Full-H/native取得を使う。F-HのSOURCEに一致する独自scalar `c_H + extracted_full_H_identity` を元順で比較し、LD2のtail-only値を使わない。

Full-HへH4 PR用Accountingを適合検証せず流用しない。本F-H案で価格S/Dは未取得ならnullとし、CをSで埋めない。PRを別途取得する場合は既存adapterのpaired S/C/Dをそのまま保存する。

## 出力・資源・失敗条件

各選択枠の新公開出力案は `docs/research/d_s1_v1_price_execution_20261114/<cell>/run01/`、privateは `/home/AbeHiromu/projects/d-s1-v1-price-20261114-private/<cell>/run01/`。`<cell>`は表のP-S2/P-M5-CAL/P-Y4/P-M5/F-H。未存在をmetadata確認しただけで、今回作成していない。実行SOURCEが変わる新request/consumer keyを生成し、旧IDやcache meanを付け替えない。

軸別compiled RZ/CX数・depth/全depth/size、measurement付きwrapper fingerprint、compiler/lowering/native/source、保存input/H/state/prep/tail/scalar provenance、cohort/role/seed/event digest/order・component/basis診断、cacheと代表作用gate、compile/全wall/peak RSSを記録する。既存fast writer/metrics/cachesを再利用できる部分だけ再利用し、未検証のFull-H fast Accountingを追加しない。新compile前のcache hitはfull fingerprint・compiler・SOURCE/physical条件が一致する場合だけ。独自の古いFull-H cacheを新policyのlaw meanとして採用しない。

元cost_trialは軸ごとのmetricsを返し、fingerprintはcompile_cached側のrecord/引数へ渡す。必要なcaller差分は、既存compile_cachedを呼ぶ小さいobserverで既に求めたfingerprint・compiler・compile時間を取得し、元resultへsource-bound sidecarを添える範囲で足りる。gate列やcompilerを再実装しない。今回その科学caller/observerは実装・起動していない。

CPU/RAM/workerに人工capを追加しない。process-localのlowering/hash切替は単一threadで扱い、独立pathのprocess並列は承認範囲と実資源で選ぶ。今回の各1 pathから並列母平均取得を開始しない。OS/共有library・他人process/filesは変更しない。一時的な競合のみでretryせず、実OOM/継続不能・source/scalar/measurement/gate不一致では部分結果を保全してSTOP。

自動retry・別seed/round/path、X/Y片軸化、追加価格/fit/外挿、全round再集計、N/Gは0。原本不足や元source/contextを安全に束縛できなければ代替を推測せずSTOP。小セル成功後のlong-q/R認定とmean精度は[V1](V1_LONG_TIME.md)の条件に戻る。

新scientific SOURCEが未固定なので、本案を現状のまま実行可能な承認済みjobと扱わない。現作業は技術成果の公開照合後 **mandatory STOP**。
