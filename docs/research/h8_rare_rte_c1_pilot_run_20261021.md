# H8希少RTE C-1 pilot：本番driver実装・runtime fork gateで科学実行前STOP

2026-10-21 JST。基準commit `77bf6472f901feab8cdc4871369bc8ddfc04be7b`。**ユーザーの実科学承認を受領したが、実runtime import後の親が64 threadsとなり、固定された単一thread fork gateを満たさずSTOPした。** Phase1/Phase2とも0/43、compile0。有効grant発行/消費0。科学load・prefix復元・law形成/数値整合・RTE抽選・simulation・回路生成・compile・pilot会計は実行していない。再試行・環境変更・Phase2進行は行わない。

## 完了した技術準備

[C-1固定契約](h8_rare_rte_c1_pilot_contract_20261021.md)とC-0.5 adapter/scheduler/sampling/会計を保全した。[本番driver](../../scripts/h8_rare_c1_run.py)を新SOURCE `3d2478945a9d51e52aff356419b429f9024cd53b`としてローカルcommitした。

- 43本・S2_SINGLEのみ、両条件のq/r/K・候補・予算・private registryを厳格照合する。
- replica000を先行し、native exit/reap、同event両軸、control/oracleを受入後だけ残42本をdynamic trajectory/両軸fork-COWへ投入する。
- 43本すべてのevent/count/axis/metricsを照合し、mean・不偏variance・SEと別実装raw momentsを一度だけ会計する。
- grant消費前のbyte/import/library/技術forkと、科学load後・各fork前のthread/library gateを分け、非返還予約・新marker・部分結果STOPを接続した。
- 実memory/thread/IO需要の根拠がなければ起動を拒否する。固定の小さいworker/CPU/RAM/thread capを新設しない。cacheは固定契約どおり無効。

元のpr_cost_native/pr_cost_contract/pr_cost_integrationを745a6745の固定Gitからprivate bridgeへbyte不変で取得し、元Pのimport recipeと実module SHAを使用した。元context bindingは固定契約と一致した。旧SOURCE・raw・grant・ledger・marker・STOPは変更しない。

[新8人工テスト](../../tests/test_h8_rare_c1_run.py)はsymbolic providerで両qの43本完了・正確予算・独立moment照合、代表失敗による残42本未投入、改変registry/予算拒否、thread STOPとgrant未消費を確認した。既存55件もPASS。計63件で実H8科学成功を主張しない。今回14件のmock native receiptで所有子・孫の回収を確認した。別AIへの委任・科学reviewは0。

## 実runtimeの開始前gateとSTOP

元Pythonと元runtime/providerの固定byteを、分子load前の独立した技術processで照合・importした。自分のprocessに`PYTHONDONTWRITEBYTECODE=1`を追加し、旧環境へpycacheを書かない。OMP/OPENBLAS/MKL/NUMEXPR/RAYON/Qiskitのthread環境や共有設定は変更しない。

|項目|実測・状態|
|---|---|
|元byte照合|35,304 unique physical file一致（旧35,292＋bridge3＋今回のexecution source9）|
|critical module SHA|7/7一致|
|Qiskit version gate|1.3.0 PASS|
|import後の親native threads|64|
|固定fork条件|single thread必須。不合格|
|停止理由|`ValueError:UNSAFE_MULTITHREADED_FORK_PARENT`|
|SessionEquivalenceLibrary signature・親子照合|thread gate後なので未到達|
|fork probe・科学worker|未起動|
|科学loader後のfork安全性|未評価、科学loadなし|
|数値law・canonical 1e-12照合|未実行・適否未確定|
|実H8 resource footprint|未取得、値を補わない|
|技術probe elapsed|2.5080297235399485秒|
|技術probe RSS/PSS/private|115437568 / 112646144 / 112590848 bytes|
|技術probe native exit|2、親subprocessで回収済み|

[実preflight](../../artifacts/resource_applicability/h8_rare_rte_c1_pilot_run/2026-10-21/runtime_preflight.json)。64 threadsの原因を特定のlibraryへ断定していない。無変更のimport結果と固定gateの不一致を記録した。idle threadだから安全と読み替えたり、thread数を1へ変更したり、spawnへ自動切替したりしない。実runtime probeは1回のみ。失敗後は保全・構造・終了監査だけを行う。

## 条件別の科学実行状況

共通科学条件はlinear H8/1 Å/STO-3G/tol-only DF28/T=.8/P=0、二次partial PF、元finite-RTE/native DF/analytic shots/scalar/control/full Hadamard-wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。変更していない。

|条件|候補|予定n|取得n|実compile|条件付きmean/不偏variance/SE|判定|
|---|---|---:|---:|---:|---|---|
|ε=.05|B2_p8_q2_2nd_r4_K4|43|0|0|null / null / null|Phase1科学開始前STOP|
|ε=.0005|B2_p7_q16_2nd_r4_K4|43|0|0|null / null / null|Phase1不成功のため未開始|

[条件別JSON・全call予定/予約/実使用](../../artifacts/resource_applicability/h8_rare_rte_c1_pilot_run/2026-10-21/condition_results.json)。予約/実使用0は科学entryに到達していないことの記録であり、未知event費用を0としたものではない。mean/variance/SEは観測なしのnull。法則の不適合を発見したわけではなく、照合自体を実行していない。

元の予算は各43 trajectory・86compile、coarse344/tight2752event、各4control probe・1event oracle・2非測定wrapper、decode/prefix/law/count/component/会計/別数値検査も既存値のまま。[停止中の新契約](../../artifacts/resource_applicability/h8_rare_rte_c1_pilot_run/2026-10-21/README.md)は科学scopeの承認と、runtime authorizationが成立していない状態を区別する。新実行承認receiptを保存したが、有効grant・科学ledger・one-shot markerは発行していない。

S0既存P n48を新pilotへ混合・再解析していない。S2_MULTI/S4_PRESENTは未知。全層期待費用、PR/PF順位、production本数、母平均保証に新しい結論はない。

## 保全・終了・次の判断

[SHA保全](../../artifacts/resource_applicability/h8_rare_rte_c1_pilot_run/2026-10-21/preservation_audit.json)、[native exit](../../artifacts/resource_applicability/h8_rare_rte_c1_pilot_run/2026-10-21/native_exit_audit.json)、[終了・構造監査](../../artifacts/resource_applicability/h8_rare_rte_c1_pilot_run/2026-10-21/completion_audit.json)、[STOP](../../artifacts/resource_applicability/h8_rare_rte_c1_pilot_run/2026-10-21/STOP.json)。失敗した技術processと人工workerは回収済み、科学worker0。旧証拠を含む35,304ファイルをpost-STOP再hash一致。共有環境・別4次PF研究・他processへの介入0。private paths/seed/raw/tracebackはpublic候補から除外し、限定pattern/リンク構造とSHAを検査した。包括的なsecurity certificateとはしない。

専用branch `codex/h8-rare-rte-c1-pilot-run-20261021`にSOURCE・停止証拠・監査をローカルcommitし**mandatory STOP**。push、production、追加希少層、詳細GPT/Claude科学review、最終evidence freezeへ進まない。

次の担当は**ユーザー**。64-thread import結果を踏まえ、元runtimeのthread/fork条件を根拠付きで確認する技術対応か、別方式の検証契約を指示するか判断する。**Codex**は新指示まで停止する。現在のSTOPを維持し、既存gateを緩和した再実行を自動で行わない。再実行時は科学scopeと新SOURCE/技術gateを照合し、新grantを用意する。GitHub公開も範囲の別承認が必要。
