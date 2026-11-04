# 研究C：Stage P固定1-wordのper-word包囲・部分実行STOP

2026-11-04 JST。**固定41-call契約を一度実行し、30 attempts／29 COMPLETE／記録されたFAIL 0／未完了1で中断を観測した。** Compiled→raw、raw→literal atomの全512列包囲と、その和によるper-word上界は取得済み。Compiled actual inverseの検証中に実行プロセスとsessionが消失し、runnerの最終result／STOP／resourcesは生成されなかった。原因・終了時刻は未確定であり、科学的correctness FAILとも正常完了とも認定しない。Retry/resume 0、残り11 unitsは転用しない。

## 固定条件・SOURCE・preflight

- 科学SOURCE：[f658f72d5f309538cba23b0745c7e9389411758f](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/f658f72d5f309538cba23b0745c7e9389411758f)。旧runner／adapter／数値条件を無変更で使用した。
- 実行checkoutの親：[Head結果369e2dfad9a4251ba5eb2fda28f183d559a791cc](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/369e2dfad9a4251ba5eb2fda28f183d559a791cc)。科学SOURCEと結果収録commitを区別する。Headの26-callは再実行していない。
- Branch：`codex/research-c-stage-p-execution-20261104`。
- [固定P契約](research_c_stage_p_literal_contract_unapproved_20261104.json)：`C_PR_R32_ONE_WORD_LITERAL_STAGE_P_V1`、最大41 logical calls、Arb192、retry/resume 0。
- 契約SHA256：`ab40f08c28c32bf803a95571f296779875864db2ba6b32a9d4b7eea1d8223cab`。
- [SOURCE manifest](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/technical/SOURCE.json) SHA256：`61684d17fa849d67f6717b019f3098edaef0c32daa06c85af8ccf07c7ca7775a`。162 SOURCE file、48 saved input。
- Fresh出力：[P-run01](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/P-run01)。旧runのretry/resumeや残枠を使用していない。

対象は直鎖H4・1.00 Å/STO-3G、number CDF8、元順LD2、partial ordered symmetric S2/s1/r32/K0、h=δ=.1。Nα=Nβ=2の物理sector36に限定せず、今回のkernel証明はsystem全256基底・controlled全512入力列／全出力を対象とする。固定wordは`CDF_proposed:F6:0`、`CDF_proposed:F7:6,7`の交互16回だけ。別wordの選定、sampling、lawの再生成はない。Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、κ・scalar・frame順・既存P0、own-DF ground oracle／状態準備除外、共通CA/RPE政策を維持した。

[Preflight](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/P_execution_preflight.json)はPASS。SOURCE／契約／入力を固定commitと照合し、fresh出力不在、package、FE_TONEARESTを確認した。Python3.11.1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、python-flint0.8.0／FLINT3.3.1。ユーザーの今回の明示許可で実行し、歴史的unapproved名・metadataを変更せず、承認証明・承認原文保存・新grantは作成していない。

入口は既存`prepare_research_c_head_bridge_stage_p.py --stage P --execute-science`。実際の実行は一回だけで、SOURCEの作り直し、数値allowance変更、fallbackはない。

## Call実績と中断

[原journal](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/P-run01/science_calls.jsonl)のattempt／terminal eventをそのまま保全した。

| 区分 | attempts | COMPLETE | 未完了 |
|---|---:|---:|---:|
| metadata／input／candidate／head／model／catalog／binding | 7 | 7 | 0 |
| Gaussian basis | 5 | 5 | 0 |
| relative Gaussian | 4 | 4 | 0 |
| raw kernel build／P0 compile | 2 | 2 | 0 |
| raw／compiled records | 2 | 2 | 0 |
| literal reference | 1 | 1 | 0 |
| compiled→raw／raw→literal | 2 | 2 | 0 |
| per-word triangle | 1 | 1 | 0 |
| actual adjoint build／record | 4 | 4 | 0 |
| actual adjoint certificates | 2 | 1 | 1 |
| shell load／certificates、finite signal、summary | 0 | 0 | 0 |
| **合計** | **30** | **29** | **1** |

12:07 UTC頃のsession pollでは実行中だったが、12:27:53 UTC（21:27:53 JST）の確認では当該jobが見つからず、session照会も`Unknown process id`となった。原journalの最後はcompiled actual adjoint certificateのattemptで、COMPLETE／FAILのどちらもない。中断時刻・終了code・原因は捕捉できていない。OOM、計算の不整合、実行基盤の制約など、特定の原因へ推測で帰属させない。

Runner最終化が行われなかった事実を保持するため、P-run01内へrunner由来を装ったSTOP／resultを補わない。[観測後のmandatory STOP](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/P_execution_STOP.json)と[部分summary](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/P_execution_partial_summary.json)はrun外に保存し、記録の作成者・由来を明記した。未完了callを記録上のFAILへ書き換えていない。

## 取得できたper-word包囲

以下は**exact captured gate／saved atom targetに対する、Arb192でroundingを含む数値包囲**であり、有限probe残差からの外挿ではない。全512入力列のcolumn2上界からFrobenius上界を作り、operator2 normを包囲した。保証に用いる正本は各JSONのexact rational endpointであり、表は表示近似である。

| 対象 | operator2誤差上界・表示近似 | 正本 |
|---|---:|---|
| Compiled C → raw R | 1.585733058571e-11 | [compiled_raw.json](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/P-run01/compiled_raw.json) |
| Raw R → literal atom D | 3.957579638642e-14 | [raw_literal.json](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/P-run01/raw_literal.json) |
| Compiled C → literal atom D：二項の和 | **1.589690638209e-11** | [per_word.json](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/P-run01/per_word.json) |

Same raw record digest、compiled digest、word／target／angle identityを照合した。Literal referenceは25 spin-count blocksが全system基底256個を重複なく覆い、full controlled action側のleakageも比較に含む。Gaussian途中gateを物理sectorへ射影した代用ではない。Controlled相対位相とglobal phaseを保持し、phase fitはない。

C→Rは同一unitary gate作用間の既存certificate。R→Dはliteralな保存U・full-CAR head・saved angleを保持し、polar化やκの射影、別headへの置換はない。Dをunitaryと仮定せず、R→Dと合計に根拠のないcap2を使用していない。Arb／CARの数値誤差は各legに含み、別項として再加算しない。

[Stage R](research_c_population_range_schema_execution_20261104.md)のexact a/Mをそのままbindingし、aを1、Mを2へ丸めていない。今回のDはStage Rのfull-CAR head targetと同一であり、[既存Head bridge](research_c_head_bridge_execution_20261104.md)のB2への追加legはper-word合計に加えていない。

## Actual inverse・shell・signal

Rawとcompiledのactual inverseは同じ実現回路から各一件生成し、計4 QPYを保存した。これは別wordのinverse引き直しではない。

Raw actual inverse対exact raw-record adjointは全512列で完了し、上界表示近似 **3.535574829163e-53**。[raw_actual_inverse.json](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/P-run01/raw_actual_inverse.json)にexact endpointを保存した。Raw→Dのforward上界との和でD†への上界を保存したが、非unitary Dの逆行列との同値性は主張しない。

Compiled actual inverseはbuild／recordだけ完了し、certificateは未完了。Head／X／Y shellの読込・certificate、単一occurrenceのraw complex signal包囲は未実行。これらのboundは**null**であり、0や旧4-wordの数値pilotで補完しない。Shellを含むend-to-end signal certificateやStage P全体の成功を認定できない。

## 資源・検査・保全

開始時CPU affinity32、空きRAM約52 GiB、空きdisk約75 GiBを確認した。既存実装がliteral block用25 worker、全column用32 workerを選び、process-local Arb／gate cacheを使用した。Jobの32 workerが稼働することも実行中に読み取り確認した。恣意的なCPU/RAM/worker/wall上限、共有venv・共有cache・他jobの変更はない。

Journalに記録された完了処理は、compiled→raw約1000.44秒、raw→literal約166.94秒、raw actual adjoint約337.86秒。全runの正常終了wall／peak RSSは未取得であり、観測までの経過時間を完了runtimeと扱わない。速度向上率は測定・主張していない。

[技術検査](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/P_execution_verification.json)はPASS。SOURCE／契約コピー、call予算、29 COMPLETE＋1未完了、三つの完了certificate各512列／Arb192／phase保持、同じrawを用いたexact二項和、25 block全基底coverageを保存JSONから照合した。科学evaluatorを再実行せず、未取得のcertificateを新規に計算していない。準備時の人工43／関連355 PASS・7 deselectedは[既存技術証拠](research_c_head_bridge_stage_p_readiness_20261104.md)として参照し、今回の未完了pipelineの全面PASSへ流用しない。

旧SOURCE・入力・両契約・Head26-call全原本など238 fileを実行前hashと照合し、一致した。P-run01の16 file／10,812,853 bytesを保全した。Raw／compiled／両actual adjointの4 QPY、4 gate records、literal reference、完了certificate、契約／SOURCEコピー、原journalを含む。原journalを追記・修正せず、既存H2/H4/H6結果、Stage R・旧失敗、grant・ledger・marker・STOPを変更していない。

## 結論の範囲と次担当

固定1-wordのC→R→D per-word operator誤差について、数値観測ではなく全512列の包囲を実際に取得できた。一方、compiled actual inverse・shell・signalの未取得により、**固定41-call Stage P全体は未完了**である。1-word compile成功は全合法wordのcompiler成功保証ではない。全law一様synthesis、期待bias校正、Stage C、RPE shots、actual CA、重要科学レビューは実施・認定していない。

次担当はユーザー／研究C。今回の部分証拠を踏まえ、未完了箇所を対象とする独立の後続検証が必要か判断する。Codexが追加検証を行う場合も、今回のrunのresume／残枠流用ではなく、必要範囲を固定した別契約と明示許可が必要となる。今回は新SOURCE・新契約を作らず、追加計算へ進まない。

結果・SOURCE・契約コピー・journal・QPY・観測STOPをcommit／公開し、remote SHA/tree・本文・主要blobを照合する。**公開照合後mandatory STOP。**
