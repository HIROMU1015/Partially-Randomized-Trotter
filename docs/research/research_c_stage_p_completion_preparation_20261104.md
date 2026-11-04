# 研究C：Stage P未完了項目だけの後続実行準備

2026-11-04 JST。ユーザーの今回の再実行指示の範囲を、compiled actual inverse、Head/X/Y shell、単一occurrence raw complex signalの取得に限定した。旧[30-attempt部分実行](research_c_stage_p_literal_execution_20261104.md)と残り11 unitsは維持し、旧runのresumeや残枠流用を行わない。後続契約は**独立した最大15 logical calls**である。

## 再利用と新しい差分

起点は旧結果commit `5e1450b1071359b81819fe5de481cdeccc9d3f35`。旧SOURCE／raw／契約／STOPは変更しない。完了済みC→R、R→D、C→D、raw actual inverseのexact証拠をhash・同じraw/compiled digest・word/angle/target・位相・512列coverageで照合し再利用する。保存compiled actual inverseのrecordを使用し、新build／compile／samplingは0。固定F6 Z／F7 ZZ交互32-event wordを変更しない。

新module [c_stage_p_completion.py](../../src/trottertracks/c_stage_p_completion.py)は、旧`c_stage_p_literal`のshell／signal、`fixed_certificate._init_pair/_pair_column/resource_plan/adjoint`、Arb192を再利用する。Full512のcolumn算術は元のまま。`as_completed`で各完了列をfresh `columns.jsonl`へflushし、`progress.json`をatomicに更新する。集計前にcolumn番号順へsortし、元のFrobenius／lower算術と順序を維持する。保存列からのrestart機能はない。

新entry [run_research_c_stage_p_completion.py](../../scripts/run_research_c_stage_p_completion.py)は、foreground childを待つ小さなsupervisorを持つ。Workerのstdout/stderr、faulthandler、終了code／signal、heartbeatを記録する。Own childへのSIGTERM/INT/HUPを転送し、workerは捕捉できたsignalをpartial／STOPとして保存する。Timeout上限、detach、OSやtool制約の回避、他job操作は導入しない。

**前回の消失原因は未確定のまま。** これらは観測・終了記録を改善する変更であり、同じ障害が必ず防げるという主張ではない。親子ともSIGKILLされた場合などは最終記録を保証できないが、完了した列は途中証拠として残る。科学的な誤差条件や不明値を変更して成功扱いにしない。

## 固定科学条件とcall数

保存直鎖H4・1.00 Å/STO-3G、number CDF8／元順LD2、partial ordered S2/s1/r32/K0、h=δ=.1、controlled512。既存P0、κ/scalar/frame順、own-DF ground oracle／状態準備除外、共通CA/RPE政策を維持する。今回のsignal対象は固定wordの単一occurrenceだけ。13round、期待bias、shots、actual CAを新しく評価しない。

[後続契約](research_c_stage_p_completion_contract_20261104.json)はscope `C_PR_STAGE_P_MISSING_OUTPUTS_V1`。新SOURCEは旧162 file＋module／entry／testの計165 file、入力69件。Fresh出力は`artifacts/research_c_stage_p_completion/2026-11-04/run01`、launch記録は同`launch01`。Retry/resume 0、mandatory STOP。両directoryの既存を実行前に拒否する。

| 区分 | 最大calls |
|---|---:|
| 保存証拠読込 | 1 |
| 保存compiled／actual inverse records読込 | 1 |
| identity／既存完了証拠の適用gate | 1 |
| compiled actual inverse certificate・全512列 | 1 |
| head/X/Yのraw／compiled QPY読込 | 6 |
| 3 shell certificate group（各3組の2×2比較） | 3 |
| 単一occurrence signal包囲 | 1 |
| summary | 1 |
| **合計** | **15** |

Inverseのtargetは同じcompiled recordのexact adjointであり、そのcertificateと保存C→D上界の和をD†へ接続する。Nonunitary Dの逆行列ではない。Shellのancilla-only条件・controlled相対位相・terminal measurement除去規約は旧実装のまま。Signalは保存Stage Rのexact a/Mと各compiled→ideal shell項を一度ずつ使用し、未知値を0にしない。All-q compiler totality／全law／校正／CAは未資格のまま。

## 人工検証と実行前gate

[追加人工28 PASS](../../artifacts/research_c_stage_p_completion/2026-11-04/technical/tests.txt)、[関連回帰96 PASS・2 deselected](../../artifacts/research_c_stage_p_completion/2026-11-04/technical/regression_tests.txt)。Invented full512 gate列で並列column結果・exact endpointが既存serial engineと一致した。列別保存、fresh拒否、途中FAIL、SIGTERM、supervisor終了code／signal、同一target・phase・word・typed schemaの不一致STOP、人工pipelineの15 callsを確認した。

除外2件は旧H/Pの消費済みfresh pathへcold preflightを要求する歴史的テスト。旧testを改変せず、新しい後続契約のcold passive gateで置き換えた。新テストは実分子配列や保存H4 QPYをdecodeしていない。

[Passive preflight](../../artifacts/research_c_stage_p_completion/2026-11-04/technical/preflight.json)はbytes／JSON／packageだけでPASS、科学call／配列decode 0。[SOURCE](../../artifacts/research_c_stage_p_completion/2026-11-04/technical/SOURCE.json)と契約をcommitへ固定した後、実際の空きCPU/RAM、環境、fresh条件を再確認して一回実行する。CPU/RAM/worker/wallへ恣意的上限を追加せず、元の実資源ベースprocess poolを使用する。共有venv・共有cache・他ユーザーのjobを変更しない。

今回のチャット指示でこの固定された後続範囲の実行は許可されている。承認原文保存・ID・証明書・新grantは作成しない。SOURCE/input/phase/target/column/shellのcorrectness FAILでは依存処理を止め、partialとSTOPを保存する。正常終了でも公開照合後mandatory STOPし、全law／校正／Stage C／CAや別wordへ進まない。
