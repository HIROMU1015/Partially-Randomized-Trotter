# M1-S独立correctness：別途実施承認とone-shot実行計画（2026-10-16）

ユーザーの最新発言「次の作業に進んで大丈夫」を、直前に対象を特定した独立one-shot科学計算への別途実施承認として記録する。
旧推奨だけを承認とは扱わず、[発言と直前文脈](../../../artifacts/track_b_m1s_independent_execution_preparation/2026-10-16/v1/user_scientific_execution_approval.json)を保存する。
[正式契約](m1s_gate0_v5_independent_correctness_contract_adopted_20261016.md)自体のcontract_grants_execution=falseは変更しない。
新しい[専用authorization](../../../artifacts/track_b_m1s_independent_execution_authorization/2026-10-16/v1/execution_authorization.json)を別commit Aに固定し、採択契約C3ec5721b、SOURCE4787cf90をhashで結ぶ。

H4 linear1.00 Å/STO-3G/8 spin orbitals/保存DF rank12・G3/G6/G9、deltaなし、保存数値sector oracle、v5比較結果cdcb7f6f。
occupation/CAR/全Pauli trace/tensor再構成と整数/有理数W-v-rhoを一回だけ計算し、全独立reference固定後に旧v5を照合する。
採択精度・閾値・モデルを変更しない。未保存full-H/HDの全scope留保はUNDETERMINED、FAIL優先、available scopeを別報告する。

[環境・性能preflight](../../../artifacts/track_b_m1s_independent_execution_preparation/2026-10-16/v1/runtime_and_performance_preflight.json)はraw bytes/metadata/静的構造だけ。
CPU1 worker、GPUなし、各thread1、2 GiB、result/log各128 MiB、5億primitive events、integer bit65536、時間上限なし。
元SOURCEのcache/整数演算を維持し、並列workerや精度・算術経路を変更しない。speedup・runtime成功は実測前に主張しない。
最初の数値検証は同じ一回内のexact代数・基底・状態・trace整合gateで行い、追加の模擬/実入力テストは起動しない。

[実行計画](../../../artifacts/track_b_m1s_independent_execution_preparation/2026-10-16/v1/execution_plan.json)に従い、clean A・commit順・全hash・runtimeを再確認後、外部排他markerを作り子runnerを一回だけ起動する。
正式one-shotはoutput directoryの排他予約で消費。失敗/未決でもdirectory、context、log、result、markerを保存し、自動retryしない。
旧v4 STOP/null・v5 raw/消費済marker・既存SOURCE/契約・親/他worktreeは保全する。旧v5再実行・研究scope拡張は含めない。
保存後は新しい科学処理を追加せず、保存監査と公開を終えて**mandatory STOP**。
