# Track A H6精度一致・測定込み資源比較：補完v3 起動時STOP

状態：**H6_MATCHED_SUPPLEMENT_V3_EXTERNAL_PRECLAIM_LAUNCH_STOP / mandatory STOP**。
科学runnerを一回起動したが、出力先の親directoryが存在せず、`out.mkdir(exist_ok=False)`で`FileNotFoundError`、exit1。
Codexのlaunch準備で親directory作成を漏らした。Phase Aのbuild/fingerprint/transpile/metricには未到達。
再起動・retry/resume・source修正・上限変更は行っていない。これは新しいMemoryErrorの証拠ではない。

## 固定identityと一次記録

Repository `HIROMU1015/Partially-Randomized-Trotter`、branch `track-a-ax2b-h4-post-review-20261010`。
準備公開commit `604e593e184903cfb7e34c118a153251783d9cba`、science source `5a903aea2492c2d211c1e15e7737df0d50694c12`。
[凍結runner](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5a903aea2492c2d211c1e15e7737df0d50694c12/scripts/resource_applicability/run_track_a_h6_matched_v3.py#L40)、
[固定契約](track_a_h6_matched_supplement_execution_contract_v3.md)、[sealed manifest・reuse registry](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v3/2026-10-11/sealed_manifest_v3.json)。
manifest digest `630179bfdfd6904f7e94746b5ca86470e0df49b49ee8409cdd66f6e065fec482`、SHA256 `7796f92d442207a985fe2594b2d88ce74694d69a4e041e4188a05a16e5200f4c`、reuse digest `c82aa9223c3f67f52085cf821547b92c37b2e110c78e98c2bb256f7c521ead70`。

[authorization-only grant](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/authorization_v3.json)はcommit `c83e55095aaeb2a758826427ffec873e93d5c2bf`、
SHA256 `b467d053d93b104223ff1ac3c99a14b647f831fcd59d9d41087995cbf10eadda`。事前にGitHubから独立bareへ再取得し、source/validation/旧raw/入力/主要文書/grant 5,486 pathのbytesを照合。
[preflight](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/local_preflight_v3.json)、[remote事前照合](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/remote_prelaunch_verification_v3.json)。

主証拠は[原stderr/stdout](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/runner_stdout_v3.log)、[一回起動command](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/launch_once_v3.json)、
[process起動記録](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/process_start_observation_v3.json)、[exit記録](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/runner_exit_v3.json)、
[外部観察status](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/execution_observation_v3.json)。外部観察は凍結runnerのterminalではない。
出力leafを作る前の例外のため、新raw namespace・execution claim・worker terminal・ResourceTrace・execution inventoryは存在しない。
過去の科学結果に新statusを上書きしていない。

最初のdetached補助ownerはsandbox namespace終了で回収され、runner未起動・出力未使用をhost側で確認した。
[その記録](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/launch_transport_observation_v3.json)と[owner起動receipt](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/owner_launch_v3.json)を保持。
foreground session内のownerが行った固定science runnerの起動は一回のみ。元の補助processと科学runnerの回数を区別する。

## 科学条件とcoverage

linear H6、1.00 Å、STO-3G、12 modes、alpha3/beta3 sector400、保存state、T=0.8。
tol-only DF 1e-8/rank19、cutoff0、signed generation orderと採用済みweighted Hermitian projectionを維持。
B0/B2 prefix5/10/15、B1 full19 S2/global Yoshida S4、q1/2/4/8、B2 r1/2/4・K2/4。
92候補、epsilon_sig 0.05/0.01/0.005/0.001、数値allowance・normalization・shot式・seed/選抜・compilerは未変更。
数値allowanceは経験的で厳密certificateではない。費用の母平均winner、PR優位、ground-state/energy精度は未認定。

| 項目 | 旧v2受理済み | v3新取得 | 現在の欠測 |
|---|---:|---:|---:|
| signal | 92 | 0 | 0 |
| B3 normalization row | 84 | 0 | 0 |
| 原登録探索group | 449 / 592 | 0 | 143 |
| 探索wrapper | 902 | 0 | missing探索分286 |
| 独立確認 | 未選抜 | 0 | 未登録、最大8構成×32 trajectoryは未開始 |

[missing全一覧・旧group/source対応](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/missing_coverage_v3.json)。
原index447 `B1_p19_q8_4th`もworker未起動。旧447のTASK_STOP・450の未完了・451〜591の未dispatchを保全。
旧449は再計算していない。旧result commit `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb`、source `54986a9f54515c8c2289e38fa16fd81e2b6008c2`。
[旧結果・監査](track_a_h6_matched_accuracy_result_v2.md)、[旧原terminal](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/execution_terminal.json)。
原STOP `H6_MATCHED_RESOURCE_STOP` / `RuntimeError:MemoryError:`はそのまま。
旧中断attempt総数はnull、不明のまま。今回のcompile/trajectory/signal/DF/state呼出しはrunnerの制御フローとpreclaim例外から0。

全592探索が揃っていないためresource map・候補選抜・独立確認・新SD/SE・order診断・allowance感度mapは作成していない。
未観測値をproxyや0費用で埋めない。新しい科学的結論はない。

## 資源・保存証拠の監査

契約はcoordinator AS soft2/hard8 GiB、cost worker8/8 GiB、CPU [0,2,5,6]、最大2 process/threads1、aggregateRSS18 GiB。
phase/total wall cap=null、GPUなし、compiler basis rz/sx/x/cx・opt1・seed17。これらの固定値は変更していない。
ただし今回は`orchestrate`以前の失敗のためcoordinator limitsもworker limitsも未設定で、worker memory peakは欠測。
実H6 S4 q8の8 GiB feasibilityを検査できていない。

[事後静的identity監査](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/static_identity_audit_v3.json)では新runtime224/validation5、旧source216/旧raw5220、親入力33、
manifest/grant、environmentが実行前のidentityと一致。凍結v3 raw auditorは入力output自体が存在せず適用不可、未呼出し。
これを数値・coverage PASSと呼ばない。[保全監査](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/preservation_v3.json)、[公開証拠inventory](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/execution_evidence_inventory_v3.json)。

一回のrunner起動認可は使用済みとしてretireする。ただしleaf作成に失敗したためexclusive output claimおよびscience claimは作られていない。
grantの原bytes・消費条件の文言は改変せず、この二つを[観察記録](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/execution_observation_v3.json)で別記する。
**H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION / next_stage_authorized=false / H8_authorized=false / mandatory STOP**。

## GPT・利用者へ戻す事項

新しいH6 mapは存在せず、PR優位性・H8移行を判断する材料は増えていない。
今回の起動失敗はlaunch準備の問題であり、旧MemoryErrorの根本原因を説明しない。
将来の再準備では、既存sourceを変える必要があるかとは別に、親directoryの存在・leaf未使用を起動前に検査する必要がある。
ここでは変更・新grant・再実行を認可しない。次の具体的な準備や実行は利用者の指示へ戻す。
公開後のremote bytes/hash確認は[公開照合receipt](../../artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/remote_result_verification_v3.json)を参照。
