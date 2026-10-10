# Track A H6精度一致資源比較 v1：一回実行認可と実行前seal

2026-10-11。利用者は公開commit `12183db780bef14c25c93ba6523c44c52edd7828` の[結果前契約](track_a_h6_matched_accuracy_execution_contract_v1.md)・sealed manifestに従う一回科学実行を明示認可した。今回の新grantはこの固定scopeだけを認可し、H8・追加探索・入力再生成を認可しない。

science sourceは `057ba97ba4db66775cf713c65cb6955e9dcb0d85`、[manifest](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v1/2026-10-11/sealed_manifest_v1.json) digestは `ed1841e008441a947ad76a91ee4b2157276fffd114426299a5e11179ec4cfcb7`。211 runtime source、親入力33記録、環境、plan、出力未使用を再照合した。source・候補・estimator・compiler・capsは変更しない。159件の準備時synthetic検査を再利用し、新しい計算を検査目的で重ねない。

[新authorization](../../artifacts/resource_applicability/track_a_h6_matched_execution_v1/2026-10-11/authorization_v1.json)、[launch preflight](../../artifacts/resource_applicability/track_a_h6_matched_execution_v1/2026-10-11/launch_preflight_v1.json)、[launch seal](../../artifacts/resource_applicability/track_a_h6_matched_execution_v1/2026-10-11/launch_seal_v1.json)を先にcommit/pushし、別bare repositoryからremote bytesを再取得して確認した後だけ起動する。grant bytesをSHA-256でpinする。この文書とsealは実行前記録であり、科学結果や工程完了を示さない。

対象はlinear H6/1.00Å/STO-3G/tol-only DF rank19/sector400、保存state、T0.8、B0 generation-prefix5/10/15・B1 S2/S4・B2の92 signal候補、ε_sig=0.05/0.01/0.005/0.001。q1/2/4/8、B2 r1/2/4、R=qr、K2/4、delta0.8/0.4/0.2/0.1。B3は登録normalization scalar診断のみ。signal先行後、適格候補の費用探索n8と登録選抜の独立確認n32、最大1708 measured wrapperを同じ固定規則で実施する。

signalはCPU IDs [0,2,5,6]・Numba/OMP4/BLAS1、costは[0,2]/[5,6]の2 disposable process・各thread1。phase/total wall capはnull。各worker AS8GiB、coordinator AS2GiB、live aggregate RSS18GiB、全output2GiB/log1MiB、call/instruction guardを維持する。grantと出力claimはone-shot、retry/resumeなし。旧pilotのSTOP・32/36 wrapper・欠測を保全する。

停止条件・数値allowance・shots・費用統計のscopeは原契約どおり。empirical uを精度証明にせず、sample mean/SEをµや正式winnerの保証としない。整合性・資源guardでSTOPしたら原出力・欠測を保存し、同一grantで再起動・source救済しない。

科学runが終わったら、保存結果・static byte/coverage audit・source/input/environment後照合・保存inventoryをGitHubへ公開し、remote再取得照合後にmandatory STOPする。科学的判断はGPTへ戻す。`H6_NOT_AUTHORIZED`は他のH6 stageの一般認可を与えないことを示し、今回の一回許可は別grantで管理する。`DRAFT_NOT_AUTHORIZATION`、H8未認可、`next_stage_authorized=false`を維持する。
