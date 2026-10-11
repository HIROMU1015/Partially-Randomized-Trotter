# Track A H6精度一致資源比較 v2：一回grant・起動・終了後公開のseal

2026-10-11。利用者が固定source `54986a9f54515c8c2289e38fa16fd81e2b6008c2` の[公開v2契約](track_a_h6_matched_accuracy_execution_contract_v2.md)とsealed manifestに従う一回科学実行を明示認可した。[authorization-only grant](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/authorization_v2.json)だけをcommit `2157da1d208a04102166fa1ae3c154490546c90f` で公開した。grant SHA-256は `a006c3b933db79d372a09366aeb0e7ae508842e96bd3d786e7d6fb9729fd2e39`。

[local preflight](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/local_preflight_v2.json)と[remote起動前照合](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/remote_prelaunch_verification_v2.json)で、science source216 path、親入力33 path、環境・資源条件、未使用出力を確認した。GitHubのgrant commitを独立bare repositoryへ再取得し、253 pathのSHA-256を照合した後、固定runnerを一回だけ起動した。[process起動観測](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/process_start_observation_v2.json)は外部ホスト観測であり、科学精度の証明ではない。

linear H6／1.00 Å／STO-3G／tol-only1e-8のsigned generation-order DF rank19／sector400／採用済みweighted Hermitian projectionと保存state／T=0.8。B0/B2 prefix5/10/15、B1 full19 S2/S4、q1/2/4/8、B2 r1/2/4・R=qr・K2/4、delta0.8/0.4/0.2/0.1、ε_sig=0.05/0.01/0.005/0.001。科学plan、92候補、shot会計、独立exploration/confirmationの選択・seed規則、最大1708 measured wrapperを変更していない。旧v1を再実行していない。

signal CPU [0,2,5,6]・Numba/OMP4／BLAS1、costは[0,2]／[5,6]の2 disposable process・各thread1。coordinator AS soft2 GiB／hard8 GiB、signal/cost AS soft/hard8 GiBを実receiptで確認した。RSS18 GiB、output2 GiB／log1 MiB、call/instruction guardを維持し、phase/total wall capはnull。GPUへbackendを変更していない。source準備時の176 local synthetic PASSを再利用し、追加検査目的の科学runは行っていない。

この文書保存時点では**runは継続中**。92 signal recordと490組×3 primitive probeの記録は保存済み。固定の適格性規則がexploration 592 groupを登録した。これは上限596に収まる実登録数であり、候補追加ではない。[partial checkpoint](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/running_checkpoint_v2.json)はTASK_COMPLETEのworker directoryだけを抽出した不変の途中証拠。最終coverage・confirmation・資源優位の結論ではない。進行中のworkerのprivate outputや最終inventoryを先取りして公開しない。

終了後の公開だけを行う[publication helper](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/finalize_matched_run_v2.py)を準備した。このhelperは科学runnerを起動せず、PID/start identityで終了を待ち、既存の静的auditor、原結果hash・欠測・外部事後identity・保全記録を作成する。科学source216 pathを変更せず、独立確認候補を自分で追加・再選択しない。元terminalやfreeze・過去の監査は改変しない。新しいtests・数値検証結果を科学的証拠として追加しない。

helperは既存HEAD/indexと[開始前保全baseline](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/baseline_before_execution_v2.json)が想定と異なる場合、他の作業を上書きせず公開処理を停止する。許可された新raw/metadataと今回の文書追記だけを明示的にstageし、dirty文書ではHEADから今回の追記だけをstageする。通常commit/push後、別bare repositoryへGitHubから再取得して原結果・source・入力のbytesを検証し、その証拠を公開・再取得する。`git add .`、force push、科学的retry/resumeの経路はない。公開処理に障害があっても科学runを再実行しない。

原結果・欠測・監査の公開とremote再取得が終わった時点でmandatory STOPし、GPT・利用者へ判断を戻す。現時点では結果公開の完了を主張しない。一般の`H6_NOT_AUTHORIZED`／`DRAFT_NOT_AUTHORIZATION`と`next_stage_authorized=false`を維持し、今回の固定run一回だけを別grantで許可している。H8、追加探索、旧v1再実行、入力生成、その他の科学計算は未認可。以下の「準備時は未認可」という既存文書・manifestは当時の記録として保存する。
