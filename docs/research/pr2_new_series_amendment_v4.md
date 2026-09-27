# PR-2 new-series amendment v4：snapshot再基準化V1–V3

日付：2026-09-28  
series ID：`pr2-rebaseline-de7a5492-v1`  
specification base commit：`bf5d3b2405b2bec4f88dd0589b75dd737b13e052`  
実行許可：V1–V3のみ  
V4/S1′実行許可：なし

## 1. 根拠と位置づけ

本amendmentは次の外部レビューと作業パッケージを受け、旧入力回収不能というV0判定後に結果前固定する。

- `docs/research/pr2_s0_review_and_research_reset_bf5d3b2_20260928.md`  
  SHA-256：`94b82d5a373721e270c62c846a2f76f50d9b236806a8da8b755a9f9f96fd4202`
- `docs/research/pr2_codex_restart_workpackage_bf5d3b2_20260928.md`  
  SHA-256：`fcad5959db8136eb4d0f68017814464d06a1a6fbd9648988789c1ec91e2a7d7f`
- `docs/research/pr2_v0_input_recovery_audit_bf5d3b2_20260928.md`

旧系列の`STOP_INPUT_REPRODUCTION_MISMATCH`、`S1_authorized=false`、expected hash `d8b4aaf2…a3dc3`、observed hash `de7a5492…12424`は変更しない。新系列は旧pilotの継続data claimではなく、新しい固定入力で同じPR-2 RQを検査する別系列である。PR-2仮説は現時点で未検証である。

## 2. 固定入力と情報開示

### 2.1 Development

- path：`artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz`
- raw file SHA-256：`3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`
- Hamiltonian hash：`de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`
- model：linear H4、1.0 Å、STO-3G、DF rank 12
- sectorとして固定する事実：8 qubits、N=4、N_alpha=2、N_beta=2、S_z=0
- 「singlet」はN_alpha=N_betaだけから証明済みとは記述しない
- 保存shape/dtype：constant `()/float64`、one_body `(8,8)/complex128`、lambdas `(12,)/float64`、g_matrices `(12,8,8)/complex128`、sector indices `(36,)/int64`、full state `(256,)/complex128`、sector state `(36,)/complex128`

既知情報は旧S0のinput hash、energy/lambda差、保存metadataである。rank別candidate signal、compile、trajectory performance、resource順位はこの入力では未評価である。

### 2.2 Held-out

- path：`artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p30_rank12_held_out_v1.npz`
- raw file SHA-256：`ad7e3e7165c55dbaa395eef7a1dd74db89e1f7ab29a69ac64333f4aebf8b3e37`
- V1–V3で許す操作：path存在確認とraw file SHA-256照合のみ
- 禁止：NPZ内部配列・metadataのload、signal、cost、ranking、candidate適格性の評価、再生成

## 3. 固定する研究対象

PR-2 RQは、固定DF Hamiltonianに対するgeneration-prefixまたはweight-prefixの決定論部分とrandom residual補完の構造が、後続の公平な同一task比較へ接続できるかである。

- reference rank：12
- primary prefix rank：6
- control prefix ranks：3、9
- method labels：`B2-G` generation prefix、`B2-W` weight-ranked prefix
- identity policy：`extract_identity_phase`
- coefficient tolerance：0.0
- weight rule：既存`rank_df_fragments`の登録済みrule
- rank、method、partition policyはhash mismatchを理由に変更しない
- G/Wのordered indicesが全rankで同一なら重複候補を統合する。異なる場合は別候補として保持する
- prefix境界をまたぐ縮退factorの差を自動的に無害なgaugeとして消さない

V1–V3は構造・実装妥当性の検査であり、resource比較、最適rank選択、論文のwinner判定ではない。

## 4. V1：snapshot integrity・model validation

V1は保存済みdevelopment snapshotだけを二回独立loadする。分子build、SCF/DF、再保存は禁止する。

### 4.1 事前固定する検査と閾値

1. raw file SHA-256が§2.1と完全一致する。
2. 二回のloadでHamiltonian hash、sector hash、state hash、全array hashがmetadataおよび相互に完全一致する。
3. key、shape、dtypeが§2.1と完全一致する。
4. constant、全arrayの実部・虚部がfiniteである。
5. one-bodyと各g matrixのHermiticity relative Frobenius residualがそれぞれ`<=1e-12`。
6. full stateとsector stateのnorm errorがそれぞれ`<=1e-12`。
7. full stateのsector外max amplitudeが`<=1e-12`、sector成分とsector stateのmax absolute differenceが`<=1e-12`。
8. 固定sector dense Hamiltonianに対するRayleigh residual `||H|psi>-E|psi>||_2`が`<=1e-9`。
9. metadataがlinear H4、1.0 Å、STO-3G、rank 12、N=4、N_alpha=N_beta=2、S_z=0と一致する。

Rayleigh energyは記録するが、旧pilot energyとの近さを旧入力同一性の受理条件にしない。canonical hashを新たに導入しない。raw file hashと既存layer hashを保持する。

### 4.2 V1 status

- 全項目通過：`V1_PASS`
- file/hash/metadata/shape破損：`STOP_V1_SNAPSHOT_INTEGRITY`
- Hermiticity/state/sector/eigen residual不適合：`STOP_V1_MODEL_VALIDATION`

## 5. V2：partial構造の決定論的検査

V2はV1通過時だけ、同じdevelopment snapshotからrank 3、6、9をこの順に検査する。

各rank・methodについて次を記録する。

- deterministic ordered indices、unordered set、randomized indices
- deterministic fragment別index、lambda、fragment hash
- partition hash、tail hash、preparation hash
- exact RTE `lambda_R`
- sampling coefficientの符号、絶対重み、probability
- probability sum error
- tail identity phase
- full H、H_D、抽出H_Rを含む`H_D+H_R=H`再構成relative spectral error

### 5.1 事前固定する閾値

- partitionがdisjointかつ全12 fragmentをexact coverする
- deterministic件数が指定rank、randomized件数が`12-rank`
- component probabilityはfiniteかつ非負
- nonempty tailのprobability sum error `<=1e-12`
- sampling coefficient signと元coefficient signが全componentで一致
- `H_D+H_R=H`のrelative spectral error `<=1e-10`
- identity phaseとtail hashが二回構成で完全一致

G/Wはordered tuple、unordered set、H_D spectral difference、H_R spectral differenceを別々に比較する。全rankのordered tupleが同一なら`collapse_B2_G_and_B2_W=true`、一つでも違えばfalseとする。違い自体は失敗ではない。

### 5.2 V2 status

- 全rank・method通過：`V2_PASS`
- partition、probability、sign、identity、再構成の不整合：`STOP_V2_PARTIAL_STRUCTURE`

## 6. V3：記録実装と限定unit tests

V3では新系列専用module、runner、testを追加し、旧S0 artifact/moduleの値を書き換えない。

必須事項は次である。

- `input_files_read`、`snapshot_loads`、`molecular_calculations`、`operator_reconstructions`、`signal_evaluations`、`wrapper_probe_trajectories`、`candidate_trajectories`、`circuits_compiled`、`quantum_shots`を別counterとして記録
- V1–V3で期待する値は、development load 2回、held-out raw hash照合1回、molecular/signal/trajectory/compile/shotは0
- build成功metadataとoperator/controlの数値検査を別fieldにする
- raw input改変検出、出力artifact上書き禁止、V4非承認guardをtestする
- rank 3/6/9のpartition/reconstruction/probability/sign/identity規約をtestする
- corrected estimator、finite truncation、shot式の既存testは旧moduleの正しい規約を再利用し、新V1–V3 runnerでは実行しない

V3のtestはtoyまたは保存済みdevelopment配列の決定論的処理に限る。testが生成した一時ファイルは研究artifactではない。

## 7. S0′統合判定と停止

V1–V3のすべてが通過した場合の統合statusは`S0_PRIME_PASS_V4_REVIEW_REQUIRED`とする。これはS1′/V4の実行許可ではない。

必ず次を満たす。

- `V4_authorized=false`
- `S1_prime_authorized=false`
- `automatic_next_stage=null`
- `held_out_signal_cost_ranking_evaluated=false`
- `S2_executed=false`
- `S3_executed=false`

V1またはV2が失敗した場合は対応するSTOP statusを保存し、それ以降のstageを実行しない。task、target、rank、thresholdの変更が必要なら新しい結果前amendmentなしに変更しない。

## 8. 明示的に未承認の処理

- V4/S1′のsignal、control、wrapper probe、compile smoke
- expected-cost Monte Carlo、32/128 trajectory
- resource winner、総費用、shot優位性の判定
- S2/S3、長RPE、precision/rank/geometry sweep
- held-outのNPZ load、signal/cost/ranking評価
- 旧pilot数値のsame-input baselineとしての再利用
- PR-3/PR-4/PR-5/PR-6への自動移行

S0′ packetを独立確認した後、V4を行う場合は別の明示的authorizationを必要とする。

## 9. 成果物とprovenance

V1–V3のmachine-readable resultは新規pathへ一度だけ書き、上書きしない。resultには、specification commit、source commit、実行時dirty-worktree状態、input file/layer hash、各検査値、counter、未実行項目、deviation、result fingerprintを保存する。

specification/amendmentを先にcommitし、sourceを別commitで固定した後に実行する。実行結果commitはさらに分離し、manifestが自分自身のcommitを要求する循環を作らない。
