# O2-SCV224：採択済み一campaignの実行仕様

2026-11-07 JST。ユーザーが本条件の最小activation、人工検証、実H4 metadata評価一回、最大224 physical shots、保存・公開を明示許可した。承認証明書・grant発行は要求しない。旧契約・STOPには触れない。

## 固定対象

保存linear H4/1.00 Å/STO-3G/legacy DF12、8qubits/全4電子sector、L_D3、m0/q1/δ=.02/r1/K2/coefficient_atol0、X/Y、理想gate。
N1 `744e1a09661ebba6841fe5faf456f88c41c47780` actual finite-grid Q/selected full、N2 `5e9bd08068738cdf1070bdc43bee8194b113ae33` 保存保証state/witness、engine `fdf00cee3c87d804845497afc89ac6a71f3b3f59` のexact Fraction演算と保存strict誤差証明をそのまま使用する。
理論基準は元Hの厳密ground。保存保証stateとの数値的な差は既存engine保証の範囲で扱い、H_D/PF有効状態へ変更しない。
準備SOURCE `560c6092bfb5e71af8eedc4a4be5416922764f98`、設計 `84e021b6a8decde7fa3ed3fa44d7148cc5428061`、理論 `8d8a290fa4d4cf24f968aea27953ad5422e418b3`、旧752結果 `3c8e55a2728f86a1eda17a2e0105050a637e7c86`。

## 群選択と推定器

既存`scv_scalar.metadata_from_bytes/select_fragment_control`を実H4に一回だけ適用。正質量fragmentを文字列辞書順、非空proper subset、ν1∈[1/8,7/8]、|ν1−1/2|最小、同点はcanonical bit iの整数mask昇順。
S、全5層のν/E[u²]/Σ、input/law/catalog/control identityをtrain前に保存・凍結。群選択にbit・state・Born・旧pilot情報を渡さない。適格群なしは測定前STOP。
h1のみscalar intercept OLS、X/Y独立、rank2・finite slope・D1=2+|λ1|≤4、binary64 slopeをそのままFractionへ凍結。他4層λ=0を取得前から固定。
SCV=Σp_h[B̄_h−λ_h(ū_h−ν_h)]、B1-OL=Σp_h B̄_hを同じfresh評価recordで比較する。

| 段階 | X | Y |
|---|---:|---:|
| h1 train | 32 | 32 |
| h0/h1/h2/h3/h4 fresh evaluation | 4/64/4/4/4 | 4/64/4/4/4 |
| 計 | 112 | 112 |

metadata固定→X train32→Y train32→X/Y fit・凍結→X eval80→Y eval80→paired合成。
評価160件を共通使用、physical総上限224。一campaign、science retry0、追加campaign0。calibration/保証付きproductionなし。

## 最小接続と再利用

新`scv224_activation.py`と`run_scv224_science.py`のみを追加し、準備CLI・旧SOURCEは変更しない。
`connect_legacy_pipeline`、`Driver`、`LegacyCollector`、exact engine、saved-service loader、conditional sampler/Streams、価格provider、qualified cron/outer/manager/runtime/pool/worker/journal/reconcileを再利用する。
旧752用routingのdispatch/manager本体を、今回のpath・verify・semantic exitに限定したprocess内のscoped設定で呼ぶ。旧ファイル・旧jobは変更しない。submitも同じconsume-once cron手順。Restart=no、triggerをapplication開始前に撤去し、cron.service cgroupを確認する。
`OrderedDriver`の小さいacquire adapterはX/Y trainを先に両方取得し、検査済みY batchを一回のfitまで保持する。再測定・再課金なし。基底driverのfit/range/加重合成/会計/finalizerはそのまま使う。
metadata binderとsaved-serviceが作るsamplerはlaw/bank/a/b/patternの完全一致を検査後、metadata側の同一objectへ接続する。targetやCDFを作り直さない。

## stream・価格・call上限

Campaign `O2-SCV224-H4-m0-q1-20261107-v1`、purpose `O2_SCV224_PAIRED_DIAGNOSTIC`、seed **20261109**。
12 domain/24子stream、PCG64/SeedSequence、token辞書順、各path child→bit child。
A1/X,Y/train/1の2 domain、B1-OL/X,Y/prod/0…4の10 domain。tagは取得用。`prod`はfresh診断評価の旧schema名で、保証付きproductionを実行する意味ではない。
独立uniform53-bitは統計モデル仮定。PCG64の数学的独立性証明と呼ばない。
path≤64 word/trial、測定word1、合計≤14,560、一trial一claim。旧397/B1339/752のbit/係数/word/stream/cacheを混ぜない。

metadata群/ν/Σ一回、saved bundle decode一回、exact engine/Born/測定各≤224、selected full build≤448、compile≤224。
N1再生成、state solve/eigh/spectral、engine再認証、全実path列挙、未取得pathのcompile、float/GPU置換、別control/分子/roundは0。
固定価格はQiskit1.3.0、basis=(rz,sx,x,cx)、opt2、seed17、selected full no-prep。cache hitも全量子価格を課金。
実支出=train64+eval160。B1-OL参照=eval160、SCV参照=train64+eval160。参照二つの和を実支出にしない。
各軸Σp_h²s_B,h²/n_h、Σp_h²s_Z,h²/n_h、paired covariance、差のsample-based variance、rank/λ/range/feature count/all-equalと実現RZを保存する。
精度ε/失敗α・十分N・期待RZ改善・最適配分B1-OLへの優越・RPE energy精度の保証や科学的GO/NO-GOは行わない。

## 資源・保存・STOP

既存の二窓CPU idle admissionと最大30分の有限待機を再利用。恣意的なCPU/RAM/worker上限を設けない。資源待機不成立ならmetadata/測定前にSTOP、同attempt再投入なし。
他ユーザー/共有設定・process・fileを変更しない。実資源/pressure/cgroup/親worker終了を既存observerと読取専用観測で確認する。
bitを価格より先にdurable保存。STOP要求をworker回収前に保存。欠測bit/価格・未完了attempt・未測定古典費はUNKNOWN。
identity/source/環境/support/群/平均/stream/word/bit/価格/call不整合、rank/range失敗、worker/finalizer重大異常、非干渉不成立は条件緩和せずSTOP。
追加shot、seed変更、refit、control変更、科学retryは禁止。完了/中断いずれも結果を保全・公開照合してmandatory STOP。

独立worktree `/tmp/o2-scv224-science-20261107`、branch `codex/o2-scv224-science-20261107`。
科学output `artifacts/o2_scv224/2026-11-07/campaign_run01`、manager `manager_run01`、独立cache同output/cache。
元main/dirty/untracked、旧SOURCE/input/raw/結果/STOP/grant/ledger/markerとC1凍結を保全する。
