# H6 full m5_best：失敗した1構成の再計算・診断保存run01起動

2026-10-19 13:56 JST snapshot。**ユーザー承認に基づき一回起動済み、native回路構築中。完了未確認。** 旧runの部分STOPを保持したまま、H6 full m5_best/p19/q2の1構成だけを別SOURCE・別grant・別ledger/outputで再計算している。成功済み58軸は再取得しない。

SOURCE：`f743a88225f7cb2c8d2c91d7e19813a5780f803b`。branch：`track-a-h6-m5-full-recovery-20261019`。元science SOURCE：`c7d53f04c3be131078ee91e8bef3c78ca434ed3e`、元結果：`b38898429af3a2a48d7057d9d828256c057685f8`、静的調査：`232ec931e2ebad93e85591205b888795223c828c`。[元STOP](track_a_h46_m5_tol_pilot_run01_results_20261019.md)、[静的調査](track_a_h46_m5_compile_stop_static_analysis_20261019.md)、[今回summary](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01_start/2026-10-19/summary.json)。

## 1. 承認範囲と不変の科学条件

ユーザーの「それだけ再計算して」と継続指示を、固定1構成・最大2軸の新しい一回実行承認として記録した。[承認projection](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01_start/2026-10-19/approval_projection.json)、[grant projection](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01_start/2026-10-19/grant_projection.json)、[固定plan](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01_start/2026-10-19/plan.json)。plan SHA256は `ffe5d880ecdebdc6616a0e4a41bd3432115503e2feba16f2414a081a2ac098da`。

H6直鎖1 Å/STO-3G、保存tol-only1e-8（rank19は観測値）、full PF/prefix19/q2、T=.8、P=0。m5は元binary64版 `PF_M5_BEST_20D64C2_BINARY64_V1`、11 signed S2 piece、symmetric-directional、scalar一回、full Hadamard cosine/sine。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17を維持する。

cosineを一回compileし、元の全検査にPASSした場合だけsineへ進む。**各errorがfiniteかつ1e-9未満**を維持する。cosine FAILなら証拠を残してSTOPし、sine予約・compileは0。原因調査用の位相調整値をPASS判定へ流用しない。

旧59予約は保全し返却しない。新scopeは追加最大2軸で、累積予約は最大61。旧runのSTOPを新runの結果で書き換えない。新PR標本・SCF/DF/state生成・新full-H reference・候補探索・shot会計・再fit・ランキングは行わない。固定1点のsignalとnative/occupation/Hadamard正しさ検証は元の処理を再実行する。これは新signal探索ではない。

元H6 full-H referenceを元SOURCE・receipt・NPZ SHA・state/input/preparation lineageで結合して再利用する。参照位相を合わせ直さず、新reference solverを呼ばない。主比較へのm5採用AD03・後続Coreは今回も保留。

## 2. 新adapter：元science式を保持し、判定前に証拠を保存

新実装は `h6_m5_recovery_contract.py`、`h6_m5_recovery_native.py`、`prepare_h6_m5_recovery.py`、`launch_h6_m5_recovery.py`。旧16 science fileを変更していない。元 `_run_task` のASTを再利用し、変更は共有referenceのSOURCEを元SOURCEへ結合する箇所と、診断保存3hookだけ。hookとlineage変更を取り除いたASTが元と完全一致することを人工testで確認した。数値不一致を解消するcompiler/係数/シミュレータ修正を行ったわけではない。

新しい専用quarantineに次を保存する。

1. compile前：元wrapper QPYと初期lowベクトル、SOURCE/plan/fingerprint/global phase。
2. compile後・数値受入前：compiled QPY、global phase/layout、compiler identity、未検証integer費用。
3. ベクトル比較前：before/after/初期low/独立expected作用のNPZとSHA。raw差・norm・内積・枝別差・Hadamard値・診断専用の共通位相調整後差。

全quarantineは `UNVALIDATED_NOT_FOR_ACCOUNTING`。原受入規則にPASSした後だけ通常のcost fileを作る。位相調整は診断用の別計算で、保存raw vector・full-H reference・signal・受入規則を変更しない。nonfiniteや保存失敗もSTOPし、欠測費用を0補完しない。

QPY/NPZは専用partial fileへ書き、flush/fsync・hash・排他的な最終名への確定を行う。保存失敗で既存証拠を上書きしない。今回も数値不一致が再現した場合、数値不一致はそのまま記録し、保存証拠を使って後続の切り分けを行えるようにする。

## 3. 人工検証・封印

[検証summary](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01_start/2026-10-19/validation_summary.json)：最終15人工testがPASS、wall3.533秒。初回14件PASS後に、実compilerの人工回路往復と終了照合を補強して最終15件を再確認した。14+15を独立29項目とは数えない。実H6入力や新分子計算をtestで使っていない。

人工testは、元science ASTの一致、phase-only raw FAILをFAILのまま保持、cosine FAILでsine抑止、sine FAILでcosineだけ保存、実toy transpile/QPY roundtrip、nonfinite保存STOP、disk失敗時のcompile抑止、spawn exit/reap、無効grantのCLI拒否を含む。別reviewerの独立検証ではなく、作者自身の技術確認である。

実configのmetadataを複製して候補追加・prefix変更・位相受入・rank/reference/native policy/origin SOURCE/sealの8種を変え、全て拒否した。元configを変更していない。無効grantで実CLIがexit2かつrun root未作成になることも確認した。

SOURCE20 file、既存provider/runtime/input/seed、原STOP・終了証拠、保存reference、新しいQPY runtime10 fileを計330 fileへ封印した。QPY sourceは固定Qiskit1.3.0のRECORD SHA/bytesにも一致。[closure projection](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01_start/2026-10-19/source_runtime_closure_projection.json)。元science16 file・元grant/ledger/marker/STOPを含む629保護fileは不変だった。

## 4. 起動・資源・非干渉

13:55:17 JSTに一回dispatchし、fresh gateを通過した。観測は有効affinity64、idle CPU約61.07相当、空きRAM約469 GiB、空きdisk約458 GiB。[fresh gate](../../artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01_start/2026-10-19/fresh_gate.json)。対象は1点、軸は依存順序があるため、productive workerは1。旧30/24 workerを恒常capにしているわけではない。内側thread/compiler並列設定は元の条件を維持し、共有環境を変更しない。

入場時のRAM/disk計画proxyは同一点の旧観測peakを3倍した約8.44 GiBで、runtime上限や厳密な必要量の上界ではない。host PSIやCPU競合だけでworkerをkillしない。実OOM・継続不能な不足・own異常・input/SOURCE破損・数値不一致は保護を維持する。

保存full-H referenceと既存preparation/fingerprint cacheを再利用し、58軸の重複計算を避ける。新しい全探索・重い高速化改変は入れていない。他worktreeのHEAD/statusを凍結せず、他chatへ書込・signalを行わない。

private提案・実行原本は `/home/AbeHiromu/projects/h6-m5-full-recovery/20261019/private/` 配下に保存。元proposal/INVALID grantはそのまま残し、今回承認は別authorization_run01で有効grantへ結合した。元run01の629保護fileは変更していない。

## 5. 現在の状態と次の担当

13:56 JST snapshotはnative_build、compile予約0・受理費用0。**再計算は進行中で、数値不一致の解消・両軸PASS・native exitをまだ確認していない。** バックグラウンドで継続する。launcherは終了時にworker exit/reap・SOURCE/input・結果を確認してcompletion witnessを保存し、外部wait4もnative終了code・reap・wall/RSSを記録する。

**Codex（終了後）**：新runの終了・回収・新旧SOURCE/ledger/quarantineの整合を確認し、PASS/STOPのどちらでも保存結果を報告してlocal commitする。追加のsampling/compile/自動retryは行わない。

**ユーザー（結果確認後）**：保存された診断と受入結果から次の対応を判断する。閾値変更・別scopeの再compile・主比較採用・後続Coreは今回の実行へ含めない。

今回の起動資料はlocal commitする。公開pushは行わない。計算終了後はmandatory STOPし、次Phaseへ自動移行しない。
