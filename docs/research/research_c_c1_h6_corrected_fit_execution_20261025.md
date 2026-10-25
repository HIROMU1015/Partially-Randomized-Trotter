# 研究C・C1：H6 rank14・修正codecの限定1 fit実行

2026-10-25 JST。固定科学SOURCE `33432e78659b8b3e654e39f1392328cb0ba5c487`、branch `codex/research-c-c1-h6-corrected-fit-execution-20261025`。

**登録20/20 logical unitsを完了し、CDF fitは1回、retry/resumeは0。初期同値性・optimizer成功・終了DF予算はすべてPASS。修正CDF14のsector誤差は1.0689213237e-4 Ha、guard込み1.0803058514e-4 HaでCA/8以内。旧CDF14よりsector誤差が93.9132%小さい。一方、保存XDF初期値に対してはtensor残差が減少してもsector誤差は4.3116%増加した。この1 fitからPF・回路・総資源の結論へ拡張しない。公開照合後mandatory STOP。**

[実行要約](../../artifacts/research_c_c1_h6_corrected_fit/2026-10-25/execution_summary.json)／[4状態比較](../../artifacts/research_c_c1_h6_corrected_fit/2026-10-25/state_comparison.csv)／[要約gate](../../artifacts/research_c_c1_h6_corrected_fit/2026-10-25/gate_summary.csv)／[科学call](../../artifacts/research_c_c1_h6_corrected_fit/2026-10-25/calls_summary.csv)／[原本・生x/jac・J/U・STOP](../../artifacts/research_c_c1_h6_corrected_fit/2026-10-25/run01/)／[postflight](../../artifacts/research_c_c1_h6_corrected_fit/2026-10-25/postflight.json)。

## 対象と固定条件

ユーザーが固定契約に基づくrank14の新規1 fitをチャットで明示許可した。契約の歴史的なunapprovedファイル名・metadataは変更せず、承認原文・承認証明・新grantは作成していない。独立worktree `/tmp/prt-c1-h6-corrected-fit-science-20261025`、fresh `artifacts/research_c_c1_h6_corrected_fit/2026-10-25/run01` を使用した。

直鎖H6・1.00 Å・STO-3G、中性singlet、Nα=Nβ=3／sector400。通常number CDF、actual rank14、`real_schur_column_gauge_v1`、ffsim0.0.83、L-BFGS-B、tol1e-8、options=None／callback=None／jac=True。元g補正κ・scalar・元frame順・既存guardを維持した。PFのsplit L_D・時間窓は非適用で、PF・native回路・shots・費用は評価していない。

保存XDF14を過去に検証済みの同条件initializerとして再利用した。新Cholesky分解・rank探索は0。`XDF_full[:14]`との成分照合でJ/U差は両方0、before snapshotは元XDF14とbyte同一。旧CDF終了結果は初期値として使っていない。

| 固定資料 | identity |
|---|---|
| [契約](research_c_c1_h6_corrected_fit_contract_unapproved_20261025.json) SHA256 | `996bb93c44d42a10108dd49ba3ea70dbd5d4dd32e315748d6146dc22b552e55d` |
| [SOURCE manifest](../../artifacts/research_c_c1_h6_corrected_fit/2026-10-25/technical/SOURCE.json) SHA256 | `bf1b942618301ef5a9c8fc466b746bef9ef115e0a631ed57e870e1671c848b14` |
| 元H6 input SHA256 | `b82ee9864d9c76071944834ae32d3d6eea7e95959708c2e19bca011936bce447` |
| 保存XDF14 SHA256 | `559dc43bf923a421ccc62d05c4cefa0bdf39a48ba68824abf3c8f770c04642b1` |
| 元入力・XDF SOURCE | `33c380d8aa792e5119bb3adec5dd07d1e1377214` |
| 初期値identity診断 SOURCE | `14cf35589a8501ef5db9a33700ed0c98bc1aa853` |
| 前段round-trip SOURCE | `43941ed8fbb0f12f0b028b2c379b872f5ba50f75` |

科学SOURCE47 files・入力18 hash・固定ffsim util/fitter・両環境のpackage versionsを実行前後で照合した。SOURCE、runner、依存lock、数値allowanceに変更はない。実行前の人工接続15＋runner12 testsがPASS。[準備時の人工161 tests＋29 subtests](research_c_c1_h6_corrected_fit_preparation_20261025.md)もそのまま保持する。

## 最適化前の同値性

修正codecのJ差0、最大density projector差3.3306690739e-15、tensor round-trip F5.6291061954e-15（固定allowance2.7729823971e-10）。before／initialのsector operator差はmax-entry7.5495165675e-15、F9.1776080238e-14、spectral norm1.2411077755e-14 Ha。max-entry gateは既存1e-10／rtol0を保持し、これをoperator normの厳密上界と扱っていない。

before／initialともsector norm＋guardがDF予算内であることを確認した後だけ、fit workerを開始した。実ffsim fitterのpackが返す504成分のx0は資格済みx0と完全一致した。prepare時とfit経路内のcodec診断JSONもbyte一致。旧codecへのfallbackは0。

## 誤差とDF予算

| 状態（すべてrank14） | tensor残差F | sector operator norm Ha | guard Ha | norm＋guard Ha | CA/8 |
|---|---:|---:|---:|---:|---|
| 保存XDF・変換前 | 7.7299273070e-5 | 1.0247385485e-4 | 1.1384527782e-6 | 1.0361230763e-4 | PASS |
| 修正codec後・optimizer開始値 | 7.7299273070e-5 | 1.0247385485e-4 | 1.1384527782e-6 | 1.0361230763e-4 | PASS |
| **今回のCDF終了値** | **6.4294638171e-5** | **1.0689213237e-4** | **1.1384527737e-6** | **1.0803058514e-4** | **PASS** |
| 旧CDF終了値（保存診断のみ） | 8.7489708782e-4 | 1.7561186635e-3 | 1.1952370989e-6 | 1.7573139006e-3 | NOT_MET |

予算は固定CA/8=`0.00019875 Ha`、guardは既存`sector_operation_scale_guard8_proposal_v1`。元physical h/g/cのsector400 Hamiltonianと同じCAR/contextを再利用し、before／initial／finalを登録した3回の`rank_assessment`で評価した。旧CDFの行は[保存診断](../../artifacts/research_c_c1_h6_initialization/2026-10-25/run01/)から引用し、新しい旧結果のnorm計算はしていない。比較元scalarのhashもpostflightに保存した。

初期値からtensor残差Fは16.8237%減少し、tensor objectiveは9.2068855986e-10減少した。一方sector誤差は4.4182775170e-6 Ha（4.3116%）増加した。最適化するtensor Frobenius目的関数と物理sector operator normは異なる評価量であり、この実行でも両者の増減は一致しない。増減は数値点比較であり、厳密な符号enclosureではない。

旧CDF14との比較ではtensor残差Fが92.6512%、sector誤差が93.9132%減少し、NOT_METから今回の予算PASSになった。旧fitの失敗結果は改変しておらず、初期値への自動fallbackもしていない。保存XDF14自体も予算内で、今回CDFがXDFより物理sector精度を改善したとは主張しない。

## optimizerの返却状態とgradient

| 項目 | 開始 | 終了 |
|---|---:|---:|
| actual `scipy_func` objective | 2.987588808603715e-9 | 2.06690024874864e-9 |
| gradient L2 | 1.0244846686917593e-4 | 6.065062527031419e-5 |
| gradient infinity | 3.573873205362524e-5 | 1.2034710620884496e-5 |
| tensor max-entry残差 | 2.4809821656e-5 | 1.9044877356e-5 |

`success=true`、`status=0`、`message="CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH"`。**nit=1、nfev=3、njev=3**。objective減少による既存停止条件であり、gradientが0になった・大域最適解へ到達したとの判定ではない。optionsや許容差を結果後に変更していない。

実際に返された生x/jacは[optimizer_parameters.npz](../../artifacts/research_c_c1_h6_corrected_fit/2026-10-25/run01/fit/optimizer_parameters.npz)へ各504成分のfloat64で保存した。開始x0/gradient、終了x/再評価gradient、最終J/Uも別保存した。終了actual gradientと返却jacの最大差は0。返却funと最終tensorからのobjective差3.5639076692e-21は固定照合allowance内。旧fitはnit254／nfev263だったが、今回のiteration数やwallから一般的な収束・速度優位は推定しない。

## call・実行・資源・保全

| 群 | 実績／上限 |
|---|---:|
| prepare：保存load、identity、codec、tensor2 | 5/5 |
| fit：input load、実fitter、前後value-gradient、final tensor、objective照合 | 6/6 |
| sector親：snapshots2、context、physical、assessment3、開始gate、終了判定 | 9/9 |
| **合計** | **20/20** |

3つのcall journalで20 attempt／20 COMPLETE／FAIL0を確認した。実fitter／optimizerは1回、codec packはprepare1＋fit内1、optimizer内部nfev/njev=3とは別に登録diagnostic value-gradientを前後各1回実施した。新初期分解・rank探索・追加fit・retry/resume・PF・native build/compile・shots・費用計算はすべて0。10項目の要約gateは全PASS。[STOP](../../artifacts/research_c_c1_h6_corrected_fit/2026-10-25/run01/STOP.json)のfailed=falseに加え、別の結果項目でoptimizer成功とDF予算PASSを確認した。

起動前はCPU affinity32、idle約99.56%、利用可能RAM約52.7 GiB。GPU driverは利用不可で、固定JAX CPU backendを使用した。疎CAR／sector400、初期2状態の2 workers、process-local BLAS/OMP1で小行列とworker間の過剰並列を避けた。物理Hamiltonian/contextは再利用。CPU/RAM/worker/wallへ恣意的なcapを設けず、共有環境・他ユーザーのjobに変更はない。

runner wall4.040692秒。fit worker内区間wall0.489627秒、prepare worker0.014255秒（import等を含む全process wallとは区別）。親process lifetime CPU2.866550秒、fit worker1.808571秒、prepare0.384774秒。peak RSSは親433512 KiB、fit worker426600 KiB。既存との比較速度実験ではない。

元H6 screening／初期化診断／round-trip／H2を含む既存294 filesと、SOURCE47／入力18 hash、今回raw34 filesをpostflightで確認した。H4を含め旧tracked SOURCE・raw結果・grant・ledger・marker・STOPに変更はない。旧契約のunapprovedというmetadataも保存した。今回成果物は固定SOURCEからのlocal evidenceで、外部独立再現・immutable CIではない。

## 到達点と次の作業

修正codecを実ffsim最適化へ接続したこのrank14の1 fitでは、初期分解の忠実性を保ち、optimizer成功とDF予算PASSを両方得た。旧codecの不忠実性は重大な原因候補だったが、この比較から他条件における唯一原因や一般的な最適化性能を断定しない。

**DF予算の数値PASSは、最終エネルギーのCA達成や同CA総資源優位の証明ではない。** engineering guardは厳密なrounding enclosureではない。H6の一般K接続・ground/alias・PF誤差・native費用・shots・総費用は今回未評価。

次の判断はユーザー／研究Cが担当する。返却CDF14を対象に、既存sector経路を用いた一般K同値性・ground/alias・決定論S2 PF限定検証の準備へ進むかを決める。Codexはその具体的依頼後に必要最小限の技術準備を行い、新科学評価は別の固定範囲と明示許可後だけ実行する。停止条件変更・再fit・rank変更は今回の次処理として行わない。**公開照合後mandatory STOPし、追加科学計算へ自動進行しない。**
