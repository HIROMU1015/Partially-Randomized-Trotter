# H4非DF/DF PR・PF：実行基盤と承認待ち契約

2026-10-23 JST。**H4専用runner・production event記録・入力封印・4方式統合を実装。H4実分子計算は未実行。完成契約は未採択・未承認で、execution_authorized/grant_valid/dispatch_enabledはfalse。**

基点は`f7da77da0581c7a79c31c981cd9c6c1fcb05d98f`、旧技術SOURCEは`8255fd0ed4e2e79b4394f8d1a91b67e9e20f6813`。GitHubの`codex/non-df-h4-readiness-20261023`が基点と一致し、H2結果branchは`c8990de7721545faff88b636108034dedb80a463`、科学SOURCEは`7a6c4263d96153ecbe721dcfb6941d0d2684b3b0`であることを確認した。ローカルにはこの二つのbranch/worktreeは存在せず、取得済み固定Git objectsを使って、新しい`codex/non-df-h4-execution-foundation-20261023`へ分岐した。他worktreeは書き換えていない。

[完成契約案](../artifacts/non_df_h4_execution_foundation/2026-10-23/H4_execution_contract_DRAFT.json)、[一括承認案](../artifacts/non_df_h4_execution_foundation/2026-10-23/H4_batch_approval_proposal.md)、[開始手順](../artifacts/non_df_h4_execution_foundation/2026-10-23/EXECUTION.md)、[技術証拠入口](../artifacts/non_df_h4_execution_foundation/2026-10-23/README.md)を一組として読む。

## 実装と旧SOURCEの保全

既存の`fixed_signal_evaluation.py`、`pauli_signal_adapter.py`、`rte.py`、DF sampler/builders、H2 runner、旧supervisorは変更していない。新しいAPIはopt-inであり、旧H2の科学kernel・記録済みhashを暗黙に置き換えない。

| ファイル | 役割 |
|---|---|
| `scripts/run_h4_non_df_pilot.py` | contract表示・将来activation・guarded実行・内部人工fixtureのCLI |
| `src/trotterlib/h4_execution_contract.py` | stdlib認証、固定契約・SOURCE・runtime・承認・grant・namespace・消費状態の照合 |
| `src/trotterlib/h4_pilot_execution.py` | 入力生成1process→入力封印→4 fresh interpreterのsupervisor管理、部分結果・STOP保全 |
| `src/trotterlib/h4_molecular_inputs.py` | H4 RHF/MO/Pauli/DF/共通状態/FCI/sector gateの実装。今回はmockのみで呼出し |
| `src/trotterlib/rte_event_audit.py` | production sampler/requestの実戻り値観測とevent/trajectory/価格のhash join |
| `src/trotterlib/h4_audited_evaluation.py` | 旧評価関数の数値順序を保つ追加経路。adapter選択と観測・join以外のAST一致を試験 |
| `src/trotterlib/h4_pilot_science.py` | 保存入力復元、固定case評価、captured-event演算照合、5精度会計、4方式統合 |

## 入力生成と4固定case

実行承認後の入力生成は、linear H4・1.00 Å・STO-3G・charge0/spin0、canonical real RHF1回、MO積分1組、tol-only1e-8 DF1回を用いる。4 spatial/8 spin、Nalpha=Nbeta=2のsector36を検証する。非DF sector最低状態を4ケースで共有し、DF sectorとPySCF FCIを照合する。Pauli/独立JW/DF全演算子/既存DF sector演算子・state norm/residual/energy・reference signalを既存閾値で検査する。入力失敗時は4ケースへdispatchしない。

OpenFermion 1.6.1の`spinorb_from_spatial`には内部の`EQ_TOLERANCE`ゼロ化がある。H4のproduction DFには、既存Pauliの`from_spatial_integrals`と同じ未閾値の明示的spin liftを用い、同じMO係数と1/2規約を両表現へ渡す。独立legacy JW照合は元関数の挙動を保持する。これは利用者指定の「同じ未閾値MO積分」を実装した差であり、旧H2を遡及修正していない。1e-17/2e-17の人工係数保持を検査する。

Pauliの`(-abs(c),label)`順、DFの既存`abs(lambda)*Frobenius(g)^2`順位とoriginal-index tie-breakで、それぞれ重み50%以上の最初のprefixを一意に決める。DF one-bodyは常時保持、正の非空tailを必須にする。結果を見た再選択・空tailのPF代替・適応samplingはない。実Pauli項数・DF rank・prefixは未生成で、旧H4/H2から転写しない。

全入力をsnapshot.npz、pauli.json、receipt.jsonに固定する。receiptはsnapshot/Pauli JSON/state/Hamiltonian/prefixのhashを持ち、4caseのmarkerはreceiptとsnapshot hashを共有する。各workerは復元時に両hash・8 qubit・36 sector・state・prefixを照合する。状態準備費用P=0、各表現自身のfull-H referenceを使用する。

固定caseはPauli PF/PR、DF PF/PRの4件。T=.8/q8/δ=.1/S2、PR r4/K4、各256 paired trajectories、ε=.05/.01/.005/.001/.0005、既存Hoeffdingと経験的allowance、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/topologyなし、ordinary-controlled full cosine/sine wrapperである。正規化・event位相・積公式順・compiler設定の変更はない。

## 実eventを再抽選せず記録する仕組み

Pauli adapterは旧APIと同じ`sample_rte_events(..., sample_count=q*r, seed=seed)`を一度呼び、その戻りtupleを保存して同じ順に回路化する。DF adapterは旧APIと同じ`make_df_partial_s2_repeated_request`を一度呼び、返された`rte_occurrences`と`step_seeds`を保存して、同じrequestを旧builderへ渡す。monkeypatchやthread共有observerはproduction経路にない。

trajectory IDはcase・SOURCE/input/grant binding・index・seedから決まり、event recordを回路生成前に排他的保存する。イベント位置、outer/inner index、Taylor order、component選択と適用順、符号、回転角、event/scalar位相、event確率、distribution ID/normalization、seed階層が記録される。実objectから得られない情報を推測で埋める仕様ではない。両軸compile後、同じIDへRZ価格・compiler設定・wrapper fingerprintをjoinする。event・join・親resultのhash、位置の重複/欠測、seed・価格・input bindingを照合する。

独立trajectory演算照合は、記録のために再sampleしたeventではなく、実際に回路に渡したcaptured objectを使う。先頭3trajectoryのoperatorとDF ordinary controlを独立行列式と比較し、全取得wrapperのcompile前後の測定確率を確認する。強制order0/2/4は代数検査であり、production MCや費用標本へ混ぜず、追加compileもしない。

audit有無・旧adapterの比較では、production呼出回数/引数、返却event列、seed、回路fingerprint/global phase、compiled価格、reference/raw/corrected/bias/normalization/N/Gが一致することを人工入力で検査する。event単独価格を足してfull-wrapper費用を再構成したり、この記録から希少eventの母平均保証を主張したりしない。

## one-shot・拒否・実行監視

無効/未発行grantでは、科学dispatcherをimportする前にCLIが拒否する。将来のactivationも、clean SOURCE・完成契約digest・正確なpack namespaceを明記した明示承認を要求し、固定package版・Python/package SOURCE/binary hashを封印して新grantを作る。今回activationは実行しておらず、live科学grantは存在しない。テストのin-memory bindingは`TEST_ONLY_NOT_A_GIT_COMMIT`等の非実行fixtureで、科学grant発行ではない。

承認claim directoryはSOURCE/契約/承認/namespaceへ固定する。run IDだけの変更はpack/grant照合で拒否し、固定case claimは返却しない。入力生成と各caseはone-shotで、入力stage失敗、case非ゼロexit、中断、STOP要求、source/input不一致ではretryしない。CONSUMED/STOP/既存runを新たな実行の成功として扱わない。

各段階は既存`isolated_qiskit_supervisor`のfresh interpreter/session、pidfd・subreaper管理で監視する。inputとcasesでそれぞれ独立の監視batchを使う。case失敗時はpeer停止・自分の子孫だけを回収し、stdout/stderr/process/ledger/STOPと取得済みeventを保存する。正常計算に一律deadlineを置かず、PSI/CPU競合だけでは終了しない。supervisor自体のSIGKILL・kernelのuninterruptible状態・machine/storage failureには普遍的回収/STOP保存保証を付けず、未回収をPASSにしない。これらの異常系は既存supervisor回帰で検査する。

科学call範囲はSCF1、MO積分1、Pauli JW1、DF1、非DF ground1、DF sector照合1（36列）、FCI1、入力reference2と4固定case。決定論2trajectory＋PR512、最大full-wrapper compile requests1028、production RTE events16384を契約に明記した。Pauliは256 batch、DFは8outer×256=2048 occurrence batch。cache命中時のcompile再利用は記録し、samplingを省略しない。各caseの固定trajectory上限を開始前に照合し、reserved/actual call accountingとcompile request/cache件数を保存する。

## 数値結果・人工技術検証の読み方

共通の既存empirical allowance、安全係数10・roundoff64・floor1e-12・u/headroom≤.01を維持する。精度不適格/数値証拠不足はN/G nullで保存し、主要resultと5精度表の双方に同じ適格性を適用する。paired covarianceとMC SEは観測標本の診断で、未観測tail・母平均順位・外部再現の保証ではない。

人工8 system/9 wrapper qubitの最終runは、新H4 runnerを通じ、mock入力封印・4 process・event/価格join・4×5精度表を検査する。人工データをH4分子の結果とは呼ばない。固定q8/r4の8 qubit operation proxyでは、全20行がN/G nullとなることもそのまま保存した。別の小さい人工評価でaudit有無の正のC/N/G一致を検証する。分子入力生成経路はSCF/積分/DF/FCIをすべてmockに置換し、各1callと8 qubit/sector36、operator/state/reference/snapshotの処理を試験した。

最終PASS/FAIL/skip/除外件数、人工runの最大誤差、call/seed/hash監査、旧SOURCEの保全件数はartifactの`validation_summary.json`・`verification.json`を正本とする。人工の反復・初回開発試験と最終試験の件数を合算して独立検証数と呼ばない。旧H2のsampling replay、旧run再開、実H4 SCF/FCI/DF/state/signal/MC/compile、科学C/N/G取得、詳細科学レビュー、pushは0である。

## 完了と次の担当

必要な実装は追加済みであり、人工/回帰結果と技術SOURCEをローカルcommitへ固定する。H4実分子に対する精度適格性、runtime/time-to-solution、DF rank/prefix数値、N/Gの存在、希少eventの観測は未確認である。実分子でgateに失敗した場合、追加実装や閾値変更で勝手に科学retryせず、部分証拠を保全してSTOPする。

次の担当は**ユーザー**。完成契約・SOURCE・namespaceを対象に、提案されたH4入力生成と4固定caseを一括で承認するか判断する。承認があれば**Codex**が同じ契約内のseal→新grant→one-shot実行→保存・ローカルcommitを進め、各工程の再承認は要求しない。現在は**mandatory STOP**。追加科学計算・詳細科学レビュー・別campaign・pushへ進まない。
