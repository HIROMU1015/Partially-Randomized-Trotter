# H6 PR独立確認：Phase 0統合readiness報告

2026-10-19 JST。状態：**PHASE0_COMPLETE_MANDATORY_STOP**。
H6専用adapter・runner・終了witness・暫定契約・人工検査・入力bindingの準備を完了した。
**最終n・新確認seed・科学call総数・最終科学SOURCEは未固定、科学実行は未承認。**
今回の新規sampling/native compile/signal/shot再計算/保存値の数値解析はすべて0。
テスト内のcost・seed・providerは人工データで、実H6の新しい科学結果ではない。

[暫定契約とbinding資料](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_phase0/2026-10-19/README.md)。
Phase 0 adapterのlocal SOURCEとfile SHAは同bundleの`source_binding.json`で確認する。
既存SOURCEを変更せず、`codex/h6-pr-confirmation-phase0-20261019`の専用worktreeで実装した。

## 固定対象と判定

linear H6、隣接1.00 Å、STO-3G、tol-only1e-8 DF rank19、T=.8、P=0。
元の二次PR・無条件finite-RTE law・DF順序・状態・normalizationを継承する。
Qiskit1.3.0、rz/sx/x/cx、opt1、seed_transpiler17、既存symmetric-directional full Hadamard wrapper。
主費用は保存shotで重み付けした`G_i=N_cosine*C_i,cosine+N_sine*C_i,sine`（compiled RZ）。
通常signed bias、allowance倍率1の2条件だけを対象とする。

|ε|固定PR|固定PF|期待するsign(PF G−PR mean G)|
|---|---|---|---:|
|.05|B2_p4_q2_2nd_r4_K2|B0I_p7_q1_2nd|−1（PR不利）|
|.0005|B2_p4_q8_2nd_r4_K2|B0I_p11_q4_4th|+1（PR有利）|

`64c85f7a…`の通常・倍率1・strengthened_S2_S4の保存比較と、元A signal・完成Dの直接実測PF行を照合した。
候補を新しく探索・再最小化していない。中間精度・非相殺の副endpointは追加していない。

判定は期待方向との一致を`reproduction_status=REPRODUCED/NOT_REPRODUCED`に保存する。
PRの方が安いかは独立した`PR_CHEAPER`に保存する。exact同費用は`INCONCLUSIVE`、`PR_CHEAPER=null`。
結果の方向による候補差替え、途中平均による終了、n変更、再試行、外れ値除外は行わない。
CI・p値・母平均優位の保証は出さず、正常実行と科学的再現は別に記録する。

## 既存実装の再利用と必要なadapter

|既存実装|再利用した処理|
|---|---|
|`pr_cost_native.py`|元finite-RTE sampler/step seed API、event列、Writer、native kernelへの接続|
|`pr_cost_batch_science.native_context`|H6 A_COST本番と同じ保存p19/p5ブロック＋元prefix_viewによるp4復元|
|`h8_pr_confirmation_parallel.py`|evolution共有、4control＋独立event oracle、cos/sin paired compile、fingerprint/compiler確認、子process終了receipt|
|`h8_key_compile_parallel.py`|owned fork pool、before-call予約、native wait/reap、異常時の自分のworker回収|
|`h8_pr_confirmation.accept/point_account/run`|native結果受入、paired mean/SD/SE、one-shot・scratch・STOP・終了時byte監査|
|`pr_cost_contract.py`|SOURCE/input/runtimeのSHA照合、Budget、exclusive marker、first STOP保持|
|`launch_h8_pr_confirmation.py`|所有sessionのidentity、wait4、停止・回収primitive|
|H4完成SOURCE `8a18b277…`|方向判定、seed階層・衝突、終了・保全監査の設計を確認。H4の実行系は変更しない|

H8の`N`・cohortを参照する関数は、関数bodyを変えず、独立したglobals辞書でH6用に束縛する。
H8 moduleの定数・SOURCE・保存結果は変更しない。mock fixtureも既存H8から再利用した。
H6専用の差分は候補/PF/期待方向、可変の最終n、seed namespace、K2 support検査、H6入力bridge、承認・witness接続。
H8 runnerのH8専用context・n48・副評価は、そのまま呼び出せないため小さいscheduler adapterで接続した。
科学kernel・control・compiler・paired統計の式は書き直していない。

## 入力・compiler・SOURCEの固定状況

指定された6 commitはすべて取得し、設計/結果の本文またはSOURCEを確認した。取得SHAは`fixed_references.json`に保存した。
H4のscience/contract/completion auditorも固定commitから読んだ。現在の別worktreeのHEAD/statusを停止条件には使わない。

94 input記録のbytes/SHA、H6旧Aのexecution SOURCE blob、2 rootのprovider closure計877 fileを照合した。
runtimeは明示inventoryの34,273 fileを一度full SHA照合した。次の監査では同じinode/size/mtimeのSHAを再利用し、重複I/Oを省いた。
Python executable・compiler profile原本・Hamiltonian/state/snapshot・元signal/shot/PFの来歴を結合した。
H6 profile原本のSHAは照合済みだが、旧compilerの全outputとの完全同一性は未検証のまま保持する。

p4 preparationの保存hashは`ce0c96f4774e07c8980d830056f012ab357b3c4db725b2d817f6deac49ef8319`。
本番Aの`preparation_receipt`から転記し、元p19/p5 archiveとprefix_view SOURCEに結合した。
Phase 0では実preparationの復元やGaussian/native生成を行っていない。

両候補の保存lawのsupportは**K2・次数[0,2]**。K4 law/次数4の混入はnative build前に拒否する。
保存B診断のp5/q8/r2候補は今回のp4/q2・q8/r4と異なるため、その条件付き費用は移送しない。
未知層の費用を0へ補完せず、無条件平均の保証は未確定として保持する。

## seedとcohort

新namespaceは`H6_NEW_PR_HOLDOUT_V1|SOURCE|seed_plan_digest|cell|replica`。
step導出は元の`direct_master_to_step_seeds_v1`をそのまま再利用する。
旧master/派生seed receiptと新A全128 taskの識別子を除外用private台帳へ結合した。
その後のH6 campaignと未完了予約の網羅性は最終seal前に照合するため、`seed_inventory_complete=false`を維持した。
新しい確認seedの割当は0。人工fixtureだけでmaster/派生seed衝突、異なるn、cohort分離を検査した。
実行validatorは旧除外台帳とA128の欠落も拒否する。衝突時のredrawは行わない。
seed非重複はstream分離の証拠であり、数学的独立性そのものの証明とはしない。

## 標本数の設計案と次の承認範囲

以下は元の保存集計の転記。SD/分散・差・削減率・必要nは今回計算していない。

|対象|保存n8 mean G（exact）|保存paired SE|固定PF G（exact）|
|---|---:|---:|---:|
|p4/q2/r4/K2|5037196305/8|8425392.116288794|356097424|
|p4/q8/r4/K2|65509543717397/2|254602554677.51428|36384720355183|

提案する数値設計scopeは、上の2候補の保存値を入力にした一度の計画算術。
H8の既存案を参考にfamily α=.05、目標power=.8、SD倍率f=1/2/4、2件の事前固定familyを提案する。
quantum shot側のα・誤差配分は変更しない。

`s8=SE8*sqrt(8)`、`Δ=abs(PF_G−mean8)`、
`n=max(2,ceil(((z_(1−α_family/2)+z_power)*f*s8/Δ)^2))`をシナリオ式として固定した。
H8の`normal_scenario`は正のPR優位marginだけを受け付けるため、数値設計時には符号付き比較を扱う最小adapterが必要。
現段階ではその関数を実科学値で呼んでいない。

保存K2 lawのp2/Rから`π2=1−(1−p2)^R`、`visibility(n)=1−(1−π2)^n`を評価する案も承認対象に含める。
H8の48本を共通参考点として比較するが、H6への必要性・十分性はまだ判定していない。
小さい選抜n8からのnormal plug-inと分散倍率は感度シナリオで、実power・tail分散・CLT・母平均優位の保証ではない。
条件付きsampling、bootstrap、raw費用の分散再集計、新signal、shot再最適化は数値設計案にも自動追加しない。

## 科学call予算・並列化・終了契約

最終nを`n_q2,n_q8`とし、現段階は式だけを固定する。

|prospective call|固定する式|
|---|---|
|trajectory/native evolution|n_q2+n_q8|
|full-wrapper compile|2(n_q2+n_q8)|
|occurrence|2 n_q2+8 n_q8|
|RTE event|8 n_q2+32 n_q8|
|代表control|各候補4、計8|
|独立event oracle|各候補1、計2|
|新PF compile/SCF/DF/state/signal/shot再計算|0|

候補ごとにgrant/output/marker/STOPを分離し、代表controlの受入後に残りを並列投入する。
p4 contextを一度復元しforkで共有、同一trajectoryのevolutionをcos/sinで共有する。
worker数・CPU/RAM/wallの恒常capは設けず、科学実行直前の有効CPU/memory・共有jobとの非干渉を確認して決める。
Phase 0の読取観測ではaffinity64 CPUだったが、将来のworker数・資源予約ではない。
内部thread1はnested oversubscriptionを避けるprocess内設定。OS/cgroup/共有venv/他jobは変更しない。
一時的PSI・CPU競合・無関係worktreeの更新だけでは停止せず、実OOM・自分のnative異常・入力不一致はSTOPする。
失敗時のbefore-call予約・marker・最初のSTOPを保持し、自動再試行・自動拡張はしない。
終了は全固定n・両軸・台帳・SOURCE/input/runtime・native exit0・全worker回収を監査して保存する。

## 人工検査と限界

H6人工31件PASS、既存H8人工22件PASS（計53件）。作者による技術検査で、別reviewerの独立レビューではない。
候補ID/q/r/K/PF/方向、cohort、入力/shot/SOURCE、seed衝突、K2 support、paired統計、反対方向・同値を検査した。
両候補のmock control/paired compile、H6 scheduler全経路、owned forkの正常終了・全回収も確認した。
終了witnessは科学処理をしない小さいchildでexit0/exit3とSTOP保存を検査した。
未承認grant・Phase 0契約はinput/provider読み込み前に拒否される。

最初の人工実行ではNumPy内部threadが既存forkのsingle-thread-parent検査に抵触した。
テストprocess内のBLAS threadを1にしてPASSし、失敗logもprivateに保持した。既存SOURCEの変更は不要だった。
実H6の新規sampling・native compile・preparation復元・signal・保存値統計は未実行で、live科学経路の成功は主張しない。

## 保全と停止後の担当

原input/result/ledger/grant/marker/STOPには書かず、新worktreeと新規privateディレクトリにだけ成果を保存した。
94 input記録を終了時にも照合し、実際に再利用するH6/H8 code bytesは固定SOURCEと一致した。
Codex BのH4 tol-only統一worktreeやjobには変更・停止・凍結要求を行っていない。
原本のprivateパス、seed値・event列、runtime inventory本体はlocal private領域に保持する。
今回のcommitはlocalのみ。公開pushは行わない。

**ユーザー（次、未承認）**：上記の保存値に限定した標本数・powerシナリオ・K2 visibility計画算術のscopeを承認する。
**Codex（その承認後）**：数値設計を実施して推奨nを提示し、最終nの判断後にSOURCE/input/runtime/seed/call予算の最終sealを準備する。
**ユーザー（さらに後、別承認）**：最終sealを確認してH6独立sampling/compileを承認する。
**Codex（現在）**：Phase 0を保存・local commitしてmandatory STOP。次stageを自動開始しない。
