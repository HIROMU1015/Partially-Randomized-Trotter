# H6希少Taylor次数：s2限定pilotの実行準備契約 v1

2026-10-13。利用者が指定した最小の追加情報取得案を実装・固定した。**準備完了、実8 trajectory・16 compileは未承認・未実行**。[一つの資料入口](../../artifacts/resource_applicability/track_a_h6_conditional_s2_pilot_preparation_v1/2026-10-13/README.md)にseal・未承認authorization草案・予算・58限定tests・byte監査を置く。GPT科学レビューを繰り返さず、通常の技術判断で準備を完結した。

## 固定対象と変更範囲

基点 `track-a-h6-rare-order-cost-analysis-20261013` / `eee1343f6e99aa7661892ddfbe1067e974e2b74a`。専用branch `track-a-h6-conditional-s2-pilot-preparation-20261013`、独立worktree `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-conditional-s2-pilot-preparation-20261013`。

新SOURCE **843265b624eac8fbb9acea47ef21cab1207e1880**。SOURCE commitは新3 modules・3 scripts・test1だけ。runtime closure257件＝既存251件byte不変＋新6件。科学SOURCE `5a903aea2492c2d211c1e15e7737df0d50694c12` の224件を保全する。探索592 group・独立確認256 group・そのtask/seed/結果/消費済みgrantを変更しない。H4・他研究・共有設定・private venvを変更しない。

linear H6/1.00 Å/STO-3G、12 modes、alpha3/beta3 sector400、保存state、T=0.8。tol-only DF1e-8/rank19/cutoff0/signed generation order/weighted Hermitian projection。既存gridのprefix5/10/15/full19、delta0.1–0.8を保全し、今回だけ **B2_p5_q8_2nd_r2_K2 / prefix5 / delta0.1 / q8 / r2 / R16 / K2** のwhole trajectory s2層を対象にする。探索と独立確認へ追加標本をpoolしない。

同じraw-concatenation partial S2、symmetric_directional、測定込みfull Hadamard wrapperを両軸で取得する。state preparation除外P=0。Hamiltonian/DF/保存state・finite-time signal・経験的numerical allowance・B・normalization・保存shot会計を変更せず、量子shotをsampleしない。新signal/SCF/DF/state/input生成0、H8・追加探索・次stageなし。

## 元の分布を条件付ける方法

K2の1 occurrenceの確率はp0=0.998948186619147、p2=0.001051813380852956。s2は16 occurrence中にorder2が1個以上ある層、`pi2=1-p0^16=0.01669690613968732`。

1. Mを `Binomial(16,p2) | M>=1` から引く。1–16の全supportを残す。
2. 16位置からM個を一様なsubsetとして選ぶ。該当位置がorder2、残りがorder0。
3. 各eventのk+1 componentを、元の独立component選択確率から引く。元の `_make_event` による角度・順序・sign・phase・coefficient・event probability・normalizationをそのまま使う。

従って次数列の確率は `p2^m*p0^(16-m)/pi2`。条件付き費用標本の法則だけを変え、量子event重みをpi2で割り直さない。元finite lawとの一致はsampling直前に検査する。q8の元master→step seed規則とPCG64をcomponent streamに保持し、条件付きorder用の別streamを設ける。回路の基底列・隣接基底reuse・原builder/compilerの意味を変えない。

新domain `H6_CONDITIONAL_S2_PILOT_V1_FRESH_COST` とphase `conditional_s2_pilot_v1`。master8/order8/component-step64の80 seedは既存832標本に由来する3744 master/step seedと衝突せず、新stream同士も衝突0。これは保存seedの一致検査で、擬似乱数の数学的独立性の証明ではない。paired cosine/sineは同じsampled eventsを共有し、再sampleしない。

## 環境・入力・source binding

既存専用Python `/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python` を変更せず使用。Python3.11.0rc1、Qiskit1.3.0、NumPy1.26.4、SciPy1.14.1、Numba0.60.0、OpenFermion1.6.1、PySCF2.7.0。native依存・critical source/binaryも既存profileで照合する。

profile SHA `4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75`、compiler SHA `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing=None。旧compilerとのoutput bit完全一致は未検証のまま。

凍結入力33件をstreaming byte SHAで検査し、科学arrayをdecodeしない。既存roundoff準備archive5670237 bytes / SHA `ad52f9f6a3150981d7be00191d8b3a9f9af992c37c9102cba6d94fcffc8e3010` と受理receipt `e11c37c0cc5813a49d837d04ed7c32bf153a3f0e4ce6cdcb288bec44ba84378b` をbyte/既存証拠でbindingした。新しいroundoff診断は実行しない。元完了native証跡・exit0・保存callsをhash照合し、旧科学計算/tests/proof campaignは再実行しない。実行承認後のworkerだけが既存archive/保存stateを復元する。

sealはSOURCE/closure/profile/input/旧結果commit/carry/finite law/8登録seed/resources/output/one-shot lockに結合する。REVIEW資料はSOURCEを変更しない別commit。外部raw証跡とprivate environmentを使うためGit単独で実科学runが再現できるという主張はしない。

## 新規・累積予算

| counter | 既知消費 | 旧予約 | 新pilot最大 | 合計 | 提案上限 |
|---|---:|---:|---:|---:|---:|
| compile | 1700 | 20 | 16 | 1736 | 1736 |
| trajectory | 834 | 10 | 8 | 852 | 852 |
| occurrence | 3136 | 64 | 64 | 3264 | 6656（維持） |
| control_probe | 360 | 4 | 0 | 364 | 1024（維持） |
| primitive | 1470 | 0 | 0 | 1470 | 10000（維持） |
| reference_matvec | 400 | 0 | 0 | 400 | 20000（維持） |

compile **1720→1736**、trajectory **844→852** だけが次回明示承認を要する変更案。旧消費・予約を再利用/reset/返却しない。unknown historical actualはnull。1 trajectoryはq8のoccurrence call8を予約し、その内部でRTE event16を引く。新8本でoccurrence64/event128、event数とregistered call数を混同しない。各子はcompile2/trajectory1/occurrence8を作業前にchargeし、全未受理groupは失敗・未dispatchも含め全額予約を保持する。

## CPU・memory・STOP・保存

単独disposable worker8回、内部thread1。worker候補CPU20、coordinator候補CPU22/23。worker AS32 GiB/RSS STOP9 GiB、coordinator AS soft2/hard32 GiB/RSS reserve2 GiB、aggregate RSS40 GiB、job RAM予算64 GiB。AS/RSS・headroomは既存H6を保持し、kernelによるRAM予約ではない。GPUを使用・queryしない。既存H4へaffinity/STOPを適用しない。準備ではtaskset/実workerを起動せず、`allowed_cpus=[]`。

元supervisorが別processのworkerを1秒間隔で観測し、所要/間隔staleness5秒超・欠測・worker/aggregate/CPU・host/cgroupメモリ不足・新OOM・host full PSI>=5%・cgroup full PSI>0でfail-closed STOP。fresh host216.25 GiB/effective64 GiB、running host168.25/effective16 GiB、H4 margin152.25/headroom16 GiBを保持する。最初のSTOP理由を保存し、所有identityを確認したown-run process groupだけを停止、wait/reap/log/journal closeまで記録する。旧guard/cleanupを再実装しない。

新output `artifacts/resource_applicability/track_a_h6_conditional_s2_pilot_v1/2026-10-13/launch_pilot_v1/`、lock `.server-preparation/h6-conditional-s2-pilot-v1/one-shot.lock` は未使用。旧supervisorのleaf名 `cost_confirmation_fast_0000`–`0007` は新parent namespace内だけで使い、新phase/seed/sourceを照合する。旧確認結果との混同・再利用はしない。

保存はJSON events/metadata/metrics/fingerprintとJSONL監視、QPYや巨大回路tree/NPZは追加保存しない。output2 GiB、各group32 MiB、全worker log1 MiB、journal32 MiB/10秒emitを維持。phase/total wall capは既定null、自動retry/resumeなし。原instruction guardはnative1,000,000/compiled5,000,000を維持する。journalが満杯でもmandatory STOPとなるため無期限完走保証はしない。

[容量案](../../artifacts/resource_applicability/track_a_h6_conditional_s2_pilot_preparation_v1/2026-10-13/capacity_plan_v1.json)はgroup256 MiB＋journal32＋log1＋親metadata16、atomic一時payload32、cache余裕512、8768 inode allowance×4096-byte block slackを合わせ約0.863 GiB。cache含む工学的見積りであり全allocationの証明ではない。起動前にfree10 GiB/100000 inodeとquotaを再検査し、実行中aggregate outputにcache/一時fileも数える。

準備時の読取snapshotではhost/effective available約464.98 GiB、filesystem約400.41 GiB/219933929 inode、user/group/project quotaはdisabledと判明、affinity0–63、AS/FSIZE hard unlimited、全full PSI0。cgroup祖先の既存oom_kill1はbaselineとして残し、発生時刻や今回増加0を推測しない。起動直前に再検査し新しいbaselineを取る。CPU空きや64 GiBの排他的割当を保証するsnapshotではない。

## 限定検証と実行方法

SOURCEに対応する58人工/mock testsが3.74秒で合格。条件付きcount/vector/複数order2、元component/weight/phase、2-system-qubit+ancillaのnative evolution/両軸wrapper意味、draft拒否、source/law/input guard、carry/partial予約、mock pressure STOP/cleanup/両軸受理、default dry childを検査した。test process1、dry child最大1、thread1/AS2 GiB/wall180秒/file16 MiB。transpileとnp.loadはraising guardで拒否する。実H6 trajectory/compile/production worker/入力array decode0、CI・独立科学レビューのPASSではない。

read-only auditorとdefault runnerも合格。未承認draftは新grant gateで拒否される。起動用templateは[execution_handoff_v1.json](../../artifacts/resource_applicability/track_a_h6_conditional_s2_pilot_preparation_v1/2026-10-13/execution_handoff_v1.json)に記載したが、`absolute_launch_command=null`、approved=false、runtime_authorization=falseを保持する。

次回は利用者が固定SOURCE・8条件付きtrajectory/16 compile・上限1736/852・資源政策を明示承認した後、新exact grant（旧grantは禁止）とその実byte SHA、remote照合、fresh gate、未使用output/lockを結合して一度だけ起動する。完了または最初のSTOP後に終了し、追加計算へ進まない。承認前はdefault dry-runしか実施しない。

## 残るリスクとmandatory STOP

実条件付きH6 wrapperのcompile壁時計・RSS・完走は未測定。s2のM>=2確率は約0.7873%で、pilot8本に一度もM>=2が出ない確率は約93.87%。正しいsupportを実装しても8本で希少component列や複数rareの費用を覆えない。条件付きmean/SD/SEはs2標本の記述値で、mother mean tail boundやpopulation winnerの証明ではない。後続解析では保存pi0/pi2とs0推定を使い、探索/確認/pilotを区別する。科学比較の新契約やH8判断をここで確定しない。

軽量source/tests/準備資料を公開・remote byte照合した後 **mandatory STOP**。実8 trajectory/16 compile、新approved grant、共有/venv変更、H4/他job操作は今回0。
