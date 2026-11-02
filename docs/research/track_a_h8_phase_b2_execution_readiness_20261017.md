# H8 Phase B2：固定S4 4軸の実装・実行準備

**B2のnative adapter・直列launcher・人工検証・SOURCE sealと無効grant案を準備した。実H8 circuitの復元・新compile・signal・費用fit/再会計は0。新規compileを起動するには、固定4軸に対する別途明示承認とfresh gate通過が必要。公開後mandatory STOPする。**

branch：`track-a-h8-phase-b2-readiness-20261017`。起点はB1正本 `83a3c4a6688e66576f85471bf41a45e29cbfb705`。別のB1実装を統合・変更・再実行していない。M1-SのSOURCE・実装・契約・成果物とは分離した。

## 入力と検証順序

| 順序 | 候補 | 軸 | 固定予測RZ/軸 |
| --- | --- | --- | --- |
| 1 | B0 p15/q1 S4 | cosine | 1175933/7 |
| 2 | B0 p15/q1 S4 | sine | 1175933/7 |
| 3 | B0+I p21/q8 S4 | cosine | 1852640 |
| 4 | B0+I p21/q8 S4 | sine | 1852640 |

予測はB1の保存値を転記し、今回再fit・別係数移送・再会計をしていない。丸めた167,990.43を判定に使わず、有理数1175933/7を使用する。model canonical SHAは `8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0`。B0のholdout 0%は独立1条件、B0+Iの0%はtraining anchor残差であり、S4精度の保証にはしない。

compile同時数は1。1軸のworkerがexit0でwait4回収され、結果SHA・wrapper/compilerを確認してから `abs(predicted-measured)/measured < 1/50` を判定する。2%以上（ちょうど2%を含む）・非正測定・control/identity/資源異常なら、その軸までのrawと非返却予約を保存してSTOPし、次軸を投入しない。軸平均・自動refit・追加候補・retry/resumeで救済しない。

## 科学条件とnative経路

H8直線鎖、隣接1Å、STO-3G、16 modes、Nα=Nβ=4/sector4900。保存tol-only DF rank28、tol1e-8/cutoff0、signed generation order、weighted Hermitian projection。T=.8、今回q1/8、δ=.8/.1、L_D=p15/21、S4は負時間中段を含む非mergeの3 S2段。full-H referenceは保存未回転、経験的allowance factor1/10と厳密bias/PR母平均保証の未確立事項を保持する。

providerは `c31ef424175c02379eafebe8fc6c7548c1d9aabe` の265 files。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/topologyなし。compiler `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、完全profile `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`、34,273 runtime filesを固定する。

B2 adapterは、承認とfresh gateの後にだけ保存snapshot/stateとfull preparation操作列を復元する。新SCF/DF/固有分解/Gaussian basis生成・state生成をしない。p15は保存full p28のprefix viewとして新しいview recipe SHAへ結合し、保存されていないdiscard:p15のroot hashを捏造しない。p21は既存discard:p21のrootにも照合する。復元したlive Gateを再記述し、basis/qubit/order/matrix/diagonal/constantを保存metadataと照合する。

B0はoriginal constantを一回だけ含め、tail identityを加えない。B0+IはPhase Aに保存されたp21 full-Fock Cをoriginalへ一回だけ加え、Cを新しく計算しない。symmetric-directional control・state preparation費用を除くfull measured Hadamard wrapperを維持する。保存されたPhase Aのcorrected/independent-forward signal、full-H reference・allowanceを参照する。

各caseで左branchのidentity、右branchと保存独立signal、cosine/sine wrapperのX/Yを検証し、非finiteまたは1e-8以上ならcompileを起動しない。これらは承認後のB2に含む限定control検証であり、新signal探索やsamplingではない。

## 資源・高速化とSTOP

snapshot load・preparation decodeは各1回、prefix view2、S4 evolution2、wrapper4/probe copy4/control probe8、compile4/model error check4を上限とする。fit/再会計/new signal/full-H/SCF/DF/state/PR/law/tail scalar/未登録compileは0。1 caseのnative/controlを共有し、その2軸を順番にcompileして回収後に次caseへ進む。B2の軸ごとのSTOP契約を守るため、compileの並列数は増やさない。

| 資源 | 条件/上限 |
| --- | --- |
| process/thread/core | driver1＋compile worker最大1＋observer1、各thread1、別3物理core |
| AS/RSS | driver64/32 GiB、worker64/32 GiB、observer512/256 MiB |
| aggregate RSS/admission reserve | 64.25 GiB/16 GiB、実効空きRAM80.25 GiB以上 |
| 累積CPU/phase wall | 24 CPU時間/12時間（所要時間予測ではない） |
| observer CPU/間隔/gap | 1800秒/1秒/5秒超STOP |
| stage/output/logs | 8 GiB/256 MiB/128 MiB |
| 起動前disk/inode | 16 GiB/40,000以上 |
| native/transpiled命令 | 4,000,000/10,000,000 |

保存操作数にproviderの静的配列確保式を適用したnative wrapper上限は、p15/q1で60,785、p21/q8で673,205。これはcompile費用・性能測定・実分子の回路生成ではなく、pre-build allocation guardのmetadata確認である。transpiled費用やRAMの保証ではないため、実行中の上限監視を維持する。

起動前にfull SOURCE/input/provider/runtimeを照合し、driver照合後にもう一度30秒以内のCPU/RAM/quota/disk/PSI/OOM/保全fresh gateを通す。CPU raw busyはSMT siblingを含め5%以内、自身の負荷を引かず、失敗時に自動再選択しない。他jobの終了・affinity変更・共有環境修正で条件を作らない。管理用Gitはthread1/optional lock無効を維持し、thread違反はPID/start/parent/uid・exe・thread数をcleanup前に記録する。

## 実装と技術検証

固定実行SOURCE：`244c45e752fa703e286d40ece9ad641430ea9c6e`。21 SOURCE files、provider265、runtime34,273、入力15件をGit/disk/bytesで結合し、別のprivate準備領域にconfigと無効grantを保存した。公開するのは無効grantとpathを含まないhash receiptである。新しい有効grant・科学one-shot lock・output/control・fresh resource gateは作成していない。

固定Python3.11.0rc1で新19＋既存100＝119 unittest methodsがPASS。grant拒否・古いscope・3 core/資源/call上限・2%境界・各失敗位置での後続抑止・control失敗時compile0・worker失敗時の未settle保持・native回収・CPU失敗receipt・試行域再使用拒否・科学module import前のpermitを検証した。

2 qubitの人工provider fixtureでは11項目を確認し、4軸の人工compileのみ実施。保存分子入力は開かず、独立dense S4（負時間中段を含む）とB0/B0+I scalar一回、左/右/X/Y、wrapper/compilerを照合した。最大control discrepancyは約1.56e-14。誤ったindependent signal・prefix記述は拒否した。実H8 object復元・compile0、実データfit0。主エージェント自身のlocal検証であり、独立レビュー・CI・外部再現とは呼ばない。

## 並行作業と保全

準備中に別チャットの `h8-b1-stop-acceptance-reconciliation-20261017` が更新され、最初のmetadata準備はPROTECTED_WORKTREE_CHANGEDで拒否された。その時点で準備directory・科学grant・実行試行は未作成。旧baselineと差分記録を保持した。

利用者が許可した別B1の文書整理commit更新を読取確認し、実行保全gateを緩めず、更新後の状態を別baselineとして記録した。初期122 worktreeのうち121は不変、残る1件はこの許可済み更新であり、このチャットによる変更ではない。更新後baselineの122既存worktree、旧683 grant/ledger/carry/marker等、旧V2 untrackedを照合した。元B0 run02の53 raw filesとB1の34 raw filesも不変である。

これは科学実行のretryやSTOP解除ではない。元B0 run02 STOP、B1正本の正常完了、別B1実装、V1・V2 CLOSED_NO_GO・H5/H7・M1-Sを変更していない。将来B2の実行に際しては、別チャットの保全対象worktree更新を止め、現在状態へ結合したgrantとfresh gateを用意する。準備中の更新履歴を使って実行中の変更を許可する構成ではない。

## 次の承認範囲

承認対象は上記4軸・最大一回のB2だけ。承認後もSOURCE・compiler・preparation・入力hash・資源gateが一致する場合に限り起動する。モデル不一致・技術異常ではSTOPし、自動retry/refit/追加compileなし。全4軸PASSでもmandatory STOPし、5精度の再会計Cは別途承認する。

[公開bundle/checker](../../artifacts/resource_applicability/track_a_h8_phase_b2_readiness/2026-10-17/README.md)をcommit・pushし、remote SHA確認後、準備段階としてmandatory STOPする。
