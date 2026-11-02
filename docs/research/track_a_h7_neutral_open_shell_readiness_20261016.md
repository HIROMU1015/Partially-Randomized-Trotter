# H7中性開殻：限定pilotの実行準備

2026-10-16 JST。**PREPARATION ONLY：実装・人工/mock検査・SOURCE/profile/carry固定を完了。実H7科学計算0、実行grant未発行。**

- branch：`track-a-h7-neutral-open-shell-readiness-20261016`
- SOURCE：`f93ad35b05a75c5e17e2c53b0f3de2af4fc443f2`
- 起点：H5完了結果 `bde58ba23f1fecdf77ee950795cb80c199c0f86b`
- [一つの資料入口](../../artifacts/resource_applicability/track_a_h7_open_shell_readiness_v1/2026-10-16/README.md)

利用者の「同様の計算をH7でも行えるか　まだ実行しなくてよい」「その方針で進めて」を、H5と同じ限定pilotの実行準備として扱う。実H7起動の認可へ読み替えず、科学方針の再レビューや全面resource mapは実施しない。

## 固定する科学条件

Linear neutral H7 / 1.00 Å / STO-3G / charge0 / ROHF / 共通real spatial MO / spin1・multiplicity2候補 / 7 electrons / Nα4,Nβ3 / 7 spatial・14 spin orbitals / sector1,225 / T=.8。

DFはH5と同じtol-only1e-8、cutoff0、signed returned generation order、weighted Hermitian projection。primary N14と補助N7、既存9.9e-11 Hartree engineering gateを維持する。実rank Lは未取得で、p=ceil(L/4)。他系のrankへ合わせたfragment切落しはしない。

評価stateはDF sector solver state。ROHF determinantは軌道取得とsolver初期値だけに使う。固定Ms=1/2をdoublet証明に使わず、実stateのS²=.75 target residual・norm・Hamiltonian residual・独立referenceを確認する。不成立時にstate/SCF/DF政策を自動変更しない。

| 構成 | 固定設定 |
|---|---|
| B0 | prefix p、S2、q1 |
| B0+I | prefix p、S2、discarded-tail identity位相 |
| Full PF S2 | 全returned rank L、q1 |
| Full PF S4 | 全returned rank L、global Yoshida、負時間係数保持、q1 |
| PR K2 | prefix p、S2、q1/r2/K2 |
| PR K4 | prefix p、S2、q1/r2/K4 |

最初に各1group＝12 full-wrapper compileでcorrectnessと実資源を確認する。全6構成が完了しguardを満たした場合だけ、いずれかの要求精度に適格なPRへ残7trajectoryずつを取得する。
最大20group/40compile/16PR trajectory/32RTE events、sampling API request16。cosine/sineは同じtrajectoryを共有する。全PR不適格なら12compileで終了し、費用0として順位付けしない。

要求signal精度 .05/.01/.005/.001/.0005。P0のshot-weighted full-wrapper RZを主指標とし、CX/RZ depth/CX depth/total depth/circuit sizeを保存する。B0+Iのidentity位相はsignal側へ適用し、full-H referenceは回転しない。finite-RTE law/component選択/normalization/shot会計/既存compilerは維持する。

## 実装と高速化

H5の既存source/tests/resultsを編集せず、新package `src/trottertracks/resource_applicability/h7_open_shell/` とprepare/run scriptsを追加した。SOURCE commitは9file追加だけで、起点の10,493fileに変更・削除0。

- `input.py`：H7のcharge/spin・7×7共通MO・14-mode積分layoutを明示。旧charged odd-chain helperは使わない。DFのweighted投影とarray identityは固定H5実装を再利用。
- `reference.py`：α4/β3のoccupation basisは各35次元でも別々に扱う。H5のSpinPrimitive/S²を再利用し、独立ladder-enumerated CSRのone-body/DF作用を一度準備してfull/truncated/tailで共有する。
- `science.py`：1,225次元のbounded eigsh adapter、Numba/thread1/chunk1の既存production action、疎referenceを接続。H5の400次元solverや100次元dense referenceの制限を削除して流用する方法は採っていない。
- `execution.py`：H7入力と二段階dispatchを接続。実測済みH5のpidfd/own-run停止・ESRCH・wait/reap・FD cleanupと独立coordinator監視を再利用。H7向けwall/driver memory/全stage合計output・inode観測も加えた。

本番経路の独立照合は1,225列を一列ずつ走査し、全体の差のFrobenius normを累積する。1,225×1,225のdense行列・dense expm/eigenvalue列を作らない。CSRのexpm actionとsplit-time/eigen residualを比較し、独立CSR eigshも別の固定generic probeから実行する。
これはengineering最低固有値整合であり、全spin/Fock空間の厳密基底状態certificateではない。独立solverがより低いenergyを返す、doublet gateが成立しない、数値差が既存gateを超える場合はSTOPする。

3種preparation、α/β spectra、時間別phase LRU、content-checked streaming hash、evolutionのcos/sin両軸再利用、fork/COWによるread-only準備共有を維持する。通常group最大4並列・内部thread1、S4は単独。実H7のspeedup倍率・時間・RSSは未測定。

## 人工・mock技術検査

**54検査PASS**：H5回帰34＋H7追加19の53件（32.148秒）、追加の1,225次元Numba chunked作用1件（.481秒）。単一test process・内部thread1、AS8GiB・wall180秒・出力16MiB/fileの人工検査計画内。

- H7のneutral ROHF provider引数、共通MO、積分layoutとsigned/tol-only DFをinjected providerで検査。
- 35×35の異なるα/β basis、phase往復、1,225次元solverの予算・shape・nonfinite・出力workspace aliasを検査。
- 小系の独立dense行列とCSR action/trace/full/truncated/tail/referenceを照合。全column検査の不一致と高spin stateを正しく拒否。
- 1,225次元の非対角・signed・人工6fragmentについて、Numba chunk1/thread1作用と独立CSRを3vectorで照合。
- 14-mode＋ancillaの人工対角S4回路でcos/sinとrelative phaseを検査。非可換小系・Gaussian・S4負時間・PR fixed-K law/native同値・fingerprint/6metricsはH5回帰を保持。
- H7 input stageと20group/40compile二段階coordinatorはmockで接続確認。実workerを起動せず、親AS soft8GiBとhard limitの継承を検査した。親hardを8GiBへ下げてworker32GiBを妨げない。
- H5旧grant拒否、新namespace/seed/source/input binding、first STOP保持、pidfd/ESRCH/wait/reapは限定mock回帰で確認。OSの実H7 end-to-end成功として報告しない。

主担当が実施した技術検査であり、別担当の独立review・CI・外部再現ではない。実H7入力・state・registered primitive全coverage・実compileの正しさは未検証で、起動後のgateとして残す。

## 環境・call・資源案

既存private Pythonを変更せず再利用する。
`/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python`

Python3.11.0rc1 / Qiskit1.3.0 / NumPy1.26.4 / SciPy1.14.1 / Numba0.60.0 / OpenFermion1.6.1 / PySCF2.7.0 / OpenFermion-PySCF0.5。
Compilerはrz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing None、symmetric_directional。
Compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。
環境14,817fileのbyte profile digest `70ebab6aec9b79b21592da2d928f3b68c15ba5b152be52560014596a8dca6992` はH5時と一致。source closure267fileを固定した。旧hostのcompiler output bit完全同一性は未確認。

| 処理 | H7新増分上限 |
|---|---:|
| ROHF / 積分 / DF / DF state solve | 各1 |
| production solver matvec（residual含む） | 10,000 |
| production/CSR全column照合のproduction作用 | 1,225 |
| CSR reference作用（独立solver・expm等を含む） | 100,000 |
| 独立CSR最低固有値solve | 1 |
| full/truncated reference exponential action | 6 |
| ensemble signal | 6 |
| native primitive probe | 実rankのunique時間点×3 |
| tail作用 / 決定論的作用 | 各500,000 |
| full-wrapper compile / PR trajectory / occurrence | 40 / 16 / 32 |
| control probe | 32（初回6構成の予定24を含む保守枠） |

Input前のrank100はcall admission上限の計算用で、rank予測・DF切落し条件ではない。実rankが対応する予約を超えればrawを保持してSTOPし、自動増額しない。
H5成功run40compileと過去partial/予約/unknown、H4/H6/H8の旧lineageはcarry draftに保持する。新H7 actual/reservation0、過去分を0に補完・reset/refundせず、新H7最大40/16/32を別増分とする。H5 partialの再利用はしない。

資源は**未承認の候補・観測STOP閾値であり、必要量の予測や予約ではない**。
Worker最大4CPU＋coordinator1CPU、内部thread1、worker AS hard32GiB/RSS観測9GiB、worker aggregate38GiB、driver AS soft8GiB/RSS2GiB、headroom16GiB、fresh admission56GiB。
実CPU番号は未割当。読み取りsnapshotは有効空きRAM約464GiB・home空き約573GiB・quota KNOWNだが、開始時に再検査する。

Monitor原則1秒/最大5秒、host full PSI1%以上・cgroup PSI非zero・OOM増加・memory不足はfail-closed。Own-runのCPU affinity/pidfdだけを扱う。wall STOP候補4時間は所要時間予測ではない。実際のH7時間/RSSは最初の6group等で測定する。

全output観測STOP1GiB、free2GiB以上、10,000inode、terminal reserve4MiB、worker file32MiB、stage journal64MiB。観測間のaggregate bytes/RSSのovershootまでhard quotaで抑えた保証ではない。raw arraysはrank100の場合でも数MiBの規模だが、checkpoint・ログ・native build/transpileのmemoryは別。グローバルSTOPが個別file上限の単純和より先に作用する。

## 起動前に残ること

1. 利用者による**H7 ROHF/DF/state生成＋この限定pilot**の明示的な実行承認。
2. 固定SOURCEを独立runtime worktreeへcheckout。REVIEW資料commitのHEADから直接起動しない。
3. Source/profile/carry bytes、fresh CPU/core・RAM/pressure/OOM・filesystem/inode/quota、未使用home output/controlを検査。
4. 新H7専用manifest/authorizationをSOURCE/profile/target/compiler/carry/outputへ結合し、exclusive one-shotとnative coordinator wait記録を用意。Python -P -B・thread1・home内cache設定を維持。
5. 承認後のinput生成結果について実rank・weighted DF・S²/reference/native gateを確認し、snapshot byte SHA/array identity/registry/call caps/seedをcost前にbinding。
6. 完了またはfirst STOPで全workerのwait/reapとnative exitを保存しmandatory STOP。次stage・自動retry/resume・q/r/K/rankのfallbackへ進まない。

未実行command template（新grantとfresh gateが必要）：

```bash
<private-python> -P -B <SOURCE-runtime-worktree>/scripts/run_h7_open_shell_pilot.py \
  --execute --manifest <new-H7-manifest.json> --authorization <new-H7-authorization.json>
```

現時点では `approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`、`sealed=false`、`absolute_launch_command=null`。実rank/input/outputが未生成なのでseal済みと扱わない。準備用private証拠は `/home/AbeHiromu/projects/h7-handoff-evidence/20261016/open-shell-readiness/`。

**実H7のROHF/DF/state/reference/signal/trajectory/compileは0。H7は未起動のまま、資料公開後STOPする。** GPU query/use・install/update・共有環境/旧H5/H6/H8/H4の実行変更は行っていない。
