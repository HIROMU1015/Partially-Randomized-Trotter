# H8 B0 anchor：adapter・launcher・実行準備の完成

**固定4構成・8軸の実行adapter、4 worker・2 waveのlauncher、SOURCE/input/profile gate、資源監視・回収を実装した。人工検証とSOURCE sealは完了。実H8の復元・control probe・compile・fit・残差・再会計は未実行で、実行承認は未取得。公開後mandatory STOPする。**

branch：`track-a-h8-b0-anchor-implementation-20261016`。
計画起点：`93bf0bf3420683d9d20755cea856d3814b951233`。
実行SOURCE：`572a337d0dc3fa78b2fd6b25ace6a37845db384e`。最終の報告commitはこのSOURCEの子孫で、実行closureのbytesを変更しない。
[公開契約・無効grant・SOURCE/profile/input照合](../../artifacts/resource_applicability/track_a_h8_b0_anchor_execution_readiness/2026-10-16/README.md)。

## 固定した実測scope

|用途|構成|実測軸|wave|
|---|---|---|---:|
|TRAIN|B0 p14/q1 S2|cosine / sine|1|
|TRAIN|B0 p21/q1 S2|cosine / sine|1|
|TRAIN|B0 p14/q8 S2|cosine / sine|2|
|HOLDOUT|B0 p21/q8 S2|cosine / sine|2|

linear H8、隣接1 Å、STO-3G、16 modes、Nα=Nβ=4、sector4900。保存tol-only DF rank28、tol1e-8、cutoff0、signed generation order、weighted Hermitian projection。T=.8、q1/8、δ=.8/.1。既存の5精度・経験的allowance factor1/10・保存未回転full-H referenceを維持する。

[前段計画](track_a_h8_b0_anchor_acquisition_plan_20261016.md)のJSON SHA `c7755c88…` を実行契約内で固定し、候補、TRAIN/HOLDOUT、call上限、資源、後続Phase境界を変更しない。今回新旧B0を混ぜず、B0+Iによる代用、H6係数移送、旧B0の広域来歴回収を行っていない。

## native adapterと来歴

[adapter](../../scripts/h8_b0_anchor_adapter.py)は新scopeのpermit確認後にだけ科学moduleをimportする。保存snapshotを一回読み、保存exportのfull preparationを一回復元し、p14/p21の二つのviewをq間で共有する。SCF・DF・state solver・再対角化・Gaussian再生成を呼ばない。

復元したGate操作列を再記述し、qubit順序・角度・matrix hashを保存操作記述と比較する。live blockから再構成したordered block SHA、保存root preparation hash、新view recipe SHA、original scalarを別々に検査する。保存hashを単に「観測hash」として転記することはしない。nativeはsymmetric-directional、S2、original constant一回で、tail identity overlayは0。state preparation費用を含めないmeasured full Hadamard wrapperを使う。

各構成で左右branchとX/Y wrapperの最大4 control probeを実施し、保存B0 corrected/independent-forward signalと照合する。非finiteまたは最大誤差1e-8以上ならSTOP。full-H referenceは再生成・回転しない。これは取得に必要なcorrectness検証であり、新signal・shot・biasの科学評価ではない。

各costの記録は新SOURCE、完全profile SHA、preparation/basis/view、scalar/control、compiler hash、wrapperのcompile前後fingerprint、測定軸・実測RZ、子process identity・exit0・wait4 reapを結合する。compile APIに入る前の独立markerも保存する。旧B0 cost receiptや費用値を新測定として採用する経路はない。

## 並列実行・予算・STOP

[launcher](../../scripts/launch_h8_b0_anchors.py)→[driver](../../scripts/run_h8_b0_anchors.py)→owned compile workerの経路を接続した。最大4 worker・thread1、2 wave。親が保持する構成は最大2つ。**各waveの2構成のcontrol検証が両方PASSするまで、そのwaveのcompileを一つも起動しない。**

driverは新scope grantとSOURCEを検査し、実在する監視親のPID/start/parent/UIDにも結合する。workerはparent-death signal、個別CPU/AS/CPU時間、専用cache/tempを設定し、親が正常終了・wait4回収を確認してから結果を受理する。pool原receiptのresult SHA欄だけに依存せず、全task resultのSHAを受理時に検査する。

call上限はsnapshot1、保存decode1、prefix view2、native evolution4、measured wrapper8、probe copy8、control probe16、compile8。fit/residual/reaccount/S4/B0I compile/new signal/full-H reference/SCF/DF/state solver/PR/law/tail scalarは0。予約はfork前にjournalへ記録し、失敗してもrefund/resetしない。STOP時の未完了予約はunknownとして保持し、「実際のcompileが0だった」とは扱わない。

control、basis、scalar、fingerprint、compiler、資源、exit/reap、receipt、予算の失敗で次waveを止め、進行中のown workerだけをterminate/reapする。同じwaveの最大4軸は既に進行中の場合がある。自動retry/resume、追加compile、refitはない。8軸すべて正常でも取得終了でmandatory STOPし、TRAIN/HOLDOUTの測定ファイルを分けて保存する。

資源監視は既存のowned-process supervisorとpoolの固定コードを新scopeから利用する。旧grantや旧whitelistは変更しない。新launcherの最終gateも別receiptに記録し、driver/旧generic supervisorの正常終了だけで全体PASSとしない。

## 起動gateと資源上限

|条件|固定値・検査|
|---|---|
|CPU|driver1＋worker最大4＋observer1、6つの異なる物理core、各thread1。SMT siblingもidleを確認し、階層cgroup quotaが6 CPU未満なら拒否|
|実効空きRAM|aggregate RSS160.25 GiB＋reserve16 GiB＝176.25 GiB以上。host/cgroupの小さい側を使用|
|AS/RSS|driver64/32 GiB、各worker64/32 GiB、observer512/256 MiB|
|CPU/wall|累積24 CPU時間、phase12時間、observer CPU1800秒。所要時間予測ではなくSTOP上限|
|storage|stage8 GiB、output256 MiB、combined logs128 MiB、20,000 entries。fresh空き16 GiB・inode40,000以上|
|instruction|native4,000,000、transpiled10,000,000|
|監視|1秒間隔、gap5秒超でSTOP。host full PSI≥1%、cgroup full PSI>0、OOM増加、未知の観測はSTOP|
|quota|実output filesystemのmount optionsを照合。quota有効・未知のfilesystemは起動せず、無断にstatvfsだけで通さない|

runtimeは完全profile `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`、34,273 files。inventory自体のdigestも固定し、同じ件数の別inventoryへの差し替えを拒否する。native providerは `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、265 files。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/couplingなし、compiler hash `1239304c…` を維持する。

launcher/driverはSOURCEのGit原本・disk、provider、保存input、runtime bytesを科学import前に検査し、driver終了前にも再照合する。長いhash照合後は監視親とのhandshakeでfresh gateを更新し、30秒以内の資源・保全観測を得てから科学処理に入る。fresh gate専用ファイルを新規作成し、config/grant/既存receiptを上書きしない。既存control/output/one-shot lockがあれば起動しない。

今回の準備では上記SOURCE14、provider265、runtime34,273、保存入力7＋font seed1のbyte/hash照合がPASSした。**これは承認後のfresh gateではない。CPU割当は空、fresh資源gateは未発行。6 core・RAM・quota等は承認後、起動直前に再確認し、満たさなければ起動しない。** 他jobの停止・affinity変更・venv修正・CPU数の自動変更で条件を満たした扱いにしない。

## 人工検証と証拠の限界

[unittest](../../tests/test_h8_b0_anchor_execution.py) 17 methodsがPASS。scope/未承認grant/古いfresh gate/不足RAM/未知quota/monitor親identity、wave barrier、次wave抑止、4 workerの故障注入と全回収、資源上限・予約非返却を検査した。旧計画の10 methodsもPASS。

[pinned provider fixture](../../tests/check_h8_b0_anchor_synthetic_provider.py)は2 system qubitだけを使い、8項目PASS。保存操作記述の復元、実際のnative左右branch・X/Y wrapper、独立dense S2とのscalar一回、q1/8と実compilerへの接続を検査した。人工4軸だけをcompileし、最大control差は `5.772695288511e-15`。実H8の入力を開かず、H8 compileは0。既存SOURCEに対するprimary agent自身のlocal技術検証であり、独立レビュー・CI・H8実測成功の証明ではない。

## 承認対象と後続境界

[完成契約](../../artifacts/resource_applicability/track_a_h8_b0_anchor_execution_readiness/2026-10-16/execution_contract_v1.json)と無効grant案を公開した。SOURCEはseal済みだが、`approved_by_user=false`、`runtime_authorization=false`、CPU[]、command=null、valid=falseで、現状のdraftではlauncherが副作用前に拒否する。準備toolに有効grantを発行・承認する機能はない。

次に承認できる範囲は、**この新SOURCEと固定資源に基づくB0 S2の4構成・8軸の取得一回だけ**。取得後のB1は別承認で、3点TRAIN＋p21/q8 HOLDOUTを各軸独立に検証し、相対誤差2%以上ならSTOP。B0+Iは既存16軸を再利用する。旧B1のSTOPをこの準備で解除せず、旧B0を混合せず、HOLDOUTをfitへ入れない。B2のS4最大4軸、Cの5精度再会計も各々別承認で、自動開始しない。

旧673 grant/ledger/carry/one-shot保護ファイル、114旧worktreeのHEAD/status、旧V2未追跡scriptを前後照合した。V1、V2 NO_GO、H5/H7、既存SOURCE/raw/grant/ledgerを保全する。今回の公開は実装・人工検証・無効契約だけで、実H8 Gate復元・control・compile・数値fit・残差・再会計・新有効grantは0。remote SHA/tree確認後mandatory STOP。
