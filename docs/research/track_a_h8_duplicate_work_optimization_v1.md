# H8重複処理削減・ケース並列化の実装

2026-10-13 JST。利用者の「その実装を行って」を根拠に、稼働中のH8 feasibilityを変更せず別worktreeへ実装した。科学計算・新trajectory・実transpile・切替・旧run停止は行わない。

SOURCE `f7c461356eacea091261388af58b44dbb96121c1`、起点REVIEW `22197f9901c37d98889fb0df3fbfac081de7680b`、branch `track-a-h8-duplicate-work-optimization-20261013`。旧SOURCE5c207fb8のruntime245件を全件保全、新4moduleを含むruntime closure249件、test/benchmark2件を別hashへ固定する。profile4e32bb9b・input d4c43461・rank28/sector4900・compiler1239304cを準備bindingに記録するが、science launch seal/grantではない。

## ハッシュ

`h8_hash_stream_reuse_v1.py`は旧全体guard prewalkを除き、canonical streaming中で現在のoperationを検査する。旧content-key付きoperation payload cacheと、bit位置payloadの上限付き再利用を使う。SHA256へ与えるJSON canonical v2 bytesは同一で、matrix dtype/shape、phase、control state、順序、signed zero、非finite STOPを維持する。top-level全payloadを作らず、operation cache最大32MiB・wire cache最大256KiB、object IDの跨call再利用0。旧policyと同様にcall中の回路変更は認可しない。

不変回路のfinal digestを無検査で返すcacheは追加しない。graphを毎回検査し、反復hashも正しいbytesをSHAへ与える。初回hashの余分な走査を減らす一方、既に温まった小fixtureのhashは旧方式と同程度であり、常に速いとは主張しない。

## control/X/Y検証

`h8_control_reuse_v1.py`は実回路Eに対するa=E|psi,0>・b=E|psi,1>を二回実シミュレーションし、(a+b)/sqrt(2)にHadamardの末尾HまたはSdg/Hを作用させる。グローバル位相を含む線形性を使い、stateを再正規化しない。low branch identity/high leakage、固定signal、ordinary/directional一致、X/Y expectationの検査を保持する。

旧方式のexpensive E作用4回（両branch＋両wrapper）を2回に減らし、validation wrapperの構築・hashも2回省く。logical control_probeは4のまま、実計算は2full-evolution/2small-suffixとして別記録する。検証方式は変わるため、将来のplan/grantに`validation_method_changed=true`と手法を明記する。旧runの独立full-wrapper検証を実施した証拠に置き換えない。

`h8_fast_cost_port_v1.cost_port`へこの検証と新hashを接続した。旧sampling・paired event・constant/control/gate順・single-owner wrapper・compiler・metrics・call予約・before-call記録は保持する。cosine/sine compileは引き続き別2callで、コンパイル済み費用値の再利用はしない。失敗時STOP・GC/ownershipは既存compile pathを使う。現runnerは旧portのまま、新adapterを起動するrunner/grantは作成していない。

## ケース並列化

`h8_fast_parallel_plan_v1.case_waves`は元task/index/seedをコピーし、memory-homogeneousなwaveを作る。未測定/partial/STOP receiptはheavy、normalの候補にするのはcomplete＋全reap＋source/input/compiler binding済み最終peakだけ。RSS/ASに25%余白を計算し、normal RSS9GiBは最大8、heavy RSS24GiBは最大3、各AS32GiB、aggregate76GiB/coordinator＋margin4GiB/admission96GiBを越えない。

CPU20/21/24–29・co22/23は提案で、allowed_cpus=[]/approved=false/runtime_authorization=falseのまま。8実workerを起動していない。memory receiptは分類用で、complete-result reuseの認定ではない。初回heavyの途中RSSをnormal分類や全campaignの保証に使わない。future fresh資源・headroom・quota/FS/inode・monitor・STOP/own cleanup・実測classificationを結合してから実行契約へ接続する。

## 検証と測定

[44local artificial/mock tests](../../artifacts/resource_applicability/track_a_h8_duplicate_work_optimization_v1/2026-10-13/test_result.txt)が8.70秒でPASS。実input/NPZ/科学sampling/実transpile0、single test process/thread1・AS8GiB・wall180秒・file8MiB、cache/tempはhomeのみ。

検査はcanonical bytes/digest、256×256人工complex parameter、nested/custom gate、signed zero/dtype/shape/control/phase/ordering/condition、跨call mutation、非finite・symbolic/recursive拒否、cache上限、alias restoration、class/definition override、2/4/16-mode人工full-X/Y回路のglobal phase/ancilla混合、control budget/failure、mock compiler接続、同一trajectory/seed/metrics/compile input、wave ownershipと資源算術を含む。実compilerをmockで置き換えた同一性は実transpile成功・実環境完全同一性の証明ではない。

[microbenchmark](../../artifacts/resource_applicability/track_a_h8_duplicate_work_optimization_v1/2026-10-13/synthetic_benchmark.json)：人工2400instructions、各3回中央値。初回hash旧.4273秒/新.1566秒（2.73倍）、warm旧.1483秒/新.1553秒（.95倍、約4.7%遅い局所結果）。8-system人工control181instructionsの状態計算は約2.00倍、state最大差6.06e-16。旧wrapper構築/hashを時間比較から除いたため、状態計算部分だけの比である。実H8全体の倍率・必要RAM・完了時間は未測定、今回の値をその予測へ直接掛けない。

## 次の適用条件

現H8 run/one-shot lock/grant/SOURCE/outputはそのまま。新SOURCEの切替やretry/resumeは今回実施しない。本格H8も未認可。次の実行準備でadapter選択・新source closure・profile/input・新validation method・task/seed/namespace・carry・新grant/output・fresh gateを結合する。現在のSOURCE連動seedを新SHAから勝手に生成し直さず、同一実験の引継ぎでは固定task/seedを保全する。SOURCE/validation変更や費用系列の混合条件を明示する。計算予算の返却/resetはしない。

[一つの資料入口](../../artifacts/resource_applicability/track_a_h8_duplicate_work_optimization_v1/2026-10-13/README.md)。共有環境・既存venv・他job・H4/source/inputを変更せず、GPU query/use0。
