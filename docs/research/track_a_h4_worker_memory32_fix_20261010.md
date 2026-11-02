# H4 worker AS/RSS32GiB修正

利用者の「上限を３２Gで修正して」を、現在の4workersのAS/RSS上限8→32GiBの明示承認として実装した。
driver8GiB、observer AS256MiB/RSS64MiB、headroom16GiB、worker CPUs[2,4,5,6]/driver16/observer18、thread1を維持する。
必要な有効空きメモリは8+4×32+16+0.25=152.25GiB。fresh gateとOwnedRun二次gateの双方が同じ算式を使う。
旧generation・legacy8GiBのdefaultは保持し、32GiBは閉じたmemory_budget_amendmentを持つ新host4workerの経路だけに適用する。
科学条件・compiler options・凍結6入力・17GiB/74805/72h・carry0・pressure条件は不変。

## 実効上限と所有確認

親のhard8を先に固定すると、childは32へ上げられない。親はkernelが強制するsoft8を維持し、全child準備完了まで継承hard ceilingを保持する。
継承hard<32は起動前STOP。workerもsoft8でpermitを読み、認可・所有/CPU index・SOURCE/profile照合後にsoft/hard32を設定してgetrlimitで一致を確認する。
ready後、driverをsoft/hard8へ固定。kernel soft8は起動中も有効であり、起動中driverを無制限にしない。
observerはtrusted driver PIDを8、native登録済workerのPID集合を32で監視する。sampleの順序や未認証role名で上限を選ばない。
未登録/欠測/退出/所有不一致・32超過・driver8超過・observer上限・5秒違反はfail-closed。
元のpressure profileはhost-only<1%/nonroot全0/OOM基準一致/例外時120.25GiB、zero-host headroom16GiBを保持。これとは別に起動時の空き条件を152.25へ上げる。

## cleanup修正と限定検査

独立reviewが、最終driver clamp失敗後のcleanup signal errorでpool/pipe/observer/budget掃除が飛ぶP2を指摘した。
全cleanup actionを試み、元の設定/停止errorを保持する。poolも最初のcleanup errorを保持し、全child wait/pipe/executor掃除後にraiseする。nonESRCHや所有不一致を成功扱いしない。
SOURCE b652fff9d2907015d3b8b7b23ce8bf0fe4fce33a、closure65。先行SOURCE5b0e671a7a5993fe75bfb9f09dff4e77595c5f5dは履歴として保持する。
最終53限定検査PASS（fail/error/skip0）、wall1.481438秒、peakRSS30,932,992B。単一test process/thread1、AS256MiB/RSS128MiB/wall45秒/output4MiBを事前固定。
継承hard8拒否、soft8/hard32保持、認可後worker32/全ready後driver8、role別AS/RSS、152.25境界、部分spawn/ready/clamp失敗、signal/wait/FD error後の掃除、旧pressure33をmockで検査。
実worker/observer/child/affinity変更・科学array・追加transpile・本計算0。初回tests-01はTHREAD_ENV不一致で検査前STOP、tests-02は51PASS、P2修正後tests-03は53PASS。全logを保持する。
これを全74784wrapperの完走保証、元native signalの確定、旧compiler完全同一性とは扱わない。run07で確認したstderr破棄とformatter二次MemoryErrorの診断2経路は今回未修正。

## SOURCEと準備binding

[固定SOURCE/旧new blob・hash](../../artifacts/resource_applicability/track_a_h4_worker_memory32_fix/2026-10-10/source_audit_v1.json)、
[32GiB profile](../../artifacts/resource_applicability/track_a_h4_worker_memory32_fix/2026-10-10/worker_memory_profile_v1.json)、
[利用者の承認](../../artifacts/resource_applicability/track_a_h4_worker_memory32_fix/2026-10-10/USER_MEMORY_AUTHORITY_v1.json)を一組として読む。
旧SOURCE62の変更7/不変55、新module/runner/test3でclosure65。SOURCE65のGit blob/checkout/byte SHAを照合した。
[準備binding](../../artifacts/resource_applicability/track_a_h4_worker_memory32_fix/2026-10-10/preparation_binding_v1.json)はSOURCE・固定environment/compiler/cache/pressure profile・凍結input identity/freeze・carry0・cap32/admission152.25を結合する。
run07開始前の旧stop_evidence_receiptに加え、latest_stopped_attempt_receiptへrun07停止要約・元native receiptのbyteSHA参照を保存する。将来seal時は新runの停止証拠へこの最新証拠を組み込む。
旧profile4件のbyteSHA不変。入力再生成・科学array読込・受領済proof campaign反復はしない。venv/install/共有設定/他job/GPU変更0。
新SOURCEは将来のseed identityへ再結合し、旧partial/random/cacheと混合しない。
旧run07の消費済one-shot/output/controlとlaunch認可を新SOURCEへ使い回さない。overall approved/runtime=false、allowed_cpus=[]、未seal、absolute_launch_command=null。
次回の起動ではfresh unique run/output/controlとSOURCE/profile/input/carry/plan/auth/reviewを再結合し、fresh CPU/memory/pressure/OOM/容量/inode/quota検査を行う。
今回の指示に基づくworker32承認は保持し、上限承認を再質問する必要はない。本計算は停止中、今回の修正turnでは起動しない。
[資料・独立review入口](../../artifacts/resource_applicability/track_a_h4_worker_memory32_fix/2026-10-10/README.md)。

独立最終review v2はPASS、blocking/nonblocking実装指摘0。SOURCE65と現在のbinding・最新run07停止証拠参照・53回帰を照合した。
新runへのlaunch signoffは未発行であり、全map完走は未検証。
[最終review v2](../../artifacts/resource_applicability/track_a_h4_worker_memory32_fix/2026-10-10/independent_worker_memory32_review_v2.json)。
