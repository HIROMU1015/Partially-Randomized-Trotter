# H8 A_COST：候補単位の並列投入と保存結果引継ぎの準備

2026-10-18。利用者の「並列実行の準備をして」に基づく**実装・人工検査・契約草案まで**。
現在のH8 runは停止・変更していない。新方式の実科学launch、有効grant発行、preparation復元、
sampling、compile、費用再会計、科学レビューは今回0。既存run自体の計算は進行している。

SOURCE：`87f1f0d528478ad212ba98159db06eec3c78433e`。
branch：`track-a-h8-pr-candidate-parallel-readiness-20261018`。
起点REVIEW：`1e3c8450af50f662c6d2f3451d09714cfebcba4a`。
現H8のSOURCE：`31ef62a12407010e9c1a07f9b37facedb8dff5fe`。
[準備bundle](../../artifacts/resource_applicability/track_a_h8_pr_candidate_parallel/2026-10-18/)。

## 変えた実行順序

旧方式は16候補の代表軌道（replica0）の全件終了後に、残る112軌道へ進む。
新方式は、**各候補の代表軌道がcontrol検証・paired compile・native exit0・同一親によるreapを
終えたら、その候補の残り7軌道を投入可能**にする。他候補の遅い代表軌道を全体の待機条件にしない。
control.jsonが途中で作られただけのworkerは、後続を解放しない。

不足する代表軌道を優先し、RTE event枠R・qが大きいtaskから開始する。同順位はIDで固定する。
これは独立taskの投入順序であり、量子演算順序や乱数列を変えるものではない。
空きslotへ実行可能なtaskを割り当て、未充足の依存だけが残って実行中workerもなければSTOPする。
どれかの検証/workerが失敗した場合は、当該H8 attemptの所有workerだけを回収し、最初の理由を保持する。
PSI・他worktree更新だけで正常workerを停止しない。

`scripts/h8_key_compile_parallel.py`へ後方互換の`ready`/`on_complete` callbackを追加した。
元interfaceの実動作も人工fixtureで確認した。新adapterは自分のSOURCEに固定したpoolを明示loadし、
旧provider snapshot側の同名moduleを誤って使わない。
`scripts/pr_cost_batch_science.py`と`pr_cost_batch.py`へ新policy専用のgateを追加し、
H4/H6・B/Dの通常経路は維持する。

## 引継ぎで再利用できる結果

対象は今回のH8 A_COSTの固定16候補×探索n8、計128 taskだけ。
linear H8/1Å/STO-3G/tol-only DF28/T=.8/P=0、元S2 PR・元RTE law・元compilerを保持する。
旧確認n32や別のgeometry/旧partial/cacheを、この引継ぎへ混入しない。

1. 元SOURCE blob/input bytes/profile/bindingと、元task ID・candidate・replica・seedを照合する。
2. 元coordinatorと全記録workerのidentityが退役し、native parent receiptが全worker回収を示すことを確認する。
   PID再利用は開始identityで区別し、無関係なprocessを停止しない。2回の静止観測とreceipt不変を要求する。
3. 個別workerのexit0/reaped/result SHAを要求し、paired軸・event digest・seed・compiler・fingerprint・
   元入力binding・before-call予約・代表control（4 probe＋1 event oracle）を検査する。
4. 元mapがPASSなら通常受入れ。意図的引継ぎで`OWN_RUN_SIGNAL_15`のSTOPなら、
   明示されたhandoff認可receiptに基づく別の受入れ経路で、証拠の揃った個別workerだけを受け入れる。
   科学的不一致のSTOPを、この経路でPASSへ変換しない。元STOP自体は保持する。
5. result fileだけ、native SHA未記録、非zero exit、途中のcontrol/片軸だけの結果は再利用しない。
   不確定なtaskを再実行対象として残し、旧予約・消費は返却しない。
   記録済みSHAとの不一致は再利用拒否・STOPとし、検査を緩和しない。

既存runの途中結果を今すぐ受け入れたという報告ではない。
元run稼働中の`--finalize`は拒否される。現在は入力/コードの結合と無効草案だけを作成した。

新scheduler SOURCEは上記87f1f0d5、**seed導出SOURCEは元31ef62a1のまま**。
同じ128 taskを続けるための明示的な継続契約であり、新SOURCEから新乱数を引き直さない。
科学kernel `scripts/pr_cost_native.py`は元SOURCEと同じbytes/SHAを要求し、RTE抽選・
native lowering・scalar/control/oracle・paired compile・compiler optionsは変更しない。
各taskの実行SOURCE/元directory/hashとseed導出SOURCEを別々に保存する。
探索n8を新しい独立確認と呼ばず、旧n32へpoolingしない。

## 残call予算と資源

元全体はtrajectory/evolution各128、wrapper256、occurrence736、RTE event3584、
control probe64、event oracle16。これは元task全体の値であり、追加で全件を取得する認可ではない。
切替時に検証済みtaskを除き、残taskだけのcall上限を再計数する。
新attemptはtask起動前に親台帳へ予約し、childの個別予約も残す。
再利用する完了compileは再課金せず、元の失敗・未確定部分はcarryとして保持する。
追加signal・SCF/DF/state生成・fit・量子shot・ランキング・確認samplingは0。

worker数はfresh CPU・有効memory・残task数から選び、CPU番号はまだ認可していない。
元の起動見積り24GiB/worker・parentと、host余裕案32GiBを維持した例では、
15 worker＋parentで416GiB。観測時のMemAvailableは約422GiBで、この見積りでは15 workerが目安だった。
実際の切替前にCPU競合・cgroup・memory・disk・inode・quota方針を再確認し、数を調整する。
15を恒常上限にしない。RSS/AS/wall hard capを新設せず、worker内部thread1・
preparationのprefix単位復元/COW共有・event/evolution共用・fingerprint cacheを保持する。
現在のrunのCPUや他jobの資源割当は変更しない。

## 人工検証・固定資料

新しい人工24検査、既存承認境界18検査、無効/未seal CLI拒否2件がPASS。
遅いB代表の終了前にA後続が実際に走るfixture、失敗時のowned cleanup/reap、
局所gate、元seed保持、live origin拒否、正常native受入れ、SHA/closure/seed改竄拒否、
科学STOP拒否、欠測native hashの不採用/carry、synthetic/real混合拒否を含む。
同時child最大2の最小人工caseだけで、新synthetic compile・実科学callの追加起動は0。
旧40検査/benchmark/campaignを再実行していない。作者の技術確認であり、独立科学レビューではない。
AI subagentは0。

- 実装：`scripts/pr_cost_candidate_parallel.py`
- metadata草案/finalize CLI：`scripts/prepare_pr_candidate_parallel.py`
- 検査：`tests/test_pr_cost_candidate_parallel.py`
- private草案：`/home/AbeHiromu/projects/h8-pr-candidate-parallel-preparation/20261018/private/draft02/`
- 公開する計画：bundleの`plan_UNSEALED.json`と`grant_INVALID.json`
- 固定closure：19 fileのbytes/SHAを`source_closure.json`へ記録

全草案はapproved=false/runtime_authorization=false/allowed_cpus=[]、technical_sealed=false、
resume_manifest=null、absolute_launch_command=null。最終native receiptとhandoff指示が揃ってから
`--finalize`で技術sealを行うが、そのCLI自体は有効grantも起動も行わない。

## 次の担当と現在のrun

準備中の観測では、既存H4/H6 A_COSTは完了、H8は15/128のpaired task完了・control16件PASS。
現在の残作業は進行しており、これらは時刻付きsnapshotである。旧runtime SOURCE/承認grantは保全した。
control全件待ちの解消による利益は、切替時点で旧runがどこまで進んだかに依存する。
新方式の実入力での速度・peak memoryは未測定で、倍率の保証はしない。

**ユーザー**が引継ぎを指示した場合、**Codex**が直前進捗と資源を確認し、旧runの所有範囲だけを
意図的停止・回収、最終receipt検査、残task計数、技術sealと新grant結合、一度の継続実行を行う。
旧runが既に完了していれば追加compileは不要。今回は停止・切替・新方式の科学起動を行わず、準備報告で停止する。
