# 通常入口の自動準備と停止復帰

- JobStore、DistributedExecutor、stage先行関係、保存結果の受入と低優先価格検証を再利用した。
- 最適化器と操作入口の固定SOURCEは `f03f2fcf49a7b7ab726fce5085dfed933787a402`。workerと準備通信adapterは `8313cea128710be5c5d7b6804160cbf42b1c9187`。PBS回収Gatewayは `95a74e4486131a97bd5235221bf2d8ab9f99f591` と既存の専用Python。科学Q3は `28d976b99cdf8329adcc1b79327e329e947ef774`。準備providerと価格workerの固定版は既存manifestから継承した。
- 採用したcurrentは `romutail-auto-preparation-f03f2fcf4`。基点の通常版は `romutail-pbs-current-95a74e448-cpu100-qg230-laptop`。中央の索引とlockは `.rpe-central/state` を使用する。新規runの成果はcontroller checkout内の `search-evidence/<run_id>` から固定Git locatorで参照する。既存runはそれぞれの固定設定を使う。
- signal、compile、auditの新規run既定値は既存の `UNLIMITED` 表現を使う。受入済み契約・探索範囲・選択規則が生成する要求だけが対象。明示した工程予算と0を優先し、Q3予算と別に計上する。Previewは解決した値を表示する。
- `7fd6b2291` のPauli較正view宣言・workspace配置と `5ffac784b` の終了待ち・復旧照合を取り込んだ。未対応mapperは幅family単位で理由を保存する。PFの準備工程の保留は同じqueueで処理中の独立候補へ波及させない。

## 検収

通常Gatewayから隔離人工run `search-0452b9f95e3381757659` を準備予算省略のscopeで開始し、8件を一度ずつ採用した。同じworker SOURCEで、RomuTailは信号2件と監査1件、PBSはcompile 1件とQ3 4件を担当した。PBS `267199.CPUcluster-PRISM` は `workq`、1ノード・2 CPU・4 GB・4分で、最終状態F、exit 0。完了Resumeは `REUSED_COMPLETED` で、追加起動、採用、PBS再投入はいずれも0件。人工の台帳・成果・価格storeは検収専用に置いた。

PBSのPR-Pauli信号・compileについて、保存入力71項目ずつの照合と新規workspaceへの取込み、compileの13ab viewを確認した。見積りは信号120秒、compile 180秒で、drain 30秒を加えて判定する。provider名とSOURCEに一致する工程に適用する。監査・Q3の科学実行は見積り不足でPBS受付外。100 CPUのPBS総上限、qg2 30 CPU、RomuTailとRomulaptopに追加のCPU上限なしを維持した。

RomuTail・qg2・PBS・local0-wslへ固定版を配置。WSLのd887 PR-Pauli信号だけ、Qiskit 1.3.0がある既存のPython 3.12.3に対応付けた。保存REQUESTのruntime要求に一致した。DP並列16、初期1枠autoと端末の受付状態を維持した。server1は復帰時に同じ配置と検収を適用する。

WindowsとRomuTailの通常Previewは明示指定、release省略のどちらも新releaseと同じrun IDを返した。旧登録runのStatusは固定の `romutail-6da8554e0-r4` を返した。関連テストは110件、subtest 10件が通過した。

## 保存runと残る科学条件

親run `search-b1a74e00581b89ea42db` の読取り照合で、採用済みsignal 16、compile stage 8、残枠signal 0・compile 0・audit 8・Q3 32を確認。保存compileは8件とも価格証拠待ちで実回路compile 0。親台帳、終了記録、元成果を保全した。継続時は固定問題、範囲、消費、owner・attempt、STOPを照合し、実行可能な候補を通常の選択規則へ戻す。

最終controllerで継続のPlanOnlyと通常の§10.5照合が通った。保存された候補のうちLD5 / κ58、LD4 / κ64を含む4件が、保存packet・価格・物理scheduleの再照合後に登録できた。継続用scopeのPreviewは `search-a8c8ce3e5d57f171cb97`、新規科学callは0件。

大きい要求幅のPR-DF mapperには非整数PF gridの拒否が残る。別の丸めや分割は採用していない。LD8 descriptorとPR-Pauli LD7は既存providerが不足を識別する経路を使う。各descriptorの新規科学生成とB0Iの新規科学準備は今回の人工検収には含まれない。

この統合作業による新規の実信号、実compile、実Q3、価格fitおよび本番探索は各0件。
