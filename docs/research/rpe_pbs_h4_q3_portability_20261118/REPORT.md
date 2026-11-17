# 実H4 Q3一件のPBS移植検収（2026-11-18 JST）

**移植検収PASS。実Q3の新規起動1、完了1、中央ACCEPT1、ADOPT1。**
元OUTCOMEとPBS OUTCOMEは、exactなMSE上下界・判定・closure・identityがすべて一致した。
OUTCOME全体の差は`finished_UTC`だけである。[exact比較](results/EXACT_COMPARISON.json)。
同じrunの再開による追加Q3、二重採用、科学予算追加消費、PBS再投入はいずれも0。
検収workerはEXITED、PBS267197はF/exit0、自分の残存PBSジョブは0。
信号取得・回路compile・価格fitは0件。共有currentと他担当のrun・relayは操作していない。

## 選定と固定版

専用branch：`codex/rpe-pbs-h4-q3-portability-20261118`。
実行前のcaller・入力・選定・設定・dry-checkはcommit `5040a1813`で固定した。
中央・workerの実装は登録済みrelease `romutail-pbs-integrated-925985b4e`の
**`925985b4ea76a15f8a07c507179d4fa6c492eabd`**。
科学Q3 checkoutは**`28d976b99cdf8329adcc1b79327e329e947ef774`**、
timed adapterは`349558b748a82f384a4589434a63ef9b5572032c`、
route/providerは元REQUESTの`pf_b_pf6` / `cbc471c913072de45c25a0ed22d51dff8a9b7034`を使った。
元REQUESTの`optimizer_source=53bb0a1d7b1da70644da1c22552224291205af78`は履歴identityとして保持した。
それを今回のcontroller SOURCEと呼び替えていない。

着手時の公開中央branchは674dc7874、価格branchは8ba82646eで、staging Gatewayが統合作業中だった。
既存中央run4件はplan-only。自分のPBSジョブ0、専用spoolの既存8 assignmentはすべて人工だった。
releaseのSOURCE・配置HEAD・SERVICE fingerprint・固定設定13ファイルを照合した。
[環境確認](environment/READ_ONLY_PRELAUNCH_CHECKS.json)、[重複照合](selection/DUPLICATE_CHECK.json)。
終了後、中央担当側のcurrent採用が`romutail-price-fleet-e917165ba`、実装SOURCE
`e917165ba1f36b7627614cd25e88b5108d9b20b9`へ進んだことを読取確認した。
今回の明示925 runはその変更を取り込まず、925のまま完了した。
[終了時の中央状況](environment/FINAL_CENTRAL_PROGRESS.json)。このPBS担当によるregistry/current書込みは0である。
参照できた共通指示10-23/10-25版には§10.5がなく、読了したとは扱っていない。
利用者が明示した同条件の保存・実行中成果の重複照合と、参照可能な再利用規則を適用した。

保存REQUEST5,372件と正常終了記録5,332件の技術的な棚卸しから、保存wall timeが最短のFullPFを選んだ。
元の完全入力・exact OUTCOME・資源記録が公開commitにそろっている。
他方式との科学的な優劣による選定ではない。[選定理由](selection/README.md)。

| 項目 | 選択した値 |
|---|---|
| 方式・公式 | FULL_PF / `4th(m5_best)` |
| configuration | `70f7b31744ecd87fe277511bad7d1fc7dace9b1a097c315729c566f706c75edd` |
| 元job ID | `98693b829295fc884330` |
| N_X = N_Y | `[30,28,26,24,22,20,17,15,13,11,9,5,3]` |
| 保存価格によるcost | `497219638 / 1` |
| 評価contract digest | `2e3dac22b1065151` |
| 系・geometry・basis | linear H4、1 Å、STO-3G |
| DF・split | tol-only `1e-8`、観測rank10、`L_D=10`、FULL_H、random tailなし |
| 時間窓 | 保存binary64の`h≈0.1`、要求PF幅`16h≈1.6`、round `m=0..12`。各roundの実幅は保存値を維持 |

decoder、IID binomial law、all-event squared energy loss、閾値、保存信号・価格・モデル根拠は
[評価契約](selection/EVALUATION_CONTRACT.json)と[科学条件](selection/SCIENTIFIC_CONDITIONS.json)をそのまま使用した。
新信号、Hamiltonian生成、回路build/compile、価格fitは行っていない。

## 通常エンジンでの実行・回収・採用

正本はRomuTailの`.rpe-central/state/runs/pbs-h4-q3-portability-20261118/output`。
専用SERVICE/ROUNDは登録済みSOURCEを固定した一件用overlayで、登録済みSERVICEの書換えではない。
元PROBLEM、元REQUEST、Q3予算`{FULL_PF:1}`、他工程予算0、PBSのみ、同時Q3数1、旧結果importなしにした。
入力原本のraw hashとcanonical hashは[実行前記録](PRELAUNCH.json)に保存した。

固定925の通常`scripts/run_rpe_schedule_search.py:main(driver_factory=...)`から、
既存`make_executor`、DistributedExecutor、JobStore、PBS manager、worker、通常のingest/adoptを使った。
外部technical callerは[check_rpe_pbs_saved_q3.py](../../../scripts/check_rpe_pbs_saved_q3.py)。
元の科学job IDを変えず、新しいattempt `18c2c7bc2d7b4a5581ef2543b1359e1b`を追跡した。
保存OUTCOMEは比較専用で、実行前の回答源・prior runとして渡していない。
DP cacheも専用の空namespaceから開始し、実測はMISS・再利用0 round・本体START/END各1だった。

925の通常Windows/Gateway Start CLIには「保存完全REQUESTを強制再評価1件」の指定がない。
今回は既存のプログラマティック入口hookでその有限タスクを渡した。
**未変更のWindows/Gateway Startから実H4を開始した、またはfull探索の次候補選択まで検収した、とは主張しない。**
人工controller用hookは使用していない。共有RUNS/CURRENT_TARGET・currentを変更せず、専用runを中央state配下に保持した。

完全REQUESTの事前metadata検査はPBS267196（8CPU/8GB/3分）、CPUcluster035でPASSしF/exit0。
実workerはPBS267197、CPUcluster031の割当で起動し、固定SOURCE・入力・route/runtime・実割当を確認した。
管理サーバーで科学workerを動かしていない。
実行→数値gate→保存→共有home spool→RomuTail回収→hash照合→ACCEPT/ADOPTを完了した。
closureは`VERIFIED_FINITE_BINARY64_SUPPORT`、checked pairs110,670で元と一致。
supervisorのerrors/monitor_errorsは空、記録された子process95件のexit codeはすべて0だった。

元判定と今回の判定はともに`MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD`。
**移植性の検収PASSであり、科学的な閾値通過・十分N・物理的ground-energy認証・方式優位性の新しいGOではない。**
価格storeは専用で0モデル、通常の科学比較表への追加登録は0。既存科学評価の二重登録を避けた。

## 固定locatorとexact比較

元OUTCOMEはcommit `13326577297673151fd9735e5ee060c20c6d06f5`の
`docs/research/h4_pf_optimizer_search_20261116/run01/FULL_PF/jobs/FULL_PF/98693b829295fc884330/OUTCOME.json`。
SHA256は`1102d87631bce963df7ea917ea02fefe2672761f5b6c3553d00c2f1e4d52e5b2`。

PBS OUTCOMEはこの公開版の[results/run/jobs/FULL_PF/98693b829295fc884330/OUTCOME.json](results/run/jobs/FULL_PF/98693b829295fc884330/OUTCOME.json)。
SHA256は`68e9ee56923e6d8cee8af92b897552a30dcb944246c33c1d50255b589c13ab1f`。
元REQUESTのraw SHA256は`b83d89b37c208ce20b510fab80721b7b44a4ed88ef8dbb414852e8d72c75486d`、
canonical SHA256は`a499fd8f2e1006bae4ba128261c479f310c0e24228f7dde7e5bde81bfbfd5d97`。
原本と検収の両OUTCOMEに巨大なexact numerator/denominatorを保存し、decimal丸めを介さずdictを直接比較した。
REQUEST、shot配分、configuration、評価契約、信号binding、SOURCE/input hashも一致。
[原本locator/hash](selection/ORIGINAL_FILE_LOCATORS.json)、[回収原本一覧](results/PUBLISHED_RUN_FILES.json)。

## 資源・計時

要求はworkq、1node、8CPU、8GB、15分。DP8×library threads1、同時Q3数1。
元の8並列とsampled peak1,764,976 KiBを根拠に、task memory6GiB、割当8GiBを確保した。
`task_seconds.q3=480`、drain60秒。workerはマシン全96 CPUではなくqstatの実割当8CPU/8GiBを上限にした。
設定根拠は[RESOURCE_PLAN](deployment/RESOURCE_PLAN.json)。本番探索の最適資源設定を認定した値ではない。

| 区間・資源 | 測定値または上限 |
|---|---:|
| タスク予約READY→PBS投入意図 | 1.275秒 |
| PBS投入意図→worker起動 | ≤3.321秒（中央が最初のworker pollを受けた上限） |
| worker開始→Q3本体 | 13.057秒（同じ計算ノード時計） |
| workerのtask durable保存→本体 | 7.426秒 |
| Q3本体 | 49.641秒 |
| adapter execute全体 | 55.650秒（元ホスト58.225秒） |
| supervised route全体 | 52.172秒 |
| sampled process-tree peak RSS | 1,603,456 KiB、約1.529 GiB（元1,764,976 KiB） |
| process-tree CPU | 74.844 CPU秒（APIと成果保存まで） |
| 本体終了→bundle ready | 2.369秒 |
| 成果保存→中央採用 | ≤17.457秒（既存のB+C envelopeと中央採用時間による保守的上限） |
| 中央receive→executor ADOPT | 0.154秒 |
| 中央の検査・保存 | 0.148秒 |
| worker upload→ACK | 3.037秒（同じworker時計） |
| 初回の不足入力配置 | 71,096 bytes、中央4.363秒／cluster0.242秒、SOURCE配置は再利用 |
| 回収した成果payload | 11,886,494 bytes、36ファイル（wire bytesは欠測） |
| 新規DP cache保存 | 48,474,848 bytes、0.749秒、再利用0 round |

異なるhostのepochを直接差し引いていない。PBSのqtime/stimeは秒精度で同値だったが、厳密な待ち時間0とは扱わない。
本体とadapter全体は別区間。元ホストには本体clockの保存がなく、単発・異なるhost/runtime/cache条件なのでspeedupは主張しない。
RSSはサンプリングしたprocess群の合計で、共有ページを重複計上し得る。厳密なpeak CPUは欠測。
PBSのbilling値はcput80秒・wall105秒・mem63,608KBで、後者をprocess-tree peakへ置換しない。
[計時・資源・cache](results/MEASUREMENTS.json)、[既存計時器の集計](results/EXISTING_TIMING_SUMMARY.json)。

## 再開、終了後回収と残る不足

完了後に同じcaller・SERVICE・出力で再開し、JobStoreの予約/割当/開始/採用・attempt・OUTCOME hash・PBS tokenがすべて不変だった。
[再開差分](results/RESUME.json)。終了後にも既存gatewayのharvestで元attempt36ファイルを読み出し、すべて中央回収済み原本と一致した。
[終了後harvest](results/AFTER_EXIT_HARVEST.json)、[最終PBS/worker](results/FINAL_PBS.json)、[残存確認](results/FINAL_OWN_JOBS.json)。
managerの保存STATEにあるRはcontroller終了前の最終pollであり、終了確認は別保存のFINAL_SNAPSHOTを正本にする。

終了後harvestの初回だけsystem Pythonの4300桁制限により、7168桁のexact整数のJSON読込みが失敗した。
通常の最初の回収・採用はすでに完了していた。回収clientだけを既存専用Python3.11.0rc1へ向け、
同じSOURCE925のgateway・同じallocation/attemptから回収した。再計算・再採用・固定run設定の変更は0。
[復旧記録](results/POST_EXIT_HARVEST_RECOVERY.json)。中央担当にはこのgateway runtime設定を引き継ぐ。

今回実確認したのは、このFullPF・M5・保存REQUEST・runtime・PBS経路の一件。
TRUNCATED_PF、PR_PAULI、PR_DFの実Q3、実信号、実回路compile、価格fit、別候補・別配分・full探索、
通常Windows/Gateway CLIからの実H4起動は今回の検収範囲外。
他方式の過去metadata PASSを、今回の実Q3 PASSに広げない。

選定/環境/コード/結果の読取照合とcaller実装はsub-agentへ分担し、実Q3の起動・監視・回収・再開は親Codexが実施した。
人工テストは4件・6 subtests PASS。sub-agentの照合は技術的な成果同一性・scope確認であり、重要な科学レビューではない。
[担当と検査範囲](results/TECHNICAL_CHECKS.json)。
次は中央統合担当が設定とこの公開commitを取り込む。共有currentの切替は同担当に委ねる。
追加の科学Q3・他方式・他工程は、利用者の別の採択指示がある場合に担当Codexが固定scopeで実施する。
