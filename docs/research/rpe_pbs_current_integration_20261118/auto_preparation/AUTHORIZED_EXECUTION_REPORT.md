# 指定範囲の自動準備と実探索

中央正本はRomuTailの `.rpe-central/state`。専用の作業treeと既存の研究branchで変更を管理した。継続前に§10.5の照合を行い、Q3と準備成果の再利用を個別に判定した。親runのSTOP・予算台帳・実行attempt・原結果を保持した。

## 固定版と検収

| 用途 | 固定版 |
|---|---|
| PR-Pauliの実行 | `romutail-auto-preparation-04191fec4`、controller `04191fec461a3040816946bec5db2fc2fe0ed029` |
| B0Iの最終準備 | `romutail-auto-preparation-02bf08a51`、controller `02bf08a51235e52c36ab9c8d0a1ceb9be3364e2c` |
| B0IのQ3 | `romutail-auto-preparation-ca8030e68`、controller `ca8030e689c7e8de2d2af63212b5805e63424faa` |
| PR-DFの実行 | `romutail-auto-preparation-2584454cd`、controller `2584454cd5b107e98dd511ff8b5c8d5f7a7385df` |
| 最終current | `romutail-auto-preparation-a1d4b45aa`、controller `a1d4b45aad61bb3e6efc6ff9ff2184cac7a544ee` |
| 科学Q3とtimed adapter | `28d976b99cdf8329adcc1b79327e329e947ef774`、`349558b748a82f384a4589434a63ef9b5572032c` |
| 準備wrapper、通信worker、価格worker | `d887abdcadf47a969aa2f69c1dd3bdc88c20f1c2`、`8313cea128710be5c5d7b6804160cbf42b1c9187`、`a18d5a04df5ec8c789f274730bff87a2c0a7cc89` |

[工程別manifest](AUTHORIZED_RELEASE_COMPONENTS.json)にprovider、runtime、hostとの対応を記録した。RomuTail、qg2、local0-wsl、PBSでは既存環境と固定viewを再利用した。server1は停止扱い。PBS総上限100 CPU、qg2は30 CPU、RomuTailとRomulaptopは追加のCPU上限なし。Romulaptopは電源と受付状態に従って使用する。従来の同時Q3枠auto・初期1枠・DP並列16を保持した。

WindowsからWSLへ検証ファイルを渡し、200015バイトの転送前後でSHAを照合した。固定Git入力の不足分198オブジェクトをWSLとqg2の作業用viewに補い、PR-Pauliの4候補で同じ保存入力の事前照合が通った。長い引数をWindowsのコマンド行に積まず、WSLが転送済みファイルを開く。成果の取込みでは保存commitの兄弟履歴も移送する。B0Iの受理済み較正とPR-DFの固定providerをこの経路で再利用した。

currentを `romutail-auto-preparation-f03f2fcf4` から `romutail-auto-preparation-02bf08a51`、`romutail-auto-preparation-2584454cd` を経て最終版へ切り替えた。最終切替では実行中LD7のcontrollerとrelayの所有を確認し、29 runの固定設定のSHAが変わらないことを照合した。WindowsとRomuTailで明示版・版省略が同じ `search-c810f9a73bfed5090f53` を返した。工程予算省略時の `UNLIMITED` を確認し、Previewの起動は0件。Windowsの通常Statusは進行中B0Iの固定版を返した。前段releaseの関連テスト90件とsubtest 8件、最終差分のテスト60件とsubtest 8件が通過した。各対象で既存fixture条件によるskipは7件。PR-Pauliの新規split descriptorは共通の準備queueに接続し、人工入力で経路を検査した。

## PR-Pauli

親 `search-b1a74e00581b89ea42db` の停止後、継続 `search-9263482821b08a34a00a` でRomuTailが実Q3を32件開始・完了し、32件を一度ずつ採用した。未完了・失敗・残Q3枠は各0。対象はLD4～7、絶対κの両端込み [2,64]、幅h/2h/4h/8h、shot auto。評価に進んだのはLD4～5の8候補、幅h。LD5・κ58は8件、LD4・κ64は1件を採用した。保存成果で入力・価格・scheduleを照合済みの4候補を登録し、信号を再計算せずに開始した。新規準備の投入は0。診断で価格・信号不足が示された候補はQ3枠の対象外となった。

採用数はLD4でκ2が2、κ13とκ64が各1。LD5でκ2とκ7が各3、κ33が10、κ約48.610549が4、κ58が8。指定の2候補を準備済みの優先登録に含め、各候補の実行順は既存の選択規則が決めた。

親台帳のsignal 16・compile stage 8の消費を引き継いだ。旧compileの実wrapperは0件。工程上限はsignal 16・compile 8・audit 8で、追加の準備枠は0。枠の残りはsignal 0・compile 0・audit 8。[中央の公開結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/283b23c3c6d5222eba89535c7c9530b19517d73b/docs/research/romutail_controller_runs/search-9263482821b08a34a00a/SEARCH_PUBLICATION.json)

## B0I

元計画 `search-6c38979a9c8d450af7f5` の登録済み36構成とQ3上限16を保持した。範囲はp6/7/8、2nd・4th・4th(m5_best)、幅h/2h/4h/8h、shot auto。保存p7・Yoshida4・32h/5は比較点として取り込んだ。

通常の選択規則で最初に開いた候補はp8・4th(m5_best)・8h。信号13点のうち保存5点を再利用し、RomuTailで8点を実計算した。qg2で較正12wrapperを実compileし、C監査は12件を受理した。保存した較正と監査成果からモデルの入力を復元し、qg2で独立holdoutの2wrapperを実compileした。RomuTailでholdoutの監査2件が通り、価格確定stageが `PRICES_READY` を返した。holdout原結果は `9b6c39f5c4b1c712e11d547d2f653e77feba9fd0`。

`search-62da44b27f99dc95511b` では価格確定後のQ3入力登録が旧hostの固定パスで停止した。RomuTailに配置した同じ固定B binderから登録を復旧した。保存Q3テンプレート `c8a2c017f01ef5b6d0d8267c9f63084f84538384` のGitオブジェクトを、Q3の固定checkoutと対象workerへ配布した。Q3の継続runは `search-8c5b56e7d2e6d5d599c7`。復旧の準備stageと新規科学準備はいずれも0。

Romulaptopは当初のB0I自動配布がタイムアウトしたため、この候補に限定したGit差分packを一度配置した。一部既存オブジェクトを含む約97 MBの転送だった。以降、同じ固定入力で通常のdry-checkが通った。

実評価はp8・4th(m5_best)・8hのshot auto。RomuTailで実Q3 16件を開始・完了・採用し、予算上限で正常終了した。失敗・未完了・残枠は各0。他の35構成には新規Q3を投入していない。qg2とRomulaptopでも実入力dry-checkを通し、Q3の実行先は全件RomuTailだった。current切替後の通常Resumeは開始時の固定版で `REUSED_COMPLETED` を返し、追加起動・追加採用は各0。[中央の公開結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/568599263a9ddf3cc9157aaa7bb28b9a8a5beeb9/docs/research/romutail_controller_runs/search-8c5b56e7d2e6d5d599c7/SEARCH_PUBLICATION.json)

追加準備の累計上限はsignal 1・compile 3・audit 5 stage。最初に不足した信号1、較正と独立holdoutのcompile 2、監査4を見積もり、固定入力が欠落した技術失敗分のcompile 1とaudit 1を別に追加した。開始・消費は1/3/5、成功は1/2/4、失敗は0/1/1、待機と残枠は各0。元の失敗分は消費のまま保持した。較正原結果は `805f4db414cb636b218f5d2f9dae6789ce5a21da`、C監査の価格証拠は `56957d16f631ec1b0f358bd46350d5b29f677bec`。

## PR-DFとFull PF

PR-DFの対象はLD7～9、幅h/2h/4h/8h、保存計画のκ規則とshot条件、Q3上限16。LD8のdescriptorは保存blobから取得し、固定入力と照合した。旧descriptor stage 4件は固定view欠落により開始前に失敗し、消費は保存した。LD8・κ1・hでRomuTailが実信号13点を取得し、原packet `0a05f02c53c56bb064b0f42e93177792c91c4e75` を次の継続で再利用した。候補に必要な準備を既存依存関係に従って補う設定を使い、準備枠は数値上限なしの `UNLIMITED`。各stageの消費はrunの中央台帳に残る。

PR-DFの価格runtimeは保存設定のSHAと封印済みprovider/runtime 278ファイルを照合した。中央の既存cacheと各hostのprofileから実パスを解決する。前段 `search-e3288a841e437a13ab6a` の較正compile stage 1件はprovider依存のimportで回路を作る前に失敗した。元の消費とattemptを記録したまま、同じ封印済みproviderから不足モジュールを解決した。

LD8・幅h・κ1の `search-0071392407117fd5e51d` は固定native価格providerの検査で停止した。保存したdescriptorを受け入れた一方で、価格providerの準備ハッシュまたは正確なtail λの一致条件を満たせなかった。較正回路の開始前の停止で実compileは0件。科学条件と検査は維持した。

LD9・幅hの保存構成は独立した継続 `search-541575a96257b73ce1ac` で処理した。保存較正のm6の8軸を再利用した。m0の4経路・8軸はSTARTEDを記録した後に `STOP_PARTIAL`。実compileの開始・完了は各0。旧bootstrapがProcessPoolに渡す固定関数をpickleできないことを保存設定で再現し、最終版で固定関数の転送と人工関数の実行を検収した。原STOPと4経路のSTARTEDを保持し、同じ要求の再投入は保留した。比較価格の再fitや幅の調整は行っていない。

2h・4h・8hの9セルは、固定mapperが要求するPF gridが非整数になるため保留した。LD7・幅hはR06の保存比較を取り込み、旧論理routeとホストprofileを分けて供給した。新旧routeの差分がhostのパス2項目だけであることを照合し、`search-4bd3a237a18ed0429847` の§10.5で保存Q3 7件が再利用可になった。RomuTail・qg2・Romulaptopで保存入力の事前照合を通過した。

PR-DFの準備はRomuTailで信号stage開始5・成功1・失敗4、compile stage開始3・成功0・失敗3、audit開始0。実信号13点、実compile 0。LD7継続で追加の準備stageは0。実評価はLD7・2nd・幅h・保存κ約24.305275で、shot autoの新規Q3 16件をRomuTailで開始・完了・採用し、予算上限で正常終了した。科学評価jobの失敗・未完了・残Q3枠は各0。元のR06 jobは再利用7件に含まれる。[中央の公開結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/390a7b6206d89a8e52889df52aadda4e1dcf792a/docs/research/romutail_controller_runs/search-4bd3a237a18ed0429847/SEARCH_PUBLICATION.json)

Full PFのM5・16hは保存した比較点を使用した。新規16h探索とFull PFの新規科学実行は0。比較点の物理条件と固定locatorは [保存比較点](AUTHORIZED_SAVED_COMPARISONS.json) に記録した。

## 公開と再開

PR-Pauliの公開結果をCの通常同期で取り込み、Windowsの実閲覧ファイルまで確認した。Cの確認済みcommitは `fab7484bb3df0ecd2fced3d6e9b097a9b7b86de8`。Windowsは `search-9263482821b08a34a00a` を `FINAL_RESULT_PUBLISHED` として表示した。閲覧先は `C:\Users\romuk\OneDrive\ドキュメント\ChatGPT\研究\H4-search-map\index.html`。

B0IのWindows同期タスクは開始時に起動し、実行中の状態も確認した。C `00860e8462384f510bab1dd4f19cbf13e252c5fb` からWindows実閲覧ファイルまで `search-8c5b56e7d2e6d5d599c7` の `FINAL_RESULT_PUBLISHED` を確認した。今回は新規Q3計64件、実信号21点、実compile 14 wrapper。Q3と実信号はRomuTail、実compileはqg2で実行した。Romulaptopでは保存入力の照合が通り、PBSは今回の工程条件と照合できる所要時間枠がないため実計算0件だった。

LD7の通常Resumeも固定版で `REUSED_COMPLETED` を返し、追加のQ3起動・採用は各0。最終のC公開commitは `7d7e100d532e7615667b6031183bbc37f01f91d5`。Windowsの実閲覧ファイルで `search-4bd3a237a18ed0429847` の `FINAL_RESULT_PUBLISHED` を確認した。各継続の間は、共通queueの稼働ownerが終了してから次のrunが同じ入口を取得した。最後に中央索引の待機・割当・実行・転送待ちは各0で、今回の探索は停止した。

通常のStatus、Stop、Resumeはrun IDを指定する。既存runは登録済みの固定版と入力を参照する。操作と保存先は [HANDOFF](AUTHORIZED_EXECUTION_HANDOFF.md)。
