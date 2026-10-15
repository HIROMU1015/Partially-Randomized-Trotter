# H6 native B0+I 全12構成：実行準備・技術監査（2026-10-15）

実装・人工検証・独立技術監査を完了した。**実科学計算は未認可・未起動**。固定SOURCEは `731826b3bb82dba18b31989a39607fe426d34960`。新branch `track-a-h6-native-b0i-full-family-readiness-20261015` は旧実行worktreeのcommit `d1fa0974064ae4feeb1c8168a88b43f89fc58340` から独立作成した。旧science/provider・結果・契約・grant・ledger・worktreeは変更していない。

[監査bundle入口](../../artifacts/resource_applicability/track_a_h6_native_b0i_full_family_readiness_v1/2026-10-15/README.md)／[未承認契約案](../../artifacts/resource_applicability/track_a_h6_native_b0i_full_family_readiness_v1/2026-10-15/execution_contract_draft_v1.json)。この資料は全面実測の結果ではなく、将来の一度の取得に必要な実装と承認対象である。

## 固定参照と受領

- [pilot結果 ae3b992f](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae3b992f22aead88515ef594cde7e5f867aefa87/docs/research/track_a_h6_native_b0i_pilot6_results_20261015.md)：既存3構成・6軸。実行SOURCE `2e733d8a683d415792ce92a56c7d149b78f42716`、grant/seal `d8cdc1a7141d65658308de0fb66a4ed67233023c`。
- 科学provider `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`、独立確認結果 `321cd8c3066194fe98741cbebbcdba1515d1deaa`。
- [H4/H6 D0 8dbdb682](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dbdb68293c988a999645e21a82b53966f9ef6cd/docs/research/track_a_d0_saved_value_reanalysis_result_20261014.md)。今回はこの保存値解析を再実行していない。
- [GPT統合レビュー 3d910ee1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md)：全12構成の追加対照を科学的に採択済み。科学レビューは再実施していない。実行認可は別である。

上記7つの固定GitHubファイルはローカルGit objectとremote bytes/SHAを照合した。新契約には旧科学5入力＋pilot JSON12件＋公開D0 JSON1件の**18 byte receipts**を登録した。科学5入力は既存受領receiptを保持し、NPZの再hash・科学array読込を今回の準備で繰り返していない。追加12 JSONは元ファイルのbytes/SHA、D0は固定Git blobとregistry metadataを確認した。保存された旧scalar/precisionを検査したが、新しいH6 scalar・位相・bias・allowance・shotの計算は行っていない。起動時には全18入力を改めて照合する。

## 登録12構成と再利用6軸

科学条件はlinear H6、1.00 Å、STO-3G、保存DF rank19/state、T=0.8、S2、symmetric_directional、既存native Gaussian lowering／compiler optionsを維持する。

| prefix | q1 | q2 | q4 | q8 |
|---|---|---|---|---|
| 5 | 旧pilot再利用 | 新規候補 | 新規候補 | 新規候補 |
| 10 | 新規候補 | 旧pilot再利用 | 新規候補 | 新規候補 |
| 15 | 新規候補 | 新規候補 | 新規候補 | 旧pilot再利用 |

旧6軸の分類は **REUSE_WITH_DECLARED_LIMITS**。H/DF/state/input、prefix/q/T/PF、full-Fock規約、compiler hash/options、旧numeric fingerprint、軸別6指標、成功terminal/native wait/launcher wait・6worker cleanup、旧grant/profileを照合した。元SOURCEの実測として引き継ぎ、新SOURCEでのcompiled bytes完全同一性は主張しない。matched B0のprofile/numeric同一性は未取得なのでdeltaはnull。再利用条件が崩れればSTOPし、旧3の自動再compileや18軸を超える補完は行わない。

旧provider/profileで確認済みのprefix別preparation metadataからqだけを静的に展開した12 expected blueprintを固定した。`_prepare_discard(ham,prefix)` にq引数がないことが根拠で、新9構成の実preparation診断ではない。旧legacy/currentのpreparation_hash差も保全し、実行時の実metadataが期待値と異なればfield差を保存してSTOPする。旧環境とのbit同一性も未証明のままである。

## 将来の処理順序と取得上限

Aで旧3の保存precisionを再利用し、新9だけtail scalar・保存B0 signalの解析的位相回転を行う。full-H referenceは回転しない。signed λ、Hermitian G、DF生成順、full-Fock tail identity、ancilla relative phaseを保持し、retained/nuclear/one-body scalarを二重加算しない。元のempirical uの政策、`EMPIRICAL/certified=false`、正規化、軸別整数shot/headroom規則も変更しない。

ε=0.05/0.01/0.005/0.001/0.0005 × factor1/10の10条件を全候補について揃える。全10条件が確定INELIGIBLEならnative build/probe/compileを省略し、費用はUNMEASUREDのまま。未解決は不適格扱いせずSTOPする。適格な候補の費用欠測は比較完成にしない。

全screen終了後に取得対象を固定する。Bで対象全構成のcontrol/X-Y/保存独立forward reference/numeric fingerprintを検証し、**最後の構成までPASSして初めてCのcompileを開始**する。Cは検査済みwrapperをfork継承し、最大6並列・各thread1で取得する。新loader/buildをworker内で追加せず、QPY転送も不要。開始前に件数を予約し、失敗予約を返却しない。

| 新規操作 | 最大 | 適格な新構成がn件の場合 |
|---|---:|---:|
| snapshot load | 1 | 1 |
| tail scalar／analytic overlay | 各9 | 各9 |
| native evolution build | 9 | n |
| measured wrapper | 18 | 2n |
| unmeasured probe copy | 18 | 2n |
| control probe | 36 | 4n |
| compiled axes | **18** | **2n** |
| 新SCF/DF/state/trajectory/signal exponential action | **0** | **0** |

nは0〜9。旧pilotの操作は歴史的消費に保持し、新規取得数に再計上しない。失敗・未確定消費はnull/UNKNOWNを0にせず、carryをreset/refundしない。

Dの主比較はP=0のshot-weighted RZ。固定D0のH6_CONFIRMATION_UNION8（28 baseline構成×10条件＝280行）とB0+I12構成を結合する。PRは**確認union8/n32だけ**で、探索n8を混合しない。精度・factorごとに適格で費用が揃ったB0/B0+I/full S2/full S4の最小を分母とする。他5指標と仮想共通状態準備費用Pのlower envelopeは補助比較。取得済み集合内の点最小であり、母平均・全候補最適性や新科学判断を主張しない。

## 未承認の資源・時間・容量案

| 項目 | 案／条件 |
|---|---|
| driver／compile workers／metadata observer | 1／最大6／既存launcher1、常駐最大8（同期Gitを別記） |
| CPU | driver1＋worker最大6の異なる空きphysical core。H8・他jobと重複回避。既存metadata launcher/observerはdriver coreを共有。番号未固定、allowed_cpus=[] |
| numerical AS／RSS | driver・各workerとも32GiB／9GiB |
| launcher AS／RSS | 512MiB／256MiB |
| memory admission | 7×32GiB＋0.5GiB＋16GiB＝**240.5GiB** |
| phase／total wall | **null（上限なしの新契約案）** |
| CPU上限 | driver＋全compile子の累積86,400秒、metadata1,800秒、いずれも別途承認対象 |
| monitor | 1秒独立観測、5秒staleness fail-closed、host/cgroup pressure・OOMは旧厳格政策維持 |
| final／pending／temp | 各512MiBの容量予約。temp全体の独立hard使用量capは未実装 |
| combined stdout/observer log／journal／terminal | 512MiB／128MiB／1MiB |
| block・inode admission | 4GiB、fresh40,000 inodes。JSON output cap20,000 inodes、terminal4別予約 |
| quota | 起動直前の実filesystem・effective user/group/project quotaを確認。quota有効で有効上限を証明できなければSTOP |

旧pilot契約SHA `1cf99b426aab08f738e6d71e64e89e001941180f2648e37e08307c54997f9db4` は**wall14,400秒・CPU14,400秒**で実行された事実を残す。今回の指示§7が述べる今後のwall上限なし方針に合わせ、旧4時間を新契約へ流用せずnullを明示した。原本の共通作業指示だけにwall-null条項があるとは扱っていない。新wall政策と別管理CPU予算を含めて実行承認が必要であり、旧grantを変更・転用しない。

pilot37分42秒・worker最大RSS約3.90GiBは過去実測で、新9構成の完走時間・RAMの保証ではない。全対象をbuildしてからcompileするため、最大9構成のwrapperを保持するdriverのRAMは未実測であり、上限に達せばSTOPする。今回の観測を将来のfresh launch証拠に使わない。

4GiB予約はfinal/pending/temp各512MiB＋log512MiB＋journal128MiB＋terminal1MiB＋4KiB×40,000 block/inode paddingと余裕を含む。JSON/journal/combined logの直接計数と、FS空き容量/inode/quota guardは実装されている。temp/control全treeの独立hard capや正確な使用量課金は実装していないため、容量予約と区別する。容量見積りを上回る使用もfresh FS/headroom guardでfail-closedに止める。

## 人工検証・独立監査

単一process/thread1、AS2GiB/RSS512MiB観測上限、CPU/wall180秒、log16MiB、home配下の専用temporaryを事前計画した。小さい人工2-mode行列、string回路、mock provider/identity/workerだけを用いた。science imports・実NPZ・実fork/exec/subprocess・fixture外writeを拒否するharnessで**44件PASS**、最終実行wall約4.46秒、peak RSS32.5MiB。実worker/compileの動作・速度・本番profileとの同一性の検証ではない。

検査にはregistry/reuse/max18、全不適格skip、部分取得、未解決STOP、source/input/compiler/profile・full-Fock/T/preparation改変、最後のscreen/correctness失敗時compile0、失敗queue停止/first STOP、cleanup未確認、one-shot、旧n8拒否、deterministic最小、matched delta nullを含む。

別担当のread-only Codex技術監査は **PASS_WITH_LIMITS、open P1/P2=0**。compile後に子CPUの累積検査が途切れるP2を発見し、wait4値をcleanup後もdriver Monitorへ保持し、contract/driverに結合したcarryをobserverへ渡す修正を行った。Phase D/最終tickの累積CPUとfirst STOP回帰を追加し、該当差分を独立再確認した。監査担当による別のtests実行や科学レビューは行っていない。

再現用人工検査は、SOURCE checkoutで新しいhome内directoryを指定する：

```bash
python3 -P -B tests/run_h6_b0i_family_artificial_tests_v1.py \
  /home/AbeHiromu/projects/h6-handoff-evidence/20261015/full-family-readiness/another-exclusive-artificial-test
```

これは人工検査用commandで、本番launch commandではない。

## 次の一括承認・fresh gate

1. このSOURCEと新REVIEW_BUNDLEを対象に、同じ固定profile/compilerを既存venv無変更で使い、旧6軸のREUSE_WITH_DECLARED_LIMITSを受け入れる。
2. 新9構成のscreen9・overlay9、適格なn構成のみbuild n/probe4n/compile2n（最大18）と、保存値による6metric/P比較を一度実行する。
3. wall-null、累積CPU、AS/RSS、worker数、observer、memory/FS/inode/quota案を承認する。CPU番号はfresh空きcoreの提案後に固定し、その許可と分ける。
4. 起動直前にSOURCE全closure・provider・materialized profile・18入力bytes/SHA・旧carryを照合し、未使用exclusive output/control・永続one-shot lockとCPU/memory/pressure/OOM/block/inode/quotaをfresh検査して最終binding/sealを作成する。
5. 完了またはfail-closed STOP後にmandatory STOP。自動retry、旧case再取得、matched B0、q16/H8/H5/H7/T/PF拡張、次stageを行わない。

現在は`approved=false`、`runtime_authorization=false`、`grant_created=false`、`execution_plan_sealed=false`、`allowed_cpus=[]`、`absolute_launch_command=null`。軽量SOURCE/tests/資料だけを公開対象とし、NPZ/raw runtime/checkpoint/credential/内部接続情報は含めない。既存rootのdirty/untrackedと旧2worktreeのHEAD/statusは保全監査で一致している。**新科学計算・実native build/probe/compile・新grant・共有venv/他job変更・原稿改訂は全て0。準備完了によりmandatory STOP。**
