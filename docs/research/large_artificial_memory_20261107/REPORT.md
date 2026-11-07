# 長round向け人工flat回路のメモリ・時間検証（2026-11-07）

**人工回路の技術検証PASS。実H4 m17の実行可能性は未確定。公開照合後mandatory STOP。**

約4千、6.5万、26万instructionsで生成方式のgate列・位相が一致し、旧/新canonical fingerprintも一致した。
最大条件のtracemallocなし測定ではX/Y whole compileが各約1.1秒で完了した。
旧hashの観測peak RSSは3.30 GiB、streamed hashは0.42 GiBだった。これは人工回路1条件の観測である。
実H4追加build/compile、価格MC、強PF価格、総RPE費用算出、科学条件変更は実施していない。

## 成果物とSOURCE

- [集約JSON](../../../artifacts/large_artificial_memory/2026-11-07/summary.json)・[全66 stageの正規化CSV](../../../artifacts/large_artificial_memory/2026-11-07/summary.csv)。
- 主測定：[raw JSON](../../../artifacts/large_artificial_memory/2026-11-07/run01/result.json)・[raw CSV](../../../artifacts/large_artificial_memory/2026-11-07/run01/stages.csv)。
- 補足測定：[raw JSON](../../../artifacts/large_artificial_memory/2026-11-07/run02/result.json)・[raw CSV](../../../artifacts/large_artificial_memory/2026-11-07/run02/stages.csv)。
- [測定script](../../../scripts/benchmark_large_artificial_memory.py)・[テスト](../../../tests/test_large_artificial_memory.py)・[保存テスト結果](../../../artifacts/large_artificial_memory/2026-11-07/tests.json)・[PLAN](PLAN.md)。
- 専用branch：`codex/track-a-b2-large-artificial-memory-20261107`。基点`96efc6123b8bea3ff66d5dea729e0dcc3b9908b4`。
- 主測定SOURCE：`67c0eb21510bea767745b0c4b974ac7ec77d819d`。観測境界修正・補足SOURCE：`a02f528f808df2391fd6aa2f95d3eb4c055ee318`。
  各rawに実行SOURCEと6ファイルのSHA256、集約JSONにrawのSHA256を記録した。
  旧SOURCE・結果・grant・marker・STOPは保全し、既存ライブラリのコードは変更していない。

## 条件と測定方法

既存`synthetic_circuit('controlled', 1)`の先頭4 controlled primitiveを3 qubit unitにし、既存compilerで147 basis instructionsへ一度変換した。
9 qubit（system8＋ancilla8）の巡回pairにcomposeし、opaque gateを残さないflat回路を生成した。
unitのcompiler global phaseを保持する。人工block数28/446/1784から実instruction数4,116/65,562/262,248となる。
**このblockはB2 outer stepでもRTE eventでもなく、Q=131072に対応しない。H4入力を使っていない。**

全中間tailをlist保持する`materialized`と、同じiteratorを1個ずつcomposeする`stream`を比較した。
既存X/Y no-prep wrapper、canonical/streamed fingerprint、metric helperを再利用した。
compiler probeは既存moduleのQiskit transpile engineを同じ引数で呼び、hash処理を時間から分離する。架空のfingerprint/cost recordは作らない。
Qiskit **1.3.0、basis rz/sx/x/cx、optimization 1、seed 7**、backend/target/couplingなし、logical all-to-allを維持した。
X/Y wrapperのcopyはwrapper生成時間に含む。native copy、入力DAG、compiled DAGも別stageで測った。

条件ごとにspawn workerを使い、parentがそのworkerだけを`/proc`から公称5ms間隔で観測した。
begin/end handshakeで返却objectを終了観測まで保持する。peak RSSはsample値と、そのstageで新たに増加したkernel HWMの大きい方。
既存HWMより低い短時間peakなどは未観測の可能性があり、観測peakは厳密な上限ではない。
Python値はstage開始後のtracemalloc割当peakであり、プロセス内のPython総使用量ではない。
RSSにはRust/C、import、allocator残留、計測器自身の負担を含む。RSSとPython peakの差をRust/Cの厳密な量とはしない。

主測定run01は全条件でtracemallocを有効にした。**run01のraw `maximum_sample_gap_seconds`はloop開始間隔で、終了handshake境界を含まない不備があった。**
正常な主測定を完了後に観測境界を修正し、同じ最大stream条件をrun02でtracemallocなし測定した。主rawは変更していない。
正規化CSVはrun01の旧値を`legacy_loop_start_gap_ms`へ移し、`maximum_sample_gap_ms`を空欄にした。
run02はbegin/endの外部RSS読取・実時刻を含み、全15 stageで境界の時間整合を確認。実際の最大sample gapは8.717msだった。
run02のPython peakは未測定/null。tracemallocは時間とRSSを変えるため、主測定の値を通常実行の必要RAMとして使わない。

## 段階的な生成比較（run01、tracemallocあり）

RSSはプロセス全体。増分は「観測peak − stage直前RSS」で、解放量・専有量ではない。MiB=2^20 bytes、GiB=2^30 bytes。

| instructions | 方式 | 秒 | peak RSS MiB | RSS増分 MiB | Python peak MiB |
|---:|---|---:|---:|---:|---:|
| 4,116 | materialized | 0.018 | 281.69 | 0.00 | 0.201 |
| 4,116 | stream | 0.018 | 281.23 | 0.00 | 0.143 |
| 65,562 | materialized | 0.210 | 287.99 | 6.50 | 2.536 |
| 65,562 | stream | 0.211 | 283.98 | 2.50 | 0.581 |
| 262,248 | materialized | 1.346 | 335.46 | 54.22 | 9.804 |
| 262,248 | stream | 1.023 | 293.04 | 12.50 | 1.240 |

最大条件ではstream化で中間tailの保持が減った。最終flat circuit自体は保持するため、全体が定数メモリになるわけではない。
最小条件のRSS増分0は追加割当なしを意味しない。allocator再利用とRSS観測の分解能がある。

## 最大262,248 instructionsのstage別観測

run02はtracemallocなしの時間/RSS。右列は同一入力のrun01で測ったPython peakであり、同時測定値ではない。
全stageには、それ以前から生存するobjectとallocatorの履歴が含まれる。

| stage | run02 秒 | peak RSS MiB | RSS増分 MiB | run01 Python peak MiB |
|---|---:|---:|---:|---:|
| flat生成 stream | 0.265 | 291.88 | 10.00 | 1.240 |
| gate列digest | 1.740 | 291.88 | 0.00 | 0.002 |
| native copy | 0.016 | 309.38 | 17.50 | 0.003 |
| X wrapper | 0.368 | 327.39 | 28.00 | 0.068 |
| X input DAG | 0.051 | 384.89 | 57.50 | 0.016 |
| X whole compile | 1.088 | 451.32 | 96.50 | 64.898 |
| X compiled metrics | 3.233 | 387.32 | 0.00 | 0.004 |
| X compiled DAG | 0.054 | 429.32 | 42.00 | 0.004 |
| Y wrapper | 0.391 | 416.82 | 29.50 | 0.007 |
| Y input DAG | 0.042 | 438.82 | 22.00 | 0.004 |
| Y whole compile | 1.049 | 502.95 | 106.00 | 64.159 |
| Y compiled metrics | 3.319 | 437.32 | 0.00 | 0.002 |
| Y compiled DAG | 0.046 | 478.82 | 41.50 | 0.004 |
| streamed canonical hash | 50.015 | 432.48 | 5.22 | 11.534 |
| legacy canonical hash | 66.144 | 3,378.48 | 2,946.00 | 2,815.935 |

copy・wrapper・DAGではPython peakが小さくてもプロセスRSSは増えた。Pythonだけの監視ではnative割当を捉えられない。
入力DAGを保持したままwhole compileする測定ではない。診断用DAGは解放し、whole compile内のDAG構築はそのstageに含めた。
両者を合計してproduction compile時間にしない。X/Y両方のwrapperとcompiled circuitを保持したpipeline終端RSSは437.26 MiBだった。
hash前には両wrapperとcompiled circuitを解放しているが、allocatorの残留はある。

最大条件のrun01 hashはstreamed **259.188秒 / RSS417.13 MiB / Python11.53 MiB**、legacy **311.495秒 / RSS5,843.93 MiB / Python2,815.94 MiB**。
run02でも旧hashの全payload構築が最大の一時メモリ消費だった。streamed hashの省メモリ効果を確認した一方、50秒の走査時間は残る。
1条件1観測であり、速度倍率・tracer overhead・厳密な漸近量の推定は行わない。旧hashは後に実行するためallocator履歴も異なる。

## 整合性・資源・証拠の範囲

3規模すべてで生成方式間のinstructions、qubits、global phase、gate counts、gate列digestが一致。
各stream workerで旧/新canonical hashが一致し、補足測定も最大主測定と入力digest・位相・canonical hash・X/Yの6 metrics・DAG operation数が一致した。
最大X/Y出力は各262,254 instructions、CX=92,768、RZ=128,451。これは人工回路の診断値でありH4価格ではない。
benchmarkは7 worker・66 stage・whole compile8回・setup unit compile2回。機能テスト内の短いcompileはこの回数に含めない。
4種類の機能テストが初期/修正版の両cohortでPASS。16 MiBのmmapでPython traceに現れないRSS増分を確認し、短いstageの終了境界も検証した。
これは主agentによる自己点検であり、独立科学レビューではない。

CPU affinityは64。条件は時間比較を自worker同士の競合から分離するため順次実行し、CPU/thread/RAMの上限は追加しなかった。
次サイズの需要予測とhost MemAvailable・読取可能な有限cgroup headroomを使い、実行中も実際の空き容量を観測した。
run01最低空き454.78 GiB、run02最低434.38 GiB。読取範囲に有限cgroup memory上限はなかった。資源不足STOPなし。
共有設定・他ユーザーのprocess・他worktreeは変更していない。現時点の空き容量が将来も得られる保証はない。

## m17の未確定事項と次の改善

**Q=131072の実H4 m17について必要メモリ・処理時間・実行可能性は未確定。**
今回確認したのは、最大約26万の人工basis instructionsで既存wrapper/hash/compilerが動作し、Python以外のメモリ増加も観測できること。
実H4のouter step当たりのgate数、gate mix、非可換構造、native展開、最適化挙動、event分布、同時保持量を測っていないため、Q比例外挿はしない。

次の技術改善候補は次の順序とする。今回これらのpipeline変更や実H4検証は実行していない。

1. Xをcompileしてmetricsを抽出後にwrapper/compiled circuitを解放し、Yへ進む。既存metric-only cacheを再利用し、miss返却値のcircuitを呼出側に残さない。
2. 生成streamingと同じcanonical identityを保つstreamed fingerprintを活用する。実回路への適用範囲は別途指示に従い、既定cache policyやmutation検出を弱めない。
3. native compiler workspaceとmetric走査を個別にprofileし、実測メモリと実際の空き容量に応じて待機event/workerの同時保持を調整する。
   今回はmetric走査も各約3.3秒を要した。科学比較を維持する検証なしに、分割compileをwhole compile価格の代替へ採用しない。

**mandatory STOP：本成果物のcommit/push・remote SHA/tree/公開内容照合をもって停止する。実H4追加検証や価格MCへ自動進行しない。**
次の担当はユーザー：上記改善の実施範囲を判断する。Codex：次の具体的指示があれば、その範囲内の改善を実装する。
