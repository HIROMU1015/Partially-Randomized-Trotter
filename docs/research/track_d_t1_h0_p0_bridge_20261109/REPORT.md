# Track D T1〜H0：P0境界bridgeと保存H4の限定照合

2026-11-09 JST。**CXの両wire隣接相殺を扱う最小bridgeを実装し、保存54人工case・指定3 P0・保存H4 2pathの照合を完了した。全H4のP0価格をpre-P0情報だけから予測する段階には未到達である。** 対応範囲は局所CX規則と確定native列の計数で、1Q再合成・Gaussian・CSWAP等の全体bridgeは残る。

[PLAN](PLAN.md)／[P0接口分析](P0_INTERFACE_ANALYSIS.md)／[bridge仕様](BRIDGE_IMPLEMENTATION.md)／[保存H4](SAVED_H4_COMPARISON.md)／[次段階案](NEXT_EXECUTION_PROPOSAL.md)／[STOP](STOP.md)。

## SOURCEと実施範囲

実行SOURCEは **2712a73f6e609e0f883563c400b4d1facecd1aa8**。基点T0はc583abba、同じリソース評価branchを継続した。旧SOURCE・科学builder・T0・旧STOPを変更せず、新module [track_d_p0_cx_bridge.py](../../../src/trottertracks/track_d_p0_cx_bridge.py)と[一括runner](../../../scripts/run_track_d_t1_h0.py)を追加した。正式共通指示2026-11-09と適用AGENTS／PROJECT_MAPを確認し、ユーザーの一括許可の範囲で実施した。承認証明／新grantは作らない。

通常Git pushはcredential cacheの認証切れで失敗したため、接続済みGitHubツールで[公開SOURCE bae6a11b9d2b71bc3734acfe17db9e6a5226a53c](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/bae6a11b9d2b71bc3734acfe17db9e6a5226a53c)を作成した。実行2712a73fと公開bae6a11bのGit treeはともに **3a8d8a2bd017049a7fa82ef05b263a4a869a4e94** で全ファイルbytesが同じ。実行記録のSOURCE SHAは書き換えず、元2712a73fもローカルtag `track-d-t1-h0-execution-20261109`で保全した。結果はこの公開SOURCEの子に追加する。

| Phase | 実施結果 | 証拠 |
|---|---|---|
| A・保存人工trace | 54caseを全件検査。raw16＋compiled54＝70 native列で保存7指標と一致。raw38は未展開gate等で対象外 | [全case比較](../../../artifacts/track_d_t1_h0_p0_bridge/2026-11-09/run01/artificial_trace_comparison.json) |
| B・P0分類 | CX隣接／control・target側RZ／parity／1Q／Gaussian・actual inverse／CSWAPを区別 | [分析](P0_INTERFACE_ANALYSIS.md) |
| C・最小bridge | 両wire frontierで隣接する同方向CXだけ相殺。可換RZ移動・1Q再合成は実装しない | [実装](BRIDGE_IMPLEMENTATION.md) |
| 技術回帰 | 新8＋既存T0 22＋既存設計mock7＝37件PASS | [新test](../../../artifacts/track_d_t1_h0_p0_bridge/2026-11-09/technical/bridge_tests.txt)／[T0](../../../artifacts/track_d_t1_h0_p0_bridge/2026-11-09/technical/T0_regression.txt)／[旧mock](../../../artifacts/track_d_t1_h0_p0_bridge/2026-11-09/technical/design_regression.txt) |
| D・固定追加人工P0 | 許可されたfull＋二chunkの3回だけ。全3 correctness PASS、max-entry残差0、phase fitなし | [3 P0結果](../../../artifacts/track_d_t1_h0_p0_bridge/2026-11-09/run01/artificial_new_P0.json) |
| E・保存H4 | q4_sample00／q64_sample00、Strong／Reference X/Yの8 compiled列すべてで保存絶対価格と一致、HOLDなし | [H4比較](../../../artifacts/track_d_t1_h0_p0_bridge/2026-11-09/run01/saved_H4_comparison.json) |

人工SOURCEはC7-2固定709d6ed9（実行34929f8a）、P0はQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approx1／initially_zero=False／topology-free／既存MPFR160宣言位相保持。新人工3回は同じ既存compiler経路を使い、小8×8 Operatorの位相付きcomponent abs1e−10・rtol0を維持した。新H4 compile／state-actionは0。

## 確認できたP0対応と限界

適格raw16列では、bridge後のCX countとCX depthは各15列、全7資源は10列で保存P0と一致した。残る6列の不一致を保存し、ruleを結果に合わせて拡張して消していない。literal RZ(0)の削除や1Q runの合成が欠けており、全P0と等しい一般規則とは主張しない。54 compiled列ではbridgeによる追加CX削減は0だった。

固定人工 `CX01,CX12 | CX12,CX01` は、full P0ではCX/size/depthすべて0、独立chunkは各CX2／depth2で、価格和ではCX4／depth4になる。同じbridge状態をchunk間で維持するとfullと同じ0へ相殺できた。**T0 transferを独立に合成するだけでは、P0のこの跨界面相殺を復元できない。**

fixed native word／bridgeが定義した変換後wordの計数は厳密である。純CX列での構造的対応候補と指定3 P0の一致は得たが、任意native列・full PR wrapperの全P0閉包を証明したものではない。自動出力は `P0_UNQUALIFIED` を維持する。

## H4への接続

分子条件はLinear H4／隣接1.00 Å／STO-3G／Nα=Nβ=2、元H、number-CDF8／元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1。元H厳密groundを仮定し準備費を除外。q4/cut2とq64/cut32、C1追加head-tail融合OFF、強いsymmetric-directional・whole-path Reference一gadgetと既存共有・融合・scalar/χ条件を維持した。

8保存compiled列の再計数では全絶対資源が一致した。これはcompiled答えを読んだ確認で、未compile構造からの予測証拠ではない。rawでは全未対応命令をopaque barrierとし、適格native連続segmentの内側だけにbridgeを適用した。q4では両方式各14CX、q64では両方式各134CXを局所的に取り除けるが、**これを保存compiled価格から追加で差し引いてはいけない**。Gaussian/RZZ/CRZ/CSWAP/readout等の未対応部分とP0界面を含む全価格はnullのままである。

## 長qの実現可能性と次の判断

T0は確定native streamならO(L)時間／O(W)memory。今回bridgeはfrontierへのpush/pop自体O(L)、最終計数O(L)だが、履歴は最悪O(L+W)memoryを使う。32命令のprefixを保持した後、逆順suffixで全消去する人工回帰を保存し、一定長frontierが常に十分とはしなかった。q1024〜4096の新実測やspeedup評価は行わない。

既存emitterから意味を保ったnative streamを得る点、1Q run・zero/special angle・Gaussian/parity・CSWAP／χを跨ぐP0規則が未解決である。主A・比較F/Sは維持し、長q bias／law／E[max]を0にしない。208−224qの全path定理、母平均、N、全13-round会計も更新しない。

推奨は、既存の不一致を使う1Q run／zeroの小さな境界検討とnative emitter出口の整備である。別H4／追加q／新sampling／大回路compileを先行させる根拠は今回得ていない。O1/O2/O3との共同最適化にも、対応済資源と未資格な全P0価格を区別して渡す必要がある。[次段階案](NEXT_EXECUTION_PROPOSAL.md)。

## 保全と終了

[SOURCE](../../../artifacts/track_d_t1_h0_p0_bridge/2026-11-09/run01/SOURCE.json)に8実装pinと9固定input pinを保存し、[前後照合](../../../artifacts/track_d_t1_h0_p0_bridge/2026-11-09/run01/preservation.json)はPASS。q4 archive・q64選択8 QPYも保存hashと再照合した。過去の結果・SOURCE・grant・ledger・marker・STOPは不変。人工6 QPYは新local保存、H4/event/QPY/NPZ原本と内部運用情報は追加公開しない。

[結果](../../../artifacts/track_d_t1_h0_p0_bridge/2026-11-09/run01/result.json)／[実行STOP](../../../artifacts/track_d_t1_h0_p0_bridge/2026-11-09/run01/STOP.json)。新P0は固定人工3回、H4再compile／H4 simulation／sampling／fit／N／全RPE再計算／詳細科学レビューは0。公開SHA・tree・公開bytes照合後mandatory STOPし、次の判断をユーザー/GPTへ返す。
