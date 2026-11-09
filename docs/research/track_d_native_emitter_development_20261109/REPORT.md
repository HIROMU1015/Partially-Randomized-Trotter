# Native emitter・1Q/Gaussian界面・P0構造bridge：技術結果

2026-11-09 JST。リソース評価系列 `codex/resource-eval-track-d-longq-structure-20261109`、基点 b49ee3fc153bed786635b352a6ffb89c944b5adb。[固定PLAN](PLAN.md)の技術開発を完了した。**既存emitterからnative streamを取得し、保存54人工caseすべてで部分bridgeの全7資源が保存P0価格と一致した。一般P0／H4／長qへの資格は取得していない。**

## 実行と証拠

実行SOURCEは **3faeaf87eb243f671bd802953f90f097541cccfb**。接続済みGitHub経由の公開SOURCEは **92ecfe3860df5691db88ca26db2e2febc0f16d7a**。両者のtreeは **0104daf4168e525dec25089710e6183734d9ae6b** で同一。APIでcommit metadataが変わるためSHAは異なるが、実行SOURCEの記録は変更しない。元commitはローカルtag `track-d-native-emitter-execution-20261109`で保全する。最終公開はこの公開SOURCEの子とし、ref・tree・各公開bytesを照合する。

| 確認 | 結果 | 証拠 |
|---|---|---|
| 既存head／relative Gaussian／step境界、tail chunk、Reference actual inverse／χ／X/Y | 新10人工tests PASS | [test](../../../tests/check_track_d_native_emitter_bridge.py)、[実行log](../../../artifacts/track_d_native_emitter_development/2026-11-09/technical/native_tests.txt) |
| 旧T0 22＋CX 8＋設計mock 7 | 37 PASS、新10と合わせ重複なし47件 | [T0](../../../artifacts/track_d_native_emitter_development/2026-11-09/technical/T0_regression.txt)、[CX](../../../artifacts/track_d_native_emitter_development/2026-11-09/technical/CX_regression.txt)、[mock](../../../artifacts/track_d_native_emitter_development/2026-11-09/technical/design_regression.txt) |
| C7-2保存人工raw QPY | 公開hashに一致する54/54原本をread-only使用、HOLD／未対応0 | [SOURCE](../../../artifacts/track_d_native_emitter_development/2026-11-09/run01/SOURCE.json)、[全54結果](../../../artifacts/track_d_native_emitter_development/2026-11-09/run01/artificial_results.json) |
| native loweringの位相付き作用 | 54/54 PASS、最大成分誤差5.43632399492795e−13 | 同上。component atol1e−10、rtol0、phase fitなし |
| 部分bridgeの位相付き作用 | 54/54 PASS、最大成分誤差5.390889565733785e−13 | 同上。閾値緩和なし |
| bridge後のRZ/CX count・RZ/CX/total depth・size・幅 | 各54/54、全7同時一致54/54 | 保存P0のCSVをoracleに使用。新P0は0 |

この54は固定した人工case集合であり、PR lawの標本やH4への性能保証ではない。最大5qubitの保存人工原本を使い、小規模Operator比較と2×2合成を行った。分子geometry／basis／DF rank／LD指定を持つ新科学条件を評価していない。旧H4条件（Linear H4／1 Å／STO-3G／Nα=Nβ=2、元H／number-CDF8／元順LD2／S2/s1/r32/K0／h=δ=.1、元H厳密ground仮定・状態準備費除外、C1追加融合OFF、P0 Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／topology-free／位相保持）は変更せず、その結果の新再解析も行わなかった。

## 到達点と未解決事項

1. PR emitter→native stream→T0／CX bridgeの入口は成立した。既存C3のheadとReference／readoutの実コードを再利用し、全wrapperのQuantumCircuitを作らない経路を人工入力で確認した。tailやGaussianの既存小QC、人工oracle用QC／小dense行列は残る。[接続詳細](NATIVE_EMITTER_INTEGRATION.md)。
2. 旧6不一致はすべて解消した。Qiskit公式ZSXXの2×2 run合成を使い、厳密に短くなる1Q置換と既存CX相殺を反復する。full transpile／DAG最適化passは呼ばない。[接口結果](P0_INTERFACE_RESULTS.md)。
3. Gaussian／parity／CSWAPも54case内では全指標一致した。ただしtranslation経路・pass順・任意角・長い連鎖の閉包は未証明。`NATIVE_EXACT`は確定したnative列の整数価格、`P0_UNQUALIFIED`は一般P0対応という区別を維持する。
4. T0逐次計数は全wordを保持しない。今回の1Q＋CX固定点処理はO(L)記憶を使い、最悪時間O(L²)。固定幅frontierへ無条件に切り詰めない。q1024〜4096の実測速度／RAM、H4からのstream価格は未取得。[計算量と限界](PERFORMANCE_AND_LIMITATIONS.md)。
5. 技術的には小さな範囲で構造評価を続ける価値がある。全P0の再実装は勧めず、次は保存H4の限定照合を一度行うか、部分構造モデル＋既存A/F/Sを使ってPR共同最適化へ重点を移すかをユーザー／GPTが判断する。[次段階案](NEXT_EXECUTION_PROPOSAL.md)。科学的GO／研究終了の決定は行っていない。

## 保全とSTOP

新P0・H4 build／compile／sampling／保存H4再解析・q≥128科学実行・fit・N算定・13-round再会計・詳細科学レビューはすべて0。q1024〜4096 benchmarkも0。A主モデル／F frozen／S探索モデル、既存208−224qの証拠階層は不変。局所CX削減を保存P0価格から追加控除していない。

旧SOURCE・科学builder・T0／CX実装・入力索引・原本・結果・grant／ledger／marker／STOPは変更せず、新namespaceへ技術結果だけを保存した。原本QPY／event／NPZや内部保存先は公開しない。[STOP](STOP.md)／[実行STOP.json](../../../artifacts/track_d_native_emitter_development/2026-11-09/run01/STOP.json)。公開照合後mandatory STOP。次の担当はユーザー／GPTであり、Codexは新しい明示範囲が決まるまで後続を開始しない。
