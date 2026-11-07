# Track D C8-1：q64独立価格検証の固定PLAN

2026-11-08 JST。ユーザー指示による一campaign。基点B/C最終8828b298。**q64/cut32、fresh8 path、512 occurrence、16384 event、raw8、最大32 P0/postだけ**。最初の固定sample00のraw・4 wrapper/post・pairedχが全PASSした後だけ残り7 pathへ進む。preflightの4 compileは総32に含む。失敗pathの除外・追加sampling・retry/resume・閾値緩和はしない。

## 固定科学条件と対照

Linear H4/隣接1.00 Å/STO-3G/Nα=Nβ=2、元H/number-CDF8/元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1。元H厳密groundを仮定・準備費除外。元C ba0e2bf2のSOURCE/input/保存lawを既存C6a/C5/C2 preflightで照合する。packet1d6087f6、endpointa8813aca、lawc347d7f2の完全hashは旧契約とSOURCEに固定。新ground/DFfit不要。

strongはuncontrolled Gaussian、directional/head relative fusion/tail basis sharing、63 step境界融合、scalar/identity集約。Referenceはwhole path一gadget、B前32step/A後32step、actual A.inverse、analytic χ、二CSWAP、内部62融合とcut一境界損失。短いgadgetの反復・弱い対照へ変更しない。

P0はQiskit1.3/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=false/topology-free/宣言位相保持。component abs1e−10、observable Frobenius/signal2e−10、rtol0、phase fitなし。全256system入力、CAR/AB/actual adjoint/vacuumχ/scalar/complex coherence/X/Y/Reference leakage/raw-P0を既存checkerで照合する。

## 再利用と高速化

C7-3のprivate domain bindingとQ32Requestからq64 requestだけを最小拡張し、元C7-1/C3/C5/StrongBuilder/Reference/CAR cache/統計を使う。正常q32/q16のguardを変更しない。新sampler/Gaussian/compiler/simulatorは実装しない。

rawとpostの両方でC7-3 exact gate cache、公式Aer double unitary（9qubit以下のみ）、既存BatchedStateActions（Reference17、全256列）をworker内選択する。17qubit全Operatorは作らない。これは精度改善を認定する変更ではなく、同じ閾値を保つ既存高速経路の選択である。C7-4 native版は未採択。

sample00 gate後はraw/compile/postを独立task化し、既存CPU/RAM allocatorで十分に並列化する。実affinity/quota/MemAvailableを観測。初期memory admissionは全256列17qubit state配列の16倍という暫定作業estimateで、実sample00 raw peakの2倍を反映する。これは資源使用上限・科学的boundではない。CPU/RAM/workerの低い人工上限を設けず、共有OS/venv/scheduler/他job/cacheは変更しない。小行列BLASとP0内nested parallelだけを抑え、割当native threadsを使う。

## Seeds・event固定

master=2026110801、domain=`track_d_c8_1_q64_independent_price_v1:fresh_independent_holdout:q64`、master tuple=(master,domain,8828b298...)。既存trajectory_seedを使い、endpoint、固定key `C8_1_PAIRED_PRICE_ONLY`、sample0..7/occurrence0..63を含む。512 seedとC6a/C6b/C7-1/C7-3・Stage A seedの非重複をsampling前に確認する。完全key/seedと過去plan hashはtechnical/seed_plan.jsonを正本とする。seed非重複はIID証明ではない。

canonical configure/sample_occurrenceを512回だけ実行し、全8 event JSONをlocalへ保存/hash固定してからCAR/回路/価格を扱う。過去path連結・prefix共有・再抽選なし。Reference/strong/XYは同じwhole pathを共有する。

## モデル・評価規則

主A=C7-4 affine、F=元C6a frozen affine、S=C7-4 design[1,q,sqrt(q)]。固定公開64072ba8/SOURCEdd31e1e2から係数/係数共分散hashと保存q64予測をsampling前に固定する。q64のモデル校正共分散は保存係数の線形伝播だけ。technical/frozen_models.jsonに予測/校正共分散と固定参照hashを保全する。新fit/bootstrapは0。basis/lane版は固定参照を保持するが、今回の母平均価格入力にはせず、追加のfeatureモデル診断も必須にしない。

最優先CX count・total depth。全RZ/CX count/depth・total depth・size・peak幅を取得。差はReference−strong。各pathの整数残差ΔCX+14128を保存し、0でも全path定理へ昇格しない。

全8 pathが適格に揃った場合だけ既存n8/ddof1統計で絶対平均/SD、paired平均/SD、min/max、方式/資源/XY共分散を作る。XYをn16としない。A/F/Sの絶対24量・paired12量の予測残差、保存校正変動と標本変動を別記する。今回5%幅/αfamilyや確認的PASSは採用せず、連続残差とfinite観測だけを報告する。モデル不一致/整数反例は有効結果で、correctness failureと区別する。

rawのbasis/sign/support/zero angle/head-tail/cut/lane/CSWAPとcompiled DAG featureは既存extractorから保存する。compiled DAGを未来予測の事前featureとしない。sample00の最大数値残差はC7-4保存q32残差との条件差付き比較に留める。

## SOURCE・出力・STOP

人工・回帰テスト後、SOURCE/本PLAN/seed/model packetをGit固定して明示起動する。新namespace `artifacts/track_d_c8_1_q64/2026-11-08/run01`。進捗、preflight、path check、価格/統計/モデル診断/整数残差、SOURCE/input hash、payload hash、result/STOPを保全。event/QPY/NPZ/conditions/fusion/内部環境はlocalのみ。公開は必要なscalar/check/hash/戦略文書だけ。

SOURCE/law/seed/科学条件不一致、公平な対照不能、raw/P0/χ/adjoint/vacuum/leakage/固定精度不成立、保全/非干渉不成立なら依存処理を停止し、active taskは適切に完了/回収してpartialを保持する。phase fit/renormalization/列削減/seed変更/閾値緩和・科学retry禁止。q≥128、追加sampling/compile、fit、N、全13-round費用更新、actual保証、routing/T/noise、詳細レビューは0。

成功/不成立いずれでも、保存証拠による長q戦略整理は継続する。q1024/2048/4096の大きい費用寄与を考慮し、次qをq128へ自動固定せず、少数長q compile・構造モデル・中間推定の必要精度/負担/holdout/数値可能性を比較する。最終成果公開・remote SHA/tree/bytes照合後mandatory STOP、次の判断はユーザー/GPT。
