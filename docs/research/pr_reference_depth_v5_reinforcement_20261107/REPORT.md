# V5 depth保証：独立レビュー指摘2点の限定補強

2026-11-07 JST、SERVER 2。基準 `285b77023542c3b39caea767e2a937dffa046a3b`、専用branch `codex/pr-reference-depth-v5-reinforcement-20261107`。

**Stage A・BともPASS。既存の全18round期待depth上下界と、削減率の保証下限40.995952643300015%を維持した。** 結果を所与として採用せず、F†側の実source inverseのlocal DAGと、B′下界を支える非X依存鎖を確認した。Stage Cは成立条件と保存値の対応照合だけを実施し、18round総計を再計算していない。

| 項目 | 判定 |
|---|---|
| F†側のD span上界 | **PASS**。inverseの依存行列は異なるが、全20 recordのspanが既存上界以下 |
| B′のD/frame下界保持 | **PASS**。実際の下界対象19種類×4 contextはX-freeで、非X依存鎖を保持 |
| 18round期待depth上下界 | **既存維持**。新総計再計算0回 |
| 40.995952%保証下限 | **維持**。固定モデル下の条件付き境界比較 |
| 既存CX・RZ・shot保証 | 変更なし |
| mandatory STOP | [実施済み](../../../artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/STOP.json) |

これは著者による技術検証と構造論証であり、新たな独立科学レビュー・形式証明・第三者再現・CIではない。指定名 `PR_reference_V5_depth_independent_review_2026-11-07.md` は基準treeと既知ローカルpathで未提供・未確認であり、本文読了とは扱わない。今回提供されたCodex指示に列挙された2論点を対象とした。

## 固定条件・再利用

Linear H4 1Å/STO-3G/DF12、元H厳密ground供給・準備費除外、Partial S2 LD3 h=.02 finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、保存18round q/r、energy精度1.5936e-4 Ha、失敗率.05、phase cap .4rad。Qiskit1.3.0/compiler A/native RZ-SX-X-CX/全結合、B′4-CX controlled-RZZ、reference F/G・actual inverse・vacuum P・8-CX CSWAP、D融合・basis共有・X/Y wrapperを固定した。[SCOPE](SCOPE.md)参照。

既存の `QiskitDFPartialS2CircuitBuilder._append_block`、`compile_reference`、`pr_reference_depth_v5.stats/transfer_matrix/frontier_depth/catalogues`、fence、保存common cacheを再利用した。旧V5のmoduleは変更せず、新module `src/trottertracks/pr_reference_depth_v5_reinforcement.py` に逆Dのcontext検査・非X鎖の資格判断・条件付き対応照合だけを追加した。旧runner、旧STOP、旧成果物は再開・変更していない。

## Stage A：実source inverseのDAG

保存正方向nativeのadjointを代用せず、既存builderが生成する固定Dの**source回路に `QuantumCircuit.inverse()` を適用してから**compiler Aでlowerした。実行前の[PREDECLARED.json](../../../artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/PREDECLARED.json)でD0〜3 half-step(.01)、D0 full-step(.02)、referenceのrare/Z有無×X/Yの4 contextを固定した。20 inverse recordをbarrierで分離した4 batchへまとめ、独立4 workerでcompileした。正方向Dを再compileしていない。

| D構造 | 保存正方向span上界 | 実source inverseのspan（4 contextの最大） |
|---|---:|---:|
| block0 half | 223 | 223 |
| block1 half | 291 | 285 |
| block2 half | 301 | 291 |
| block3 half | 291 | 285 |
| block0 full | 223 | 223 |

各recordについて、実native列からのmax-plus span、Qiskit DAG depth、既存wire frontierが整数一致した。post-unroll gate族も保存contextと一致した。sourceの演算順序反転、実角符号、global phase反転、最終pass前後の行列・fence・実native列を保存した。[STAGE_A.json.xz](../../../artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/STAGE_A.json.xz)、[raw 4 batch](../../../artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/stage_A_raw.tar.xz)。

**正方向とinverseの依存行列は一般には一致しなかった。** 旧V5のmultiset説明だけでは上界移送の根拠が不足していた。今回確認したのは、実source inverseをコンパイルしたspanが、必要な全block/contextで保存上界以下となることである。`Compile(C†)=Compile(C)†` や同一matrixを仮定しない。

q=1では既存SOURCEのFがreverse deterministic half列であり、そのinverseは今回の4 inverse half blockを逆順に並べたものになる。q≥2のlaneでも、fusion済みD0 full blockとhalf endpointを含む既存の正のblock多重度がsource inverseで逆順になる。各局所spanが既存値以下なら、その直列span和は順序にかかわらず安全な上界である。全laneの正確なfrontierや、巨大なwhole回路の構造一致を証明したとは扱わない。

13 basis SOURCEについて、RZ/XXPlusYY族の実角・wire順序を追い、double inverseが元のbinary parameter列へ戻り、対応するbasis/basis_inverseが保存catalogueにあることも確認した。[SOURCE_INVERSE_CHECKS.json](../../../artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/SOURCE_INVERSE_CHECKS.json)。RTEの逆向きbasis-runはこの閉包と保存pair spanを使う。reference下界は正方向G laneだけでも支持されるため、Fのδを正方向値と同一と仮定する必要はない。

## Stage B：CX→X相殺と非X依存鎖

B′全4 contextのD/frame 144 recordを確認した。保存された最終local native列は全てXを含まなかった。そのうち旧下界が実際に参照するものは、D chunk 5種類と、保存全18round・両軸に現れる7 basisのforward/inverse 14種類、合計19種類×4 contextの76 recordである。旧δとXを全て除いた部分列のδ・max-plus行列を照合し、変更がないことを確認した。[STAGE_B.json.xz](../../../artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/STAGE_B.json.xz)。

保存localのX-freeは、**pre-last内部にもXがなかったという観測ではない**。その全命令列は保存されていないため未取得と記録した。whole QPYの一部にXがないという観測を、全supportの根拠にもしていない。

必要な論証は次の通り。

1. SOURCE Gaussian frameはRZ/XXPlusYYで、初段InverseCancellationの自己逆gate対象ではない。Dのdirectional CXは局所chunk内で処理され、境界は保存frameで隔てられる。旧B′のSOURCE境界論証と固定contextの列挙を引き継ぐ。
2. 最終passは**CXを処理した後にXを処理する**。対象D/frameのCX fenceはX削除前のものを確認している。したがって、このCX phaseで外部runがchunk内部の保護CXへ入ることはない。
3. その後のX phaseは、保存X-free局所鎖のCX/RZ/SXノードを削除できない。局所内又は境界のXが追加で消えても、保存した非X列のwire順序部分列は残る。逆にXが残る場合も、非X列へgateが挿入されるだけなので、その依存鎖のdepth下界を小さくしない。
4. よって下界は**保持された非X部分回路**のδにより支えられる。Xを含む一般の依存鎖の保護を、CX fenceだけから推論していない。

固定pass順序はQiskit1.3.0の[Python wrapper](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/passes/optimization/inverse_cancellation.py)と[Rust実装](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/inverse_cancellation.rs)を確認した。gate-list順でrunを収集・消去し、CXへ戻らないという範囲に限定する。

人工例 `X—CX—CX—X` では、CX削除後にX-Xも実際に消えた。この一般的な危険を否定せず、今回の下界にXを含めないことで切り分けた。source inverseの順序反転と、非一様frontierに対する非X部分列の下界も含む[人工3検査](../../../artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/SYNTHETIC_TESTS.json)を通過した。これらを科学性能の証拠にはしない。

## Stage C：旧会計を維持

Aのinverse span上界は全て旧値以下、Bのδは全て旧値と同一だった。保存18round・X/Yの適用条件を対応付け、旧 `DEPTH_18ROUND.json` の総計・比較値をそのまま参照した。[STAGE_C.json](../../../artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/STAGE_C.json)。新しい総計の再計算は0回。より小さいinverse spanを使った上界の引下げや、改善率の再最適化は行っていない。

| 18round shot加重期待native DAG depth | B′ | Reference |
|---|---:|---:|
| 維持した下界（安全側の概数） | ≥9.19338940794×10¹¹ | ≥4.17942775235×10¹¹ |
| 維持した上界（安全側の概数） | ≤1.14643841539×10¹² | ≤5.4244718400×10¹¹ |
| 十分shots | 8,598 | 8,606 |
| qubits | 9 | 17 |

reference上界<B′下界により、既存の相対削減率下限40.995952643300015%を維持する。これは正確な期待depthの算出ではなく、固定モデル下の境界比較である。reset/測定込みの保存depth規約、既存CX削減・RZ増加、V3全任意角未資格・総T未取得も変更しない。

## 実行・未資格事項・STOP

新local compileは4 batch／20 inverse recordのみ。科学対象の新whole compile、trajectory、sampling/simulation、Hamiltonian/ground生成、T合成は0。保存common cacheを再利用し、旧科学suite・QPY照合・18round計算を不要に再実行していない。実affinity64、MemAvailable約493GiBを観測し、独立4 contextは4 workerで処理した。共有設定・他者process/fileへ変更を加えていない。[SOURCE・入力・環境・実行記録](../../../artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/README.md)。

今回補強したのは指摘2点の成立条件であり、固定compiler/保存H4・canonical law・算術／信号モデルという制約は残る。形式証明、最適depth、実機時間、routing/noise、space-time、総T/T-depth、全資源優越、新規性・論文化価値の認定は対象外。追加8 qubit・RZ増加をdepthやCXへ恣意的に合算しない。

**mandatory STOP。次担当はGPT／ユーザー**：補強した構造論証と残存制約を評価し、研究上の判断を行う。Codexは自動的に追加最適化・T合成・長round取得・別Trackへ進まない。旧SOURCE・結果・STOPは保全した。
