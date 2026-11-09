# Streaming・frontier・長qの計算量と限界

[全結果](REPORT.md)。Lは展開後native命令数、Wは量子＋古典wire数、Bは同時保持する最大の既存bounded QC/chunkの命令数とする。以下は理論評価であり、q1024〜4096の新benchmarkは行っていない。

| 経路 | 時間 | 記憶・保持対象 |
|---|---|---|
| NativeSink→T0直接 | O(L)、bounded gate展開コストを含む | T0はO(W)、sinkは全wordなし。callerのchunk O(B)・再帰definition・既存frame cacheは別に残る |
| 旧CX bridgeを一回 | O(L)、各gateのarity固定 | O(L+W)。生存nodeのwire stack、消したnodeのtombstoneも最終まで保持 |
| 1Q run合成を一回 | O(L)、各runの2×2積と有界Euler合成 | O(L+W)。現実装はpending run・出力列を持つ。2×2積だけでrunを捨てる実装ではない |
| 1Q＋CXを固定点まで | O(RL)、最悪O(L²) | O(L+W)。各非終端反復でsizeが少なくとも1減るので有限、固定小R上限なし |
| 既存T0 template | capture O(LW)、dense max-plus連結 O(W³) | 三transfer O(W²)。確定wordの連結／mappingを再利用できる |

T0 count／depthの整数計数はnative列に対して厳密。浮動小数点global phaseや1Q合成精度の全q保証まで意味しない。stream→T0だけなら全wrapper QC／full DAG／CAR targetを持たないが、今回の部分P0 bridgeは全native履歴を保持する。全P0 compile・全入力state simulationより軽い可能性はあるものの、実測RAM／速度倍率は未取得である。

## Frontierを確定できる条件

以後そのwire集合にgateが来ないことが入力契約で確定したlaneや、対象ruleに対して消えない依存blockerを証明できる境界なら、その前を確定してtemplateへ落とせる可能性がある。単にchunkを読み終えたこと、現在のwire finish time、固定個数のgateを見たことだけでは十分でない。

例 `a b … z z … b a` のような可逆native列は、中央の相殺から古い端まで連鎖できる。1Q再合成でblockerが消える場合もあるため、CX-onlyで安全に見えた境界が1Q＋CXでは安全とは限らない。今回の保存actual_inverse_pairも30反復で空になった。これは長q計算量の観測ではなく、小人工caseで固定小frontierを仮定できない例である。

T0 transfer／countだけへ早期圧縮するとopcode・角・順序を失い、後から相殺を復元できない。chunkの価格を独立加算せず、同じbridge状態へ順に送るか、未確定wordを保持する。新試験は同じwordの全cutで一致を確認したが、chunkごとに独立optimizeした結果を足す仕様ではない。

## 長qへの障害と再利用

Actual A†はdescriptor／stepを逆順に走査し各gateを反転する方式なら、全wordの反転QCを作らず取得できる。ただしその全H4 driverは未実装・未実行。event原本や各frameのSOURCEを読み飛ばしてはならない。実際のPR stepは異なるordered pathなので、同一templateをq乗する仮定を置かない。

独立A/B laneは位相付き命令生成や小2×2処理を分けられるが、shellの共通ancilla／readoutまで含むdepthは全wire依存から計数する。`shell+max(lane)`式へ置き換えない。routing／実機時間／合成精度の新コストは評価していない。

Templateは最適化済みと確定したnative列に使い、wire mapping・native word・phase・Qiskit規約が同じ条件で再利用する。今回新しい長さLの全word cacheや科学SOURCE別の管理層は作っていない。既存basis/frame/definition cacheを保持する。

q1024〜4096については、T0単体の逐次計数は構造上有望。一方、partial bridgeのO(L) Python object記憶と最悪O(L²)反復、bounded chunkを用意するコスト、P0対応不足が残るため、今回の結果だけで実行時間・RAM・P0価格精度の実用達成を判断できない。次の限定照合で残差と実資源を測り、全compiler再実装が必要なら打ち切って部分構造＋A/F/Sへ重点を移す判断材料とする。
