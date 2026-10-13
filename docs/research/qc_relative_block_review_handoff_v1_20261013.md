# 相対時間発展block事前検証：次の科学判断

2026-10-13 JST。[技術結果・構成・誤差・費用](qc_relative_block_prevalidation_v1_20261013.md)、[完了済み方針レビュー](quantum_chemistry_ground_energy_algorithm_direction_review_2026-10-13.md)、[未認可pilot草案](../../artifacts/qc_relative_block_prevalidation/2026-10-13/pilot_design_NOT_AUTHORIZED.json)を一組として読む。

方法差の照合、数学構成、可逆encoding/parity設計、共通detuning近似の誤差会計、全励起を含む費用式、13固定synthetic検査まで完了した。新分子・固有値・科学数値・compile benchmark0。
算術で全occupation枝列挙を避ける構成を定義できたが、既知conditional SU2＋関数算術への対応があり、非自明な独立差は未特定。相殺による係数精度緩和も、強い既存compressed対照へ同等に与える。
function evaluatorの全gate/garbage/guard precision、具体的なT/depth/ancilla定数は未閉鎖。構造IRとmath.angle referenceを完成quantum circuitとしない。

次に判断する事項：

1. 相殺に対応した精度・生成法のどこに既知手法を超える寄与を求めるか。独立差を特定できない現constructorを区切るか。
2. 具体的な関数backendと資源比較を閉じる投資に価値があるか。未計測情報の補充だけを研究目的にしない。
3. 第一候補を継続するなら、H2単独でなくH4を含むdevelopment pilotと強いcompressed/native/DF対照の必要scopeを採択するか。現草案はsource/input未freeze、launch不可。
4. 第一候補不成立の場合、基底エネルギー低コスト化の目的を維持し、第二候補等を改めて比較するか。自動採択しない。

元レビュー第12節の方法差・coherent access/費用に関する科学判断の境界。source/test/公開だけを理由に同じ科学レビューを繰り返す提案ではない。
新しいGPTレビューは未開始で、開始は改めて明示承認。分子科学batch/compile/第二候補起動は別の認可、Mandatory STOP。
N1停止/N2保留/N3 closure、旧Track結果とSTOP、global manifest既存description欠落FAILを保持する。
