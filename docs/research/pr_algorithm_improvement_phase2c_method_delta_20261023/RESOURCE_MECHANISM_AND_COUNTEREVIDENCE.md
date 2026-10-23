# 資源機構と反証材料

本書は設計上の交換を整理する。新しい数値、保存raw再集計、改善倍率、価格取得はない。basis／childのactual native占有率も測っていない。

## 1. 何が減り、何が増えるか

反可換対のinvolution化でP側の辞書atom数と係数のL1質量は減り得る。RTE補正normalizationと統計負担への好影響が期待される一方、Aを実装するW/W†を各childとproductに追加する。q/r固定ならchild個数は変わらない。新atomが非Pauliになったことだけを回路単価削減としない。

| 層 | 追加案による変化 | 相殺と未確定部分 |
|---|---|---|
| model・表現 | 理想のH/Rは同じ。係数辞書は再編 | finite bitの新表現差は別。元DF誤差は消えない |
| PF | D/R、順序、δ、q固定 | PF bias改善の新機構なし。finite-RTE targetへの接続条件はPOへ残る |
| normalization／stat | λ、p、次数law、θ、Bが変わる | 必要N、二次moment、rangeを共通policyで再評価する必要。VQE測定節約をPRへ転記しない |
| 1-shot | Aの中央P操作の前後にW/W† | 通常Pauli childよりarbitrary rotationとparityが増え得る。Wは係数依存角度で、既存equiangular rotationの合成上の利点も自動継承しない |
| sharing／connection | 同じAが隣接すればW†W相殺 | 既知basis共有そのもの。異なるpairのWは一般に消えず、event順序を変えられない |
| control／native | basis非制御・中央制御を共通適用 | W内部も含めRZ/CX/T/depthを別に数える。T換算・native合成は未実施 |
| classical | bit判定、pair選択、ρ/φ、sampler再構築 | 前処理とprecision・cache構築費は無料ではない。E0や最適価格oracleなしでも安価性は未確定 |
| sector／prep/readout | 初期stateと測定taskは共通 | sector-only実行を使えない場合がある。Nが変われば共通per-shot準備費も総費用では相殺しない |

## 2. 過去の証拠が制約すること

H2 Pauli/DF比較（`EV-0179`）とREP-09（`EV-0197`）は、小さいnormalizationやNの利得を生成費が上回る固定例を与える。今回その性能値を再掲せず、新Aの倍率や一般のno-goへ移さない。前者はPauli／DFの物理splitも異なるため、純粋な辞書効果の因果対照ではない。後者はsynthetic小系・固定q1/compilerの限定結果であり、あらゆるinvolution化の不能性ではない。

WP06b（`EV-0027/0028`）と実DF builder／反復builder（`P2-BUILD/REPEAT`）は、support/full選択、非制御basis、同basis相殺、境界half-step融合を既に持つ。それらを対照から落としてW共有の利得を作れない。

P-Aのinterval負結果（`EV-0032`）は強対照とのplan一致であって、すべての回路設計の無効性ではない。Gaussian積生成closure（`EV-0128`）は固有手順未定義による停止で、性能負けではない。BM-0.5（`EV-0084`）は既知構成への書換えを独立法と呼ばない制約として使う。

今回の停止理由は、これらの過去negativeを新案へ移したことではなく、追加案のAとWが既知unitary partitioningへ直接対応したことである。[出典と確認範囲](SOURCE_EVIDENCE_INDEX.json)。

## 3. 改善を主張する前に必要だが、今回は取得しないもの

既知法の応用として評価する場合であっても、同じ精度・wrapper・合法compilerで、追加Wの実費、相殺後の1-shot、B/N、準備・読出し、古典取得費を戻す必要がある。単に辞書数・λを減らしたことから総資源改善を推論できない。

独立方法差の有無は、この性能測定をすべて済ませるまで判断不能というものではない。今回はoperatorと生成recipeの既知対応が分かったため、性能pilotを新法探索の救済として開始しない。
