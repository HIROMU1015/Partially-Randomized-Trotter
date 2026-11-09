# 非full-compile価格評価法の比較と実装優先順位

2026-11-09 JST。[結論・固定domain](REPORT.md)。ここでの「exact」は対象表現と条件を必ず併記する。主資源指標・新しい価格精度・α・RPE成功確率は本作業で採択しない。

## 方法の比較

| 方法 | CX/RZ/size | RZ/CX/total depth | P0対応 | q依存・memory | 全wrapper／全入力検証 | 負担・主な用途 |
|---|---|---|---|---|---|---|
| 1 完全解析式 | 閉包済domainならexact候補。208−224qのfull P0導出は未完成。差だけで絶対価格は得られない | ordering/maxの依存が強く、qだけの式は未資格 | 今は未証明 | q等だけならO(1)、必要特徴があれば抽出O(qr) | full wrapper不要となり得る。意味の証明・資格検証は別途必要 | 証明負担が最も大きい。まずCXの限定閉包を狙う |
| 2 構造特徴による直接評価 | event/support/run/zero/boundaryのcount自体はexact。特徴から価格への写像は別 | lane-max等は説明特徴で、wire依存を省くと近似 | C7-4 featureだけでは未資格 | O(qr)、集約は小。actual basis/template cache等は別 | descriptor入力でwrapper不要。全256列は特徴計数に不要 | 最小拡張。O1/O2/O3等の入力変更を安く表せるが、それだけで優劣保証にはならない |
| 3 native stream／圧縮DAG・max-plus | 固定native列ならexact | 固定列・全wire transferならexact | 未compile descriptor→P0列のbridgeが必要 | stream O(L)/O(W)、cache適用O(KW²)。frontier閉包は未確定 | full QuantumCircuit／dense target不要。新emitter/最適化の意味資格は別 | **最優先の基盤**。既存counterとemitterを再利用し、保証の範囲を明確にできる |
| 4 構造＋経験モデル | 閉じた部分をexact、残差をproxyに分離可能 | 未閉包のP0・max/law部分をproxyとして残す | 成立済部分だけ適格化。全体の誤差をゼロにしない | 特徴抽出＋小モデル評価。大回路不要 | 予測のたびにfull correctnessは不要。校正／holdoutは別scope | 実用までの中間案。既存A/F/Sとの残差設計や新fitは未実施 |
| 既存full compile | 実出力のcountはexact integer | 実出力のdepthはexact dependency metric | その固定P0そのもの | 全IR・中間DAG等、時間は実passに依存 | 現pipelineは全入力検査を伴うが、計数とは分離可能 | 参照・限定oracleとして保存。巨大qへの一律拡張を既定にしない |

O(W)等は評価器の作業memoryであり、保存event原本・SOURCE・来歴のstorageを消すという意味ではない。Referenceは2n+1幅とCSWAP/χ負担を含む。routing/T/noise費は別途未評価。

## A/F/Sの扱い

| 既存model | 現在の役割 | 今回維持する条件 |
|---|---|---|
| A＝all-q affine | 現在の主モデル | q2..32・88pathを使用した保存仕様/係数/共分散のまま |
| F＝frozen affine | 元C6a比較 | q2/q4由来の凍結係数を変えない |
| S＝sqrt(q) | 探索的depthモデル | design列[1,q,sqrt(q)]。q64で点残差が小さくても主モデルに昇格しない |
| CX構造式 | 条件付き厳密化候補 | −224の部分説明と208/P0残差の未解決を併記 |
| DAG/interface | orderingを保持する候補 | 固定wordのexactとP0 bridge未資格を別に表示 |

q64報告のSの良さは一点群の探索診断で、q1024/4096のbias評価ではない。既存のbootstrap/係数共分散は校正変動の一部を表すだけで、全law・未観測support・model biasの上界ではない。今回新しいmodel score、SD/CI、fit、価格幅を作らない。

## 置換できるもの／まだ置換できないもの

固定native列のcount/size/depthを算定する段階では、統計モデルも状態vectorも不要になる。必要なnative列がP0後と同じことが証明できたdomainでは、該当資源をmodel proxyからexactへ置換できる可能性がある。

未解決なのは、actual P0の境界簡約、絶対価格のpath依存、全sampling lawの特徴分布、E[max]、長q biasである。最大を取る前にlaneを平均しない。C7-4保存DAGのwire endpointを未来qの入力に使うのはcompiled answerの漏洩で、非full-compile法にはならない。新しい構造評価器を使って保存H4を再計数すること自体も、今回は実施していない。

`G_g=Σ N E[C_g]`へ接続するとき、各方式の絶対価格と差の共分散、axis依存、q間の共有係数依存を維持する。幅はpeakとして別会計とする。価格精度の資格だけから共通Nやactual成功保証を更新しない。

## 実装順の提案

1. **native stream＋max-plus基盤**：既存`weighted_depth`を共通counterとして使い、全wire transferだけを追加する。既存emitter／feature抽出を薄く接続し、価格とCAR/全入力検査の生成を分離する。最初はmockで仕様を固める。
2. **CXのP0 interface bridge**：Gaussian/parity、step/cut、CSWAP/χを優先。既存反例を受け入れるdomainを分類する。証明できた範囲だけをexactにする。
3. **必要なら構造＋既存modelの残差案**：未解決項を明示し、A/F/Sを維持して比較する。新しいfitや保存値の再解析が必要なら別の科学実行範囲とする。
4. **完全解析式の拡張**：閉包証拠が揃ってから。208−224qの有限一致を先に正式価格として登録する順番にはしない。

これは設計上の優先順位で、後続実装・P0検証の開始承認ではない。q128を自動的な次の実測地点にせず、まず既存q≤64を生かせる閉包課題を絞る。
