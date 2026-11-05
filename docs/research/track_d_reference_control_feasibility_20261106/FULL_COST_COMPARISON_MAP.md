# Track D：full interrogationの費用比較地図

2026-11-06 JST。保存済み価格の引用と構成要素の整理。新しい資源数値・改善率・総費用の計算はない。信号の必要条件は [SIGNAL_PRESERVATION_MAP](SIGNAL_PRESERVATION_MAP.md)。

## 1. 同じtaskと会計単位

元H、元Hの厳密基底状態、精度、成功確率、path/law、round/axis、weights、normalizationを同じにする。比較指標aごとに

\[
G_{\mathrm{total}}^{(a)}
=\sum_{m,b}N_{m,b}\,
\mathbb E[C_{m,b}^{\mathrm{full,no\text{-}prep},(a)}]
\]

を使う。RZ、CX、T、depthは別々。qubit数と古典処理も別欄。shot depthの和を並列実行のwall timeとは呼ばない。

基準となる比較集合は、既存diagonal-only DF/Pauli control、既存basis共有とboundary fusion、適用可能なnative symmetric-directional、phase保持compileの各構成である。一つのSOURCEで同時に実現していない最適化の数値を合成しない。全armに同じ合成精度、compiler、routing条件と許された最適化を与える。

Referenceでは、同じ **full ordered pathの連続部分** A/Bを両laneへ置く。全gate数の総和はA†とBの実実装に依存し、並列化によるdepth減少とgate count減少を分ける。cutで失う融合も会計する。各stepを別のgadgetへ置換してCSWAPを増やし、途中back-actionを無視する比較はしない。

## 2. 強baseline後に残る操作

| 構成要素 | 現baselineの必要操作 | Referenceで除ける候補／新たに必要なもの | 既存最適化との重複・確定度 |
|---|---|---|---|
| Gaussian basis | 非制御forward/inverse。既存run共有・support/full選択 | 制御除去としての追加利益なし。A†側の実inverse、cutに露出したbasisが必要 | 全gate制御との比較は過大。既存builder確認済み、cut後価格は未取得 |
| random rotation | 中央CRZ/CRZZ、parity、ancilla接続 | 通常RZ/RZZへ置換し制御由来rotation/entanglerを除ける可能性 | symmetric構成でも中央tailに残る具体的対象。新compiled差は不明 |
| random product | supportへのCZ、sign phase | 非制御Z productへ。既知signはreference補正へ | Clifford entanglerの削減候補。任意角T数の同率削減とはしない |
| deterministic half | ordinaryでは両half中央を制御。nativeでは前半uncontrolled／後半directional | directionalのancilla parity接続除去の候補。既に任意角rotation削減済みの部分を再計上しない | symmetric最適化と重複が強い。Gaussian以外の残余CX/depthは別途比較 |
| same-basis・D境界 | event/run共有、legacyの同block half fusion | cut両側を独立に合成することで相殺・rotation融合を失い得る | Referenceで同じ共有が可能か未確認。通常の共有は全arm共通 |
| scalar/event phase | ordinaryのancilla P、directionalのphase処理、既存集約 | path依存reference phaseのPと古典角度計算。省いたglobal scalarも復元 | control phaseを全て無料で除去できない。合成精度・集約規則は未評価 |
| 測定wrapper | H、YならSdg、H、ancilla測定 | 同じ読出しに加えgadgetの干渉操作 | 既存wrapperの費用を両armで保持 |
| register間干渉 | 直接制御には不要 | SE原著の前後2 CSWAP cascade、control配線 | n組の交換を2層。native分解・routing・T/depthは新compileなしでは不明 |
| qubits | n system＋ancilla | n-qubit reference追加。cat方式なら更にfan-out workspace | 初期state準備除外は追加qubit維持やcat生成/逆演算無料の意味ではない |
| layout/connectivity | 既存pathとancillaのrouting | 二つのn-register間のmapping、SWAP経路、必要ならcat配線 | topology-free保存価格からhardware利益を推定しない |
| reference reuse/reset | 不要 | 初期化、必要ならreset/測定、保持。途中reuseの条件 | post-selection時はtrial分母と必要Nを含め直す |
| fault-tolerant合成 | 任意角rotationと既存Clifford | A†、reference補正、CSWAPの合成と許容誤差 | RZ数だけでTを決めない。同じenergy精度の合成誤差を共有 |

原著 [SE-QPE v2](https://arxiv.org/abs/2604.14921v2) §III/Table IのCSWAP分解・cat overheadは原著のcost modelであり、今回の新資源評価ではない。その表を既存Qiskit実測値へ直接足してreference価格としない。balanced二laneのdepth利得も、同じrandom pathのcutとroutingが対応するときの候補である。

## 3. 保存された実回路価格の例

以下は新しい評価ではなく、[AX-2B H4 v5報告](../track_a_ax2b_h4_pilot_v5_result.md)と既存個別cost JSONの転記。public carrierは [main固定版](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/546bca7ed2cece59b714b986a264d7b0b0871a01)。

条件：linear H4、1.00 Å、STO-3G、legacy DF rank12、8 system qubits、generation-prefix L_D=6＋deterministic one-body、二次PF、T=0.8、q=4、δ=0.2、R=8/r=2、short-step h=0.1、K=2。保存normalized stateはN=4、Nα=Nβ=2のsector内で、元報告はground-state資格を認定していない。今回仮定する元H exact groundの科学結果へ読み替えない。

同じreplica 0のexplicit trajectory、cosine、測定付きfull wrapper、state preparationなし。Qiskit1.3.0、`rz,sx,x,cx`、opt1、seed17、backend/couplingなし。

| 既存個別record | RZ count | CX count | total depth |
|---|---:|---:|---:|
| [ordinary](../../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/H4_B2_K2_rep0_ordinary_cosine_cost.json) | 44,422 | 24,806 | 42,182 |
| [symmetric_directional](../../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/H4_B2_K2_rep0_symmetric_directional_cosine_cost.json) | 20,906 | 9,766 | 11,656 |

両recordのevent digestは同一。これは単一保存trajectoryのcompile価格であり、母平均、Reference価格、必要N、全RPE総費用や最良control policyの判定ではない。新しい差・比率・再平均は計算していない。

SOURCEはbase b2e1bf65＋実行前manifestの未commit closureであり、SOURCE全体の単一固定commitは本資料から認定できない。[baseline監査](CONTROL_BASELINE_AUDIT.md) §5の3部品hash照合と、過去runのclosureを区別する。この由来の制限を隠してmainから再生成した値とは呼ばない。

## 4. 価格証拠の混同を避ける

| 証拠 | 今回利用できる範囲 | 使えない読み替え |
|---|---|---|
| 上記full-wrapper実compile | 強い既存controlを実測した事実、同一trajectoryの対応 | new referenceの価格や同精度の全RPE改善 |
| [native理論A/B](../track_a_native_df_theoretical_gate_model_20261016.md) | 構成要素、固定lowering、許された相殺境界 | compiled CX/RZ/depthの推定実測、T保証 |
| WP06などのevent/occurrence価格 | same/different basis、非加法性、support選択の既存知見 | additive primitive価格をfull-wrapperと呼ぶこと |
| [SE原著§III–IV](https://arxiv.org/abs/2604.14921v2) | diagonal-only対照、追加registerとCSWAPによる交換 | symmetric最適化後にも同じ原著削減率が残ること |
| [Simon–Love §III](https://arxiv.org/abs/2511.13855v1) | 対称PFのcontrol任意角負担削減 | 非対称random pathにも同じ削減が自動成立すること |
| [PR App.E21–E24](https://arxiv.org/abs/2503.05647v2) | 全roundのshot×費用、det/randomの別会計、決定論側の既存削減 | 原著RMSE会計を任意成功確率や本repositoryのfull-native価格へ直結すること |

## 5. 全RPE費用へ接続できる条件

一shotの条件付き測定law、weights、trial分母、fresh-IID、decoderと合成誤差資格を保てれば、同じNを使って各資源指標の価格差だけを全roundへ接続できる。どのroundをreferenceで置換するかは、各roundのfull価格と追加qubits/depthを確認してから決める。高powerだけの混在自体はSE原著に既知の考え方である。

分布保存が未確認ならNは未決定で、単価減少だけでは総費用の低下にならない。合成精度変更、post-selection、reference leakageを伴うなら追加の統計/誤差接続が要る。今回の資料には最適化済みsymmetric/共有構成とReferenceの同条件native比較がなく、改善の有無・率・全RPE順位は確定できない。
