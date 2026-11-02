# Paper Claim Sheet：Track B A/B

2026-10-18。〔既存〕の証拠と〔技術推論〕の主張候補を区別する。**以下の仮題・主張は未採択で、論文草稿・新定理ではない。** [詳細対応](prior_art_method_comparison.md)／[固定証拠](evidence_index.json)。

## A：局所return係数に基づく有限乱択時間発展の生成

**仮題**：Local generation of return-aggregated finite randomized time evolution with explicit finite-bit accounting.

**研究問題**：全形式語の辞書を展開せず、同じ有限奇数次数Taylor targetを表す乱択回路を生成できるか。係数取得・有限bit補正・controlled位相・native費用を含め、同じ情報の既知法と異なる資源上の価値があるか。

**核となる主張案**：正の有理数p、Hermitian involution Qᵢ²=I、奇数m、0<x≤1の既存仮定の下で、free-word returnを先に集約し、parent局所query、ordinary envelopeによるzero-fill、補正weightを接続する。P5では少数群からnormalizerを取得する専用経路もある。**生成可能性は既存成果、既知法との差・資源上の優位は未確立の主張候補**である。

**支持される内容**：G6のideal有限target・非負性・係数query・finite-bit bias/range会計、G7/G8の局所生成interfaceと条件付きnative経路、G9の固定P5 native比較、G10の次数別比較と固定辞書の任意proposal下界。一般kernelの既存算術量はsetup O(Lm²)、有効parent query O(m³+Lm²)。bit費用・回転取得費用は別。G10では一般fullがclosed P5+tailに勝たない固定例もある。

**最も近い既知法・不確実性**：Aomoto–KatoのGreen構造、Wan–Berta–CampbellのEuler-paired RTE、Zhao–Yuanの低次数寄与吸収、SCU/CTSのcollection・layer sampling、cost-aware IS、SPS。個々の原理は既知。全有限return・局所生成・有限bitの結合が、同情報・同最適化・同誤差条件で既知手順へ還元されるかは未確定。文献で完全一致を発見していないことは新規性の証明ではない。

**決定的な不足／不成立条件**：強い非列挙対照にも同じ係数query、収集、価格取得、最適化を許した対応が未完。未知normalizerを避けても、range・拒否・online合成を含む総費用利益は自動的に得られない。通常の既知処理で同じlaw・保証・計算量が得られるなら、新方法としての中心主張は不成立。独立差があっても取得費用が利益を消すなら「資源効率のよい方法」という主張は支持されない。

**図表候補と限界**：既存の生成手順図、有限bit誤差台帳、次数別固定比較・負例。新scaling曲線や新provider表はまだ存在しない。現状の技術分類は **`A_INSUFFICIENT_EVIDENCE`**。独立アルゴリズム論文としては現在未成立、狭い構成ノートの可能性は判断留保。

## B：有限RTEの保証・価格・統計費用を接続する方式選択

**仮題**：When normalization minimization fails to minimize native resources in finite randomized time evolution.

**研究問題／主張案**：同じ精度・confidenceの下で、係数L1、実weight moment/range、bias、離散native価格、prep/readoutを分離し、方式選択を予測できるか。保存された順位反転の説明を越え、未知の入力にも有効な選択規則を与えられるか。

**現在の支持**：D1–D3は同じsynthetic providerにおけるr・保証・価格・literal相殺の寄与を分離。S4は訂正済み70 lawで、common対noncommonのbest-to-bestに約14%（ordinary）・19%（P5）のT低下を保存している。このS4の2比較は3-system-qubit development synthetic、finite P5、r=1、x=τ=5/7、state000、strict primitive 10⁻⁶、各axis ε=1/200・α=1/40、加法native価格、h_T=0の条件。分子geometry/basis/DF rank/L_Dは非該当。proposalも異なり配分単独の因果効果ではない。N・CX増加を伴う。

**既知法との差と不足**：shots×回路価格、誤差budget、安いgateとweightの交換自体はIS/SPS/SCUで既知。保存反例・条件付き診断はあるが、それらを越える新しい評価指標、一般化可能なselector、独立入力での予測保証はない。固定価格KKTも標準凸最適化で、native離散価格の最適化ではない。

**不成立条件／図表／限界**：既知の総費用式への保存値代入と事後説明だけで完結するなら、独立方法論の核にはしない。利用可能なのは既存D3の保証×価格4ケース、S4のT/N/CX trade-offと既存pairのprep感度、G10の制約付き負例。新しいh最適包絡線は未計算。多stepのnormalization下界を総T比へ読み替えない。現状は **`B_DESCRIPTIVE_RESULTS_ONLY`**。

**判断への引継ぎ**：A/Bとも独立論文の主張としては保留を推奨する技術的根拠がある。まずAの結合構成に残る差の価値を判断する限定的な科学的受理が必要かをユーザーが決める。追加レビュー、研究方針採択、新計算はそれぞれ別途承認。現在のD＋Eとmandatory STOPは不変。
