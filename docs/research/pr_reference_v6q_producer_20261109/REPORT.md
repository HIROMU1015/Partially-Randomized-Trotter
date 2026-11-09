# 新A V6-Q：canonical finite-bit event producer

2026-11-09、SERVER 2。**producerを実装し、人工8テストPASS。固定確率モデルの下でcanonical lawとconsumerへのSOURCE接続を確認した。** 実H4のsampling、4 kernel fixture、m17 whole、回路build/compile/simulation、RPE・総RZ再会計は実行していない。

[PRODUCER_CONTRACT](PRODUCER_CONTRACT.md)に、単一draw・event kernel・列の分布、独立性の前提と実装の境界を記録した。これは著者の実装・技術検証であり、独立科学レビューや新しい研究方針の正式採択ではない。

## 1. 実装と再利用

基点は`87e9b11a672e380915866a73253be100e75ff427`。新しい[producer module](../../../src/trottertracks/pr_reference_v6q_producer.py)だけを追加し、既存CDF／draw／_make_event、provider、caller、consumer、compiler、reset SOURCEを変更していない。

| 追加部分 | 使用する既存部品・意味 |
|---|---|
| EventKernel | integer_cdf／draw_cdfを直接使用。order選択→n+1 component選択→_make_eventを一度呼ぶ。floatによる事前正規化や再配列なし |
| fixed_parameters | 旧transferと同じtolerance式、tail、h=.02、保存r、K2をmake_rte_configへ渡す。旧NumPy条件付き抽選は使用しない |
| CanonicalFiniteBitProducer | 明示されたgetrandbits sourceを保持するsupplier。保存metadataを照合し、q*r個の新eventを旧順序で返す。既定RNG／seed方式、event cache、reweightingはない |
| consume_canonical_axis | 既存consume_saved_roundへ実producerを渡し、指定した1軸のnative回路だけを返す。次のaxis-shotでsupplierを再呼出しし、同じtrajectoryの流用を避けるinterface |

実供給器のtypeとfresh-independent-uniformという仮定を返値で区別する。RTEConfigの旧seed/PRNG欄はRNG初期化に使わない。consumerのSOURCE文字列だけを資格根拠とせず、実際の呼出し先と[分布論証](PRODUCER_CONTRACT.md)に対応付けた。

## 2. 人工検証

[test_pr_reference_v6q_producer.py](../../../tests/test_pr_reference_v6q_producer.py)の**8テスト全PASS**。既存project venvで`python -B tests/test_pr_reference_v6q_producer.py -v`を実行した。前段のcaller／consumer／Q1 mock helperを再利用したが、旧test suite・科学runner・検算は再実行していない。

- 人工nominal binary64のexact-rational正規化、CDFの端点・境界・zero cell、正support消失の拒否。
- 2-bit人工分布の最大4 drawを全256整数列で列挙し、選択結果の全kernel質量がorder×component積と一致。実samplingやMonte Carloではない。
- 人工component／tauについて既存の**実_make_event関数**を使い、order0/2、rotation/product順序、符号・scalar位相、angleのhex、normalizationが直接constructor参照と一致。bit幅4/66/88で確認した。
- sourceの継続利用、1 eventに2/4回のdraw、新event tuple／object、再seedなし、範囲外word・不正metadata／axisの拒否。
- 実supplierから人工consumer・compiler/resetまでの接続、X/Y物理shot要求で別event生成、m0のlegacyとm17 opt-inの受渡し。他17roundのselectorは既存SOURCEを不変更で継承。m17 testはkernel構築直前で止め、q*r列を生成していない。

人工RTE constructorのためNumPyと既存RTE dataclassを読んだが、H4 cache・Hamiltonian・groundは一切読んでいない。量子回路はmockだけで、実QuantumCircuit／compile／simulationを呼んでいない。人工テストPASSを実H4の有限数値結果へ読み替えない。

## 3. 分布・独立性・既存保証

SOURCEは、旧integer floor CDFで定まるcategory mass `(C_i−C_(i−1))/2^b`をそのまま実装する。fresh independent uniform wordモデルの下で、event kernelはorder質量×n+1個のcomponent質量の積となる。variable長drawでも未使用IID suffixは保持され、1 event最大4 drawなので旧4qrのproposal disk条件に合う。

_make_eventが持つevent_probability等は旧nominal metadataのままで、finite-bit実選択質量へ書換えたり、weight補正に使ったりしない。既存finite-bit proposal error／disk、normalization、raw signal rangeの扱いを維持する。新しいlaw・importance sampling・確率補正はない。

実装はgetrandbitsを呼び、新しいwordの利用とstateを戻さないことを保証する。**任意の実乱数器について一様性・数学的独立性を証明したものではない。** 具体的なRNGは今回採択しておらず、固定確率モデルの入力条件として残る。shot／axis／round／worker間で同じseedやstateをcopyしてよいという意味でもない。

## 4. Q3-N・RZ・V6-Pへの移送資格

固定条件はLinear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外、Partial S2／LD3／h=.02／K2、ordinary diag(I,V)、保存18roundのq/r/bits、Qiskit1.3.0／Compiler A、B′4-CX controlled-RZZ／既存Reference構成である。精度1.5936e-4 Ha、failure .05、phase cap .4、αaxis1/720を保持する。

参照する科学結果は[Q3-N](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/543b711a4da6cb291af1bb5665db602d609ad272/docs/research/pr_reference_v6q3_numeric_20261109/REPORT.md)と[V6-P](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f672ce6884bd49231366875a9e24a539fcaec7a/docs/research/pr_reference_v6p_results_20261109/REPORT.md)。元科学SOURCEはf2638a7845685a119619382b850441f5d8d33713である。

| 対象 | 結論と前提 |
|---|---|
| Canonical producer SOURCE | **実装・対応確認済み**。fresh independent uniform bitsを仮定する固定lawそのもの |
| Consumer接続 | **完了**。固定cache context→保存metadata→実supplier→旧F/G/caller→Compiler A/reset。実H4起動は未実行 |
| Q3-N | 同じcache、basis／angle文法と旧算術モデルを保つため、13 basis／135 generator／ℓ=65のnormal域を継承 |
| B1誤差 | exact binary取得／F+G和／最後1回RNと`δ_B1≤B_norm+2uM`を保持。旧normalizationや他diskは消さない |
| RZ単価 | P角だけの変更、同じcompiler context／resetなので既存構造保証を維持。新compile・価格取得は不要 |
| RPE／総RZ | **元の固定モデル内で、実装前提を具体的SOURCEへ移送できる**。今回、shot・総計の再計算や本番RPE実行を認定していない |

従って、前段で未実装だったcanonical producerの穴は、元の確率・算術モデル下のSOURCEとして埋まった。V6-Pの位相上界5.172494657527835e-08 rad、m17 Reference 630→626 shots/axis、全18roundで強B′比少なくとも10,360,815,280.326 native RZ削減という保存不等式へ、同じ入力・測定／独立性・compiler条件の下で接続できる。一般の乱数器・実機や別条件へ無条件に拡張せず、総Clifford+T優越とも混同しない。

## 5. 未確認事項・次の判断・STOP

具体的な乱数供給器の選択と運用、数学的IIDモデルを実行環境でどう扱うか、本番shot loop／backend、実機RPE・総T/T-depthは未評価。これはSOURCEで確認した固定モデル内の分布保証と区別する。既存結果をそのモデル下の保証付き実装評価として正式採択するかはGPT／ユーザーが判断する。

SOURCE証拠と人工検証で今回の接続条件は整理できたため、新しい実H4数値検証を必須とはしない。必要と判断された場合だけ、[既存4 kernel fixture案§2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/87e9b11a672e380915866a73253be100e75ff427/docs/research/pr_reference_v6q_consumer_20261109/NEXT_VALIDATION_PLAN.md)を再利用する。m0 r1/bits66とm17対応r32/bits88、各order0/2、index規則・CDF cell左端を保持し、その時点のproducer固定SHA・入力・call／STOPを確定して別途実行承認を受ける。今回この4 fixtureを評価せず、新しいNEXT_VALIDATION_PLANは重複作成していない。

既存branch `codex/pr-reference-v6q3-numeric-20261109`を87e9b11aからfast-forwardで継続する。階層はアルゴリズム改善→新A/Reference Interference→V6-Q canonical producer。変更は新module・人工test・2文書の4ファイルだけ。旧SOURCE/input/raw/results/grant/ledger/marker/STOP、他Track、共有設定・他利用者process/fileは不変更。11-09版共通指示・AGENTS・PROJECT_MAPを適用し、AI subagentは使用していない。

**commit/push・remote SHA／4ファイル内容照合後mandatory STOP。** 次担当はGPT／ユーザー：固定モデル内のV6-P結果の採否、乱数供給と後続実行の必要性を判断する。Codexは実H4 sampling、4 fixture、m17 whole、追加compile／RPE／総RZ再会計、CT評価へ自動進行しない。
