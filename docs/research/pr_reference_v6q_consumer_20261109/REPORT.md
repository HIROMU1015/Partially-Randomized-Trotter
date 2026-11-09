# 新A V6-Q：外側consumer接続

2026-11-09、SERVER 2。**固定cache／入力interfaceと、既存caller→Compiler A→compile後resetの接続を実装し、人工9テストPASS。Canonical finite-bit event producerは未実装・未資格。** V6-Pの約103.61億RZ削減は引き続き固定モデル内の条件付き結果であり、end-to-endの実装済み科学成果へは昇格しない。新しい科学計算は0件。

[接続構造と保証境界](CONSUMER_INTEGRATION.md)／[残った箇所の次scope](NEXT_VALIDATION_PLAN.md)。これは著者の技術検証であり、独立科学レビュー・研究結論の正式採択ではない。

## 1. 再利用と不足していたSOURCE

基点はcaller接続`5fdb68e4fc59f4e4c95270381043d46f074bff35`。Q3-N `543b711a4da6cb291af1bb5665db602d609ad272`、V6-P `5f672ce6884bd49231366875a9e24a539fcaec7a`、元科学SOURCE `f2638a7845685a119619382b850441f5d8d33713`を保全した。保存schedule、provider、F/G、compiler、過去の科学runner・検算は再実行していない。

| 既存資産 | 確認した役割・今回の扱い |
|---|---|
| track_c_h4_finite_bits.integer_cdf／draw_cdf | exact-rational正規化・integer floor CDF／getrandbitsによるdraw部品。固定H4のorder/component選択からeventまでの既存producerは見当たらない |
| rte.iter_sample_rte_events／sample_rte_events | PCG64＋NumPy choiceによるclassical sampler。保存finite-bit lawそのものとは扱えない |
| transfer.events_for／sample、pilotの価格cohort | rare/nonrare条件付き価格取得。RPE用のfresh independent finite-bit samplerへ流用しない |
| pr_reference_v6q_schedule | 保存m/q/r/bitsの照合、m17のみ明示opt-in、同一F/G・coreを共有するX/Y。byte不変更で再利用 |
| Compiler A、transfer.sampleのreset部分 | 固定compilerの後で17q/1clbit回路へreset(8…15)を置き、nativeをcomposeする規約を再利用 |

検索は新A関連、既存finite-bit helperとその参照、RTE event生成に限定した。他Trackに別条件のsampling codeが存在しても同条件のproducerとは認定せず、旧研究を再開・変更しない。今回、新しいlaw／event samplerを設計・採択・実装していない。

## 2. 追加実装と人工検証

[新module](../../../src/trottertracks/pr_reference_v6q_consumer.py)を追加し、旧SOURCEは変更していない。

- load_fixed_inputs：固定pickle hashを検証し、旧read-only common()を使うcontext factory。別prepが既にLRUへ入っているprocessでは再ラベルやcache clearをせず拒否する。返却contextは各requestで再利用できる。
- consume_saved_round：保存scheduleを照合し、明示supplierへ同じmetadata/prepを渡す。返されたcfg/dist/eventsを既存callerへ渡し、旧Compiler Aとreset規約へ接続。既定samplerやcohort fallbackはない。
- 拒否範囲：cache hash／active common identity、prep identity、宣言されたlaw／producer SOURCEの欠損、tail ID/hash／λ、保存scheduleの不一致。既存callerのr/h/K/dist・event長検査とbuilderのbasis metadata検査も保持する。
- 限界：lawやproducer SOURCEは科学的来歴の申告であり、独立性の証明ではない。返却値にも`NOT_ESTABLISHED_BY_THIS_INTERFACE`を明記し、申告だけでcanonical資格を与えない。

[人工test](../../../tests/test_pr_reference_v6q_consumer.py)を`python3 -B tests/test_pr_reference_v6q_consumer.py -v`で実行し、**9テスト全PASS**。前段Q1/callerのmock・lane helperを再利用したが、過去のtest suiteは再実行していない。m17 opt-in／他round legacy、同じevents・F/G・coreの受渡し、X/Y共通core、compiler入力、compile→reset回路作成→native composeの順、reset対象8…15、誤入力拒否、cache bindingと環境復元を検査した。人工cacheは単なる短いbyte列で、pickleをunpickleしていない。Qiskit／NumPy／trotterlibもimportせず、実科学sampling・回路build/compile/simulationは0件。

## 3. 保証・価格・V6-Pへの移送判定

科学条件はLinear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外、Partial S2／LD3／h=.02／K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit1.3.0／compiler A。m17/q131072/r32/bits88、精度1.5936e-4 Ha、failure .05、phase cap .4、αaxis1/720、B′4-CX controlled-RZZ／旧Reference構成を保持する。

| 項目 | 到達点 |
|---|---|
| 固定cache→実assembly | byte hash・loader identity・prep/tail/λを結ぶ実装と人工検証まで完了。実cacheで新consumerを起動していない |
| Canonical law | **未資格**。CDF/draw部品はあるが、固定lawのevent producerのSOURCEが未接続。申告・有限event適格性は分布全体の証明と区別 |
| Q3-N | 同じ固定SOURCE／適格event生成／旧算術モデルならℓ=65を継承。新cache・任意event lawへは拡張しない |
| B1／M・B_norm | 既存exact和＋最後1回RN、`δ_B1≤B_norm+2uM`を維持。normalizationや他diskを削除せず、新δ評価なし |
| Compiler／reset | 既存順序を接続。resetはコンパイル処理の後に挿入し、物理実行ではnativeの前。新pass／control／relative phase変更なし |
| Native RZ | 固定contextでP角だけが違い、resetは同じ8本でRZを追加しない。既存の構造的価格不変条件を保持。新compile／価格取得不要 |
| 626 shots／全18round総RZ | [V6-P保存値](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f672ce6884bd49231366875a9e24a539fcaec7a/docs/research/pr_reference_v6p_results_20261109/REPORT.md)を再計算しない。producer未資格のためend-to-endへの無条件移送は未成立 |

保存V6-Pは、位相上界5.172494657527835e-08 rad、m17 Reference 630→626 shots/axis、強B′比で少なくとも10,360,815,280.326 native RZ削減という条件付き結果である。今回、compiler/reset側の未接続を技術的に埋めたが、producerの不足を0誤差・実装済みとして補完していない。総T-count／T-depthの結果ではない。

## 4. 保全・次担当・STOP

研究階層はアルゴリズム改善→新A/Reference Interference→V6-Q consumer。同じ作業線の`codex/pr-reference-v6q3-numeric-20261109`をfast-forwardで継続する。変更は新module、人工test、本ディレクトリの3文書だけ。旧SOURCE/input/raw/result/grant/ledger/marker/STOPと他Trackは不変更。11-09版正式共通指示§8/§10.4、AGENTS、PROJECT_MAPを適用し、共有設定や他利用者のfile/processを変更していない。AI subagentも使用していない。

**commit/push・remote SHA／5ファイル内容一致の照合後mandatory STOP。** 次担当はGPT／ユーザー：canonical producerのSOURCEをどう確定・採択するかを判断する。新しい科学samplerの実装と実入力による検証は今回開始しない。既存compiler contextの再compileや全18round再会計を必須とはせず、残るproducer部分にだけ次scopeを限定した。
