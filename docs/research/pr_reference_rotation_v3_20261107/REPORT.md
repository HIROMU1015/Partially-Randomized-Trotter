# 新A V3：回転分類と18round再会計

2026-11-07 JST、SERVER 2。固定commit `43bfc5918703e8dcd020be4fcf96bb8dfd99f909`、独立branch `codex/pr-reference-rotation-resources-v3-20261107`。

**確定した任意角部分はreferenceでB′比2.818200%減少した。ただし未確定角度を含む全任意角数の削減は未資格。** 保存nativeのbinary64 pi定数をexact Clifford/Tとして無条件に認定しなかった。判定は `PARTIAL_CLASSIFICATION_COMPLETE_ARBITRARY_ADVANTAGE_UNQUALIFIED`。T総費用の優越は認定していない。

V3範囲を完了し、[mandatory STOP](../../../artifacts/pr_reference_rotation_v3/2026-11-07/STOP.json)。V5、新compile、新しいClifford+T合成、precision最適化へ進んでいない。これは著者の技術検査で、独立科学レビュー・第三者再実行・CIの結果ではない。

## 固定条件・分類規約

Linear H4、1.00Å、STO-3G、DF12、元H ground供給・準備費除外、Partial S2/LD3/h=.02/K2、ordinary diag(I,V)、canonical finite-bit law、保存q/rの18round。精度1.5936e-4 Ha、失敗率.05、位相cap.4rad。compiler A/Qiskit1.3.0、RZ/SX/X/CX、全結合、B′4-CX中央回転、reference F/G・vacuum補正・2-register・8-CX CSWAPを維持した。保存shot schedule8598/8598/8606、native誤差・alpha・aliasも不変更。

[SCOPE](SCOPE.md)、[分類規約・例外・一次SOURCE](CLASSIFICATION_RULES.md)を参照。主記録 `literal_native` と、解析的SOURCE由来の参考記録 `analytic_provenance` を分けた。

固定gate定義からkπ/4という**解析的由来**を追えても、保存native角が非零binary64近似なら、そのnative演算そのものを厳密なClifford/Tとは認定しない。参考記録には理想的な由来と丸め差を残す。数値計算で得たGaussian phase/betaにもsymbolic branchが保存されていないため、近さや同じfloat値だけから型を認定しない。未確定を任意角へ埋めず、両者を区別した。

## 3方式の全18round比較

以下は期待総個数。小数表示は概数で、exact有理値・外向き端点・round別値は[CLASSIFIED_18ROUND.json](../../../artifacts/pr_reference_rotation_v3/2026-11-07/CLASSIFIED_18ROUND.json)に保存する。未確定がある行の「認定済み」は、その型の全個数ではない。

| 指標 | 登録14-CX baseline | B′ | reference | 資格 |
|---|---:|---:|---:|---|
| 総RZ | 3.735644724×10¹² | 3.571960961×10¹² | [3.577340247, 3.577340249]×10¹² | 既存有理値／上下界と一致 |
| 認定済みClifford型RZ、実native基準 | 4.054363231×10¹¹ | 4.054363231×10¹¹ | 4.072290256×10¹¹ | 厳密zero RZとして認定した分 |
| 認定済みT型RZ、実native基準 | 0 | 0 | 0 | 非零pi近似角は未確定。T費用0の意味ではない |
| 確定した任意角RZ | 3.194040386×10¹¹ | 3.194040386×10¹¹ | 3.104025932×10¹¹ | 既存lawの有理期待個数 |
| 未確定角度、実native基準 | 3.010804362×10¹² | 2.847120599×10¹² | [2.859708628, 2.859708631]×10¹² | 未確定数の期待値／上下界 |
| 総CX | [5.931622390, 5.931622391]×10¹¹ | [5.038801864, 5.187605286]×10¹¹ | ≤4.802896135×10¹¹ | 既存保証値を継承 |
| 十分shot数 | 8,598 | 8,598 | 8,606 | 保存schedule |
| qubit数 | 9 | 9 | 17 | 固定構成 |

T型行の0は、厳密と認定できたnative型の個数である。未確定角度の中身や、任意角の合成Tを0扱いしていない。総T・depth・sizeは引き続き未取得／未資格。

確定任意角部分の期待差は **9,001,445,403.795599以上、2.8182002469784466%以上の減少**。これは未確定を除いた部分の比較であり、全任意角回転の改善率として引用しない。

実native基準で未確定を含めた総任意角数は、B′が `[3.194040386×10¹¹, 3.166524637×10¹²]`、referenceが `[3.104025932×10¹¹, 3.170111224×10¹²]` に入り得る。安全な差の区間はゼロを跨ぐ。したがって、全任意角の優越・改善消失のどちらも断定しない。未確定のSOURCE型を推測して区間を縮めていない。

### 解析的SOURCE由来の参考内訳

| 指標 | 登録baseline | B′ | reference |
|---|---:|---:|---:|
| Clifford由来slot | 2.756868849×10¹² | 2.697347481×10¹² | [2.709275989, 2.709275992]×10¹² |
| T型由来slot | 1.041623946×10¹¹ | 0 | **963,872（有理値は整数）** |
| 確定任意角slot | 3.194040386×10¹¹ | 3.194040386×10¹¹ | 3.104025932×10¹¹ |
| 由来も未確定のslot | 5.552094413×10¹¹ | 5.552094413×10¹¹ | 5.576607005×10¹¹ |

これは実native角を丸め直した回路の価格ではない。binary64との残差を無視してexact Clifford/Tを無料実装した結果や、回路の最小T数へ読み替えない。

登録RTEの24-RZ controlled-ZZは、解析的Clifford由来8・T型由来14・物理的な半角2。B′では半角2だけが残る。このため登録baseline→B′の22-RZ減は、この分類ではClifford/T由来slotの削除であり、確定任意角部分の数は変わらなかった。reference中央回転は1 RZで、確定任意角部分が減る経路になる。

referenceのCSWAPは1個あたりClifford由来4・T型由来7の11 RZ。16 CSWAP/shotと8,606 shotsでT型由来slot963,872になる。真空位相補正は別の1 RZ/shotで、全supportの分類は未確定として残す。このslot数も実T総数ではない。

## 構造会計とangle key

D block非融合寄与と境界融合差分、RTE中央回転、basis frame/inverse、product、event sign、scalar/vacuum補正、CSWAP、X/Y wrapperを分けた。融合差分はsignedな会計補正で、負のgateを実装する意味ではない。

既存のfinite-bit floor-CDFのexact有理確率、basis-run期待値、D融合式を、各分類とsigned angle keyへ適用した。reference product-Zの消去は既存の全残し上界／全捨て下界を維持した。角度key・親role・round・multiplicityは[ANGLE_KEY_MULTIPLICITIES.json.xz](../../../artifacts/pr_reference_rotation_v3/2026-11-07/ANGLE_KEY_MULTIPLICITIES.json.xz)に保存した。keyをまとめることをgate削減やword実行の共有とはみなしていない。

全round・両軸・3方式で、分類個数の和が既存RZ有理価格／上下界にexactに一致した。既存CX保証（B′比で少なくとも4.681782%減）とRZ増加保証（少なくとも0.150597%増）は変更しない。新しい成功保証・shot最適化は行っていない。

## 保存cohortとの照合・テスト

- 保存local catalogueの2,165 recordについて、解析したRZのsigned binary角ヒストグラムと実保存loweringが一致した。全contextのD recordについても一致を確認した。[LOCAL_INVENTORY.json.xz](../../../artifacts/pr_reference_rotation_v3/2026-11-07/LOCAL_INVENTORY.json.xz)。
- 64保存path×2軸×3方式の384比較で、構造分解のRZ数が保存whole価格と一致した。referenceの差は、SOURCEでのみ消去され得るproduct-Zに限定して既存の範囲内で記録した。[COHORT_CHECKS.json](../../../artifacts/pr_reference_rotation_v3/2026-11-07/COHORT_CHECKS.json)。
- 保存native QPY14回路を読み、RZ角度ヒストグラム全体と構造分解が一致した。QPYがない他のpathに角度観測があるとは扱わない。今回の3方式に入らない旧weak `existing_direct` QPYは対象から除いた。[QPY_CHECKS.json](../../../artifacts/pr_reference_rotation_v3/2026-11-07/QPY_CHECKS.json)。
- 人工15検査で、符号、zero、pi由来、由来欠落、隣接floatへの摂動、未束縛symbol、CRZZ/CSWAP由来を確認した。近い角をTへsnapする検査にはしていない。[SYNTHETIC_TESTS.json](../../../artifacts/pr_reference_rotation_v3/2026-11-07/SYNTHETIC_TESTS.json)。人工PASSを科学性能の証拠にしない。

技術的修正として、Gaussian betaにsigned-zeroの分岐に対応する−3π/2が存在することを確認した。これを近いCliffordへ昇格させず、数値betaは未確定に保った。QPY readerも旧weak armを除外するよう修正した。最終結果は修正後の照合に基づき、旧価格の再取得・新compileは行っていない。

## 再利用と実行記録

新SOURCEは `src/trottertracks/pr_reference_rotation_v3.py`。既存のB′保存path復元、`distributions`、D primitive、区間helper、SOURCE/context・RZ会計を再利用した。旧moduleのrunner・価格・合成取得関数は呼んでいない。

`native_df_h6_cost_adapter.rotation_classes` は許容誤差によるdiagnosticなので、今回のexact分類に使用しなかった。既存C2-Bのsigned/positive binary angle keyとstrict target規約は確認した。保存strict合成7,603 record（963 positive angle key、PASS）を読取確認したが、新A用precision・誤差配分を選ばず、T価格・word・guardを流用していない。[再利用確認](../../../artifacts/pr_reference_rotation_v3/2026-11-07/SYNTHESIS_REUSE_AUDIT.json)。

新SCF、DF fit、ground solve、sampling、回路build、native compile、Clifford+T合成、strict guard再評価は0。実affinity64、MemAvailable約463GBを観測し、独立14 QPY読取を14 processで処理した。既存venvを読取使用し、共有環境・他者process/fileを変更していない。

[SOURCE.json](../../../artifacts/pr_reference_rotation_v3/2026-11-07/SOURCE.json)、[成果物README](../../../artifacts/pr_reference_rotation_v3/2026-11-07/README.md)に固定入力、SOURCE、実行順、環境、検証資産、保存先を記録した。旧SOURCE・raw・結果・STOPは保全する。

## 未取得のFT資源・次の判断

今回取得したのは回転の部分分類・期待個数と安全な区間であり、FTの総T、最小T、合成depth、space-time、routing/noise/wall timeではない。解析的C/T由来slotにはbinary encoding差があり、未確定の数値phase/betaとvacuum補正も残る。これを既存strict合成のPASSやT価格で埋めていない。

**次担当はGPT／ユーザー**：SOURCEの型と実nativeの厳密性のどちらをFT資源の主指標にするか、未確定を含む結果をどう評価するかを判断する。既存CX改善、RZ増加、追加8 qubitはそのまま考慮する。V5を別途承認する場合は、今回の回転部分比較を全資源優越とせず、固定B′/referenceのcritical pathを独立に評価する必要がある。

**V3終了でmandatory STOP。CodexはV5・新しい角度分解・precision・T合成へ自動進行しない。**
