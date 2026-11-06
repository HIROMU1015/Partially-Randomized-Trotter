# 新A V5：全18round native DAG depthの構造的資格

2026-11-07 JST、SERVER 2。基準 `224440f740051ed5794e8db87a57908fc02dd02f`、専用branch `codex/pr-reference-depth-v5-20261107`。

**固定モデル下で、referenceの期待総native DAG depth上界が4-CX強baseline B′の下界を下回った。削減率の保証下限は40.995952643300015%。** 期待depthそのものの厳密値は求めていない。判定は `CONDITIONAL_ALL18_NATIVE_DEPTH_REDUCTION_VS_BPRIME`。

V5-A〜Eを完了し、[mandatory STOP](../../../artifacts/pr_reference_depth_v5/2026-11-07/STOP.json)。科学対象の新build/compile、simulation/sampling、価格再取得、Clifford+T合成は0。新しい回路最適化や角度分類解消へ進んでいない。以下は著者による構造論証・保存データ照合であり、独立科学レビュー、第三者再現、形式検証、CIの結果ではない。

## 固定条件と主指標

Linear H4、1.00Å、STO-3G、DF12、元H厳密groundを各shotで供給・準備費除外。Partial S2/LD3/h=.02/finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、保存18roundのq/r。energy精度1.5936e-4 Ha、失敗率.05、phase cap .4rad、Qiskit1.3.0/compiler A/RZ-SX-X-CX/全結合を維持した。[SCOPE](SCOPE.md)参照。

B′の4-CX controlled-RZZ、referenceのF/G・actual adjoint・vacuum P・8-CX CSWAP、D融合・basis共有・符号・scalar・X/Y wrapperを変更していない。保存成功資格に対応するshots8598/8606と9/17 qubitを使用し、誤差disk、alpha、alias、PF、law、precisionも変更していない。

主指標は保存価格と同じ**reset/測定を含む単位gate native DAG depthのshot加重和**。回路間・shot間の実機並列化、gate duration、routing、noise、通信、初期化時間、wall timeを意味しない。

## 全18roundの比較

表の数値は読みやすい概数／安全側に桁を落とした端点。exact有理境界・外向き浮動端点は[DEPTH_18ROUND.json](../../../artifacts/pr_reference_depth_v5/2026-11-07/DEPTH_18ROUND.json)に保存した。

| 指標 | B′ | Reference | 資格 |
|---|---:|---:|---|
| 期待総CX | 約[5.03880, 5.18761]×10¹¹ | ≤約4.80290×10¹¹ | 既存保証を継承 |
| 期待総RZ | 約3.571960961×10¹² | 約[3.577340247, 3.577340249]×10¹² | 既存保証を継承 |
| 期待総native DAG depth下界 | ≥9.19338940794×10¹¹ | ≥4.17942775235×10¹¹ | 今回の構造下界 |
| 期待総native DAG depth上界 | ≤1.14643841539×10¹² | ≤5.4244718400×10¹¹ | 今回の構造上界 |
| 十分shot数 | 8,598 | 8,606 | 保存schedule |
| qubit数 | 9 | 17 | 固定、referenceは追加8 |
| T・T-depth | 未取得 | 未取得 | 対象外 |

reference上界とB′下界の差は **376,891,756,799.6996以上**、相対削減率は **40.995952643300015%以上**。境界同士の比較で保証しており、cohort平均や点推定を代入していない。全round・X/Yの1-shot価格、有理確率、shots、lane penaltyはJSONと[ROUND_DEPTHS.tsv](../../../artifacts/pr_reference_depth_v5/2026-11-07/ROUND_DEPTHS.tsv)に保存した。round寄与は保存したshotsと有理価格の積で表され、総和を上記会計に使用している。

既存のCX削減≥4.681782%、RZ増加≥0.150597%は不変更。V3の確定任意角部分≥2.818200%減もその部分に限定したまま、未確定を含む全任意角優越は未資格。今回のdepth結果から総T・全資源優越を推論しない。

## frontierと構造境界

[DERIVATION.md](DERIVATION.md)に式と適用範囲を記録した。既存 `pr_reference_transfer_qualification.frontier_depth` はwireのmax+1更新で保存QPYの実depthを再現する。新実装は、保存local loweringを整数max-plus行列へ写し、非一様な入力frontierへの作用を保持する。新しいquantum builder/compilerは実装していない。

B′下界は、SOURCEと最終CX/X相殺から保護されたD/frame内の依存鎖を残す。システム8 wireすべての入力深さがt以上なら、出力の最小深さがt+δ以上となる局所δを用いる。ancillaが同じt以上という仮定は置かない。中央parity CX・rotation・product・phase・wrapperの寄与を捨てた保守的下界である。露出parity CXの相殺数がreferenceと等しいとは仮定しない。

reference上界では2 registerを別laneとして扱い、F†とGの並列性を保つ。各laneの融合後Dのspan上界と、RTE occurrenceごとのrandomな上界cost Wを使う。Wの期待値は保存finite-bitのexact basis-run式とrotation/product価格から計算する。2 laneのmaxには、全support範囲W∈[0,M]からVar(W)≤M²/4を用い、`E max(S_F,S_G)≤n E W + M sqrt(n/8)` を適用する。有限cohortの分散や経験的余裕を使っていない。sqrtは128-bit二進格子への有理ceilingで評価した。q=1は固定の非対称F/G分割を別扱いする。

source-levelで消え得るproduct-Z、最終passのCX/X相殺を上界では残す。D境界融合は実際のblock多重度で評価し、q/rへの経験的な比例外挿を行わない。basis共有は既存のcanonical floor-CDF有理確率とrun期待式を再利用する。平均event数で作った回路のdepthを期待depthとみなしていない。

保存8-CX CSWAPを8個接続した一つのcascadeのfrontier spanは72。8×局所span16=128という直列和を実depthとは呼ばず、wire依存で得た72を2 cascadeの上界会計へ用いた。16 CSWAP/shot、128 CX/shot、vacuum P、X/Y wrapper、測定を含む。

referenceの保存SOURCEはnative compile後にfresh registerの8 resetを明示的に付けている。resetは互いに並列のdepth 1、ancilla上の初期Hのnative depth 3と並列であり、初期frontier上界はmax(1,3)=3。8 resetを捨てたり直列8層として足したりしていない。unitary-only depthとは区別する。

## 保存資産との照合

- 保存local catalogueの**5,284 context-record**でmax-plus行列が対応contextと一致した。[LOCAL_FRONTIERS.json.xz](../../../artifacts/pr_reference_depth_v5/2026-11-07/LOCAL_FRONTIERS.json.xz)。
- D/frameの**268境界**で、最終pass前の露出CX/X prefix/suffixが空であることを確認した。[FENCE_CHECKS.json](../../../artifacts/pr_reference_depth_v5/2026-11-07/FENCE_CHECKS.json)。SOURCE初段passの成立条件は旧B′補遺と今回の導出に記載した。
- 保存64 path×X/Y×2方式の**256条件**で、実whole-native価格のdepthがpathwise構造上下界に含まれた。これは区間包含検査であり、全256条件のnative topologyを完全再構成した検査とは区別する。[COHORT_DEPTH_CHECKS.json](../../../artifacts/pr_reference_depth_v5/2026-11-07/COHORT_DEPTH_CHECKS.json)。
- 保存**8 QPY**（reference q1/4/8・各rare/nonrare、B′ q4・各rare/nonrare、すべてX）で、再利用frontierの深さがQiskit depthと保存whole価格にexact整数一致し、構造上下界にも入った。QPYがない他pathへ観測を拡張しない。[QPY_DEPTH_CHECKS.json](../../../artifacts/pr_reference_depth_v5/2026-11-07/QPY_DEPTH_CHECKS.json)。
- **人工13検査**で、非一様frontier、並列gate、barrier、reset/測定、gate削除の単調性、整数有理sqrt上界を確認した。6本の人工回路は技術検査用で、科学対象の新回路ではない。[SYNTHETIC_TESTS.json](../../../artifacts/pr_reference_depth_v5/2026-11-07/SYNTHETIC_TESTS.json)。

最初のQPY検査では「resetなし」という新コードのmetadata仮定が誤っており、命令数assertが停止した。保存SOURCEの8 resetを確認して修正し、対応する人工並列検査を加えた後の照合・会計だけを採用した。depth値の不一致や旧結果の変更ではない。新しい科学条件・compileで補っていない。

## SOURCE再利用と実行範囲

新SOURCEは `src/trottertracks/pr_reference_depth_v5.py`。既存DF/repeated-S2/RTE builder、B′ rewrite、reference F/G、compiler A、correctness・誤差・shot qualificationを変更していない。

再利用した主な関数・資産は、`frontier_depth`、`pr_reference_bprime.saved_samples`、`pr_reference_bprime_accounting.distributions`、`track_c_h4.common/save_json`、局所catalogue・相殺fence、whole価格、QPY、`TOTALS_18ROUND`/`STAGE4_RPE`。既存 `df_partial_s2_repeated.py`、`df_rte_qiskit.py`、`rpe_resource_accounting.py` のSOURCEと費用接続を確認した。既存のCX/RZ会計は非線形なdepthの期待maxを扱わないため、max-plus行列と2 laneの期待上界だけを専用SOURCEへ追加した。旧runnerや価格pipelineは再実行していない。

環境は既存venvを読取使用し、affinity64・MemAvailable約419GiBを観測した。独立8 QPYは8 worker、局所行列はcache、期待値は有理計算で処理し、巨大な長round回路を生成していない。共有設定・package・他者process/fileへの変更はない。[SOURCE・入力・環境・実行順](../../../artifacts/pr_reference_depth_v5/2026-11-07/README.md)を参照。

## 未資格事項と次担当

保証は固定保存H4・compiler A context・canonical独立finite-bit law・既存の算術／信号モデル内に限る。元H表現の無条件保証、別PF/law/compiler、最適なdepth、正確な期待depth、実機時間、space-time、routing/noise、総T/T-depth、V3未分類の解消は示していない。追加8 qubitはdepthやCXへ重み付け合算しない。

**次担当はGPT／ユーザー**：構造境界の科学的妥当性、depth/CX改善とRZ増加・追加8 qubitのtradeoff、残存FT資源を踏まえて研究上の意味と後続作業の要否を判断する。Codexは追加最適化・T合成・別depth候補・旧Track再開へ進まない。

**V5終了でmandatory STOP。旧SOURCE・raw・結果・STOPを保全した。**
