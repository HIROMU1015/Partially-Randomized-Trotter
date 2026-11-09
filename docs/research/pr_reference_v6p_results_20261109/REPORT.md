# 新A V6-P：保存上界による条件付きRPE総RZ比較

2026-11-09、SERVER 2。**判定：CONDITIONAL_B1_TOTAL_RZ_ADVANTAGE。** 固定SOURCEによる科学campaignを**1回**実行し、旧N・区間算術・費用会計の検算を完了した。B1を実現・資格化できるという仮定の下で、Reference m17は630→626 shots/axisとなり、全18round総native RZの上界が保存B′下界を下回る。

これは**実装前の条件付き数値screen**である。B1 provider、相対位相correctness、新回路、実機成功、総Clifford+T T-count削減を認定した結果ではない。

## 1. 固定条件・SOURCE・実行

Linear H4、1.00 Å、STO-3G、DF12、元H厳密ground供給・状態準備費除外、Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit 1.3.0／compiler A／rz,sx,x,cx。B′4-CX controlled-RZZ、ReferenceのF/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、D融合・basis共有・scalar/event相対位相・X/Y wrapperを保持した。

対象m17はq=131072、r=32、bits=88。精度1.5936e-4 Ha、failure=.05、phase cap=.4 rad、αaxis=1/720、他17roundの条件・N・単価は不変。E0、gap、groundの古典記述を補正へ追加使用していない。

| 固定資料 | 内容 |
|---|---|
| [実行SOURCE a2c54558](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a2c54558e1b4cc052aeb835a9e6d3b2796767e98/scripts/diagnose_pr_reference_v6p.py)／[準備REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a2c54558e1b4cc052aeb835a9e6d3b2796767e98/docs/research/pr_reference_v6p_preparation_20261109/REPORT.md) | 公開したadapterをbyte不変更で使用。準備時の人工11テストは再実行していない |
| [e5d6e388計画 §2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e5d6e388ee2543b47ebcc1dfd9e19862d08ac327/docs/research/pr_reference_v6_static_20261108/NEXT_VALIDATION_PLAN.md) | 単一B1式、m17 N、価格不変を仮定した18round RZ再会計 |
| [科学入力正本 f2638a78](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/f2638a7845685a119619382b850441f5d8d33713) | 6保存入力と3再利用SOURCE。blob／SHA256は下記SOURCEに保存 |
| [実行SOURCE・入力記録](../../../artifacts/pr_reference_v6p/2026-11-09/P_run01/SOURCE.json) | 実行commit、adapter hash、全固定入力のprovenance、Python情報 |
| [完全RESULTS](../../../artifacts/pr_reference_v6p/2026-11-09/P_run01/RESULTS.json) | 三条件、phase／disk／ρ／β／ceil区間、36セル・9role・exact Fraction総計 |
| [CHECKS](../../../artifacts/pr_reference_v6p/2026-11-09/P_run01/CHECKS.json)／[検算SOURCE](../../../scripts/check_pr_reference_v6p.py)／[STOP](../../../artifacts/pr_reference_v6p/2026-11-09/P_run01/STOP.json) | campaign再呼出しなしで結果の検算・外向き表示を実施。元のraw出力とSTOPは不変更 |

実行は専用worktree／branch、fresh `artifacts/pr_reference_v6p/2026-11-09/P_run01`。CLI呼出し1回、exit 0、tool報告wall time 0.195342565秒。既存project venvのPython／mpmath 1.3.0、192-bit directed intervalと既存Fraction会計を利用した。D0 runner、phase_bound、qualify、common、量子builderは呼び出していない。

共有環境では起動前にMemAvailable 515750280 kB、CPU／memory PSI avg10=0.00を観測した。小さい保存値算術なので重複workerを起動せず、CPU/RAMへの独自上限、共有設定変更、他利用者／他worktreeへの変更は行っていない。

## 2. 位相上界とRPE十分shot数

B1の仮定は、**同じbinary literalをexactに集約し、F/G合計を最後に一度だけbinary64最近接丸め**すること。片側ずつの丸めや別precisionを試していない。保存Mとnormalization上界からδ_B1=B_norm+2·2⁻⁵³Mを外向き評価した。

| 位相項 | 保存／条件付き上界 |
|---|---:|
| 保存exact-prefix M | 16375.943495244212 |
| 旧naive accumulation | 0.00045372585145842106 |
| 保存normalization B_norm（保持） | 5.172131038537467e-08 |
| 一回丸め項2uM（外向き） | 3.6361899036761377e-12 |
| 条件付きB1位相上界 | **5.172494657527835e-08** |

旧phase_arithmetic.phase_error_upperは0.00045377757276880644、旧raw会計でchargedされた安全側phaseは0.0004537775727688065。旧検算には後者を使用した。normalizationを削除しておらず、SOURCE／event／finite-bit proposalとnative joint由来のraw diskも保持した。

| m17条件 | phase raw項 | total raw disk上界 | ρ下界 | β下界 | shots/axis |
|---|---:|---:|---:|---:|---:|
| 保存B′（不変） | vacuum追加項なし | 3.6648120444602677e-06 | 0.5726873297263416 | 0.38598699711850315 | 626 |
| 旧Reference再構成 | 0.0004537775727688065 | 0.00045743900771371104 | 0.5722335555306725 | 0.38519464269240195 | **630** |
| 条件付きReference-B1 | 5.172494657527835e-08 | 3.713159891479787e-06 | 0.5726872813784947 | 0.3859869126962799 | **626** |

Referenceのphase以外のdisk上界は3.661434944904509e-06。B′とReferenceでnative joint等の内訳が異なるため、B′のdisk／NをReferenceへ代入した結果ではない。改善後もReferenceのρ／βはB′と同一ではないが、整数ceilは同じ626となった。X/Yに共通の保存worst-case保証を使用した。

ρとβの外向き端点を保存してから、192-bit区間で `ceil(4 log(1440)/(ρ² sin²β))` を計算した。m0〜m16のNは再計算していない。旧内部intervalをbitwise再現したとの主張はせず、保存安全側成分からの再構成で旧630とB′626が再現することを検算した。

## 3. 条件付き全18round総native RZ

**価格不変の条件**：F/G・SOURCE gate列は同じで、最終vacuum Pだけが新角度となり、固定compilerの同じcontextで1 RZへlowerされること。今回新compileや価格取得を行わず、保存単価区間を用いた。

次の小数はexact Fraction区間を既存D0表示関数で0.001 RZ単位へ外向きに表示したもので、実現可能な端点policyや統計信頼区間ではない。

| 総RZ／差 | 下界 | 上界 |
|---|---:|---:|
| 保存B′（不変） | 3,571,960,960,634.802 | 3,571,960,960,634.803 |
| 保存Reference | 3,577,340,246,876.263 | 3,577,340,249,201.294 |
| 条件付きReference-B1 | **3,561,600,143,038.349** | **3,561,600,145,354.476** |
| 条件付きReference−B′ | **−10,360,817,596.453** | **−10,360,815,280.326** |
| 条件付きReference−保存Reference | −15,740,103,846.819 | −15,740,103,837.914 |

従って、指定した条件が実現できれば、保存強baseline B′に対して全18round総native RZを少なくとも **10,360,815,280.326 RZ** 減らす区間が得られた。これは旧Referenceの約＋53.79億RZという不利をそのまま改善結果と読み替えたものではない。m17の余分なshot費用を除くことで、D0に保存されたB′-shot基準の単価差寄与（約−103.61億RZ）が総差として残ることをexact恒等式で確認した。

N差分はm17の両axisだけに掛け、RTE basis/frame/inverseを含む全9roleを再会計した。D融合のsigned補正も保持。負のN差分と旧総区間を独立に加算せず、同じ単価端点を新旧費用へ対応させて総計を計算した。CX／depth／size／qubit／T-countの新しい総計は作っていない。

## 4. 検算と結果の限界

[post-run checker](../../../scripts/check_pr_reference_v6p.py)はcampaign、candidate_bound、bound_and_shots、shot_ceiling、recostを再呼出しせず、保存結果を次のように確認した。

- 固定adapter・入力hash、target／scheduleと科学条件の整合。
- exact FractionによるB_norm+2uMの包含と、normalization保持。
- shared/native raw成分からのdisk、原ROUNDSからのnominal radius、ρ／βの外向き端点。
- 直接4-log式と元のepsilon／√2式の区間整合。上端ceilだけでなく、区間下端もN−1を上回ることを確認。旧630、旧626、新626の整数判定に曖昧さなし。
- 36セルの費用、9role・signed融合の和、m0〜m16不変、B′全18round不変。
- 新総差とD0の保存単価差、除かれた費用とD0の保存追加shot項のexact一致。

**全検算PASS。** RESULT bytes=70824、SHA256=`854073ac7cae86b5b3fb103d62ec21946a5665537aabb740d592462bd1226a26`。post-run検算は著者自身の検算であり、独立科学レビューではない。科学campaignの追加実行／retryは0回。

B1は未実装で、最終RNのruntime適合、実binary入力の取出し、F/G合計の一回丸め、relative Pのcorrectness、SOURCE inverse／copy／compose経路、同じnative価格を保てる実装の接続は未資格。旧算術モデル・全support M・normalization保証を条件として再利用したscreenであり、元H表現やruntime全体の無条件保証を新設したものでもない。m17 CT、domain／emitter／word policy、総Tの既存課題・負結果・STOPは解消していない。

## 5. 科学的判断材料とmandatory STOP

1. **十分shot削減の余地あり**：条件付きB1上界でm17は630→626となった。
2. **総RZ優越の余地あり**：価格不変の条件下では、全18roundのReference上界が強B′下界より小さい。
3. **V6-Qを検討する数値的動機は得られた**：ただし実装・runtime・correctness・価格接続を別途資格化する必要があり、今回それを採択・実行していない。

旧SOURCE・raw・結果・grant・ledger・marker・STOP・準備文書は不変更。専用branchに今回の結果・SOURCE記録・検算・STOPと本REPORTを追加し、commit/push・remote SHA／内容一致を確認して**mandatory STOP**する。

次担当はGPT／ユーザー：この条件付き利益を踏まえ、既存V6-Q案の実装・資格検証へ進むか、その固定scopeと実行承認を判断する。CodexはV6-Q、新phase provider、回路生成、合成、総T評価、他候補の探索へ進まない。
