# Track D B6b＋B/C統合準備：最終報告

2026-11-08 JST。**承認範囲の技術準備を完了。公開照合後mandatory STOP。**

Referenceの理想X/Y測定同値性を任意system入力へ導出し、既存SOURCEのDF構成条件と対応付けた。B4-2の保存NとC側価格を接続する薄いadapter、全13-round coverage、人工会計テストを整備した。**実H4の新sampling・回路生成・compile・価格再解析・N算定・全RPE総資源の数値評価は行っていない。**

## 成立したことと条件

V=AB、actual A†、真空reference v、既存CSWAP構成とancilla P(+χ)に対し、complex coherence演算子は

\[
T_{\rm Ref}=e^{i\chi}\overline{\langle v|V|v\rangle}\,V.
\]

従ってVv=e^{iχ}vなら、任意の純粋/混合system入力でdirect controlled-VとX/Y誘導observable・条件付き±1分布が一致する。強いsymmetric-directionalは位相付きW0†W1=Vという同じcoherenceを実現することを要する。単なるglobal-phase同値ではない。

既存DFの各halfが真空rayを保存する条件は、信号同値性に加えて最終referenceの真空復帰を保証する。whole Vだけがrayを保存する弱い条件では最終referenceが非真空の場合もあり、既存DF checkerのzero-leakage条件を緩めていない。任意A/Bで無条件に同値とは主張しない。[導出・SOURCE対応・lawとN条件](MEASUREMENT_EQUIVALENCE.md)。

この一般化は、SOURCEが正しいsigned/ordered/scalar/identity/χ/actual adjoint・合法融合を実現する**ideal PR blueprint**を対象とする。有限H4 pathのPASSを全support証明へ外挿せず、floating実装・全round compiler誤差・finite-bit lawの資格は別に残す。

同じwhole-path lawと同じfresh測定条件なら両方式の理想測定分布が一致する。さらにB4-2のideal finite-RTE IID、元H厳密ground、attenuation・誤差予算・prior/decoder/targetが同じなら、保存されたround/axis別Nを共通利用できる。保存合計14,738は既存条件付き十分値の引用で、新算定でもactual成功保証でもない。共通finite-bit Qでの方式間同値性だけではB4-2のideal Pへの距離やconcentrationを保証しない。[条件判定record](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/common_N_conditions.json)。旧handoffの共通N未資格fieldは改変せず、B6bの条件付きideal判定を別recordにした。

## 検証結果

|対象|実施・結果|意味する範囲|
|---|---|---|
|人工・回帰・会計|24件PASS（新12、既存C5 strong8、C3 4）|局所技術検証。独立レビュー/immutable CIではない|
|人工whole-path|4 case、正負回転・Gaussian・非自明χ・scalar・cut・actual inverse|3件DF適格、1件whole-V-rayだけの代数拡張例|
|人工P0|16 full wrapper、全postcheck PASS|同P0・位相保持、全system入力、閾値緩和なし|
|negative controls|χ欠落/逆符号、forward A、逆順AB、V-ray非保存の5件を検出|不適格条件・反例を保全|
|B/C接続|固定43 Git input binding、13 coverage行、52人工価格recordで会計/共分散/欠損guard|実H4 N×価格は未評価|

最大compile後component差9.473989258092931e−15、X/Y observable最大Frobenius差4.1595923209015305e−15。精度はcomponent1e−10、observable2e−10、rtol0、phase fitなし。詳細は[人工検証報告](VALIDATION.md)、[raw theorem check](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/raw_theorem_checks.json)、[postchecks](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/postchecks)。人工direct controlは数理比較器で、新しい弱いH4価格baselineではない。

## B/C接続の現在地

固定条件はLinear H4/隣接1.00 Å/STO-3G/Nα=Nβ=2、元H、number-CDF8/元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1。元H厳密groundを仮定し準備費を除外。B4-2の0.00159 Ha・failure .01・βtotal π/6・βstat π/12・αaxis .01/26・13-roundは変更していない。

q1にはevent16 cutの保存32本、q2/4/8/16/32には各16/16/32/16/8 whole-pathの直接compile有限標本がある。X/Yはpairedでnを倍にしない。q1はcomplete-step-cutモデルへ無断移送しない。q64〜4096は直接compile証拠がなく、C7-4のq-only探索的proxy等を母平均保証へ昇格できない。C7-4公開固定版64072ba8は別版metadataとして読取り、worktreeへの介入・再fitはない。

[B/C仕様・13-round表](BC_INTEGRATION_SPEC.md)／[coverage CSV](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/coverage13.csv)／[価格metadata](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/price_catalog_metadata.json)。

未来の総和hookは既存aggregate_totalsへ委譲し、全52期待価格record、identity・q・scope・compiler・SOURCE・標本単位と共通N条件を確認する。finite sampleを無条件の期待価格と扱わず、明示したconditional modelと仮定は受入可能。欠損はnull/拒否とし0補完しない。6資源を分離、peak qubits非加算、attenuation二重会計なし、共有calibration/paired共分散を独立扱いしない。**実H4の13-round数値totalはnull。**

actualのfinite-bit/joint-law、native/P0/χ/adjoint/leakage、全support・長q raw誤差と旧整数Nの適否は未資格。[actual不足と二重計上防止](ACTUAL_GAPS.md)。これらの完成は、明示した仮定による理想条件付き資源点推定の絶対的な前提ではない。

## SOURCE・原本・公開保全

実行SOURCEは`13114b2bd9db7dcf1ee7f027ab202f72786e2552`、treeは`73ba2523b728d257eb616553bd8308bdeb251d1f`。基点はC7-3最終`acea43987edae0d5eb0c186a89aa8cd7139fbd7b`。最終公開commitはこのSOURCEの子で、報告書・metadata・索引だけを追加する。実行前[PLAN](PLAN.md)と実行SOURCEの6固定file、7再利用file、元Cの4 exportを前後照合し、変更しない。[SOURCE](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/SOURCE.json)／[固定input hash索引](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/input_bindings.json)。

人工raw/compiled QPY32＋target NPZ4の36原本はlocal保全し、公開は[hash索引](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/payload_index.json)だけ。実H4 raw payload・内部環境/絶対保存pathは公開しない。既存SOURCE・結果・grant・ledger・marker・STOPを改変せず、共有設定・他job・他worktreeに変更しない。

## 次の担当とSTOP

**ユーザー/GPT**：まず[次段階案](NEXT_EXECUTION_DRAFT.md)のclaim水準、q1の扱い、q≥64価格versionを判断する。最小案は既存Nと採択proxyによる理想条件付き全13-round点推定で、新しい長q sampling/compileやactual保証完成は必須ではない。そのH4数値会計自体は今回の対象外で、新しい実行許可が必要。価格根拠を先に補うなら別scopeのq64等、actual保証が必要ならlaw/raw誤差資格を別途選ぶ。

**Codex**：今回の公開SHA/tree/bytesを照合して停止する。後続科学計算・数値再解析・N算定・詳細レビューへ自動進行しない。[完了record](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/result.json)／[STOP](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/STOP.json)。
