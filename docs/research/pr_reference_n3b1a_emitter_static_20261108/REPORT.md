# 新A N3-B1a：emitter厳密性・誤差保証の静的調査

2026-11-08 JST、SERVER 2。固定参照 **`9c60c43135666a9ab5497a43a4b1695ab371650f`**、専用branch `codex/pr-reference-n3b1a-emitter-static-20261108`。

**Stage A〜Eの静的調査を完了し、mandatory STOP。** 既存lower_nativeの順序／actual adjointを追跡し、固定binary64定数のW加算に8状態閉包が成立する条件を導いた。これに基づき、宣言した有限具象化経路ではN3-B0残余を満たす**記号的な十分上界**を構成できる。

判定は**EMITTER_CERTIFIABLE_FROM_SOURCE（明示したC_phiと経路への適合が条件）**。新しい43 word、実provider、全serialization経路を数値確認したPASSではない。実dictionary／実装適合・全support CT RPE成功・T価格は引き続きUNQUALIFIED。既存emitterのW処理が必ず変更を要するという結論にはしない。

## 固定条件と保存値

Linear H4・1.00 Å・STO-3G・DF12、元H厳密ground供給・状態準備費除外。Partial S2・LD3・h=.02・finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、m7/q128/r1/73 bits、最終native RZ strict precision10^-11。

基本辞書b=36、指数-36…6、43 key、基本Frobenius要求10^-13を保持する。Qiskit1.3.0/compiler A／rz,sx,x,cx／全結合、B′4-CX controlled-RZZ、Reference F/G・actual inverse・vacuum P・2 register・8-CX CSWAP、D融合・basis共有・scalar/event・X/Yも不変更。energy1.5936e-4 Ha、failure.05、phase.4、alpha1/720、199 shot/axisを維持する。

[N3-B0 FIXED_PARAMETERS](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/FIXED_PARAMETERS.json)、[ERROR_BUDGET](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/ERROR_BUDGET.json)、[COVERAGE](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/BASIC_KEY_COVERAGE.json)を読み、再選択／再計算していない。保存coverage0/43、量子化上端約5.144879e-12、emitter残余R_emit^lo約5.55121031385e-13を引用する。残余正本は保存Fractionであり、今回新しい数値guardを行っていない。

up_to_phase=False、exact W・相対位相、負角の連結word全体のactual adjointを必須とする。phase fit、理想pi snap、無条件2pi key同一視、新control後付けを行わない。

## Stage A〜E

| Stage | 到達点 | 資格 |
|---|---|---|
| A：実装意味論 | word／written product逆順、named gate、W float加算、compose／inverse、Rust setter、QPYを追跡 | SOURCE読取で確認。Python8ファイルは公式1.3.0とbyte一致 |
| B：連結とadjoint | scalar Wを保持した固定順の全文字列と、その全体inverseで対応可能 | 代数的条件。基本wordの新取得／連結は未実施 |
| C：phase境界 | 8状態閉包と有限phase経路からe_emit<2^-41<R_emit^loという十分条件を記号的に導出 | EMITTER_CERTIFIABLE_FROM_SOURCE、C_phiと経路に条件付き |
| D：再利用 | exact_word_matrix／guard／lower_native／CT-1／CT-4Bを再利用できる。新emitter必須とはしない | 実word取得後のprovenance／実装対応は未資格 |
| E：次scope | 取得前のSOURCE適合と、取得後のrecord／composite／保存形式を分けた | N3-B1bは未開始。今回の作業自体は完了 |

詳細：[EMITTER_SOURCE_AUDIT](EMITTER_SOURCE_AUDIT.md)、[PHASE_ERROR_BOUND](PHASE_ERROR_BOUND.md)、[MINIMAL_EMITTER_ADAPTER](MINIMAL_EMITTER_ADAPTER.md)、[NEXT_SCOPE](NEXT_SCOPE.md)。

## 記号的な誤差保証の意味

Pi0を保存binary64 pi、P=Pi0/4、tau=2Pi0=8Pとする。Pのsignificandは50 bitなので、0…8倍はすべてbinary64で表現できる。既存localのW加算はsetterで毎回正規化され、通常IEEE条件の下でglobal phaseが`(K mod8)P`へ閉じる。従って、未知W数Kを推測して丸め誤差へ代入する必要がない。

一方、tauは真の2piではない。localの定数差、outer composeの加算／wrap、inverseと必要な再読込は別に数える。named Clifford+T gateのideal意味論を保持し、数値phaseと通常IEEE/remainderの条件C_phiを満たすとき、最大43 carrier＋最終join、1回の全体inverse、合計2回までの再設定／QPY読込を覆う保守的bookkeepingは

\[
e_{emit}<[(43+1)64+32+2\cdot16]u<2^{12}u=2^{-41}<R_{emit}^{lo},
\quad u=2^{-53}
\]

となる。最小経路はwhole concatenated wordを1回emitする形で、上はそれより粗い十分上界である。操作回数は未実装pipelineに対する観測値ではなく、後続で固定すべき具象化経路の条件。任意の後続処理／回数への保証ではない。比較は既存の外向き下端と整数不等式の静的導出で、新しい科学数値評価を実行したものではない。

このFrobenius上界は局所wordにscalar増分を帰属させる意味であり、巨大joint行列のFrobenius上界と読み替えない。whole接続ではspectral telescopingへ移す。先行wordの誤差をvacuumだけへ付け替えず、基本guard／量子化とemitterを二重計上しない。

## 未資格と最小設計

実runtimeの定数bit／通常IEEE条件、named gate保持、具体的なphase経路、必要QPY往復をSOURCE証明へ対応させる最小適合確認が残る。CT-1/CT-4Bの有限Operator診断を任意43-word連結のcertificateとは扱わない。C_phiを使えない経路はW数／word長等に依存する一般境界へ戻し、CONDITIONAL_ON_WORD_METADATA又はUNQUALIFIEDとする。

既存lower_nativeが使えるなら、まずcomposite record／providerだけを限定実装する候補が最小である。integer W ledgerを必要とする場合は、per-wordの整数K mod8とboundedなphase設定だけを新SOURCEへ分離し、元のword/hash・gate順・負角inverse・out.composeを維持する。これは必要時のMINIMAL_ADAPTER_REQUIREDであって、今回新emitterを正式採択／実装したものではない。

bare Qiskit circuitからWを捨てたり、symbolic phase／sidecarというだけでe_emit=0としたりしない。exact carrier表現とfloat具象化の保証を分ける。他のC0 wordの局所許容・全support取得／停止・RPE disk会計も別途残るため、完成辞書／CT RPE成功／T優越は認定しない。

## 参照と保全

[N3-B0 preflight](../pr_reference_n3b0_preflight_20261108/PREFLIGHT.md) `385c24a00fb9b47d6639fea1e7aecb0c37d55b23`、[N3-A](../pr_reference_n3a_static_20261108/REPORT.md) `79dfa01e6796d1ecab50e30694f5b3f73b95f4ab`、[N1](../pr_reference_n1_numeric_20261108/REPORT.md) `0a6ba582d246b3b724d42c8b56475c16ebef1dd9`、[CT-4B](../pr_reference_ct4b_20261107/REPORT.md) `8012c3e9547690f0d722edac9d0b9a9391ba2f8e`、[CT-5B](../pr_reference_ct5b_20261107/REPORT.md) `0dbf1fe265176677e026ea982c0bd435987f4338`、CT-2／CT-3Aの固定保証調査を参照した。

正式共通運用指示2026-10-25、親／worktree AGENTS.md、PROJECT_MAPと既読研究概要を適用。Qiskit1.3.0の固定commitは`896d1c8b72888a52a46081a6d1f4de8c56db8aea`。確認した一次SOURCEと対象箇所は[SOURCE監査](EMITTER_SOURCE_AUDIT.md)、blob参照は[SOURCE_REFERENCES.json](../../../artifacts/pr_reference_n3b1a_emitter_static/2026-11-08/SOURCE_REFERENCES.json)。これは著者Codexの静的技術調査・記号導出であり、独立科学レビュー／機械語までの形式証明ではない。

新規合成／基本word取得、guard数値実行、辞書／連結adapter／emitter実装、Operator検証、回路build/compile、sampling/simulation、T価格／極値／期待値、b／precision再選択、N2／SK／18round総T／T-depthはいずれも0。科学moduleをimportせず、source byte／ASTと保存JSONを読んだだけ。旧SOURCE/raw/result/grant/ledger/marker/STOP、共有設定、他ユーザー・別Trackを変更していない。

指定の独立した静的項目を完了し、残る実証は今回範囲外なので[mandatory STOP](../../../artifacts/pr_reference_n3b1a_emitter_static/2026-11-08/STOP.json)。次担当は**GPT／ユーザー**：C_phiの条件付き証明を評価し、[NEXT_SCOPE](NEXT_SCOPE.md)の最小実装・適合確認・43 key取得へ進むか判断する。CodexはN3-B1bを自動開始しない。
