# 新A V6-Q3-N：有限generator監査・整数検算

2026-11-09、SERVER 2。**PASS_IN_FIXED_ARITHMETIC_MODEL。** 固定計画のN1〜N3を1 campaignで完了した。全有限basis catalogの受入れ条件を確認し、完成済みgeneratorの半角込み共通格子は **ℓ=65**、保存出力の独立整数検算もPASSした。

固定SOURCEのbasis対・event文法・旧correctly-rounded binary64 add/sub/remainderモデルの下で、m17の全canonical finite-bit supportに対する入力受入れとnormal域の十分条件を得た。**実m17 callerの接続、RPE・総RZ再評価、Clifford+T資格を完了した結果ではない。** 実wheel／libm全体の形式証明や新保証の正式採択でもない。

## 1. 固定範囲・再利用

| 固定資料 | 使用内容 |
|---|---|
| [計画 ba616902 §1〜4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba6169028d6d71105aa1f474dfcd285461339070/docs/research/pr_reference_v6q3_static_20261109/NEXT_VALIDATION_PLAN.md)／[静的補題](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba6169028d6d71105aa1f474dfcd285461339070/docs/research/pr_reference_v6q3_static_20261109/ALL_SUPPORT_ACCEPTANCE.md) | 受入れ文法、basis対相殺、normal格子、M/B_norm・価格接続の条件。範囲・選択規則を変更しない |
| [実行SOURCE 2a3a90b0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/2a3a90b047c55730a396372505e564923b3ba081)／[PREPARATION](PREPARATION.md) | driver／checker／人工8テストを科学実行前に公開・内容照合し、そのまま実行 |
| 元科学SOURCE・入力 `f2638a7845685a119619382b850441f5d8d33713` | 保存common.pkl.xz、A1、V2角witness、D/RTE/basis・repeated SOURCEを再利用 |
| provider `eaf70d886d42364a01b7e949ebc21f3b026bbe72` | _binary_inputだけを型・exact ratio監査に利用。phase providerやcore builderは呼ばない |
| [Q2 9560e310](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9560e310e9f848fe59a1dafce4b506d5a08aae76/docs/research/pr_reference_v6q_q2_20261109/REPORT.md)／[V6-P 5f672ce6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f672ce6884bd49231366875a9e24a539fcaec7a/docs/research/pr_reference_v6p_results_20261109/REPORT.md) | 既存有限correctness、M/B_norm・条件付き費用結果を保持。再実行しない |

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外、Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit1.3.0／compiler A／rz,sx,x,cx。B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、D融合・basis共有・scalar／signed event・X/Yを不変更。

対象はm17/q131072/r32/bits88。energy精度1.5936e-4 Ha、failure .05、phase cap .4、αaxis1/720、保存Reference 630／条件付きB1 626 shots/axisとm0〜m16の旧条件を維持。shot・確率・RPE・費用を新計算していない。

## 2. N1：全有限catalogの受入れ

[N1_CATALOG.json](../../../artifacts/pr_reference_v6q3/2026-11-09/N_run01/N1_CATALOG.json)に全recordを保存した。

| 対象・条件 | 結果 |
|---|---|
| 完全catalog | D basis 4＋RTE registry basis 9＝13。既存A1のD index／全registry IDと一致 |
| 操作／parameter | 414操作、724 parameter。全parameterの実型はnumpy.float64で、既存_binary_inputが受け入れるfinite値 |
| genuine gate・shape | 真正RZGate／XXPlusYYGate、所定width・parameter数、wire範囲・重複なし、clbit0・conditionなしを確認 |
| 保存metadata | name／wire／parameter文字列と保存basis metadataが全一致。matrix fingerprintを再生成していない |
| tail support | 全218 componentが非identityの1/2-site supportで、basis ID/hash／保存operationsへ結び付く |
| inverse／basis相殺 | 旧SOURCEの同じbasisを逆順・符号反転して使う条件を保持。inverse回路やcompiled wordを新しく構成していない |

Q2の4 trajectoryの型分布を外挿した結果ではない。固定pool全体を走査した。basis parameterは受入れだけを監査し、指数・誤差・価格は解析していない。D等の数値生成経路とnumeric global_phaseの型閉包は前段の静的補題と併用する。

## 3. N2：完成済みgeneratorとnormal域

[N2_GENERATORS.json](../../../artifacts/pr_reference_v6q3/2026-11-09/N_run01/N2_GENERATORS.json)に135 recordの実型・hex・exact numerator／分母指数・半角込み必要ℓを保存した。

- Dのhalf 4 contextとblock0 full 1 contextを、旧diagonal_primitivesで各1回だけ評価した。完成済みglobal_phase／RZ／RZZを使用し、fullをhalfの数学的2倍へ置換していない。
- scalarは既存repeated SOURCEと同じ`-.02*n*(constant+extracted_identity)`、n=65536を1回評価した。式を再結合していない。
- 中央2角は保存V2のr32/order0,2 actual_hexを読み、lambdaのSOURCE bindingと既存exact_equal witnessを照合した。atanやeventを新生成していない。
- D phase、scalar、0、binary π／τをphase generatorとし、D／中央RTE角にはexact半角の分母1 bitを加えた。符号±は同じ格子に閉じる。

得られた十分値は**ℓ=65≤1022**。従って、静的補題の条件下で

\[
S=S_F+S_G\in2^{-65}\mathbb Z,
\qquad
S=0\ \lor\ 2^{-65}\le |S|\le2M_{\rm saved}.
\]

非零exact和はnormal閾値2^-1022以上になり、現providerのsubnormal reject域を避ける。上側は同じ固定SOURCEに対する保存Mのcertificateを継承し、Mを再評価していない。

**この結論の算術条件**は、前段に記録したbasis対の厳密相殺と、global_phase加算・数値周期による剰余が共通格子を保つ旧correctly-rounded binary64 add/sub/remainderモデルである。単なるγ誤差上界や有限な剰余標本から閉包を推定したものではない。g_F/g_Gの実到達値、実S、RN64(S)、実phase誤差は計算していない。全trajectory列挙もm17 whole F/Gも不要だった。

normal域と同じMの条件が揃うので、前段の受入れ入力向け命題`|RN64(S)−S|≤2uM`、`δ_B1≤B_norm+2uM`へ**同じモデル内で**接続できる。上流normalization、scalar／signed event、controlled SOURCEとの対応を0誤差へ変更せず、旧B_normと他diskは保持する。新δの数値評価やRPE成功条件の再会計はしていない。

## 4. N3：保存出力だけの検算

[CHECKS.json](../../../artifacts/pr_reference_v6q3/2026-11-09/N_run01/CHECKS.json)は`N3_SAVED_OUTPUT_CHECKS_PASS`。checkerは科学cacheをunpickleせず、primitive・providerを再実行しなかった。

保存hexを独立にratioへ戻し、分母指数・numerator・半角1 bit・両符号・共通分母2^65での整数性と最大ℓを確認。全basis／全component、5 D contextのglobal／RZ／RZZ、保存中央2角、scalar式・periodのcoverageと集計、SOURCE／cache・結果hash、1 campaign／retry0／禁止call0、STOPの一致を確認した。結果・SOURCE・STOPのbyteは検算前後で不変。

RESULTS SHA256は`62abc409fa586ffbda7f097838ff45991a545ca4a8f0df2e471a7d20eef54988`。全保存入力と出力の来歴は[SOURCE.json](../../../artifacts/pr_reference_v6q3/2026-11-09/N_run01/SOURCE.json)とCHECKSに記録した。これは著者自身の技術・数値検算であり、別担当の独立科学レビューではない。

## 5. 科学callと実行状態

| 処理 | 実行数 |
|---|---:|
| 固定common loader／unpickle | 1 |
| diagonal_primitives | 5 |
| 固定scalar評価 | 1 |
| 保存中央角読取 | 2 |
| phase provider／新atan | 0／0 |
| trajectory／回路build／compile／simulation／sampling | すべて0 |
| RPE／shot／総RZ再計算／CT・T価格 | すべて0 |
| 科学campaign／科学retry | 1／0 |

科学CLIは1回、exit0。driver内wall time約2.01秒。科学実行前の人工8テストPASSを保持し、旧Q1/Q2等を再実行していない。固定Qiskit1.3.0のcondition属性読取でdeprecation warningが出たが、所定のconditionなし検査を完了しSOURCEを変更していない。

既存project venvを使い、専用worktreeとfresh出力だけへ書き込んだ。小さい有限集合の監査なので重複workerを起動せず、CPU/RAMの独自恒常上限・共有設定変更・他利用者のfile/process操作・AI subagentはない。common cacheコピーはGit対象外でlocal保持し、元の固定archiveを公開入力とする。

## 6. 残る条件・成果保全・STOP

今回解消したのは、**固定SOURCE文法と旧算術モデル下の全support受入れ・normal域の有限入力条件**である。以下は引き続き別義務として残る。

- 実m17 callerとq/r/bitsのbinding、m0〜m16のlegacy保持を運用callsiteへ接続すること。今回は未実装。
- 実callerが同じF/G・law・event順・registerを保ち、relative P角だけを変えること。既存のnative RZ数不変補題の実consumerへの適用条件を確認すること。
- V6-Pの条件付き626 shots／総RZ利益は今回再計算・実装済み利益へ昇格していない。実機RPE、classical長round走査費、CT domain／emitter／word、総T／T-depthも未資格のまま。

driver／checker／testは実行SOURCE commitのまま。旧SOURCE・raw・結果・grant・ledger・marker・STOP、前段の静的文書と負結果を変更していない。[RESULTS](../../../artifacts/pr_reference_v6q3/2026-11-09/N_run01/RESULTS.json)／[STOP](../../../artifacts/pr_reference_v6q3/2026-11-09/N_run01/STOP.json)／入力参照・検算を専用branch `codex/pr-reference-v6q3-numeric-20261109`へ公開し、remote SHAと全追加成果物の内容一致を照合する。

**N1〜N3完了。公開・remote照合後mandatory STOP。** 次担当はGPT／ユーザー：このモデル条件付き資格を踏まえ、実caller接続と既存compiler資格への接続を次に許可するか判断する。Codexはcaller実装、新phase変更、追加科学計算・価格・RPE・CTへ自動進行しない。
