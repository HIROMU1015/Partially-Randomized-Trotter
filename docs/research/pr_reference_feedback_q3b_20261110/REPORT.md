# 新A Q3-B：区間branch・有限count・hull丸めの検証

2026-11-10 JST／SERVER 2。**B0→B1→B2を承認済み固定scopeで連続実施し、既存192-bit区間モデル内でPASSした。** 新selectorは旧prefix成功event上で正しいbranchを一様に選び受理する。Feedback算術とenergy-hull丸めも、Q0の保存余裕内に接続できた。一方、native全contextの追加誤差は未資格であり、**実m17の340-shot成功保証・全18round総コスト削減は未認定**。

基準commitはQ3-A `c624114d99056373c7cbf5e933fb8ee786259c42`。同じbranch `codex/algo-improve-new-a-total-cost-20261109`を継続し、Q3-B専用領域へ追加した。旧Q0／Q1／Q2／Q3-A、他Track、SOURCE・raw・results・grant・ledger・marker・STOP・KEEP契約は変更しない。

## 1. 固定条件・実装

Linear H4／1.00 Å／STO-3G／DF12、元H exact ground供給・準備費除外、Partial S2／LD3／h=.02／K2、canonical independent finite bits、既存18round。m17 q131072/r32/bits88、energy精度1.5936e−4 Ha、failure .05、phase cap .4を維持。

[新opt-in adapter](../../../src/trottertracks/pr_reference_feedback_q3b.py)は、Q1のcount→192-bit argを再利用し、`(qm Cprev−pm)/(2π)`の区間全体が一つのstrict最近整数cell内にある場合だけjを選ぶ。signed liftを保存し、旧float decoderは選択後の診断に限定した。Q1原本は変更しない。feedback、readout、confidence decoderもQ1の同じ関数・型をそのまま使う。

数学的成功event上でRは正しい整数から1/5未満、半整数境界から3/10より大きく離れる。B1の全count幅と192-bit basic arithmetic boundにより、その区間幅は2⁻¹³⁰未満。m0の開始、m1〜16の帰納、period／aliasと中心区間までの論証は[BRANCH_AND_NUMERICAL_GUARANTEE.md](BRANCH_AND_NUMERICAL_GUARANTEE.md)。判定は**`PROVED_ACCEPTANCE`（固定成功event・区間モデル内）**で、科学的成功event自体が観測できるという意味ではない。

## 2. B0とB1

[人工6テスト](../../../tests/test_pr_reference_feedback_q3b.py)はPASS（[log](../../../artifacts/pr_reference_feedback_q3b/2026-11-10/B0_TESTS.txt)）。旧fixtureを再利用し、正負・period cut・strict cell／tie・zero・誤alias・arm binding・Q0 decoder接続・旧float診断の故障を確認した。旧suiteは再実行していない。

保存Nsは両armで同じ11種類。fixed f2638a78のm0〜16を不変更で使い、同じNのarg計算を共有した。[INPUTS](../../../artifacts/pr_reference_feedback_q3b/2026-11-10/run01/INPUTS.json)に全scheduleとdomainを保存。

| B1項目 | 結果 |
|---|---:|
| 重複を除くcount位置 | 587,815 |
| 非ゼロvector | 587,808、すべてPASS |
| ゼロvector | 7、拒否を確認 |
| 両arm×全prefixへの対応位置 | 1,652,068（非ゼロ1,652,048／ゼロ20） |
| 完全監査row | 2,495 |
| m16 feedback対象 | N347の121,104位置すべて |
| 最大arg区間幅 | 2⁻¹⁸⁹ rad、約1.27448e−57 |
| 最大feedback算術誤差 | 約2.22034e−16 rad |

件数はcount-domainの位置数であり、実測shotsを生成した数ではない。E0による成功履歴の選別・history直積の列挙・samplingは行わない。

包含は既存directed-interval primitiveモデルとexact Fraction入力に基づく。全非ゼロ位置で整数座標によるarg区間との整合も確認したが、これを独立したlibmpの形式検証とは呼ばない。有限sampleで全域を代用せず、全対象を実際に処理した。

**[B1結果](../../../artifacts/pr_reference_feedback_q3b/2026-11-10/run01/B1.json)／[全row記録](../../../artifacts/pr_reference_feedback_q3b/2026-11-10/run01/B1_ROWS.jsonl.xz)。** row記録は全kYの判定範囲・zero・最大幅・witness・case hashを保存する。全angle値の一覧を重複保存したものではない。B0.jsonのpending記載はB1開始前の段階記録で、最終判定はRESULTS／CHECKSへ保存した。

## 3. B2：保存余裕への接続

Q0固定RESULTS／CHECKSのn340下側PASS bracketと共通δsafeを使い、旧scan・反転・二分探索を再実行せずexact rationalで判定した。

新しいchargeは、κ（energy中心区間幅）、ωE（energy-hull復元丸め）、ωQ0（逆正弦端点を含む追加数値slack）。PF・finite-bit・native diskの再加算ではない。逆正弦の端点ではsqrt／Hölder boundを使い、単純なO(u)や誤差0と置かなかった。

| 項目 | 両arm共通の安全側値（表示） |
|---|---:|
| κ | 約3.42121e−47 Ha |
| ωE | 2⁻¹⁷⁰ Ha、約6.68192e−52 |
| ωQ0 | 2⁻⁶⁴ rad、約5.42102e−20 |
| `κt17+ωEt17+ωQ0` | 約5.42102e−20 rad |

| arm | 保存`δlo−δsafe`（表示） | 判定 |
|---|---:|---|
| B′ | 約1.26151e−7 rad | PASS |
| Reference-B1 | 約8.61444e−13 rad | PASS |

判定には完全な有理数を使い、表示値へ丸めてPASSを作っていない。[B2.json](../../../artifacts/pr_reference_feedback_q3b/2026-11-10/run01/B2.json)に全項と正の残余を保存した。共通cap `.000748774257 rad`、phase cap、failure条件を変更していない。

## 4. 検算・実行記録

[checker](../../../scripts/check_pr_reference_feedback_q3b.py)は全row coverage、各rowの極値、arm／round対応、中心・cell・energy幅bound、有理数のB2不等式を検算した。11個の最大arg幅witnessと1個の最大feedback誤差witnessだけを同じ固定domain内で再現し、全gridは再実行しなかった。[CHECKS.json](../../../artifacts/pr_reference_feedback_q3b/2026-11-10/run01/CHECKS.json)はPASS。著者Codex自身の技術検算で、独立科学レビューではない。

B1では実際のCPU affinity64に対して64 processを使用。起動時CPU／memory PSI avg10=0、利用可能memory約410GiBを観測し、恣意的な資源上限は置かなかった。B1/B2主処理の所要約2.84秒は当campaignの記録であり、一般的な性能比較ではない。共有設定・他process・packageへ変更なし。

[実行SOURCE](../../../scripts/run_pr_reference_feedback_q3b.py)／[SOURCE.json](../../../artifacts/pr_reference_feedback_q3b/2026-11-10/run01/SOURCE.json)／[最終RESULTS](../../../artifacts/pr_reference_feedback_q3b/2026-11-10/run01/RESULTS.json)。SOURCEのprohibited_callsはこの主処理についての記録である。科学cache／ground、科学回路build／compile、sampling／RPE、native価格、全round会計、strict合成／Tはいずれも0件。

## 5. B2後の許可された技術確認

科学入力を使わない1qubitの自由symbol φに対し、既存Compiler AでP(−φ)→H→Measureを1回だけlowerした。[人工test](../../../tests/test_pr_reference_feedback_q3b_lowering.py)はPASSし、φ依存のinstructionはRZ(−φ)のみ、共通global phaseのφ係数は−1/2だった。[POST_B2_TECHNICAL.json](../../../artifacts/pr_reference_feedback_q3b/2026-11-10/run01/POST_B2_TECHNICAL.json)に保存。

これは追加の**人工compile1件、科学compile0件**であり、新しいH4回路・科学価格を取得したものではない。孤立readoutで同じ軸parameterを保つ条件は具体化したが、m17全core／trajectoryのbody lowering・誤差charge保持、trajectory共通角、canonical consumerへの実接続を自動的に資格化しない。

## 6. 結論とmandatory STOP

| 主張 | 今回の判定 |
|---|---|
| 区間selectorの成功event上の受理 | 固定192-bitモデル内で証明・全count幅で確認 |
| Feedback算術・中心・hull丸め | 固定scope内でPASS |
| Native全contextの追加誤差 | UNQUALIFIED |
| 実m17の340-shot成功保証 | native条件等に条件付き、正式未採用 |
| 全18round総費用・総T | 未評価 |

native条件が閉じれば、fresh independent trialsに対する旧Q0の十分条件と`34/720+1/360=.05`へ接続できる。今回はその前提を0や有限fixtureで代用していない。Q改善は両arm共通で、Reference固有の改善として二重計上しない。

[NEXT_VALIDATION_PLAN.md](NEXT_VALIDATION_PLAN.md)に残るnative保証と費用接続の最小判断を示した。**次担当はGPT／ユーザー。** 数値保証の採否と、次のnative資格scopeを判断する。Codexは公開・remote照合後[mandatory STOP](../../../artifacts/pr_reference_feedback_q3b/2026-11-10/run01/STOP.json)。科学レビュー、m17 whole、追加sampling、全18round会計、Clifford+Tへ自動進行しない。
