# SOS-PF A3正式採用：現分割生成アルゴリズム探索の終了

2026-10-16 JST。区分：完成済み科学レビューを受けたユーザー採用判断の記録。今回、新しい科学レビュー・数学的検証・科学計算は行っていない。

## 正式採用と適用範囲

ユーザーが「A3でよい」と明示した旨を今回の正式指示で確認し、**A3を正式な研究方針として記録する。SOS-PF候補Aを、新しいPF-native分割生成アルゴリズムとして追究する現在の研究系列を `CLOSED` とする。**

これは現系列への研究資源配分判断である。SOS-PF一般の不可能性、SOSを用いる誤差保証の無価値、量子化学Hamiltonian simulation研究全体の終了は主張しない。

正式状態の機械可読正本は[今回の採用decision](../../../artifacts/qc_sos_pf_a3_state_adoption/2026-10-16/decision.json)。過去のmethod status・decision・STOPを上書きせず、この新しい採用decisionで現在状態を区別する。

| 対象 | 正式な扱い |
|---|---|
| SOS-PF候補A：現PF-native分割生成アルゴリズム探索 | `CLOSED`（A3採用） |
| K0・K0.5の既存技術結果 | 保存・変更なし |
| K0全体の技術判定 | `UNDETERMINED_DATA_REQUIRED` を維持 |
| A2：有限step誤差保証理論 | `NOT_ADOPTED`。可能性として保存し、研究は開始しない。A2をCLOSEDとはしない |
| K1-F数値検証 | `NOT_AUTHORIZED / NOT_EXECUTED`。準備拡張も行わない |
| SOS-PF一般 | 一般的no-goは主張しない |
| その他の独立研究・D1/D2 | それぞれの既存状態・範囲を維持し、本decisionでは変更しない |
| 次の研究方向 | 未決定。自動探索・採択なし |

**研究系列の正式なCLOSED判断と、K0の技術的な未判定を混同しない。** A3採用は新アルゴリズムの正式採択でも、A2への自動移管でもない。

## 採用理由と完成レビュー

以下は完成レビューとユーザー指示に基づく採用理由の保存であり、Codexによる科学的再評価ではない。

1. K0・K0.5を経ても、既存SOS・DF・BLISS・平方完成・Gram因子化などと区別できる、独立した有利な分割生成規則が特定されなかった。
2. SOS gapを利用する数学的な誤差評価の成立だけでは、新しいアルゴリズムの独立性を確立しない。
3. 有用な有限step保証、実際の基底固有位相誤差の改善、総計算費用の削減は未確立である。
4. 現段階でK1-Fの数値検証へ進む研究上の優先度は低いと判断された。

根拠は[完成GPT独立科学レビュー](../../reviews/quantum_chemistry_pf/qc_sos_pf_post_k05_gpt_scientific_review_20261016.md)の§0.1–0.2、§4.3、§5.3–5.4、§7.1、§10、および同レビューが対象とした[完成Claude科学レビュー](../../reviews/quantum_chemistry_pf/qc_sos_pf_post_k05_claude_scientific_review_20261016.md)の§0.1、§5、§7.1–7.3である。両レビュー間の留保・修正も原文に保持する。レビューの提案・未認可草案を採用済みの実行条件に格上げしない。

## 原本の同一性と時点の区別

ユーザー提供 `SOS_PF_A3_Claude_GPT_review_originals_20261016.zip` を原本とし、同梱 `SHA256SUMS.txt` と照合して保存した。変更は保存先ファイル名だけで、本文・改行・空白は変更していない。

| ZIP内の原本 | 保存先 | byte数 | SHA-256 |
|---|---|---:|---|
| `レビュー.md`（GPT完成版） | 上記GPTレビュー | 56,985 | `ab31421c04774a84c8898c4a047debaab226683807e277595491412c406b0e77` |
| `レビューoutter(1).md`（Claude完成版） | 上記Claudeレビュー | 72,558 | `576cb72c470139ab9176d6611f8a61788fc03f2b3a9d3b3acbc74702a8219797` |

レビュー内の「正式採用前」「ユーザー判断待ち」「まだ正式CLOSEDにしない」等はレビュー時点の記録として残す。今回のユーザー採用は本ノートとdecisionで記録し、原文へ遡及反映しない。

**Claude原本は今回のA3採用記録作業で取得した。** K0・K0.5実施時点の「Claude原本未取得」は過去時点の正しい記録として保持する。ACM出版完全本文403・arXiv v3との版差分未確認など、既存の取得限界も変更しない。

## 固定した既存成果

- 既存研究状態：[`78b50c515d9df9556f261e74ca7c57af461b7199` の採用decision](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/artifacts/qc_pf_exploration_state_adoption/2026-10-15/decision.json)。他系列への今回の状態変更はない。
- K0：[`58d8672702284d73321cc67a26c4640adaddde5f` の資料索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/README.md)、[技術判定](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/method_status.json)。
- K0.5：[`01bdc56a2970dcb749390b0540846143f775d6f9` の資料索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/README.md)。今回の公開branchの基点もこのcommitである。

## 完了境界

今回の承認はA3採用判断の記録・原文保存・専用branchへのcommit/push・独立再取得照合に限定される。既存科学source・results・authorization・STOP・global manifestの既知FAILを保持する。新規科学計算・保存科学数値再集計・新たな科学レビューはいずれも0件。

block-encoding/qubitizationの扱い、絶対基底エネルギー以外への研究対象拡張、次の研究方針は別の判断事項である。K1-F、A2、候補B/C、旧系列の再開は認可しない。

**公開と独立照合後にMandatory STOP。次の研究方向をCodexが独断で決めない。**
