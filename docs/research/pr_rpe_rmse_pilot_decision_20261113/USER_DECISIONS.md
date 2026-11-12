# C：ユーザーが決めること

**今回、科学計算を始めていない。** 次は下記の提案を一括して選ぶか、変更箇所を指定することで条件を固定できる。各値/式は推奨候補であり、元の[入力template](../pr_rpe_rmse_pilot_preparation_20261113/PILOT_INPUT_TEMPLATE.json)は未定のまま保全した。

## 推奨する一組：提案A＋保存再解析限定S0（未採択）

| 固定する項目 | 提案A |
|---|---|
| 系・二時点 | H4 DF10、PR m0/m6 |
| 外側条件 | 既存H/state/snapshot、S2/L_D1/ΔPF=.1/K2/h=.1、prior/decoder/profile/P0/無shiftを保持 |
| 保存対象 | m0 R2 `53e00d8b07d1547d19a7`、m6 R128 `2a8c6718220eec59d419` |
| 共通κ | `κ_A=(255/128)/(λ_1² h²)`。保存候補に整合する記号帯の中点という決定式。数値は未計算 |
| 目標RMSE | `εE=0.00159360010199040 Ha` をRMSE metricとして新たに採択する候補 |
| 予算・合成則 | `εqpe=εtrot=εE/2`、既存`minkowski` rule。実際の成分RMS上界の資格は未取得 |
| 最大round index | `M=max(6,ceil(log2(0.10π/(h εqpe))))`。PR経験的モデルによる決定式、旧M12を自動流用しない |
| normalization | `finite_B_diagnostic`。経験的nominal Nのみ |
| 科学scope | 固定SOURCE値から式を具体化し、PR二候補の保存gate・局所bias/phase/radius/logB・nominal N_X/Yを再解析。**新mean/新reference/normalizer再計数0**、欠測や不一致はSTOP |

この提案はS2一piece・初期例外なし・保存P1再利用という理由から最初のpilotを小さくする案である。κ/N/Gの最適化案、finite-Kや全decoderに対するRMSE保証ではない。SOURCEはA ee15/model510、B f09、P1公開71ba/科学a1a79213、保存λのSOURCE_INDEX/contextを[実行計画](EXECUTION_PLAN.md)に固定している。

## 変更したい場合の最小指定

1. **対象・κ**：H4のもう一つの適合組R=(4,256)、またはH6 m4/m8の適合3組から選び、同じκを決める帯内点の規則かliteral値を指定する。候補条件は[REPORT](REPORT.md)の記号表。全roundに別々のκを割り当てない。
2. **目標・予算・M**：別εE/εqpe/εtrot、paper_quadrature、literal Mを使う場合は明記する。Mは包含条件とPR経験的horizonを両方満たす候補にする。paperの最適予算・Cmaxを現finite-Kの保証へ読み替えない。
3. **normalization・追加call**：paper_kappa_envelopeや不足取得S1を希望する場合は明記する。S1は固定二候補に限り最大mean2・内部対照2・必要reference4 action。SOURCE不一致/保存FAILを自動再実行する承認ではない。

## 次の承認で明記すること

ユーザーは、採択する**提案A（または変更条件）とS0/S1の科学実行を承認するか**を明記する。式で固定したM/κは、採択した保存SOURCEから一度具体化する処理まで含められる。numeric literalを先に確認したい場合は、その具体化だけを別の有限scopeとして指定できるが、手続上必須の段階は新設しない。

方針への賛同だけを、未指定の条件や科学実行承認へ広げない。今回の文書を作る許可で実κ・RMSE値を評価していない。科学的に適格な十分Nや全RMSE保証を必須とする場合は、局所pilotとは別にstate/finite-K/alias/全decoderの証拠を選ぶ必要がある。

**担当**：ユーザーが条件と有限科学scopeを選択・承認する。Codex Aは明示承認された一組・二時点だけを実行して結果と未資格を報告する。B/Cの統合/価格/全Gへ自動進行しない。今回の条件整理は公開照合後mandatory STOP。
