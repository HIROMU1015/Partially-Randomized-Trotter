# GPT・ユーザー向けhandoff — C1-3-R2/C1-4完了、mandatory STOP

1. **Reference不一致の原因**は旧複素算術`real+1j*imag`によるIEEE754 zero符号bitの変化。固定R1保存局所行列で再現し、全59旧不一致がzero符号bitだけと確認した。前回selected順の最初はbasis_002/operation 0、byte offset15。新helperは期待値を変えず実虚別格納で復元する。[原因報告](REFERENCE_REPAIR.md)
2. **13 basis・全412局所行列の新bytes/hashが一致**。operation/descriptor順、basis ID/hash、R1原資料を保持し、独立referenceを維持した。[Phase B](phase_b/summary.json)
3. **C1-3-R2は`C1_3_CORRECTNESS_PASS`、新規1回・retry 0**。新SOURCE/契約/authorizationを先に公開固定した。旧2系列の失敗を変更・resume/retryしていない。[実行identity](result/execution_identity.json)・[terminal](result/terminal.json)
4. **固定6 eventは事前GateすべてPASS**。state/hash/norm、schedule/short step、F1/F3 P0-P5、positive proposal・weight/root/angle・Evidence、strict controlled active/inactive/full matrix、actual adjointとX/Y診断。全label/event/path/native/物理準備への一般化はしない。[correctness報告](CORRECTNESS_REPORT.md)
5. **C1-4の技術契約草案を完了**。追加科学計算・native合成・benchmarkなし。epsilon_complex=0.05、alpha_familywise=0.05は未採択候補のまま、新N/資源winnerは未計算。[誤差契約](C1_4_ERROR_CONFIDENCE.md)・[費用/対照契約](C1_4_NATIVE_BASELINES.md)
6. **残件**はrigorous DF/model binding、PF定数、全path mean/統計/終端、uniform strict DF/native全support、準備/readout、非列挙の認証価格/期待費用、canonical他方式の有限bit/DF接続、強いUNKNOWN対照。既存の一般保証へ条件を結ぶ作業と、必要になった新証明を分ける。

## 次の担当・必要な承認

| 担当 | 次に判断/実施する内容 | 承認境界 |
|---|---|---|
| ユーザー・GPT | 保存DF対R1のtarget、最終精度/confidence、比較family、native費用の到達層、state/access/相殺、UNKNOWN baselineの扱いを判断 | C1-4草案の正式採択は未実施。技術成功だけで研究価値・新規性・C1-5実行GOを採択しない |
| Codex | 上記判断と明示scopeが来た後、必要な既存certificate/SOURCE回収、whole-path/他方式/provider/Evidenceの不足実装・人工確認を行う | 今回は未実施。新実装/詳細検証/新非自明証明は必要な別承認の範囲で実施 |
| ユーザー | SOURCE・入力・全誤差/failure・coverage・native取得profile・baseline・one-shot/STOPを固定した次の科学実行を明示承認するか判断 | C1-5、本格native synthesis/費用取得、shots/Monte Carlo比較、全round RPE、Full-N/Pilotは今回未承認 |

C1-5のGO前提は共通target/比較契約採択、必須誤差/coverage/価格/Evidenceの接続、bias控除後の正の統計残余と全trial予算の事前固定、強い対照への公平な最適化機会である。UNKNOWNを0として救済せず、新科学条件・新非自明保証が必要なら判断へ返す。科学的解釈・研究方向・投稿判断はユーザー/GPTが行う。今回別agent/外部reviewerによる独立reviewは行っていない。

## 公開・保全

- branch: `track-b-c1-3-r2-reference-recovery-c1-4-20261021`
- 基点: `49dae5247ff0ad2c4efb9056e095b70913cc9ddb`
- SOURCE: [`2be82f7bff78b7d11ffb11013b03a49e296acd1d`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/2be82f7bff78b7d11ffb11013b03a49e296acd1d/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021)
- 新authorization: [`324634a37538eea2bf6153467fa0216ffa8a5c56`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/324634a37538eea2bf6153467fa0216ffa8a5c56/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/authorization_receipt.json)
- 結果commit・公開bytesは`publication_receipt.json`。receipt自身を含む最終HEAD/公開再照合は専用runtime directoryの外部receiptと最終チャットにも記録する。

旧SOURCE/raw/grant/claim/ledger/marker/STOP、R1・旧失敗原成果、非公開レビュー原本を保全した。共有venv/cache/OS・他者process・無関係worktreeは変更していない。新namespace追加だけで公開する。

**公開・保全確認後mandatory STOP。次の承認まで追加実装・数値解析・科学計算・合成・比較へ進まない。**
