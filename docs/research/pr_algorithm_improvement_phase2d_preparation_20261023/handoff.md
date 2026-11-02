# Phase 2D-P0引継ぎ

2026-10-23 JST。**準備文書のみ。実行権限なし。状態PREPARATION_INCOMPLETE。**

1. **科学的問い**：既知のfragment別Pauli／frame辞書をPRへ接続して、同じH・finite target・精度でactual資源を減らせる場合があるか。RQ-01/03を終了しない。分類KNOWN_METHOD_APPLICATION_CANDIDATE、INDEPENDENT_METHOD_DELTA_UNESTABLISHED。R1反可換pair案は除外。
2. **F1**：H一致、finite corrected mean、law/sign/order、actual controlled operator／adjoint／ancilla phase、I/c、full/sector、basis共有／D融合、X/Y readout、literal価格DPを別々に検査する。same R→same polynomialだけでは認証しない。
3. **F2**：F1後の別承認で全L² trajectoryのactual full-wrapper費用、lambda/B/N、N加重RZ/CX、depth、prep/readout、古典生成・選択・cache費用を比較する。高次数・分子一般・全RPEは対象外。
4. **固定候補**：4mode、2粒子X0X1、ancilla4、LD1、density-square residual2、exact rational係数／Givens frameを[入力案](FIXTURE_AND_INPUT_IDENTITY_DRAFT.md)に記載。SS/SP/PS/PP、同情報・cache・compiler・最適化、selectorと既知四候補全列挙を区別。mixed最良は必要条件ではない。
5. **SOURCE/input**：base ef2eb7d55ea37bf93e0b85773df20b63811d1f9b、参照2978/f29/ae743を[出典台帳](SOURCE_AND_PROVENANCE_INDEX.json)に固定。回路builder版差とstrict DF wrapper接続不足を明記。新SOURCE、input SHA、env lock、arm Lはnullで、生成済みとはしていない。
6. **call範囲**：F1は3sum L component probes、32motif×3版=96controlled builds、X/Y64wrapper/compile、catalog68×2=136builds、各gate。F2はQ=2sum L²、prep/readout診断2を加えQ+2 logical wrapper/compile requests、unique missesはIR binding後に確定。詳細は[台帳](CALL_INVENTORY_DRAFT.json)。全件未実行。
7. **閾値とSTOP**：候補H1e-11、control1e-10、mean/signal2e-10、compiled5e-10、rtol0。90/128digits参照は数値診断で厳密certificateではない。phase/order/sign/adjoint/sector/scalar/hash/call/保全違反は即STOP、自動retryなし。成否を問わず実行後mandatory STOP。
8. **未確定**：新prototype/closure、input採用と生成hash、L/coverage、価格provider、interval/ceil資格、env patch/wheels、API truncation_tolerance、finite-bit/bias割当、最終call整数／出力容量。最新private GPT review原本は未取得で、ユーザー指示の採択済み方針を利用した。
9. **実装・科学実行の必要性**：契約案だけではmixed adapterを検証できない。P0は禁止範囲を守り、実装／import／fixture生成／simulation／sampling／build／compile／非自明証明／新科学grant発行は全て0件。SOURCE/inputを固定するためI（実装のみ）とB（限定input binding）を提案するが、今回未承認。
10. **次の担当・承認**：ユーザーが[承認境界](STOP_AND_AUTHORIZATION_BOUNDARIES.md)のI/B範囲を判断。Codexは明示承認後だけ最小prototypeとbindingを行い、最終SOURCE/input/call契約を提示してSTOP。その後ユーザーがF1を明示承認、F1結果確認後にF2を別承認する。GPTレビューを自動依頼しない。
11. **公開**：branch `research/pr-algorithm-improvement-phase2d-preparation-20261023`。固定文書commit／tree／bytes/SHA256/blobのreceiptは[PUBLICATION.json](PUBLICATION.json)。入口は[専用branch](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/research/pr-algorithm-improvement-phase2d-preparation-20261023/docs/research/pr_algorithm_improvement_phase2d_preparation_20261023)。receiptの検証対象commitを使えば、公開receipt自身を含む最終commitと混同せず追跡できる。
12. **mandatory STOP**：P0文書の公開・remote HEAD／変更files／既存file非変更／bytes／SHA256／Git blob／文書リンクの照合後に停止する。この公開はF1/F2、旧C1、RQ-04、性能GO、新規性GO、投稿GOの承認ではない。

管理検査の範囲は[VERIFICATION.md](VERIFICATION.md)。既存SOURCE・input・raw・研究記録・privateレビュー・grant／ledger／marker／STOP・他worktree/job・論文原稿を保全した独立namespaceの追加だけとする。
