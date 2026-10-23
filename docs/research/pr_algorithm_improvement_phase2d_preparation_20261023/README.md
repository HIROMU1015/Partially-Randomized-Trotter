# Phase 2D-P0：F1・条件付きF2の検証契約準備

2026-10-23 JST。**文書・静的監査・公開のみ。科学実行を許可する文書ではない。**

準備状態は `PREPARATION_INCOMPLETE`。数値仕様は静的な候補として具体化したが、新prototype SOURCE、生成後input identity、arm別component件数をまだ固定できない。F1承認可能状態 `READY_FOR_F1_APPROVAL` ではない。これらをP0で科学処理して埋める権限はない。

科学的問いは「既知のPauli／frame辞書構成をPRへ接続したとき、同じHamiltonian・finite target・精度条件で実際の資源を減らせる場合があるか」。分類は `KNOWN_METHOD_APPLICATION_CANDIDATE`、独立方法差は `INDEPENDENT_METHOD_DELTA_UNESTABLISHED`。RQ-01/03を終了せず、元のPhase 2Cの応用検証を準備する。R1の反可換Pauli対・unitary partitioning案を含めない。mixed最良を成功条件にしない。

## 成果物

| 文書 | 用途 |
|---|---|
| [F1契約案](F1_EXECUTION_CONTRACT_DRAFT.md) | 正しさの検査と限定call |
| [条件付きF2契約案](F2_CONDITIONAL_CONTRACT_DRAFT.md) | 全有限trajectoryのactual費用・N |
| [入力候補](FIXTURE_AND_INPUT_IDENTITY_DRAFT.md) | 係数、frame、時間、環境、identity |
| [motifの固定表](MOTIF_CATALOG_DRAFT.md) | 8件×4arm、catalog68件の指定 |
| [比較公平性](COMPARISON_AND_FAIRNESS.md) | 4arm・selector・既知全列挙 |
| [意味論・数値gate](SEMANTICS_AND_NUMERICAL_GATES.md) | 別々の合格条件、閾値の根拠 |
| [実装接続案](SOURCE_IMPLEMENTATION_PLAN.md) | 最小prototypeと既存版の相違 |
| [call台帳案](CALL_INVENTORY_DRAFT.json) | 件数式、入出力、失敗、重複防止 |
| [承認・STOP](STOP_AND_AUTHORIZATION_BOUNDARIES.md) | 現権限、将来承認、禁止事項 |
| [preflight設計](PREFLIGHT_CHECKLIST_DRAFT.md) | 将来の実行前確認。今回未実施 |
| [出典台帳](SOURCE_AND_PROVENANCE_INDEX.json) | 固定commit、bytes、SHA、確認範囲 |
| [準備状態](READINESS_AND_BLOCKERS.md) | 解消条件・担当 |
| [引継ぎ](handoff.md) | ユーザー・GPT向け判断材料 |
| [管理検査](VERIFICATION.md) | 文書検査とその限界 |
| [公開receipt](PUBLICATION.json) | 検証済み文書commit・公開識別子 |

## 固定資料と時点

元設計：[Phase 2C](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/0c94a3d973b0ba0d24b3d277ab8316829739b278/docs/research/pr_algorithm_improvement_phase2c_design_20261023)。限定方法差監査：[Phase 2C-R1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/ef2eb7d55ea37bf93e0b85773df20b63811d1f9b/docs/research/pr_algorithm_improvement_phase2c_method_delta_20261023)。RQ・強対照：[Phase 2A](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/fbb83680247a4565336ca177278255e32843cfb2/docs/research/pr_algorithm_improvement_phase2a_20261023)。必須13ファイルを固定commitから再取得し、固定checkoutのbytes／Git blobと照合した。

最新の非公開 `PR_Phase2C_R1後_研究方向再選定レビュー_2026-10-23.md` 原本は取得できなかった。原本を読了したとはしない。ユーザーのP0指示に明記された採択済み方針と固定資料を基準とする。この代替は指示で許可されている。非公開レビュー、private raw、認証情報、内部絶対pathを公開しない。

実装起点は `ef2eb7d55ea37bf93e0b85773df20b63811d1f9b`。参照する後続の回路共有SOURCEと起点の回路builderには版差がある。起点に既に全機能があるとはしない。[接続案](SOURCE_IMPLEMENTATION_PLAN.md)に差を記録した。

実施したのは固定資料の取得・テキスト読解・hash照合・契約案・文書管理検査・公開。新規実装、SOURCE import、fixture／数値行列／Pauli係数／科学input生成、sampling／trajectory生成、回路build／compile、simulation、非自明な証明、科学grant／ledger／marker作成は全て0件。過去のSOURCE・結果・契約・レビュー・STOPは変更しない。

**公開・照合後mandatory STOP。次の担当はユーザー。F1にもF2にも別途明示承認が必要。**
