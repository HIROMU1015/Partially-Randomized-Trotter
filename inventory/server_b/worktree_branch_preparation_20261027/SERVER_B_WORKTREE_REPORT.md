# SERVER_B worktree整理準備・GitHub branch分類

2026-10-27 JST。PUBLIC_REDACTED。本人所有のGit登録と既存索引の非破壊差分調査。

**登録checkoutは379件（linked worktree375・共通Git owner checkout4、4 Git管理領域）。主リポジトリの登録344件だけで打ち切らず、既知worktree親directory直下のGit markerから別管理領域を特定した。実際の削除・登録解除・branch削除・merge・科学実行は0件。**

**B3は現在稼働中と認定できない。10:23 JST以降のread-only確認でcontroller/workerを検出せず、user serviceはnot-found/inactive、既存journalに09:36 JSTの停止と失敗終了がある。保存されたSCIENTIFIC_ACQUISITION_RUNNINGは古いcounterのままで、科学取得完了を示さない。停止要求の主体・理由は今回scopeで確定していない。今回の調査による停止・signal・再設定・再開は0。B3関連資産を保護したまま扱った。** 詳細は[B3保護報告](B3_PROTECTION_REPORT.md)。

## 件数と分類

| 分類 | 件数 | 扱い |
| --- | ---: | --- |
| KEEP_ACTIVE_B3 | 3 | B3用保護ラベル。実稼働の認定ではなく、未完了・停止観測のためSOURCE/baseline/共通Git ownerを保持 |
| KEEP_REQUIRED | 59 | 他の共通Git owner、可視利用/lock、前回の固定path/code bindingを保持 |
| BULK_APPROVAL_CANDIDATE_PENDING_OWNER | 83 | 前回依存scope・同HEAD・現在clean/未追跡0/ignored0・live GitHub tipを確認。利用終了と他環境の判断待ち |
| PENDING_OTHER_ENVIRONMENT | 63 | cleanで公開tip保持だが、新規/変更scope等の依存・他環境条件が未完了 |
| PRESERVATION_ACTION_REQUIRED | 171 | 変更/未追跡/ignored/private資産、exact公開tipや保持ref未確認などの保全条件 |
| UNRESOLVED | 0 | 登録metadataエラーによる主分類は0。各分類のscope外条件は別に残る |

83件は今回の技術的な一括判断候補であり、実行承認や全依存の不存在証明ではない。171件も全てが未pushと断定しない。同名branchの既存graphを対象限定で確認した5件中2件は現在tipの祖先、3件は祖先でない。残るtip非一致について別branchからの包含・remote保持の全閉包は未確認であり、公開版の未検証HEAD/refは非公開として伏せた。

全行は[worktree分類CSV](SERVER_B_WORKTREES.csv)、具体的な保全手段は[保全計画CSV](SERVER_B_PRESERVATION_PLAN.csv)、正確な83 ID集合は[一括整理計画](SERVER_B_BULK_CLEANUP_PLAN.md)を参照する。B自身のIDを新規に付け、LOCALの75/139 IDや結論と混ぜていない。

## 再利用した証拠と現在差分

2026-10-21の本人専用worktree監査（274 directory・当時候補99件・参照4077件）と、そのGit status/参照の区別方法を再利用した。現在登録379件中257件に旧監査行が対応した。旧99件という結論をそのまま採用せず、同HEAD・現在status・保持ref・可視利用を確認して83件へ絞った。

現在のGit statusはoptional locksとfsmonitorを無効化してread-only確認し、B3/共通Git owner/可視利用の6件はstatus走査を保留した。全file再hash・全raw/checkpoint再走査・du全tree再集計は0。旧容量を現在解放量として表示しない。前回のmetadata bindingは当時の証拠であり、今なお科学的必須か未解決のものは安全側でKEEP_REQUIREDとした。単なるHEAD/status観測や文章上の引用を新しい保持契約にはしていない。

変更trackedのあるtreeは3件、untracked entryのあるtreeは17件、ignored entryのあるtreeは116件（重複）。非B3・非可視利用treeの列挙済み小型text/code47 file、4,076,634 bytesを本人専用領域へ独立copyし、source前後のsize/mtime/inode安定・保存SHA一致・source非hardlinkを確認した。stage済みindexとworking copyの差、削除済みfile、未走査directory、private rawの全保全は別条件で、分類は自動的に候補へ進めていない。

SOURCE/provider/input/runtime/基準結果・元STOP・grant・ledger・marker・共通Git・branchは保持。研究branchでcheckout/rebase/reset/clean/prune/gcはしていない。追加保全の書込みは本人専用領域と独立公開cloneだけであり、原本移動0。原本にcredential等があっても公開版へ転記しない。

## GitHub branch分類

開始時のlive GitHub headsは801件、800固有tip。現在tipと一致する既存LOCAL/A/統合の分類行があるbranchは778件。801 tipのうち利用可能な800固有commit metadataを読み、全履歴walkは行わなかった。Claim–Evidence Matrix43行、既存paper outline・synthesis・研究索引の固定SOURCE/result参照を再利用した。代表の明示roleと重要証拠bindingを区別し、科学的採択・結果順位を新たに変更していない。

| 分類 | 件数 |
| --- | ---: |
| REPRESENTATIVE_KEEP | 5 |
| ACTIVE_KEEP | 0 |
| IMPORTANT_HISTORY_KEEP | 331 |
| REDUNDANT_CANDIDATE_PENDING_REVIEW | 0 |
| UNRESOLVED | 465 |

ここでACTIVE_KEEP 0はB3の終了を意味しない。B3のSERVER_B専用branchは開始時live GitHub branch一覧に存在せず、GitHub表の対象外である。local checkoutと未公開変更の保全を別に保持する。

代表5件は既存の同tip代表roleに基づく `main`、`codex/df-pauli-evidence-paper-plan-20261026`、`codex/h46-long-time-pf-pr-20261025`、`codex/pauli-jw-research-summary-20261026`、`codex/track-a-data-systematization-20261024`。全研究をこの5件へ科学的に統合したという認定ではない。

中間準備・技術履歴のroleがあり後続を検証できていない441行、固定SOURCE/resultの正確なtip参照18件、同tipの相互関係2行を[対応表](BRANCH_EVIDENCE_RELATIONS.csv)へ記録した。18件の索引参照は履歴包含の証明ではない。完了/STOP等は先端commit subjectの記述を区別して表示し、現在の全科学stage成功と呼ばない。類似branch名・同HEAD・祖先関係だけで独自科学資料やprivate custodyが閉じたとは扱わず、branch削減候補は今回0件とした。

[801 branch分類CSV](GITHUB_BRANCH_CLASSIFICATION.csv)には現在SHA、既存系列/role、索引・SOURCE、結果locator、B checkout対応、独自資料・private原本の未確認条件を記録した。公開branch名に含まれるhost tokenも匿名化し、commit URLとSHAによる参照を残した。未取得の固有資料を重複なしと偽認定していない。

## 公開と限界

整理branch開始HEADは `f7c0d3b4dc47a64144eaaf38c1a14da8ef8a0dcf`。指示の基準 `b71c6e9cc074ae51aeeab986600d1b1d6354cbbb` より新しいLOCAL側更新を保持し、既存inventoryは上書きしない。公開直前にもHEADを確認し、A側更新があればその先へ新directoryだけを追加する。2つのLOCAL依存照合結果をSERVER_B自身の削除証拠へ転用していない。

異なるhost/namespace、未記録job・手動再開予定、固定path契約の全解除、private custody・独立backupの全閉包は未検証。最近追加されたtreeの全依存再走査やGitHub801 branchの全固有科学資料監査も未実施である。制約を保留として公開し、削除可能数を水増ししない。

今回の出力は運用調査であり重要科学レビューではない。正式共通指示2026-10-25版の§8.1・§8.4・§10を適用し、sub-agent委任0、科学call0、B3操作0、実削除0。公開commit・local/remote一致・GitHub読み戻しは実測後の最終通知と非公開公開記録で示す。

**次の担当・判断は[NEXT_ACTIONS.md](NEXT_ACTIONS.md)。今回の調査・保全・計画・匿名化公開後はmandatory STOP。**
