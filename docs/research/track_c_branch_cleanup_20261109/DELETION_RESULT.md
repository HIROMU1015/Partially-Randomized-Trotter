# Track C リモートブランチ削除結果（2026-11-09）

**指定21本のリモートブランチを削除し、Track C は25本から4本になった。削除保留0本。**

## 前回保留の解消

ユーザーが候補branchを今後使用しないと明示したため、前回の継続用途未確認は解消した。checkout登録だけを実使用と扱わず、Git履歴・成果の保全と、remote ref削除がローカル原本を変更しないことを確認した。既存の研究契約やSTOPの解除、科学的再計算は行っていない。

21候補の最新tipは監査時から変化しておらず、すべて元の固定研究commit `d6fd10c1c43f3d1220733c9f754534fffeff11f0` と現在の統合保全先から到達できる。原REPORT・SOURCE・raw・結果は元SHAで取得できる。前回監査文書は過去時点の記録として保持する。

## 未追跡原本の保全

- 9,610ファイルは、統合済みRAW_INDEX/RAW_BUNDLESとarchiveのSHA256・元fileのbytes/SHA256を照合して一致。公開archive11件も既存索引と一致。
- 1ファイルは統合先の通常Git blobとbyte一致。
- 残る3,292ファイルは非公開cache、失敗runのpartial原本、lock、旧archiveなどで、元worktreeにそのまま保持する。これらすべてが公開統合済みとは扱わない。
- ローカルbranch・worktree・私有原本を削除・detach・移動・変更していない。remote ref削除後も25 worktreeのtracked/untracked状態は前回記録と同一。

既存 `check_track_c_*` のbyte/hash照合部分とRAW_INDEX/RAW_BUNDLESを再利用し、数値検証・科学module import・再解析は行っていない。元SOURCE bundle4件の検証結果は前回記録を維持する。

## 削除21本

| branch | 削除時tip SHA |
|---|---|
| `codex/track-c2-a-round-pf-pilot-20261106` | `ce84fa8e2d2f1d487df3f05e63d519ebb03f83ad` |
| `codex/track-c2-limited-investigation-20261106` | `6f6c084bdfaf04ba951fd5d8ea0c521313ceb0b3` |
| `codex/track-c-closed-p3-integration-20261108` | `2e38af3b12200c48c4f1b793638d223baff24e76` |
| `codex/track-c-closed-p3-joint-opt-20261108` | `c5a3c102199fe97f4d5c21837c059ae2684f5ef2` |
| `codex/track-c-compiler-a-18round-qualification-20261106` | `3258a89f225d9530fba38e9d2343bc0165dd6d7d` |
| `codex/track-c-compiler-a-correctness-20261106` | `f45e9b07bf730f7272933b28055914cfe76fdaab` |
| `codex/track-c-compiler-static-investigation-20261106` | `05d9c5d2eb984e0f3ce89f44a3f2928685b8d4bf` |
| `codex/track-c-d-internal-20261108` | `b9dacb639eabdd3cf499e89f8807ff50b68e23ef` |
| `codex/track-c-df-cancellation-evidence-20261108` | `12e74dd68568f493ed39937f8856f26ce48d33e1` |
| `codex/track-c-df-cancellation-p1-p2-20261108` | `92495a04d52c65142ed16b3abe977345065f9745` |
| `codex/track-c-df-truncated-pf-20261108` | `da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13` |
| `codex/track-c-diagonal-fusion-20261108` | `94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9` |
| `codex/track-c-guarantee-cn-price-20261106` | `720416c5b88b227cf76ff7aa729d91a0a98ed58b` |
| `codex/track-c-h4-input-readiness-20261106` | `7db4792d4cd0ed9a7abc255565f207dd31a437bd` |
| `codex/track-c-h4-parallel-pilots-20261106` | `a78054dc48737b06b9396e649f53ecb9efd4bd3b` |
| `codex/track-c-joint-isometry-20261108` | `2745d57bea5c28604739bf9c01338e14e649c467` |
| `codex/track-c-mixed-formula-20261108` | `48209b088ca0dffd22a8d3fddcb82888eb29d3a5` |
| `codex/track-c-o3-pilot-plans-20261106` | `66553d948aa9df684987972789c6f3ee8a741312` |
| `codex/track-c-o3-summary-20261108` | `100377dac8635fc49074147bfd8b42f52f2b7311` |
| `codex/track-c-o3-technical-preflight-20261106` | `4b625e5bef06ac51f2b05cafe66f709df2f8bddb` |
| `codex/track-c-resource-bottleneck-20261108` | `10bf5a4813559cea807fcd8c50f3b966407747d1` |

## 保持4本

| branch | 結果文書追記前のremote SHA |
|---|---|
| `codex/track-c-b-connection-survey-20261108` | `4d48a6f2097ea81ff3b7c64114e30bef35b1b685` |
| `codex/track-c-o3-integrated-20261108` | `a8b4c6f02a3cd7dc58c300af11c8b20f9011d11a` |
| `codex/track-c2-b-block-synthesis-pilot-20261106` | `78004943ed5b718c12841e80f55fddf1d1644364` |
| `codex/track-c2-b-strong-baseline-20261106` | `602ca4b67b1037e221e910a2733444630c5add62` |

最終統合branchは、この結果文書2件の追記だけで上表のcommitからfast-forwardする。元研究固定commitと科学treeは変更しない。最終公開SHA・全文byte一致は公開後検証記録と完了チャットに記す。

## 実行と事後検証

削除は明示allowlist21本だけに `git push --atomic --porcelain` と個別の期待SHA leaseを適用し、全21 refの削除が成功した。remote headsは317本から296本、Track Cは25本から4本。残る全refは削除前と同じSHAで、main・独立成果3本・他Trackに意図しない変更はない。削除した21 tipは統合保全先から引き続き到達可能。対応するorigin追跡refだけが削除され、広域pruneは行っていない。

機械可読な結果は [DELETION_RESULT.json](DELETION_RESULT.json)。ローカルの詳細なraw照合・前後ref・削除push結果・公開後照合は `_workspace/04-maintenance/track_c_branch_cleanup_20261109/` に保持する。非公開原本と未追跡file一覧を公開しない。

**今回のリモート整理に関する次の作業なし。公開後照合を終えてmandatory STOP。**
