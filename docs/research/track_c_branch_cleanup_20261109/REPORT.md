# Track C ブランチ保全監査・整理結果（2026-11-09）

**削除0本。21候補すべて HOLD_UNVERIFIED、明示保持4本を含む25本を保持した。**

研究保全先は `codex/track-c-o3-integrated-20261108`、元の固定研究commitは `d6fd10c1c43f3d1220733c9f754534fffeff11f0`。21候補の最新remote tipはすべてこのcommitの祖先。Git管理の原REPORT・SOURCE・入力・raw・結果・STOPは、その元commitを指定して統合履歴から取得できる。元研究commitや科学ファイルは変更しない。

## 削除を保留した理由

21候補はすべて、このホストで専用worktreeにcheckoutされている。7候補には計12,903件の未追跡ファイルがあり、公開archiveとの全原本一致と継続用途は未確認。cleanな14候補についても利用終了を確認できていない。既存整理方針の `HOLD_WORKTREE_OR_LOCAL_DIFFERENCE` と今回の「利用中・使用状況不明は保持」を適用した。worktreeの削除・detach・job停止で削除条件を作っていない。

確認環境は `GPUA100-gleap02`。可視の自分のUIDのprocessをpath/cmdlineで読取確認し、監査自身を除く関連processは観測しなかった。scheduler、他ユーザー、別サーバー・別clone・他チャットの現在の利用状況は確認していない。SSH設定・SSHディレクトリ・SSH agentがなく、SERVER 1/2全体を確認済みとはしない。

## 保全確認

- 最新25 refを `git ls-remote --heads origin` で再取得し、この25本だけをfetchした。21/21の `git merge-base --is-ancestor` が成功。候補からのみ到達するcommitは0件。
- 固定統合commitから到達する68,008 Git objectに欠落0件。sparse checkoutで原SOURCEやarchiveが未展開でも、Git blobから取得して確認した。
- Closed P3、共同最適化、DF比較、P1/P2の `SOURCE_HISTORY.bundle` は一時領域に元blobを読み出し、4/4で `git bundle verify` 成功。公開snapshotと元科学SOURCE commitの区別を維持した。
- 既存のRAW_INDEX・分割archive・SOURCE・原bundleのblob一覧を [AUDIT.json](AUDIT.json) に記録。Git管理外のcommon.pkl、common_arrays.npz、ground_reference.npzは既知の元worktreeに存在することを確認し、変更・公開していない。全rawの再hashや新しい数値処理は行っていない。
- 42件のTrack C文書の保全文言と既存管理分類を確認。検索範囲内でbranch存続を要求する固有KEEP/HISTORY KEEP契約は見つからなかったが、これは契約解除を意味しない。SOURCE/raw/grant/ledger/markerと過去のSTOPを維持した。
- 統合snapshotの文書・コード・CI・索引を候補名とURLエンコード名で検索し、15行の歴史的branch表記を確認。検索範囲でbranch URLや実行時照会への依存は見つからなかった。全外部依存を確認済みとはせず、文書を書き換えていない。

## 21候補の分類

全件 `HOLD_UNVERIFIED`。先端SHA・REPORT/SOURCE/raw等のblobと理由コードは [AUDIT.json](AUDIT.json) を参照。

| branch | tip SHA | 未追跡件数 |
|---|---|---:|
| `codex/track-c-closed-p3-integration-20261108` | `2e38af3b12200c48c4f1b793638d223baff24e76` | 0 |
| `codex/track-c-closed-p3-joint-opt-20261108` | `c5a3c102199fe97f4d5c21837c059ae2684f5ef2` | 0 |
| `codex/track-c-compiler-a-18round-qualification-20261106` | `3258a89f225d9530fba38e9d2343bc0165dd6d7d` | 0 |
| `codex/track-c-compiler-a-correctness-20261106` | `f45e9b07bf730f7272933b28055914cfe76fdaab` | 0 |
| `codex/track-c-compiler-static-investigation-20261106` | `05d9c5d2eb984e0f3ce89f44a3f2928685b8d4bf` | 0 |
| `codex/track-c-d-internal-20261108` | `b9dacb639eabdd3cf499e89f8807ff50b68e23ef` | 2032 |
| `codex/track-c-df-cancellation-evidence-20261108` | `12e74dd68568f493ed39937f8856f26ce48d33e1` | 0 |
| `codex/track-c-df-cancellation-p1-p2-20261108` | `92495a04d52c65142ed16b3abe977345065f9745` | 0 |
| `codex/track-c-df-truncated-pf-20261108` | `da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13` | 0 |
| `codex/track-c-diagonal-fusion-20261108` | `94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9` | 801 |
| `codex/track-c-guarantee-cn-price-20261106` | `720416c5b88b227cf76ff7aa729d91a0a98ed58b` | 0 |
| `codex/track-c-h4-input-readiness-20261106` | `7db4792d4cd0ed9a7abc255565f207dd31a437bd` | 2 |
| `codex/track-c-h4-parallel-pilots-20261106` | `a78054dc48737b06b9396e649f53ecb9efd4bd3b` | 6713 |
| `codex/track-c-joint-isometry-20261108` | `2745d57bea5c28604739bf9c01338e14e649c467` | 1555 |
| `codex/track-c-mixed-formula-20261108` | `48209b088ca0dffd22a8d3fddcb82888eb29d3a5` | 191 |
| `codex/track-c-o3-pilot-plans-20261106` | `66553d948aa9df684987972789c6f3ee8a741312` | 0 |
| `codex/track-c-o3-summary-20261108` | `100377dac8635fc49074147bfd8b42f52f2b7311` | 0 |
| `codex/track-c-o3-technical-preflight-20261106` | `4b625e5bef06ac51f2b05cafe66f709df2f8bddb` | 0 |
| `codex/track-c-resource-bottleneck-20261108` | `10bf5a4813559cea807fcd8c50f3b966407747d1` | 1609 |
| `codex/track-c2-a-round-pf-pilot-20261106` | `ce84fa8e2d2f1d487df3f05e63d519ebb03f83ad` | 0 |
| `codex/track-c2-limited-investigation-20261106` | `6f6c084bdfaf04ba951fd5d8ea0c521313ceb0b3` | 0 |

## 明示保持4本

| branch | 監査時tip SHA | 理由 |
|---|---|---|
| `codex/track-c-b-connection-survey-20261108` | `4d48a6f2097ea81ff3b7c64114e30bef35b1b685` | 独立成果、統合先にないcommit 1件 |
| `codex/track-c-o3-integrated-20261108` | `d6fd10c1c43f3d1220733c9f754534fffeff11f0` | 長期統合保全先 |
| `codex/track-c2-b-block-synthesis-pilot-20261106` | `78004943ed5b718c12841e80f55fddf1d1644364` | 独立成果、統合先にないcommit 1件 |
| `codex/track-c2-b-strong-baseline-20261106` | `602ca4b67b1037e221e910a2733444630c5add62` | 独立成果、統合先にないcommit 2件 |

## 実行・事後状態

削除対象確定結果はDELETE_READY 0本、HOLD_UNVERIFIED 21本、KEEP_REQUIRED 4本。remote/local branch削除・remote-tracking ref削除は0本。監査前後のremote headsは316本から316本、Track Cは25本から25本。監査中のremote変更は0件。mainおよび他Trackへの変更操作は行っていない。

既存管理資料の隣 `_workspace/04-maintenance/track_c_branch_cleanup_20261109/` に、before/after ref一覧、25 worktree状態、可視process観測、保全資料の抽出、bundle検査結果を保存した。原本や未追跡ファイル名の全一覧は公開しない。このREPORTとAUDIT.jsonだけを既存の最終統合branchへfast-forwardで公開し、公開後にremote SHA・2文書の全文byte・元treeとの差分と保全refを照合する。最終公開SHAは公開後の検証記録と完了チャットに記す。

## 次の担当とSTOP

ユーザー：削除を続ける場合は、21 worktreeの継続用途・SERVER 1/2等での利用終了、7 worktreeの未追跡原本と非公開cacheの保全状況を確認する。今回の削除許可はそのまま有効であり、重複の削除承認は不要。

Codex：上記の事実確認が揃った後の再開指示で、同じ21候補の最新tipと全削除条件を再確認し、適格な候補だけを整理する。今回自動再開しない。新科学計算・科学レビュー・他Track整理・mainへのmergeは実施していない。**公開後照合を終えてmandatory STOP。**
