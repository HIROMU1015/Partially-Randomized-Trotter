# 1週間以上前のbranch：成果保全確認後の削除（2026-11-09）

**古いtipを持つ10本を確認し、1本を削除、9本をHOLDとした。全remoteは297→296本。** 今回のユーザー指示に基づく実行であり、前回の「監査のみ」のSTOP後に自動削除したものではない。

更新日の作業上の基準は先端commitのcommitter日時。開始時点の7日前は `2026-11-02T18:29:23+09:00`。GitHubのref最終push日時とは区別する。古いcommitを指していても、既知の削除後再出現・具体的な未公開差分・固定参照の問題があれば保持した。枝名の日付は判定に使わない。

## 削除した枝と結果の整理先

- branch：`track-a-ax2b-h4-post-review-20261010`
- 削除前full SHA：`054bb7b2b2ca788b47ebdc5c14c115f24c8b6379`、committer日時 `2026-10-11T21:35:57+09:00`。
- 結果：main固定SHA `546bca7ed2cece59b714b986a264d7b0b0871a01` の `artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/remote_result_verification_launch_v2.json` が元branchのblob `e58e76588951468cbbf3d4b39182f9119dffa5d9` と同一。
- 全履歴：既存tag `refs/tags/archive/research-map-20261102/0dd5c3d204da`、tip `0dd5c3d204dab9b731268a1aa2add621080f3d20`。元tipの祖先到達を削除前後に確認。新しいarchive tagは作成していない。
- 結果の案内：既存mainの `docs/research/RESEARCH_RESULTS_INDEX.md` → `branch_archive_20261102/README.md` → `branches.csv` で、元branch/full SHA・結果root・資料path・保存tagを参照できる。今回、削除条件を作るための新しい科学的統合は行っていない。
- 現ホストの主repositoryと既知の別6 repositoryのworktree登録を再確認し、このbranchをcheckoutしている結合はなかった。今回の年齢条件の指示を一般的な将来用途不明のHOLDに適用し、具体的なSOURCE/未公開差分/実在checkoutの保護は保持した。

## 10本の結果

| branch | 削除前/保持SHA | tip committer日時 | 結果 | 理由 |
|---|---|---|---|---|
| `all-r-coherent-opt2-reoptimization` | `2c865c66171fc9d1972b67c1257c57e5684fe26c` | 2026-10-27T23:17:39+09:00 | HOLD | 前回LOCAL台帳に未push/分岐・worktree保持理由。現LOCALの未公開履歴と継続利用が未確認。 |
| `codex/h46-long-time-b3-20261026` | `ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed` | 2026-10-28T11:45:31+09:00 | HOLD | local/remote各1 commit分岐。SOURCE contextと未push側を保持。 |
| `pr2-v4-s2-parallelization-20260928` | `b2e1bf65e21893b6c617223b42313623d3186f12` | 2026-10-09T15:50:17+09:00 | HOLD | 現行input_contractが当該名の他環境worktree内の固定科学入力を参照。原本・利用状態の確認が必要。 |
| `track-a-ax2b-h4-post-review-20261010` | `054bb7b2b2ca788b47ebdc5c14c115f24c8b6379` | 2026-10-11T21:35:57+09:00 | DELETED_VERIFIED | 結果blobがmainに同一収録、既存archiveに全履歴・索引あり、現地checkout結合なし。 |
| `track-b-g6-w2c-b3-closed-port-bundle-integration-20261019` | `62967269e611b61e73b70abe20e998d9c7eaaf50` | 2026-10-19T23:33:20+09:00 | HOLD | 既存verify_publication.pyまたはverify_fixed_references.pyが実remote branch名/HEADを照会。固定参照の扱いが未解決。 |
| `track-b-g6-w2c-b3-limited-pilot-16896-publication-20261020` | `f27e63b78518c9e807b91f0028fd29078439ec4f` | 2026-10-20T13:23:11+09:00 | HOLD | 既存verify_publication.pyまたはverify_fixed_references.pyが実remote branch名/HEADを照会。固定参照の扱いが未解決。 |
| `track-b-g6-w2c-b3-real-api-adapter-contract-closed-v1-20261020` | `80b7bf61fe0a735d4031bb719c4b3e788611c3ba` | 2026-10-20T01:22:02+09:00 | HOLD | 既存verify_publication.pyまたはverify_fixed_references.pyが実remote branch名/HEADを照会。固定参照の扱いが未解決。 |
| `track-b-g6-w2c-b3-trace-reservation-accounting-repair-20261019` | `596781daeba253418ba35f4b7964312a54caf233` | 2026-10-19T15:48:58+09:00 | HOLD | 既存verify_publication.pyまたはverify_fixed_references.pyが実remote branch名/HEADを照会。固定参照の扱いが未解決。 |
| `track-b-g6-w2c-b3-v2-independent-acceptance-server-20261019` | `77126767c00585d0967558c0756907df2d69d1ff` | 2026-10-19T17:15:44+09:00 | HOLD | 旧削除名が再出現。tip日付をref更新日と同一視しない。既存verify_fixed_references.pyも実remote名を要求。 |
| `track-b-g6-w2c-b3-v2-production-bundle-bridge-candidate-20261019` | `d24cccbecdf0928fe977080dd7b28c591df0089e` | 2026-10-19T19:18:07+09:00 | HOLD | 既存verify_publication.pyまたはverify_fixed_references.pyが実remote branch名/HEADを照会。固定参照の扱いが未解決。 |

9本は年齢だけで削除しない。Track Bの保存済み検証コードには `git ls-remote ... refs/heads/<branch>` の実照会・HEAD検査が残る。これを履歴資料としてどう扱うかを定めず、参照先を消す操作は保留した。pr2のworktree pathは現行固定入力contextに使われる。元コード・契約・STOPを削除に合わせて変更していない。

## 実行と検証

既存 `retire_archived_remote_branches.py` の `verify_origin`、`remote_refs`、`deletion_arguments`、`run` を再利用し、旧352候補runnerの実行はしていない。明示1 refだけをatomic pushと期待SHA leaseで削除した。削除実行は `2026-11-09T18:34:04.196489+09:00`〜`2026-11-09T18:34:06.740112+09:00`、終了コード0、直後の実GitHub refは不在。保持tagは元SHAで残り、元tipは引き続き祖先である。

削除中の対象外ref変更0件。main、その他研究branch、全103既存tag、元local branch/worktree・未追跡/ignored原本、実行中job・SOURCE・入力・結果・grant/ledger/marker/STOPを変更していない。GitHub branch protectionフラグはfalse、open PRとrulesetは観測上0件。科学計算・数値再解析・回路compile・詳細科学レビューは0件。

[全10本・実行と保全確認のJSON](results.json)を参照。親の297本分類表は削除前の履歴snapshotとして保全し、この追補の296本と区別する。本追補を既存研究C整理branchへ文書だけ公開し、公開bytes・実ref・保全先を再照合する。

## 次の担当とSTOP

- ユーザー：残る9本も整理する場合、LOCAL未公開差分の所在・pr2固定入力worktreeの利用状態を提供し、Track Bのbranch名依存を固定SHA/archive参照へ改める範囲を判断する。
- Codex：上記の事実と必要な参照変更の範囲が定まった後、今回の年齢条件・成果保全条件の許可内で残りを再判定する。同じ削除許可を重複要求しない。今回自動で固定契約・科学SOURCEを変更しない。

**公開後照合を終えてMANDATORY STOP。今回の条件で確認できた削除作業は完了。**
