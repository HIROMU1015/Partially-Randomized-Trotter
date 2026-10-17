# M1-S worktree条件付き整理：agent用ignore反映後の完了記録

2026-10-17 JST。**C01–C31の31件を通常の `git worktree remove` で削除。保留0件。全branch保持。公開検証後mandatory STOP。**

repository：`HIROMU1015/Partially-Randomized-Trotter`。
固定監査：[893da184e74e682dfb1bc97102c61bbe8ce2895e](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/893da184e74e682dfb1bc97102c61bbe8ce2895e/docs/maintenance/m1s_worktree_audit_20261017)。
前回は未追跡link2項目を理由に削除0件・31件保留とした。
その[記録da4dd39d5197c16592bc870de04145135d82b6c3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da4dd39d5197c16592bc870de04145135d82b6c3/docs/maintenance/m1s_worktree_cleanup_20261017/README.md)は当時の事実として維持する。
今回はユーザーがエージェント用に追加した項目と説明し、ignoreを指示したため、設定後に全候補を再検査して整理した。
旧監査・前回報告・科学成果の内容は上書きしていない。

## 1. 実施結果

| 確認項目 | 結果 |
|---|---|
| 削除候補C01–C31 | 31件全て削除成功、保留0件 |
| 削除方法 | 候補ごとに通常の `git worktree remove -- <固定絶対path>` を1回。forceなし |
| C31 | 既知のignored bytecode4件を含むcheckoutを通常removeで除去。追加のcache削除操作なし |
| 元repositoryのworktree | 217件 → 186件。削除した31候補だけが登録から消失 |
| M1-S branch | 全42件保持。local／GitHub同名branchのHEADは固定値と一致 |
| 元repositoryの全local branch | 227件の名称・HEADが前後不変 |
| P01–P03、H01–H08 | 全11拠点を保持。HEAD・index・全ローカルfile manifestが前後一致。再分類なし |
| 対象外175 worktree | 登録・branch・HEAD等のmetadataは前後不変。個別content監査・変更なし |
| 親worktree | HEAD・index不変。既存dirty変更をstage／commitしていない |
| 親エージェント用資料 | `.agent/docs` の実file2件のmanifest、`CLAUDE.md` のbytes／SHA256が不変 |
| 候補の追跡file | 削除直前に延べ228,390件をHEAD Git blobと照合、差異0 |
| 管理記録の整合性検査 | 38件PASS・0件FAIL。科学的な数値testではない |
| 科学計算・科学値の再解析・科学SOURCE変更 | 0件 |
| 科学実行authorization／grant／markerの発行 | 0件 |
| branch削除／rename／merge／rebase／reset／checkout変更／force／prune／clean | 0件 |

削除した各ID・branch名・絶対path・固定HEADは[candidate_outcomes.csv](candidate_outcomes.csv)に全31件を記録した。
[execution_record.json](execution_record.json)には、個別のpreflight、直前確認、削除command、開始終了時刻、exit code、
stdout／stderr、path・登録消失の確認を保存した。失敗・retry・部分削除は0件。

残存186件の一覧は[remaining_worktrees.csv](remaining_worktrees.csv)。
内訳はM1-SのP01–P03／H01–H08の11件と、対象外175件。
対象外の行は登録metadataであり、内容・利用状況を個別監査した記録ではない。

## 2. ignore設定と消失内容の確認

ユーザーの追加指示と承認scopeは[scope_and_ignore_change.md](scope_and_ignore_change.md)。
repository共通の**ローカル** `.git/info/exclude` にroot相対の `/.agent/docs` と `/CLAUDE.md` を追加した。
既存excludeを保持して追記し、tracked `.gitignore`、global Git設定、各研究branchの追跡fileは変更しない。
この設定はこのrepositoryのcommon Git directoryを使うworktreeに適用され、GitHubや別cloneへ自動配布されない。

ignore自体を保全保証とは扱わない。全31候補で、両pathがそれぞれ親領域の同名pathを指すsymlinkであることを確認した。
通常removeで候補側のlinkだけを除去し、親側の実directory・実fileは保持した。
親側の内容manifest／hashとexclude設定のhashを、削除前後および個別削除の間で比較し不変を確認した。
その他のignored項目はC31の既知のPython bytecode4件だけを許可し、旧監査のpath・SHA256と一致した。
bytecodeの対応SOURCEは追跡済みで、全tracked fileのGit blob照合にも含まれる。

[local_ignore_verification.json](local_ignore_verification.json)はignore直後・削除前の確認イベントである。
このイベントにある `worktrees_removed=0` は当時の状態であり、本資料の最終結果31件と矛盾しない。
前回の31件保留記録と同様、時間の異なる記録を同一時点の結果として混ぜない。

## 3. 削除直前の安全確認

候補は並列に削除せず、個別preflight → 通常remove → 保全確認の順に処理した。
逐次依存と衝突の検出を優先する管理操作であり、科学計算のworker制限ではない。

- 固定CSVのID・branch名・絶対path・HEADと、現在のHEAD／symbolic ref／worktree登録を照合。
- 候補ごとに実GitHub同名branch HEADを `git ls-remote` で再照会し、固定値との一致を確認。
- staged／unstaged／untrackedは0。ignoredはユーザー指定の2 linkとC31の既知cache4件だけ。
- 全tracked fileの実bytesをGit blobと照合し、特殊index flag・特殊file・入れ子repositoryの問題なし。
- hash照合・remote照会の前後でindexが安定。削除直前にもindex／HEAD／ref／lock／未追跡／ignored inventoryを再確認。
- Git lock・worktree lockなし。読み取り可能な同UID process metadataに候補path参照なし。
- ユーザーのC01–C31利用終了確認を継続して適用。process情報だけによる全session不使用の証明とは表現しない。
- `sd-pam`／`sshd`の既知の4 processはcwd／fdにPermissionErrorがあることを記録した。
  新たな読み取り不能workerや候補への参照があれば保留する設計とし、他UIDのprocess内容には立ち入らない。
- 各削除後に全local refs、残存登録、親エージェント資料を再照合し、想定外の変化がないことを確認。

既存worktreeへの照会は `GIT_OPTIONAL_LOCKS=0`、`GIT_NO_LAZY_FETCH=1` を適用した。
無視設定の変更と承認された通常remove以外の修正・強制続行は行わない。
安全確認は観測時点の情報であり、未来の外部process動作まで排除する証明ではない。

## 4. 保全・provenanceと証拠索引

全42 M1-S branchの固定HEAD・local／remote前後値は[preservation_checks.json](preservation_checks.json)。
worktreeの除去はcheckoutの除去であり、branchとcommit historyを保持している。
既存SOURCE、raw、科学契約、authorization、grant、ledger、消費済みmarker、レビューへの到達可能性を維持する。
過去の失敗・null・未解像結果やone-shot消費状態を修正していない。
hash照合は保全検査であり、新しい独立科学再現や数値correctness testではない。

主要anchorの保全拠点：

- P01 Stage S：`f2bfb8770e6858de56a63e50e68f07ff2d6464b6`。
- P02 統合成果：`5f079dbe67bb35eff35613857b169fdf812af4b8`。
- P03 別履歴の先行研究調査：`4301c0f87a16be45b620304c3fa935d5aa2d721a`。

C23の固有の旧独立correctness準備SOURCEも、checkout除去後はbranch
`track-b-m1s-gate0-v5-independent-correctness-preparation-20261016` の
`d6d35cfb79b97f9cb218c9169b427aacf6a4a334` で保全する。
主要anchorだけに集約したことにはしない。

| 証拠 | 内容 |
|---|---|
| [execution_record.json](execution_record.json) | 31件の個別検査・通常remove・終了確認 |
| [candidate_outcomes.csv](candidate_outcomes.csv) | 削除済み31件のID・branch・path・固定HEAD |
| [remaining_worktrees.csv](remaining_worktrees.csv) | 残存186件の登録metadata |
| [before_snapshot.json](before_snapshot.json)／[after_snapshot.json](after_snapshot.json) | 全local refs／登録／42 remote HEADとP/H file manifest digest |
| [parent_agent_before.json](parent_agent_before.json)／[parent_agent_after.json](parent_agent_after.json) | 親側のエージェント用資料とexcludeのtype・hash |
| [preservation_checks.json](preservation_checks.json) | 全42 branch、P/H、対象外、親worktreeの保全確認 |
| [verification.json](verification.json) | 管理記録の整合性検査 |
| [publication_receipt.json](publication_receipt.json) | 固定payloadのremote HEAD・公開bytes・SHA256照合 |

P/H全11拠点ではtracked／untracked／ignored／symlinkを含む延べ85,212 fileの前後manifestを比較した。
Git adminはindexとlockを別記し、symlinkはtarget文字列を記録して先を追跡しない。
公開snapshotには各拠点の件数・canonical manifest SHA256とcanonicalization規則を保存した。
完全manifestは独立した `/tmp/m1s-cleanup-agent-ignore-before-full-20261017.json`／
`/tmp/m1s-cleanup-agent-ignore-after-full-20261017.json` に保存した。これは科学証拠の唯一の保管先ではない。
対象外研究のcontent全体を監査したという主張はしない。

## 5. 公開領域と停止

管理成果のみを独立clone `/tmp/prt-m1s-worktree-audit-20261017` の管理branch
`track-b-m1s-worktree-audit-20261017` に新directoryとして追加する。
この管理branchは元repositoryの固定42 M1-S branchに含まれない。
既存管理報告を上書きせず、今回の追加指示と前回保留との関係を記録した。
`.git/info/exclude` 自体はローカル設定のためcommit対象ではなく、変更内容と検査結果を公開する。

**今回承認された整理は完了。追加作業は不要としてmandatory STOP。**
全branch保持、P/H保持、H再分類禁止を維持し、追加削除・branch整理・新規研究段階へ自動移行しない。
今後別の整理を希望する場合の担当はユーザーであり、具体的範囲をあらためて指定する。
