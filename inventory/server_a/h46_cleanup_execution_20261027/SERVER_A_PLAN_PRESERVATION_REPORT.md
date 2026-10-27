# H46 PLAN保全・6参照の位置付け

2026-10-27 JST。保持対象の新worktreeとPLANは変更せず、旧H46専用worktree0038/0039/0040/0041だけを解除した。前回 `5916f4baac80b1180c6f40c57a9dcd4e0aadb694` で集合保留とした6参照に対して、今回の明示指示に従い保全・取得可能性・現在の実行依存を限定確認した。

## PLANの保持

保持branchは `codex/pr-boundary-phase-a-20261027`、観測HEADは `a4373f51d7acb5d70fda894b4ad66f8bc781177d`。PLANはこのcommitのGit blobと作業fileが同一bytesで、branch/保持refも存続した。削除前後でPLAN本文・branch・HEAD・新worktreeのclean状態は不変。本人専用領域へ同一bytesのPLAN追加copy1件を作り、SHA一致・元との非hardlinkを確認した。本文・内部絶対path・hash値・復旧先は公開しない。

PLANの保存statusは `PHASE_A_FIXED_AWAITING_USER_EXECUTION_PERMISSION`。現在の境界研究の実行設計にもPhase B未承認・mandatory STOPが記載されている。古いPLANのlocatorを現在実行中の契約と自動認定せず、本人processのcwd/引数と既知controller lockも確認した。旧4pathへの可視process一致0、既知controller lockのOS保持未検出。未知namespaceや他所有者の全jobの不存在を主張するものではないが、今回確認範囲で旧0039を必須とする現在有効なjob/bindingは検出しなかった。

## 6参照の確認

| 参照 | 元の役割 | 保全・取得確認 |
| --- | --- | --- |
| H4入力receipt | 0039配下の入力宣言locator | 宣言SHA・元bytes・保持Git blob一致。新保持worktreeにも同一版 |
| H4入力snapshot | 0039配下の入力locator | 同上。配列の科学計算はしない |
| H6入力receipt | 0039配下の入力宣言locator | 同上 |
| H6入力snapshot | 0039配下の入力locator | 同上 |
| H6 snapshot receipt | 0039配下の入力宣言locator | 同上 |
| 公開結果出力root | draft/過去実行の公開出力先locator | 元0039の結果16ファイルはGit blobで保持。新保持worktree側の16結果もGitと実bytesを照合 |

入力5件はPLAN宣言SHA-256、元0039のfile bytes、元commitのGit blob、新保持worktreeのfile/同一blobを個別に照合した。元0039は唯一の供給元ではない。snapshotのpickle/array内容を数値ロード・再解析していない。

結果は単なるdirectory名の有無で判断せず、元16fileの版を保持Git blobとbytesで確認した。新保持worktreeの結果16件もそのcommitのblobと実bytesが一致した。古い結果版と新しい保持版はそれぞれのcommit/blobで区別し、後続結果が旧cohort/失敗/STOPを置換するとは扱わない。

## 削除後の将来利用制約

PLANの本文はそのままなので、旧0039を指す絶対pathは削除後に存在しない。したがって、**保存PLANを現在のまま即時実行可能とは認定しない**。

- 入力5件・結果の各版・科学SOURCEは保持worktreeまたは完全commit/Git blobから取得できる。私有の復旧索引に旧pathと元相対階層・ref・file hashを残した。
- 古いpathを変えずにPLANを将来使うなら、別途指示後に保持refから元pathのworkspaceを再作成する必要がある。今回、再作成・restore・旧copyの上書きはしていない。
- 新しいworkspace/pathを使う実行設定・科学契約への変更が必要なら、別の具体的な許可・固定条件の確認後に扱う。PLAN/SOURCEを無断書換えして削除gateを通していない。
- 0041の古い私有copy5件は更新前の歴史snapshot。最新5件は保持Git blobを復旧元とし、古いcopyを最新fileへ戻さない。

この制約を明示し、PLAN保存と古い絶対pathの永久維持を分離できたため、4件を通常removeした。PLANの実行・Phase B・新科学計算は今回の許可に含めず、開始していない。mandatory STOP。
