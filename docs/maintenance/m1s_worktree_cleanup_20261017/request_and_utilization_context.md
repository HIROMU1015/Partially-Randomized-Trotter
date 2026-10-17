# 条件付き整理の承認範囲と利用終了確認

2026-10-17 JST。このファイルは管理作業の来歴記録であり、科学実行authorization・grant・markerではない。

## 今回の依頼

ユーザーは、監査commit `893da184e74e682dfb1bc97102c61bbe8ce2895e` にあるC01–C31について、
**安全性を直前に再確認できたworktreeだけ**、通常の `git worktree remove` で削除することを承認した。
31件すべてを削除する必要はなく、条件を満たさないものは保留する。

明示された条件には、HEADとremote公開状態の一致、staged／unstaged／untrackedの未保存データなし、
ignoredを含む消失内容の確認、indexの安定、Git lock・実行中jobなし、他Codex・利用者の不使用、
SOURCE・raw・契約・authorization・grant・ledger・markerの保全が含まれる。

- 全42 M1-S branchを保持し、削除・変更しない。
- P01–P03、H01–H08は削除しない。Hの再分類もしない。
- 他研究のbranch/worktreeには手を加えない。
- `--force`、prune、clean、強制削除、branch削除・rename、merge、rebase、reset、checkout変更は禁止。
- 想定外の状態を独断で修正して続行しない。
- 管理記録のみを独立作業領域からcommit・pushし、公開同一性を確認してmandatory STOP。

## 利用状況の追加確認

旧監査README §2にある利用者・他Codexの利用終了確認について、process情報だけでは
idle editorやremote sessionの不使用を確定できないため、利用状況の事実確認を行った。

ユーザーへの質問：

> C01–C31を、現在ほかのCodex・ターミナル・エディタで利用していないと確認できますか？

ユーザーの回答原文：

> C01–C31はすべて利用終了済み

この回答により利用終了の確認を得た。ただし、未追跡ファイルなど他の削除条件を解除するものではない。
今回の質問は既に与えられた削除承認の再申請ではない。

## 実施した判断

新しいpreflightで全31候補に、旧監査では0件だった未追跡項目 `.agent/docs` と `CLAUDE.md` を確認した。
どちらも親作業領域を指すsymlinkだが、未追跡のままであり、ユーザーの削除条件を満たさない。
生成主体・生成目的は帰属しない。link先の内容を開いたり削除したりしていない。

**全31候補を保留。worktree removeの試行0件、削除0件。**
symlinkを別途取り除く、ignoreに追加する、強制削除する、といった修正は行わなかった。
新しい科学計算、SOURCE修正、科学結果の再解析、科学実行authorization／grant／markerの発行も行っていない。
