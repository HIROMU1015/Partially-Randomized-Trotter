# エージェント用ローカル設定のignore指示と整理scope

2026-10-17 JST。これは管理作業の来歴記録であり、科学実行authorization・grant・markerではない。

## 以前の条件付き削除承認

ユーザーは、監査commit `893da184e74e682dfb1bc97102c61bbe8ce2895e` のC01–C31について、
直前の安全性確認を満たしたworktreeだけ、通常の `git worktree remove` で削除することを承認済み。
全42 M1-S branchは保持し、P01–P03とH01–H08および他研究のworktreeは削除しない。
force／prune／clean／branch削除・rename／履歴変更／checkout変更は禁止。

利用状況について、ユーザーは次のように確認した。

> C01–C31はすべて利用終了済み

前回は全31候補に未追跡 `.agent/docs` と `CLAUDE.md` が見つかったため、削除せず保留した。
当時の管理記録はcommit `da4dd39d5197c16592bc870de04145135d82b6c3` に固定されており、上書きしない。

## 今回の追加指示原文

ユーザーは上記2項目を指定し、次のように説明・指示した。

> これは新しく私がこのプロジェクトでのエージェント用に追加したものなので gitignore か何かで無視してくれればよい

この指示に基づき、repository共通のローカルexcludeだけに、次のroot相対patternを追加した。

```gitignore
/.agent/docs
/CLAUDE.md
```

設定先：`/home/abe/Project/Partially Randomized Trotter/.git/info/exclude`。
既存excludeのbytesを保ち末尾へ追記した。tracked `.gitignore`、各branchの追跡ファイル、global Git設定は変更しない。
末尾 `/` を付けないことで、親worktreeのdirectory／fileと候補worktreeのsymlinkの両方が対象になる。
scopeはこの2つのroot相対pathのみで、`.agent`全体や全階層のCLAUDE.mdを一律除外しない。
common Git directoryを共有するworktreeへ効くローカル管理設定であり、cloneやGitHubへ設定自体が配布されるものではない。

31候補全てで `git check-ignore` を確認し、設定直後のstaged／unstaged／untrackedは0件だった。
ignoreは内容の保全を保証する仕組みではない。削除前には次を別途確認した。

- 候補の `.agent/docs` と `CLAUDE.md` は、いずれも親worktree内の同名pathを指すsymlinkである。
- 親側の実directoryと実fileのmanifest／hashを取得し、候補削除の前後および各削除間で不変を確認する。
- 候補側のlinkのtype・targetは直前に再確認し、link先を削除する操作は行わない。
- その他のignored pathは許可しない。例外は既存監査に列挙されたC31のbytecode4件だけで、旧hashとの一致を確認する。

## 整理の承認境界

元の条件付き削除承認と利用終了確認は継続し、今回の指示で明示された2項目のignoreを反映して再検査する。
削除範囲はC01–C31から拡張しない。各候補を個別に検査し、通常のremoveが拒否した場合はretry・強制修正せず保留する。
全branch、P/H、他研究の保全条件を解除しない。Hの再分類もしない。

科学SOURCE・raw・科学契約・authorization・grant・ledger・markerの内容変更、新規科学計算、simulation、compileは行わない。
管理記録の公開・検証を終えたらmandatory STOP。新たな整理や研究段階へ自動移行しない。
