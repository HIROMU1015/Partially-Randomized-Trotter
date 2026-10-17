# M1-S関連branch・worktree：読み取り専用監査と整理計画

2026-10-17 JST。**監査のみ。削除・prune・既存branch変更・研究実行の承認はない。mandatory STOP。**

対象repository：`HIROMU1015/Partially-Randomized-Trotter`。
基点：[M1-S統合成果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/5f079dbe67bb35eff35613857b169fdf812af4b8)。
最初に[研究サマリー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md)、
[証拠索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/evidence_and_results_index.md)、
[再開ハンドオフ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/codex_restart_handoff.md)を確認した。
[共通指示2026-10-17 §8.4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/research_project_instructions_unified_2026-10-17.md)も適用する。

## 1. 結果と推奨する整理範囲

| 区分 | 件数 | 推奨 |
|---|---:|---|
| 削除候補 | 31 | 後日の明示承認と直前再確認後にworktreeだけを除去できる候補。branchは全て保持 |
| 保全対象 | 3 | 統合成果・Stage S・別履歴の先行研究調査の3拠点を残すことを推奨 |
| 判断保留 | 8 | 監査中のindex bytes変化を観測。利用者・変更理由・安定性が未確定のため除去しない |

全体は227 local branch・217登録worktree。そのうちM1-S名を発見の手掛かりにした42 branch・42 worktreeだけを個別監査した。
各branchの直近commitにM1-S専用SOURCE、artifact、契約または文書の変更があることを確認し、名前だけで関連性を確定していない。
その他185 branch・175 worktreeは対象外で個別監査していない。
全branchの名称・最新subjectという発見用metadataも照合し、M1-S名を含まない追加候補は見つからなかった。関連性未確定のものは除外を維持し、他研究の内容を個別調査していない。`track-b-*`全体を対象にしていない。

**42 branchのHEADは、読み取り時点のGitHub同名branch HEADと全て一致。全42 branchはlocal mainに未merge。branch削除の提案は0件。**
upstream設定がないbranchも、実remoteのHEADで公開状態を確認した。tracking設定の欠如を未公開と扱わない。
remoteだけに存在する追加のM1-S名branchは見つからなかった。

追跡M1-Sファイルと保存入力について、worktree内容とHEAD Git blobを延べ15,125件照合し差異0。
全対象でstaged/unstaged/untrackedは0、特殊index flag（assume-unchanged/skip-worktree等）は0。
ignoredはStage S準備のPython bytecode4ファイルのみで、対応する追跡済み.pyを確認した。科学結果・SOURCE・契約・markerの唯一の保存先とは扱っていない。
この4ファイルの除去も未実施で、候補の影響欄に記録する。

候補一覧は[candidate_worktrees.csv](candidate_worktrees.csv)、全件は[all_worktrees.csv](all_worktrees.csv)。
機械可読計画は[proposed_cleanup_plan.json](proposed_cleanup_plan.json)で、`deletion_authorized=false`。
以下のC/P/H IDとHEADを、将来の承認対象の識別に使える。現時点で実行しない。

## 2. 監査方法と限界

- 既存worktreeへのGit照会は`GIT_OPTIONAL_LOCKS=0`、`GIT_NO_LAZY_FETCH=1`を付け、index refreshを要求しない。
  HEAD/ref、差分、staged、untracked、ignored、Git lock、特殊index flag、追跡fileのGit blob一致を読む。
- GitHubの公開状態は`ls-remote`による実branch HEAD照会。固定commitの到達関係と、3主要anchorにないbranch固有blobも調べた。
- processは同UIDの読み取り可能な`/proc`のcwd・fd target・明示worktreeパス参照だけを確認した。
  コマンド全文・環境変数・他ユーザーのprocess内容は公開せず、停止/設定変更は行わない。
- 読み取れたprocess metadataには対象worktreeの参照0、Git lock0。ただし4件のPermissionErrorがあり、
  他UIDは対象外。idle editor、remote Codex session、共有daemon、未来のjobまで不使用と証明したものではない。
  **全候補に、削除直前の再監査と利用者・他Codexの利用終了確認を要求する。**
- 8件でindexのbytesが監査中に変化したという観測がある。SOURCEやrawの差異は0、HEADも一致したが、
  原因を他Codexまたは監査コマンドのどちらにも帰属していない。後続の安定したsnapshotだけで保留を解除しない。
  初回終了時index hashの値そのものは保存しておらず、初回before hash・変化検出bool・後続hashを記録した。この限界を隠さない。
- worktreeの最終access時刻やmtimeだけから使用中/未使用を判定していない。全他worktreeの未追跡内容を調べたという主張もしない。
- 科学moduleのimport、raw科学値のparse/再解析、数値test、simulation、compileは0。保全はfile bytes/Git identityの確認。

既存worktree・branchを増減させないため、監査成果は独立clone `/tmp/prt-m1s-worktree-audit-20261017`、
専用branch `track-b-m1s-worktree-audit-20261017`で作成した。このcloneは元repositoryの登録worktreeではなく、上記42件に含めない。
cloneは`--shared`により元repositoryのobject storeを参照する。今回は元object storeの削除/GCやbranch除去を行わない。
長期に元repositoryごと廃棄する計画を作る場合は、cloneのobject依存も別途確認する必要がある。

## 3. 各候補の一覧

S/U/?/Iはstaged/unstaged/untracked/ignoredのファイル数。remoteは全件HEAD一致。
「未保存科学データなし」はこの監査範囲での意味であり、他process/外部領域までの全証明ではない。

### 削除候補

| ID | branch | worktree（絶対パス） | HEAD | S/U/?/I | 推奨処置・理由 |
|---|---|---|---|---|---|
| C01 | `track-b-m1s-finite-pilot-results-scientific-review-20261017` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-finite-pilot-results-scientific-review-20261017` | [`e5ed0db6b8f50ed02ee9bc17cc6d0e2d4ed0898b`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e5ed0db6b8f50ed02ee9bc17cc6d0e2d4ed0898b) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C02 | `track-b-m1s-finite-signal-model-accounting-v2-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-finite-signal-model-accounting-v2-20261016` | [`3e742894e2c79e7e46b14de9801e7e1519062879`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/3e742894e2c79e7e46b14de9801e7e1519062879) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C03 | `track-b-m1s-finite-signal-preparation-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-finite-signal-preparation-20261016` | [`a398a3fab6f68abc2a47799d7454b312082af292`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a398a3fab6f68abc2a47799d7454b312082af292) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C04 | `track-b-m1s-finite-signal-v2-completion-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-finite-signal-v2-completion-20261016` | [`025c5a90888b02b73d2815fbb10d6bd1d2823602`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/025c5a90888b02b73d2815fbb10d6bd1d2823602) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C05 | `track-b-m1s-finite-signal-v2-formal-adoption-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-finite-signal-v2-formal-adoption-20261016` | [`b13a44b3fff56003679f961f05917ed9ca48f702`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/b13a44b3fff56003679f961f05917ed9ca48f702) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C06 | `track-b-m1s-finite-signal-v2-one-shot-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-finite-signal-v2-one-shot-20261016` | [`0bb3fbafb64f1b94b4bf8824f64cfdd7c1689ffe`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/0bb3fbafb64f1b94b4bf8824f64cfdd7c1689ffe) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C07 | `track-b-m1s-g0-contract-preparation-20261015` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-g0-contract-preparation-20261015` | [`0f0952614dcf5c06d3d63c21b5d49091b8ecce37`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/0f0952614dcf5c06d3d63c21b5d49091b8ecce37) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C08 | `track-b-m1s-g0-pre-adoption-survey-20261015` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-g0-pre-adoption-survey-20261015` | [`8b7666d0270f7fac01e430e195bef7b58ac0116c`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/8b7666d0270f7fac01e430e195bef7b58ac0116c) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C09 | `track-b-m1s-g0t1-s0-contract-repair-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-g0t1-s0-contract-repair-20261016` | [`0d84c2181de128e496bdc1e6161264cdf84eeba1`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/0d84c2181de128e496bdc1e6161264cdf84eeba1) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C10 | `track-b-m1s-g0t1-s0-formal-adoption-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-g0t1-s0-formal-adoption-20261016` | [`373b8e4b9831f4d00bdc00ffc741ac61e03f0841`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/373b8e4b9831f4d00bdc00ffc741ac61e03f0841) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C11 | `track-b-m1s-g0t1-t1a-theory-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-g0t1-t1a-theory-20261016` | [`4e7b18dd2aa14c127bce9e248c63996c62a055cc`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/4e7b18dd2aa14c127bce9e248c63996c62a055cc) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C12 | `track-b-m1s-g0t1-t1b-theory-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-g0t1-t1b-theory-20261016` | [`54336028edd7ef759f8a0c73bd0a96ade13f47f9`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/54336028edd7ef759f8a0c73bd0a96ade13f47f9) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C13 | `track-b-m1s-gate0-formula-contract-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-formula-contract-20261016` | [`e749bf8844f5d78817f18f738ade87269c2d70e1`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e749bf8844f5d78817f18f738ade87269c2d70e1) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C14 | `track-b-m1s-gate0-precision-cause-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-precision-cause-20261016` | [`a058ef14d4ea162547ec73b707a4662ec8fcd7d6`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a058ef14d4ea162547ec73b707a4662ec8fcd7d6) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C15 | `track-b-m1s-gate0-precision-contract-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-precision-contract-20261016` | [`4019d045cbfad94ba9333b544f7950a6d10d467d`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/4019d045cbfad94ba9333b544f7950a6d10d467d) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C16 | `track-b-m1s-gate0-source-preparation-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-source-preparation-20261016` | [`83fbac28bca8e308b3039d8532492e6510492f89`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/83fbac28bca8e308b3039d8532492e6510492f89) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C17 | `track-b-m1s-gate0-tail-bound-proof-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-tail-bound-proof-20261016` | [`b5a185cc0c3aae0e477bfc52b93061e86feb7dec`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/b5a185cc0c3aae0e477bfc52b93061e86feb7dec) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C18 | `track-b-m1s-gate0-v4-contract-candidate-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v4-contract-candidate-20261016` | [`44277e03448ee51dbe9afdd2dc6560337757cc6d`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/44277e03448ee51dbe9afdd2dc6560337757cc6d) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C19 | `track-b-m1s-gate0-v4-formal-adoption-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v4-formal-adoption-20261016` | [`5e57955baa79e42f37aa233ac011c60539e431fd`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/5e57955baa79e42f37aa233ac011c60539e431fd) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C20 | `track-b-m1s-gate0-v4-one-shot-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v4-one-shot-20261016` | [`c937ed37e73bb9b0bb39f2fa7116d255a3215269`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/c937ed37e73bb9b0bb39f2fa7116d255a3215269) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C21 | `track-b-m1s-gate0-v5-formal-adoption-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v5-formal-adoption-20261016` | [`e46386abb70d601b973850b82e9bcfb0430ad0ee`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e46386abb70d601b973850b82e9bcfb0430ad0ee) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C22 | `track-b-m1s-gate0-v5-independent-audit-prep-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v5-independent-audit-prep-20261016` | [`2ba0858c3c92b61581cfbf6cdfb5039d25a174c9`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/2ba0858c3c92b61581cfbf6cdfb5039d25a174c9) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C23 | `track-b-m1s-gate0-v5-independent-correctness-preparation-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v5-independent-correctness-preparation-20261016` | [`d6d35cfb79b97f9cb218c9169b427aacf6a4a334`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/d6d35cfb79b97f9cb218c9169b427aacf6a4a334) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C24 | `track-b-m1s-gate0-v5-one-shot-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v5-one-shot-20261016` | [`cdcb7f6f151df217960611f502eb1bac8be63f58`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/cdcb7f6f151df217960611f502eb1bac8be63f58) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C25 | `track-b-m1s-gate0-v5-source-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v5-source-20261016` | [`47da54279838a6b3e1497f36204ba7e53c4857f9`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/47da54279838a6b3e1497f36204ba7e53c4857f9) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C26 | `track-b-m1s-independent-contract-adoption-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-independent-contract-adoption-20261016` | [`0956fd477a54a7f4f0570944a1265e0c27cb3fbc`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/0956fd477a54a7f4f0570944a1265e0c27cb3fbc) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C27 | `track-b-m1s-independent-one-shot-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-independent-one-shot-20261016` | [`cc2f451324b7292ae50cef94ff794d46f225d892`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/cc2f451324b7292ae50cef94ff794d46f225d892) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C28 | `track-b-m1s-post-checkpoint2-preflight-20261016` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-post-checkpoint2-preflight-20261016` | [`05513eea807fbfb841c6012d2fe2133d70b5f1f9`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/05513eea807fbfb841c6012d2fe2133d70b5f1f9) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C29 | `track-b-m1s-rpe-feasibility-conditions-20261017` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-rpe-feasibility-conditions-20261017` | [`870af3f65e3e31e19d2b002c556216c7340c06f0`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/870af3f65e3e31e19d2b002c556216c7340c06f0) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C30 | `track-b-m1s-shot-cost-feasibility-preparation-20261017` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-shot-cost-feasibility-preparation-20261017` | [`fb5519bf3f44392dc050feb9db44057dbc5207b1`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/fb5519bf3f44392dc050feb9db44057dbc5207b1) | 0/0/0/0 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持 |
| C31 | `track-b-m1s-stage-s-preparation-20261017` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-stage-s-preparation-20261017` | [`0dd660ece1f2e342d2666d37d4b4536adb5e8ca7`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/0dd660ece1f2e342d2666d37d4b4536adb5e8ca7) | 0/0/0/4 | 追跡変更なし、固有ローカル科学データなし、公開commitから回収可能。branchは保持。ignored bytecode4件の消失は再生成可能cacheとして明示 |

### 保全対象

| ID | branch | worktree（絶対パス） | HEAD | S/U/?/I | 推奨処置・理由 |
|---|---|---|---|---|---|
| P01 | `track-b-m1s-stage-s-execution-20261017` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-stage-s-execution-20261017` | [`f2bfb8770e6858de56a63e50e68f07ff2d6464b6`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/f2bfb8770e6858de56a63e50e68f07ff2d6464b6) | 0/0/0/0 | 統合成果・Stage S・別履歴の先行研究調査の再開拠点として保全を推奨 |
| P02 | `track-b-m1s-integrated-handoff-20261017` | `/tmp/prt-m1s-integrated-handoff-20261017` | [`5f079dbe67bb35eff35613857b169fdf812af4b8`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/5f079dbe67bb35eff35613857b169fdf812af4b8) | 0/0/0/0 | 統合成果・Stage S・別履歴の先行研究調査の再開拠点として保全を推奨 |
| P03 | `m1s-prior-art-survey-20261017` | `/tmp/prt-m1s-prior-art-20261017` | [`4301c0f87a16be45b620304c3fa935d5aa2d721a`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/4301c0f87a16be45b620304c3fa935d5aa2d721a) | 0/0/0/0 | 統合成果・Stage S・別履歴の先行研究調査の再開拠点として保全を推奨 |

### 判断保留

| ID | branch | worktree（絶対パス） | HEAD | S/U/?/I | 推奨処置・理由 |
|---|---|---|---|---|---|
| H01 | `track-b-m1s-gamma-pi3-execution-20261017` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gamma-pi3-execution-20261017` | [`81d6103ec89d485cb7336c3a90bd95dad66992aa`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/81d6103ec89d485cb7336c3a90bd95dad66992aa) | 0/0/0/0 | index bytes変化を観測。データ差異なしでも利用状況未確定、除去保留 |
| H02 | `track-b-m1s-gamma-pi3-formal-adoption-20261017` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gamma-pi3-formal-adoption-20261017` | [`b17b056deec60820413c0881978f7b77caf0861b`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/b17b056deec60820413c0881978f7b77caf0861b) | 0/0/0/0 | index bytes変化を観測。データ差異なしでも利用状況未確定、除去保留 |
| H03 | `track-b-m1s-gamma-pi3-preparation-20261017` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gamma-pi3-preparation-20261017` | [`e3828965dec292e9c66f6e7d8480e46059701183`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e3828965dec292e9c66f6e7d8480e46059701183) | 0/0/0/0 | index bytes変化を観測。データ差異なしでも利用状況未確定、除去保留 |
| H04 | `track-b-m1s-v1-gamma-completion-20261017` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-v1-gamma-completion-20261017` | [`b34572770e7c2105e16da003af050d789f490662`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/b34572770e7c2105e16da003af050d789f490662) | 0/0/0/0 | index bytes変化を観測。データ差異なしでも利用状況未確定、除去保留 |
| H05 | `track-b-m1s-v1-gamma-formal-adoption-20261017` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-v1-gamma-formal-adoption-20261017` | [`e7791c9ffe59577528e6095da5b2419bf6fadda4`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e7791c9ffe59577528e6095da5b2419bf6fadda4) | 0/0/0/0 | index bytes変化を観測。データ差異なしでも利用状況未確定、除去保留 |
| H06 | `track-b-m1s-v1-gamma-one-shot-20261017` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-v1-gamma-one-shot-20261017` | [`bc62ce6fb45ce3c67404ac8ae693953c588effac`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/bc62ce6fb45ce3c67404ac8ae693953c588effac) | 0/0/0/0 | index bytes変化を観測。データ差異なしでも利用状況未確定、除去保留 |
| H07 | `track-b-m1s-v1-gamma-preparation-20261017` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-v1-gamma-preparation-20261017` | [`fcd1c464fae48743e829fbe765a38141304f879d`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/fcd1c464fae48743e829fbe765a38141304f879d) | 0/0/0/0 | index bytes変化を観測。データ差異なしでも利用状況未確定、除去保留 |
| H08 | `track-b-m1s-v1-v0-preparation-20261017` | `/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-v1-v0-preparation-20261017` | [`be2da2c3c4ed1d826a76e8a08a98ff6c92ce90c5`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/be2da2c3c4ed1d826a76e8a08a98ff6c92ce90c5) | 0/0/0/0 | index bytes変化を観測。データ差異なしでも利用状況未確定、除去保留 |

## 4. 候補ごとの影響と削除前条件

C01–C31の各行には共通して次を適用する。例外のcacheと固有成果は次節・個別カードに明記する。

| 項目 | 削除候補の意味／必要な確認 |
|---|---|
| 消えるもの | 該当checkoutの追跡file実体とGit worktree登録。Stage S準備のC31に限りignored bytecode4件も対象に含まれる |
| 保持するもの | local/remote branch名とHEAD、履歴の全SOURCE/raw/契約/authority/grant/ledger/marker/review。main未mergeでも保持 |
| 操作上の影響 | 旧絶対パスは利用できなくなる。参照するeditor/jobを閉じ、必要時は保持branch/HEADから別途checkoutを復元する |
| one-shotへの影響 | 作業場所を取り除いても消費済み状態を未消費にしない。過去A/grant/markerを新runへ流用しない |
| 当日確認 | 対象パス・branch・HEADが表と同一、remote公開と到達可能性、staged/unstaged/untracked/ignored、lock/process/index安定性を再確認 |
| 利用者確認 | idle/remoteのCodex・IDE・他研究からの参照を確認。未確定なら保留し、branch名だけを理由に除去しない |
| 新データが見つかった場合 | 操作せず保全/保留へ変更。勝手にcommit、copy、削除、force処理で解消しない |
| 実施権限 | 後日ユーザーの具体的なworktree削除承認が必要。今回の監査・commit/push承認では削除不可 |

P01–P03はデータがdirtyだからではなく、研究再開の3拠点を明示的に残すという保守的な計画上の推奨である。
将来ユーザーが別の保持方針を選ぶことまで禁止するものではない。
H01–H08は利用状況の判断保留。理由が解消され、別途承認と最新preflightが揃うまでCへ自動変更しない。

## 5. 固有成果と固定commitの保全

統合成果・Stage S・先行研究調査の3anchorはそれぞれ対応branchが同じHEADで公開済み。
local/remote branchは全42件を保持する。branchを保持すればworktree除去だけでcommitやblobは削除されない。
この説明をbranch削除やGCの許可へ拡張しない。

3anchorから到達しない固有commitを持つbranchは以下の3件。これらを「最新統合に全て入っている」と表現しない。

| branch | 固有commit数 | 3anchorのpath/blob集合にないM1-S成果 | 保全先 |
|---|---:|---|---|
| `track-b-m1s-gamma-pi3-formal-adoption-20261017` | 1 | `artifacts/track_b_m1s_gamma_pi3_formal_adoption/2026-10-17/v1/publication_receipt.json` | 同名local/remote branch HEADを保持 |
| `track-b-m1s-gate0-v5-independent-correctness-preparation-20261016` | 1 | `scripts/tracks/algorithm_codesign/run_m1s_gate0_independent_v1.py`<br>`src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1/control.py`<br>`src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1/reference.py` | 同名local/remote branch HEADを保持 |
| `track-b-m1s-v1-gamma-formal-adoption-20261017` | 1 | `artifacts/track_b_m1s_v1_gamma_formal_adoption/2026-10-17/v1/publication_receipt.json` | 同名local/remote branch HEADを保持 |

独立correctness準備branchの旧SOURCE3ファイルは後続SOURCEとは別versionで公開されている。
未完成版・旧試行であっても履歴として保持し、branchを削除しない。正式採択branchの2つのreceiptも同様。
その他のbranch HEADは少なくとも一つの主要anchorから到達できるが、その理由でbranchを削除する提案はしない。

SOURCE/runner/test、raw結果・監査、契約、authorization/grant/ledger/marker、reviewのpath別件数と固有blobは
[audit_inventory.json](audit_inventory.json)の各branchの`evidence_category_counts`と固有blob一覧へ記録した。
追跡fileの実体照合は[tracked_blob_and_stability_checks.json](tracked_blob_and_stability_checks.json)。
カテゴリ件数はfile名による技術的索引であり、契約や結果の科学的再評価ではない。

## 6. 個別カード：保全・影響・削除直前の確認

### C01 — finite-pilot-results-scientific-review-20261017

- 分類：**削除候補**。local/remote HEAD：`e5ed0db6b8f50ed02ee9bc17cc6d0e2d4ed0898b`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-finite-pilot-results-scientific-review-20261017`。所有UID 1001、確認したM1-S/input tracked blobs 495件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`e5ed0db6b8f50ed02ee9bc17cc6d0e2d4ed0898b`。
- 関連性の根拠：Document scientific review of completed M1-S finite-step pilot。変更path例：`docs/research/m1s_finite_step_pilot_results_gpt_scientific_review_20261017.md`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C02 — finite-signal-model-accounting-v2-20261016

- 分類：**削除候補**。local/remote HEAD：`3e742894e2c79e7e46b14de9801e7e1519062879`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-finite-signal-model-accounting-v2-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 414件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`3e742894e2c79e7e46b14de9801e7e1519062879`。
- 関連性の根拠：Record finite-step v2 public-byte verification and mandatory STOP。変更path例：`artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v2/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C03 — finite-signal-preparation-20261016

- 分類：**削除候補**。local/remote HEAD：`a398a3fab6f68abc2a47799d7454b312082af292`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-finite-signal-preparation-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 386件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`a398a3fab6f68abc2a47799d7454b312082af292`。
- 関連性の根拠：Record finite-step pilot preparation publication and zero-science STOP。変更path例：`artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C04 — finite-signal-v2-completion-20261016

- 分類：**削除候補**。local/remote HEAD：`025c5a90888b02b73d2815fbb10d6bd1d2823602`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-finite-signal-v2-completion-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 494件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`025c5a90888b02b73d2815fbb10d6bd1d2823602`。
- 関連性の根拠：Record completed finite-step publication identity and final scientific STOP。変更path例：`artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C05 — finite-signal-v2-formal-adoption-20261016

- 分類：**削除候補**。local/remote HEAD：`b13a44b3fff56003679f961f05917ed9ca48f702`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-finite-signal-v2-formal-adoption-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 430件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`b13a44b3fff56003679f961f05917ed9ca48f702`。
- 関連性の根拠：Publish finite-step v2 adoption receipt and maintain scientific STOP。変更path例：`artifacts/track_b_m1s_finite_signal_v2_formal_adoption/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C06 — finite-signal-v2-one-shot-20261016

- 分類：**削除候補**。local/remote HEAD：`0bb3fbafb64f1b94b4bf8824f64cfdd7c1689ffe`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-finite-signal-v2-one-shot-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 447件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`0bb3fbafb64f1b94b4bf8824f64cfdd7c1689ffe`。
- 関連性の根拠：Record public pre-execution STOP evidence and unchanged one-shot state。変更path例：`artifacts/track_b_m1s_finite_signal_v2_preflight_stop/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C07 — g0-contract-preparation-20261015

- 分類：**削除候補**。local/remote HEAD：`0f0952614dcf5c06d3d63c21b5d49091b8ecce37`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-g0-contract-preparation-20261015`。所有UID 1001、確認したM1-S/input tracked blobs 12件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`0f0952614dcf5c06d3d63c21b5d49091b8ecce37`。
- 関連性の根拠：docs: record M1-S precontract publication byte verification。変更path例：`artifacts/track_b_m1s_g0_contract_preparation/2026-10-15/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C08 — g0-pre-adoption-survey-20261015

- 分類：**削除候補**。local/remote HEAD：`8b7666d0270f7fac01e430e195bef7b58ac0116c`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-g0-pre-adoption-survey-20261015`。所有UID 1001、確認したM1-S/input tracked blobs 6件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`8b7666d0270f7fac01e430e195bef7b58ac0116c`。
- 関連性の根拠：docs: complete M1-S G0-pre literature and protocol decision packet。変更path例：`artifacts/track_b_m1s_g0_pre/2026-10-15/v1/evidence_index.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C09 — g0t1-s0-contract-repair-20261016

- 分類：**削除候補**。local/remote HEAD：`0d84c2181de128e496bdc1e6161264cdf84eeba1`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-g0t1-s0-contract-repair-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 18件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`0d84c2181de128e496bdc1e6161264cdf84eeba1`。
- 関連性の根拠：docs: record G0-T1 S0 candidate remote byte verification。変更path例：`artifacts/track_b_m1s_g0t1_s0_contract_repair/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C10 — g0t1-s0-formal-adoption-20261016

- 分類：**削除候補**。local/remote HEAD：`373b8e4b9831f4d00bdc00ffc741ac61e03f0841`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-g0t1-s0-formal-adoption-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 25件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`373b8e4b9831f4d00bdc00ffc741ac61e03f0841`。
- 関連性の根拠：docs: record S0 formal contract remote byte and adoption verification。変更path例：`artifacts/track_b_m1s_g0t1_s0_formal_adoption/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C11 — g0t1-t1a-theory-20261016

- 分類：**削除候補**。local/remote HEAD：`4e7b18dd2aa14c127bce9e248c63996c62a055cc`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-g0t1-t1a-theory-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 33件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`4e7b18dd2aa14c127bce9e248c63996c62a055cc`。
- 関連性の根拠：docs: record remote identity for M1-S T1a theory packet。変更path例：`artifacts/track_b_m1s_g0t1_t1a_theory/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C12 — g0t1-t1b-theory-20261016

- 分類：**削除候補**。local/remote HEAD：`54336028edd7ef759f8a0c73bd0a96ade13f47f9`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-g0t1-t1b-theory-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 43件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`54336028edd7ef759f8a0c73bd0a96ade13f47f9`。
- 関連性の根拠：docs: record verified publication of M1-S T1b theory。変更path例：`artifacts/track_b_m1s_g0t1_t1b_theory/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### H01 — gamma-pi3-execution-20261017

- 分類：**判断保留**。local/remote HEAD：`81d6103ec89d485cb7336c3a90bd95dad66992aa`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gamma-pi3-execution-20261017`。所有UID 1001、確認したM1-S/input tracked blobs 671件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：いいえ（保留）。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`81d6103ec89d485cb7336c3a90bd95dad66992aa`。
- 関連性の根拠：Record verified publication of completed gamma pi over three gate and mandatory stop。変更path例：`artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01/publication_receipt.json`。
- 推奨処置・影響：除去しない。index更新の原因・所有者・他Codexの利用状況を確認する。SOURCE/rawの一致は保留解除の代わりにならない。解消後も新preflightとユーザーの削除承認が必要。

### H02 — gamma-pi3-formal-adoption-20261017

- 分類：**判断保留**。local/remote HEAD：`b17b056deec60820413c0881978f7b77caf0861b`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gamma-pi3-formal-adoption-20261017`。所有UID 1001、確認したM1-S/input tracked blobs 654件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：いいえ（保留）。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから未到達の固有履歴あり。同名remote branchから回収可能。branchは保持。直近の関連commit：`b17b056deec60820413c0881978f7b77caf0861b`。
- 関連性の根拠：Record verified pi over three formal contract publication and stop before execution。変更path例：`artifacts/track_b_m1s_gamma_pi3_formal_adoption/2026-10-17/v1/publication_receipt.json`。
- 推奨処置・影響：除去しない。index更新の原因・所有者・他Codexの利用状況を確認する。SOURCE/rawの一致は保留解除の代わりにならない。解消後も新preflightとユーザーの削除承認が必要。

### H03 — gamma-pi3-preparation-20261017

- 分類：**判断保留**。local/remote HEAD：`e3828965dec292e9c66f6e7d8480e46059701183`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gamma-pi3-preparation-20261017`。所有UID 1001、確認したM1-S/input tracked blobs 643件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：いいえ（保留）。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`e3828965dec292e9c66f6e7d8480e46059701183`。
- 関連性の根拠：Record verified publication of unadopted pi over three gate preparation。変更path例：`artifacts/track_b_m1s_gamma_pi3_preparation/2026-10-17/v1/publication_receipt.json`。
- 推奨処置・影響：除去しない。index更新の原因・所有者・他Codexの利用状況を確認する。SOURCE/rawの一致は保留解除の代わりにならない。解消後も新preflightとユーザーの削除承認が必要。

### C13 — gate0-formula-contract-20261016

- 分類：**削除候補**。local/remote HEAD：`e749bf8844f5d78817f18f738ade87269c2d70e1`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-formula-contract-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 129件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`e749bf8844f5d78817f18f738ade87269c2d70e1`。
- 関連性の根拠：Record remote byte identity and Gate 0 formula preparation STOP。変更path例：`artifacts/track_b_m1s_gate0_formula_contract/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C14 — gate0-precision-cause-20261016

- 分類：**削除候補**。local/remote HEAD：`a058ef14d4ea162547ec73b707a4662ec8fcd7d6`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-precision-cause-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 209件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`a058ef14d4ea162547ec73b707a4662ec8fcd7d6`。
- 関連性の根拠：Record precision-cause publication identity and preserved STOP。変更path例：`artifacts/track_b_m1s_gate0_precision_cause/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C15 — gate0-precision-contract-20261016

- 分類：**削除候補**。local/remote HEAD：`4019d045cbfad94ba9333b544f7950a6d10d467d`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-precision-contract-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 104件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`4019d045cbfad94ba9333b544f7950a6d10d467d`。
- 関連性の根拠：docs: record verified Gate0 scope precision publication。変更path例：`artifacts/track_b_m1s_gate0_precision_contract/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C16 — gate0-source-preparation-20261016

- 分類：**削除候補**。local/remote HEAD：`83fbac28bca8e308b3039d8532492e6510492f89`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-source-preparation-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 78件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`83fbac28bca8e308b3039d8532492e6510492f89`。
- 関連性の根拠：docs: record verified M1-S Gate0 source publication。変更path例：`artifacts/track_b_m1s_gate0_source_preparation/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C17 — gate0-tail-bound-proof-20261016

- 分類：**削除候補**。local/remote HEAD：`b5a185cc0c3aae0e477bfc52b93061e86feb7dec`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-tail-bound-proof-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 223件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`b5a185cc0c3aae0e477bfc52b93061e86feb7dec`。
- 関連性の根拠：Record tail-bound proof publication and non-execution boundary。変更path例：`artifacts/track_b_m1s_gate0_tail_bound/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C18 — gate0-v4-contract-candidate-20261016

- 分類：**削除候補**。local/remote HEAD：`44277e03448ee51dbe9afdd2dc6560337757cc6d`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v4-contract-candidate-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 158件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`44277e03448ee51dbe9afdd2dc6560337757cc6d`。
- 関連性の根拠：Record Gate 0 v4 publication byte identity and mandatory STOP。変更path例：`artifacts/track_b_m1s_gate0_v4_contract/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C19 — gate0-v4-formal-adoption-20261016

- 分類：**削除候補**。local/remote HEAD：`5e57955baa79e42f37aa233ac011c60539e431fd`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v4-formal-adoption-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 174件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`5e57955baa79e42f37aa233ac011c60539e431fd`。
- 関連性の根拠：Record formal adoption publication identity and science STOP。変更path例：`artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C20 — gate0-v4-one-shot-20261016

- 分類：**削除候補**。local/remote HEAD：`c937ed37e73bb9b0bb39f2fa7116d255a3215269`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v4-one-shot-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 197件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`c937ed37e73bb9b0bb39f2fa7116d255a3215269`。
- 関連性の根拠：Record Gate 0 one-shot publication identity and mandatory stop。変更path例：`artifacts/track_b_m1s_gate0_execution_audit/2026-10-16/v4_one_shot/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C21 — gate0-v5-formal-adoption-20261016

- 分類：**削除候補**。local/remote HEAD：`e46386abb70d601b973850b82e9bcfb0430ad0ee`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v5-formal-adoption-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 266件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`e46386abb70d601b973850b82e9bcfb0430ad0ee`。
- 関連性の根拠：Record v5 formal adoption publication and preserved non-execution boundary。変更path例：`artifacts/track_b_m1s_gate0_v5_formal_adoption/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C22 — gate0-v5-independent-audit-prep-20261016

- 分類：**削除候補**。local/remote HEAD：`2ba0858c3c92b61581cfbf6cdfb5039d25a174c9`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v5-independent-audit-prep-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 315件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`2ba0858c3c92b61581cfbf6cdfb5039d25a174c9`。
- 関連性の根拠：Record independent correctness preparation publication and zero-computation STOP。変更path例：`artifacts/track_b_m1s_gate0_independent_preparation/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C23 — gate0-v5-independent-correctness-preparation-20261016

- 分類：**削除候補**。local/remote HEAD：`d6d35cfb79b97f9cb218c9169b427aacf6a4a334`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v5-independent-correctness-preparation-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 251件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから未到達の固有履歴あり。同名remote branchから回収可能。branchは保持。直近の関連commit：`d6d35cfb79b97f9cb218c9169b427aacf6a4a334`。
- 関連性の根拠：Preserve halted independent Gate 0 correctness source draft。変更path例：`scripts/tracks/algorithm_codesign/run_m1s_gate0_independent_v1.py`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C24 — gate0-v5-one-shot-20261016

- 分類：**削除候補**。local/remote HEAD：`cdcb7f6f151df217960611f502eb1bac8be63f58`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v5-one-shot-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 290件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`cdcb7f6f151df217960611f502eb1bac8be63f58`。
- 関連性の根拠：Record Gate 0 v5 one-shot publication identity and mandatory STOP。変更path例：`artifacts/track_b_m1s_gate0_v5_execution_audit/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C25 — gate0-v5-source-20261016

- 分類：**削除候補**。local/remote HEAD：`47da54279838a6b3e1497f36204ba7e53c4857f9`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-gate0-v5-source-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 247件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`47da54279838a6b3e1497f36204ba7e53c4857f9`。
- 関連性の根拠：Record v5 source candidate publication and byte identity。変更path例：`artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C26 — independent-contract-adoption-20261016

- 分類：**削除候補**。local/remote HEAD：`0956fd477a54a7f4f0570944a1265e0c27cb3fbc`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-independent-contract-adoption-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 332件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`0956fd477a54a7f4f0570944a1265e0c27cb3fbc`。
- 関連性の根拠：Record independent contract adoption publication verification and STOP。変更path例：`artifacts/track_b_m1s_independent_contract_adoption/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C27 — independent-one-shot-20261016

- 分類：**削除候補**。local/remote HEAD：`cc2f451324b7292ae50cef94ff794d46f225d892`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-independent-one-shot-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 358件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`cc2f451324b7292ae50cef94ff794d46f225d892`。
- 関連性の根拠：Record independent one-shot public identity verification and mandatory STOP。変更path例：`artifacts/track_b_m1s_independent_execution_audit/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C28 — post-checkpoint2-preflight-20261016

- 分類：**削除候補**。local/remote HEAD：`05513eea807fbfb841c6012d2fe2133d70b5f1f9`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-post-checkpoint2-preflight-20261016`。所有UID 1001、確認したM1-S/input tracked blobs 58件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`05513eea807fbfb841c6012d2fe2133d70b5f1f9`。
- 関連性の根拠：docs: record verified M1-S post-checkpoint2 draft publication。変更path例：`artifacts/track_b_m1s_post_checkpoint2_preflight/2026-10-16/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C29 — rpe-feasibility-conditions-20261017

- 分類：**削除候補**。local/remote HEAD：`870af3f65e3e31e19d2b002c556216c7340c06f0`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-rpe-feasibility-conditions-20261017`。所有UID 1001、確認したM1-S/input tracked blobs 507件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`870af3f65e3e31e19d2b002c556216c7340c06f0`。
- 関連性の根拠：Record publication identity for M1-S RPE conditions packet。変更path例：`artifacts/track_b_m1s_rpe_conditions/2026-10-17/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### C30 — shot-cost-feasibility-preparation-20261017

- 分類：**削除候補**。local/remote HEAD：`fb5519bf3f44392dc050feb9db44057dbc5207b1`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-shot-cost-feasibility-preparation-20261017`。所有UID 1001、確認したM1-S/input tracked blobs 691件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`fb5519bf3f44392dc050feb9db44057dbc5207b1`。
- 関連性の根拠：docs: record M1-S shot-cost packet public identity and preservation receipt。変更path例：`artifacts/track_b_m1s_shot_cost_preparation/2026-10-17/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### P01 — stage-s-execution-20261017

- 分類：**保全対象**。local/remote HEAD：`f2bfb8770e6858de56a63e50e68f07ff2d6464b6`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-stage-s-execution-20261017`。所有UID 1001、確認したM1-S/input tracked blobs 748件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`f2bfb8770e6858de56a63e50e68f07ff2d6464b6`。
- 関連性の根拠：docs: seal Stage S result publication receipt and mandatory STOP。変更path例：`artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/publication_receipt.json`。
- 推奨処置・影響：再開用拠点としてworktreeも残す。公開済みでも削除候補には含めない。保持方針を変える場合はユーザー判断と新しい利用状況確認が必要。

### C31 — stage-s-preparation-20261017

- 分類：**削除候補**。local/remote HEAD：`0dd660ece1f2e342d2666d37d4b4536adb5e8ca7`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-stage-s-preparation-20261017`。所有UID 1001、確認したM1-S/input tracked blobs 723件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 4。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`0dd660ece1f2e342d2666d37d4b4536adb5e8ca7`。
- 関連性の根拠：docs: record Stage S public identity and preservation receipt。変更path例：`artifacts/track_b_m1s_stage_s_preparation/2026-10-17/v1/publication_receipt.json`。
- 推奨処置・影響：後日の承認・直前preflight後にこのcheckoutだけを整理する候補。旧絶対パスは消えるが、公開されたbranch/HEADの成果は維持。
- cache例外：`src/trottertracks/__pycache__/__init__.cpython-310.pyc`（301 bytes）、対応追跡source `src/trottertracks/__init__.py`。削除未実施。
- cache例外：`src/trottertracks/algorithm_codesign/__pycache__/__init__.cpython-310.pyc`（326 bytes）、対応追跡source `src/trottertracks/algorithm_codesign/__init__.py`。削除未実施。
- cache例外：`src/trottertracks/algorithm_codesign/m1s_stage_s_v1/__pycache__/__init__.cpython-310.pyc`（344 bytes）、対応追跡source `src/trottertracks/algorithm_codesign/m1s_stage_s_v1/__init__.py`。削除未実施。
- cache例外：`src/trottertracks/algorithm_codesign/m1s_stage_s_v1/__pycache__/spec.cpython-310.pyc`（4244 bytes）、対応追跡source `src/trottertracks/algorithm_codesign/m1s_stage_s_v1/spec.py`。削除未実施。
- 削除前の確認：§4の全条件とC/P/H ID・HEADの一致、他Codex/IDE/jobの利用終了。変化があれば除去せず保留。branch削除・force・pruneは計画に含めない。

### H04 — v1-gamma-completion-20261017

- 分類：**判断保留**。local/remote HEAD：`b34572770e7c2105e16da003af050d789f490662`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-v1-gamma-completion-20261017`。所有UID 1001、確認したM1-S/input tracked blobs 620件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：いいえ（保留）。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`b34572770e7c2105e16da003af050d789f490662`。
- 関連性の根拠：Record verified publication of completed gamma V1 analysis and mandatory stop。変更path例：`artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/publication_receipt.json`。
- 推奨処置・影響：除去しない。index更新の原因・所有者・他Codexの利用状況を確認する。SOURCE/rawの一致は保留解除の代わりにならない。解消後も新preflightとユーザーの削除承認が必要。

### H05 — v1-gamma-formal-adoption-20261017

- 分類：**判断保留**。local/remote HEAD：`e7791c9ffe59577528e6095da5b2419bf6fadda4`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-v1-gamma-formal-adoption-20261017`。所有UID 1001、確認したM1-S/input tracked blobs 566件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：いいえ（保留）。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから未到達の固有履歴あり。同名remote branchから回収可能。branchは保持。直近の関連commit：`e7791c9ffe59577528e6095da5b2419bf6fadda4`。
- 関連性の根拠：Record formal gamma V1 adoption publication and mandatory stop。変更path例：`artifacts/track_b_m1s_v1_gamma_formal_adoption/2026-10-17/v1/publication_receipt.json`。
- 推奨処置・影響：除去しない。index更新の原因・所有者・他Codexの利用状況を確認する。SOURCE/rawの一致は保留解除の代わりにならない。解消後も新preflightとユーザーの削除承認が必要。

### H06 — v1-gamma-one-shot-20261017

- 分類：**判断保留**。local/remote HEAD：`bc62ce6fb45ce3c67404ac8ae693953c588effac`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-v1-gamma-one-shot-20261017`。所有UID 1001、確認したM1-S/input tracked blobs 582件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：いいえ（保留）。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`bc62ce6fb45ce3c67404ac8ae693953c588effac`。
- 関連性の根拠：Record verified gamma one-shot failure publication and mandatory stop。変更path例：`artifacts/track_b_m1s_v1_gamma_execution_audit/2026-10-17/v1/publication_receipt.json`。
- 推奨処置・影響：除去しない。index更新の原因・所有者・他Codexの利用状況を確認する。SOURCE/rawの一致は保留解除の代わりにならない。解消後も新preflightとユーザーの削除承認が必要。

### H07 — v1-gamma-preparation-20261017

- 分類：**判断保留**。local/remote HEAD：`fcd1c464fae48743e829fbe765a38141304f879d`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-v1-gamma-preparation-20261017`。所有UID 1001、確認したM1-S/input tracked blobs 556件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：いいえ（保留）。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`fcd1c464fae48743e829fbe765a38141304f879d`。
- 関連性の根拠：Record gamma-dependent V1 publication identity and mandatory stop。変更path例：`artifacts/track_b_m1s_v1_gamma_preparation/2026-10-17/v1/publication_receipt.json`。
- 推奨処置・影響：除去しない。index更新の原因・所有者・他Codexの利用状況を確認する。SOURCE/rawの一致は保留解除の代わりにならない。解消後も新preflightとユーザーの削除承認が必要。

### H08 — v1-v0-preparation-20261017

- 分類：**判断保留**。local/remote HEAD：`be2da2c3c4ed1d826a76e8a08a98ff6c92ce90c5`、一致。
- 対応worktree：`/home/abe/Project/Partially Randomized Trotter/.worktrees/track-b-m1s-v1-v0-preparation-20261017`。所有UID 1001、確認したM1-S/input tracked blobs 529件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：いいえ（保留）。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`be2da2c3c4ed1d826a76e8a08a98ff6c92ce90c5`。
- 関連性の根拠：Record verified M1-S V1 V0 publication and mandatory stop。変更path例：`artifacts/track_b_m1s_v1_v0_preparation/2026-10-17/v1/publication_receipt.json`。
- 推奨処置・影響：除去しない。index更新の原因・所有者・他Codexの利用状況を確認する。SOURCE/rawの一致は保留解除の代わりにならない。解消後も新preflightとユーザーの削除承認が必要。

### P02 — integrated-handoff-20261017

- 分類：**保全対象**。local/remote HEAD：`5f079dbe67bb35eff35613857b169fdf812af4b8`、一致。
- 対応worktree：`/tmp/prt-m1s-integrated-handoff-20261017`。所有UID 1001、確認したM1-S/input tracked blobs 759件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`5f079dbe67bb35eff35613857b169fdf812af4b8`。
- 関連性の根拠：docs: seal M1-S handoff publication identity and STOP receipt。変更path例：`docs/research/m1s_integration_20261017/publication_receipt.json`。
- 推奨処置・影響：再開用拠点としてworktreeも残す。公開済みでも削除候補には含めない。保持方針を変える場合はユーザー判断と新しい利用状況確認が必要。

### P03 — m1s-prior-art-survey-20261017

- 分類：**保全対象**。local/remote HEAD：`4301c0f87a16be45b620304c3fa935d5aa2d721a`、一致。
- 対応worktree：`/tmp/prt-m1s-prior-art-20261017`。所有UID 1001、確認したM1-S/input tracked blobs 684件、差異0。
- 未保存データ：staged 0／unstaged 0／untracked 0／ignored 0。再生成cache以外のlocal file 0。
- lock 0／観測できたprocess参照 0。初回index bytes安定：はい。非可視sessionの不使用保証はない。
- 固有成果・保全：主要anchorから到達可能。branchは保持。直近の関連commit：`4301c0f87a16be45b620304c3fa935d5aa2d721a`。
- 関連性の根拠：docs: record verified M1-S literature publication and mandatory stop。変更path例：`docs/research/m1s_prior_art_survey_20261017/publication_receipt.json`。
- 推奨処置・影響：再開用拠点としてworktreeも残す。公開済みでも削除候補には含めない。保持方針を変える場合はユーザー判断と新しい利用状況確認が必要。

## 7. 禁止操作・監査後の停止

今回、worktree remove/prune、branchのlocal/remote削除、clean、rm、merge、rename、履歴変更、既存worktree checkout、
他job停止/設定変更、既存SOURCE/raw/契約/authority/grant/ledger/markerの変更は実行していない。
監査成果だけを専用clone/branchへcommit・pushする。科学状態や既存索引・manifestは変更しない。

結果の公開同一性は[publication_receipt.json](publication_receipt.json)、技術検査と元repository保全は
[verification.json](verification.json)を参照。indexの観測差を「全metadata無変化」と偽らず、上記8件を保留した。

**mandatory STOP。次の担当はユーザーで、削除候補C01–C31のどれを整理するか明示承認する。
Codexによる削除はその後の新しい指示と直前再監査まで実施しない。**
