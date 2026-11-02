# H4 新サーバー引継ぎ資料のGit公開（2026-10-07）

今回の変更は引継ぎ資料と索引だけである。旧サーバーでcommit済みのsourceをGitHubから
取得可能にするため、元branchと資料用branchへnon-force pushを試す。公開完了はpush成功と
remote SHA一致で確認する。commit作成だけでは公開済みと扱わない。

## 取得identityと資料入口

- 元branch：`track-a-h4-lazy-identity-run05-20261006`
- 元引継ぎcommit：`8f77bebf99c5bd15fa6419c1f58556c3bd2837a9`
- science SOURCE_COMMIT：`6d365257770e99022b91d6a38dbee49ee0077503`
- 資料公開branch：`track-a-h4-new-server-handoff-20261007`（元引継ぎcommitの子）
- [新サーバーCodexへの指示](handoffs/h4-new-server-20261007/NEW_SERVER_CODEX_INSTRUCTIONS.md)
- [停止状態と入力hash](handoffs/h4-new-server-20261007/HANDOFF_STATE_v1.json)
- [公開対象・source19・plan/auth/review不変性](handoffs/h4-new-server-20261007/publication_manifest_v1.json)
- [既存source/plan資料](../../artifacts/resource_applicability/track_a_h4_lazy_identity_run05/2026-10-06/README.md)

旧`.server-preparation/handoffs/`にある6軽量資料をbytes不変のまま公開pathへコピーした。
そのREADME/manifest/指示書の「未公開」「bundle未作成」等は各作成時点の履歴である。
後のbundle作成・検証結果は同梱のGIT_BUNDLE_DELIVERY_v1.jsonにある。
publication_manifestのremote未確認も準備時点の観測で、成功後のremote SHA確認とは区別する。
旧資料、旧branch/worktree、既存source19、plan、authorization、stage reviewは変更しない。

公開に成功した場合、新サーバーの既存cloneで次の取得を行う。commandは新サーバー側で未実行。

```bash
git -c maintenance.auto=false -c gc.auto=0 fetch origin \
  track-a-h4-lazy-identity-run05-20261006 \
  track-a-h4-new-server-handoff-20261007
git rev-parse origin/track-a-h4-lazy-identity-run05-20261006
git cat-file -t 8f77bebf99c5bd15fa6419c1f58556c3bd2837a9
```

元branchのSHAが上記40文字SHAと一致することを確認する。既存の同名branchやworktreeを
上書きしない。公開資料branchのcommitも報告された40文字SHAで確認する。
資料branchに科学sourceの変更はなく、source/plan bindingは旧SOURCE_COMMITのままである。
新hostでの独立worktree作成・環境監査・修正・新binding草案固定は上記Codex指示に従う。

## 現在の停止状態と引継ぎ予算

run05は2026-10-06 23:28 JSTに`monitor interval/freshness`でSTOP。
run05完成compile recordsは0、signal recordsは0/1308、全owned processesは終了済み。
75 local人工testsの合格はproduction成功を意味しない。H4 geometry mapと最終total-cost評価は未完了。
旧run05の起動準備記載は当時の履歴であり、本補足とHANDOFF_STATEを停止後の記録として読む。

累積actual invocations消費/予約は20、chargeは165214360 bytes、次attemptへの保守的wall carryは
5466.188392877579 s。未完了予約を返却せず、campaign予算をresetしない。
既存上限74784 actual invocations/10 GiB charge/72 h wallを維持する場合、残invocationsは74764。
行列parameter serializationと監視の干渉は未解決で、新hostでは監視/serialization修正と
対象を絞った人工testsまでを進める。deadlineやAS/RSS等の上限を勝手に緩和しない。

H4 linear neutral singlet/STO-3G、4 spatial/8 system+ancilla1、DF12。
距離0.70/0.80/0.90/1.10/1.40/1.60 Å、T=0.8、二次DF-prefix PF/canonical finite-RTE、
L_D=0/3/4/5/6/9/12、q=1/2/4/8、delta=0.8/0.4/0.2/0.1、固定r/K、random32 paired trajectories。
科学条件、compiler、source、計算上限を今回変更しない。

## Git公開と別転送の境界

GitHubから取得できる対象は既存のcommit済みsource/準備資料と今回の軽量引継ぎ資料である。
6凍結NPZ、generation-freeze、旧runtime/checkpoint/cache/control/log実体は今回Gitへ追加しない。
これらは上記Codex指示にある元directoryから専用evidence領域へ別途コピーし、bytes/hashを照合する。
source/資料のGit取得だけなら旧サーバーへのSSH認証は不要になるが、未追跡入力・停止証拠の
転送には引き続き既存SSH等の受領経路が必要である。未受領を入力再生成で補わない。

新サーバーからのSSH接続・認証と実転送は、この資料作成時点では未確認/未実施。
GitHub HTTPSのpush認証と旧hostのSSH認証は別であり、clone成功だけでpush可能とは判断しない。
認証失敗時は設定を変更せず、未公開commitと手動push commandを報告する。

## 認可と停止条件

新hostのCPU使用・最終review・明示launchは未承認。旧hostのCPU許可やapproved=true記録を
新hostへ流用しない。今回の公開は移行監査・監視修正・人工tests・source/plan/認可草案固定まで。
本計算、入力再生成、taskset/worker起動、GPU query/use、共有設定や他jobの変更は行わない。
新hostでは必要なCPU承認・最終review・launch指示が揃うまでSTOPする。
