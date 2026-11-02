# H8 Phase C：修正後SOURCE再固定・契約結合・公開認証待ち

利用者から通常signed biasに基づくPhase Cの一回実行承認を受領し、SOURCEと実行契約を再固定した。
人工49件、入力12件のbytes/SHA・Git object、B2登録4軸受入はPASS。
ただしHTTPS pushの認証確認が失敗した。今回の利用者指示は「公開やSOURCE照合が失敗した場合は、その状態を解消しないまま実計算を開始しない」と明記している。
このため科学実行0・有効grant発行0で公開待ち。科学条件や一回実行の承認は取得済みで、再承認を求める状態ではない。

## 固定来歴

- SOURCE：`931755c4c84aa0751dbde7ddf84cb5187b5172ae`
- 旧統合SOURCE：`29ed76544441d534ad99e5c708a2ae17eec78c18`（履歴を保持）
- 修正起点：`b7be02fcf705425a0d98e871b19d0c1aebf60c09`（公開branchからfetchし、remote先端と照合）
- 新plan：`79d624b9bc1f10232b7cd5e48ce0d4468f7a4375ba6608d6ab832a5f27947232`
- 旧plan：`9e42fc76078568c9aaad4945539ffc46293214f71d4d36db1f0f53beea20d132`（新SOURCEでは拒否）
- source closure：`e00443937b8fef05300cd920851504da8d92b38fda95416250e41c0c69ef7066`（8 files）
- [固定bundle](../../artifacts/resource_applicability/track_a_h8_phase_c_execution_binding/2026-10-17/README.md)

## 科学契約は維持

linear H8、1.00 Å、STO-3G、保存tol-only DF rank28、16 spin modes/sector4900、T=.8、q=1/2/4/8。
ε=.05/.01/.005/.001/.0005、allowance倍率1/10、通常signed bias、既存のshot・normalization・適格性規則、P=0。
旧32決定論候補と強化184候補を別に評価し、PR確認union6（各n32）とのshot-weighted full-wrapper RZを比較する。
探索n8と非相殺shotは混ぜない。通常会計1,900行・比較20回・会計pass1の上限、exact rational算術を維持する。
新signal/shot生成・compile・sampling・fit・B2再実行・次stage進行は許可しない。

B2最終受入 `fef2db7a79886ab682b652da7266b463c98c7d3a` に基づき、B0 p15/q1 S4の両軸168353、B0+I p21/q8 S4の両軸1852640を受け入れた。
保存3軸・新規1軸の軸別SOURCE、native exit/reap、最終gate等は公開契約のgateで照合済み。
旧STOPをPASSに変更していない。単価順位はB2実測→B1 anchor→旧full PF保存実測（旧/不明profileを保持）→固定B1モデル。
未測定S4はモデル推定であり全格子の精度保証ではない。PRはn32有限標本点費用であり母平均優位・厳密bias保証を主張しない。

## SOURCE差分と人工検証

`b7be02fc` の観測範囲修正・owned child cleanupを確認した。追加の変更はplanに隔離/cleanup方針を明記する6行と、旧plan拒否の人工test10行。
会計module・入力adapter・registry・入力契約のbytesは旧SOURCEから不変。科学planの既存fieldも完全一致する。
無関係なworktreeはHEAD/statusの観測のみ。自身のSOURCE・入力・Python等21 receiptは厳密照合する。
cleanupは未回収direct childのPID/start/PPID/UIDを確認してpidfdで送信し、TERM→必要時KILL→wait4。
ESRCHでもnative waitは必須、他error・所有不一致・未回収は成功扱いしない。最初のSTOP理由を保持する。

固定private Pythonで人工49件PASS（元37件＋前修正11件＋旧plan拒否1件、wall約5.69秒）。
実launcher→人工driver E2E、出力/one-shot再使用拒否、SOURCE改変拒否、非相殺混入拒否、missing/unresolvedの扱い、正常/例外/TERM/KILLとreapを含む。
未seal draftとsealed INVALID draftの両方を実launcherへ渡し、副作用の前に `NEW_SEALED_C_USER_GRANT_REQUIRED` で拒否された。
これは主担当によるlocal検証であり、別担当による独立reviewやCIではない。subagent0。

## 実行用固定領域

実行用checkout（資料branchとは独立）：
`/home/AbeHiromu/projects/h8-phase-c-integration/20261017-03-execution/execution-repo`

HEADは上記SOURCEへdetachし、source8件をGit object/bytes/SHAで照合した。
入力12件はhome下のexclusive directoryへbyte-only copyし、固定契約と照合した。科学arrayやPhase C数値行は解析していない。
Python SHA：`ebc2fae9fa0845b8c1fc09ed10196c0e93c26cb8fbd2a84d64548326ef93209b`。
profile：`622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`、
compiler：`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。
private config/grant原本は `20261017-03-execution/metadata_draft/` と `stage01/` に別保存し、旧資料を上書きしていない。

sealed config digest：`6ce100df4f8e975e9e37347949380cb855e6e664640957eef61f573f8a762238`。
sealed INVALID grant digest：`5c034f387ae68fb65e92bbf9b414a152533a45b5a95565a11e752ada4534148e`。
これは有効grantのdigestではない。有効grant・allowed CPU・実行fresh gateは公開後に新規発行/検査する。
`output01/control01/C_ONE_SHOT/C_LAUNCH_ATTEMPT` は未作成で、科学attempt0。

## 資源・非干渉

読取監査時MemAvailable約468.42 GiB、disk空き約486.2 GiB、free inode219282610、ext4 mountにquota optionなし。
これは直前fresh gateではない。公開確認後にCPU・実効memory/cgroup/OOM・容量・inode・quota・未使用output/lockを再確認する。
保存値会計はindexed一回走査とaxis cacheを使う軽い単一processであり、不必要なworker増設はしない。
エージェント独自のCPU/RAM/wall capはなく、PSIは観測のみ。他ジョブのCPU割当や共有環境/venvは変更していない。
B2/M1-S・旧grant/ledger/marker/STOPには書き込んでいない。root/他worktree登録・共有ref更新0、広域のfile探索・大量hash監査0。

## 公開と次の作業

HTTPS pushの事前確認はexit128、`could not read Username for 'https://github.com': No such device or address`。
認証・remote設定は変更しない。資料とSOURCEをlocal commitに保全し、手動pushコマンドをチャットへ提示する。
実pushとremote SHAの最終結果はprivate公開receiptと最終チャットに記録する。push失敗を公開済みと扱わない。

1. ユーザー：提示された資料branchのnon-force pushを行う。
2. Codex：remote SHA/公開SOURCE・契約を照合し、既存承認に基づきPhase Cだけの新grantを発行し、fresh gate後に一回実行する。追加の科学承認を求めない。
3. Codex：正常終了なら結果・native wait・回収・最終gateを保存/公開してmandatory STOP。異常STOPなら証拠を保持し、自動retryせず原因と新SOURCE/grant案を報告する。

現段階では5精度の最良費用・PR削減率は未計算。正常終了/不合格の科学結果はまだ存在しない。
