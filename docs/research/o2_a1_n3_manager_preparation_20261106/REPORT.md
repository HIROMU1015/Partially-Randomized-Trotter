# A1 N3：持続manager整備・人工検証完了、mandatory STOP

2026-11-06 JST、SERVER1 GPUA100-gleap-01。
**管理経路は条件付きで技術使用可能。unit13件・cron経路6ケースPASS。実N3/新科学計算は0。**

## 完成した範囲

既存のユーザーcronを利用し、自分の人工jobが`/system.slice/cron.service`で動くことを確認した。
Linger=noのまま、会話/投入clientの終了からapplicationの寿命を分離した。自分の一時crontabは開始前に撤去し、元の不存在状態へ戻した。
共有cron.service・systemd/linger/cgroup等は変更せず、実際のSSH・他user/serviceを操作していない。

消費済receiptによる再起動/二重実行拒否、外側recorder、親/worker identityとraw exit/signal、実child/ancestorのresource、call start/return/exception、STOP先行保存、bit/価格欠測分類を接続した。
旧lifecycle/atomic writer/journal parserを再利用し、旧科学SOURCEと数値kernel/乱数/統計処理は変更していない。
管理SOURCEは公開固定`5cb542ece4d208878a1dfc7011007b99297a5a55`。初回source d139cf8も保全した。

## 人工結果・限界

正常終了、worker exit23、親SIGKILL、投入client終了、明示STOP、manager SIGKILLの6ケースを同じcron経路で検証した。
STOPはworker回収より先、人工bitは価格欠測時も保全、同じSPECの再投入はexit72で拒否した。
初回は集計fileとworker identityの混同でsuiteが失敗。source/logを保全し、回帰test追加後の人工run02で6ケースを完了した。

外側recorder/host/cron全体の同時消失、実SSH logout、実engineの長時間負荷は未保証。
親wait ownerが失われたworkerのexitcode、manager消失時に未保存の親exitcodeはUNKNOWN。これを推測で補わない。
共有daemon既存Restart=on-failureは変更せず、applicationのrestart=noと区別する。
Python3.12 private pool registry依存を明記し、別版へ無条件に移送しない。

## 新campaignの準備と次の条件

[案B契約草案](N3_NEW_CAMPAIGN_CONTRACT_DRAFT.md)を作成した。旧397件を新train/cal/prodへ混ぜず、旧結果/費用/STOPを保全し、同N1/N2/engine・科学条件を再利用する。
候補identity、候補seed20261107と独立stream、一条件4方式×X/Y・一campaignの有限scopeは**提案のみ**。正式採択やword生成はしていない。
本番B用の薄いentryと新SOURCE/環境lockは採択内容で別途固定する必要がある。現在のdispatch CLIは人工専用で、実N3は投入できない。
次はseed/stream・契約の採択、本番entry/SOURCE固定、有限科学実行への別途明示許可。新しい承認証明やgrant発行は要求しない。

## 成果物・保全

- [N3_MANAGER_CONNECTION](N3_MANAGER_CONNECTION.md)：方式・接続・持続性の保証範囲。
- [N3_LIFECYCLE_TESTS](N3_LIFECYCLE_TESTS.md)：人工結果、失敗履歴、取得不能事項。
- [artifact SOURCE](../../../artifacts/o2_a1_n3_manager_preparation/2026-11-06/SOURCE.json)、[qualification](../../../artifacts/o2_a1_n3_manager_preparation/2026-11-06/qualification_run02/QUALIFICATION_RESULT.json)、[STOP](../../../artifacts/o2_a1_n3_manager_preparation/2026-11-06/STOP.json)、[保全](../../../artifacts/o2_a1_n3_manager_preparation/2026-11-06/PRESERVATION.json)、[manifest](../../../artifacts/o2_a1_n3_manager_preparation/2026-11-06/EVIDENCE_MANIFEST.json)。

独立worktree/branch `codex/o2-a1-n3-manager-preparation-20261106`を使用。
元main HEAD/statusと3087既存file、旧科学/診断worktree、SOURCE542、旧397件/価格/係数/STOP・grant/ledger/marker、C1凍結を保全する。
人工job終了後は追加実験を行わず、docs/証拠保全・commit/push・公開本文/blob/treeとremote SHA照合後にmandatory STOPする。
