# B終了とmandatory STOP

固定runnerの実statusは **B_COMPLETE_MANDATORY_STOP**。本報告の段階状態 **B_BINDING_COMPLETE_MANDATORY_STOP**はこの成功STOPと対応する。binding summaryの **B_INPUTS_BOUND_F1_REVIEW_REQUIRED**を、自動レビュー依頼の許可とは扱わない。

- 新規B grant：1件発行、固定gate照合後に1回consume。
- Consumed marker：1件存在し、内容は新authorizationのdigestと一致。以後のretry/resumeは禁止。
- 実行ledger：11attempt／11completed、manifest、24行log。独立ledger fileを追加したとは主張しない。
- STOP：原本を保全、SHA256 `484139aba19a4125b09a518b77dd03977af3188c4835769b41e1cd41058ece34`。
- Failure：0、runtime errorなし、stderr空、process exit0。
- F1/F2、sampling/event/trajectory、build/compile、Operator/mean/signal、price/interval資格：未実施。

科学処理は11case完了時に停止済み。以後は保存fileのread/hash/文書化・公開照合だけで、input/lawを再計算していない。旧grant／ledger／marker／STOPを操作・再利用していない。原本・共有env・他jobも変更しない。

B成功はF1 validated、F2 ready、物理精度certificate、資源改善GO、新規性GOではない。次の担当はユーザー。F1の残る準備の範囲を判断し、科学実行は別途明示承認する。公開照合後もmandatory STOPを保持する。
