# 次段階：ユーザーとGPTの判断待ち

1. **ユーザーとGPT**：RQの必要範囲、n64/256/1024、S/P1二条件か単条件か、reference codec第3条件の要否、path coverage、CPU分解能/反復条件を決める。推奨検討案はn256・二条件16,896 callsであり採択済みではない。
2. **ユーザーとGPT**：科学precision/lawを保持すること、新seed namespace、cooperative STOP、in-flight/unknown/no-refund、parent arrival order、結果/log予算と新trace粒度、parallel CPUを運用費用とするestimand、retry0/resumefalseを正式契約として判断する。
3. **Codex（別の明示指示後）**：採択されたscopeだけの新production bridge、全条件を含むglobal one-shot budget/identity、独立offline adapterを作り、実科学importを伴わない人工受入・SOURCE固定・公開を行う。科学SOURCEを変える必要があれば差分を再び判断へ戻す。
4. **ユーザーとGPT**：その固定SOURCE/input/runtime/契約の正式採択と、別の科学authorizationを行う。trusted store/approval原本を決める。既存claimを再利用しない。
5. **Codex（さらに明示された実行scopeだけ）**：admission前後照合、単一atomic claim、実correctness/採択pilotを一回実行し、障害ならSTOP。結果をraw/receipt/費用/unknown付きで公開して再びSTOPする。

これは次の作業の区分であり、現在の作業や現在の承認を意味しない。今のSOURCEにboolを渡して実行可能にする手順は存在しない。本packet公開後はmandatory STOPし、Full-Nやpilotへ自動移行しない。
