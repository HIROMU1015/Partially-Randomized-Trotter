# Q3復旧に必要な最小条件（未実行）

新しい信号は不要。完成13roundのsource-bound packetと、指定の同一Y4 B_PRICE列を再利用できる。今回のraw/STOP/source・未取得MSEを保存したまま、Q3の復旧を別に判断する。

1. 現実行Python3.12.3と、Acceleratorの期待AST schemaの不一致を解消する方法を決める。候補は、期待schemaを再現するproject専用interpreterの利用、又は**空のtype_paramsのみ**を対象にした厳密なcanonical AST比較。後者でも非空type parameterやその他SOURCE差は拒否する必要がある。現guardを無効化、期待hashを無検査で置換、数値gateを緩和する方法は採らない。元B実行interpreterの所在・完全対応は本成果では未確定。
2. 技術修正後は、実worker入口でAccelerator初期化が通り、人工DPで元のexact loss/質量/残余/状態遷移と一致することを確認する。今回追加した静的schema診断は原因特定であり、その互換修正の同値性検証を済ませた証拠ではない。
3. 新caller/SOURCE・runtime依存・保存packet・同じX/Y配分・新出力namespace・最大1 Q3を固定する。旧q3_run01はresume/retryしない。新SOURCEに元数値・guard条件との差分を明示し、§10.5で同候補/同配分の後続結果・進行jobを再照合する。
4. ユーザーがこの復旧Q3を明示承認した場合に限り1件実行する。今回のretry0・mandatory STOPは解除されておらず、科学API呼出し0件だったことを自動的な再投入許可と解釈しない。

固定する科学契約は元Z1＋T1、DF10 solver proxy、IID deterministic raw、全事象平方loss、exact τ、指定Y4配分。AST比較は技術SOURCE識別の修正案で、科学的なscalar/step/decoder/loss定義変更を提案していない。ただし、修正の実装・検証と再実行SOURCEを確認するまでは互換性を認定しない。

新mean/reference/step・回路build・compile・価格・fit・別配分は復旧案にも不要。Q3後も粗幅費用は未評価のまま。価格pilot等は別の有限案と承認が必要。
