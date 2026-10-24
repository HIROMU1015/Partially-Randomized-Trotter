# H4 Pauli central 条件付きC/N/G

## SOURCE固定前

ユーザーの明示承認で固定10条件/15比較・guard端点・paired G統計1batchを準備。
既存SOURCE/16人工を再利用し、新14人工PASS。科学条件変更なし。
新SOURCEを先にcommitし、未使用one-shot grantで実行する。実ρ・母平均tailは未保証。
DF会計・quantum/compile/sampling/C-L実行へ範囲拡大しない。

## 初回停止と原因特定

初回SOURCE da979af72、入力bindingのみ実施。cross-representation reference全体の一致要求が過剰で停止。
元SOURCEでは表現別full Hからreferenceを作るため、hash/信号は別。
表現内一致と同一ray/T/methodを検査する修正で、各保存reference/biasは保持する。
新16人工PASS。N/G科学callは0、成功科学値の重複なし。新SOURCE/seal/grantで回復する。

## 回復完了・科学STOP

SOURCE d96dfe5ca0a6c950a5f1791ff61659e6dc8f1c34、新grant 37a5812aa9787e3a366576fde95fa02176621d449ee928c8d04c47265e288bf2。
10条件/20軸+20端点/15比較/paired G統計1batch成功。PR/PF G比.12434〜.12788。
PFとPRそれぞれ対応するDFより低G、全15比較は独立guard全域で反転なし。
実ρと母平均tailは未保証。旧genericよりPFの低下幅が大きいため、PR/PF比は旧約.06から約.125へ上昇した。
主agent124保存check PASS、code13/input600/runtime4896/旧9165/output28 SHA照合・旧manifest180行保全。
native exit0・回収、初回失敗STOPは維持。科学作業mandatory STOP。
結果commit/push・remote SHA確認のみ継続し、残差/tail/D3等へ移行しない。

## 公開監査

結果commit ef5c13d94639c5235cfd6486a17ac998344397f0。pushはHTTPS認証情報なしでexit128、remote branch検索0件。
新remote SHA未取得、失敗logと再実行コマンドを保存。後続は公開metadata commitのみ。
科学STOP・旧N/G null・元native UNKNOWN・hash FAIL/原因UNKNOWNを維持。
