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
