# O2-SCV224：一campaignのSOURCE固定

2026-11-07 JST。ユーザーが[実行仕様](EXECUTION.md)のmetadata一回・224 physical shots一campaignを採択/実行許可した。
準備固定560c6092の旧SOURCEを変更せず、新activation/入口/人工テストだけ追加した。新activation12件＋既存106件＝118人工テストPASS。
初回のmock parent testにはcollectorのclassical timer用raw属性が欠けており、fixtureだけを補って再検証した。実科学入力を使った修正/retryではない。初回logも保存した。

新SOURCE/依存環境/inputのbyte lockを実入力前に公開し、remote SHA/本文/blob/tree照合後、既存qualified managerで一回だけ起動する。
ここまでの実H4群/ν/Σ、science PRNG、Born、測定、実build/compileは0。科学結果はまだUNKNOWN。
固定条件は保存linear H4/1 Å/STO-3G/legacy DF12/全4電子sector/8qubits/L_D3/m0/q1/δ.02/r1/K2/atol0、X/Y、N1 actual Q/selected full・N2保証state・認証済exact Fraction engine。

SOURCE/環境はPROFILE_LOCK.json、入力の静的bindingはINPUT_BINDING.json、人工検証はARTIFICIAL_TESTS.log。
metadata固定→X/Y train計64→両fit/freeze→同一fresh評価160のpaired比較。一campaign最大224、science retry0。
既存成果397/B1339/752を混入しない。元main・旧SOURCE/結果/STOP・C1凍結を保全する。

完了/中断時に取得済み/未完了call・費用・結果を本REPORTへ追記し、公開照合後mandatory STOPする。
