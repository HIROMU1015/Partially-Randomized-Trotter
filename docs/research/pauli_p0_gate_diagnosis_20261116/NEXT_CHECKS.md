# 元P0を再開せず原因を確定する最小追加検証（未実行）

現在の保存JSONは残差・scalar・post-JW係数を持つが、DF tailのnormと集積中の係数を持たない。
下記を科学的な保存配列再解析・追加実H4診断として区別する。**今回の実行範囲には含めない。**
新しい実行許可があった場合も、既存P0のSTOPを解除したretryとはせず、別診断SOURCE・fresh outputで保存する。

## A. 演算子の大きさと後段の除外

保存`9fa9525b3:.../LD{7,8,9}/LD*_OPERATORS.npz`だけから、各splitの
`||tail||_F`、`||Pauli_nonidentity||_F`、trace/identity/traceless残差を取得する。
保存JW JSONから同じ8mode・bit reverseでPauli行列を1回再構成し、保存Aとの一致を確認する。
LD6を対照とするなら同じ処理1件を追加する（目的に必要かは実行計画で決める）。
上限は失敗3分割の保存operator評価3件・JW行列再構成3件、対照を含める場合は各4件。
Hamiltonian/state生成、固有値問題、reference、mean、RPE、compileは不要。

- 判定：後段再構成が保存Aと一致すれば、実測残差をcollector/行列化より前へ局在できる。
- 保存値との不一致・非finite・binding不一致が出た場合はそこでSTOP。閾値は1e-9のまま。
- scalarのみの説は既存のsaved-scalar対照で既に不十分なので、scalarを修正したP0再実行は不要。

## B. 実factorでの最小中間trace

最初にLD9の残りfactor1個（index9）だけを、固定snapshot・元weight/順序・P0/M1の同じ経路で追う。
比較用の保存tailはAで使う既存operatorを再利用する。
新診断では実行前にPython/OpenFermion等の実runtimeと内部閾値を記録し、元carrierとの確認不能な差を残す。

1. `one_body`へ渡す非zero成分と、FermionOperator集積前後の係数。
2. `a*a`とweight乗算の係数、`residual +=`で消えた寄与。
3. JWのFermion termごとの寄与とQubitOperator部分和で消えた寄与。
4. default経路は変更せず、局所的な無除去shadow係数集積で失った寄与を追う。
   libraryのglobal EQ_TOLERANCE・既存module・既存辞書は変更しない。
5. identityと非identityを分け、default/shadowの差が元のPauli–DF残差を説明するか確認する。
   shadowの正しさは保存tailとのoperator一致でも確認し、単に残差が小さくなっただけで採択しない。

この1分割traceでone_body/weighted/JW/複合のどこで失われるかを区別できる。
LD7/8まで原因確定するには、それぞれ残存factor集合`[7:]`（3個）、`[8:]`（2個）で同じtraceを
各1件実施する。**全失敗分割を解く有限範囲は3 split trace**であり、LD9のみの結果を他LDへ移植しない。
対照LD6の残存4factorを1件加えるかは、失敗traceに対する対照の必要性に基づいて判断する。

成功条件は、破棄された具体的label/係数・最初の演算段階が記録され、それらの差が保存残差と
同じ固定gate内で整合すること。そうでなければ未確定を維持し、原因未特定の追加探索へ自動進行しない。
新しい辞書を本番へ採用するには、原因診断とは別に変更方針・scientific SOURCE・P0再資格の承認が必要。

今回の人工例は機構の検証だけで、上記A/Bは未実行。価格・signal・schedule・λ_P再採択は含まない。
