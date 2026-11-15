# C→統合担当/B/D1：Full-H κ2 m6の下界調査

**順位未判定。保存根拠から今回数値化・正当化できた `L6_lower=0 RZ` は、p7 S2の593,555,352 RZに届かない。不足額593,555,352 RZ。**

入口：[REPORT](REPORT.md)／[DECISION.json](DECISION.json)／[SOURCE_INDEX.json](SOURCE_INDEX.json)。対象m6のphysical ID `FULL_H_4ec8d73571760a6830b8`、q64/R566/K2、X/Y各57 shot、元full measured controlled wrapper・new ZZ・Qiskit1.3.0/rz-sx-x-cx/opt1/seed17/P0を保持。

0下界は全合法path・任意lawに対するcompiled RZ count非負性。実際の最小価格が0だとは主張しない。非自明な生存下界や適用可能な片側誤差評価を、確認した保存SOURCEでは得られなかった。

- 決定論PFのQPY反復障壁はFull-Hの構造と異なる。Full-Hはdeterministic headなし、basis共有とwhole-wrapper最適化を行う。event数/構造S/単独wrapper価格はそのままpost-RZの下界ではない。
- D7の1 fresh path×X/Y、モデル部分額、小さいholdout誤差は下界ではない。fit/比較/予測は再計算せず、原role・qualification・STOPを保全した。
- 旧Full-H contextはH/state/snapshot一致でも、prep hashが旧 `0e3989…1046d` とD7 `923253…aa100` で異なる。旧368 component/probability表の現m6移送は保留。旧価格もK4/q16/R16で、population boundはnull。原因は未同定で、D7自身のreceipt受入を取り消すものではない。
- 次の重要な証拠は、同じD7 descriptorと全合法外部continuationを対象とする `C_X+C_Y≥h(ω)` のpost-compiler生存不等式。D7 hashに一致する保存完全support descriptorが入力として必要。

## 次の有限案

[NEXT_PLAN.md](NEXT_PLAN.md) を参照。まず新path/compile0の保存descriptor回収と生存不等式を優先する。

取得が必要な場合の具体案として、事前H=11,000,000 RZに対する成功事象を使う片側検定を提示した。仮定付きパラメータα_price=1/20、事象law差δ_A≤1/200では、prospective **61 fresh path／最大122 wrapper** の全成功規則でサイズ≤5%を示せる。61はその提案規則での最小。現PCG64 joint law・独立性・δ_Aは未資格、confidence policyも未採択なので、**現在compile実行可能という引渡しではない**。事象lawの片側差を直接扱えば価格全support変動幅を要しないが、その差を0とは置かない。

旧m6 pathは設計参考のみで、新標本へ数えない。1件失敗/欠測/gate不成立で不成立/UNKNOWNを保存しSTOP。閾値やcohortを変更したretry、他roundへの展開は行わない。

## ロードマップ差分・保全

「Full-H κ2固定1.5倍配分のm6単独p7比較は未判定。全supportの0下界のみ、母期待部分和のpost-compiler証拠不足。旧Full-H prep hash差を明示し、D7 hash一致descriptorを次入力とした。条件付き有限片側検定案は未採択」。ロードマップ本体は統合担当が更新する。

今回：保存SOURCE読取・下界導出/比較・未採択案の数学的サイズ確認。sampling/build/compile/fit/Q3・D7/F1再計算0、新CIなし。F1全額nullと旧STOP・係数・資格を保全。公開・remote照合後mandatory STOP。
