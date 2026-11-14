# 事前採択用：許容差と比較policyの提案

**提案のみ。採択していない。m6の新価格は読んでおらず、取得・fit・予測もしていない。**

## 推奨：H-b、固定相対2%、観測価格を分母

各方式・公式・軸に、同一の次の判定を用いることを推奨する。

`|C_obs(m6) − [a+b q_m6]| ≤ (1/50) C_obs(m6)`

`a,b`は同方式・同公式・同actual幅・同軸のm4/m5だけで求め、m6取得前に固定する。X/Yを平均せず、Full p10とTruncated p7 B0Iも別々に判定する。全12判定がPASSであることを、この検証campaignの成功条件とする。任意の1判定がFAIL、欠測、SOURCE/gate不一致なら部分結果を保存してSTOPする。fit更新、feature追加、別幅、許容差緩和、追加compileを自動で行わない。

既存`price_model_validation.validate`の`RELATIVE`は **`scale=observed`**。そのまま使う入力は：

```python
tolerance = {
    'kind': 'RELATIVE',
    'value': {'numerator': 1, 'denominator': 50},
    'source': USER_ADOPTED_POLICY_SOURCE,
}
```

現時点で`USER_ADOPTED_POLICY_SOURCE`は未確定であり、実validationへ渡さない。元案H-a/H-c/H-dに書かれた`C_pred`分母とは区別する。元の検証器・許容差は変更しない。

2%は決定論価格に対する**運用上の事前合格条件**であって、統計CI、2点fitの科学的正しさ、全roundの2%誤差保証ではない。有限cohortのSEで決まる値でもない。固定5%はより緩い別案で、ユーザーが結果を見る前に選ぶ場合だけ使用できる。2%を推奨する理由は、比較marginが未確定の段階で約9%や5%より粗いmodelを容易に通さず、少数holdoutによる初期検証として単純で全12domainに同じ条件を使えるためである。

## H-a〜H-dで確認できる範囲

|案|検証するもの|今回の判断|
|---|---|---|
|H-a、構造proxyの価格差の1/3、約9%|宣言した許容差にm6のmodel誤差が収まるか|proxy p7/p10≈0.727はcompiled価格差の証拠ではない。両方式の誤差、共有systematic、全roundでの逆転を排除できず、今回の正式価格精度の根拠にしない|
|H-b、固定2% / 5%|指定した相対誤差のm6点での適合|2%を推奨。5%も含め事前に一つを選ぶ。どちらも長qへの保証ではない|
|H-c、絶対誤差の下限付き|小価格で相対判定が過敏になることの調整|今回の決定論PFで、結果を見る前に絶対許容RZ数を正当化する根拠がない。追加しない|
|H-d、SEまたはcohort区間込み|有限cohortの標本変動を含む検証|今回の固定決定論回路には不要。n1を標本分散0の証拠にせず、価格modelの系統誤差へ流用しない|

m6 holdoutは**較正・model選択に使わない点**という意味で独立であり、決定論compiler価格を確率的IID標本と呼ばない。したがって、この12判定を統計的信頼区間や全候補の成功確率へ読み替えない。

## 比較policy案（未採択）

- 共通profile：Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、`symmetric_directional`、`full_Hadamard_with_measurement`、P0。
- native policy：両方式とも`H46_LONG_TIME_NATIVE_COMPOSE`、RTE loweringは`NOT_APPLICABLE_NO_RTE`。
- Full PF scalar quantity：`HAMILTONIAN_CONSTANT`。Truncated B0I：`HAMILTONIAN_CONSTANT_PLUS_B0I_TAIL_IDENTITY`。ともに総Tに一回。同じ数値であることを要求するのではなく、各H/切断定義に対応した値を要求する。
- METHOD・保持prefix・scalar・作用gate・SOURCE・physical ID・actual幅・軸・quantityはC guard/Dの元検査で確認し、旧receiptを新しい取得cohortへ読み替えない。
- 比較対象は全3公式、同round・同軸・同compiled quantity。piece数のみで公式を排除しない。

## 外挿とG_paperでの扱い

直接価格が受け入れられたm4/m5/m6は`DETERMINISTIC_DIRECT`にできる。m6 PASS後も`validate`は`qualified_for_candidate=False`を維持する。modelは外部資格なしでは`UNQUALIFIED_MODEL`である。`QUALIFIED_MODEL`への昇格は本件では行わない。

m7〜m12（最大q S2=1792、Y4=768、M5=256）は未検証外挿のまま。m0〜m3の異なるactual幅へも移植しない。m6での成功は他のq・SOURCE・native/loweringに対する同時誤差界を与えない。

全cellで根拠のある同時誤差界`|C_hat_i−C_i|≤b_i`と固定非負Nがあるときだけ、`|G_hat−G|≤Σ_i N_i b_i`を使える。例えば全cellに同じ観測値相対誤差rが保証されれば、各価格は`[C_hat/(1+r), C_hat/(1−r)]`で囲え、方式のG区間が離れるかで順位を判定できる。**今回のm6判定だけではこの前提は成立しない。** 相関を独立として誤差を縮小しない。

第1水準G_paperはnominal Nを使う別の比較で、MSE/risk-qualifiedな正式Gではない。direct価格を受け入れても他roundの欠測は残り、全RPE G_paperを算出したことにはならない。未資格modelを用いる場合は未資格scenarioのtierを維持する。現時点でG_paper・正式G・winnerはいずれも未計算。
