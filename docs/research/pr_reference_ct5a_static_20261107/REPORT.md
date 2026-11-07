# 新A CT-5A：T-count差の構造的分解（静的調査）

- 日付：2026-11-07 JST
- 固定参照commit：`8012c3e9547690f0d722edac9d0b9a9391ba2f8e`
- branch：`analysis/pr-reference-ct5a-static-20261107`
- 状態：**STATIC ANALYSIS COMPLETE / MANDATORY STOP**
- 範囲：公開済みCT-4B/V3/CT-0/CT-1/CT-3Aの文書・SOURCEと保存値の照合、および条件式の記述のみ。
- **新規合成、科学数値再解析、word価格取得、回路build/compile、sampling、simulation、precision/shot最適化は行わない。**
- 本文は技術的な静的整理であって、独立詳細科学レビュー・全support T優越保証・新しい正式policyの採択ではない。

## 1. 目的と固定条件

CT-4Bの **q=128、r=1、strict RZ精度 `epsilon_G=10^-11`** に対する2保存trajectory（rare/nonrare、j=0、X/Y、B′/Reference）のT価格差を、trajectory非依存の構造項と依存項に分ける。これは、新たな全域fallbackの実装が本当に必要か判断するための技術的な前処理である。

物理・回路条件は保存CT-4Bに従う：Linear H4 / 1.00 Å / STO-3G / DF12、元H厳密groundを各shotで供給し準備費除外、Partial S2・LD3・h=0.02・finite RTE K2、ordinary `diag(I,V)`、canonical independent finite-bit law、Qiskit1.3.0/compiler A（`rz/sx/x/cx`、全結合）、B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・8-CX CSWAP、D境界融合・basis共有・scalar/event・X/Yを固定する。

主資源は**採用strict wordのT+T†の物理出現数**。最小T、T-depth、誤差込み18round総T、実機space-timeとは異なる。qubits B′9/Reference17も変更しない。

## 2. 保存済みの有限結果（新規再集計なし）

[CT-4B REPORT](../pr_reference_ct4b_20261107/REPORT.md) による、1 axis／1 shot当たりの実word価格：

| 保存trajectory | B′ T+T† | Reference T+T† | Referenceの削減 |
|---|---:|---:|---:|
| nonrare/j0 | 9,307,406 | 9,133,736 | 173,670（1.865933%） |
| rare/j0 | 9,230,078 | 9,056,398 | 173,680（1.881674%） |

X/YでT数一致。299 word（既存111、新規188）のstrict guard、25局所対象、保存199 shot/axisの必要十分条件のうち**報告された十分条件**はCT-4Bの範囲でPASS。全supportのword取得・価格・CT版RPE成功は未資格。

保存[ROLE_T_COUNTS.json](../../../artifacts/pr_reference_ct4/2026-11-07/ROLE_T_COUNTS.json)とCT-4B報告における両trajectoryの共通部分：
- B′のD未融合T価格：8,235,520、D境界のsigned補正：−420,624。
- ReferenceのD未融合T価格：8,174,080、D境界のsigned補正：−536,760。
- Referenceの16 CSWAPの保存T価格：17,472、B′は当該roleなし。
- RTE basis frame/inverseはpaired eventに対し両方式の保存T価格が同じだった。
- RTE中央の費用、event/scalar/vacuum価格は、両保存trajectoryで値が異なり得る。
- 一部のroleのT=0は、**採用された有限word policyの結果**であって、実binary RZをexact Cliffordと認定した結果ではない。

上記のD固定価格とCSWAP価格は、固定`q,r,epsilon_G`・保存価格policyと、後述するrole分解／compiler適用条件の下でのみ全supportへ移す候補である。これだけで全trajectoryのT削減は証明しない。

## 3. SOURCEに見える構造的分割

[V3 SOURCE](../../../src/trottertracks/pr_reference_rotation_v3.py) の `dparts(arm,q)` は `q` と固定D primitiveからD未融合／D境界補正を構成し、event列 `omega` を引数に取らない。ReferenceのD境界には`n=q//2`とhalf/full blockの差分を使う。

`saved_path_parts(row,arm,axis)` がevent列から構成するのはRTE basis run、中央回転、product、B′ event位相と、Reference trajectory固有のvacuum Pである。`CSWAP`はReferenceに固定16出現、X/Y wrapperは固定構造。

V3の `pair(basis_id)` はforward＋actual inverse basisのRZ profileを共通に表現し、同じpaired event列で両armのframe価格は一致する**構造的候補**を与える。ただし、whole compilerのsource/final passで境界・product Zの取消しやcontext変更が生じる場合、各roleのliteral-native key histogramと一致する証明が別途要る。CT-4Bでは固定8 wholeで照合済みだが、全supportの証明ではない。

中央回転について、V3のroleモデルは、各eventの符号付きnative角 `theta` に対しB′が `RZ(theta/2)` と `RZ(-theta/2)`、Referenceが一つの `RZ(±theta)` を持つ。符号反転はstrict wordのactual adjointでありT数は変えない。したがって、同一政策・literal角・contextが成立する場合、個別eventの**T価格差の候補**は `2*t_e(abs(theta/2))-t_e(abs(theta))` である。angle arithmeticとcompiler後の実angle identityは必要な条件として保持する。

product、B′ event位相、Reference vacuum Pの価格はこの固定D/central型だけから決まらない。Reference vacuumのbinary64角は逐次加算順序を含み、CT-0で到達域を覆えても全support strict word/T価格を保証したわけではない。

## 4. 全supportの優越を語るための不足項目

詳細な**条件式と必要な境界**は [CONDITIONAL_BOUNDS.md](CONDITIONAL_BOUNDS.md) に分離した。

主要な論点は次のとおり。

1. 固定`q128/r1/epsilon_G=10^-11`のD・CSWAP・wrapper等を全trajectoryで同じliteral-native/strict word価格として保持できるか。特にV3のsigned D融合補正は**負の物理gateではない**。
2. Paired B′/Referenceでframe価格を全supportで同じと証明できるか。cancelled product Zやその他contextの影響を除く必要がある。
3. 有限event alphabetの中央回転・product・B′ event位相について、必要な全keyの実strict価格／境界を得られるか。2 trajectoryの観測最大をworst-case capとしない。
4. Reference vacuum Pの**到達全角度に対するword policy・上界**をどう扱うか。全域終了保証の不足（CT-2/CT-3A）は残る。
5. 全supportで実行可能なword policyと誤差保証を持たないまま、価格差の条件式だけでCT成功保証や実際のT削減を認定しない。

判定目標は二種類ある：trajectoryごとの `DeltaT(omega)>0` の**pointwise保証**と、`E[DeltaT]>0` の**canonical lawに対する期待優越**。後者の方が弱い条件だが、全support価格・確率会計が必要なことは変わらない。

## 5. 次に進める範囲とSTOP

この段階で、CT-3B（SK初期net・fallback実装）へ進む必要性は確定しない。

次はまず、Codexが**固定SOURCEと保存contextを使い、全supportのrole価格同一性・有限event alphabetのkey閉包・未資格境界を技術的に確認する限定scope**を別途検討する価値がある。保存値からの新しいT会計・極値・探索、key合成、native compileには新しい科学実行承認が必要。sourceで導ける条件だけで証明が閉じれば、その範囲でSTOPする。

現時点で既存保存2 trajectoryの「Referenceで1.87〜1.88% T減」は有効な有限結果。**全support T優越／全18round T優越・T-depth・FT全資源の優越は未資格。** 追加8 qubitと既存RZ増加を保持。

ここで **mandatory STOP**。旧SOURCE/raw/result/grant/ledger/marker/STOPを変更せず、次の科学的判断はGPT／ユーザーに戻す。

## 参照・再利用

- [CT-4B REPORT](../pr_reference_ct4b_20261107/REPORT.md)／[SCOPE](../pr_reference_ct4b_20261107/SCOPE.md)
- [CT-4B role価格](../../../artifacts/pr_reference_ct4/2026-11-07/ROLE_T_COUNTS.json)／[finite T](../../../artifacts/pr_reference_ct4/2026-11-07/PILOT_T_COUNTS.json)／[STOP](../../../artifacts/pr_reference_ct4/2026-11-07/STOP.json)
- [V3回転SOURCE](../../../src/trottertracks/pr_reference_rotation_v3.py)／[REPORT](../pr_reference_rotation_v3_20261107/REPORT.md)
- [CT-0 precisionと位相](../pr_reference_ct0_20261107/PHASE_PRECISION_FEASIBILITY.md)
- [CT-1有限correctness](../pr_reference_ct1_20261107/REPORT.md)
- [CT-2価格未資格](../pr_reference_ct2_20261107/FULL_SUPPORT_PRICE_APPLICABILITY.md)
- [CT-3A fallback未資格](../pr_reference_ct3a_20261107/REPORT.md)
