# 新A N1：保存入力によるvacuum T価格境界の限定数値評価

2026-11-08 JST、SERVER 2（GPUA100-gleap02）。専用branch `codex/pr-reference-n1-numeric-20261108`。

**指定境界と検算を完了し、mandatory STOP。判定は`CONDITIONAL_THRESHOLD_EVALUATED`。** CT-5Bのcompiler条件Cと同じ採用wordを維持する全support policyの完了を前提とした、保守的十分条件の要求値を求めた。実vacuum上界・全support T優越を証明した結果ではない。

## 固定scope・入力・SOURCE

科学入力基準は **`0dbf1fe265176677e026ea982c0bd435987f4338`**、準備commitは **`bf0a7b1b5d0afb6ead3cc83ce909bbe53514b24a`**。[承認対象のNUMERIC_EXECUTION_SCOPE](../pr_reference_n1_preparation_20261108/NUMERIC_EXECUTION_SCOPE.md)と[入力／算術仕様](../pr_reference_n1_preparation_20261108/N1_INPUT_AND_ACCOUNTING.md)に限定した。

Linear H4・1.00 Å・STO-3G・DF12、元Hの厳密ground供給・状態準備費除外。Partial S2・LD3・h=.02・finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、m7/q128/r1、保存73 bits、一様strict epsilon=10^-11。Qiskit1.3.0/compiler A、rz/sx/x/cx、全結合・Targetなし。

B′の4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum補正・2 register・8-CX CSWAP、既存D融合・basis共有・scalar/event位相・X/Y wrapperを保持。energy精度1.5936e-4 Ha、全失敗率.05、phase cap .4、alpha1/720、保存199 shot/axisは不変更。これらは保存条件の照合であり、新しいcompiler実行や誤差再最適化ではない。

入力は準備時の[INPUT_REFERENCES.json](../../../artifacts/pr_reference_n1_preparation/2026-11-08/INPUT_REFERENCES.json)に固定した22ファイル。各ファイルを科学入力基準commitから読み、SHA256・bytes・Git blobを照合した。CT-4Bの最終ROLE_T_COUNTS／PILOT_T_COUNTS／KEY_REUSE_AND_SYNTHESIS、4保存strict record、FIXED_INPUTS、B′の保存p2、V3/CT-0のm7条件を使用した。旧科学module全体のimportや旧runnerの再実行は行っていない。

新しいSOURCEは[run_pr_reference_n1_numeric.py](../../../scripts/run_pr_reference_n1_numeric.py)。標準ライブラリのみで保存入力の整数／Fraction会計を行う。CT-0の既存`positive/validate_record`関数をASTで取り出し、その純粋metadata処理だけを再利用した。旧module全体のimport、guard再評価、runner再開は行っていない。新SOURCEのSHA256と入力参照hash・実行条件・全検算は[RESULTS.json](../../../artifacts/pr_reference_n1_numeric/2026-11-08/RESULTS.json)に保存した。

## 境界の厳密値

単位はすべて、**固定q128/r1・1 axis・1 shot当たりの採用物理T+T†数**。最小Tや199-shot積、18round総Tではない。

| 固定項 | 値 |
|---|---:|
| D_B*（未融合＋signed境界補正） | 7,814,896 |
| D_R*（未融合＋signed境界補正） | 7,637,320 |
| Csw*（Reference固定16 CSWAP） | 17,472 |
| SB*（B′ scalar） | 118 |
| K = D_B* − D_R* − Csw* + SB* | **160,222** |
| d0 = 2t(h0) − t(f0) | **106** |
| d2 = 2t(h2) − t(f2) | **114** |
| dmin | **106** |
| V_threshold = K + 128 dmin | **173,790** |

保存lawのp2を文字列からFractionとして読み、分布を再生成していない。

```text
p2 = 642545340934082409/9444732965739290427392
U_threshold = 1602930468830332421754729/9223372036854775808
```

`U_threshold = K + 128[(1-p2)d0+p2 d2]`の40有効桁の外向き表示は

```text
[173790.0696649054561171268739119821766792,
 173790.0696649054561171268739119821766793]
```

である。正本は厳密Fraction。小数下端はROUND_FLOOR、上端はROUND_CEILINGで整数分子／分母から直接生成し、その区間への厳密値の包含を検算した。

### Pointwiseと期待の条件

[CT-5B条件Cと共通word policy P](../pr_reference_ct5b_20261107/CONDITIONAL_T_ADVANTAGE.md)の下で

\[
\Delta T(\omega)=K+\sum_{i=1}^{128}d_{o_i}+E_B(\omega)-v_P(\phi_\omega).
\]

全supportで`v_P(phi_omega) <= Vmax`を別途保証でき、**Vmax < 173790**なら、pointwise T優越の保守的十分条件を満たす。canonical lawで`E[v_P] <= UV`を別途保証でき、**UV < 1602930468830332421754729/9223372036854775808**なら、期待T優越の十分条件を満たす。等号では厳密優越を保証しない。

E_B>=0を下界から落とした条件であり、一般の優越に対する必要条件ではない。vacuum・order・product符号は同一trajectoryに依存する。期待式には線形性を使い、それらの独立性を仮定していない。共通frame価格の取消しはword取得／合成誤差の取消しではない。signed D補正を負の物理gateやnormとして使っていない。

## 指定検算

| 検算 | 結果 |
|---|---|
| 固定入力22ファイルのhash・schema／row一意性 | PASS |
| 8保存行のprojection資格、D各項・CSWAP・B scalarの一致 | PASS |
| 中央4 literal keyのe=-11 record／spec／sequence SHA／W／strict metadata／実word文字数 | PASS。strict guardの再実行ではない |
| 保存p2の一致・有理確率の範囲・m7/q128/r1/bits73/199shot条件 | PASS。新確率生成なし |
| 期待閾値と等価式`K+128*d0+128*p2*(d2-d0)`の一致 | PASS |
| 8行のsigned role T和と保存直接native Tの一致 | PASS |
| 同じ保存trajectoryを使う4 paired差分恒等式 | PASS。下表 |
| 外向き小数区間への厳密Fraction包含 | PASS |

| 保存trajectory | axis | order0 / order2 | B′ event T | Ref vacuum T | 式／保存直接会計のΔT |
|---|---|---:|---:|---:|---:|
| nonrare/j0 | X | 128 / 0 | 0 | 120 | 173670 / 173670 |
| nonrare/j0 | Y | 128 / 0 | 0 | 120 | 173670 / 173670 |
| rare/j0 | X | 127 / 1 | 0 | 118 | 173680 / 173680 |
| rare/j0 | Y | 127 / 1 | 0 | 118 | 173680 / 173680 |

2 trajectoryは既存有限価格式の検算にだけ使った。vacuum120/118を全support価格上界や期待値へ転用せず、rare/nonrareを無加重平均していない。採用recordの`T`はT+T†合計なので、内数のTdaggerを重ねて課金していない。中央負角にはactual adjointを使う既存policyを保持した。

## 未資格・結果の限界

実際に返すpolicyに対するVmaxとUVは**取得していない**。全support vacuum wordの取得・有限停止・strict保証、common frameの未取得word、到達する場合のB′ b2 event word、全support CT RPE成功は引き続き**UNQUALIFIED**。入力角域が有限／有界というだけでT上界や有限停止を資格しない。

N1で得たのは、後続に要求するvacuum価格境界の数値である。CT-4Bの有限結果を全supportへ拡張したり、Referenceの総RPE費用優越・部分ランダム化一般の改善・新規性を認定したりしていない。未知価格を0にせず、改善率や新しいT価格は取得していない。全18round総T、T-depth、FT費用は未取得／未資格。追加8 qubitやnative CX/RZ/depthとのtradeoffをTへ合算しない。

保存RPE誤差・shot条件は照合のみ。compiler diskと合成diskの新しい再会計、phase/radius変換、shot・alpha最適化は行っていない。全supportの成功を将来資格するには、全物理位置の正しいstrict誤差を旧diskへ同じ単位で一度だけ加算する証拠が別途必要になる。

## 実行・保全・STOP

SOURCEを構文確認後、指定scopeの再解析を1回実行した。Python3.12.3、64 CPU利用可能なSERVER 2で、保存入力処理は1秒未満だった。小さい有限算術であるためworker追加は不要で、CPU/RAMの独自上限や共有設定変更は設けていない。著者Codexの再会計検算であり、独立科学レビューではない。

新規合成・word生成・strict guard・回路build/compile・新価格取得・sampling/simulation・確率再生成・precision/shot/alpha/fusion最適化・SK/CT-3B・N2/N3・199-shot積／18round総T／T-depth計算はいずれも**0**。新しい専用SOURCE／出力／本報告だけを追加し、旧SOURCE・raw・result・grant・ledger・marker・STOPを変更していない。他ユーザー／別Track／共有環境も操作していない。

[STOP.json](../../../artifacts/pr_reference_n1_numeric/2026-11-08/STOP.json)のとおり、承認範囲は完了し**mandatory STOP**。次担当は**ユーザー／GPT**：この要求値に対し、vacuum上界・取得policy・残る有限word資格のどの証拠を次に求めるか判断する。Codexは追加の明示承認なしにN2/N3・科学レビュー・合成・全support価格・18round計算へ進まない。
