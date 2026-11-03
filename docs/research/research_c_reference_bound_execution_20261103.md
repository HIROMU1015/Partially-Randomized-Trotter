# 研究C：B2状態非依存reference上界・固定27-call実行結果

2026-11-03。完成済み固定B2契約だけをfresh `run01`で一度実行し、**27/27 calls COMPLETE、FAIL/retry/resume0**で終了した。
Statusは `B2_REFERENCE_OPERATOR_BOUNDS_COMPLETE`、wall timeは3.771047710441053秒。
科学計算はこの27-callで停止し、Stage C・native・sampling・shots・重要科学レビューは開始していない。

## 固定条件・SOURCE

対象はLinear H4・隣接1.00 Å/STO-3G、Nα=Nβ=2、sector36、Number CDF rank8、元frame順LD2、
partial ordered symmetric S2、s1/r32/K0、h=δ=.1、m0..12。CA=.00159 Ha、DF reserve=CA/8、own-DF ground oracle・状態準備費用除外。
各event/coherent occurrence/measurement shotで独立な53-bit IID-gridを条件とする。実PCG64のIID保証ではない。
B2は任意の正規化sector状態へのoperator上界を評価し、保存ground stateを読み込まず、ground solveもしない。
保存P0（Qiskit1.3/opt1/seed17/rz,sx,x,cx/MPFR160 declared phase/Gaussian・relative fusion）は変更していない。

- Endpoint SHA256：`a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
- 元PR科学SOURCE：`4df8eca29c59ffd55e2dd898f05da16bb246ddbe`。
- Stage A科学SOURCE/結果：`ddd8333488e07a3b3af9095b5cb193ef9f90b3c3` / `ae9a4e13d1ab6f48fc4d38cbbcd0fddca30e9c53`。Stage Aの38-call結果は再実行・変更していない。
- B2実装SOURCE：`918c63cc2a959f59f2817388bf6be549b905e3e6`。実行時commit：`01baf2b5e400773e00a5ae548941265f85b9882b`。
- [固定準備・対象定義](research_c_reference_bound_contract_readiness_20261103.md) / [固定契約](research_c_reference_bound_contract_unapproved_20261103.json)。
- SOURCE manifest SHA256：`07eef25fe16ee42beaad79f02caacb406a7ca08b71ce72e973c7f6b1521ac443`。
- 契約SHA256：`a346b066f65e071f4e8c5d6912dfb8e62a45d9eea260161d8c4e04617b29c44b`。

ユーザーの実行許可は固定27-callに対する会話の明示許可として適用した。準備時の未承認proposalは原本のまま保存した。
`execution_authorized=false`を含むその履歴文言を書き換えたり、新grant・証明書・承認原文保存を要件化したりしていない。
科学SOURCE174 file、契約8保存入力、継承SOURCE/input/STOPのbyte照合・fresh preflightを通過した。

## 実際に囲った二項と結果

Mathematical atom map M_atomは、保存216atomの正確な係数/p、literal saved Uで構成するCAR Z/ZZ、保存angle bit、
正確なhead半時間1/20とtail identity phaseから定義する。
C_mathは保存G0..8とλ/τ/identityを正確なdyadic入力としたK0 corrected kernel。
R_math,m=(1+τ²)^(-16*2^m) C_math^(2^m)。
R_num,mは保存corrected matrixをbinary64で反復二乗し、各保存attenuationをbinary64で乗じた評価行列。

```
B_m >= ||M_atom^(2^m) - R_math,m||2
U_m >= ||R_math,m - R_num,m||2
|psi†(M_atom^(2^m) - R_num,m)psi| <= B_m+U_m  (||psi||=1)
```

Arb384/python-flint0.8.0/FLINT3.3.1で指数・積・冪・scalar・差・Frobenius normの丸めを含めて評価した。
上界は全行列成分のoutward Frobeniusからoperator2へ換算したもので、浮動小数SVD残差の再分類ではない。
保存stateの精度・ground oracleへの距離を認定せず、任意正規化状態への不等式でraw complex signal誤差にする。
Re/Im・位相を保持し、attenuationの再乗算や2によるclampはしない。

下表は保存exact rational上界を12桁で上向き表示したもの。正本は[result.json](../../artifacts/research_c_reference_bound_contract/2026-11-03/run01/result.json)の`upper_exact`。
単位は無次元raw complex signalの絶対誤差。科学入力や条件を変更した追加集計・精度再実行はない。

| m | Bridge B_m | Reference uncertainty U_m | B_m+U_m |
|---:|---:|---:|---:|
| 0 | 2.80589760013E-17 | 1.00434921484E-14 | 1.00715511244E-14 |
| 1 | 5.60959211281E-17 | 2.06002983908E-14 | 2.06563943119E-14 |
| 2 | 1.12024925813E-16 | 4.08928655770E-14 | 4.10048905029E-14 |
| 3 | 2.22777896755E-16 | 8.15060650564E-14 | 8.17288429531E-14 |
| 4 | 4.36389675805E-16 | 1.60876826293E-13 | 1.61313215969E-13 |
| 5 | 8.20397228689E-16 | 3.08199510991E-13 | 3.09019908219E-13 |
| 6 | 1.47416109437E-15 | 5.66688690246E-13 | 5.68162851340E-13 |
| 7 | 2.66722992362E-15 | 1.01550279105E-12 | 1.01817002098E-12 |
| 8 | 5.11599414154E-15 | 1.89236066926E-12 | 1.89747666340E-12 |
| 9 | 9.90536497909E-15 | 3.66190233363E-12 | 3.67180769861E-12 |
| 10 | 1.92948905342E-14 | 7.14525972898E-12 | 7.16455461952E-12 |
| 11 | 3.69761099130E-14 | 1.37229719450E-11 | 1.37599480549E-11 |
| 12 | 6.84027481910E-14 | 2.53822642422E-11 | 2.54506669904E-11 |

最大表示値はm12のB_m≤6.84027481910e-14、U_m≤2.53822642422e-11、合計≤2.54506669904e-11。
これは**固定した保存入力の数学的referenceに対する包囲**。元Gの生成精度や物理Hamiltonian全体の誤差を含む主張ではない。

B_mはatom→数学CARの表現bridge、U_mは数学CAR→binary64 referenceの評価誤差であり、別々のtriangle leg。
Arb半径は各legの上界へ包含済みで、別のreference uncertaintyとして二重加算していない。
Categorical p→qのStage A上界は再計算・再加算していない。PF/corrected-RTEとDFもこの二項へ混ぜていない。

Stage Aのideal-atom対CAR m12 operator差観測約1.08677e-10と、今回B_mを同じ量として比較しない。
前者はbinary64で構成したatom kernelとCAR replayの差の観測。今回B_mは数学的atom mapと数学的CARの差で、
U_mは数学的CARとbinary64 CAR replayの差。Stage Aのbinary64 atom計算誤差やsaved-state expectation自体の誤差を取得したとは言わない。
旧Stage Aのnull・観測値を上書きせず、新しい固定target定義の二項を別の証拠として提供する。

## Calls・資源・保存成果

| 登録単位 | COMPLETE |
|---|---:|
| Metadata / identity / arrays / sector / atom context / atom mean | 6 |
| Head exponential | 3 |
| Head products / CAR tail / kernel pair / shared powers | 4 |
| Round bound m0..12 | 13 |
| Summary | 1 |
| **計** | **27** |

[call journal](../../artifacts/research_c_reference_bound_contract/2026-11-03/run01/science_calls.jsonl)は27 attempt/27 COMPLETE、FAIL0。
[STOP](../../artifacts/research_c_reference_bound_contract/2026-11-03/run01/STOP.json)はretry/resume0、first_failure=null、mandatory_stop=true。
出力は[実行SOURCE](../../artifacts/research_c_reference_bound_contract/2026-11-03/run01/SOURCE.json)、
[実行契約](../../artifacts/research_c_reference_bound_contract/2026-11-03/run01/contract.json)、result、journal、STOPの5科学記録。
run専用Matplotlib cacheは科学証拠としてstage・公開しない。原本は削除しない。

CPU affinity32、実行内のavailable memory約55.43 GB、head3 process・round13 processを使用した。
二本のArb chainと一本のbinary64 chainを共有し、各chainの12依存squaringとモデル上2^m個のcoherent occurrenceを区別した。
各workerのBLAS1でoversubscriptionを避け、Arb contextはprocessごとに分離した。
CPU/RAM/wallの人工capや共有環境設定変更はない。実測比較がないspeedupは主張しない。

## Qualificationと未解決事項

B2のresultは`reference_bridge`と`reference_uncertainty`の各13row、`rigorous_bound`/`rounding_enclosed=true`を出力した。
これは指定のliteral saved-atom / mathematical saved-G CAR / numeric reference間の二項だけの資格である。
全体の固定Hamiltonian・native law・実PCG64・CAの保証へ拡張しない。
既存統合schemaのidentity・sector36・raw units・disjoint tagsは維持するが、今回は統合adapterやStage Cを科学的に実行していない。

未資格は引き続き次のとおり。

- Whole-kernel P0の全word一様synthesis、inverse/shellを含むB1の不足。
- Literal Uのatom mapからcategoricalが仮定するexact unitary trajectory mapへの対応。
- 保存generator構築の数値誤差、元積分/DF/物理モデル誤差、既存PF/RTE numerical pilotの保証水準。
- 保存ground stateの精度認定、実PCG64 joint stream、Stage C十分条件・actual/rigorous CA、保証shots・総資源。

新ground solve、native build/compile、sampling、PF誤差評価、shots/energy評価、重要科学レビューは0。
科学処理は27-callでmandatory STOPした。統合の不足を理由に次の科学stageへ自動継続しない。

## 保全・次の担当

固定SOURCE・入力・元契約を変更せず、旧Stage A/B1/B2/統合のraw・STOP・grant・ledger・markerを保全した。
元checkoutのHEAD/index/開始時の未コミット状態と、固定SOURCE174 file・各入力を照合する。
追加した研究記録はこの結果報告、概要・dated note・manifestと必要な入口だけ。

[実行・静的確認記録](../../artifacts/research_c_reference_bound_contract/2026-11-03/execution_verification.json)で、
全27 call/13row、5科学出力のdigest、SOURCE174 file・契約入力の一致、消費済みfresh拒否、報告リンクを確認した。
準備時の355人工・回帰PASS（既知の旧2件除外）から実装SOURCEに変更はなく、科学処理の再実行・追加pytestは行っていない。
Manifestの旧commit entry[37]に`commit`キー・`sha/description`欠落の既知schema不整合があり、
全体checkerは起点と同じ理由でFAIL。今回の追加entryだけのschema・参照検証はPASSし、旧配列prefix・監査statusは保持した。
元checkoutはHEAD `e098c54c78f589055082f9cfc2b13de50c90ca94`、index SHA256
`68a93531c7bc1993d49f6750ac8d98a54fafb8e153ddd0b887dd17b28ddd9c4c`、開始時status178行と一致した。

次の担当はCodex（技術統合）：公開したB2のtarget/誤差会計とB1のcoverageを照合し、統合に残る前提を整理する。
Stage Cや追加の科学評価・重要レビューが必要なら、具体的な固定範囲へのユーザー許可を別途求める。
今回は結果のcommit・push・remote SHA/tree/本文照合で終了し、mandatory STOPを維持する。
