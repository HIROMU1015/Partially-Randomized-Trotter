# N1：別途承認後の最小数値実行scope

2026-11-08 JST。[REPORT](REPORT.md)、[入力と算術](N1_INPUT_AND_ACCOUNTING.md)。**未実行の提案仕様**であり、今回の実行許可ではない。数値実行を始めるには、この固定入力による限定的な保存値再解析へのユーザーの明示承認が必要。承認証明書・grant・approval packetは不要。

## 1. 対象と入出力

基準`0dbf1fe265176677e026ea982c0bd435987f4338`と本準備branchの保存入力定義を用い、別のfresh出力領域へ記録する。H4/1Å/STO-3G/DF12、Partial S2/LD3/h=.02/K2、ordinary diag(I,V)、m7/q128/r1/73 bits、epsilon1e-11、compiler A/Qiskit1.3.0、B′/Reference構成とX/Y、元H ground供給／準備費除外を変更しない。energy1.5936e-4 Ha、failure.05、phase.4、alpha1/720、199 shot/axisを保持する。

必須の科学入力は[INPUT_REFERENCES.json](../../../artifacts/pr_reference_n1_preparation/2026-11-08/INPUT_REFERENCES.json)に記録した保存role価格、採用record、m7のp2、保存cohort・V3/CT-0の条件資料。実ファイルの固定commit/hashを対応させる。過去stageごとのSOURCE snapshotは保全し、現在moduleのhashと異なるだけで過去証拠を上書きしない。使用価格の正本はCT-4Bの最終保存会計と同一eの採用inventoryである。

**科学callはread-onlyの有限整数／有理数再会計だけ。** K、d0、d2、dmin、V_threshold、U_thresholdを入力文書の式で各1回評価する。D star価格はsigned role和を用いる。p2を保存文字列からFractionへ読み、確率・normalization・finite lawを再生成しない。

| 後続出力 | 型／意味 | 今回の状態 |
|---|---|---|
| D_B*, D_R*, Csw*, SB* | 固定anchorの入力処理と来歴 | star和は未評価、個別保存価格のみ引用 |
| K | 整数の固定T価格差 | `NOT_EVALUATED` |
| d0, d2, dmin | 整数の中央差 | `NOT_EVALUATED` |
| V_threshold | 整数のpointwise十分条件閾値 | `NOT_EVALUATED` |
| U_threshold | 厳密Fractionの期待十分条件閾値 | `NOT_EVALUATED` |
| 任意の小数表示 | 桁数・丸め方式付き外向き区間 | 未生成 |
| 条件・未資格・STOP | 入力検算と条件付き結果の限定 | 本準備のSTOPとは別に保存 |

Vmax、UV、vacuum価格分布、改善率、新しいword policy、full supportの総Tそのものは取得しない。保存shot199を掛けたround価格や全18round総Tもこの最小scopeには含めない。

## 2. 必要な検算

承認後、関連作業をまとめて次だけ行う。科学runner再開や大規模test suiteの再実行は不要。

1. 固定path/hash、schema、row/key一意性、m7/q128/r1/bits73、epsilon1e-11を確認する。roleにnull／欠損／不一致があれば資格せず停止する。
2. 8保存行のprojection資格を確認し、D未融合・signed補正・Ref CSWAP・B scalarがtrajectory/axis間で一致することを確認する。`Tdagger`をTへ二重加算しない。signed補正を物理gate／normへ使わない。
3. 中央4 keyのrecord key/spec、e=-11、strict guard metadata、sequence SHA、W、actual adjoint規約を保存recordと対応させる。必要なら**4保存wordだけ**の文字数をmetadataと照合する。新word生成やguard評価ではない。欠損metadataを補作しない。
4. p2は指定された保存文字列と一致し、有理確率として範囲内であり、m7のlawを表すことを確認する。lawの数値再生成は行わない。
5. K/d/閾値を厳密算術で計算する。等価な期待式`K+128*d0+128*p2*(d2-d0)`でも一致を検算し、点wise/期待の単位をともに**1 axis・1 shotのT+T†**とする。
6. CT-4Bの8保存行について、既存roleのT和がPILOT_T_COUNTSの直接native会計と一致することを再会計する。FIXED_INPUTSの保存event列のorder0/2出現を用い、一般差分式を**同じtrajectoryの**保存central/event/vacuum価格と照合する。新しいevent生成、native histogram再compile、basis価格期待の再計算はしない。
7. 外向き小数を付ける場合は厳密Fractionが区間内にあることを確認する。結果の科学的判定に丸めたfloatを使わない。

これらは後続N1内の検算範囲であり、今回は価格和・event数・差分の再計算を行っていない。実装が必要ならread-only JSONと標準ライブラリによる小さい処理とし、既存SOURCE／会計の定義を再利用する。既存`ct4.count_hist`等を不用意にimportして旧出力/cache/workerを起動しない。

## 3. 結果の解釈

pointwise条件は`v_P(phi_omega) <= Vmax`を全supportで別途保証した上で、**Vmax < V_threshold**を要求する。期待条件は同じcanonical lawで`E[v_P] <= UV`を別途保証し、**UV < U_threshold**を要求する。等号では厳密優越を保証しない。

両条件はE_B>=0を落とした保守的十分条件。右辺が非正なら、この非負vacuum capの十分条件では優越を資格できない。Reference劣位・優越不可能・研究no-goの証拠ではない。結合差／eventとの相関を使う別の境界が必要なら、その不足証拠を報告して停止し、新たな計算へ進まない。

vacuum・order・product符号は同一trajectory由来。期待式は線形性を使い、それらの独立性を仮定しない。saved rare/nonrareを無加重平均しない。保存2 trajectoryのvacuum価格をVmax／UVに代入しない。

数値が得られても資格は**`CONDITIONAL_THRESHOLD_EVALUATED`**まで。compiler条件Cと共通word Pの完了を仮定した要求値であり、全supportT優越／canonical CT RPE成功／最小T／新規性の認定ではない。条件Cの再証明や正式科学レビューをN1内で自動開始しない。

## 4. 引き続き未資格／scope外

| 不足項目 | N1数値化後にも必要な別の証拠 |
|---|---|
| 全support vacuum word | 到達literal角に対する実取得・有限停止・strict位相保証 |
| Vmax、UV | 実際に返す共通policyの一様価格上界又はcanonical joint lawの期待価格上界 |
| Common frame未取得key | strict wordと誤差資格。差で価格係数0でも実装に必要 |
| B′ b2 event | 到達ならstrict word、又はcanonical非到達の証拠。未観測だけで非到達としない |
| 全support CT RPE | compiler diskと全物理位置の合成diskを同単位で一度だけ合算し、位相・半径・shot条件を保証 |
| 18round総T、T-depth | 本q128/1shotの条件式からは認定できない |

N1に新合成、strict guard、回路build/compile、新価格取得、simulation/sampling、precision/shot/alpha/fusion最適化、SK/CT-3B、N2/N3を含めない。common frameの価格取消しやevent非負性は誤差取消しではない。旧誤差予算を数値的に再最適化しない。

## 5. STOPと次担当

後続実行で指定有限境界と必要検算が完了した時点、又は入力／record／確率／単位の本質的不一致、欠損、条件Cの不成立、固定条件変更・対象外callが必要となった時点で**mandatory STOP**。残る独立した静的項目は報告可能な範囲で完了し、不一致結果をPASSにしない。改善を得るための探索範囲・候補・precision拡張や旧STOP解除はしない。

**ユーザー／GPT**：本書の限定数値再解析を実行するか判断し、実施するなら明示承認する。**Codex**：その承認範囲内でだけ保存値の再会計を実行・公開し、STOPする。本準備作業は完了・STOP済みであり、後続作業は開始していない。
