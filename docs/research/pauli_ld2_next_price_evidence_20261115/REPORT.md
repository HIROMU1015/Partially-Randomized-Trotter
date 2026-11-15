# PR-Pauli LD2：次の有限価格証拠の選択

**支配的不確かさはまだ識別できない。** m5は2 path、m6は1 pathで、有限path変動とround移送時の
モデルのずれを分離する根拠が足りない。一方、全13 roundへ向けた明確な証拠の空白はm7以降の独立価格比較である。
**第一案：B、m9の新cohort・fresh 2 path×X/Y、最大4 compile。**
**得られる判断：凍結モデルの長q領域での2軸の有限cohort記述差と、同roundの2 pathの違い。**
誤差上限・全law E[C]・CI・モデルPASSは得られない。提案・未採択・科学実行なし。

## 1. 二つの不確かさ

### 同じroundでの有限path平均

固定holdoutの保存個別receiptでは、m5はX/Yともpath0=68,228、path1=68,300 RZ、
m6はX/Yともpath0=149,435 RZ。m5の同軸2 path平均68,264と、m6の1 path値149,435に対する
既存記述差は、それぞれ**+0.044612%、+0.157666%**（P−O、分母O）。
これは[705f59ea8 HANDOFF_C2.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/705f59ea863e1b9fe90e023f1cb1057d00366b16/docs/research/d_pauli_ld2_b1_holdout_results_20261115/HANDOFF_C2.md)
の引用で、新しい平均・誤差・分散を算出していない。

m5では同じroundでも2つの価格が異なることは分かるが、2 pathだけで母平均の必要精度や裾・
再現的な分散を認定できない。m6は単一観測なので、当該cohortから標本分散・SE・CIを推定できない。
m5の変動をm6へ移すこともできない。小さい記述差は「標本変動が小さい」「系統誤差がない」の証明ではない。
X/Yは同一pathの別測定軸で、holdoutは3 fresh path・6 receipt・4比較単位。
較正は8 fresh path＋親S66 path0→path1から作った1 derived path。P2R66は親に依存し、
9独立pathでも18独立観測でもない。fit単位は各軸5 cellである。
実LD2 order2 product・非自明event/D2位相の未観測UNKNOWNは、保存holdoutの状態をそのまま維持する。

### roundを変えたモデル予測

固定モデルは軸別の`C_a(q,R)=β0,a+β1,a q+β2,a qR`、特徴量順`[1,q,qR]`。
較正点は下表。Tは総時間で、全cellのouter step時間は0.1。較正はwhole-round系列そのものではない。

| CAL cell | q | R | T | qR | path |
|---|---:|---:|---:|---:|---|
| S66 | 1 | 66 | 0.1 | 66 | fresh 2 |
| S526 | 1 | 526 | 0.1 | 526 | fresh 2 |
| S4205 | 1 | 4205 | 0.1 | 4205 | fresh 2 |
| Q8R66 | 8 | 66 | 0.8 | 528 | fresh 2 |
| P2R66 | 2 | 66 | 0.2 | 132 | derived 1 |

全13 roundのmetadataを固定A packetから転記した。特徴量は各行`[1,q,qR]`。
Tは読取用の十進表示で、実行時はpacket内の元exact/binary64表現を保持する。

| m | q | R | T | qR | このモデルの証拠状態 |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 2 | 0.1 | 2 | W配線1 pathのみ。CAL/独立モデル比較ではない |
| 1 | 2 | 3 | 0.2 | 6 | 独立価格比較なし |
| 2 | 4 | 5 | 0.4 | 20 | 同上 |
| 3 | 8 | 9 | 0.8 | 72 | 同上。Q8R66とはRが違う |
| 4 | 16 | 17 | 1.6 | 272 | 独立価格比較なし |
| 5 | 32 | 33 | 3.2 | 1056 | 指定fresh 2 path×XYの独立記述比較済み |
| 6 | 64 | 66 | 6.4 | 4224 | 指定fresh 1 path×XYの独立記述比較済み |
| 7 | 128 | 132 | 12.8 | 16896 | 長round独立価格比較なし |
| 8 | 256 | 263 | 25.6 | 67328 | 同上 |
| **9** | **512** | **526** | **51.2** | **269312** | **次の候補。S526と同じR、長qは未検証** |
| 10 | 1024 | 1052 | 102.4 | 1077248 | 独立価格比較なし |
| 11 | 2048 | 2103 | 204.8 | 4306944 | 同上 |
| 12 | 4096 | 4205 | 409.6 | 17223680 | S4205と同じRでも長q全体は未検証 |

CALのqは1/2/8に限られ、m5/m6は独立比較済みでも母平均資格はない。m0〜m4はRがCALより小さく、
一部の特徴量がCAL範囲内でも正確な組合せ・law条件での独立比較はない。
m7以降はqと総event数が比較済み領域を超える。RがCALと一致するm9/m12でも、head反復・
境界を跨ぐcompiler最適化・連結回路全体のずれは、単一step CALだけでは決まらない。
+0.044612%/+0.157666%を全13 roundの誤差上限やモデルPASSへ変換しない。

## 2. 比較する二案と選択

| 項目 | A：m6の新fresh cohort | B：m9の新fresh cohort（第一案） |
|---|---|---|
| 対象 | m6 q64/R66/T6.4 | m9 q512/R526/T51.2 |
| 範囲 | 2 fresh path、X/Y、最大4 compile | 2 fresh path、X/Y、最大4 compile |
| 再利用 | b74eのm6保存予測、物理入力・reader・gate。旧m6 pathは新cohortに算入しない | b74eの係数・domain、A canonical metadata、辞書・head・reader・gate。m9予測は別途観測前に固定が必要 |
| 新しく分かること | 既存1 pathしかないm6で、別cohort2 pathの違い・旧観測との再現性 | 初のq512/長連結回路での予測差、同じroundの2 pathの違い |
| 残ること | 長roundへの移送、全13 roundの証拠不足。2 path追加だけで必要精度・CIは決まらない | 2 pathではモデルずれと標本変動を完全分離できず、m6の分散・m10〜m12・全law資格も残る |

Aを選ぶ判断は「m6の記述差を別cohortで確かめたい／同roundの精度目標に向けた標本設計を進めたい」場合。
m5にも同趣旨の追加は可能だが、既に2 pathがあるため、Aの具体案は単一観測しかないm6へ絞った。
Bを選ぶ判断は「全13 roundに使う前に、長qへの移送仮説が未観測のままなのを解消したい」場合。
**現時点ではB。** 全13 roundの期待価格に向け、既存round内の繰返しより異なる領域の証拠を先に増やす。
これは不確かさの大小を測定済みという判断でも、情報量/計算時間を最適化した結果でもない。
精度目標・信頼水準・N/G重みが未採択/未評価なので、総Gのどのroundが支配的かは認定しない。

m9はR526の単一step較正を保ちつつ、長い反復・境界の効果を調べられる。
m7を次にするだけよりq領域を広げ、m12のq4096・17,223,680 event/pathへ直行する前の有限点とする。
m10/m11/m12を自動追加しない。2 pathは新roundで少なくとも2つの個別価格を観測するための最小対であり、
旧path数、95%信頼、2%精度を採択したものではない。Bも母平均検証の完了ではない。

取得負荷は同一ではない。m6既存jobはwall4:35.03、peakRSS946,088 KiB（705f REPORTの引用、
import/context/gate/XY compile込み）。m9には269,312 event/pathがあり、wall/RSSは未測定。
m6実測を比例倍して見積もらず、長round用の実経路・進捗・再利用可能な高速化を先に確認する。
資源継続性や同等性が確認できなければ科学投入を保留し、m6へ無断で差し替えない。

## 3. §10.5照合と固定SOURCE

| 照合対象 | 判定・今回の扱い |
|---|---|
| CAL正本5dacの10 record/18 receipt | 既存fitの根拠として再利用。S526はq1、m9はq512なので価格の代替不可 |
| 705fのholdout REPORT/RECEIPT_ADMISSION/元0e1ce raw | 6個別locatorのaxis/index/cohort/fingerprint/RZを照合。一致。元の独立比較として再利用し再集計しない |
| 043cの元cohort/seed、b74eのfit・予測 | 固定根拠として再利用。新cohort・m9はこの旧実行許可の対象外 |
| LD2 W、LD1 B0、旧DF/Full-H/Pauli-JW | 旧043c/705f監査のlaw/head/K/lowering/用途差を引継ぐ。Wは配線のみ。新CAL/holdout標本へ転用しない |
| m9 canonical IDとm6新cohort候補 | 既知のLD2価格索引と関連LD2履歴に同等trial/receiptなし。m9のhitはA schedule metadata。新規候補は各案最大4 compile、同時実行しない |
| 関連公開ref・自ユーザーの関連process・提案namespace | C2=f448、D2 Pauli=705f。照合時、同等の関連科学jobなし（検索自身を除く）。提案public/private namespaceは未存在 |

探索は指定LD2の既知catalog/receipt/履歴・関連refと既知保存先に限定し、Full-H規則案branch・
D2 PR-DF branchは作業に使っていない。外部非公開trial・他ユーザーjobまで未取得と断定しない。
新cohortのseed表・新prediction locator・実行SOURCE・資源適合は**保留事項**であり、取得済み証拠と混同しない。
実行前に対象/cohort/seed・関連job/保存receiptを再照合し、同等保存成果が判明したら再利用の適合性を確認する。
証拠不足や競合なら当該項目STOP。元cohortの用途と、再利用結果の来歴を維持する。

| 固定commit | path / 用途 |
|---|---|
| `043c7567451e8f502a1cf0f425d32efe0a50f5dc` | `docs/research/d_pauli_b1_caller_preparation_20261114/CAL_AFTER_W_METADATA.json`：LD2入力・旧cohort/seed |
| `5dac74959ed9ea49465d393fc792ef5b4727edba` | `docs/research/d_pauli_ld2_b1_cal_results_20261115/C_HANDOFF.json`：較正正本 |
| `b74e4e536392c2bbe94cd371436615fc56c0564f` | `docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json`：軸別モデル・4旧予測 |
| `f4480ee3f831a7b106d287c401ee78fbd62f8615` | 同dir `HOLDOUT_RULE_PROPOSAL.md`：今回の明示指示・705fで採択確認、数値閾値は未採択 |
| `705f59ea863e1b9fe90e023f1cb1057d00366b16` | `docs/research/d_pauli_ld2_b1_holdout_results_20261115/`：HANDOFF_C2、REPORT、RECEIPT_ADMISSION、COMPARISON |
| `714247619afb4714fd663eb15c0470ef32bdc7af` | `docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json`：prefix2のconfiguration/round identity metadataのみ転記 |

[有限実行案](FINITE_EXECUTION_PLAN.md)／[HANDOFFとロードマップ差分](HANDOFF.md)。
今回は新sampling/compile/fit/CI/E[C]/N/G、希少イベント調査、LD1・Full-H移送を行わない。
計画公開後mandatory STOP。
