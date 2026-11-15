# PR-Pauli LD5 較正cellの取得・監査・価格の規則（科学実行前に固定）

LD5段階1の正本 [HANDOFF_B](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf0a930fd75fb2eb693622edcf3b7204e30f5eff/docs/research/df_head_pauli_tail_ld5_stage1_20261116/HANDOFF_B.md)（`bf0a930fd`）と利用者の指示（chat A1109）に従う。LD3の段階3（[規則](../a_pauli_ld3_cal_price_20261116/AUDIT_PRICE_RULES.md)、`158bbcc7b`）と同じ設計で、LD5の値だけを替える。規則・driver・監査code・人工testを同じcommitで固定し、その後に変えない。

## 1. 対象と上限

| 項目 | 内容 |
|---|---|
| cell | **Q8R4**：q＝8、R＝R_12(LD5)＝4、T＝0.8、K＝2 |
| path | 独立normal path 2本。各pathをX/Yで共有する（独立標本ではない） |
| compile | 最大4 |
| QPY読取 | 最大8 |
| cell ID | `PAULI_PRICE_CELL:53f1503d5e53f9190730006845b8e89605dcbbc270349fd79d24701745256318`。m3（q＝8、R＝1、`RMSE:c943efea…`）とも、m12（q＝4096、R＝4、`RMSE:795c5535…`）とも別物で、receiptはround価格のreceiptではない |
| cohort・seed | `A_PAULI_LD5_CAL_Q8R4_N2_V1`、16個。既存のLD1〜LD3の1,341個と重複しない（[BINDING.json](BINDING.json)） |

## 2. 入力・SOURCE・出力先

- 科学code：D2 `13ab1acfa`（LD2 CAL・C2 m9・LD3 CALと同じ）を、専用checkout `a-pauli-ld3-cal-source-13ab` から無変更で使う。
- driver：`scripts/run_pauli_split_cal.py`。LD3 driver（`158bbcc7b`）の汎用部分を無変更で使う。LD3のspecで、公開済みLD3 bindingを `driver` 欄以外すべて再現することを確認済み。
- LD5の入力
  - 辞書：`6255fbc42` の `LD5_DICTIONARY.json`（sha `2947784c…`、λ_P＝0.061570372477616604）
  - operator：`6255fbc42` の `LD5_OPERATORS.npz`。common blocksは `4220f3006`
  - configuration：`4d6e2c9c…`（`39a1fc7df`）
  - scalar：c＋保存s_R＝2.302014585108299、総Tで1回
- 実行前の入力確認（[INPUT_PREFLIGHT.json](INPUT_PREFLIGHT.json)、sampling・build・compileは0）
  - native LD5 headは one-body＋fragment 0〜4 の6 block
  - nativeが抽出したtail identityは保存s_Rとbit単位で一致
  - bit順・head・tailのgateはPASS
- 出力：公開は `docs/research/a_pauli_ld5_cal_price_20261116/run01/`、privateは `/home/AbeHiromu/projects/a-pauli-ld5-cal-20261116-private/`。いずれもfresh、再試行なし。

## 3. step境界と数え方

- 1 stepは `U(1b) U(f0) … U(f4) [4 event] D(f4) … D(f0) D(1b)`。
  - 期待されるD blockは6、D(1b)が障壁になる。
  - 区切りの規則はS0-r1（`5a18b792a`）をそのまま使う：identity対の除外、ancilla CXの順序まで完全一致、最後のD CXでの区切り、区切りをまたぐ多qubit tokenがないこと。
- wireごとのF（最初の入口）・P（2番目以降の入口）・S（末尾）・E（終端）の定義は、LD3と同じ（`pauli_ld3_cal_audit.edges`・`bodies` を無変更で使う）。
- 係数（RZ数は全wireの合計。直接数えた値で、fitではない）

  ```
  A5 = F + S + E + 最終run − (P + S)
  H5 = P + S
  E5 = （2 path × 第2〜第8 step、計14区間の本体RZ数の合計）/ (14 × 4) = … / 56   [RZ/event]
  C_ref(m, a) = A5 + H5·q_m + E5·q_m·R_m
  ```

  - X軸だけを使う。同じpathのYは独立標本として二重計上しない。
  - 単一のqでの総RZ数から、切片と傾きをfitしない。

## 4. 規則が支持されたとみなす条件

LD3と同じ条件をすべて満たすこと。

1. P1 4/4
2. P2 4/4、全16 stepでevent＝4
3. P3 (a)〜(e)
4. P5：X/Yの一致、sdgだけの差、wrapperの入力run、`rz·sx·rz`、θ₁・θ₂が一意、LD5の13 round × X/Yで余裕が1e−6以上
5. P6：2 path × X/Yで分解が厳密

満たさない場合は、確認できた直接値（receipt、F・P・S・E、本体）を保存し、価格欄はnullにしてSTOPする。規則の変更、再取得、再compileはしない。

## 5. 価格と比較

- 26軸はすべて「モデル適用」と明記する。直接観測は q＝8・R＝4 の2 pathだけである。R＝1・2のroundや、長いqへの適用はモデル値である。
- 2つの列で、LD5の条件付き総額を、同じ列のLD3保存総額と並べる。
  - VP（LD3：173557305044/165、`f25a05f8a`）
  - FULL_PF__B_PRICE（LD3：625936421828/1155、`fb4d78781` の `achieved_conditional_cost`）
- 表示する差
  - H5−2685
  - 列ごとの切片・head・eventの費用差（重みはshot Σ2N、step Σ2N·q、event Σ2N·q·R）
  - LD3側の分解は、LD3の係数から保存総額をexactに再現することを確認してから使う。
- 段階1の分岐値（約1,535・2,408 RZ/step）は、事前の目安として残すだけで、比較には使わない。
- LD5のQ3適否は未評価である。

## 6. 行わないこと

Q3、新しいshot配分、fit、E[C]・CI・G、winner、固定SOURCE・原結果・旧STOPの変更。
