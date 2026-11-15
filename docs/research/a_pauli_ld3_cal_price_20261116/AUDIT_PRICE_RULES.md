# PR-Pauli LD3 段階3：較正cellの取得・監査・価格の規則（科学実行前に固定）

計画 [`a50dce8f1`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a50dce8f1cbb38b2914bb2fb4ff79bab35e6e8f6/docs/research/a_pauli_ld_split_plan_20261116/REPORT.md) の段階3を、利用者の指示（chat A1109）に従って具体化した。この規則・driver・監査code・人工testを同じcommitで固定し、その後に変えない。

## 1. 対象と上限

| 項目 | 内容 |
|---|---|
| cell | **Q8R660**：q＝8、R＝660、T＝0.8（binary64の0.1×8）、K＝2。m12と1 stepあたりのRが同じ較正用cell |
| path | 独立なnormal path 2本（index 0・1）。各pathをX/Yで共有する（X/Yは同じpathの別軸で、独立標本ではない） |
| compile | 最大4（各path 2）。それを超える呼出しはdriverが拒否する |
| QPY読取 | 最大8（2 path × X/Y × wrapper・compiled） |
| cell ID | `PAULI_PRICE_CELL:9cd5c434acf2eebf43f2f8795db4725a3db4a940a9c1c4f9112d6ab8f8946eb8`（非configuration cell）。**m12のphysical ID `RMSE:95014c1c…`（q＝4096）とは別物**で、そのreceiptはround価格のreceiptではない |
| cohort・seed | `A_PAULI_LD3_CAL_Q8R660_N2_V1`、16個のstep seed（既存の `seed(candidate_id, cohort, index, outer, 0)`）。LD1・LD2のCAL・holdout・W・m9の1,325個と重複しない |

## 2. 入力・SOURCE・出力先

- 科学code：D2の `13ab1acfa`（LD2 CALとC2 m9と同じ）を、専用のdetached checkout `a-pauli-ld3-cal-source-13ab` から無変更で使う。
  - `acquire_path`／`pilot_core`、normal sampler、native DF head、Pauli D2 lowering、測定付きwrapper、compiler（Qiskit 1.3.0、`rz/sx/x/cx`、opt1、seed 17、compiler hash `1239304c…`）。
- driver：`scripts/run_pauli_ld3_cal.py`。C2のm9 worker（`cb034e547`）の観測・event行列memo・同一byte fingerprint・2 process・receipt書出しを、pinned blobからそのまま使う。
- LD3の入力
  - 辞書：`50640b74c` の `LD3_DICTIONARY.json`（sha `be4ee9cd…`、λ_P＝0.8137790393725127）
  - operator：`50640b74c` の `LD3_OPERATORS.npz`、common blocksは `4220f3006` の `COMMON_OPERATORS.npz`
  - configuration：`dd5b881d…`（`421258d40`）。B total view `6757998a…`（`2de33a4f4`）
  - scalar：c＋s_R＝2.3857164132121906、総Tで1回
- cell identityの作り方：旧 `request_for` と同じ（configurationのbinding・split・profile、route、lowering、q・R・T・K）。この作り方で、保存LD2 Q8R66のrequest（`PAULI_PRICE_CELL:d901db3c…`）がbit単位で再現することを確認済み。
- 実行前の入力確認（[INPUT_PREFLIGHT.json](INPUT_PREFLIGHT.json)、sampling・build・compileは0）
  - native LD3 headは one-body＋fragment 0〜2 の4 block
  - nativeが抽出したtail identityは保存s_Rとbit単位で一致
  - bit順・head・tailのgateはPASS
- 出力：公開は `docs/research/a_pauli_ld3_cal_price_20261116/run01/`、private cacheは `/home/AbeHiromu/projects/a-pauli-ld3-cal-20261116-private/`。いずれもfresh（再試行・再開はしない）。

## 3. step境界（LD3のheadから）

- 1 stepは `U(1b) U(f0) U(f1) U(f2) [R個のevent] D(f2) D(f1) D(f0) D(1b)`。
  - 前半のheadは制御しない（symmetric_directional）。
  - ancillaに作用するのは、tailのeventと後半のD領域だけである。
- D領域は `cx(anc,o)·rz(o)·cx(anc,o)` の対からなり、ancillaのrzでblockに区切られる。最後のblock D(1b)が全wireに対の作用を置き、障壁になる（S0-r1で確認した仮説H_loc）。
- 区切りの方法はS0-r1（`5a18b792a`、[AUDIT_RULES_R1.md](../a_pauli_ld2_step_audit_20261115/AUDIT_RULES_R1.md)）をそのまま使う。
  - compile前の角度の絶対値が `1e−12` 未満の対（identity対）は、期待列から除く。
  - compiledのancilla CXのtarget列が、期待列と順序まで完全一致すること。
  - wire wの区切りは、そのstepのD領域でwを対象とする最後のancilla CXとする。
  - 区切りをまたぐ多qubit tokenがないこと。
- LD2との差はD blockの数（3→4）だけで、規則の関数は変えない（`pauli_ld2_step_audit.audit_path` をそのまま呼ぶ）。

## 4. 決定論部分とevent本体の数え方

2本のpathのX軸のtoken列から、wireごとに次を求める（S0-r1 §4と同じ定義）。

| 記号 | 定義 |
|---|---|
| S_w（末尾） | 全16区間に共通する最長の末尾 |
| F_w（最初のstepの入口） | 2本の第1区間からS_wを除いた部分に共通する最長の先頭 |
| P_w（2番目以降の入口） | 14本の第2〜8区間からS_wを除いた部分に共通する最長の先頭 |
| E_w（終端） | 最後の区切りより後（system wire） |
| 本体 | 区間から入口（FまたはP）と末尾Sを除いた部分 |

**係数**（RZ数は全wireの合計。直接数えた値で、fitではない）

```
A3 = F + S + E + 最終run − (P + S)
H3 = P + S
E3 = （2本のpathの第2〜第8 step、計14区間の本体RZ数の合計）/ (14 × 660)    [RZ/event]
C_ref(m, a) = A3 + H3·q_m + E3·q_m·R_m
```

- 各pathについて `A3 + H3·8 + Σ本体` が保存rz_countと一致すること（X・Yとも）。
- Y軸は同じpathなので、E3には使わない（二重計上しない）。
- 単一のqでの総RZ数から切片と傾きをfitすることはしない。

## 5. 規則が支持されたとみなす条件

次をすべて満たしたときだけ、26軸の価格を作る。

1. P1：4/4 PASS（sha256、cache record、receiptの6指標、tokenのRZ・CX・size、basis）。
2. P2：4/4 PASS。全16 stepでevent数＝660。
3. P3：(a) compile前のD領域、(b) 障壁、(c) compiledのancilla D領域が全stepで同一。(d) S_wがD(f0)以降を全wireで覆う。(e) system wireの終端が2 pathで同一。
4. P5
   - compiledのX/Yが、最終run以外の全wireでtoken一致する。
   - compile前のX/Yの差は、ancillaのsdg 1個だけである。
   - wrapperの入力runの照合（`p＝−scalar·T`、ZSX合成）が成り立つ。
   - 最終runが `rz·sx·rz` で、θ₁・θ₂が軸ごとに2 pathで同一である。
   - LD3の13個のTについて、`θ₁(m) = wrap(θ₁(0.8) − scalar·(T_m − 0.8))` の絶対値が1e−6以上である（26軸すべて）。
   - θ₁のTに対する関係そのものは、LD2のS0-r1で5種のTについて確認済みであり、今回の1つのTでは再検定しない。
5. P6：2 path × X/Y で、上の分解が保存rz_countと厳密に一致する。

**一つでも満たさない場合**
- 該当箇所と取得済みの値（receipt、監査結果）を保存し、価格欄をnullにしてSTOPする。
- 規則の変更、再取得、再compileはしない。

## 6. 価格表と同一VPの比較

- 26軸（m0〜m12 × X/Y）：固定scheduleのq_m・R_mに上の式を当てはめる（すべて「異なるq・Rへのモデル適用」）。
- 較正cell自身の直接の数え値（2 path × X/Y）は、別の行として区別して載せる。
- 総額：VP `N = [48,45,42,39,36,33,29,26,23,19,16,13,9]`（X/Y各軸）で `T = Σ N_{m,a} C_ref(m,a)` を、exactな有理数で求める。
- 比較の基準：LD2の同一VPの保存参照総額 `a7eceecba`（T_ref＝305225682539751998/90343469）。差額とLD3/LD2の比を示す。
  - LD2のOWN-Bは別の配分なので、基準にしない。
  - LD2のT_refはC2の凍結線形モデル（`b74e`）によるもので、今回のLD3の直接分解とは作り方が違う。この点を併記する。

## 7. 行わないこと

- 新しい信号、Q3、shot配分の探索、fit、E[C]・CI・G、winner。
- 固定SOURCE・原結果・旧STOPの変更。
