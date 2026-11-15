# S0+ 判定規則：C2の保存m9回路のstep分解（m9のQPYを開く前に固定）

**対象**：C2の保存m9回路。q=512、R=526、cohort `D_PAULI_LD2_M9_LONG_NORMAL_N2_V1`、2 path × X/Y。
- 読むもの：wrapper・compiled QPY 計8件と、cache JSON 4件。読取専用。
- 入力：[INPUTS_M9.json](INPUTS_M9.json)。C2の公開receipt（`a7fdd200d`）と、価格の束縛（`c9986c82f`）から作った。
- C2の既存のm9価格とモデル比較（`cb034e547`、`adeae8f83`）は、そのまま再利用する。再計算しない。

**規則の出所**
- path単位の監査：S0-r1の固定code `scripts/pauli_ld2_step_audit.py`（`5a18b792a`）の `audit_path` を**変更せずに**使う。出力先だけを `run03/` に切り替える。
- 本書の追加分：run02で確定した決定論部分との照合と、統計の出し方。実装は `scripts/pauli_ld2_s0plus_m9.py`、人工test `tests/check_pauli_ld2_s0plus_m9.py`。

## 1. S0-r1の規則をそのまま使う部分（path・軸ごと）

- **P1 束縛**：QPY・cache JSONのsha256、fingerprint、compiler、6指標、`cache_reused=false`、tokenから数えたRZ・CX・命令数。
  - QPYのsha256不一致、RZ数の不一致はSTOP。
- **P2 step区切り**：identity対（角度の絶対値が1e−12未満のD対）を除いた期待列と、compiledのancilla CXのtarget列を、順序まで完全一致で照合する。
  - 不一致は、最初の位置・前後の項目・差分ブロックを記録してFAIL。
  - 区切り位置、多qubit tokenの区間の一貫性、D対の `cx·rz·cx` と角度の照合も、S0-r1のとおり。
- **X/Yの本体一致**：最終run以外の全wireのtoken列を、区切りの成否と無関係に比べる。
- **最終run**：`rz·sx·rz` の形、`p` の角度が `−scalar·T` とbit単位で一致すること、ZSX合成との一致。

## 2. run02の決定論部分との照合（S0+の追加。X軸）

**参照**：[REFERENCE_RUN02.json](REFERENCE_RUN02.json)
- run02（`43755e0ca`）で全13 pathに共通だった、wireごとのtoken列を固定したもの。最初のstepの入口 F、2番目以降のstepの入口 P、各stepの末尾 S、終端 E。
- 取り出し元：Fと終端はS66/p0、Pは Q8R66/p0 の2番目のstep、SはS66/p0。
- RZ数の合計：F＝1092、P＝942、S＝958、E＝156、最後のrun＝2。
- ここから、切片 308、stepあたり 1900 が決まる。

**照合**（区切りに成功したpathだけ）
- 各区間が、wireごとに、入口（step 1はF、step 2以降はP）のtoken列で始まり、末尾Sのtoken列で終わること。
- system wireの終端が、Eと一致すること。
- token単位で完全一致を要求する。最初の不一致（step・wire・入口か末尾か・位置）を記録してFAILとし、規則の変更や再試行はしない。

**総数の復元**
- 本体RZ ＝ 区間RZ − （入口のRZ ＋ 末尾のRZ）（run02の定数）。
- `308 + 1900·q + Σ本体RZ` が、保存rz_countと一致するかを報告する。
- 意味を持つのは、上の照合がPASSしたときである。照合がFAILでも、計算した値はそのまま記録する。

**最終runの予測との照合**
- θ₁が、run02の関係から予測したm9の値（X：−0.8275034047898195、Y：−2.398299731584725）と1e−9以内で一致し、θ₂のhexがrun02と同じであること。

## 3. P4

- 適用外とする。m9には、親子の対応がある合成回路がないため。

## 4. 統計（区切りに成功したpathだけ）

- X軸だけを使う。Yは同じpathの別軸なので重複として除く。
- R=526の2番目以降のstep（各path 511区間）について、本体RZの件数・平均・平均/R・不偏分散・標準偏差・分散/R・最小・最大を、path別に出す。
- 2 pathをまとめた値も出す。独立path数は2、step区間数は1,022。
- 最初のstep（各path 1区間）は、件数が1なので値だけを示す。
- 本体RZは§2の定義による。記述統計だけで、CIやfitではない。

## 5. 実行・STOP

- 2 pathをprocess並列で、各1回だけ実行する（`--run --workers 2`）。続けて `--summarize`。`--measure`（余分な読取）は行わない。
- 出力：`run03/`。token配列 `run03/work/` は再生成できるため、commitしない。
- STOP：AUDIT_RULES.md §8 のとおり（Qiskitの版、QPYの読取、sha256、RZ数の不一致、workerの異常。private fileへの書込みは禁止）。
- 範囲外：新規のsampling・build・compile、再fit、S1、価格予測、E[C]・CI・G、希少eventの調査。run01・run02とC2の結果は書き換えない。
