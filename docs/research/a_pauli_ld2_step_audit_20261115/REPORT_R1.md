# PR-Pauli LD2 S0-r1：同じ保存QPYの再解析（run02）

2026-11-15 JST。Codex Aの担当分（chat A1109が実施）。

**位置付け**
- run01（[REPORT](REPORT.md)、P2 FAIL）と同じ保存標本（QPY 52件）を、利用者が承認した修正規則S0-r1で1回だけ再解析した。
- run01の判定と原結果（`run01/`）は変更していない。r1の結果は `run02/` に別に記録した。

**規則とSOURCE**
- 修正規則 [AUDIT_RULES_R1.md](AUDIT_RULES_R1.md)・監査code・人工test 14件（PASS、[TESTS_R1_FROZEN.txt](TESTS_R1_FROZEN.txt)）を `5a18b792a` で固定・pushし、remoteとの一致を確かめてからQPYを開いた。
- 新規のpath・sampling・build・compile、fit、価格予測、E[C]・CI・Gは0件。

## 結論

| 検査 | run02（S0-r1） | 参考：run01 |
|---|---|---|
| P1 束縛 | PASS 26/26 | PASS 26/26 |
| P2 step区切り（順序まで完全一致） | **PASS 26/26** | FAIL 26/26 |
| P3 境界寄与の形 | **PASS** | 未評価 |
| P4 P2R66と親S66の合成関係 | **PASS_EXACT**（RZの照合は1個差、§3.4） | 未評価 |
| P5 X/Yと終端run | **PASS**（X/Yの本体一致は13/13） | 終端runのみPASS |
| P6 区間の表 | **PASS**（153区間、13/13で分解が厳密） | 未評価 |

1. **H_locは、保存した実回路（q ≤ 64）で成り立った。**
   - compiled回路は、stepごとの障壁で区切れた。区切りをまたぐ多qubit tokenはなかった。
   - 各stepの区間は「決定論的な入口 ＋ eventに依存する本体 ＋ 決定論的な末尾（D(f1)〜D(1b)の全3 block）」で、入口・末尾・終端はRにもqにもよらず全pathで共通だった。
   - 全13 pathで、次の分解が厳密に成り立った（各項は保存回路から直接数えた値で、fitではない）。

     ```
     compiled RZ ＝ 308 ＋ 1900 × q ＋ Σ（各stepの本体のRZ）
     ```

2. **合成関係も成り立った。** P2R66（S66のpath0とpath1を連結）の2番目のstepは、単独で compileしたS66 path1と、入口の決定論部分を除いてtoken単位で一致した（全9 wire）。
3. **stepごとの観測変動**（X軸、P2R66を除く）
   - 2番目以降のstep：R=33で分散232（標準偏差15、n=62）、R=66で分散385（標準偏差20、n=77）。
   - 最初のstep（単一step cellを含む）は、R=66で分散38（n=5）など小さめだった。理由は未解明である（§4）。
4. **観測範囲の外へ延長するときの仮定**
   - q>64への延長（m9のq=512は、C2の記述比較 +0.274% だけ）。
   - R=1052・2103のstep law。
   - Rの大きいstepが、2番目以降の位置でどれだけばらつくか。
   - order2・非自明な位相、実sampler law。

## 1. §10.5の照合と実行

**照合**（22:21〜22:25 JST）
- run01以後の新しいcommitは、D2 PR-DF LD2の価格評価（`8dc41b354`）とC Full-H F1（`a1aa38df4`）だけだった。Pauli LD2のstep分解の再解析は、ほかに完了も進行もしていなかった。同じjobはなかった。
- private cache 3 directory（120 file）は、run01の後からr1の前まで変化がなかった。

**実行**
- 入力：`INPUTS.json`（`2f3bf20ce`）をそのまま使った。同じQPY 52件とcache JSONを読取専用で1回だけ開いた。`--measure` は行っていない。
- 資源：13 pathをprocess並列で実行した。wallは14.2秒、workerの最大RSSは391 MiB（m6）、すべての子processがDONEで終了codeは0（[RUN_ALL.json](run02/RUN_ALL.json)、[PROGRESS.jsonl](run02/PROGRESS.jsonl)、`RUN_time.txt`）。
- 保全：private cacheは、r1の前後で変化がなかった。

## 2. P1・P2・P5

**P1：PASS 26/26**（run01と同じ照合）

**P2：PASS 26/26**
- 期待列から、各stepのidentity対5対（D(f1)の単一Z項、角度は約10⁻¹⁹）を除いた。identity対の数は、全306 stepで5だった。
- 修正後の期待列は、compiledのancilla CXのtarget列と、順序まで完全に一致した（26/26）。
- 残りのD対（stepあたり72対、X軸で計11,016対）は、すべて `cx·rz·cx` の形で残り、rzの角度はhexまで一致した。
- 全wireに区切りがあり、区切りをまたぐ多qubit tokenはなかった。
- 各stepのevent数＝R（全306 step）で、product tokenは0件だった。

**P5：PASS**
- compiledのX/Yは、最終run以外の全wireでtoken列が一致した（13/13）。compile前のwrapperのX/Y差は、ancillaのsdg 1個だけだった（13/13）。
- 最終runは26/26で `rz·sx·rz` だった。`p` の角度は `−scalar·T` とbit単位で一致し、ZSXで合成した結果も一致した。
- θ₁の関係の残差は、Xで最大1.8×10⁻¹⁵、Yで0。13 round×X/Yのすべてで余裕は1e−6以上（最小0.052 rad、Y軸のm7）。最後のrunのRZは全roundで2個である。

## 3. P3・P4・P6

### 3.1 P3 境界寄与の形：PASS

- (a) compile前のD領域の内容（3 block・77対・block区切りのrz）は、13 path・X/Yの全306 stepで同一だった。
- (b) 最後のblock（one-body、8対）も同一だった。
- (c) compiledのancillaのD領域のtoken列は、全153 step（X軸）で同一だった。
- (d) 全stepに共通の末尾 `S_w` は、全9 wireで、D領域の3 blockすべて（D(f1)から障壁まで）を覆った。規則が要求したのはD(f0)からだが、それより強い結果である。
- (e) system wireの終端部分は、13 pathすべてで同一だった。

**決定論部分のRZ数**（全wireの合計、[EDGES.json](run02/EDGES.json)）

| 部分 | RZ |
|---|---:|
| 最初のstepの入口 F | 1092 |
| 2番目以降のstepの入口 P | 942 |
| 各stepの末尾 S（D(f1)〜D(1b)） | 958 |
| 終端 E（system wire） | 156 |
| 最後のancilla run | 2 |

したがって、次が成り立つ。

```
RZ ＝ (F + S + E + 2) + (q − 1)(P + S) + Σ本体 ＝ 308 + 1900·q + Σ本体
```

### 3.2 P6：PASS

- 13/13のpathで、区間の和＋終端＋最終runが保存rz_countと一致した。区間はX軸で153個。
- 表：[SEGMENTS.csv](run02/SEGMENTS.csv)（step別に区間・入口・本体・末尾のRZ数とevent数）。

### 3.3 P4 合成関係：PASS_EXACT

- P2R66の1番目の区間は、S66 path0の区間と全9 wireで完全に一致した。
- P2R66の2番目の区間は、S66 path1の区間と、全9 wireでEXACT（違うのは入口の決定論部分だけ）だった。
- 終端部分も一致した。

### 3.4 P4のRZ照合の1個差（記録）

- 実際の差は `Δ = RZ(P2R66) − RZ(S66p0) − RZ(S66p1) = −307` だった。
- 決定論部分の定数から導く値は `ΣP − ΣF − ΣE + 最終runの差 = −308` で、1個ずれた。
- 位置はwire 3である（[P4_RZ_CHECK.txt](run02/P4_RZ_CHECK.txt)、run02のtoken配列だけを読んだ。QPYは読み直していない）。
- 原因は「入口」の境界の定め方にある。wire 3では、全pathに共通な先頭として、2番目以降のstepの入口 P が、最初のstepの入口 F より、共通部分に対して1 token手前で終わる。ほかのどこかの2番目以降のstepが、そこで異なるためである。そのため、2番目以降のstepの「本体」は、定義上rz 1個を余分に含む。
- tokenの一致（EXACT）には影響しない。実際の差 −307 は、入口の非共通部分の差（−149）、終端（−156）、最後のrunの差（−2）で過不足なく説明できる。
- 規則の変更や再試行はしていない。

## 4. stepごとの観測変動

[STEP_STATS.csv](run02/STEP_STATS.csv)。X軸だけを使い（Yは同じpath）、P2R66は親に依存するので除いた。

| R | 文脈 | 件数 | 本体RZの平均 | 平均/R | 分散 | 標準偏差 | 分散/R |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | 最初 | 1 | 28 | 14.0 | — | — | 識別できない（n=1） |
| 33 | 最初 | 2 | 224 | 6.79 | 800 | 28.3 | 24.2（自由度1） |
| 33 | 2番目以降 | 62 | 223.6 | 6.78 | 232.2 | 15.2 | 7.04 |
| 66 | 最初 | 5 | 436.2 | 6.61 | 37.7 | 6.1 | 0.57 |
| 66 | 2番目以降 | 77 | 430.3 | 6.52 | 384.6 | 19.6 | 5.83 |
| 526 | 最初 | 2 | 3314.5 | 6.30 | 0.5 | 0.7 | 0.001（自由度1） |
| 4205 | 最初 | 2 | 26646 | 6.34 | 1152 | 33.9 | 0.27（自由度1） |

**表の読み方**
- 本体には、step間で変わらない定数は含まれない。「最初」と「2番目以降」は入口の定義が違う（§3.4）ので、平均を直接比べない。
- 2番目以降のstepの分散は、R=33で232、R=66で385と、十分な件数で求まった。
- 最初のstepは、R=66（n=5）、526・4205（各n=2）で分散が小さかった。件数が少なく偶然を区別できないが、単一step cellのばらつきが、長いpathの中のstepのばらつきを代表しない可能性がある。原因は未解明で、今回は追加調査していない。
- 平均/Rは、R=33〜66で6.5〜6.8、R=526・4205で6.30・6.34だった（記述値。fitではない）。

**path全体のばらつきの目安**（2番目以降のstepの分散を使い、stepが独立だと仮定した場合）
- path総数の標準偏差は、m5（q=32、R=33）で約86 RZ（観測値68,264の0.13%）、m6（q=64、R=66）で約157 RZ（同149,435の0.10%）、Q8R66（q=8）で約55 RZ。
- 実際の2 pathの標本標準偏差（m5で51、Q8R66で28。各自由度1）と同じ桁である。
- 参考：分散がRに比例すると仮定すると、m9（q=512、R=526）では約1,250 RZになる。C2が公開した2 pathの標本標準偏差1,329とも同じ桁である（自由度1で、これは検定ではない）。

## 5. 直接確認した範囲と、延長の仮定

**保存QPYで直接確認したこと**（q ∈ {1, 2, 8, 32, 64}、R ∈ {2, 33, 66, 526, 4205}）
- 障壁による区切り、区切りをまたぐtokenがないこと。
- 入口・末尾・終端が決定論的で、R・qによらず共通であること。
- 合成の恒等性（q=2、R=66）。
- 厳密な分解 `308 + 1900·q + Σ本体`（13 path）。
- 2番目以降のstepのばらつき（R=33・66）。
- X/Yの本体一致と終端run。

**未観測のq・Rへ延長するときの仮定**
- **q>64**（m7〜m12）：同じ局所的な書換えが、より大きな回路でも成り立つこと。compiler binaryがtagのsourceと同じ挙動をすること、規模に依存する分岐がないこと。
  - q=512は、C2の記述比較 +0.274% だけで、区間の分解は観測していない。S0+でm9のQPYを分解すれば直接確かめられるが、今回は範囲外。
- **R=1052・2103**（m10・m11）：同じRのstepを観測していない。平均/RはR=526・4205で6.30・6.34で、その間は線形だと仮定する。
- **Rの大きいstepのばらつき**：2番目以降の位置では観測していない（最初のstepでn=2だけ）。
- **order2・非自明な位相**：UNKNOWN（今回も全306 stepがorder0）。
- **実sampler law**：理想lawとの差は未資格のまま。

## 6. 保全と、行っていないこと

- run01の判定・原結果・REPORT・HANDOFFは変更していない。
- private cache、固定SOURCE、旧SCOPE_DEVIATION、旧STOPも変更していない。
- S0+・S1・S2、追加取得、再fit、CI・期待総費用の算出、希少eventの調査は行っていない。
- r1で規則を追加変更したり、再試行したりしていない。

## 7. 変更したfile

**規則の固定：`5a18b792a`**
- `scripts/pauli_ld2_step_audit.py`：r1の確定、不一致位置の記録、出力先run02
- `tests/check_pauli_ld2_step_audit.py`
- `AUDIT_RULES_R1.md`、`TESTS_R1_FROZEN.txt`、`run02/.gitignore`

**run02の結果：`43755e0ca`**
- `run02/` 一式：`PATH_*.json` 13件、`AUDIT_RESULT.json`、`AUDIT_TABLE.csv`、`SEGMENTS.csv`、`STEP_STATS.csv`、`EDGES.json`、`FINAL_RUN.csv`、`PROGRESS.jsonl`、`RUN_ALL.json`、console、time、`P4_RZ_CHECK.py`・`.txt`
- `run02/work/` のtoken配列は、QPYから再生成できるためcommitしていない。

**報告（このcommit）**
- `REPORT_R1.md`、`HANDOFF_R1.md`
- `docs/README.md` の索引の行を更新

**mandatory STOP。**
