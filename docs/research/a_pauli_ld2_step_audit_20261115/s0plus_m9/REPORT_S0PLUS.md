# PR-Pauli LD2 S0+：C2の保存m9回路のstep分解（run03）

2026-11-15 JST。Codex Aの担当分（chat A1109が実施）。

**対象**：C2の保存m9回路。q=512、R=526、2 path × X/Y。
- 読んだもの：QPY 8件（wrapper 4・compiled 4）とcache JSON 4件。読取専用。
- 新規のsampling・build・compile、再fit、S1、価格予測、E[C]・CI・Gは0件。

**規則とSOURCE**
- 規則：[AUDIT_RULES_S0PLUS.md](AUDIT_RULES_S0PLUS.md)。path単位の監査は、S0-r1の固定code（`5a18b792a`）の `audit_path` を変更せずに使った。
- 規則・入力（[INPUTS_M9.json](INPUTS_M9.json)）・run02から固定した決定論部分の参照（[REFERENCE_RUN02.json](REFERENCE_RUN02.json)）・人工test 18件（PASS、[TESTS_S0PLUS.txt](TESTS_S0PLUS.txt)）を `85921fcf0` で固定・pushし、remoteとの一致を確かめてから、m9のQPYを開いた。
- run01・run02とC2の結果は変更していない。

## 結論

| 検査 | path0 | path1 |
|---|---|---|
| P1 束縛（X/Y） | PASS / PASS | PASS / PASS |
| step境界のancilla CX列（順序まで完全一致、X/Y） | PASS / PASS | PASS / PASS |
| run02の決定論部分との一致（入口・末尾・終端） | **PASS**（512 step × 9 wire） | **FAIL**：step 1・wire 3・入口の位置374（最初の不一致を記録して終了） |
| `308 + 1900·q + Σ本体RZ` による保存総数の復元 | **一致**（2,678,974） | 一致（2,680,853）。ただし、決定論部分の一致の裏付けはstep 1で止まった |
| X/Yの本体一致 | 一致 | 一致 |
| 最終run（run02の関係から予測したθ₁との一致、RZ 2個） | 一致 | 一致 |
| P4 | 適用外（m9には親子の対応がある合成回路がない） | 同左 |

1. **q=512でも、step境界での区切りは成り立った。**
   - 両path・X/Yとも、ancilla CX列は修正後の期待列と順序まで完全一致した。
   - 各stepのidentity対は5対、event数は526、order0のみだった。
   - 区切りをまたぐ多qubit tokenはなかった。
2. **path0では、512 stepすべての入口・末尾と終端が、run02の決定論部分とtoken単位で一致した。** したがって、q=512で `308 + 1900·q + Σ本体` の分解が厳密に成り立つことを、保存回路で直接確認した。
3. **path1は、固定規則のとおりFAILとした。**
   - 不一致は、最初のstepのwire 3で、run02の入口F（375 token）の最後のtoken（位置374）だった。
   - そのtokenは1q gate（rz）で、wire 3の最初の多qubit tokenより前にある。最初のeventとつながる1q runの中の違いである。
   - run02の入口Fは、13個の最初のstepから共通部分として決めたため、つなぎ目の1 tokenまで含んでいた（run02のP4の1個差と同じwire・同じ種類）。
   - 規則の変更・再試行はしていない。
4. **R=526の2番目以降のstep**（X軸。Yは同じpathなので除く）

   | 範囲 | 独立path | step区間 | 本体RZの平均 | 平均/R | 分散 | 標準偏差 | 分散/R |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | path0 | 1 | 511 | 3331.8 | 6.334 | 3201.2 | 56.6 | 6.09 |
   | path1 | 1 | 511 | 3335.3 | 6.341 | 3427.6 | 58.5 | 6.52 |
   | 2 pathをまとめて | 2 | 1,022 | 3333.6 | 6.338 | 3314.3 | 57.6 | 6.30 |

   - 最初のstepの本体は、path0で3333、path1で3388（各1件）。
5. **run02と並べると**（記述的な比較。fitではない）
   - 2番目以降のstepの「分散/R」は、R=33で7.04、R=66で5.83、R=526で6.30だった。eventあたりの分散は、R=526までほぼ一定である。
   - 「平均/R」は6.78、6.52、6.34だった。

## 1. §10.5の照合と再利用

**照合**（22:34〜22:40 JST）
- C2のbranchのheadは `cb034e547` のままで、m9のstep分解はなかった。
- ほかのbranchの新しいcommit（D：PFの幅の価格対応、B：PFの幅の比較計画、D2：PR-DF LD2）は無関係だった。
- 同じjobは動いていなかった。
- m9のprivate namespace（28 file）は、実行の前後で変化がなかった。

**再利用したもの**
- C2の公開記録をそのまま使い、再計算していない。
  - receipt（`a7fdd200d`）
  - 価格の束縛（`c9986c82f`。2 pathの平均が、公開値5,359,827/2と一致することを確認した）
  - fingerprintと `cache_reused=false`（RESULT）
  - 予測との記述比較（+0.274418%、`adeae8f83`）
- S0-r1の規則関数。
- run02の決定論部分。

## 2. 実行

- 2 pathをprocess並列で、各1回だけ実行した。wallは5分40秒、workerの最大RSSは6.77 GiB、両方の子processがDONEで終了codeは0（[RUN_ALL.json](run03/RUN_ALL.json)、[PROGRESS.jsonl](run03/PROGRESS.jsonl)、`RUN_time.txt`）。
- 1 fileあたりの読込みとtoken化は、wrapperが約46秒、compiledが約80秒（約686万命令）だった。
- 段階は、親processのmonitorで逐次確認した。

## 3. 各検査の詳細

[AUDIT_RESULT_M9.json](run03/AUDIT_RESULT_M9.json)、[SEGMENTS_M9.csv](run03/SEGMENTS_M9.csv)、[STEP_STATS_M9.csv](run03/STEP_STATS_M9.csv)、`PATH_M9_ROUND_p*.json`。

**P1**：4/4で、次がすべてreceiptと一致した。
- QPY・cache JSONのsha256、fingerprint、compiler、6指標、`cache_reused=false`。
- tokenから数えたRZ・CX・命令数。

**ancilla CX列と区切り**
- 4/4で、修正後の期待列と順序まで完全一致した。
- D対は1 pathあたり36,864対（512 step × 72対）すべてが `cx·rz·cx` の形で残り、角度はhexまで一致した。
- 全wireに区切りがあり、区切りをまたぐ多qubit tokenはなかった。

**run02の決定論部分との一致**（X軸）
- path0：全512 stepで、各wireの区間が入口（step 1はF、2以降はP）で始まり、末尾Sで終わった。system wireの終端もEと一致した。
- path1：step 1・wire 3・入口の位置374で不一致になり、固定規則のとおりそこで照合を終えた（記録：[MISMATCH_DETAIL.txt](run03/MISMATCH_DETAIL.txt)）。
  - 不一致のtokenはrzで、多qubit tokenではない。その後、最初の多qubit tokenは位置377にある。
  - つまり、違いは入口の最後の1q runの中だけである。
- **事後の補足**（固定規則の外。判定には使わない）：path1の2番目以降の511 stepは、全wireで入口Pと末尾Sに一致した。

**総数の復元**
- `本体 ＝ 区間RZ − （入口のRZ ＋ 末尾のRZ）`（run02の定数）とすると、次がどちらのpathでも成り立った。
  - path0：Σ本体 1,705,866、`308 + 1900·512 + 1,705,866 = 2,678,974`
  - path1：Σ本体 1,707,745、`308 + 1900·512 + 1,707,745 = 2,680,853`
- 終端（system wire）のRZは156、最後のrunのRZは2で、run02と同じだった。

**X/Yと最終run**
- compiledのX/Yは、最終run以外の全wireで一致した（2/2）。compile前のX/Yの差はsdg 1個だけだった。
- 最終runは `rz·sx·rz` だった。`p` の角度は `−scalar·T` とbit単位で一致し、ZSX合成とも一致した。
- θ₁は、run02の関係から予測したm9の値と一致した（差は約10⁻¹⁵）。θ₂はhexまで同じだった。最後のrunのRZは2個である。

## 4. 観測変動の読み方

**path全体のばらつき**
- 2 pathの本体の合計の差は1,879で、保存総数の差と同じだった。
- 2番目以降のstepの分散（まとめた値3314）から、stepが独立だと仮定すると、1 pathの標準偏差は約1,300 RZ（総数の約0.05%）になる。観測した2 pathの標本標準偏差1,329と同じ桁である。

**最初のstep**
- run02では、最初のstep（R=66でn=5、R=526の単一step cellでn=2）の分散が小さかった。
- m9の最初のstep（3333と3388）にはその傾向は見られない。件数が少なく、結論は出せない。

**Rへの依存**
- 2番目以降のstepについて、平均/Rと分散/Rは、R=33・66・526でそれぞれ6.3〜6.8と5.8〜7.0の範囲にあった。

## 5. 直接確認した範囲と、延長の仮定

**保存QPYで直接確認したこと**
- S0-r1（q ≤ 64）に加え、q=512・R=526の2 path。
  - 区切りが成り立つこと。
  - 決定論部分がrun02と同一であること（path0は全step、path1はstep 1の入口の最後の1 tokenを除く範囲。path1の残りは事後の補足）。
  - 総数の復元。
  - X/Yの本体一致と最終run。
  - 2番目以降のstepのばらつき（1,022区間）。

**未観測の範囲へ延長するときの仮定**
- q ≥ 1024（m10〜m12、event数の約97%）：同じ局所的な書換えが、さらに大きな回路（m12はm9の約64倍のevent数）でも成り立つこと。
- R=1052・2103：同じRのstepを観測していない。R=526・4205の平均/R（6.34・6.34）の間で、線形だと仮定する。
- R=4205の2番目以降のstep：ばらつきを観測していない（最初のstepでn=2だけ）。
- 入口の定義：run02から固定した入口Fの最後の1 tokenは、最初のeventによって変わりうる（path1）。総数の復元には影響しないが、本体の定義の境界として残る。
- order2・非自明な位相（UNKNOWN。今回の1,024 stepもすべてorder0）、実sampler law。

## 6. 保全と、行っていないこと

- run01・run02の結果、C2のm9の価格・比較・予測、private namespaceは変更していない。
- 新規のsampling・build・compile、再fit、S1、価格予測、E[C]・CI・Gは行っていない。
- 不一致の後に、規則の変更や再試行はしていない。

## 7. 変更したfile

**固定：`85921fcf0`**
- `scripts/pauli_ld2_s0plus_m9.py`、`tests/check_pauli_ld2_s0plus_m9.py`
- `s0plus_m9/` の `AUDIT_RULES_S0PLUS.md`、`INPUTS_M9.json`、`REFERENCE_RUN02.json`、`TESTS_S0PLUS.txt`、`run03/.gitignore`

**結果と報告（このcommit）**
- `run03/`：`PATH_M9_ROUND_p0/p1.json`、`AUDIT_RESULT_M9.json`、`SEGMENTS_M9.csv`、`STEP_STATS_M9.csv`、`PROGRESS.jsonl`、`RUN_ALL.json`、console、time、`MISMATCH_DETAIL.py`・`.txt`
- `REPORT_S0PLUS.md`、`HANDOFF_S0PLUS.md`
- `docs/README.md` の索引の行を更新

**mandatory STOP。**
