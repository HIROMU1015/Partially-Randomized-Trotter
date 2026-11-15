# PR-Pauli LD2 S0：保存QPYのstep分解監査（run01）

2026-11-15 JST。Codex Aの担当分（chat A1109が実施）。計画は [`df5e5c452` REPORT §5](../a_pauli_ld2_expected_price_plan_20261115/REPORT.md)。

**行ったこと**
- 判定規則・code・入力を、実QPYを開く前に `2f3bf20ce` で固定・pushした（[AUDIT_RULES.md](AUDIT_RULES.md)、[INPUTS.json](INPUTS.json)、人工test 12件PASSの[TESTS.txt](TESTS.txt)）。
- そのcodeを変えずにS0を1回実行した（run01）。
- 読んだもの：保存QPY 52件（13 path × X/Y × wrapper・compiled）。読取専用。

**行わなかったこと**：新規のpath・sampling・build・compile、fit、価格予測、E[C]・CI・G、private fileへの書込み（いずれも0件）。

## 結論

| 検査 | run01の結果 |
|---|---|
| P1 束縛 | **PASS 26/26** |
| P2 step区切りの復元 | **FAIL 26/26**（ANCILLA_CX_ALIGNMENT） |
| P3 境界寄与の形 | 未評価（P2が全件FAILのため） |
| P4 P2R66と親S66の合成関係 | 未評価（同上） |
| P5 X/Yと終端run | 終端runの部分は**PASS**（26/26、13 round×X/Yの余裕もすべてOK）。X/Yの本体一致は未評価 |
| P6 区間の表 | 未評価（P2が全件FAILのため） |

1. **H_loc（compiled RZ数を「stepごとの寄与＋境界・終端の寄与」に分けられるという仮説）は、今回の保存回路では検査できなかった。**
   - 固定規則が想定していなかった相殺により、step区切りの照合が全件で失敗した。
   - compiled回路のancilla CXは、規則が期待する数より、**どの回路でもstepあたりちょうど10個少なかった**。
2. **原因は事後の診断で特定した**（判定には使っていない）。
   - 欠けた10個は、D(f1) blockの単一Z項5対だった。compile前の角度は ±10⁻¹⁹ 程度で、数値上の0である。
   - Qiskit 1.3.0がこのrzをidentityとして除き、その結果、間にあったCXの対が相殺された。
   - 同じ5対が全path・全stepで同一なので、この欠落はstepごとに決まった定数である。H_locと矛盾する現象ではないが、H_loc自体は未検査のままである。
3. **規則を変えた再実行はしていない。** 修正案S0-r1（3点）と、run01の失敗を再現する回帰testを用意した（未実行・要許可。[AUDIT_RULES_R1_PROPOSAL.md](AUDIT_RULES_R1_PROPOSAL.md)）。
4. **stepごとの観測変動**
   - step区間からは、どのRについても識別できなかった。
   - ただし単一step cell（q=1）では、path全体が1 stepなので、path総数の分散がそのままstep寄与の分散になる。R=66では2、R=526では0.5、R=4205では1152（いずれも自由度1）。
   - R=2・33、および多step path内のstep分散は識別できない（§4）。
5. **B2向け**
   - S0で狭まったのは、終端run（全13 roundでRZ 2個）と、保存回路と公開receiptの束縛だけである。
   - 順位反転の境界（LD2の期待総費用がT_refより75〜82%小さいこと）を左右するround間の移送の不確かさは、run01では狭まっていない（[HANDOFF](HANDOFF.md)）。

## 1. 入力と§10.5の照合

**照合の範囲**（2026-11-15 21:47〜21:55 JST）
- C2 `cb034e547`：m9の最終結果。D2 `705f59ea8`、D `606881b04`、B `aae9f2179`、B2 `a7eceecba`（Pauli T_ref）、C `61f249e6e`、ロードマップ `ad6ab45bb`、A `1adb7e65a`。
- 最近更新された全remote branchのresearch directory名。
- 同じユーザーのprocess。

**照合の結果**
- これらの保存QPYに対するstep分解監査は、ほかになかった。名前が似ているものは、無関係な2026-10-17のdirectoryだけだった。
- 同じjobは動いておらず、出力namespaceも未使用だった。
- B2のT_ref（`a7eceecba`）は、指示どおり重複とは数えない。
- 照合の単位は「path × 軸 × QPY」の52件で、再利用0・新規52・保留0。

**入力の束縛**（[INPUTS.json](INPUTS.json)）
- W：raw `6c4709f85`、受入 `1e2639a1b`
- 較正：`5dac74959`
- holdout：`705f59ea8`、raw `0e1ce866e`
- 物理ID・cell・軸・index・cohortは、公開記録どうし（AXIS_MAP・C_HANDOFF・RECEIPT_ADMISSION・W PRICES）で一致した。
- scalar ＝ 2.515477156730702（S66 RESULTの保存値）。

**規則を固定する前に読んだもの**
- 公開metadata。
- private fileのstat。
- cache JSON 1件（W m0 X、鍵名の確認のため）。QPYは読んでいない。

## 2. 実行と資源

- 規則の固定：`2f3bf20ce` をpushし、remoteとの一致を確認してからQPYを開いた。
- 計測：最小のfile（W m0 X、4,539命令）の読込みは0.05秒、RSSは91.5 MiB（[MEASURE.json](run01/MEASURE.json)）。
- 本実行：13 pathをprocess並列で実行した。wallは12.69秒、workerごとの最大RSSは369 MiB（m6）。最大のcompiled file（m6、328,666命令）の読込みは3.7秒だった。すべての子processがDONEで、終了codeは0（[RUN_ALL.json](run01/RUN_ALL.json)、[PROGRESS.jsonl](run01/PROGRESS.jsonl)、`RUN_time.txt`）。
- 保全：D2のprivate cache 3 directory（計120 file）のfile名・bytes・更新時刻は、実行の前後で変わらなかった。
- 並列実行の形は、決定論PFの閉包監査（`d3338c7ef`）のrunnerを流用した。新しい高速化手法はない。

## 3. 各検査の結果

### 3.1 P1 束縛：PASS 26/26

compiled 26件すべてで、次が成り立った（[AUDIT_TABLE.csv](run01/AUDIT_TABLE.csv)、各`PATH_*.json`の`P1_checks`）。
- QPY・cache JSONのsha256がreceiptと一致する。
- fingerprint、compiler hash、6指標がreceiptと一致し、`cache_reused=false` である。
- tokenから数えたRZ・CX・命令数がreceiptと一致する。
- gateは rz・sx・x・cx・measure だけである。

### 3.2 P2 step区切りの復元：FAIL 26/26

**compile前のwrapperから読んだ構造は、規則どおりだった**（全13 path × X/Y、計306 step）
- 各stepのevent数（`crz`）はRと一致した。product tokenは0件で、すべてorder0のeventだった。
- 各stepのD領域は3 block・77対だった。その内容（対のtarget・角度、blockを区切るrz）は全306 stepで同一だった（記録された署名が1種類）。
- 最後のblock（one-body、8対）も全pathで同一だった。

**compiled回路との照合で失敗した**
- ancilla CXの数が、期待より**stepあたりちょうど10個少なかった**。
  - S66・S526・S4205・Wは10個、P2R66は20個、Q8R66は80個、m5は320個、m6は640個（各q×10）。
- 固定規則は、期待列との完全一致を要求する。そのため、区切り位置の決定より前に26件すべてがFAILになった。

### 3.3 失敗の原因（事後の診断。判定には使わない）

監査済みの1 path（S66/p0 X）のwrapperとcompiled QPYを読み直し、欠けたCXの位置を調べた（[DIAGNOSTIC_P2.py](run01/DIAGNOSTIC_P2.py)、[出力](run01/DIAGNOSTIC_P2.txt)）。新しいfileは読んでいないので、読んだfileは52件のままである。

- 期待列と実際の列の差は、連続した1か所（期待位置188〜197）の削除10個だけだった。
- 削除された10個は、D(f1)の単一Z項5対（wire 0・3・5・6・7）で、compile前のrz角度は +8.7×10⁻¹⁹、−8.7×10⁻¹⁹、−4.3×10⁻¹⁹、+4.3×10⁻¹⁹、+4.3×10⁻¹⁹ だった。
- Qiskit 1.3.0は、角度の絶対値が1e−12未満のrzをidentityとして除く。間のrzがなくなると、`cx(anc,o)` の対が隣り合って相殺される。
- この5対は、全stepで同じ値を持つ（§3.2の署名）。したがって、この相殺はstepごとの定数である。
- これはDF fragmentの対角化で生じた数値上の0であり、compilerの局所的な書換えの範囲内の現象である。stepをまたぐ相互作用ではない。

### 3.4 P3・P4・P6：未評価

- 区切りに成功したpathが0件のため、評価できなかった。
- run01の集計（固定code）は、P3とP6を「FAIL」と表示し、P5のX/Y本体一致を「True」と表示している。これらはいずれも、**空の集合に対する表示であり、判定ではない**。正しくは未評価である。
- この表示の不具合は、修正案S0-r1で直した（§6）。

### 3.5 P5 X/Yと終端run：終端runの部分はPASS、X/Yの本体一致は未評価

**終端run**（[FINAL_RUN.csv](run01/FINAL_RUN.csv)）
- 26/26で、最後のancilla runは `rz·sx·rz` だった。
- wrapperの入力 `p` の角度は `−scalar·T` とbit単位で一致し、入力runをZSXで合成した結果はcompiled runと一致した（26/26）。
- compile前のwrapperでは、X/Yの差はancillaのsdg 1個だけだった（13/13）。
- 観測した5つのT（0.1、0.2、0.8、3.2、6.4）で、θ₁は傾き −scalar の関係に従った。残差はXで最大1.8×10⁻¹⁵、Yで0。θ₂は軸ごとにhexまで同一だった。
- 13 roundのTすべてで、終端角の余裕は1e−6以上だった（26/26でMARGIN_OK）。最小はX軸でm3の0.289 rad、Y軸でm7の0.052 rad。
- したがって、全13 round・X/Yで、最後のrunのRZ数は2個のままである。

**X/Yの本体**
- compiledのX/Yがtoken単位で一致するかは、固定codeでは区切りの後にしか比べないため、未評価である。
- path総数のX=Yは、公開receiptの値で全件成り立っている。

## 4. stepごとの観測変動

- step区間からは識別できなかった。理由はP2の全件FAILである。
- 代わりに、保存標本から識別できる範囲を示す（[PATH_LEVEL_VARIATION.csv](PATH_LEVEL_VARIATION.csv)、作成は [path_level_variation.py](path_level_variation.py)）。
  - 値はP1で束縛したreceiptのRZ数（X軸。Yは同じpath）。
  - P2R66は親に依存するので除く。

| cell | q | R | path数 | RZ（path別） | path総数の分散（自由度） | 解釈 |
|---|---:|---:|---:|---|---:|---|
| W m0 | 1 | 2 | 1 | 2236 | — | 識別できない（n=1） |
| S66 | 1 | 66 | 2 | 2643・2641 | 2（1） | 単一stepなので、step寄与の分散そのもの（最初のstepの文脈） |
| S526 | 1 | 526 | 2 | 5522・5523 | 0.5（1） | 同上 |
| S4205 | 1 | 4205 | 2 | 28830・28878 | 1152（1） | 同上 |
| Q8R66 | 8 | 66 | 2 | 18948・18987 | 760.5（1） | 多step。step分散へ分けるにはH_locが必要 |
| m5 | 32 | 33 | 2 | 68228・68300 | 2592（1） | 同上 |
| m6 | 64 | 66 | 1 | 149435 | — | 識別できない（n=1） |
| m9（C2、参照） | 512 | 526 | 2 | 2678974・2680853 | 1,765,320.5（1） | 同上 |

**識別できないRとその理由**
- R=2：pathが1本だけ。
- R=33：単一step cellがなく、多step pathはstepに分けられなかった。
- R=66・526の多step path内のstep分散：stepに分けられなかった。

**記述上の注意**（検定ではない）
- stepが独立・同分布・加法的（H_loc）だと仮定して、多step pathの分散をqで割ると、stepあたりの分散はQ8R66で95、m5で81、m9で3448になる。
- これに対し、単一step cellの値はS66で2、S526で0.5と小さい。
- いずれも自由度1なので偶然の範囲を区別できない。ただし、「同じRでも、単一stepの文脈と多step pathの中とで、ばらつきが同じか」という問いは、S0-r1の区間分解（m5の64 step、m6の64 step、Q8R66の16 step）で直接測れる。

## 5. 直接確認した範囲と、未観測へ延長するときの仮定

**保存QPYで直接確認したこと**（q ∈ {1, 2, 8, 32, 64}、R ∈ {2, 33, 66, 526, 4205}）
- 26回路と公開receiptの束縛。
- compile前の構造：全306 stepでevent数がR、すべてorder0、D領域は同一。
- compiled回路のancilla CXの不足が、どの回路でもstepあたり10個で一定であること（数の上での確認。位置まで調べたのは1 pathだけ）。
- 終端run：観測した5つのTでの形と関係、および13 roundのTでの余裕。

**確認できなかったこと**
- H_loc（区間ごとの区切り、境界寄与の一定性、P2R66の合成関係）。
- X/Yの本体一致。
- 多step path内のstep分散。

**未観測のq・Rへ延長するときの仮定**（計画 `df5e5c452` §4.5から変わらない）
- H_loc（今回も検査できず、論証のまま）。
- q>512（m10〜m12）への延長。
- R=1052・2103で、Rについて線形であること。
- installされたcompiler binaryがtagのsourceと同じ挙動をすること、規模に依存する分岐がないこと。
- order2・非自明な位相（UNKNOWN）、実sampler law。

## 6. 修正案 S0-r1（未実行・要許可）

[AUDIT_RULES_R1_PROPOSAL.md](AUDIT_RULES_R1_PROPOSAL.md)。code・testは `28f4259f0`、人工test 14件PASS（[TESTS_R1.txt](TESTS_R1.txt)）。

**変更は3点だけ**
1. compile前の角度の絶対値が1e−12未満のD対を、期待されるancilla CX列から除く。
2. X/Yのcompiled一致を、P2と無関係に比べる。
3. 区切りに成功したpathがないときは、P3・P6をUNKNOWNにする。

**回帰test**
- test 13：旧規則のままでは、run01と同じ失敗（ANCILLA_CX_ALIGNMENT、stepごとの不足）を再現する。
- test 14：修正後は、P2〜P6が通る。

**再実行する場合の範囲**
- run01と同じ52件のQPYを読み直す（新しいfileは0）。compileは0。見込み時間は約15秒。
- run01では区間・境界・合成・分散のデータを計算も閲覧もしていない。修正は、compilerの文書化された閾値だけから決めた。

## 7. 保全と、行っていないこと

- private cache、固定SOURCE、旧SCOPE_DEVIATION、原結果、旧STOPは変更していない。
- run01の出力は固定code（`2f3bf20ce`）のまま保存した。修正案のcodeで再集計していない。
- S0+・S1・S2、追加取得、再fit、CI・期待総費用の算出、希少eventの調査は行っていない。
- order2・非自明な位相はUNKNOWNのままである（今回の306 stepもすべてorder0）。

## 8. 変更したfile

すべて `docs/research/a_pauli_ld2_step_audit_20261115/` 以下（codeとtestを除く）。

**規則の固定：`2f3bf20ce`**
- `scripts/pauli_ld2_step_audit.py`、`tests/check_pauli_ld2_step_audit.py`
- `AUDIT_RULES.md`、`INPUTS.json`、`TESTS.txt`

**run01の結果：`7be3bf244`**
- `run01/`：`PATH_*.json` 13件、`AUDIT_RESULT.json`、`AUDIT_TABLE.csv`、`SEGMENTS.csv`・`STEP_STATS.csv`（空）、`EDGES.json`、`FINAL_RUN.csv`、`MEASURE.json`、`PROGRESS.jsonl`、`RUN_ALL.json`、console、time、`DIAGNOSTIC_P2.py`・`.txt`

**修正案：`28f4259f0`**
- 上記のscriptとtestの改訂、`AUDIT_RULES_R1_PROPOSAL.md`、`TESTS_R1.txt`

**報告（このcommit）**
- `REPORT.md`、`HANDOFF.md`、`PATH_LEVEL_VARIATION.csv`、`path_level_variation.py`
- `docs/README.md` に索引1行

**mandatory STOP。** S0-r1の実行、S0+・S1・S2は、それぞれ別の判断と指示に委ねる。
