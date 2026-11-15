# 保存compiled QPY閉包監査：判定規則（実QPYを読む前に固定）

この規則は、計画 [REPORT §6](../d_pf_unobserved_round_closure_20261115/REPORT.md)（`dd435d07e`）と、利用者の実行指示（chat A1109）を具体化したものである。

- 監査code：[`scripts/pf_qpy_closure_audit.py`](../../../scripts/pf_qpy_closure_audit.py)
- 人工test：[`tests/check_pf_qpy_closure_audit.py`](../../../tests/check_pf_qpy_closure_audit.py)
- 入力参照：[INPUTS.json](INPUTS.json)

これらを同じcommitで固定する。このcommitより後に、規則・閾値・分割方法を変えない。

## 0. 範囲

**対象**
- Full PF・B0I p7 × S2・Y4・M5 × X/Y の12 group（route×公式×軸）。
- 監査するq：
  - S2：{7, 14, 28, 224}（m4・m5・m6・m9）
  - Y4：{3, 6, 12, 96}（同上）
  - M5：{2, 4, 32}（m5・m6・m9）

**読み込むもの**
- compiled QPY 44件と、m9のwrapper QPY 12件。計56件で上限どおり。
- cache JSONは読取専用で開く。
- 寸法を測るためだけに、最小のfileを1回余分に読む（同じfileなので件数は増えない）。

**行わないこと**
- private fileへの書込み、build・transpile・compile、未観測60軸の価格値の算出、係数の再推定。

**入力の束縛**
- 公開receipt：較正 `b190bd815`、m6 `a5581337`、m9 `0a351019` の RESULT/PRICES。いずれもHEADでblobが一致することを確認済み。
- 凍結係数：`12372fa6`。
- 未観測roundのT・q：固定済みの [SCHEDULE_IDENTITY.json](../d_pf_unobserved_round_closure_20261115/SCHEDULE_IDENTITY.json)（540 check、PASS。今回は繰り返さない）。

## 1. wire token（既存のwire比較を再利用）

既存の実装を流用する：51d324ea `check_xy_wrapper_locality_artificial.py` の `wires_without_final_ancilla_run`。制御の流れは変えない。

**tokenの定義**
- `circuit.data` の順に、directiveと `measure` を除いた各命令を、`(gate名, qubit番号のtuple（circuit順）, parameterの float.hex のtuple)` とする。
- 各命令は、それが作用するすべてのwireの列に追加する。

**ancillaの扱い**
- ancillaは最後のqubitである。
- ancillaの最後の多qubit命令より後ろにある1q命令（measureは除く）を「最後のancilla run」として列から外し、別に保存する。

**流用元から変えた点**
- parameterの符号化を `round(.,12)` から hex に変えた。
- tokenを整数に置き換え（intern）、qubit番号は表を引くようにした。
- 外した末尾の記録と、命令数を返すようにした。

**同等性の確認**
- 元の符号化に戻すと、元の関数と全件一致する（人工回路200個、test）。

## 2. 周期の抽出と照合（A2）

1. **周期の長さ**：wireごとに `p_w = (len(q_max) − len(q_min)) / (q_max − q_min)` を求める。
   - 正の整数であること。
   - すべての監査qで `len(q) = len(q_min) + p_w·(q − q_min)` が成り立つこと。
   - 満たさなければFAILとする。
2. **分割の決め方**：q_maxの列Sで、`S[i] == S[i+p_w]` が真になる区間のうち最長のものを `[s, s+r)` とする。同じ長さが複数あれば先頭のものを採る。そのうえで次のように決める。
   - `n = ⌊(r + p_w)/p_w⌋`
   - `H_w = S[:s]`
   - `B_w = S[s:s+p_w]`
   - `E_w = S[s+n·p_w:]`
   - `c_w = q_max − n`
3. **cは全wireで共通**：`c = max_w c_w` とする。`c_w < c` のwireでは、余る `B_w` の `(c − c_w)` 個を `E_w` の先頭に移す。
4. **判定**：同じgroupの全監査qで、全wireについて `S_w(q) = H_w + B_w×(q−c) + E_w` が厳密に（tokenの整数が完全一致で）成り立てばPASSとする。
   - `q − c < 0` になる場合はFAILとする。
   - roundごとに別の分割を選ぶことはしない。

## 3. 各規則

### A1　束縛
次をすべて満たせばPASS。

- cache JSONのfingerprintがreceiptと一致する。
- cache JSONの6指標が、PRICESのreceiptと完全に一致する。
- RESULTのtrial価格がPRICESと一致する。
- compiler hashが `1239304c…` である。
- receiptのTのhexがscheduleと一致する。
- `cache_reused=false` である。
- compiled QPYが存在する。
- tokenから数えたRZ・CX・命令数が、保存値と一致する。
  - RZ総数が一致しない場合はtoolの不具合としてSTOPする（§5）。

欠損している、または一意に束縛できないものはUNKNOWNとし、再compileはしない。

### A3　凍結係数との照合（係数は再推定しない）
次をすべて満たせばPASS。

- `Σ_w RZ(B_w)` が凍結した傾きbと一致する。
- 各監査qで、`RZ総数 − (q−c)·Σ_w RZ(B_w)` が `a + c·b` と一致する。
- 同じ値が、`H・E のRZ ＋ 最後のrunのRZ` とも一致する。

### A4　最後のancilla run
gating（これをすべて満たせばPASS）：
- 全監査qで、命令の形が厳密に `rz, sx, rz` である。
- 2番目のRZ（θ₂）のhexが、全監査qで同一である。
- 同じgroup内でscalarが共通である。
- 1番目のRZ（θ₁）が、`|wrap(θ₁(q) − θ₁(q_9) + scalar·(T_q − T_9))| ≤ 1e−9` を満たす。

満たさない場合はUNKNOWNとし、未観測5 roundもすべてUNKNOWN_A4とする。

未観測round m∈{7,8,10,11,12}の扱い（X/Y別）：
- `θ₁(m) = wrap(θ₁(q_9) − scalar·(T_m − T_9))` を求める。ΔTは有理数として正確に差を取ってからfloatにする。
- 余裕 `|θ₁(m)|` は、RZが省略される点（0 mod 2π、Qiskit 1.3.0の `ANGLE_ZERO_EPSILON=1e−12`）からの距離である。
- 余裕が `1e−6` 以上ならMARGIN_OK、未満ならUNKNOWN_SMALL_MARGINとする。

記録だけで判定には使わないもの（m9のwrapper QPY）：
- 最後のancilla runの入力命令。
- `p` の角度が `−scalar·T_9` とbit単位で一致するか。
- 入力runの行列をZSXで合成した結果（transpileはしない）が、compiled runと1e−9以内で一致するか。

### A5　障壁
- gating：各wireの `B_w` にCXが1個以上ある。
- 記録のみ：各system wire kで、本体を巡回的に見て、連続する `cx(anc,k), rz(k), 同じcx(anc,k)` があるか。
- A5が直接示すのは、観測したq点の本体に障壁が存在することだけである。
- 「compilerの全反復で、相殺が境界を越えない」ことは、§3のpassの局所性と入力の同一性から導く論証であり、観測はしていない。

## 4. 未観測60軸の区分

| 項目 | 内容 |
|---|---|
| 最後のrun | MARGIN_OK、UNKNOWN_SMALL_MARGIN、UNKNOWN_A4 のいずれか |
| 周期の延長 | A1〜A5がすべてPASSでも、監査qの外（および間）のqでcompiled回路が `H+B×(q−c)+E` になることは**仮定**として残す（passの局所性、installed Rust binary＝tagのsource） |

- 価格値は出力しない。
- 直接価格の96/156には加算しない。

## 5. 実行とSTOP

**手順**
- まず最小のcompiled file（p7 S2 m4）を1つ読み、時間とRSSを測る（`--measure`）。
- その結果と空きRAMの観測から、groupごとの並列数を決める（`--run --workers N`）。
- 段階ごとのlog（`run01/PROGRESS.jsonl`：load開始・終了、秒数、最大RSS）と、子processの結果（`RUN_ALL.json`）を保存する。faulthandlerの定期dumpは使わない。

**STOP**

| 事象 | 対応 |
|---|---|
| Qiskitが1.3.0でない、またはQPYを読めない | STOP |
| tokenのRZ総数が保存rz_countと一致しない | STOP |
| OOMやworkerの異常 | 部分結果を保全してSTOP |
| private fileへの書込みが必要になる | 禁止 |

- 欠損を埋める再compile、閾値の変更、条件を変えての再試行はしない。

## 6. 出力（`run01/`）

| file | 内容 |
|---|---|
| `GROUP_*.json` | group別の詳細 |
| `AUDIT_TABLE.csv` | 12 group × A1〜A5 |
| `OBSERVED_POINTS.csv` | 44点 |
| `UNOBSERVED_AXES.csv` | 60軸の終端角の余裕と区分 |
| `AUDIT_RESULT.json` | 集計 |
| `MEASURE.json` | 最小fileでの計測 |
| `PROGRESS.jsonl`、`RUN_ALL.json` | 進捗logと子processの結果 |
