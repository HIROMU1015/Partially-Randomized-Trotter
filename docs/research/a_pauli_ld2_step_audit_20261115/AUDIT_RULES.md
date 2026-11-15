# PR-Pauli LD2 保存QPYのstep分解監査（S0）：判定規則（実QPYを開く前に固定）

この規則は、計画 [REPORT §5](../a_pauli_ld2_expected_price_plan_20261115/REPORT.md)（`df5e5c452`）と、利用者の実行指示（chat A1109、S0）を具体化したものである。

- 監査code：[`scripts/pauli_ld2_step_audit.py`](../../../scripts/pauli_ld2_step_audit.py)
- 人工test：[`tests/check_pauli_ld2_step_audit.py`](../../../tests/check_pauli_ld2_step_audit.py)（結果は [TESTS.txt](TESTS.txt)）
- 入力参照：[INPUTS.json](INPUTS.json)

これらを同じcommitで固定する。このcommitより後に、規則・閾値・区切り方法を変えない。

## 0. 範囲

**対象**：保存13 path × X/Y。
- W m0：1 path
- 較正：S66・S526・S4205・Q8R66（各2 path）、P2R66（S66のpath0→path1を連結した1 path、独立標本ではない）
- holdout：m5（2 path）、m6（1 path）

**読むもの**
- QPY：wrapper（compile前）26件と compiled 26件、計52件。private cacheは読取専用で開く。
- cache JSON 26件（fingerprint・compiler・6指標）。
- 公開receipt・C_HANDOFF・RECEIPT_ADMISSION・W のPRICES/RECEIPT_TRACE（INPUTS.jsonを作るときに読んだ）。

**規則を固定する前に読んだもの**
- 上記の公開metadata。
- private fileのstat（存在とbytes）。
- cache JSON 1件（W m0 X、289 bytes）。鍵名を確かめるためで、QPYではない。

**行わないこと**
- build・sampling・transpile・compile、private fileへの書込み、fit、価格予測、E[C]・CI・G、希少eventの調査。
- m9のQPYとprivate namespaceは対象外。m9の公開済み比較（`cb034e547`）は参照だけ。

## 1. tokenの定義

決定論PFの監査（`d3338c7ef` `pf_qpy_closure_audit.py`）と同じ定義を使う。
- compiled回路の各命令を `(gate名, qubit番号のtuple, parameterのfloat.hex)` とし、作用するすべてのwireに置く。directiveは除き、`measure` は数えるがtokenにしない。
- 命令数（size）は、directive以外のすべての命令（measureを含む）を数える。
- ancilla（最後のqubit）の最後の多qubit命令より後の1q命令を、「最終run」として別に扱う。
- 同等性：人工回路で、流用元の `wires_without_final_ancilla_run` と完全に一致すること（test 10）。

## 2. P1　束縛

各compiled QPY（26件）で、次をすべて満たせばPASS。
- wrapper・compiled QPYのsha256が公開receiptと一致する。**一致しなければ、そのpathをSTOPする。**
- cache JSONのsha256、fingerprint、compiler hash（`1239304c…`）、6指標が、公開receiptと一致する。
- `cache_reused=false` である。
- tokenから数えたRZ・CX・命令数が、receiptの値と一致する。**RZが一致しなければSTOP**（toolの不具合）。CX・命令数の不一致はFAIL。
- compiled回路のgateは rz・sx・x・cx・measure だけである。
- 物理ID・cell・軸・index・cohortは、INPUTS.jsonを作る段階で公開記録どうし（AXIS_MAP・C_HANDOFF・RECEIPT_ADMISSION・W PRICES）を照合済みである。

## 3. P2　step区切りの復元

**compile前のwrapperから、stepの構造を読む**（ancillaに作用する命令を、作成順に見る）
- ancillaの `p` は q+1 個ある（各stepのtail位相 q 個 ＋ 最後の `p(−scalar·T)`）。数が合わなければFAIL。
- step s の**tail**：前のD領域の後から、そのstepの `p` までのancilla 2q命令。`crz` は order0 のevent、`cx`・`cy`・`cz` はproduct token（数だけ記録し、調査はしない）。
- step s の**D領域**：その `p` の後から、次の `crz`（最後のstepでは最後の `p`）まで。
  - D領域の2q命令は `cx(anc, o)` だけで、作成順に `cx(anc,o)`・`rz(o)`・`cx(anc,o)` の三つ組になっていること。
  - ancillaの `rz` はblockの区切りとして扱う。
- 各stepのevent数（`crz` の数）が、receiptのRと一致すること。一致しなければFAIL。
- 期待されるcompiled後のancilla CX列：`crz` は2個、`cx`・`cy`・`cz` は1個、D領域の `cx` は1個。

**compiled回路を区切る**
1. compiledのancilla 2q tokenは、すべてancillaを制御とする `cx` であること。そのtarget列が、上の期待列と完全に一致すること。違えばFAIL（ANCILLA_CX_ALIGNMENT）。
2. D領域のすべての対 `cx(anc,o)…cx(anc,o)` について、次を満たすこと。
   - wire o上で2つのCXの間にtokenがちょうど1個あり、それが `rz` で、角度がcompile前と `|wrap(差)| ≤ 1e−12` で一致する。
   - ancilla上では2つのCXが隣り合う。
   - 満たさなければFAIL。hexまで一致した件数も記録する。
3. 区切り位置：step s・wire w の区切りは、「step sのD領域でwを対象とする最後のancilla CX」の位置とする。ancillaの区切りは、step sのD領域の最後のCXとする。全wire・全stepに区切りがあり、単調増加であること。
4. 各tokenは、そのwireの区切り位置から決まるstep区間に属する。**多qubit tokenは、作用するすべてのwireで同じ区間に属すること。** 違えばFAIL。
5. ancillaの最終runは区間から除く。ancillaには、最終run以外の終端部分があってはならない。

X・Yの両方がPASSし、event数がRと一致すればP2はPASS。

## 4. P3　境界寄与の形

P2をPASSした全path（X軸）について、wireごとに次を定める。
- `S_w`：全path・全stepの区間に共通する、最長の末尾。
- `F_w`：`S_w` を除いた最初のstepの区間に共通する、最長の先頭（入口の決定論部分）。
- `P_w`：`S_w` を除いた2番目以降のstepの区間に共通する、最長の先頭。
- `E_w`：最後の区切りより後（system wire）。
- 各区間の**本体（body）**＝区間から先頭（`F_w` または `P_w`）と末尾 `S_w` を除いた部分。

次をすべて満たせばPASS。
- (a) compile前のD領域の内容（対のtarget・角度、block区切りのrz角度）が、全path・全step・X/Yで同一である。
- (b) 最後のD block（障壁）の `(target, 角度)` の列が同一である。
- (c) compiledのancillaのD領域のtoken列が、全path・全stepで同一である。
- (d) すべてのwireで、`S_w` が「最後から2番目のD blockでそのwireを対象とする最初のancilla CX（なければ区切り位置）から区切りまで」を、全区間で覆う。つまり、D(f0)〜D(1b) の領域が、Rにもqにもよらず共通である。
- (e) system wireの `E_w` が全pathで同一である。
- 記録のみ：`S_w` がD領域全体（最初のD blockから）を覆うかどうか。

`F_w`・`P_w`・`S_w`・`E_w` の長さとRZ数を、wireごとに出力する。

## 5. P4　P2R66と親S66の合成関係（X軸）

- 1番目の区間：P2R66とS66 path0で、全wireが完全に一致すること。
- 2番目の区間：P2R66の区間2とS66 path1の区間1を、wireごとに比べる。共通の末尾の長さを c とする。
  - **EXACT**：共通でない部分が、それぞれの入口の先頭（P2R66は `P_w`、S66 path1は `F_w`）の中に収まる。
  - **LEADING_1Q_RUN_ONLY**：共通でない部分が先頭を越えるが、越えた部分に多qubit tokenがない（入口とのつなぎ目の1q runだけ）。
  - **DIFFER**：上記以外。
- 判定：1番目の区間が異なるか、DIFFERが1つでもあればFAIL。全wireがEXACTならPASS_EXACT。それ以外はPASS_LEADING_1Q_RUN_ONLY。
- RZの関係：`Δ = RZ(P2R66) − RZ(S66p0) − RZ(S66p1)` と、決定論部分から導く値 `ΣRZ(P) − ΣRZ(F) − ΣRZ(E) + 最終run(P2R66) − 最終run(S66p0) − 最終run(S66p1)` を記録する。PASS_EXACTのときは一致を確認する。
- P2R66は親に依存するので、独立標本として数えない。

## 6. P5　X/Yの差と最終run

- compiledのX・Yは、最終run以外の全wireでtoken列が完全に一致すること。compile前のwrapperは、ancillaのsdg 1個だけが異なること。
- wrapperの入力run（ancillaの最後の2q命令より後の1q命令）について、次を確認する。
  - `p` の角度が `−scalar·T` とbit単位で一致する。scalar＝2.515477156730702（S66 RESULTの保存値）。
  - 入力runをZSXで合成した結果が、compiledの最終runと1e−9以内で一致する（transpileはしない）。
- 軸ごとの関係（PFのA4と同じ）
  - 最終runの形がすべて `rz·sx·rz` で、2番目のRZ（θ₂）のhexが全pathで同一であること。
  - 最大のT（m6）を基準に、`|wrap(θ₁(T) − θ₁(T_ref) + scalar·(T − T_ref))| ≤ 1e−9` を満たせばPASS、満たさなければUNKNOWN。
- 13 roundのT（AXIS_MAPの正確な値）について、`θ₁(m) = wrap(θ₁(T_ref) − scalar·(T_m − T_ref))` を求める。`|θ₁| ≥ 1e−6` ならMARGIN_OK、未満ならUNKNOWN_SMALL_MARGIN。

すべて満たせばPASS。

## 7. P6　区間の表と観測変動

- `SEGMENTS.csv`：全path・全step（X軸）について、区間・先頭・本体・末尾のRZ数、event数、product数、独立標本かどうか（P2R66はfalse）。
- 分解の完全性：各pathで、`区間RZの和 ＋ 終端RZ ＋ 最終runのRZ ＝ 保存rz_count` であればP6はPASS。
- `STEP_STATS.csv`：本体RZ数の記述統計。
  - X軸だけを使う（Yは同じpath）。P2R66は除く。
  - (R, 文脈) ごとに集計する。文脈はFIRST（最初のstep）とLATER（2番目以降）で、入口が異なるので混ぜない。
  - 件数、平均、不偏分散、標準偏差、分散/R。件数1は「分散を識別できない」、件数2は「自由度1」と注記する。
  - 記述統計だけで、CIやモデルfitではない。

## 8. STOPと禁止事項

| 事象 | 対応 |
|---|---|
| Qiskitが1.3.0でない、QPYを読めない、QPY内の回路が1個でない | STOP |
| QPYのsha256がreceiptと異なる | そのpathをSTOP |
| tokenのRZ数がreceiptと異なる | STOP |
| OOMやworkerの異常 | 部分結果を保全してSTOP |
| private fileへの書込みが必要になる | 禁止 |

- 再compile、閾値の変更、条件を変えた再試行、m9 QPYの読取はしない。
- FAIL・UNKNOWNはそのまま報告する。

## 9. 実行の手順

1. 最小のcompiled file（W m0 X）を1回読み、時間とRSSを計測する（`--measure`）。
2. path単位でprocess並列に実行する（`--run --workers 13`）。段階log（`run01/PROGRESS.jsonl`）と子processの結果（`RUN_ALL.json`）を保存する。定期的なframe dumpは使わない。
3. `--summarize` でP3〜P6の表を作る。
4. `run01/work/` のtoken配列はQPYから再生成できるため、commitしない（`.gitignore`）。
