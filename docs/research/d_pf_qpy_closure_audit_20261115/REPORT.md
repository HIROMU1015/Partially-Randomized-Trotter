# 決定論PF：保存compiled QPYを使った反復ブロックの閉包監査

**範囲**
- 計画：[REPORT §6](../d_pf_unobserved_round_closure_20261115/REPORT.md)（`dd435d07e`）の0-compile監査を実施した。
- 判定規則・監査code・入力参照は、実QPYを読む前に `d3338c7ef` で固定・公開した（[AUDIT_RULES.md](AUDIT_RULES.md)、[INPUTS.json](INPUTS.json)）。
- 読んだもの：compiled QPY 44件と、m9のwrapper QPY 12件（計56件、上限どおり）。
- 新規に行わなかったこと：build、transpile、compile、未観測60軸の価格値の算出、再fit。private fileへの書込みも0件。

**結果**
- 12 group×軸のすべてで、A1〜A5がPASSした。
- 未観測60軸の終端角の余裕は、すべて1e−6以上（最小0.0225 rad）。
- 監査したqの外側（および監査qの間）への周期の延長は、実装上の論証によるもので、観測ではない（§2）。
- 直接価格の96/156には加算していない。

## 1. 保存QPYで直接確認したround・qと、A1〜A5の結果

**監査したq（各group×軸）**
- S2：q = 7, 14, 28, 224（m4・m5・m6・m9）
- Y4：q = 3, 6, 12, 96（同上）
- M5：q = 2, 4, 32（m5・m6・m9。M5のm4は対象外）

| group（X・Yとも同じ結果） | c | A1 束縛 | A2 周期閉包 | A3 本体のRZ ＝ 凍結b | A4 最後のrun | A5 障壁 |
|---|---:|---|---|---|---|---|
| Full S2 16h/7 | 1 | PASS | PASS | 7791 ＝ 7791 | PASS | PASS（8/8） |
| Full Y4 16h/3 | 1 | PASS | PASS | 23535 ＝ 23535 | PASS | PASS（8/8） |
| Full M5 16h | 1 | PASS | PASS | 85653 ＝ 85653 | PASS | PASS（8/8） |
| p7 S2 16h/7 | 1 | PASS | PASS | 5513 ＝ 5513 | PASS | PASS（8/8） |
| p7 Y4 16h/3 | 1 | PASS | PASS | 16539 ＝ 16539 | PASS | PASS（8/8） |
| p7 M5 16h | 1 | PASS | PASS | 60643 ＝ 60643 | PASS | PASS（8/8） |

機械可読版：[AUDIT_TABLE.csv](run01/AUDIT_TABLE.csv)、[OBSERVED_POINTS.csv](run01/OBSERVED_POINTS.csv)（44点）、group別の詳細 `run01/GROUP_*.json`。

**A1：束縛**
- 44点すべてについて、次を確認した。
  - cache JSONのfingerprint・compiler hash・6指標が、公開receipt（PRICES・RESULT）と一致する。
  - Tのhexが固定scheduleと一致する。`cache_reused=false` である。
- tokenから数えたRZ・CX・命令数は、保存値と全件一致した。STOPの条件には該当しなかった。

**A2：周期閉包**
- 全wire（system 8本＋ancilla）、全監査qで、`S_w(q) = H_w + B_w×(q−1) + E_w` がtokenの完全一致（parameterのhexまで）で成り立った。
- cは全groupで1、各wireの `c_w` もすべて1だった。
- 先頭の `H_w` は最大71 token。末尾の `E_w` は、1周期分に −1〜+70 tokenを足した長さである。
- つまり端の効果は、最初の数十命令と最後の1 stepに閉じている。本体の1周期は、監査したqの間でbit単位で同一だった。

**A3：凍結係数との照合**
- 本体1周期のRZ総数が、凍結した傾きbと一致した。
- 各監査qで、本体以外のRZ数が凍結した `a + c·b` と一致した。
  - 内訳：端のRZ ＝ `a + b − 2`、最後のrun ＝ 2（例：Full S2では 8096＋2＝8098＝307＋7791）。

**A4：最後のancilla run**
- 全監査qで `rz, sx, rz`（SXは1個）だった。2番目のRZ（θ₂）は、q間でhexが同一だった。
- 1番目のRZ（θ₁）は、傾き −scalar の関係に残差 ≤1.07×10⁻¹⁴ で整合した（基準は1e−9）。scalarはFullで2.29310124732、p7で2.293146549384713。
- m9のwrapper QPYで、入力側を確認した（判定には使わない記録）。
  - 入力の並び：Xは `rz, p, h`、Yは `rz, p, sdg, h`。
  - `p` の角度は、`−scalar·T` とbit単位で一致した。
  - 入力runの行列をZSXで合成した結果は、compiled runと一致した（transpileはしていない）。
  - 入力runの `rz(2·gp₀)` は、FullとB0Iで同じ値だった。one-body blockの位相が共通であり、共通の切片307と整合する。

**A5：障壁**
- 全group・全wireで、本体 `B_w` にCXがあった。
- 全system wire kで、`cx(anc,k)·rz(k)·cx(anc,k)` の連続した並びが本体の中にあった（8/8）。
- A5が直接示すのは、観測したq点のcompiled本体にこの障壁が存在すること、までである。

**資源**
- 最初に最小のfile（p7 S2 m4、81,259命令）を読んで計測した：0.89 s、RSS 110 MiB（[MEASURE.json](run01/MEASURE.json)）。
- 本実行は12 group並列で、wallは100.5 s、workerごとの最大RSSは1.47 GiB、すべての子processがDONEで、終了codeは0だった。
- 最大のfile（Full M5 m9、5,752,332命令）の読込みとtoken化は65.6 sだった（[PROGRESS.jsonl](run01/PROGRESS.jsonl)、[RUN_ALL.json](run01/RUN_ALL.json)、`RUN_time.txt`）。

## 2. 未観測60軸：確認できたこと、延長の根拠、残る仮定

**終端角（観測値から延長して数値で確認）**
- 60軸すべてで、θ₁の余裕は1e−6以上（MARGIN_OK）だった（[UNOBSERVED_AXES.csv](run01/UNOBSERVED_AXES.csv)）。
- つまり、未観測roundのTでも最後のrunのRZ省略は起きず、最後のrunのRZ数は監査点と同じ2のままである。
- 公式・roundごとのX/Yの最小値は次のとおり。

| 公式 | m7 | m8 | m10 | m11 | m12 |
|---|---:|---:|---:|---:|---:|
| S2 | 0.979 | 0.485 | 0.666 | 1.431 | 0.162 |
| Y4 | 1.044 | 1.603 | 1.784 | 0.583 | 2.186 |
| M5 | 1.182 | 0.688 | 0.697 | 1.634 | 0.0225 |

**監査したqの範囲と、未観測qの位置**

| 公式 | 監査したq | 未観測のq：範囲内 | 未観測のq：範囲外 |
|---|---|---|---|
| S2 | 7〜224 | 56、112 | 448、896、1792（最大で監査最大の8倍） |
| Y4 | 3〜96 | 24、48 | 192、384、768 |
| M5 | 2〜32 | 8、16 | 64、128、256 |

**周期を延長するための実装上の根拠**
保存点での一致だけでは、任意の反復数への延長は確定しない。延長は次の論証による。

1. **入力が同一である**：未観測roundの入力piece列は、監査したroundとbit単位で同一である（固定済みのSCHEDULE_IDENTITY、540 check）。違いはstep数とancillaの `p(−scalar·T)` だけである。
2. **piece境界で回路が切れる**
   - SOURCE（`append_directional_diagonal`）では、one-body blockの方向制御が、各pieceの終わりに全system wireの `cx(anc,k)·rz·cx(anc,k)` をk順に連続して並べる。
   - この区間には、system wire同士の2q gateがない。
   - 監査では、その障壁がcompiled本体に全wireで残っていることを確認した（A5）。
3. **passはCXを作らず、局所的にしか書き換えない**（Qiskit 1.3.0。Pythonはinstalledのsource、Rustはtagのsource）
   - translation以後に新しいCXは生じない。1q再合成はrz・sx・xだけを出力し、unrollは発動しない。
   - CXはInverseCancellationによって、隣接する同一の対としてしか消えない。
   - 1q runは2q gateで区切られる。
   - 1q再合成が置換するのは、gate数が厳密に減る場合だけである。
   - 以上から、障壁は全反復で保たれ、書換えは障壁の間の区間（ほぼ1 pieceずつ）に閉じる。
4. **固定点ループはqに依存しない**
   - 区間ごとの変化は、その区間の内容だけで決まる。
   - 変化は必ずsizeを減らすので、全区間が局所的な固定点に達するまでループが続く。
   - したがって、step数が2以上なら出力は `H + B×(q−1) + E` になり、H・B・Eは監査で得たものと同じになる。

**残る仮定**
- installされたRust binaryが、tag 1.3.0のsourceと同じ挙動をすること。bit単位の同一性は確認していない。
- 大きなq（回路規模はm9の最大8倍）でも、Qiskitに規模依存の分岐がないこと。読んだsourceには見当たらなかったが、網羅はしていない。
- 未観測roundの回路そのものは観測していない。

**残るFAIL・UNKNOWN**
- 規則上は0件。
- ただし、上の延長は観測ではなく論証である。未観測60軸を「直接確認済み」とは扱わない。

## 3. Cが価格の採否を判断するための材料

1. **確認済みの事実**
   - 6候補×X/Yの監査点で、compiled回路は `H + B×(q−1) + E` の形だった。本体1周期はbit単位で同一で、そのRZ数は凍結傾きbに、端と最後のrunのRZ数は `a + b` に一致した。
   - 凍結した直線 `a + b·q` は、本体を繰り返す構造そのものとして説明できる。
   - 未観測roundでも、最後のrunのRZ数は変わらない。
2. **延長の根拠と仮定**：§2の論証4点と、残る仮定3点。採否の基準は、Cがこれらを受け入れるかどうかで決まる。
3. **直接取得は必須とは考えない**
   - 監査はすべてPASSで、延長の前提はSOURCEと観測で裏付けられている。
   - Cが、監査範囲外のqについて観測による確認を採否の条件にする場合だけ、最小案として1 groupのm12 X/Yを挙げる。
     - 候補：p7 S2（q=1792、m9の8倍。見積り約2.1 h）。
     - 理由：監査範囲を最も大きく超える点で、安価に取得できる。
   - 対象・負荷は、既存計画 `dd435d07e` §6.6のとおり。
4. **今回は行っていないこと**
   - 60軸の値をCの表へ入れること。
   - 2%の判定、MSE・N・G。
   - 凍結JSONの資格（`qualified_for_candidate=false`、`long_q_extrapolation_qualified=false`）の変更。

## 4. 実行前の照合（§10.5）と保全

**前回（`dd435d07e`）以後の照合**
- 同等の監査：なし。
- このbranchの更新：なし。
- 自分のjob：なし。動いていたのは別の研究（Full-H D7、H8）のjobで、無関係なので操作していない。
- 公開receipt：12件ともHEADとblobが一致した。
- private QPY・cache：監査の前後で新しいfileはなかった。

**再利用したもの**
- 540件のschedule検査（繰り返さず、固定結果を参照した）。
- 51d324eaのwire比較。
- 公開receiptのfingerprint。
- 凍結係数。

**保全**
- 元のSOURCE、raw、凍結JSON、旧STOP、privateのcache・QPYは変更していない。

## 5. 人工test

[TESTS.txt](TESTS.txt)：18件、すべてPASS。transpileも保存入力も使っていない。

**確認したこと**
- 流用したwire比較が、元の関数と一致する（人工回路200個）。
- ZSXの `rz·sx·rz` 形で、1番目のRZの角度が入力位相の分だけずれる。sine軸では −π/2 ずれる。
- 人工の周期回路でA1〜A5がPASSする。

**意図どおり失敗・停止することの確認**

| 入れた異常 | 結果 |
|---|---|
| 本体を1 ulpずらす | A2 FAIL |
| 凍結傾きを1変える | A3 FAIL |
| RZ総数を食い違わせる | STOP |
| 角度の関係を壊す | A4 UNKNOWN |
| QPYが欠損している | UNKNOWNになり、再compileしない |
| 余裕が1e−8しかない | UNKNOWN_SMALL_MARGIN |

**mandatory STOP。** m12の直接取得や、価格モデルの正式な採用は、別の判断に委ねる。
