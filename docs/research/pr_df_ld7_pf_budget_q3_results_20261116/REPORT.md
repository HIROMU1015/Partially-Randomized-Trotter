# PR-DF LD7：PF達成費用予算での有限shot配分のQ3結果

2026-11-16 JST、B2（Claude Code、主agentのみ。独立レビューではない）。

**生成した4列はすべてQ3モデル不適合（4件とも下界>τ）。適合点は無く、LD7の確認済み最安はA_SHAPEの1,235,340,731.912698 RZのまま。** 新規Q3は4件（各1回）で、停止理由は `FINITE_MENU_EXHAUSTED`。判定保留（INCONCLUSIVE）・UNKNOWN・retryは0件。新規信号・sampling・compile・価格再fitも0件。

## 4列と結果（費用の安い順）

対象はPR-DF LD7。条件は保存H4 linear 1.00 Å／STO-3G／tol-only DF10、one-bodyは別head、DF0..6 head・DF7..9 tail、S2・Δ=0.1・K2・共通κ、全13round R=1。configurationは `ce9c7f08…`。予算はp7 S2の395,962,678 RZとFull PF S2の559,457,386 RZ。各予算でLD7の既存規則A_SHAPE（重みV0）とB_PRICE（重みV0/C_ref）を1本ずつ適用し、各軸の上限はV0 `[64,60,56,52,47,43,38,34,30,25,21,17,12]` とした。X=Yの列になった。

| 順 | 予算・規則 | N_X = N_Y（m0〜m12） | 条件付き費用 RZ | 予算残額 RZ | m7〜m12割合 | MSE下界 Ha² | MSE上界 Ha² | 下界/τ | 判定 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | p7・A_SHAPE | [15,14,13,12,11,10,9,8,7,6,5,4,2] | 352,435,885.500000 | 43,526,792.500000 | 95.5767% | 8.5621124779798553E−4 | 8.5621126040422472E−4 | 337.1 | 不適合 |
| 2 | p7・B_PRICE | [64,60,56,52,47,43,38,34,26,10,4,1,1] | 390,742,769.682540 | 5,219,908.317460 | 82.9922% | 3.4582889791669221E−5 | 3.4582889791669221E−5 | 13.62 | 不適合 |
| 3 | Full PF・A_SHAPE | [22,21,19,18,16,15,13,11,10,8,7,5,4] | 537,571,380.119048 | 21,886,005.880952 | 95.7475% | 1.5562136302719190E−5 | 1.5562136306556324E−5 | 6.128 | 不適合 |
| 4 | Full PF・B_PRICE | [64,60,56,52,47,43,38,34,30,19,7,3,1] | 548,149,803.126984 | 11,307,582.873016 | 87.8761% | 8.7332943105188905E−6 | 8.7332943105188905E−6 | 3.439 | 不適合 |

- exact費用：列1 704871771/2、列2 24616794490/63、列3 22577997965/42、列4 34533437597/63 RZ。各列の26軸 N×C_ref、m0〜m6／m7〜m12小計は [INPUTS](INPUTS.json) と [SUMMARY](SUMMARY.json) にある。m7〜m12の割合はその列自身の費用から求めた。
- 判定は exact τ=(0.00159360010199040 Ha)²=0.0000025395612850638132820416921600 Ha² と原APIのexact有理数による。表の小数は表示用。4件とも**下界がτを上回る**ため、保守的な上界だけが大きいわけでも、判定保留でもない。
- 不適合の結論は評価したこの4列に限る。同じ予算の別の列、別の生成規則、V0を超える振替は評価していない。

### 両PF基準との関係

適合点が無いため、PFの基準と並ぶ「適合した配分での比較」は得られなかった。参考として各列の T−T_PF を示す（不適合列なので比較の根拠にはならない）。

| 列 | T − 395,962,678（p7） | T − 559,457,386（Full PF） |
|---|---:|---:|
| p7・A_SHAPE | −43,526,792.500000 | −207,021,500.500000 |
| p7・B_PRICE | −5,219,908.317460 | −168,714,616.317460 |
| Full PF・A_SHAPE | +141,608,702.119048 | −21,886,005.880952 |
| Full PF・B_PRICE | +152,187,125.126984 | −11,307,582.873016 |

**LD7の確認済み最安は A_SHAPE `[48,45,42,39,35,32,29,25,22,19,16,12,9]` の1,235,340,731.912698 RZ**（exact 155652932221/126、D2 `2b3b3bcc9` で適合）。この値はp7の約3.12倍（+839,378,053.912698）、Full PFの約2.21倍（+675,883,345.912698）である。

## 診断（判定には使わない）

| 列 | bias² Ha² | variance Ha² | any-zero確率 |
|---|---:|---:|---:|
| p7・A_SHAPE | 1.79×10⁻⁸ | 8.56×10⁻⁴ | 0.0189 |
| p7・B_PRICE | 1.84×10⁻⁷ | 3.44×10⁻⁵ | 0.00144 |
| Full PF・A_SHAPE | 9.34×10⁻⁹ | 1.56×10⁻⁵ | 0.00040 |
| Full PF・B_PRICE | 1.24×10⁻⁸ | 8.72×10⁻⁶ | 6.4×10⁻¹⁴ |

4列とも分散が支配的である。B_PRICE列は早いroundをV0のまま保ち、m9〜m12を大きく削った。A_SHAPE列は全roundを同じ比率で削った。どちらも、これらの予算では採択閾値に届かなかった。PF（p7・Full PF）とは信号もroundごとの価格も違うため、同じ予算での適否を直接比べることはできない。

## 数値gate

| 列 | 閉包（ρ=1）隣接組 | 遷移組 | 非終端frontier最大 | 残余質量 | 残余loss上界 Ha² |
|---|---:|---:|---:|---:|---:|
| p7・A_SHAPE | 10,615 | 2,851,879 | 8,192 | 1.06×10⁻¹³ | 1.26×10⁻¹¹ |
| p7・B_PRICE | 2,251,024 | 83,371,552 | 8,192 | 1.12×10⁻³² | 4.57×10⁻³¹ |
| Full PF・A_SHAPE | 47,621 | 7,632,403 | 8,192 | 4.1×10⁻¹⁷ | 3.84×10⁻¹⁵ |
| Full PF・B_PRICE | 2,275,980 | 86,698,644 | 8,192 | 3.58×10⁻³¹ | 1.17×10⁻²⁹ |

- 4件とも `VERIFIED_FINITE_BINARY64_SUPPORT`。遷移組は上限141,559,036／callの内側。残余は捨てずに上界へ会計しており、下界とτの差（最小でも約6.2×10⁻⁶ Ha²）に比べて無視できる。
- 保存結果の検査（既存の `validate_result`）を4件とも通過した。検査項目は、first-unsafe分割＋all-good=1、各round good＋unsafe=1、保持MSE寄与＋残余loss上界=DP上界、moments区間とMSE上下界の整合、exact閾値判定、原APIとRESULT.recordの一致。
- transition allowanceは既存の1×10⁻¹² rad/transition（engineering allowance）で、厳密な数値certificateではない。

## SOURCE・条件付き会計

- 科学実行SOURCE：**`cf19fc36ff76167fc2fe86d2a671c40918ea2c5f`**（caller `scripts/run_pr_df_ld7_pf_budget_q3_fixed.py`、[INPUTS](INPUTS.json)、人工テスト）。4列・exact費用・残額・費用順をQ3の前にcommit・pushした（[SCOPE_AND_PREFLIGHT](SCOPE_AND_PREFLIGHT.md)）。
- 信号・評価契約：D2 `0defa286c` の LD7 V0 Q3_INPUTS（保存13round信号・binding・target、Z1＋T1 total decoder、保存DF10 solver proxy −2.1663874486347634 Ha、全事象平方loss、保存ideal finite-K raw・IID二項・EXACT_AFFINE、exact τ）。caseはV0 caseの複製で、allocationだけを差し替えた。
- 価格：D2 `4326422a4` のLD7自身の26軸 `C_ref=α+βq`（m0/m6 full-wrapper Cの2点モデル、X/Y同値）。m7〜m12はq64より先の未検証外挿で、今回の4列では費用の83〜96%を占める。誤差上界・CIは無く、法則上のE[C]でもない。
- 生成：`e094b2d60` の有理数breakpoint `generate`。D2の保存LD7 menuを同じ関数・価格で再現できることを確認した。
- 経路：D2のLD7と同じ `run_pr_df_ld_split_q3_resume.py`（`460d77d8f` と同一bytes）の supervise／worker を使った。条件は固定SOURCE view `27e70efe5`、元AST guard、Python 3.11.0rc1、8 DP worker／batch 8。

## 実行・保存・負荷

- 1件ずつ逐次に実行した（最初の適合で止める規則のため）。4件とも SCIENTIFIC_CALL_START 1回・`WORKER_ARTIFACTS_SAVED`。workerが原API・partials・RESULT・RESOURCE_FINAL・PERSISTEDを自分で保存した。pool worker processは4件とも終了code 0で、supervisor・monitorのエラーは無い。caller全体も終了code 0（[PROCESS_RETURN](PROCESS_RETURN.json)）、全体wallは約415.8秒。
- 列ごとのwallは26.9／162.2／43.4／176.9秒。worker peak RSSは194,960／748,704／246,208／801,760 KiB。親monitorが定期観測したprocess-tree RSSの最大は501,692／4,978,072／1,422,440／5,312,468 KiB（厳密なpeakではない）。詳細は各 `run01/NN/<label>/RESOURCE_FINAL.json` と `SUPERVISION.json` にある。
- 実行中、別チャット（D2のLD9価格取得、Dのp7 Yoshida4較正）のjobが並行して動いていたが、互いに触れていない。
- 実行前テスト：新10件と、D2の再利用経路の既存2件、計12件がPASS（[TESTS](TESTS.txt)）。

## 資格の範囲とSTOP

保存ideal raw・IID二項・solver proxyと、LD7の探索用価格モデルに条件付くモデル判定である。十分N・最小N・物理的global RMSE・正式G_qualified・winnerではない。

固定SOURCE・保存LD7の原結果・旧STOPは保全した。新規信号・sampling・compile・価格再fit・retryは各0件。公開とremote照合の後 **mandatory STOP**。
