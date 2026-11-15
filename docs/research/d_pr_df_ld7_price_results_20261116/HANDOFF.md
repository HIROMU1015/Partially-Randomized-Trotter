# C・統合担当向け：LD7段階3〜5完了

**LD7はA_SHAPEがV0/A/B内の最小適合点。モデル費用1,235,340,731.912698 RZ、MSE上界5.56517630181748840998541E-7 Ha²。** X/Yとも `[48,45,42,39,35,32,29,25,22,19,16,12,9]`。新規10 path・20 compile・fit2・26価格・配分2・Q3各一回計2。全receipt有効、両Q3適合、retry/欠測/保留0。次候補は計画の分岐に従いLD9を提案する。今回の取得はLD7で終了。

## Cが取り込む固定入口

**機械可読join commit：`fd010b111e95411702be51adc73e56b0bba6c093`**

- `docs/research/d_pr_df_ld7_price_results_20261116/C_HANDOFF.json`：configuration→model/26軸価格→配分→Q3判定→26加重項/費用。各価格のmodel/sourceから元path・cohort/seed・axis・fingerprint・compiled receiptに戻れる。
- 同directory `SUMMARY.json`：LD1/LD2保存6点＋LD7 V0/A/Bの9点、共通V0、候補別最小点、m0..6/m7..12内訳、資格・次候補。
- `MENU.csv`：9点の費用・MSE・X/Y整数列。
- `PRICE_RECEIPTS.json` / `.csv`：20軸receipt、同一path S/C、cohort/index/seed、physical/configuration ID、event digest、fingerprint、原locator、gate/資源。
- `SOURCE_LOCATORS.json`：実行SOURCEと結果公開commitを分けた40桁参照。

C向けsource/receipt照合はD2が実施したもの。C自身による受入は未実施。`price_acceptance`は有限cohort由来の探索モデルを示し、厳密なlaw期待価格やCIとして受け入れるものではない。新しい承認用packetやauth手順は設けていない。

| 成果物 | 固定commit | path（当directoryから） |
|---|---|---|
| 取得前のrequest/cohort/全seed | ae623e10a | ACQUISITION_INPUTS.json |
| 較正8 path/16 compile raw | ec09e36cd | calibration/acquisition/RESULT.json、m0/m6/p*/cosine.json・sine.json・PATH_RECEIPT.json |
| 軸別fit2・26正価格 | 4326422a4633ed46968d6e348fe5ffaa24fbb468 | FROZEN_MODELS.json、AXIS_PRICES.json |
| 独立m3 raw | 13423065b | holdout/acquisition/RESULT.json、m3/p*/cosine.json・sine.json・PATH_RECEIPT.json |
| 独立記述比較 | c160ffb2f | HOLDOUT_COMPARISON.json |
| 配分/費用/予算残額 | ca7c016f5b6bc084de82e047ffc30befea73cb4c | ALLOCATION_INPUTS.json |
| Q3観測前固定入力 | 37b20a07d8ee3535c925d7abdcd15dea914b9418 | Q3_INPUTS.json |
| Q3二件のraw・API・閉包・資源 | 2b3b3bcc909c889d82d0302078725fd328579421 | q3/LD7_A_SHAPE、q3/LD7_B_PRICE、q3/COMPLETION.json |

表の短縮SHAはSOURCE_LOCATORSの40桁値へ解決済み。configuration/信号/元V0は0def HANDOFFを再利用し、新規生成・信号計算・V0再Q3は0。LD7 original configuration IDは `ce9c7f08234089f0b50fd786c20403fe02c5de9073e89d6d35a12ee9d966c5e1`。Q3のcanonical total view/binding IDはALLOCATION_INPUTS.casesに保存され、物理価格側の元IDと区別する。

## 二通りの比較

| L_D | 配分 | モデル費用 (RZ) | MSE lower..upper (Ha²) | 判定 | m7..12費用割合 |
|---:|---|---:|---|---|---:|
| 1 | V0 | 261,314,830,475.559718 | 4.72462588934653801951643E-7 .. 4.72462588934653960729534E-7 | 適合 | 99.923854% |
| 2 | V0 | 63,914,076,114.980849 | 4.72409731948051054908399E-7 .. 4.72409731948051194818804E-7 | 適合 | 99.896387% |
| 7 | V0 | 1,668,829,456.071429 | 4.28414160318294721301353E-7 .. 4.28414160318294721365383E-7 | 適合 | 96.017751% |

候補別最小点：

| L_D | 配分 | モデル費用 (RZ) | MSE lower..upper (Ha²) | 判定 | m7..12費用割合 |
|---:|---|---:|---|---|---:|
| 1 | B_PRICE | 189,232,849,610.910739 | 7.67458737806743819998359E-7 .. 7.67458737806743978776249E-7 | 適合 | 99.894849% |
| 2 | B_PRICE | 46,341,978,336.212733 | 7.67401652027366652784195E-7 .. 7.67401652027366792694600E-7 | 適合 | 99.857098% |
| 7 | A_SHAPE | 1,235,340,731.912698 | 5.56517630181748821477536E-7 .. 5.56517630181748840998541E-7 | 適合 | 95.945359% |

全9点・費用内訳はREPORT/SUMMARY参照。LD1/LD2は8ec7c882806809460d38a638a891bafcf18fafd1をそのまま再利用。表示額でなくexact有理数の実費用を比較した。LD1=E_I[S]＋残差、LD2=full-C三点、LD7=full-C二点という根拠の差を保つ。LD7の選択点はm7..12価格外挿が95.945359%を占め、未検証のまま。

## 価格モデルと独立性

m0 q1/R1の4 path平均12999/2、m6 q64/R1の4 path平均1575349/4から、X/Y別に `alpha=88523/252, beta=1549351/252` を固定。モデルは `C_ref=alpha+beta*q`、offset0の測定付きfull-wrapper Cであり、Sは診断専用で再加算しない。全R1のためbetaはhead/tailを合算した傾き。LD2係数の流用0。

モデル公開後のm3 q8/R1は独立cohort2 path。各軸P=1783333/36、O=49690、P−O=−5507/36、相対差=−5507/1788840（約−0.307853%）。`DESCRIPTIVE_COMPLETE` / `NOT_ASSESSED_NO_ADOPTED_TOLERANCE`。補間域の記述比較のみで再fit0。m3を較正へ戻していない。凍結モデルの「未開封」欄は凍結時点の記録であり、後の比較sidecarと合わせて読む。

固定H4・fragment順・one-body別head・DF0..6 head/7..9 tail・S2/Δ=.1/K2/共通κ、全R1単一piece、cとs_Rの別SOURCE・総scalar一回を維持。compilerはP0/full measured wrapper/Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、新ZZ、symmetric-directional。cohort/seedは取得前固定、元normal samplerで引き直し0。X/Yは同pathの別軸であり20独立pathではない。

取得gateは元P2と同じinput/preparation/request/lowering/compiler/fingerprintの範囲。実full-wrapper statevector・numerical action gateは未実施。人工prefix7 fixtureはこの実装経路の技術検査で、全pathの作用証明や価格モデル性能の保証には使わない。Q3は元3.11.0rc1 worker/AST guard、元decoder・target・loss・exact閾値のまま各一回。例外・数値gate停止・不適合・判定保留は今回は発生しなかった。

## ロードマップ差分・次の判断・STOP

LD7は「共通V0適合・価格未取得」から「固有価格較正→モデル固定→独立価格holdout→独自A/BともQ3適合→3点内最小A_SHAPE確認」へ進んだ。ロードマップ本体は統合担当が更新する。

f89 REPORTの規則「LD7の最小適合モデル費用がLD2より低ければ次はLD9」に該当。LD9を提案するが、今回configuration/schedule/signal/価格/配分は未生成。さらにheadを増やしtail1本となる領域で交換関係を調べる問いを残す。LD3/5が劣るとは判断せず、全L_Dの最適性・単調性は認定しない。LD7長round価格とLD2の外挿制約も解消していない。

§10.5はf89/0defの照合を再利用し、取得前と新Q3前にremote・同候補/全vector/cohortのreceipt・model・job・出力競合だけを差分確認した。同等価格/新配分Q3の再利用0、曖昧対象0。他方式のjobは除外・非介入。後続実行前は今回以後の差分を再確認する。

厳密E[C]/CI、実PCG64 joint lawへの移送、全law誤差・ground保証、正式G/winnerは未認定。SOURCE・raw・旧STOP、LD1の凍結価格・m4独立holdoutを保全した。20 compile/2 fit/26価格/2配分/2新Q3で終了し、LD9取得・追加holdout・別配分・retryには進まない。commit/push・remote照合後mandatory STOP。


読取例（exact MSEに大きな整数を含むため、既存reader同様にPythonの10進桁数制限を解除して読む）：

```python
import json, subprocess, sys
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
joined = json.loads(subprocess.check_output([
    "git", "show",
    "fd010b111e95411702be51adc73e56b0bba6c093:docs/research/d_pr_df_ld7_price_results_20261116/C_HANDOFF.json"
]))
```

この読取は保存JSONだけを開き、fit/価格取得/Q3は実行しない。費用・誤差の分子/分母をfloatへ丸めずに扱う。
