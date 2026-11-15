# C・Bへ：粗幅Yoshida4の固定配分Q3復旧結果

**取得・保存は完了。粗幅32h/5はこの固定配分で `MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD`（下界>元exact τ）。UNKNOWNではない。** 科学API1件、追加信号/reference/build/compile/価格/fit/別配分0、粗幅総費用はnull。旧q3_run01/UNKNOWNを保全している。

X/Y各軸 `[59,55,51,47,43,39,35,31,27,23,19,7,2]`。

| PF要求幅 | MSE下界 Ha²（表示） | MSE上界 Ha²（表示） | 判定 |
|---|---|---|---|
| 現幅16h/3 | 2.418371981954278179229700e−6 | 2.418371981954278179229706e−6 | 保存モデル適合・b4296b110引用 |
| 粗幅32h/5 | 3.409058131550481301103526e−6 | 3.409058131550481301103539e−6 | **この配分でモデル不適合** |

exact τの表示は2.539561285063813282041692e−6 Ha²。判定は元exact有理数で行い、端点を丸めて判定していない。粗幅全体の不可能性、shot増での救済、最小必要N、方式winnerは未判定。

## Cが結合する固定SOURCE・locator

元configuration：**`dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0`**。
total view：`97ffe0a66f2dc14fa4b3264e885428216a2716c7a8a288085ddfe3b8a0e622ec`。
binding：`ef7912d2ede9f55de01c4bcf91c5c3adb935874370aaed84a9c6d48b7e9445ba`。

保存入力は次の正本をそのまま再利用した。

- configuration：`3179680ed9b4cff251b3401cabea2f6100cf47fb:docs/research/d_p7_y4_coarse_signal_q3_20261116/SIGNAL_INPUTS.json#configuration`。
- 信号：`f23c3f10ed56d0c218e49efbdbfb31d12ffa8a53:docs/research/d_p7_y4_coarse_signal_q3_20261116/BOUND_SIGNAL_INPUT.json`。
- 固定配分・契約：`278fb5214f1091efba2a05b305bc05190cd18ff6:docs/research/d_p7_y4_coarse_signal_q3_20261116/Q3_INPUTS.json` のcases[0]、statistical_contract/options/adopted_evaluation。

**新しいQ3結果の固定commitは `537df23766e962fad027bf98ba7edf3c6854d927`**。その下で次を読む。

~~~text
docs/research/d_p7_y4_coarse_q3_recovery_20261116/run01/
  B0I_P7_Y4_32h_over_5/RESULT.json
  B0I_P7_Y4_32h_over_5/API_RETURN.json
  B0I_P7_Y4_32h_over_5/PARTIAL_COARSE.json
  B0I_P7_Y4_32h_over_5/PARTIAL_LEMMA.json
  B0I_P7_Y4_32h_over_5/PARTIAL_TOTAL_DP.json
  B0I_P7_Y4_32h_over_5/RESOURCE_FINAL.json
  COMPLETION.json
~~~

API_RETURN[0]はRESULT.recordと完全一致。exact MSE端点はRESULT.record.diagnostic.deterministic_risk.MSE_lower/MSE_upper、個別判定はRESULT.status／adopted_q3_assessment。元τは凍結INPUTSのinputs.adopted_evaluation.MSE_upper。処理正常はCOMPLETION.status=`ONE_FIXED_Q3_SAVED`で、モデル判定の適合とは別の欄である。

現幅の固定引用は `b4296b110b129d226dbef2743c14f725a27c3bf6:docs/research/pf6_cost_budget_q3_results_20261115/run01/B0I_P7_Y4_16h_over_3__B_PRICE/RESULT.json`。同じN列の再Q3は行っていない。

**Cが自動取込みする結合表の固定locator：`8a2f5c46d201b26bdaf2aad7ca556e31083779aa:docs/research/d_p7_y4_coarse_q3_recovery_20261116/C_RESULT_JOIN.json`。** 同commitの `SUMMARY.json` に比較のexact端点を保存した。

本ディレクトリ [C_RESULT_JOIN.json](C_RESULT_JOIN.json) はconfiguration/source、元/total候補ID、physical ID、m/axis/Nの26行、元配分source、exact MSE/τ/判定と上記固定result/API locatorを持つ。Cは同ファイルのresult_source/API_sourceから固定rawへ戻れる。[SUMMARY](SUMMARY.json) に両幅のexact端点・表示値・runtime・資源・gateを収めた。compiled単価やGは追加していない。

## 科学条件・runtime・gate

保存H4 linear1Å/STO-3G/tol-only DF10、B0I p7/標準Y4・32h/5、h_RPE=.1、P0除外。13roundのscalar一回・signed/fragment順・raw=corrected/B=1等は保存packetのまま。元Z1＋T1、DF10 solver proxy、IID deterministic raw、全事象loss、exact τを維持した。

実行SOURCE：**`5e2246d1233fbeebc709cc9f20a3f48e2fb3a1ad`**。実行ファイル `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python`、実体は同project内Python3.11.0rc1。元kernel710aa69b、accelerator9e294778、AST guardの期待/実hash3d043e71…が一致。SOURCE正規化・期待hash変更・guard無効化なし。

本番PF worker入口から実コンストラクタ・人工DP・API/partials/RESULT/RESOURCE保存まで検査し、元exact-DPとの同値性を含む5 testがPASS。実callでも保存粗幅signalのbinary64閉包PASS（2,551,245 pairs）、DP85,523,546遷移、残余probability約3.49e−31とloss上界約1.30e−29を残した。API1件、retry0、取得/保存UNKNOWN0、98記録child終了entryは全0。8-worker既存並列DP、監督wall134.798212 s、worker peak596,748 KiB、観測tree最大3,634,356 KiB。詳細REPORT/runtime/RESOURCE参照。

## 費用・ロードマップ・STOP

粗幅総費用は **null/未評価**。同公式現幅435,641,568 RZ、計画時点p7最安435,481,090 RZは保存参照に留め、この不適合点を達成費用更新とは扱わない。旧h規則・現幅係数の粗幅流用なし。

統合担当への差分：**粗幅Y4の指定列は、環境guardによるUNKNOWNから、有効なQ3モデル不適合へ解決した。同じ列の現幅適合は不変。追加の適合点・価格資格・費用順位はない。** ロードマップ本体は統合担当に任せる。

旧SOURCE・入力・13round原結果・UNKNOWN/STOPは不変。commit/push・remote照合後 **mandatory STOP**。別配分・shot増・追加compile・価格fit・N/G/winnerへ自動進行しない。
