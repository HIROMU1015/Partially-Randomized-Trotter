# H8希少RTE検証：正式採択・現在の主張範囲で終了

2026-10-21。**ユーザーは、H8希少RTE検証を層別の有限標本資源評価と残余費用の条件付き感度解析の範囲で正式採択し、現在の主張範囲で終了する方針を決定した。** 追加sampling・compileは現時点で不要とし、S2_MULTI/S4_PRESENTの追加取得を開始しない。既存のmandatory STOPを維持する。

今回は研究方針の文書反映のみ。新規科学計算、保存値再解析、sampling、simulation、signal計算、回路生成、compile、詳細GPT/Claude科学レビュー、最終evidence freezeは行っていない。PR資源評価研究全体や、4次PFなど並行する別研究の終了・進捗変更を意味しない。

## 論文で採用する証拠

線形H8、1 Å、STO-3G、tol-only 1e-8 DF rank28、T=.8、P=0、通常signed bias、allowance1。二次partial PF、r=4、K=4、splitと時間幅δ=T/qは下表の固定候補に従う。元input・preparation・analytic shots・native DF/full Hadamard wrapper・Qiskit1.3.0/rz,sx,x,cx/opt1/seed17の保存compiled RZ会計を維持する。

|ε|固定PR候補|S0独立確認|S2_SINGLE条件付きpilot|残余混合平均の逆転境界／S0平均|従来未観測だった質量のうち費用推定を付与した割合|
|---|---|---:|---:|---:|---:|
|.05|B2_p8_q2_2nd_r4_K4|48|43|約1,276倍|約99.67%|
|.0005|B2_p7_q16_2nd_r4_K4|48|43|約888倍|約99.56%|

数値は[固定された最終結果報告](h8_rare_rte_saved_integration_results_20261021.md)からの転記で、再計算していない。採用内容はS0とS2_SINGLEの層別の有限標本資源評価、および残るS2_MULTI/S4_PRESENTの費用感度解析である。異なるcohortの標本行をpoolせず、同じ無条件期待費用に対する層別寄与として結合した保存解析を採用する。

## 条件付き境界と主張の限界

`K_known_hat=p_S0*mean_hat_S0+p_SINGLE*mean_hat_SINGLE` とし、残余寄与は `p_MULTI*mu_MULTI+p_S4*mu_S4` と表す。上表の逆転境界は、**既知層の経験平均を固定したとき**の `p_MULTI*mu_MULTI+p_S4*mu_S4=G_PF-K_known_hat` を残余2層の確率加重平均費用で表し、S0の経験平均で割った保存値である。2層同費用の仮定や、各層単独の軸切片ではない。母平均の保証ではない。

約99.67%／99.56%は、段階Bで未観測だった希少3層の確率質量のうち、S2_SINGLE取得で経験的費用推定を付与した部分の割合である。全期待費用の誤差・分散が同じ割合で減ったという主張ではない。

次の主張は採用しない。

- PRの無条件期待費用優位の数学的保証。
- 全supportの厳密な費用上界、母平均順位の保証。
- 未観測S2_MULTI/S4_PRESENTの費用・分散が既知であるとの扱い、0補完。

標本SEとその合成は層内iid・有限分散・cohort間独立性等の仮定に条件付けられ、±kSE感度は信頼区間や厳密coverageではない。保存最大値やpretranspile gate countを全supportのcompiled RZ上界とみなさない。C-1の保存binary64整数比samplingと段階Bのcanonical lawとの1e-12 engineering gateのPASSも、期待費用biasの厳密保証に格上げしない。これらの限界は正式採択後も維持する。

## 終了・再開の条件

- 現在の論文の主張範囲に対するH8希少RTEの追加sampling・compileは現時点で不要。
- S2_MULTI/S4_PRESENTの追加取得、production、自動的な次段階移行を開始しない。
- 既存のmandatory STOPを維持し、旧SOURCE・raw結果・grant・ledger・marker・STOP・各科学契約を変更しない。
- 今後、論文の主張を強める場合に限り、別途科学判断と明示的な実行承認を行う。この終了方針は新しい科学実行承認を含まない。
- 4次PFなど並行する別研究の進行状態は変更しない。

統合報告・旧索引・研究ノートに残る「主張範囲をユーザーが判断」「追加実測を選ぶ場合の候補」は採択前の時点記録・条件付き計画として保全する。**現在の研究方針は本採択記録と、概要・進捗文書の本日付追記を優先する。** 既存数値報告の内容や科学的保証範囲を書き換えるものではない。

## 固定資料・監査への対応

|段階|固定結果commit|科学SOURCE|結果・監査|
|---|---|---|---|
|段階B|3d5ab5bb1cf0d78a3157ad7f742cfa5c7b58efbb|e16c3be8a039f6bc468e112f0ed086bde544fef2|[結果](pr_rare_rte_stage_b_results_20261021.md)、[既存監査](../../artifacts/resource_applicability/pr_rare_rte_stage_b/2026-10-21/completion_audit.json)|
|C-1 pilot|f377f8869c697c4846e3614b0ae7bc9d3ad5cb3c|383dd9068504b8f578944984fc45a60dd4ed18b2|[結果](h8_rare_rte_c1_pilot_repair_20261021.md)、[既存監査](../../artifacts/resource_applicability/h8_rare_rte_c1_pilot_repair/2026-10-21/completion_audit.json)|
|保存値統合|145b438c05b1888ab479081ef72462b93429f89a|1e1e1c82a09ab656d9dc6ca71ef0b818e04fae87|[最終結果](h8_rare_rte_saved_integration_results_20261021.md)、[既存監査](../../artifacts/resource_applicability/h8_rare_rte_saved_integration/2026-10-21/completion_audit.json)|

今回の文書採択・リンク/SHA/保全検査は[採択記録packet](../../artifacts/resource_applicability/h8_rare_rte_closure_policy/2026-10-21/README.md)を参照。既存の数値検証receiptを確認し、科学driver・数値検証関数を実行していない。

次の科学作業なし。**Codexはmandatory STOPを維持する。** 公開を希望する場合の次担当は**ユーザー**：公開範囲を別途承認する。GitHub pushは今回実施しない。
