# B V2対応と長q・大R平均compiled価格：次段階の設計（2026-11-13）

**B `PR_RPE_RMSE_V2`の既存adapterを再利用するC側受動入口を用意し、固定Bとの人工結合12件がPASSした。実H4/H6の平均S/平均compiled Cは今回取得していない。** 118候補/236軸は旧round別Rの保存metadataとして整理し、新しい共通κ scheduleへの資格はすべて未取得のまま維持した。

実装SOURCE `58382c54c`（full SHAはCHECKS）、基点`a11efd130e9dfe7086b88386570594ebe54d46af`、既存C branchを継続。[HANDOFF](HANDOFF.md)／[PRICE_COVERAGE](PRICE_COVERAGE.json)／[方式比較](METHODS.md)／[次の科学案](SCIENCE_PLAN.json)／[SOURCE_MAP](SOURCE_MAP.md)／[TESTS](TESTS.txt)／[CHECKS](CHECKS.json)。新S/C/保存数値再解析・sampling/build/compile/fit・実Gは0。

## A. 現在すぐに技術的に実現できること

[`V2MeanPriceLookup`](../../../src/trottertracks/bv2_mean_price.py)は、B固定`6a9d7fd36b687e58aecbed1278c6d7927d6827d6`の`pr_rpe_rmse_integration.MeanPriceLookup`を呼ぶ。そこから既存C MeanPriceProviderへ渡すため、controller、candidate管理、会計、mean kernelを再実装していない。旧provider/stream/kernel/C1係数は不変。

| 項目 | B V2 → Cの対応・制約 |
|---|---|
| schema/configuration | `PR_RPE_RMSE_V2`。全spec/0..M scheduleがconfiguration IDに入り、各rowはRMSE candidate IDを持つ。旧B IDを再利用しない |
| 元physical candidate | Bの`physical_candidate`と`source_candidate_id/physical_method`を保つ。Bのlookup view変換を再利用し、旧C1 recordを新IDへ付け替えない |
| H/state/snapshot | 全identity内のbinding/decomposition/contract/sourceを照合。PFは元target Hとretained H、FULL_H/TRUNCATED_Hを明示 |
| 時間/回路 | T、実PF幅、qPF、R_outer、K、公式/分割、signed weights/整数allocation/event_count。H6 regularは11piece、初期prefix5へ自動適用しない |
| 共通κ/M | Bの全schedule identity/law_bindingをecho。κやMが変わると同じ時刻のrowも別identity。CはN・κ・Mを算定しない |
| law/SOURCE | 元宣言とprice samplerを保持し、B law_binding完全一致を要求。価格calibration SOURCEはidentityのlaw SOURCEとは別の出典としてcopyする |
| compiler/control/unit | 元compiler/profile、control、X/Y→cosine/sine、measure付きfull wrapper/P0、rz_count gate_countを保つ |
| quantity | 新C入口の値ありrecordは`FULL_WRAPPER_MEAN_COMPILED_PRICE`明示を必須とする。S0a/path S/local residualを拒否。値なしrecordのquantityは要求する対象量を示すだけ |
| cohort/期待値 | DIRECT finite cohortとMODELを保持し、pool/平均再計算しない。複数cohortは明示選択が必要。有限cohortは既定のqualified_expectationへ昇格しない |
| 不確かさ/資格 | MODEL falseを維持しtrueへの昇格を拒否。systematic/joint covariance/CIの未知null、欠測/OUTSIDE_DOMAIN/FAILEDを保つ |

Bは既にquantity検査を持ち、legacy compiled recordでのquantity省略は既存仕様として残している。**今回追加したC入口だけが、新しい値ありrecordに明示quantityを要求する。** 旧recordへの自動ラベル付けはしない。B共通API/既存adapterは変更せず、SOURCEやcohortの科学的適合を宣言だけで証明する仕組みも追加していない。

C入口で接続するのは現BのA-source-bound PR view。Full PF/truncated PFには既存の決定論価格providerが必要でOUTSIDE_DOMAIN/null、Full-H RTEには全H preparation/law/event_count bridgeが必要でOUTSIDE_DOMAIN/null、未接続native PRもnullを返す。新price・model・calibration・Gへfallbackしない。

## 保存価格coverage：存在と新schedule資格を分離

[PRICE_COVERAGE](PRICE_COVERAGE.json)は固定B COVERAGEのmetadata projection。候補recipe/structure/contextと236軸のdirect/model/cohort/SOURCE/資格だけを持ち、**S/C/Nの科学値は転記・再計算しない**。旧M12とround別Rは履歴であり、新RMSEのM/κを採択していない。

| 方式 | B保存menuの候補/軸 | DIRECT存在の候補/軸 | 保存MODEL存在の候補/軸 | 現C入口・不足 |
|---|---:|---:|---:|---|
| H4 truncated PF B0/p7 | 13/26 | 3/6 | 3/6 | 決定論PF経路へ。q8/16/32・δ.1に保存価格。Full PFとは別、元Hへの切捨て/RMSE資格が必要 |
| H6 Full PF Full19 | 13/26 | 4/8 | 4/8 | 決定論PF経路へ。m3(q1/δ.8)、m4..6(q1/2/4・δ1.6)に保存価格。別δをAffineへ移さない |
| H4/H6 PR | 92/184 | 10/20 | 10/20 | 現CはH4 S2/δ.1/p1/K2・H6 m5/δ1.6/p7/K2のみ。長q/新Rの多くは欠測、MODEL資格false |
| H4 Full PF/H6 truncated PF | route-levelのみ | 新V2対応候補未固定 | 未固定 | 過去PF build/compilerは存在。対象H/retained H・公式/幅・metadata対応が必要 |
| Full-H RTE | route-levelのみ | 別の固定Tに有限cohort資産あり | 新V2へのモデルなし | 全Hのone-bodyもrandomizeするSOURCEを再利用する別bridgeが必要。DF B3は不可 |

候補/軸数は価格値の集計ではなく保存metadataの個数。34軸にDIRECTの存在情報があるが、共通κ scheduleの適格mean coverageを意味しない。236軸すべて新RMSE比較資格は未取得、現共通κ V2 identityは未供給、G/winnerはnull。

**H6初期の区別**：m0..2/prefix5/q1/R32/δ=Tは価格missing。m3/prefix5/δ.8にはV1 EXPLORATION/CONFIRMATIONの有限cohort価格がある。一方、現C mean-provider domainはregular prefix7/δ1.6なのでm0..3をOUTSIDE_DOMAINとする。保存m3が存在することとprovider接続・全law資格は別であり、δ.8の値をδ.1/.2/.4へ流用しない。

**C1の8セル**：H4 q8/16×R4/8、H6 q1/2×R16/32。各系3calibration/1validation、元n/cohort/roleを[metadata](PRICE_COVERAGE.json)へcopyした。すべて有限cohortのみで、C1 MODEL falseを保持。既存paired mean-price covarianceの所在は残すが、その値を再解析せず全schedule共分散へ昇格しない。

旧qモデルも別domainで保持する。H4 Mqは固定R4のPR q16とB0/p7のPF q32までの保存独立検証、H6 Affineは固定R16/δ1.6/p7のq1/2校正・q4独立検証。C1のR32や長q/大Rへ係数を移植しない。

## B. 新しい科学計算で確認する必要があること

三方式の比較と境界条件は[METHODS](METHODS.md)に記載した。今回のSOURCE調査で、物理C1の実用的な小状態閉包を証明する十分条件は得られなかった。

`extract_identity_phase`ではrandomized componentsからidentityを除くため、以前の人工identity-angle反例をそのまま物理supportの到達例にできない、というSOURCE上の制約を確認した。一方、pending Zのliteral zeroとbasis相殺、合法tail内の逐次binary64 π位相和、fixed scalar/wrapperとの境界を含む**価格observableの状態同値類**は未定義である。固定個数のfloat registerやS0a8/8だけから小さい閉状態を結論しない。

- 閉状態方式：合法support・遷移/reward同値性・uniform flush条件・signed lawとsource順の根拠が必要。
- 局所方式：現在のcontrolled full-wrapperに対する平均S近似とpaired D=C−Sの校正、PF/RTE・piece・outer・wrapperの境界、打切り/非局所compiler残差、独立holdoutが必要。旧raw κ(C)をSへ足さず、C1 affineと局所残差を重ねない。
- Streaming標本平均：Sの平均とcompiled Cの平均は別。保存同一path C/Sを優先するが、有限cohortだけで全law期待値を認定しない。未知のrare寄与・law距離・共有不確かさは0にしない。

## C. 次に優先する案と理由

**最初の条件は、A/Bの実RMSE pilotと同じ1系・外側条件・固定(M,共通κ)と、価格が必要なphysical domainをユーザーが選ぶこと。** 今回そこを任意の数値で埋めていない。A最新ee15のpilot/preflightは旧B f09 pinを参照しており、新V2への追従はA/B担当。CはRMSE評価器や誤差契約を追加しない。

その後の第一科学候補は、**1つの選択済みphysical preparation/law domainについて、有限supportとflush境界から平均S方式の成立性を確認する限定診断**。新compiled fitを先に増やす前に、E[S]をどう得るかを決めるためである。1 preparation descriptorと当該domainの元J signed-piece law/word templateを対象にし、新path sampling/S再計数/compileは0の案。実prep/law metadataの新生成・数学的仮説の検証なので、SOURCE/対象domainを固定した別の実行許可が必要で、今回は実行していない。全supportの証拠が作れなければUNKNOWNでSTOPし、人工状態で補完しない。

閉包が証明できなければ局所近似が現実的な効率候補になるが、正式採用はユーザー判断。Streamingは短い保存点での参照器として保持する。平均modelのSOURCEがまだない段階では、新しいmean比較を開始しない。

保存C1のdevelopment候補としては各系の新R2セル（H4 q8/q16・R8、又はH6 q1/q2・R32、各保存n8）がある。ただしBの必要domainと異なる場合は移送証拠にならない。モデル定義後に既存summaryを再利用し、新model meanの比較だけを別許可範囲へ絞る。既存C1の残差/mean/covariance統計を同じ目的で再計算しない。旧validationも新modelには既知developmentで、fresh holdoutは別途必要。

[SCIENCE_PLAN](SCIENCE_PLAN.json)に4候補の目的・対象selector・必要SOURCE/input・追加callの種類・校正/holdout・STOPを記載した。M/κ/候補ID/必要精度/新cohort数が未固定の案は**実行準備完了ではない**。旧40/16、fresh32件等は採用しない。

## D. Bへ適格な平均compiled価格を供給するまでの条件

1. 全schedule/M/κ/configurationと元physical candidate、H/state/target-retained H、law/λ/normalization、signed allocation、compiler/control/軸/wrapper/P0が対応すること。
2. 求める価格lawをN側と一致させ、finite cohort・ideal model・実samplerの差を明示すること。mean Sだけではmean compiled Cにならない。
3. 平均Sの正確性又は近似誤差と、full-wrapper C−S残差の校正/独立検証・長q/R domainを別に資格すること。q-only/R-only/joint移送を混同しない。
4. 未観測order2等、モデルsystematic、paired X/Y、共有calibration/cross-round covariance・total CIを扱う根拠を用意すること。nと量子shot N、Bを価格へ再乗算しない。
5. 必要価格精度をユーザー/A/Bが決めること。固定Nなら形式的に |δG|≤ΣN|δC|と結べるが、今回はN/Gの実数値も精度閾値も計算していない。現在のC1 falseを技術接続PASSだけで変えない。 現C入口はqualified MODELの受入をまだ実装しておらず、科学証拠が揃った後も採用する新SOURCEを別途整備する必要がある。

## 技術検証・保全・STOP

新12件は固定B controller/integration・最新Cを同じtest processへ読み、identity/law/source/unit、4方式の欠測、H6早期/11piece、cohort選択、M/κ分離、MODEL false、S0a拒否、旧ID付替え禁止、236軸metadataを確認した。Bのassessは人工recordで呼んだが、cost/select・A.evaluate・C mean/S/predictを例外化し、科学入口が起動しないことを確認した。実N/Gや人工Gも今回計算していない。

最初のtest loaderはBで更新されていたexact-scalar helperを現C版と同一と仮定して失敗した。B固定helper/pipelineをtest process内へ読む形に直し、旧C module object/実ファイルを変更せずに12件PASS。これは作者自身の技術確認で独立科学レビューではない。旧科学検証を繰り返す回帰は行わず、前段のkernel/stream/C1検証記録を参照した。

C1/C2/connected-cluster、S0a run01失敗/run02結果、原SOURCE/input/raw/cohort/grant/ledger/marker/KEEP/STOP、A/Bのworktree/process/cacheを保全した。必要成果をcommit/pushしremote SHA/主要成果物照合後mandatory STOP。次はユーザーがphysical domainと検証方式/有限scopeを判断、A/BがV2・RMSE条件、Cがそのdomainの価格検証を担当する。新しい科学計算には進まない。
