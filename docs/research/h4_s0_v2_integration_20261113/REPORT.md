# H4 DF10 S0固定条件のcanonical V2統合と13round不足証拠

2026-11-13。B`af284d853890155bacd19e61850c196bd1d5c660`の既存branchを継続。**A S0の固定inputsからV2を新規生成し、13roundすべての物理metadata一致を確認した。十分Nは26軸すべて未取得、global RMSE boundはnull、対応する適格Cは26軸すべて欠測。実G/winnerはnull。新科学計算・保存数値再解析・compile/fit/実Gは0。**

固定A：`7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75`、科学SOURCE`7c3e1e5db0c55dd8249e9030a4c7e9954496fa7e`。固定C：`1db4d6e7ebce872c069e337b76632046f60d8b56`。既存controller/integration/plannerを変更せず、S0専用metadata接続とCの受動入口を再利用した。

## V1/V2の照合

V1 IDは`6cc89360a60b6f5b6554d044193a28f4c92cafa0208e77efb4d1f2ce9f80181c`、新V2 IDは`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。

[V2正本](run01/CONFIGURATION_V2.json)は既存`B.generate(I.a_spec(S0.inputs,'PR'))`で作成した。[対応表](run01/IDENTITY_MAP.json)は元physical ID・A request ID・各V1/V2 candidate IDを分離して保持する。V1のpre-identity hashを既存_id規則で検査し、元Nのfull identity/lawとV1全specを照合したうえで、A.scheduleの全metadataとV2を比較した。

許容したmetadata差分はschema/ID体系と、PR splitの決定論側を`A_SOURCE_BOUND_PF_BASELINE`から`PREFIX`へ明示する点だけ。元physical candidate、M/κ、T/実Δ/q/R/event、λ/SOURCE、split/finite-K、H/state/snapshot、prior/decoder、normalization/error policy、compiler/control/wrapper/P0は保存条件と一致した。不一致を自動補正する処理はない。

κはA CONFIGURATIONのexact有理数を使用し、近似小数から再生成しない。S2/L_D1/ΔPF.1/K2/h.1/M12、εE=.00159360010199040 Ha、εqpe=εtrot=εE/2、minkowski、finite_B_diagnosticを保全した。H4 linear1Å/STO-3G/tol-only DF10・元保存state/snapshot、無shift、symmetric-directional/full Hadamard measure/P0が対象。S0の設計条件が固定されたことと、RMSEモデルの科学的資格は別で、資格はUNQUALIFIEDのまま。

**SOURCEの版差**：A modelはS0消費SOURCEとbyte一致。pipeline全体は現在BにH6保存価格factoryが追加されたため異なっていた。元科学版のhashを確認し、その未使用H6 factoryを除くmodule ASTが同一であることを検査した。H4の処理・元SOURCE・既存BのH6機能を変更していない。元科学実行のSOURCE lock/STOPを変更する措置ではない。

## 13roundの不足証拠

下表のnominalはAが保存した整数の転記であり、BによるN再評価ではない。旧HP Nの存在もmetadataのみ。**全行・両軸の十分Nはnull、Cは欠測、global boundはnull**。

| m | qPF | R_outer | event位置数 | S0 nominal X/Y | 同physicalの旧HP N | 対応保存C |
|---:|---:|---:|---:|---|---|---|
| 0 | 1 | 2 | 2 | 64/64・未資格 | あり・RMSE不採用 | なし |
| 1 | 2 | 4 | 8 | 未評価 | あり・RMSE不採用 | なし |
| 2 | 4 | 8 | 32 | 未評価 | なし | なし |
| 3 | 8 | 16 | 128 | 未評価 | なし | なし |
| 4 | 16 | 32 | 512 | 未評価 | なし | なし |
| 5 | 32 | 64 | 2048 | 未評価 | なし | なし |
| 6 | 64 | 128 | 8192 | 38/38・未資格 | あり・RMSE不採用 | なし |
| 7 | 128 | 255 | 32640 | 未評価 | 対応する旧menuなし | なし |
| 8 | 256 | 510 | 130560 | 未評価 | 対応する旧menuなし | なし |
| 9 | 512 | 1020 | 522240 | 未評価 | 対応する旧menuなし | なし |
| 10 | 1024 | 2040 | 2088960 | 未評価 | 対応する旧menuなし | なし |
| 11 | 2048 | 4080 | 8355840 | 未評価 | 対応する旧menuなし | なし |
| 12 | 4096 | 8160 | 33423360 | 未評価 | 対応する旧menuなし | なし |

保存範囲は固定B6aの旧catalogとC1dbのPRICE_COVERAGE。これらでm0..6のphysical recipeは一致するが、旧round別R menuと新共通κ全scheduleを同一視しない。m7以降の候補をR256/512/.../8192等へ丸めたり置換しない。旧HP N、保存normalization診断、S0 nominalはそれぞれ独立欄に保全し、十分Nへ代用しない。

C1のH4 q8/q16セルはR4/8で、今回のq8/R16・q16/R32とは異なる。元finite cohort・MODEL false・SOURCE/roleを保持する。Cの新`V2MeanPriceLookup`を供給recordなしで照会した26軸もすべてMISSING/null。R4のm1にも対応保存値はなく、固定Rモデルや小q価格を新R/長qへ外挿しない。これは対象catalog内の不存在と新V2での未取得の報告で、未確認private原本まで探索したという主張ではない。

## 元S0とplannerの保全

[元N_HANDOFF](run01/ORIGINAL_S0_N_HANDOFF.json)は公開原本とbyte一致。m0/m6のnominal以外は未評価、十分N全null、RMSE bound=null、model/global/finite-K/signed資格UNQUALIFIED、SOURCE/statusは不変。

[EVIDENCE_PLAN](run01/EVIDENCE_PLAN.json)は既存`pr_rpe_rmse_evidence.plan`のPR1構成/26軸に、source付きS0診断と過去catalogの存在・不一致理由を付加した。Aの新V2 packetはBでは生成していない。plannerに渡すN_packetと値あり価格recordはゼロで、null placeholderを取得済みとしない。S0のV1 packetは診断refとしてだけ接続し、各セルのN/C/全体RMSE不足とA/C next_workを出す。

4方式区分は維持し、他3方式はpresent=false。H4 B0はtruncated、DF B3は全H純RTEではない。保存A helperがPF metadataも照合するが、新たなPF実configurationをmenuへ追加していない。PF formula/幅の正式比較menu・純RTE bridgeは未完成で、今回のPR-only planを4方式比較や実最適化完了とは表示しない。

## 実装・技術確認

新`h4_s0_v2_intake`は固定metadataのcanonical化・診断view・保存catalog照合・将来A V2診断packetの検査のみ。Cの`bv2_mean_price.py`をbyte一致で再利用した。AのRMSE/N、Cのmean/構造計数/価格model、既存B controller/plannerは再実装していない。[SOURCE_AND_SCOPE](run01/SOURCE_AND_SCOPE.json)に固定ref/消費元を記録した。

新10＋C入口12/planner13/4方式14/既存統合12＝**61技術test PASS**。[TESTS](TESTS.json)／各log。保存fixtureの全13round一致、nominal/null/未資格、SOURCE/identity/law不一致、C1 false/finite cohort、4方式、科学/N/mean/compile/G非起動を確認した。C原testはBにないstream helperと未配置inventory scriptを仮定して初回2環境エラーとなったため、testだけをB環境へ適応した。欠けたstreamにはtest中だけ例外sentinelを置き、inventory helperは固定Gitから読む。C production SOURCEや価格意味は変更していない。作者自身の技術確認で、独立科学レビューではない。

新signal/N/RMSE数値再解析、normalization評価、H/state配列読込、実C/S、sampling/build/compile/transpile、fit、実G積和・M/κ最適化は0。SOURCE/input/raw/結果/grant/ledger/marker/KEEP/既存STOPを保全した。

## 次に必要な担当作業

**A**：まず既存二点の診断をcanonical V2 packetへ技術追従。その後の別許可範囲で、残11roundのnominal診断、全13roundの科学的十分Nとglobal RMSE/decoder/state/law条件を取得・資格する。保存mean/referenceを使える場合は重複生成を省くが、旧HP Nを採用しない。

**C**：[26件のC request](run01/C_REQUESTS.json)、特にm0/q1/R2とm6/q64/R128を使い、全必要domainを覆う平均compiled Cの方法・calibration/独立検証・law/長q/大R/不確かさを有限計画にする。26軸不足は26新compileの許可又は必要数ではない。Cが方法と上限を提案し、Bは価格を補完しない。

**ユーザー**：残る科学的RMSE資格、価格policyと各有限実行scopeを判断・承認する。S0の既存許可を他round/価格/実Gへ拡張しない。**B**は必要証拠と実G scopeの明示許可が揃った後だけ既存cost/selectへ進める。詳細は[HANDOFF](HANDOFF.md)。今回はcommit/push・remote SHA/主要成果物照合後**mandatory STOP**。
