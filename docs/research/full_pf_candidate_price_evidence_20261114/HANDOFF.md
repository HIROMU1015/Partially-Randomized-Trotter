# A/B/D向け：Full PF3候補の価格証拠

正本は [COVERAGE.json](COVERAGE.json) / [COVERAGE.csv](COVERAGE.csv)、解釈は [REPORT](REPORT.md)、未承認の有限案は [MINIMAL_VALIDATION_PLAN](MINIMAL_VALIDATION_PLAN.md)。AのSOURCEは `3ae055fa7a696b13320e00ec79a5ba15c4ed182e`、D direct catalogは `e7dca5f7301c8633b17a59b6a7161331303aef2d`。旧C3ae2の暫定6 recipeではなく、Aの現3候補を使用した。

|候補|configuration ID|保存direct metadata|不足|
|---|---|---|---|
|S2/.2|`9e96ed75f7f10b5f993ba9e2841684e52ad97b7d4e9df07b7b0070f02c852981`|m3 q4・m4 q8、XY|他11round|
|Y4/.4|`99e539037a07452f2f5a4029549980c82ec72d1b832cfcbcff59313ee3b0ae0e`|m3 q2・m4 q4、XY|他11round|
|M5/.8|`24286ac55e9b96a58fa4f11ae4fe5e6c4f9f9ca0c6a9cf24f501dc146db810e2`|m3 q1、XY|m4 catalog内欠測＋他11round|

coverageは値なしの受動metadata。`direct_evidence` に元carrier、receipt/fingerprint/SOURCE/preparation hash又はtrial locator、軸を保持する。`identity_and_law_source` のconstructorから既存C.configure/B生成を使えば元のcanonical identity/law_bindingを再取得できる。全39roundは今回再生成してAと照合済み。新price recordやmodel値は作成していない。

## Dへの最小依頼案

同native5roundの受入を前提に、新規はM5 m4 q2校正と、S2 m5 q16・Y4 m5 q8・M5 m5 q4の独立holdout。4round×XY＝最大8 wrapper。固定compile seedの決定論価格であり、PR trajectory sampling/cohort計画は不要。詳細SOURCE/役割/STOPは計画参照。初期案の成功からearly幅・長q・全Gを資格化しない。

Dが既に確認済みの元native SOURCE・scalar・full-wrapper receipt証拠を再利用する。別loweringを選ぶ場合は同compilerでも別domainであり、旧5roundを新校正値としてcopyしない。数値受入や新compileを今回開始しない。

## C/Bへの接続条件

既存 `rpe_mean_price_model.context/PriceProvider` と `I.price_context`、Bのmeasurement/cost/selectをそのまま使える。実際のrecord化は科学受入scopeが許可された後に行う。profile_extensionには元native policyとそのSOURCEを明記し、configuration/round/axis/identity/law_binding、full physical context、scalar/入力SOURCE、DIRECT/MODEL、quantity=`FULL_WRAPPER_MEAN_COMPILED_PRICE`、P0を保持する。

Full PF fixed compiler価格はDETERMINISTIC_FULL_WRAPPER。旧cohort名n1はprovenanceであり、PR有限cohort meanや全law推定ではない。normalization Bを価格へ乗算しない。MODELは検証前なら `qualified_for_candidate=false`、未取得値とmodel系統誤差はnullを保持する。S/E[S]/局所価格で欠測を埋めない。

Aのprofileにlowering_idはないため、現時点のcoverageは `SAVED_NATIVE_DIRECT_AVAILABLE_CONSUMER_LOWERING_NOT_ADOPTED` とした。Dの既存EXACTを取消した意味ではない。Bがどのloweringで費用を比べるか明示し、異なるloweringを同一価格として合成しない。

Bには候補別の適格N・比較margin/regret・必要price precisionが必要。S2のpiece数最少を価格最少とせず、Y4/M5も残す。Aの局所spectral budget screenを全RPE/N資格へ移送しない。N未知ではG/winnerはnull。

## 追加幅と保全

Aは将来も同constructorと全roundのexact actual幅・q・physical ID・科学SOURCEを供給する。要求幅やformula名だけで照合せず、共有する物理回路条件と軸の一致でreceipt再利用を判断する。元IDと新consumer IDを両方残す。旧catalogへ幅を丸めない。

production module変更0、保存価格の算術/再集計/fit/予測/compile/G0。A/B/D/監査のbranch/worktree/private rawに介入していない。原本・grant・ledger・marker・KEEP・STOPを保全。公開・remote照合後mandatory STOP。次はユーザーのlowering/科学scope判断、許可後にD/C/Bが各範囲を実行する。
