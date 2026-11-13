# D compiled価格adapterとRPE pipelineの技術統合

2026-11-13。**既存Cの価格入口を公開Dの `from_a_job/from_b/pending_price_records` へ接続し、D path結果と外部mean証拠をB会計入力へ渡すmappingを実装した。** 実H4 signal/preparation/sampling/build/compile、保存科学値の再解析、fit/E[C]推定、実N/RMSE/G、科学的winner選定は0。

[PRICE_INTEGRATION_HANDOFF](PRICE_INTEGRATION_HANDOFF.md)／[TESTS](TESTS.txt)／[NEXT_SCIENCE_PLAN](NEXT_SCIENCE_PLAN.md)。基準Cは `2f76b5750dd4c4823c122ab04e7629f731fd03a3`、Dは `0c6e16342b3c804fa0bd89a4028d6b5bea7b0322`。既存C branch `codex/r-variable-compiled-price-design-20261111` を継続。D/A/Bのworktree・実行領域へ変更しない。

## 接続内容

|既存部品|今回の差分|
|---|---|
|`rpe_optimization_pipeline.providers`|同じprivate namespaceへDの固定4 moduleを読み、`api.Price` を追加。importは受動で、NativeRuntime/Adapterを構築しない|
|`price_requests/price_context`|既存形状を保持。input/execution SOURCEを指定するとDから各round/axisのpendingを取得し、Cのphysical contextへ対応付ける。要求軸のみ返す|
|Bのfixed-A schedule adapter|`pr_rpe_rmse_integration` の公開moduleを追加読込。Aの固定LD1/LD2 configurationを元B validatorで扱えるようにした。SOURCE/ID生成規則は変更しない|
|新 `rpe_d_price_connection.bind_results`|Dのrequestを元APIで再照合、path metadataを保存、外部mean/sidecarを同じrequest/contextへ結ぶ。統計・平均・価格計算は追加しない|
|B shot配分/cost/select|既存のmean/law/model/区間/資格検査とexact会計をそのまま使用。N/価格資格をCで生成しない|

D requestにはconfiguration/physical ID、全round identity/law、T/qPF/R/K/formula/signed weights/allocation、H/state/snapshot/split/input SOURCE、profile/control/full wrapper/measurement、lowering、実装SOURCE、元context/request keyが保持される。Cは新しいcache keyやfingerprint計算器を作らず、Dのkeyとpathに保存されたfingerprint/cache情報を維持する。

## 価格の意味と結果

1. **request**：metadataのみ。D pendingは `mean.rz_count=null`、NOT_EVALUATED、law資格null。
2. **path S/C/D**：同一request・resolved preparation・軸・SOURCE/roleの診断。mappingは値を変更・平均せず、Gへ使わない。normal/conditional cohortを保持する。
3. **finite-cohort mean**：外部から供給されたrecordのみ。期待価格policyではblockし、明示したB finite-cohort scenarioでだけ条件付き費用を扱う。全law保証にはしない。
4. **適格E[C]/区間**：外部producerが元B形式で供給した証拠とSOURCEが必要。Cはその科学的正しさを認定せず、対応/欠測を検査し、元Bが資格を判定する。

新8＋既存36、**44 test method PASS**。作者自身の技術確認で、独立科学レビューではない。

- 人工PRでD.from_a_jobとfrom_bのrequest一致、X/Y・単軸、loweringと未取得meanを確認。D NativeRuntime/Adapter入口を禁止したテストでも成立した。
- A `a9abf8de7` の公開固定LD1/LD2を数値評価せずmetadataとして読み、各13round/26軸を生成。canonical configurationは不変、context keyは別で、LD2をLD1へ渡すと拒否する。これはD担当の実preparation・head/tail domain適合性診断を代行した結果ではない。
- 人工pathのみ、片軸、forced役割、価格欠測は全G UNKNOWN。Xの値でYを補完しない。平均が外部から揃う人工例だけ元会計でG=1002/216となり、12 shot最小と18 shot/G最小を区別した。明示X/Y別価格でも別々に積算した。
- 区間重複時は点最小のみ、区間優越はnull。欠測binding、未資格/数値不安定価格、異なるphysical/law/SOURCE/compiler/lowering、混在resolved tail、構造量のmean流用、重複pathを拒否又はpendingに保った。
- 既存36回帰は無変更でPASS。provider/decoder/N/価格solverを新規実装していない。

テストは架空H/state・path価格・平均価格・Nだけを使う。QUALIFIED指定はfixture内だけ。A公開scheduleテストもmetadata照合だけで、保存確率/科学結果の新数値解析をしていない。既存Dの科学的preparationを含む人工取得testはCで重複実行していない。

## 残る不足と保全

Dの現在の対象はH4 PR、非空head/tail、S2/M5、元compiler/full wrapper/効率的ZZ。baselineや別profileはOUTSIDE_DOMAINで、科学的敗北ではない。Cで代替価格経路を実装しない。実NativeRuntime/acquireはDの完全なprovider環境と別の科学承認scopeで実行し、結果をCへ渡す。Cの最小namespaceには科学取得依存graphを一括取込みしていない。

実preparation hash/tail/law・path取得、cohortから全law平均/同時区間への資格、sampler距離/rare/model/systematic/共分散はD側の後続課題。N資格producerとmeanの物理資格も未取得。Cのmappingだけでは実Gを認定できない。旧C1/C2/E[S]、SOURCE/input/raw/結果/grant/ledger/marker/科学契約/KEEP/STOPは不変。

必要な接続は完成したため、追加開発を広げず、commit/push・remote SHA/tree/主要成果物照合後 **mandatory STOP**。新しい科学実行には進まない。
