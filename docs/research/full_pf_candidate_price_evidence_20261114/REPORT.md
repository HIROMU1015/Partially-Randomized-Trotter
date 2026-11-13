# Full PF S2/.2・Y4/.4・M5/.8：compiled価格証拠の接続

2026-11-14。Aの固定shortlistの39 physical round・78軸セルをDの既存receiptへ対応付けた。**S2/.2とY4/.4は各m3/m4、M5/.8はm3に保存directがあり、計5round・10軸。** M5 m4はcatalog内欠測、他33roundは確認したcatalog外である。価格値の再集計・モデル予測・fit・新compile・Gは0。

3候補とも比較対象として維持する。AのS2優先理由はsigned S2 piece数であり、compiled RZやGの順位ではない。本作業ではpiece数による除外も、保存価格の新しい順位付けも行っていない。

## 固定SOURCEと照合方法

|用途|SOURCE|
|---|---|
|Aの3候補・全round・actual幅・physical ID|`3ae055fa7a696b13320e00ec79a5ba15c4ed182e:docs/research/h4_full_pf_heff_scan_results_20261114/C_HANDOFF.json`|
|Aの数値SOURCE/準備|`b64f22b97c641fc97d13e9872b723f789e4fa454` / `878000d8d5d956b1d2f5513986ae40d0612e0bdd`（数値処理は呼ばない）|
|Cの旧価格coverage・検証方針|`3ae2e605dd6efcc63ed3742fe3076a8be3e6afae:docs/research/rpe_price_evidence_validation_20261113/`|
|Dの確定direct catalog/receipt trace|`e7dca5f7301c8633b17a59b6a7161331303aef2d:docs/research/d_full_pf_full_h_v2_final_receipts_20261113/`|
|Dの入力・compiler・cost provenance|`af284d853890155bacd19e61850c196bd1d5c660:artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/scientific_provenance.json`|
|再利用するC/B接続|既存 `baseline_signal_pipeline.configure/prepare`、B canonical生成、D `full_pf_spec`、C `rpe_mean_price_model.PriceProvider` とB cost/select|

[BUILD_COVERAGE.py](BUILD_COVERAGE.py) は既存Cのmetadata APIでAのconstructorを再生成し、全3 configuration ID・39 physical/canonical round・actual幅・qを照合する。Dの完成済みRECEIPTS/source_checksを参照し、Dが実施した73件の追跡・private input確認・RZ照合を繰り返さない。新しいprice providerや価格推定系は作っていない。

機械可読な [COVERAGE.json](COVERAGE.json) / [CSV](COVERAGE.csv) に全78軸のSOURCE/ID/q/T/actual幅・receipt locator・未資格を保存した。保存RZの値自体はcopyせず、C/N/Gはnull。SOURCE付きlocatorから将来受け入れる形式である。Aにある旧S2/.1参照も新A runの結果へ付け替えず、元 `69ee0ff8…` のlocatorを保持した。

## 同一性：確認済みとconsumerで残る選択

対象はLinear H4/1Å/STO-3G、保存tol-only DF10、weighted-Hermitian completion、sector(2,2)の保存state、元full DF-H target、Full one-body＋全fragment0..9、元順序、h=.1/M12/P0。Aのshared bindingとD provenanceの全15物理fieldが一致した。元H/state/snapshot hashを保持し、実snapshotやprivate trialは今回開いていない。

shared bindingのT=.8は元inputのmetadataであり、評価時刻は各physical candidateのTで照合した。異なるroundをT=.8へ置換していない。

|項目|SOURCEで確認した内容|残る条件|
|---|---|---|
|公式/順序|A数値SOURCEとDの `h46_m5_plan.py` がbyte一致。S2/Y4/M5のbinary64 weights・負時間・T*w/q順序、canonical formula IDを既存adapterで照合|別公式・別係数や演算順には移送しない|
|scalar|Aは同snapshot由来constantをPF後に総Tで一回。D `cost_trial` のFULL分岐は同H constant、`compose` はpiece内scalar=0→最後に総Tのcontrol phase一回。B0I/PRのtail identity加算をFULLへ入れない|SOURCE規則の一致。値の新抽出・加算・実pathのscalar再解析は0。新consumer record作成時には同source/scalarを明示|
|control/wrapper|元symmetric_directional、`build_native_hadamard_wrapper(axis=cosine/sine, include_measurement=True)`。full evolution＋測定、状態準備P=0|uncontrolled/D-only/局所Sの価格では代用不可|
|compiler|hashに加えQiskit1.3.0、basis rz/sx/x/cx、opt1、seed17を照合|同hashだけでlowering同一とはしない|
|lowering|保存native policy `M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1`。Dのcompose/pilot/pf_common SOURCE照合とprep hash traceを再利用|Aのprice profileにはlowering_idがない。元native scenarioのconsumer採用は未実施。新ZZ等へ流用する同値性証拠は未取得|
|receipt|DのX/Y独立EXACT判定、carrier→元結果/receipt・fingerprint・SOURCE・prep hashをlocatorで保持|これは元回路条件での保存direct証拠。現Bの価格record/RMSE契約を新しく資格化したものではない|

したがって、**元native条件を明示して採用する場合の5roundの再利用証拠**はある。一方、別loweringを使う場合や元SOURCE/scalar/preparationを確認できない場合は未受入とし、新domainでの直接証拠又は移送証明が必要。形式的に全てをEXACTへ昇格したり、逆に元nativeで確定しているDの証拠を破棄したりしない。

## 既存direct coverage

|候補|保存round / physical ID|T / qPF / actual Δ|D catalog行|価格の来歴|
|---|---|---|---:|---|
|S2/.2|m3 `e2b139ad6211e2c63ae6`|.8 / 4 / .2|2|pf_common native結果、SOURCE e2179f68…、D receipt trace|
|S2/.2|m4 `f12385e615d43121d8a1`|1.6 / 8 / .2|14|長時間研究EXPLORATIONの決定論direct|
|Y4/.4|m3 `2594f35f92e76d58874c`|.8 / 2 / .4|25|M5 pilot由来の標準Y4、SOURCE c7d53f04…|
|Y4/.4|m4 `a3e91da9c7bcfdb2483c`|1.6 / 4 / .4|37|長時間研究EXPLORATIONの決定論direct|
|M5/.8|m3 `1b3aa083ff69842d8f31`|.8 / 1 / .8|46|m5_datasetのX/Y candidate receipt|

各行にX/Y両軸がある。EXPLORATION等の旧cohort名とn=1は来歴として保持するが、Full PFは固定compiler・固定seedで決定論的な回路価格であり、PRの有限trajectory cohort平均と解釈しない。X=Yの新しい仮定や一方の軸からの補完はしない。

|候補|direct証拠のあるround|catalog内欠測|catalog外|現時点で適格な全13roundモデル|
|---|---:|---:|---:|---|
|S2/.2|2|0|11|なし|
|Y4/.4|2|0|11|なし|
|M5/.8|1|1（m4 q2）|11|なし|

catalog外は世界中の価格不存在を主張するstatusではない。後続の公開receiptがあれば同じ条件で追加できる。全39roundのdirect coverageをモデルなしで埋める場合、同native受入後も34round・68軸分の証拠が不足する、というmetadata上の差分である。これを一括compileの推奨としない。

全候補のm0〜2は価格catalog外。actual ΔはS2のm0=.1、Y4のm0=.1/m1=.2、M5のm0=.1/m1=.2/m2=.4で要求幅と異なる。同じq1でも幅・signed time・scalar総Tが異なるため同価格としない。m12はS2 q2048/Y4 q1024/M5 q512で、3つともcatalog外。

## モデル・精度・最小検証案

既存Cの外部係数加法モデルは実装基盤として再利用できるが、Full PF3候補の適格なq比例/q-affine係数は未取得。旧PR C1、H6 Affine、rank12/別splitのcontrolled affine、PRのE[S]を移植しない。固定formula・actual Δ・scalar・loweringごとの `a+bq` は検証候補で、compilerの全反復境界相殺を保証する式ではない。

[MINIMAL_VALIDATION_PLAN.md](MINIMAL_VALIDATION_PLAN.md)：元native5roundの受入が可能なら、S2とY4は保存2点を校正へ再利用、M5だけm4 q2を追加校正、3公式のm5を独立holdoutとする。**追加4round・X/Y最大8 wrapper compile**の識別案。現時点では未承認・未実行であり、early幅の移送・低q・長qや正式Gへ十分な計画ではない。旧m12巨大T2は前提にしない。

Bの適格N、比較margin/許容regretが未供給のため、必要な数値価格精度はUNKNOWN。同時に有効なprice誤差bが得られた場合の `Σ N b` と全候補のG区間を使う条件を計画へ記した。ここではNの仮定、数値精度閾値、実G差を設定していない。

## 技術確認・保全・STOP

受動builderの全3 configuration・39round・78軸の照合がPASS。NumPy/SciPy/Qiskit/PySCFはimportせず、価格値copy/算術、snapshot/private raw読込、科学再解析、fit/prediction/sampling/compile/N/Gは0。production module変更0、Dの保存済みテスト・receipt監査を再実行する必要はなく記録を再利用した。作者によるmetadata確認であり独立科学レビューではない。

関連する既存C branchを継続し、A/B/D/監査のworktree・processは変更していない。旧校正・raw・grant・ledger・marker・KEEP・STOPは保全。[HANDOFF](HANDOFF.md) を公開・remote照合後mandatory STOP。次はユーザーが元native/別loweringと有限科学scopeを判断し、許可された場合のみDが検証価格を取得する。Cの保存値受入分析・fit/予測、BのN/Gはそれぞれ明示した科学scopeが必要。
