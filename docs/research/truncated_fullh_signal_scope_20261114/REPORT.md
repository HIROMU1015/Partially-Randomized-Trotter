# B0I・Full-H pure RTE：26round signalの承認対象scope

2026-11-14 JST。**既存c546計画から2系列・26roundを原本参照で分離した。新しい科学計算・実input読込み・保存signal再解析は未承認・未実行。** 正本は[SCOPE.json](SCOPE.json)。pipeline/数値module/新runnerの実装、既存人工testの再実行は不要で、今回追加していない。

## 元39round案との差分

|項目|今回の派生scope|
|---|---|
|Truncated PF|B0I/prefix7/S2/ΔPF=.1、m0..12の13新round評価|
|Full-H pure RTE|finite K2、q grouping=2^m、R_outer448、m0..12の13新round評価|
|Full PF・保存m6受入|0。AのFull PF最適化へ介入せず、旧spectral受入とdirect再計算のどちらも採択しない|
|依存関係|今回の26roundにはFull PF m6受入を前提としない。元c546の39round案自体は書き換えない|
|保存signal受入/比較|0。B0I m3/m4の旧spectral値を再資格・比較せず、元計画どおり新direct評価に含める|
|件数|最大26 candidate-round評価、Raw complex mean26個、X/Y座標52個。Raw/correctedの組を保存し、X/Yでmean計算件数を倍にしない|
|新reference・scalar抽出|0。保存13 reference pairと保存scalarを使用し、元gateを実行時に確認する。欠測時のsolver/抽出fallbackは禁止|

元manifestは `c5466946ab0ef9b50511210ca1e58fb7147bb0f0:docs/research/baseline_finite_signal_plan_20261113/BASELINE_FINITE_MANIFEST.json`。`candidates[1]`と`candidates[2]`のconfiguration・全physical ID・T/q/R・reference descriptorを既存C.prepareでmetadata照合した。26件ともNOT_EVALUATED、configuration IDは次のとおりで変更なし。

- B0I：`e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc`
- Full-H：`cfa6aee3260de664de9acd444a628faae388c459a2eaf68a786c707573029d1c`

## 固定SOURCE・科学条件

数値SOURCEは `5cc7c07e112ce97205948465c94441cba0042416` のbaseline_signal_pipeline.py/baseline_signal_numerics.py（元86f37d25fと同じ）。現在のworking bytesとの一致も確認した。直接expm/power `dd24dd329`、spectrum/polynomial `e417ffc9f`、finite-K `71ba49bb2`等の元数値関数を変更しない。既存24人工/回帰PASSと統合試験の記録を再利用し、同じ準備を繰り返さない。

Linear H4/1Å/STO-3G、保存tol-only DF10、同H/state/snapshot、sector(2,2)/36、hの元binary64 exact値、M12/P0、元full DF-Hの同sector ground-energy task・prior/decoder/branch/law/scalarを保持する。target/共通誤差契約は元manifest参照で、数値stateの厳密ground同定やbaselineへのrisk資格移送を新しく認定しない。AのH_eff/固有値シフト評価には変更を加えない。

B0Iは保持fragment0..6とone-body、元fragment順・S2を使い、qPF=2^m。元H定数のphaseとdiscarded tail identityのphaseを元SOURCEの順に総Tで一回ずつ適用し、指数のまとめ直しをしない。推定targetは保持Hではなく元full DF-H。Raw=corrected、B=1という元決定論規則を保持する。

Full-Hはone-body込み全non-scalar Hをrandomizeし、deterministic head0。PR prefix0/B3へ置換しない。q=2^mはgroupingでありqPFではない。dt=(T/q)/448、finite-K2 polynomialのR乗→q乗、raw/correctedの独立伝播、constant＋抽出identityのscalar一回を元順に保つ。保存preparationの元bindingとconsumer provenance bridgeはc546から変更しない。

## 承認後に一括実施する範囲

1. 固定scope/元configurationを読み、SOURCE bytes・候補・共有bindingを照合する。元INDEXのsnapshot/receipt/contextをread-onlyで読み、bytes/hash/array/H/state/sectorを元loaderで確認する。保存係数から既存36次元matrix contextを一度復元する。SCF/DF fit/state/ground再生成はしない。
2. prefix7 scalarが保存contextに存在することを先に確認し、Context.tail_scalarの欠測時extractorを使わない。Full-Hも保存preparation scalarを使用する。元13時刻のreference pairを同binding/Tでcopyし、参照数値gateを確認する。この保存reference/scalar利用・数値確認も今回求める科学実行許可に含める。
3. 既存C.execute又はI.produce_signalsを、同一Contextを返すfactoryとreference readerで呼ぶ。B0I→Full-Hの順に各13round、saved=[]とする。元c546と同じ薄いcallback接続だけを使い、承認後の実行SOURCE/caller provenanceを固定する。数値moduleを改造しない。
4. Full-H内部のnon-scalar spectrum、有限Taylor polynomial、normalizer、行列powerは26round評価に必要な元numericsの処理として含める。これをreference/groundの再推定に転用しない。half-step/macro-step/spectrum/polynomial/normalizerは既存Context cacheを再利用し、event列は生成しない。
5. Raw/corrected/reference、signed bias、normalization B/logB/attenuation、局所phase/radius等の元診断、gate結果・未知uncertainty・SOURCE・canonical/physical identity/lawを保存する。Rawそのものと数値診断/誤差資格を分け、厳密なRaw誤差boundを新しく認定しない。

complex128、state norm1e−12、Hermiticity/PF power norm/unitarity1e−9、reference pair1e−10、独立raw/corrected closure1e−9、Raw各座標[-1,1]等の元gateを維持。Full-H normalizer b>1/finiteとB/logB/attenuation rangeを保持し、Full-H有限K演算子へunitarityを強制しない。投影・再正規化・clamp・閾値緩和はしない。

共有context/cacheと最初のUNKNOWNで止まる既存経路を再利用する。36次元の小行列・共通cacheによる順次呼出しはCPU/RAM/worker上限ではない。必要に応じた高速化は自分のproject内で行い、他担当worktree・process・共有環境を変更しない。

## UNKNOWN・保存・STOP

入力/SOURCE/identity/law/scalar/referenceの不一致・欠測、数値gate不成立、非有限値や保存不能では、値/診断/理由を既存recordへ残し、局所NUMERICAL_UNRESOLVED等をUNKNOWNとして扱う。B packetには不適格Rawを渡さずnullと理由を保つ。数値gate失敗を方式の科学的不成立とは解釈しない。

既存executeのschedule内STOPに加え、返却statusがBASELINE_SIGNALS_COMPLETE_NOT_RMSE_QUALIFIEDのときだけ次のscheduleへ進む。その他はbatch全体をSTOPし、未着手round/系列はNOT_STARTED/NOT_EVALUATEDとする。retry・別候補/別law/別精度への切替・Full PFへの代用をしない。

fresh outputは `docs/research/truncated_fullh_signal_results_20261114/run01/` の系列別directory。旧c546案のoutput、旧screen/Full PF m6、SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOPを上書きしない。完了又はUNKNOWNで部分結果・coverage・実行環境・SOURCE・STOPを保存し、公開照合後mandatory STOPする。

## Bへの受渡しと今回含めない処理

既存C.b_mean/I.produce_signalsのpacket（configuration_id/target/source/means/producer_result）を使う。各meanはm/identity/law_binding、Raw/corrected/reference、normalization、bias/数値診断、physical candidate、provider_binding（数値/input/reference SOURCE）を保つ。全13roundのconfigurationも添付する。BへはI.signal_inputが実際に消費されるpacketを束縛できる。

Bの最新確認SOURCE `6106e4d969bb08e95166cbd2b2cc55b9840c7535` ではLD1/LD2の固定2診断が完了したが、正式N packetは未取得である。そのrisk条件やNをbaselineへ転用せず、baseline producerのSOURCE/target/uncertainty受入はB側で別途扱う。今回のmean資格はUNQUALIFIED、数値gate PASSだけでrisk/energy資格を付けない。

今回の承認対象にMC、RPE risk計算、N/shot最適化、AのFull PF/H_eff計算、実compiled価格・model fit・G/候補選択は含めない。**ユーザーの明示承認まで26round評価も保存referenceの数値確認も開始しない。** 現時点はmetadata照合だけで実input未読、新科学評価0。成果のcommit/push・remote照合後STOPを維持し、ユーザーへ本26round一括scopeの実行許可を求める。
