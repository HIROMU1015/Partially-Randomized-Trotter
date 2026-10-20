# Track B W2c-B3：限定Pilot結果の技術整理

**既存資料の整理のみ／追加実行の決定なし／mandatory STOP。** 対象は公開HEAD `f27e63b78518c9e807b91f0028fd29078439ec4f`。以下の数値は既存表・JSONの引用・表示配置の変更であり、新しい統計解析、費用外挿、順位計算、科学レビューではない。

## 1. 今回確認済みの範囲

固定finite target `P5_general_order`、p=(1/5,3/10,1/2)、x=5/7、m=5、sigma=+1。root256/H160、eta=rho=1e-12、epsilon_axis=1/200、primitive_error=1e-6、alpha_axis=1/320、familywise alpha=1/20。A6 tau=1e-6、K_phi=J=32、math-only。分子geometry・basis・DF rank・L_D・PF delta窓は適用外。

S（serial）→P1（8 worker）の各1回、correctness→baseline→instrumented、6 phase・48 pair・16,896 sampling。Accepted16,280／ZERO_R324／coin reject292、全correctness cellでaccepted≥11。A/B・S/P1・両pass・correctnessの対応prefixでResult/Event・分類・bit消費が一致し、保存値検査とreceipt/STOP joinはPASS。これは固定実装・有限prefixの技術観測である。[実行報告][H]・[保存値検査][V]

|方式|correctness各cell 16件：accepted / ZERO_R / coin|profile各cell 256件：accepted / ZERO_R / coin|
|---|---|---|
|Ordinary、Partial Return + tail、Closed P3 + tail|X/Yとも16 / 0 / 0|X/Yとも256 / 0 / 0|
|General Full Return|X=13 / 2 / 1、Y=11 / 3 / 2|X=219 / 19 / 18、Y=220 / 19 / 17|

上記は対応するA/B・S/P1・profile両passで同じ保存分類。これらを独立反復として合算・解釈しない。Canonical 3方式の全acceptedと、Full固有のZERO_R/coin観測を区別する。未観測の実failureを成功扱いしない。実F01 failureは未発生。[全cell表][T]

## 2. 費用の読み方と保存値

以下は既存[費用表][T]の値を方式・Layer・Axis・S/P1の固定順で並べたもの。**online窓はgenerator→Evidence/retentionまで**で、stream serialization/write・IPC COMPLETEを含まない。setupはX/Y共有pairに属し、axisへ配賦しない。Layer AはG7生成、Bは既存監査・angle/requery経路を含む。CPU秒は量子T/CX/RZ費用ではない。

### Baseline：profile各cell 256件のonline CPU / wall（秒）

|方式|Layer|S X|S Y|P1 X|P1 Y|
|---|---|---:|---:|---:|---:|
|Ordinary|A|4.766563697 / 4.907095653|4.771990764 / 4.911632854|4.977874871 / 5.132179259|4.990149879 / 5.143438060|
|Ordinary|B|12.373388226 / 12.523103601|12.377504983 / 12.528602688|12.946209555 / 13.102241669|12.726880008 / 12.876899972|
|Partial Return + tail|A|4.850251694 / 4.994199961|4.830270074 / 4.971519541|5.089693265 / 5.248772326|4.973619631 / 5.130792011|
|Partial Return + tail|B|12.867395582 / 13.020694303|12.841817580 / 12.995959417|13.392140185 / 13.549119331|13.178281974 / 13.333500237|
|Closed P3 + tail|A|4.752459995 / 4.893098898|4.734595680 / 4.876468144|4.929271952 / 5.090331263|4.865922755 / 5.023927783|
|Closed P3 + tail|B|12.559222543 / 12.704133678|12.540418590 / 12.688975357|13.072041965 / 13.234894451|12.887139993 / 13.046789745|
|General Full Return|A|5.719129852 / 5.865015047|5.693871969 / 5.839818706|5.835956972 / 5.998695800|5.778537778 / 5.933638598|
|General Full Return|B|11.457371501 / 11.605976807|11.448755221 / 11.599497871|12.009068193 / 12.182595844|11.792119912 / 11.953868604|

### Instrumented：profile各cell 256件のonline CPU / wall（秒）

|方式|Layer|S X|S Y|P1 X|P1 Y|
|---|---|---:|---:|---:|---:|
|Ordinary|A|9.078163171 / 9.264736280|8.949121313 / 9.133852598|9.644143623 / 9.823890548|10.493438028 / 10.637100359|
|Ordinary|B|30.103007116 / 30.299357650|30.289199254 / 30.486849784|33.126601109 / 33.391450075|31.961586419 / 32.144049828|
|Partial Return + tail|A|9.345602108 / 9.536091512|9.326586528 / 9.518644125|9.906069781 / 10.079561367|10.486116282 / 10.631702190|
|Partial Return + tail|B|30.771596002 / 30.969215429|30.632305203 / 30.830714978|34.239769545 / 34.537529577|32.820188045 / 33.043807061|
|Closed P3 + tail|A|9.241047530 / 9.432299003|9.206529734 / 9.398017496|10.154431994 / 10.347496117|10.611751939 / 10.773322671|
|Closed P3 + tail|B|30.195192395 / 30.391521050|30.111289947 / 30.310661257|33.854178035 / 34.092468617|32.341606452 / 32.511577976|
|General Full Return|A|12.043866162 / 12.234204235|12.029776094 / 12.222533895|13.324408394 / 13.529425760|13.974425988 / 14.180652794|
|General Full Return|B|27.129507476 / 27.325762971|27.128814992 / 27.325225820|30.500013567 / 30.788616673|28.786334083 / 28.986321035|

### Setup・online・全体窓

次表も既存[実行報告Q7][H]の引用。worker CPU、phase wall、setup、onlineは異なる計測範囲であり、重複加算しない。

|条件/phase/pass|calls|worker pair CPU s|phase execution wall s|coarse setup CPU s|axis online CPU s|
|---|---:|---:|---:|---:|---:|
|S correctness/baseline|256|13.750403|25.391922|4.738200|8.485730|
|S profile/baseline|4096|145.049685|161.806727|4.947334|138.585008|
|S profile/instrumented|4096|330.674752|348.154103|8.159938|315.581605|
|P1 correctness/baseline|256|14.065385|4.310520|4.778403|8.746711|
|P1 profile/baseline|4096|149.919880|30.115602|4.878957|143.444909|
|P1 profile/instrumented|4096|361.691999|71.686146|7.809742|346.225063|

Global outerはCPU **1,258.033384573秒**／wall **792.156549420秒**。CPUはparent own159.135530573秒＋reaped child hierarchy1,098.897854000秒（Git/helper含む）。worker END own CPU合計1,020.444851161秒は別観測で、再加算しない。outerはpre-admission→全join/finalization/STOP、outer export自身と後処理のarchive/offline集計を除外する。phase wallはPHASE_BEGIN→join後resource sampleで、後続offline phase検査を除外する。P1のpair/axis wall和はglobal wallではない。[保存済み集約][S]

|費用・証拠|既存の記録場所と解釈|
|---|---|
|setup・online・保存CPU/wall|[費用JSON][C]の`pair_records[].setup_coarse/coarse`、`axis_records[].online_stages`。`save.serialize/hash/file_hash/write/flush/finalize`等はpairの内訳。parent側witness保存は`outer.parent_storage_costs`。全保存費用を一つの独立値へ分離したものではない。|
|observer・law/child・監査・angle・Codec|Instrumentedの`fine_records`と[関数別診断][F]。selected callbackのexclusive partitionだがobserver効果を含む。coarseへ足さず、pass差を純粋なobserver固有費用としない。|
|IPC・drain/close・欠測|`control_rpc`は観測RPC窓。純粋IPC、observer単独CPU、正常時STOP/close単独費用、未回収resource tracker CPU等はnull/residual。欠測を0に置換しない。|
|保存量・復元限界|[保存量照合][B]：post-witness Result209,513,726bytes、Log41,737,308bytes。full original1,536件／compact15,360件。compactはfull生成後のdigest・保持情報であり、original全文の再構成・再検証可能性を保証しない。原本76filesの[公開inventory][R]を保持。|

## 3. 測定上の制約

- **Baseline／Instrumented**：詳細CachedObserverはoff／on。両方fused、同じ科学入力・seed prefix・guard・保存方針。Baselineにもcoarse timers・承認検査・記録費用がある。Instrumentedの関数別費用は診断用で、計測を外したアルゴリズム固有費用ではない。
- **順序・共有負荷**：S→P1、各condition内baseline→instrumentedの固定順で1回。fresh workerでもfilesystem/cache、GC、I/O、共有負荷を独立制御していない。資源観測は[費用JSON][C]の`condition_phases`と`outer.resource_observation`にあるが、負荷の因果効果や機械非依存の優位性は識別できない。
- **block変動**：[既存block表][T]と`axis_records[].block_statistics`に16/64/256の累積・min/median/max等を保存。例：S baseline Ordinary/A/Xの16-call block CPUは0.284621 / 0.296665 / 0.318439秒（min/median/max、既存値）。同じprefixのblockは独立反復ではなく、変動の存在を定常性・十分性の合否へ変換していない。
- **時計・統計**：[既存人工時計校正][K]はreported1ns、loop正増分min CPU320ns／wall79nsで、時計誤差の厳密上界ではない。SHA256-counterとprefix一致はIID証明ではない。n256の統計的十分性、CPU母平均CI、一般的な方式順位は未確立。量子誤差側のconfidenceをCPUへ転用しない。
- **証拠の境界**：同じprimary Codexの検査であり、外部の独立再現ではない。保存された有限成功を一般的API互換性・failure保証・物理RZ/native合成保証へ広げない。[報告Q6–Q8][H]

## 4. 次の計算が必要になる条件

以下は**求める主張と不足証拠の対応**であり、追加計算の採択・実行許可ではない。

|求める主張・判断|今回使える証拠|不足がある場合に必要となるもの|
|---|---|---|
|固定P5の有限prefixで実API・bit/監査/保存が接続し、観測費用を記述できた|16,896件、対応prefix、全cell表、保存検査・spent/STOP|この限定主張の記述のためにFull-N完走を追加の必須条件とする根拠はない。既存証拠で文書化できる。研究上十分とするかはユーザー・GPTの判断。|
|事前固定した全N workloadを実際に完走し、そのrunの総CPU/wall・保存量を観測した|今回の短いrunの実測だけ|**当該Full-Nの直接実行・保全が必要**。長時間のmemory/I/O・保存/drain挙動も、採択した監視粒度で直接記録する。Pilotからの費用外挿では確認できない。|
|CPU差をobserver・並列化の効果として切り分けたい|off/on・S/P1の記述値、同じevent/bit prefix|残る順序/cache/負荷交絡がその主張を妨げる場合、目的量・反復・順序/負荷管理・計測分離を結果前に定めた追加測定条件が候補。**単にNを増やすだけでは識別できない**。|
|CPUの精度・再現性を数値基準で保証したい|既存blockの記述値と時計校正|必要精度を先に採択し、現在の反復・clock・未分離残差では判定できない場合に追加Pilotを検討。必要n・反復数・閾値は今回決めない。既存値の新しい統計解析にも別途指示が必要。|
|未観測の実failure・別入力/precision/runtime等へ主張を広げたい|今回観測したpathと固定runtimeだけ|その未確認事項が研究上必要になった場合に限り、対象を明示した新しい検証scopeが候補。Full-Nでも未観測分岐・別入力・IID・一般保証が自動的に確認されるわけではない。|

**新規計算なしで記述できる主張範囲**は、固定条件・有限prefixの実装接続、保持されたResult/Event/bit/traceの対応、観測path、既存費用表と測定限界、one-shot完了・保全手順である。構成の一般理論・独立新規性・投稿十分性の評価や、方式の一般優位・Full-N総費用の確定は本資料の対象外。既存G6/G7理論成果の科学的再評価もしていない。

## 5. 次の担当と承認・停止

- **ユーザー・GPT**：主張を上記の有限観測に限定するか、全Nの直接観測・費用精度・因果分離等を本当に必要とするかを判断する。詳細科学レビューを行う場合は、その開始を別途明示する。
- **ユーザー**：追加Pilot／Full-Nを選ぶ場合、科学条件・目的量・N/反復/比較・seed・保存/測定・予算・STOPを採択し、固定SOURCE/manifestに対応する**新しい明示的実行承認**を与える。旧scope/claim/grantは再利用しない。研究方針採択と計算実行承認を混同しない。
- **Codex**：今回は資料の公開・照合までで終了。新しい具体的指示・必要な承認の後にだけ、その限定範囲を扱う。自動延長・再実行・ランキング更新へ進まない。

既存SOURCE・raw・grant・claim・ledger・STOPと科学分類は不変。今回の作業は文書選択・既存値の表示・出典/bytes照合だけで、新規科学計算・統計解析・外挿・ランキング計算・科学レビューは0。参照ファイルのSHA256は[出典記録](track_b_g6_w2c_b3_pilot_result_technical_organization_20261020.provenance.json)。**mandatory STOP。**

[H]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/handoff.md
[V]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/saved_evidence_check.json
[T]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/cost_tables.md
[S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/result_summary.json
[C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/cost_and_outcomes.json
[F]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/fine_cost_buckets.json
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/storage_reconciliation.json
[R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/raw_inventory.json
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/clock_calibration.json
