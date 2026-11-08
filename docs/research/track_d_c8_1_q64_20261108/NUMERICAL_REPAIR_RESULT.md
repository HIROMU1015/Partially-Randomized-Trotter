# C8-1：保存sample00・4 wrapperの数値修正結果

2026-11-08 JST。**Aer simulation内部fusionを無効化した修正版で、保存済みstrong X/Y・Reference X/Yの4 postcheckがすべてPASSした。** component1e−10、observable Frobenius/signal2e−10、rtol0を維持し、同じQPY・CAR/raw参照を使った。sampling・raw回路生成・compileの追加は0。公開照合後mandatory STOPする。

## 固定条件と変更点

Linear H4/隣接1.00 Å/STO-3G/Nα=Nβ=2、元Hamiltonian H、number-CDF8/元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1、q64、complete step32後cut、保存sample00。元Hの厳密groundを仮定し、準備費を除外する。ground vector/ground solve/DF refitを使わず、全256 system入力列で測定演算子を検査した。

strongのuncontrolled Gaussian/directional control/head relative fusion/tail共有・63 step境界融合・scalar集約、Referenceのwhole-path一gadget/B前32・A後32/actual inverse/analyticχ/二CSWAP/62内部融合とcut一境界損失を保持した。P0はQiskit1.3/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=false/topology-free・宣言位相保持の保存出力で、再compileしていない。

最初に、失敗時にscalar/observableを保存しない診断欠落を修正し、旧checkerをcallbackで包んだ。旧checker/閾値を変更せず、例外前のcheck・observable・9qubit以下のmatrixを保存する。次に、既存公式Aer double backendの**`fusion_enable=False`だけ**を選択した。gate matrix/angle/cache・位相・入力列・出力振幅を変更せず、phase fit/renormalization/漏れ切捨ても行っていない。17qubit全Operatorは構築していない。

simulation内部fusionの無効化は、価格評価対象のP0回路やstrong/Referenceの回路構成上の共有・融合を変更するものではない。新sampler/Gaussian/compiler/state simulatorは実装していない。[固定修正PLAN](NOFUSION_REPAIR_PLAN.md)。

## 同じ保存回路の結果

下表は各postcheckが保存したCAR-observable Frobenius値。旧設定欄はSOURCE46ce57e1の診断で取得した値であり、数値未保存だった旧8a6bbdacの値を捏造・補記したものではない。

|wrapper|旧設定の診断値|修正版の値|修正版|
|---|---:|---:|---|
|strong X|2.1474799974401975e−10|5.341415202426805e−11|PASS|
|strong Y|2.2578758683084535e−10|5.4908743163607594e−11|PASS|
|Reference X|2.2112325830025662e−10|3.2286933170038684e−11|PASS|
|Reference Y|2.336298580407411e−10|3.251596577512937e−11|PASS|

修正版はCAR・rawの両observableとのcomponent/Frobenius検査を通過した。Reference leakage最大componentはX=8.298263640539766e−13、Y=8.324143976272904e−13、最大L2は各9.399115681456667e−13。0へ丸めていない。

strongの全wrapper位相付き対応・unitarity、Referenceの位相を含むraw branch/isometry対応も既存のcomponent規則でPASSした。**full-wrapper matrixやbranch行列のFrobeniusは診断値であり、observable用2e−10をこれらの判定へ追加適用したものではない。** 例えばstrong Xのfull-wrapper Frobeniusは8.377740835009196e−10だが、固定判定対象のmax entryは5.6622905950939735e−11である。許容値や判定規則を変更してPASSにしていない。

AB=V、actual A†、真空ray、analyticχ、scalar/identity、signed係数・normalization、raw complex coherenceは旧raw PASSを再利用した。今回のpostcheckerは保存raw branch/位相との対応と各X/Y測定演算子を検証した。**既存C3の独立した最終paired-X/Y・compiled vacuumχ総合checkerは今回追加実行していない。** 従って、4個別postcheckのPASSと、元campaignのsample00総合gate・全8価格検証の資格を区別する。

## SOURCE・実行数・保全

|役割|SOURCE|今回の扱い|
|---|---|---|
|sampling/raw/4 P0|6b1d613ded963b7851afa1c07ba8c646124dd44f|原本を再利用|
|旧post失敗|8a6bbdac6977df9a6bc430bdd48aad2659a7e191|全4 FAIL/STOPを保持|
|失敗値保存の診断|46ce57e18b7d8cb606c0446e562ca98be63b4f5d|4回を追加、全4 FAILの値を保存|
|simulation fusion修正|f1e89590c7daea12319352b85b959928ad166213|同4回路を各一回、全4 PASS|

利用者が数値修正・再実行を明示許可した範囲で実施した。旧post4＋診断4＋修正版4という実数を保持し、追加検証を未実行と扱わない。一意なcompiled wrapperは4本のまま。修正段階の新sampling/raw/compileは0。両batchのworkerを完了まで回収し、重複起動・正常workerのkill/再起動はしなかった。4件完了後は各batchのSTOPを保全した。

[診断SOURCE](../../../artifacts/track_d_c8_1_q64/2026-11-08/run02_diagnostics/SOURCE.json)／[診断result](../../../artifacts/track_d_c8_1_q64/2026-11-08/run02_diagnostics/result.json)／[診断STOP](../../../artifacts/track_d_c8_1_q64/2026-11-08/run02_diagnostics/STOP.json)／[修正SOURCE](../../../artifacts/track_d_c8_1_q64/2026-11-08/run03_nofusion_repair/SOURCE.json)／[修正result](../../../artifacts/track_d_c8_1_q64/2026-11-08/run03_nofusion_repair/result.json)／[科学要約](../../../artifacts/track_d_c8_1_q64/2026-11-08/run03_nofusion_repair/repair_summary.json)／[保全確認](../../../artifacts/track_d_c8_1_q64/2026-11-08/run03_nofusion_repair/preservation.json)／[STOP](../../../artifacts/track_d_c8_1_q64/2026-11-08/run03_nofusion_repair/STOP.json)。

固定46原本のhash、元SOURCE/input、再利用実装、10 packageの版、実行SOURCEとGit bytes、診断/修正版NPZ hashを照合した。旧8保存event/seed・原raw/P0・研究契約・grant/ledger/marker/STOPは変更していない。診断用人工5件と修正版人工4件、計9件PASSは技術試験として別記し、H4の科学的結果と混同しない。

公開するのはscalar/check/hash/SOURCE/PLAN/結果のみ。event/QPY/NPZ、conditions/fusion、CPU/RAM/worker詳細とruntime原記録は専用ローカル領域で保全し、今回GitHubへ追加公開していない。最終公開commitはf1e89590の子として文書・公開可能な結果を追加し、科学実装や過去結果のbytesを変更しない。

## 成立した範囲・未解決事項

今回の証拠は**同じq64 sample00で、simulation fusion選択により既存数値gateを通過できた**という限定的な数値実装証拠である。誤差の全配賦、任意q/全supportでの精度保証、実機成功保証ではない。double precisionの局所観測で、rigorous enclosureはない。数値改善は速度改善の主張でもない。

残7 pathのrawと追加28 P0は未開始で、n8平均/SD/共分散・CX整数残差・A/F/Sモデル診断は未評価。4保存回路だけでq64母平均や長q価格を認定しない。C6a/C7-4モデルfit、N、全13-round総費用、q≥128、ground/DF solve、詳細科学レビューは行っていない。旧B/C条件付き総費用結果は更新しない。

次の最小候補は、保存済み出力による最終paired/χ総合gateを完了し、同じ8保存event・修正版SOURCE・固定P0/精度で残7 pathを一回処理するか判断すること。新samplingは不要。ただし、**今回の4 postcheck修正・再検証には残7への自動進行を含めていない**。実行する場合は、その未完了範囲とSOURCE切替を明示した許可が必要である。

将来の実行前には、既存batch単位の進捗を出す小さな技術拡張が所要時間推定に有用である。今回の封印済み実行へ後付け変更はしていない。

**次の担当：ユーザー/GPT**が残7と最終総合gateの実行範囲を判断する。**Codexは公開照合後mandatory STOP**し、追加科学処理へ進まない。
