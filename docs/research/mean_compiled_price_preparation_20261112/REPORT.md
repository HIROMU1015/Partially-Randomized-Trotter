# 長q・大Rの平均compiled価格：技術準備（2026-11-12）

**全trajectoryを展開しない期待値計算の核と、保存値だけを渡すB向けproviderを実装した。実H4/H6の平均compiled価格はまだ得ていない。** 固定サイズの境界状態が閉じるモデルでは、signed piece・PF/RTE・outer境界を引き継ぎ、イベント反復とouter反復を二乗法で合成できる。人工例でC1構造計数器による全列挙と一致し、新15件＋既存の純粋な人工テスト3件＝18件PASS。物理C1の境界状態閉包・平均S・compiler残差の長q/R資格は未確定であり、実候補への新価格はmissing/nullを保持する。

[実装](../../../src/trottertracks/mean_compiled_price.py)／[新テスト](../../../tests/check_mean_compiled_price.py)／[人工結果](ARTIFICIAL_RESULTS.json)／[SOURCE索引](SOURCE_INDEX.json)／[HANDOFF](HANDOFF.md)／[将来の限定科学案](SCIENCE_PLAN.json)／[保全確認](CHECKS.json)。実装SOURCE `09489eedbbd61c7396138634a896b8d8ecc75161`。基点f78bbc0d、既存C branch `codex/r-variable-compiled-price-design-20261111`を継続した。

今回実施したのはSOURCE/保存provenanceの静的調査、実装、人工イベント/小行列/mock試験だけ。新しい実分子sampling、量子回路build/compile、構造Sの科学的再計数、価格fit/再校正、保存科学値の数値再解析、実価格予測、signal/N、実N×C/G、最適候補の採択、詳細科学レビューは0。C1係数・資格false・cohort/役割・旧SOURCE/raw/結果/STOPを変更していない。

## 対象とSOURCEの確認

物理的な将来対象は、Linear H4/H6・1.00 Å・STO-3G、保存tol-only DF10/DF19・weighted Hermitian completion・元fragment順・保存H/state/snapshot、H4 S2/δ=.1/PR prefix1/K2、H6 4次m5_best/δ1.6/PR prefix7/K2。H6の元順11 signed pieceと負時間、整数allocation・R≥11を保つ。H6早期の別δ/prefix5へ移植しない。価格unitはsymmetric-directional controlled PRと測定付きfull Hadamard wrapperのcompiled RZ/測定1回、P=0。元Qiskit1.3.0・rz/sx/x/cx・opt1・seed17・compiler profileを維持する。

| 成果/実装 | 調査・再利用したSOURCE | 意味と制限 |
|---|---|---|
| C1 | `2e6a0cdf5a647058cc7afeeb5741bfafa24c974b`、科学SOURCE `1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7`、凍結`7eeabf0ae482ee610d667e28470692d209f2bb21` | 保存8セル・S/compiled対・full wrapper、3係数残差。有限cohort、model未採用。係数を読んで新価格を計算していない |
| C2準備 | `f78bbc0d6ab350a2b2534ea6d169565895f0f1dc` | 旧高確率N/P1に基づく40＋任意16等の未実行計画を保全。今回のRMSE/M/κ方向へ自動採用しない |
| connected-cluster概要 | [既存統合結果](../../rte_compiled_cost_validation_summary.md)、同f78固定tree | H4 DF12/LD3とH5 DF9/LD4等の明示条件、局所1〜3-event、条件により4-event。主に中央uncontrolled RTE、有限L4/6/8の局所検証 |
| 旧校正・予測・holdout | `rte_connected_cluster_cost_validation.calibrate_connected_cluster_cost_model`、`predict_connected_cluster_cost`、`validate_connected_cluster_calibration_holdout`、`validate_operational_connected_cluster_cost_model` | 旧runner `scripts/run_rte_connected_cluster_cost_validation.py` のcalibration/validation分岐。局所compile→固定calibration→予測/未使用full holdoutを分離 |
| 実compile経路 | `_compile_calibration_length`内の`compile_window/connected_value`、`QiskitDFRTEEventCircuitBuilder.build_sequence`、`TranspiledCircuitCostCache.get_or_transpile` | 旧local windowは`controlled=False`。C1の元`sample_requests/compose`→native full wrapper→`transpile_and_measure_cost`と回路unitが異なる |
| 既存平均native計数 | `native_df_gate_model.expected_rte_occurrence/expected_partial_trajectory`、[説明](../track_a_native_df_theoretical_gate_model_20261016.md) | basis状態DP・supplied Fraction law・phase幅を持つ。閉じたoccurrence境界、O(R)更新。C1 literal Z-fold/全境界やP0後価格の代用にはならない。既存S4は3-pieceで、H6 m5_bestの11-pieceではない |
| B公開interface | `724d313b9c8445763f0fe9c41feb573df28ba5d5`、観測済み公開head `7dd61438b64a4b015ce0b7a7db81187d9867d8a2` | `c_provider(task, requested_rows)`、candidate/axis/cohort、有限cohort/model資格。新RMSE/M/κ共通仕様の完成・接続を主張しない |

旧実artifactのprovenanceも確認した。主H4 LD3/δ.1/ref4/K2・holdout1500/375 artifactは、実行時HEAD `e07a5e60ce6213fca3e7b05f733eae2174a71eae`と**dirty/untracked状態、SOURCE SHA256、exact snapshot hash**を保存している。HEAD単独を完全な実行SOURCEとみなさない。今回参照した固定treeの校正moduleは記録hashと同一ではなく、確認した履歴から旧実行SOURCE全体を再構成したとも主張しない。元source hash/証拠statusを[SOURCE_INDEX](SOURCE_INDEX.json)へ転記し、数値係数は移植していない。旧実行のsnapshot/raw/cacheは開いていない。

## C1と局所モデルの会計範囲

C1の`Accounting`は、uncontrolled forward head、directional reverse head、RTE application順、同basis共有、identityでのbasis閉鎖、隣接inverse basis、wire-local literal Z-fold、tail相対phase、scalar一回、X/Y wrapper/測定を含む。`r_price_phase1_science.structural`はq outer×J signed piece×r_j eventを走査する。counterの作業状態は小さいが、現backendはrequests/eventsを一括保持するため全体memoryはO(qR)。qやRに比例する式だけではSを得られない。

`basis()`のpending状態はtail/head・piece・outerを跨いで継続する。`tail()`にはactive basisの閉鎖/relative phaseがあり、gateがpending basisやZをflushする。相殺を一つのRTE内部だけに閉じたり、outerごとにfinishしたりすると、元Sと異なる会計になる。1Q再合成/CX界面等のcompiler固有効果はSが完全には扱わず、C1はθ0+θq q+θL qRを同path C−Sの補正として校正した。

旧connected-clusterは、κ1=C(e1)、κ2=C(e1e2)−C(e1)−C(e2)、κ3=C(e1e2e3)−C(e1e2)−C(e2e3)+C(e2)を局所compileから得る。Taylor次数とcomponent/basisの条件付き局所平均を、イベント確率・window多重度から再構成する。旧K2 modelはL≤8のholdoutや条件移送で小さいが非ゼロの残差を持ち、LD6等では3-eventが不通過だった。古い5%/2%という運用基準、係数、独立SE合成を現在のDF10/DF19・制御構成へ移植しない。旧controlled q依存の検証もtime-evolution subcircuitで、Hadamard/測定を含む現在のunitではない。

**二重計上を避ける選択**は次の通り。

1. 構造平均が得られる場合、`E[S] + frozen C1 residual`を比較用の仮説として残す。C1の資格を上げず、長q/Rで正しいとは仮定しない。
2. 局所補正へ進む場合、同一の境界/入力/制御で測る **D=C−Sのpaired connected increments**を新たな残差modelとして使う。`κ(D)=κ(C)−κ(S)`は同じ局所定義で初めて意味を持つ。PF/RTE・piece・outer・wrapperの残差も必要で、内部RTE成分だけではfull価格を作らない。
3. 旧κ(C)をそのままSへ足さない。C1残差と重複する局所残差を二つとも足さない。κ(C)だけで全体を作る別方式を選ぶなら、Sを再加算せずfull boundary/unitを別途検証する。

新`local_residual_mean`は旧`_iid_sequence_form`を呼び、K2 order patternの期待window数を再利用する。正の重みを持つ係数が欠ければ失敗する。返すのは**内部RTEのC−S残差成分**で、boundary/full-wrapper平均はnull。`combine_mean`は残差modelを一つだけ受け、raw C cluster、C1＋localの重ね足し、不完全なfull-wrapper coverageや負値のclampを拒否/欠測にする。これは代数的な会計規則で、未校正残差を実価格に資格する機構ではない。

## 選択した期待値方式：境界状態と報酬の伝播

有限状態sで、eventまたはdeterministic blockの遷移Pと、状態ごとの期待RZ報酬gを与える。二つの処理を順に行う合成は

\[
(P_1,g_1)\circ(P_2,g_2)=(P_1P_2,\ g_1+P_1g_2).
\]

`MeanKernel.then/repeat`はこの合成と二乗法。`full_wrapper_mean`は

\[
W_{\mathrm{pre}}\left[\prod_{j=1}^{J}L_j E_j^{r_j}R_j\right]^q W_{\mathrm{post}}
\]

を元signed piece順に評価し、状態を境界でresetしない。初期法則とterminal flush報酬を明示入力とし、未知のterminalを0補完しない。scalar-once・axis・測定の固定費はprefix/suffix/terminalへ一回だけ含める。pieceごとに異なる分布を使え、signed weightとlaw SOURCEを保持する。符号からevent確率を自動生成する物理kernel builderは未実装である。

成立条件は、**固定サイズの状態空間が遷移・コストに関して閉じること、piece内の同じkernelの反復とouter kernelの反復が元の法則を保つこと**。相関を持つsamplerやbasis-planなら必要な履歴を状態へ含める。component/basis・support・角度・phaseを粗くまとめて、同じ状態だと仮定してはならない。

C1のpending Z角、tail相対phase/identity位相、literal zero判定は一般に履歴依存の連続/増大状態を持ち得る。したがって、今回の3状態人工例が正しくても**実C1のE[S]を有限3状態で計算できるとは言わない**。閉包できない場合の候補は、状態を保つstreaming S（path memory削減、時間O(qR)は残る）、または局所κ(S)と境界modelによる平均S近似である。後者も打切り/境界誤差を持ち、実データでの新計数・検証が別途必要となる。

局所IID期待値はpiece内で`(r_j−k+1)_+ × ∏p_j(order)`という重みを持つ。H6ではτ_jと整数allocationがpieceごとに異なるため、piece別に組む。headを挟む2eventやouter端部を一つのIID event列として平坦化しない。Taylor patternだけの式を使う場合、component/basisの条件付き平均はそのpiece/lawで既に校正されている必要がある。解析確率を使えることは、finite CDF samplerとの距離や全lawのcompiled価格保証を意味しない。

## 計算量・人工結果

状態数をd_s、短いevent alphabet数をA、piece数をJ、全event位置数をE=qΣr_jとする。q/Rの全列挙と、native逐次計数、閉状態期待値を区別する。

| 方法 | 処理量・memory | 制限 |
|---|---|---|
| 完成回路compile | event/primitive列生成と保持、QC/fingerprint/DAG/transpile。少なくとも出力規模に依存 | 実compiled整数価格のoracle。巨大q/Rの時間やmemoryを今回測っていない |
| C1逐次S/既存basis DP | 概ねO(E＋qJ×head量)、counter/DPは小状態だが現requests保持はO(E) | streamingへ変えても平均値取得・時間削減とは別。旧DPの閉境界費用はC1そのものではない |
| 新closed-state合成 | kernel準備O(Jd_s A)等に加え、dense合成O(d_s³[Σlog r_j＋log q＋J])、保持O(Jd_s²)、作業O(d_s²) | d_sがq/Rと独立に閉じる場合。状態生成が増大/列挙困難なら改善保証は無い。sparse最適化は未実装 |
| 局所κ期待form | 固定K2・cluster1〜3でpieceごと最大14個のform、q/Rは重みへ入る | 係数・法則・境界の準備費用を含まない。component組合せ/校正数・打切りsystematicは別 |

人工の短いsigned 2-piece/q2では、64通りの全イベント列を旧C1 `Accounting`で計数した重み付き平均と、新kernelの平均が両軸で一致した。人工Ginv–X–G blockは同basisの隣接inverse cancellationと異basis間のliteral Z-foldを実際のC1 counterへ通している。これは量子operatorや分子compiled価格の検証ではない。境界を閉じ直すと人工費用2→4になる反例も確認し、境界費を落とした加算を排した。

人工11 signed piece、q=2²⁰、各r_j=2¹⁶の755,914,244,096 event位置に対し、3状態・243 kernel合成で計算し、event配列を作らなかった。一つのP/gのarray payloadは96 bytes（3×3行列＋3-vector）で、peak RSSや全保存量ではない。局所期待formも人工E=10¹²で14項のままだった。[ARTIFICIAL_RESULTS](ARTIFICIAL_RESULTS.json)に条件を保存した。これは閉状態の計算構造の確認で、実分子の高速化率や未測定wallを意味しない。

計算はbinary64で、行確率/finite値のfailure checkを行うが、roundoff boundはnull。確率を後付け正規化したり、価格を0へclampしたりしない。厳密な期待値や長q誤差包含が必要なら、その閉包model/precisionについて別の検証が必要である。

## B providerとκ schedule

`MeanPriceProvider(task, requested_rows)`はc_provider形式の**受動lookup**。実装内でmean kernel・S・fit・価格予測を呼ばない。供給済みのexact identity/axis/cohortレコードをcopyし、無ければmissing。DIRECT finite cohort、MODEL未採用、OUTSIDE_DOMAIN、SELECTION_REQUIRED、FAILEDを区別する。候補・軸ごとに1件とし、複数cohortは`selections`による明示選択が必要。探索/確認をpoolしない。PR以外は既存PF providerへBがrouteする。

full candidate・qPF/R/formula/prefix/δ/K・signed構成、H/state/snapshot、compiler/full wrapper/P0、源付きlawを保つ。finite cohortのnと量子shot N、構造proxyとcompiled DIRECT、model-law期待値と全law保証を分ける。未知の不確かさはnull。C1のmodel資格trueへの変更は拒否し、modelはfalseのまま返す。B `run_missing`への人工結合と、未取得/未資格を受けた後の暗黙再lookup/再compileが起きないことを確認した。総費用集計器は呼んでいない。

RPEの**κ_m schedule**とconnected-clusterの**κ1/κ2/κ3**は別物。前者はBから`source/schedule_id/M/kappa_schedule`等を含む外部use-contextとして渡し、copyで対応付ける。後者は価格incrementのlabelである。前者からT・q・N・MをCが計算/最適化せず、未提供ならNOT_SUPPLIED、提供時もECHOED_NOT_SCIENTIFICALLY_VALIDATED。今回確認できたBの公開共通APIは旧dyadic qRPEを持つため、新RMSE/M/κ APIの完成互換を主張しない。非dyadic scheduleに合わせるB側のidentity/validator変更はBの責任とし、pipeline引数でそのAPIを受ける設計である。

## 次の科学範囲・STOP

[SCIENCE_PLAN](SCIENCE_PLAN.json)は未承認案。最初の候補は、C1の保存4cohort・各n8（計32 path）を**新mean方式のdevelopment/diagnosticとして**再利用するS0で、新compileは0。ただし期待S/法則の実評価・保存値再解析を含むため、今回実行していない。C1原本の校正/独立検証役割を保持し、新mean方式にとって新しい独立検証だと呼ばない。physical state/phase閉包が不足すれば平均価格は未取得のままSTOPする。

fresh compiled価格の次案は、Bの公開RMSE/M/κ scheduleがphysical candidateを決めた後、各系で同一Rの低q anchorと長q holdoutを一つずつ選ぶ有限pilot。各n4、4cohort・最大32 full wrapperを一つの予算案とする。C1 θは再fitせず、全て新cohortでmean/残差を診断する。n4は小pilotの予算で、必要精度/全law保証の根拠ではない。候補/ID/SOURCE/seedは現時点で未固定なので、このまま実行可能な契約ではない。R方向やlocal残差の校正が必要なら、その境界classと別の未使用holdoutを固定した独立計画を先に作る。旧40/16案へ自動復帰しない。

新local modelへ既存C1/C2データを使う場合は役割変更を明示し、新しい独立cohort/点を別に確保する。統計分散・係数共有・piece/round依存・model systematic・sampler距離を一つの独立SEへ潰さない。必要価格精度は、将来Bが固定Nまたは採用したRMSE会計の条件下で費用差を識別できるかから決める。今回はN×C、誤差の加重和、順位、最適M/R/κ、全RPE成功を新計算していない。

[新15件](TESTS.txt)と[旧3件](REUSED_TESTS.txt)はPASS。旧pytestの初回collectionはimport path不足で失敗し、記録を[保存](REUSED_TESTS_COLLECTION_FAILURE.txt)した上で、当該commandだけ`PYTHONPATH=src:tests`を付けて解消した。共有環境・依存packageは変更していない。作者自身の技術確認であり、独立科学レビューではない。

次：**ユーザー**がS0などの具体的な科学範囲を判断する。**Codex C**は選択範囲のphysical kernel/runner/SOURCE・データ役割を固定し、明示許可のある範囲だけ進める。**Codex B**は外部schedule use-contextと既存record形式の取込みを担当する。A/B worktree/process/cache/未公開変更を操作せず、旧原本/STOPを保全した。今回成果をcommit/pushし、remote SHA/主要成果物を照合後mandatory STOP。
