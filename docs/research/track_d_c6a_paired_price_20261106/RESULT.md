# Track D C6a：q2/q4 fresh whole-path paired価格校正pilot

**予定32 pathのraw correctness、全128 P0 wrapper postcheck、最終paired X/Y・vacuum χはPASS。q2/q4各16本の有限pilot統計と探索的校正を取得し、科学実行を終了した。長qモデルの確認的PASS・全law母平均保証・全RPE費用評価は行っていない。**

開始2026-11-06、成果整理・報告2026-11-07 JST。実行SOURCE `6b6667026ec8b670f610e41141d195883a9eb513`。[固定PLAN](PLAN.md)／[result][RESULT]／[STOP][STOP]。C5公開 `9a6920b8355643701ab7436a4fd3b34862e9e3bd` を基点とする専用branch。実行前にSOURCE tree `23cbfb237eee15914c32744266f72f862b1c427f` と公開13ファイルのbytesを照合した。新grant・承認証明は作っていない。

## 固定条件・canonical sampling

linear H4/隣接1.00 Å/STO-3G/Nα=Nβ=2、number-CDF8/元frame順LD2、ordered partial S2/s1/r32/K0、各h=δ=.1。元Hamiltonian Hの厳密ground仮定・状態準備費除外。ground vector・ground solve・DF fitは使わない。

元science SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`、endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、input packet `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、law `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5`。元frame・sector・scalar/identity・signed atoms・normalizationを既存preflightとSOURCE/input前後照合で保全した。[SOURCE][SOURCE]

既存 `configure_rte → sample_occurrence → iter_sample_rte_events` を再利用した。q2/q4各16 fresh whole-path、計96 independent-occurrence設計、各32 events。cohort `track_d_c6a_q2_q4_paired_price_pilot_v1`、master2026110606、q別calibration domain、sample/occurrence別seedを[実行前seed表][SEED]で固定した。q2_sample00..15→q4_sample00..15、各step昇順。全event列を回路使用前に一回生成・保存し[record hash binding][EVENTS]を固定した。両方式・X/Yが同じpathを共有する。旧保存cohortの巡回・短path反復ではない。fixture/seed/cutの事後変更・追加標本なし。

NumPy1.26.4 PCG64と既存binary64 probability mappingによる擬似乱数であり、ideal joint IIDやlaw距離の数学的資格は未評価。q別・sample別・occurrence別の非重複seedは擬似独立の設計で、証明ではない。元input/codeを条件としてq間の独立性を扱い、同じsample番号をq間のpaired標本とはしない。予約q8 holdout domainのseed/event/回路は未生成・未使用。

## 最小拡張・公平性

C3のsaved-file読込と同じ回路組立を`prepare_steps`へ分離し、C3/C5保存入口を残した。fresh入口だけがcanonical event・既存B2の独立full256 CAR参照・既存ordinary診断回路を渡す。`FreshRequest`にSOURCE/input/law・seed・event digest・half-step cutを結び、C5 `wrapper_for` とfull-wrapper価格検査を共有した。旧C5価格入口の固定q2 guardは保持する。

CAR bilinears/raw factors、B2 cached head指数・signed atom、StrongBuilder/Gaussian provider、C3 half-sweep builder、Reference gadget、Stage A correctness、C2 P0 compile/postcheck、state-actionを再利用した。新sampler・Gaussian合成器・compiler・state simulatorはない。各raw workerはcontext/cacheを再利用する。探索的fitも既存generic affine metric fitterで、legacy DF datasetを偽装していない（cosine=X、sine=Y）。

strong symmetricはuncontrolled Gaussian、directional control、head relative fusion、tail同basis共有、q−1 step境界融合、scalar/identity位相集約を保持する。Referenceはwhole Vに一つのgadget、同じAのactual inverse、analytic whole χ、内部q−2境界融合、追加8 qubits・16 CSWAP・χ補正を含む。cutはq2 step1後／q4 step2後で、global cutの1境界融合損失を価格へ含む。head-tail間の新融合・節約値の単純加算・弱い対照への置換は行わない。

## 人工・分子correctness

実行前の人工/回帰23件（新7＋C5 8＋C3 4＋C2 2＋Stage A 2）はPASS。[人工記録][TEST]。C3旧入口のq8は2-mode人工回帰だけで、実H4 q8生成ではない。後処理の共分散式・資格表示も人工データ5項目で確認した。[後処理人工記録][POSTTEST]。人工PASSを分子性能証拠にしない。

全32 raw PASS後に対応128 wrapperを各一回compileした。AB=V、actual A†、unitarity、真空ray/χ、scalar/identity一回、signed角度、strong両control branch、complex coherence・X/Y observable、χ省略negative controlを既存22項目＋paired X/Yで検証した。継承checker名の`saved_CAR`/`saved_raw`は今回fresh pathから作った独立CAR/ordinary診断を指し、旧cohortのpathへ置換した意味ではない。

P0：Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1.0/initial-zero false/topology-free/位相保持。component abs1e−10、observable Frobenius2e−10、rtol0、phase fitなし。directは全512 operator、Referenceはreference/ancilla=0の全256 system入力列・全出力振幅を既存state-actionで検査し、17-qubit全Operatorは作らない。以下は[検証集約][VERIFY]、個別checkは`paths/<sample ID>/`を参照する。

|q|raw PASS|postcheck PASS|raw最大成分差|compiled最大成分差|compiled observable最大Frobenius|reference最大leakage L2|
|---:|---:|---:|---:|---:|---:|---:|
|2|16/16|64/64|2.93099e-14|2.67088e-12|1.34821e-11|2.40273e-13|
|4|16/16|64/64|4.44089e-14|4.76728e-12|2.13415e-11|2.47632e-13|

最終paired X/Y・両方式vacuum χも全32 path PASS。compile前後の宣言位相を保持し、失敗後の閾値緩和・phase fitting・無断retry/resumeは0。SOURCE/input前後一致。local binary64照合であり、丸め誤差の厳密包絡、全law/全roundの実装誤差上界、独立科学レビューではない。

## q別有限pilot価格・変動

各欄は **平均 ± 標本SD [観測min,max]**。差は同じpathのReference−symmetric。SDはddof1、各qの標本単位はwhole-path16本。X/Yの全7価格は全標本で一致したため下表は一軸分で、32独立標本/qとは数えない。全精度・方式間共分散・全16差は[paired統計][STATS]、全128価格recordは[価格][PRICES]に保存する。

|q|資源|strong symmetric|Reference|paired差|
|---:|---|---|---|---|
|2|rz_count|11428.2 ± 728.717 [10245, 13549]|11766.3 ± 729.292 [10583, 13888]|338.125 ± 1.82117 [335, 341]|
|2|cx_count|4296.12 ± 222.606 [3940, 4936]|4056.12 ± 222.606 [3700, 4696]|-240 ± 0 [-240, -240]|
|2|rz_depth|2777.44 ± 158.556 [2515, 3212]|1467.06 ± 124.109 [1271, 1777]|-1310.38 ± 72.3592 [-1435, -1160]|
|2|cx_depth|2083.06 ± 85.9337 [1950, 2318]|922.125 ± 66.511 [824, 1088]|-1160.94 ± 39.6492 [-1230, -1077]|
|2|total_depth|6431.44 ± 348.06 [5853, 7384]|3254.19 ± 271.144 [2823, 3931]|-3177.25 ± 159.533 [-3453, -2846]|
|2|circuit_size|23279.1 ± 1459.62 [20922, 27516]|23569.2 ± 1460.21 [21212, 27807]|290.125 ± 1.82117 [287, 293]|
|2|peak_qubits|9 ± 0 [9, 9]|17 ± 0 [17, 17]|8 ± 0 [8, 8]|
|4|rz_count|21399.1 ± 883.164 [19734, 22729]|21618.8 ± 882.799 [19958, 22949]|219.688 ± 2.98259 [212, 226]|
|4|cx_count|8152.88 ± 272.972 [7628, 8556]|7464.88 ± 272.972 [6940, 7868]|-688 ± 0 [-688, -688]|
|4|rz_depth|5176.81 ± 225.984 [4715, 5514]|2621.25 ± 125.117 [2410, 2814]|-2555.56 ± 157.633 [-2829, -2263]|
|4|cx_depth|3974.44 ± 123.434 [3718, 4148]|1632 ± 68.0666 [1508, 1736]|-2342.44 ± 85.1587 [-2478, -2176]|
|4|total_depth|12039.3 ± 495.544 [11027, 12782]|5809.25 ± 272.291 [5334, 6222]|-6230.06 ± 345.36 [-6836, -5590]|
|4|circuit_size|43666 ± 1782.34 [40267, 46340]|43389.7 ± 1781.99 [39995, 46064]|-276.312 ± 2.98259 [-284, -270]|
|4|peak_qubits|9 ± 0 [9, 9]|17 ± 0 [17, 17]|8 ± 0 [8, 8]|

有限pilot内の差の符号（X/Y共通）：

|q|資源|減少/同値/増加（16本）|
|---:|---|---:|
|2|rz_count|0/0/16|
|2|cx_count|16/0/0|
|2|rz_depth|16/0/0|
|2|cx_depth|16/0/0|
|2|total_depth|16/0/0|
|2|circuit_size|0/0/16|
|2|peak_qubits|0/0/16|
|4|rz_count|0/0/16|
|4|cx_count|16/0/0|
|4|rz_depth|16/0/0|
|4|cx_depth|16/0/0|
|4|total_depth|16/0/0|
|4|circuit_size|16/0/0|
|4|peak_qubits|0/0/16|

各qでmethod×axis×7資源の28列sample covarianceを保存した（rank≤15、X/Y同価格や固定幅による特異性を含む）。同じpathの方式間・資源間・axis間の依存を落としていない。qubitsは幅として扱い、roundへ足し上げない。min/maxは観測範囲でありlaw supportではない。欠損/失敗IDはなく予定32本が完了した。complete-caseによる有利な平均や未証明の母平均CIは作っていない。

Referenceの追加register/CSWAP/χ/actual inverse/cut損失をfull wrapperへ含む。RZをTへ換算せず、DAG depthを実機時間と同一視しない。routing・有限精度RZ合成・実機reference維持の費用は未評価であり0とはしない。強いbaselineの最適化は同政策で比較した。

## 探索的qモデルと共有校正

[探索的affine係数][FIT]はq2/q4の有限標本平均へ既存uniform fitterを適用したもの。2平均・2係数なので残差自由度0で、校正残差0は確認的PASSや長q誤差証明にならない。q8 holdoutなし。native構造＋boundary補正やReference lane/critical pathモデルは、2つのq平均だけでは独立した項を識別できず、別の事前固定特徴・独立検証が必要である。

旧C5固定q2・C3固定q4/q8・Stage A保存q1は、今回のfresh標本へ混合していない。古い固定q8をholdoutに転用しない。C2の手動cross-step融合なしという条件差も保持する。qごとのlaw path長・cut・境界数/競合も変わるため、価格差の単独因果効果をqだけへ帰属させない。

将来の予測には同じcalibration係数が共有される。C6a内のC6b提案整理として、既存統計/係数から条件付きlinear covarianceを[後処理SOURCE](../../../scripts/summarize_track_d_c6a.py)で算定した（SOURCE SHA256は[提案JSON][PROPOSAL]）。q間の擬似独立を条件にCov(meanq)=Sq/16、b=(mean4−mean2)/2、a=2mean2−mean4を使い、全係数共分散を保存した。これは新しいcircuit/holdout実行・再fitではない。shared input/codeの不確かさ、PRNGの厳密なjoint-law誤差、model biasはこの式に含まれない。

## C6bへの提案と未解決事項

[C6b提案](C6B_PROPOSAL.md)は未採択・未実行。絶対1単位＋探索的calibration規模の5%という価格幅、古典family alpha=.05/36の名目的normal計画、独立q8 sample数の条件式を提示する。paired差はnear-zero差で割らずsymmetric価格規模を使う。主資源、RPEのenergy精度/成功確率を新たに固定していない。

今回の完全pilotから、36量中 **20量で校正分散のfloorがこの提案幅を阻む**。全量を満たす条件付きholdout数は null（nullならholdout追加だけでは提示できない）。これらは未知q8 SDを2×SD4と仮定したplanningであり、必要数・同時信頼度の保証ではない。

CXのpaired差は各qの16本すべてで同じ（q2 −240、q4 −688）で標本SD=0だった。ただし全lawやq8の分散0を認定するものではなく、条件式で得たn32は設計下限にすぎない。q8実分散、tail/support、normal近似のcoverage、finite-bit IID bridge、affine biasは未確認。floorが大きい場合は価格幅を広げるか、追加calibrationを別途承認してからholdoutを固定する必要がある。holdout結果を見てfit/seed/cut/閾値を調整しない。q8で仮に成立してもq>8や13-round期待価格へは移送しない。

B4-2 `07d18bfe1d5145d6f8de026c0d994d5cb1c783c8` の14738は元H厳密ground・ideal finite-RTE IID・exact native等を条件とした13-round Hoeffding十分shot数の保存結果として参照するだけ。再算定・価格との乗算は0。両方式に同じNを共用する全使用domainの測定同値性・実装誤差資格、全roundの期待価格は未資格であり、今回の32 pathから全RPE優位を認定しない。

## 保全・公開範囲・運用上の限界

raw/compiled QPY・参照/observable NPZ・ordered eventの32 archive、各20 member（計640）、member bytesを元出力と照合した。[hash索引][PAYLOAD]／[公開範囲][PUB]。**archive本体・event内容・参照配列はローカル保全のみで、GitHubに公開していない。索引のarchive名はローカル成果物を指し、remoteから取得できる意味ではない。**

自動承認レビューがevent/回路/参照行列を含むraw archiveの公開用読み出しを拒否した。理由はその具体的raw payloadのGitHub公開許可を確認できないため。拒否を迂回せずraw upload0とし、公開はSOURCE/PLAN、科学checkのscalar要約、個別資源/paired統計/モデル提案、hash索引に限定した。内部CPU/RAM/runtime/thread、絶対path、資格情報、非公開input raw、conditions内のevent列を公開対象から除く。必要な原出力とCPU/RAM/cacheはローカルでそのまま保全した。

新旧SOURCE/input/result/grant/ledger/marker/STOPを保全。過去の固定C5/Stage A/B4等の証拠を書き換えていない。新samplingは今回の96 occurrenceだけ、compile128、追加sampling/ground solve/DFfit/B2/B4-2/N/Born shot/RPE成功率/全RPE費用/q>4分子回路/holdout/詳細科学レビューは0。

独立path/方式/axisをprocess並列化し、CAR/head/Gaussian cacheを再利用した。一方、thread予算の切捨てと完了workerからの再配分なし、全raw完了を待つstage barrierには高速化の余地があった。実行中SOURCEをhot-editせず、正常jobを中断/再起動していない。将来はstage別実測memory、余剰CPUの配分、pathごとのraw PASS後compileのpipeline化が候補。今回は速度最適性や実測speedupを主張しない。共有OS/venv/scheduler/他job/他ユーザーを変更していない。

## 終了境界・次の担当

**C6aの科学実行と許可された整理を終了。専用branchへ公開し、remote SHA/tree/公開file bytesを照合後mandatory STOP。** rawは公開範囲から除いたままローカル保全する。

次担当は **ユーザー／GPT**。C6bの対象資源、価格幅、古典信頼度と統計前提、追加calibrationの要否、独立q8 holdout標本数・判定規則を判断する。raw payloadをGitHubにも公開する場合は、その対象への明示許可が必要。**Codexは別の明示許可後だけ**採択scopeを実装・検証する。C6b、長q移送、全RPE総費用、追加科学計算、詳細科学レビューへ自動進行しない。

[RESULT]: ../../../artifacts/track_d_c6a_paired_price/2026-11-06/run01/result.json
[STOP]: ../../../artifacts/track_d_c6a_paired_price/2026-11-06/run01/STOP.json
[SOURCE]: ../../../artifacts/track_d_c6a_paired_price/2026-11-06/run01/SOURCE.json
[EVENTS]: ../../../artifacts/track_d_c6a_paired_price/2026-11-06/run01/event_bindings.json
[VERIFY]: ../../../artifacts/track_d_c6a_paired_price/2026-11-06/run01/verification_summary.json
[STATS]: ../../../artifacts/track_d_c6a_paired_price/2026-11-06/run01/paired_statistics.json
[PRICES]: ../../../artifacts/track_d_c6a_paired_price/2026-11-06/run01/prices.json
[FIT]: ../../../artifacts/track_d_c6a_paired_price/2026-11-06/run01/exploratory_affine.json
[PROPOSAL]: ../../../artifacts/track_d_c6a_paired_price/2026-11-06/run01/C6b_planning_proposal.json
[PAYLOAD]: ../../../artifacts/track_d_c6a_paired_price/2026-11-06/run01/payload_archives.json
[PUB]: ../../../artifacts/track_d_c6a_paired_price/2026-11-06/run01/payload_publication.json
[SEED]: ../../../artifacts/track_d_c6a_paired_price/2026-11-06/technical/seed_plan.json
[TEST]: ../../../artifacts/track_d_c6a_paired_price/2026-11-06/technical/synthetic_tests.json
[POSTTEST]: ../../../artifacts/track_d_c6a_paired_price/2026-11-06/technical/postprocessing_tests.json
