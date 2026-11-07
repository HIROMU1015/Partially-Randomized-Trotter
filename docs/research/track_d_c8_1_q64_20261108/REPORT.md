# Track D C8-1：q64 numerical preflight不成立・最終報告

2026-11-08 JST。**sample00のcompiled X/Y observable検査が両方式とも固定Frobenius閾値を超え、C8-1科学処理を停止した。残7 pathは未開始。** 原本・失敗・SOURCE・旧STOPを保全し、閾値緩和・再抽選・科学retryは行っていない。

## 固定条件と結果

Linear H4/隣接1.00 Å/STO-3G/Nα=Nβ=2、元H、number-CDF8/元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1、q64/cut32。元H厳密groundを仮定し準備費を除外。新ground/DFfitなし。固定packet/CDF/law、scalar/identity/signed angle/normalization/attenuationを維持した。

強いsymmetric対照はuncontrolled Gaussian/directional/head relative fusion/tail共有・63 step境界融合と位相集約、Referenceはwhole path一gadget・B前32/A後32・actual inverse/analyticχ/二CSWAP・62内部融合とcut一境界損失。弱いbaselineや短gadget反復へ変更していない。

P0はQiskit1.3/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=false/topology-free/位相保持。component1e−10、observable Frobenius/signal2e−10、rtol0、global phase fitなし。全256system入力とReference出力全振幅を対象とし、17qubit全Operatorは構築していない。

|項目|予定/上限|実際の完了・判定|
|---|---:|---|
|fresh whole-path event保存|8|8保存・hash固定|
|canonical occurrence / events|512 / 16,384|すべて旧SOURCEで一回だけ生成|
|raw correctness|8|sample00の1件PASS、残7未開始|
|full-wrapper P0 compile|最大32|sample00の4件完了、追加28未開始|
|postcheck|最大32|sample00の4件すべてFAIL、残28未開始|
|sample00 numerical gate|1|不成立。残7を投入しなかった|
|n8価格統計・共分散・モデル残差|全8適格時のみ|未評価・null|

## 失敗した検査

|wrapper|postcheck結果|保存された例外|
|---|---|---|
|strong X|FAIL|compiled_vs_CAR_observable:observable_signal_allowance|
|strong Y|FAIL|同上|
|Reference X|FAIL|同上|
|Reference Y|FAIL|同上|

既存`check_compiled`は、CAR参照observableとのcomponent比較に続き、FrobeniusがSIGNAL_ATOL=2e−10を超えるとこの例外を出す。[checker SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a6bbdac6977df9a6bc430bdd48aad2659a7e191/tests/track_d_h4_reference_checks.py)。四つともこの固定判定へ到達してFAILした。P0変換処理が完了したことをcompile後correctness PASSとは扱わない。

**各Frobenius値そのものとcompiled observable配列は未保存**。既存checkerは例外時に途中のcheck listを返さず、呼出側も失敗observableの保存前に終了するためである。今回、値を取得するための再シミュレーション・再postcheck・高精度再解析は実施していない。個別のcomponent/isometry/leakage値をcompile後PASS記録として新たに認定しない。保存されていない値はnullのままとする。

rawではsample00のCAR/AB/actual A†/真空χ/coherence/X/Y・全256列が既存checkerでPASSし、raw Reference leakageのcomponent/L2は0だった。保存raw paired Xのmax entry差は2.882138972098861e−13、Yは2.3359092487795506e−13。[raw check](../../../artifacts/track_d_c8_1_q64/2026-11-08/run01/paths/q64_sample00/precompile_checks.json)。これはcompiled誤差の保証や全support資格ではない。

## q32との比較と原因候補の範囲

C7-4固定版でのq32 compiled observable最大Frobeniusは1.197156474600893e−10で2e−10以内だった。今回q64 sample00は4 wrapperとも同閾値を超えたが正確な超過量は未保存。q/cohort/path/thread割当が異なるため、qだけの因果効果とはしない。

両方式で同じCAR-observable判定が不成立になったことは、長いnative/P0 gate列の数値伝播、宣言位相/角度の会計、CAR参照・raw/P0対応、observable集約等の共通した数値要因を調べる動機になる。ただし、今回の保存例外だけで原因を特定・定量配賦できない。Referenceの理想測定恒等性への反例や、物理アルゴリズムの破綻と認定する証拠ではない。数値検証条件を満たした実装証拠が得られなかった、という結果である。

## SOURCE切替・並列化と保全

最初のsampling/raw/4 compile SOURCEは**6b1d613ded963b7851afa1c07ba8c646124dd44f**。compileが1 core処理なのにReference各31 CPU枠を予約する問題を確認し、ユーザーが改善と引継ぎを許可した。[承認範囲の変更PLAN](ACCELERATED_COMPLETION_PLAN.md)。

旧controllerだけを調整停止し、workerへsignal/killを送らず4 compileの完了を待った。旧STOP_FAILURE/KeyboardInterruptは調整停止の原記録として保持。これは数値失敗をPASSへ変更したものではない。

新SOURCEは**8a6bbdac6977df9a6bc430bdd48aad2659a7e191**。旧STOP成立後、完了済みraw1/compile4と全eventをhash/bytes照合してコピーし、**未実施の4 postcheckだけ**を実行した。compile予約1、raw/postへ残CPUを配る修正であり、compiler/backend/cache/精度/入力を変更していない。全4 postcheckに16 native threadsずつを配分した。19人工/回帰PASS（新handoff8、旧C7-3 handoff5、C8 guard6）に加え、旧SOURCEの29人工/回帰PASSを保全したが、これらはH4科学PASSではない。

最初のFAILで新規投入を止め、既に実行中の他postcheckを終了まで回収した。全4 FAIL後、新runもSTOP_FAILUREとなり、source_input_preserved=Trueを記録した。sampling追加0、raw再実行0、compile再実行0、postcheckのretry0。総raw1・compile4・post4で終了した。

[旧SOURCE](../../../artifacts/track_d_c8_1_q64/2026-11-08/run01/SOURCE.json)／[旧result](../../../artifacts/track_d_c8_1_q64/2026-11-08/run01/result.json)／[旧STOP](../../../artifacts/track_d_c8_1_q64/2026-11-08/run01/STOP.json)／[新SOURCE](../../../artifacts/track_d_c8_1_q64/2026-11-08/run01_accelerated_completion/SOURCE.json)／[継承hash](../../../artifacts/track_d_c8_1_q64/2026-11-08/run01_accelerated_completion/inherited_evidence.json)／[新result](../../../artifacts/track_d_c8_1_q64/2026-11-08/run01_accelerated_completion/result.json)／[新STOP](../../../artifacts/track_d_c8_1_q64/2026-11-08/run01_accelerated_completion/STOP.json)。

## 価格モデル・長q・全RPEへの意味

A=全q affine/F=元C6a frozen affine/S=sqrtモデルと係数共分散はsampling前に固定し、変更・再fitしていない。q64 CX候補−14128も保持。ただし全8適格pathが揃わないため、**独立q64価格検証、n8平均/SD/共分散、整数CX残差、A/F/S残差は未評価**。失敗pathを除外した統計や4 compileだけによる方式優位を作らない。

旧B/C全13-round条件付き点推定8828b298は変更しない。今回のq64結果で長q母平均・価格モデル・actual共通N・RPE成功保証の未資格は解消されていない。新fit、N/全RPE総費用更新、q≥128 sampling/compile、ground/DFfit、routing/T/noise、詳細科学レビューはすべて0。

**次の最小作業は、長q compileの追加より先に失敗値を失わない数値診断の設計を整えること**。[長q戦略](LONG_Q_STRATEGY.md)は、今回の不成立を踏まえ、直接長q/構造モデル/中間特徴推定を比較し、q1024〜4096の大きい寄与へ何で接続するかを示す。q128へ順番に進む計画は採択していない。

## 原本・公開・mandatory STOP

raw event/QPY/NPZ/conditions/fusion/内部環境はローカルのみ。公開はSOURCE、固定PLAN/seed/model参照、scalar検査/失敗、hash索引と技術報告だけ。旧Stage A〜C7-4/B4/B6b/B-CのSOURCE・結果・grant・ledger・marker・STOPを変更していない。

実行SOURCEと最終公開commitは別で、最終commitは8a6bbdacの子として報告・公開可能な原記録・索引を追加する。科学実装・精度・既存run原本は変更しない。remote SHA/tree/public bytes照合後**mandatory STOP**。

**次の担当：ユーザー/GPT**が数値診断・安定性検証の目的、SOURCE/入力、許可call数とSTOP条件を判断する。**Codex**は明示許可後のみ次の技術実装・科学診断を行い、現C8-1の再実行・残7 path処理・q追加・モデル更新・詳細レビューへ自動進行しない。
