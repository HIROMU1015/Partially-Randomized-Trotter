# 初回承認対象：保存受入1件 → PASS時のみ新38mean

**本計画は未承認・未実行。** 具体的な対象・SOURCE・数値条件は [BASELINE_FINITE_MANIFEST.json](BASELINE_FINITE_MANIFEST.json)。3系列各1seedの全13roundを揃える有限pilotで、baseline最適性を認定する計画ではない。

## 1. 固定対象と最大件数

|項目|今回ユーザーへ提案する範囲|
|---|---|
|物理構成|Full PF S2/Δ.1/Full10、B0I S2/Δ.1/p7、Full-H pure finite-K2/q=2^m/R448。h=.1/M12・同full DF-H/state/sector/P0|
|保存受入再資格|Full PF m6/F2-FINE 1件だけ。保存μ・reference/数値gate/allowance・SOURCEの適合性を確認し、旧spectral値として再利用|
|新規candidate mean|受入PASS後のみ Full PF残12＋B0I13＋Full-H13＝最大38件。X/Yで倍にしない。raw/correctedが同じPFと独立伝播のFull-Hを区別|
|保存の数値再解析|上の受入1件だけ。新旧mean差・phase/radius再解析・新allowance fit・旧m3/m4再資格は0|
|reference|保存同state/full-Hの13独立pairを共用、新action0。各使用時の元数値gate確認は許可scopeに含む|
|scalar|保存contextのprefix7 scalarと保存Full-H scalarをcopy、一回会計。新scalar抽出0。欠測ならSTOP|
|H/state|保存snapshot/receipt/contextの読取・hash/layout/sector照合と元36次元matrix復元を一度実施。SCF/DF fit/state/ground再生成0|
|禁止|sampling、回路build/compile、価格/残差fit、E[C]、MC/N/RMSE/G、winner、新候補・新8次PF、旧20セル再開|

受入が不成立/情報不足なら、その理由と途中recordを保存し**新38件は開始せずmandatory STOP**。m6をfreshで置き換える自動fallbackや、39件全directへ変更する処理は許可対象に含めない。

## 2. 既存関数への実行接続

数値SOURCEはC `5cc7c07e1` の `baseline_signal_pipeline.py` / `baseline_signal_numerics.py`（元provider `86f37d25f`と同一）。直接expm/powerは元 `dd24dd32`、spectrum/polynomialは `e417ffc9`、finite-Kは `71ba49bb`。新しいexpm正当性比較を追加しない。

1. manifestを固定Git objectから読み、各 `configuration` をB validatorで照合。既存C `prepare` にmanifestのtarget/reference descriptorsを渡す。candidate/値/SOURCEを変更しない。
2. 保存m6の1件だけを読み、原candidate/H/state/snapshot/profile/scalar/reference SOURCEと保存診断を照合する。保存finite値、元1e−9 norm/unitarity、reference pair、現在のoutcome rangeを確認。欠測を0で埋めず、再伝播しない。既存 `gates_passed` だけを根拠に全RMSEを認定しない。
3. 受入PASSなら、元 `signal/numerical/allowance/backend/source` をcopyしたconsumer recordを作り、現在のrequest `domain` と旧producer record locatorを別のbridge fieldへ保持。`status=NUMERICAL_MEAN_AVAILABLE` はlocal数値受入だけ、B qualificationはUNQUALIFIED。`numerical.backend` は旧 `SIGNED_S2_MACRO_STEP_MATRIX_POWER` を保持し、direct-expmとは書かない。
4. `N.Context(root,*N.load_shared_h4(root,shared_binding))` を一回構成。公開INDEXの3 input/context descriptorと元receiptのarray/H/state/sector照合を保持し、prefix7 scalarがあることを**先に**確認する。`Context.tail_scalar` の欠測時extractor fallbackを使わない。
5. 既存 `C.execute` を3 scheduleへ呼ぶ。Full PFだけ `saved=[accepted_m6_record]`、他はsaved=[]。全て同contextを返す `context_factory=lambda: ctx` を使う。m6以外は既存数値leaf、旧値の数値比較はしない。
6. 各scheduleの返却statusが `BASELINE_SIGNALS_COMPLETE_NOT_RMSE_QUALIFIED` の場合だけ次へ進む。既存executeのSTOPはschedule内に作用するため、薄いcallerがbatch全体でも次scheduleの開始を止め、残りをNOT_STARTEDとして保存する。

reference readerは既存SavedArtifactsでPHYSICAL_INPUTSを一度読み、`rounds[m].diagnostics.reference_pair`をcopyする。保存binding全physical fieldとm/Tを照合し、返り値は `{**descriptor, reference, independent_reference, signal_difference}`。descriptorのbindingは共通canonical bindingで、元physical bindingにsource/decompositionだけを加えたbridgeである。新reference solverは呼ばない。

このcallerは既存数値関数への薄い一括呼出しで足りる。承認後に実行caller/provenanceを固定し、numerical SOURCEや科学条件を変えずに実行する。候補ごとの新runner/新grant・承認証明書は要求しない。

## 3. 数値gate・効率・出力

complex128、入力state norm1e−12、Hermiticity/power norm/unitarity1e−9、reference pair1e−10、独立raw/corrected closure1e−9を維持。Full-Hのfinite-K normalizer b>1・finite、B/logB/attenuationのrange、raw各座標[-1,1]を確認する。決定論PFのBは1。投影、再正規化、modや閾値緩和はしない。Full-Hにunitarityを強制したり未観測exact-tail norm=0を作らない。

PFは元signed/fragment順の直接expm＋matrix_power、B0Iはdiscarded identityのphaseを総Tで一回、Full-Hは全non-scalar/one-body・head0・`dt=(T/q)/R`・finite polynomialのR乗→q乗とraw/corrected独立列を保持する。groupingを同総event数で別layoutへ置換しない。

一つの論理batchとして受入→Full PF→B0I→Full-Hの順に進め、重複入力・半step・macro-step・spectrum/poly/normalizerをcontext cacheで再利用する。38候補を巨大event列へ展開しない。36次元と共通cacheのため、初回は共有contextによる順次呼出しが自然であり、CPU/workerの運用上限を意味しない。実際の負荷を記録し、必要なら既存の独立process並列化を使う。共有環境の設定変更・他processへの干渉はしない。

Aの既存専用数値環境を優先し、実Python/NumPy/SciPy/versionを記録する。過去のCPU affinity/worker数を恒常capに引き継がない。個別固定契約の環境指定は保ち、環境差だけで不要な再計算を増やさず、許可SOURCE・精度条件に照らして判定する。

fresh rootは `docs/research/baseline_first_signal_results_20261113/run01/`、schedule別pathはmanifestのとおり。原旧m6は上書きせず、consumer bridge・全record・raw/corrected/reference/normalization・局所診断・SOURCE・missing/not-started・STOPを保存する。BにはUNQUALIFIEDのmean packet、Dには同physical scheduleのmetadataを渡す。

SOURCE/input/candidate/数値gate不一致、reference/scalar欠測、保存不能、実際の継続不能な資源異常では部分保存してSTOP。通常の他チャット進行や一時的競合だけで再起動しない。受入/計算の自動retryや別law/別candidateへの切替はしない。

全指定scopeの保存完了でもmandatory STOP。ユーザーが**本manifestの受入再資格1件＋PASS時新38mean・保存input/reference/scalar処理**を明示承認するまで実行しない。N/価格/Gや科学的比較採択は別の承認scopeである。
