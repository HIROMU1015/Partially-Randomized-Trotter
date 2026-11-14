# 未開始p7の高速化とFullとの並列化

利用者の「並列化と高速化を行って」に対応する技術変更。**実行中のFull SOURCE c02e2416ad8ffd63b800a4d1039e8ec1e6cc75aa、process、input、raw/cacheは変更・停止・再実行しない。** 元EXECUTION.mdのFull→p7の逐次順序はDの実装判断であり、今回は未開始p7を別SOURCEで開始して重ねる。6物理点/12軸/最大12実compileの範囲、固定予測・policy・数値gateは変わらない。

## 実装と使用

- `scripts/run_pf_m9_p7_fast.py --execute-approved-p7`：固定p7 3 cellだけの新入口。旧m9 selector/preopen/child、既存F/Tのtrial/price_bindings、元stage worker、C guard/bind/判定を再利用。Fullの実行ファイルを書き換える方法やFull完了の架空recordを使う方法ではない。
- Fullの保存jobでphysical IDがp7と非重複であることを確認する。Fullに重大STOPがあれば新p7投入を拒否する。p7の既存namespace/重複jobがあれば再実行せず保留する。
- `scripts/pf_native_price_fast.py`：元の`h46_long_time_native.compose`関数をそのまま呼ぶ。その**一回のcompose内だけ**、同じblock object列（強参照保持）と全builder引数（型・float.hex・control・capを含む）が完全一致するときに、元native builderが生成したS2回路templateを再利用する。
- 元composeは通常どおり`qc.compose(part.circuit,inplace=True)`（既定copy=True）をsigned pieceの順に行い、total scalarを一回適用する。templateの最終fingerprintも元値と照合し、変更を検出したらSTOPする。candidate/roundをまたぐcacheや新永続cacheは作らない。
- 既存`h46_b3_fast_fingerprint.py`（5a67cea7とbyte一致）をprocess内の元`use_fingerprint_reuse`呼出しへopt-in接続する。canonical v2 bytes/SHA・cache key・入力mutation検査を保持する。emitter/wrapper/Accounting/compiler/作用検査は元SOURCEのまま。
- builder/contextの一時差替えは当worker process内のみで、例外を含めて復元する。実行SOURCEにhelper/hasher/新入口も固定し、price_sourceにacceleration SOURCEを明示する。

新p7は同じ予定public/private `m9_holdout/run01` の未使用directoryを使う。旧one-shot枠は転用しない。通常p7 callerとの二重投入はdirectory/CHILD_STARTEDで拒否される。開始後にSOURCEを差し替えたりretryしたりしない。

## 確認した技術証拠

[FAST_TESTS_FINAL.txt](FAST_TESTS_FINAL.txt)：人工4-mode DFだけの4 test PASS。保存H4入力、H4 sampling/build/compileは含まない。

- S2/Y4/M5、B0I scalar、負時間Full、q反復について、旧/新の全回路fingerprint・global phase・命令数・instruction boundとlow/high-control vectorが一致。
- 測定付きX/Y両wrapperのfingerprint、compiler hash、compiled全6指標、compiled回路fingerprintが一致。compiled-vectorは元の1e-7 gateを通過。人工compileは旧/新×XYの4件で、m9実compile予算とは別の技術test。
- 実行contextを通した元cost_trialも人工入力で7 gateを完了。候補/時間の変更によるcache誤共有、RTE入力の混入、例外後のalias残留を拒否・確認。
- 人工q3のbuilder呼出しはS2 3→1、Y4 9→2、M5 33→6。全33回分等のcompose自体とgate順は維持する。
- 最終runの人工compose wallは、S2 .588→.347 s、Y4 1.402→.537 s、M5 5.260→1.453 s。最初のrunも別logで保存。単発の技術測定であり、精密なbenchmark平均や実H4の倍率ではない。

[P7_FAST_PREFLIGHT.json](P7_FAST_PREFLIGHT.json)：固定予測・p7 6軸、既存Fullとの非重複、元Full SOURCE bytes不変、資源観測。旧m9/m6 reader・元stage/F/T/native/fast-hasherの各ファイルは無変更。

## 並列性と残る負荷

Full 3 cellとp7 3 cellを独立processで並列に扱う。実CPU/RAMの余裕を確認し、worker数は元の資源観測経路から選ぶ。64 CPUを独立した64科学対象へ水増ししない。X/Yは同じevolutionを共有する元の順序を保持する。新しいaxis間IPCやcompiled回路再読込は追加しない。

全回路の保持・compose・canonical hashの走査、native/compiled statevector検査、個々のtranspileは残る。これ以上の変更は今回の未開始cellへの投入に対する費用対効果が下がるため増やさない。**実H4の同一条件での改善倍率は未測定**で、Full/p7のwall比を倍率にしない。p7実行ではcompose時間とtemplate再利用数を技術annotationとしてRESULT内に保存する。

Full/p7の実行SOURCEを区別して報告する。直接価格の有効性と固定モデルのPASS/FAIL/UNKNOWNは別に判定し、有効なmodel-FAIL観測も保全する。最大12軸の判定・公開後、または重大異常でmandatory STOP。追加compile、他round、refit、N/G/winnerへ進まない。
