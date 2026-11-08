# 真空参照CSWAP：実H4限定資源比較の技術準備報告

状態：TECHNICAL_PREPARATION_COMPLETE / SCIENCE_NOT_AUTHORIZED。新adapter・誤差評価コード・一回実行入口を実装し、差分人工検証とlevel2比較に通った。保存H4の入力はロードせず、path identity・実誤差・実compiled価格は未取得。

## 到達点

- state-free SavedPRで保存basis/block/component/policyを復元する接口を追加。旧SavedEngineのselected_circuit・phaseとRTE policyを再利用し、旧engine/prototype/SOURCEを変更しない。
- selected fullのgate列を観測するだけでsemantic unit境界を取得。先頭floor(B/2)をR、残りをLに固定し、fusion/cancellationを壊すbarrierや再探索は加えない。traced bodyと旧wrapperのgate列一致を検査する。
- scalar/event/identity phaseをexact Fractionで保持し、scalar-free L0/R0とP(ell0+r0+sum phi)へ接続した。補正角のbinary64丸めはexact差として記録。元Hのvacuum energyを代用しない。
- 新18人工tests PASS。既存14 PASS／128人工行は再利用し、旧suiteは再実行しなかった。新しい人工2-mode DF配列（2 fragment、L_D1、delta=.02、geometry/basisなし）の保存schemaを用い、2 path×X/Y×3方式×2 topology=24結果行にPASS。人工8-system descriptorは17 wires/16 CSWAP/8 resetを確認し、大きなstate列挙は行わなかった。
- Qiskit1.3.0／rz,sx,x,cx／level2／seed17を全方式に適用。既知SEとcandidateは同一IR・同一費用で、重複compileをしない。native8＋routing8、operator資格16回。全24行のcompiled回路とlayout、QPYを保存。
- level2が読出し直前のRZを省くとunmeasured isometryは変わる。このため全入力の測定POVM差を評価し、正しい分布同値を保証する方式へ接続した。relative phase誤りを検出し、出力phase fittingは使わない。
- 既存interval小gate証明を再利用し、明示IEEE実演算のroundoff/underflowと全input-column計算を保守的に囲う。人工最大座標上界は約1.383e−10。これは実H4の値でも全support認証値でもない。科学的allowance・1/32は採択していない。

[実装](H4_ADAPTER_IMPLEMENTATION.md)、[数値保証方式](H4_NUMERICAL_CORRECTNESS_AND_ERROR.md)、[次回限定実行契約草案](H4_LIMITED_EXECUTION_CONTRACT_DRAFT.md)。[人工証拠正本](../../../artifacts/o2_vacuum_h4_preparation/2026-11-08/SUMMARY.json)、[失敗を含む実行履歴](../../../artifacts/o2_vacuum_h4_preparation/2026-11-08/EXECUTION_HISTORY.json)。

## 未解決事項と次の許可

実H4の正label名とpath identityは未取得。選定規則はUTF-8正label昇順の先頭をn0、先頭3件をorder2のproducts→rotationに割り当てるcost-blind規則として固定した。実LAWの処理は次回metadata science callに含める。保存gate alphabetや大きいcompiled列でのinformativeな一様数値上界も、今回科学入力なしでは数値確定できない。

次の担当はSERVER1 Codex。公開固定SOURCEと契約候補を採択し、五入力照合/metadata選定/復元1回、body2、selected wrapper4、reference wrapper4、native8、routing8、POVM資格16、24行、retry0の限定科学実行への明示許可が必要。保存ground/state、Born、RNG、shot、preparation/CDF/solve再計算、variance/N/期待費用/全RPEは含めない。

全正規化system入力のoperator資格なので、次回もN2 stateを読み込む必要はない。実測/bit-grid/hardware reset/校正/FT合成への総engine保証や、新しい科学精度の受理は別契約。現在のコードが技術的に用意できたことを実H4の科学的成功や資源優位としない。既知SE応用という位置付けも変更しない。

## SOURCE・保全・STOP

SOURCE.jsonに実行code/helper hash、Python3.12.3／NumPy1.26.4／Qiskit1.3.0／mpmath1.3.0と人工入力を保存。最終人工全経路は約2.6秒、peak RSS約594 MiB。実H4の実行時間は推定しない。技術的不備の失敗SOURCEとSTOPも別outputで保全し、科学retryは実施していない。

新directory/branchのみを変更し、元main、旧SOURCE/raw/結果/STOP/grant/ledger/marker、前回0edc0fa5、他研究とC1凍結を維持。今回SOURCE・結果・文書を専用branch codex/o2-vacuum-h4-preparation-20261108 で公開し、remote SHA・本文・blob/tree・人工SOURCE対応を照合する。公開照合完了後mandatory STOP。実H4科学実行や別研究へ自動進行しない。
