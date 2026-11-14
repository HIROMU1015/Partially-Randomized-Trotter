# 決定論PF m0：固定された有限取得

利用者が承認したFull PF / B0I p7 × S2 / 標準Yoshida4 / M5 × X/Yの12軸だけを対象とする。保存H4 linear 1.00 Å / STO-3G / tol-only DF10、m0、T=q×actual幅=0.1、q=1、P0除外。C alignment `0b4b2d941cd669aaa45231e1236bc2d3158a9d12` のconfiguration / physical ID / axisをそのまま使用する。価格モデルの判定・予測・refit・他roundへの移送・N/Gは行わない。

## 重複確認

[DUPLICATE_PREFLIGHT.json](DUPLICATE_PREFLIGHT.json) は旧coverage `bcb67b2b2bb78776db1dd8d0625a3facb37cd298` への差分である。5 carrier 746 record、Full receipt catalog 136候補を、ID・schedule名によらずsystem/method/prefix/formula/T/q/actual幅/R/Kで照合した。該当0、既知の正確なprivate trial pathも該当0、関連実行job0。12軸とも不足確認済み。

PF commonの旧q=1 recordはT=.8、今回のm0ではない。旧direct retry106の788軸もmetadata確認：H4_DF10は122軸、今回のFull10 / B0I p7 / q1に対応する候補なし。H4_LEGACY_DF12・他prefix・別公式を流用しない。既存のsource/scalar/wrapper監査は旧coverageの固定locatorを継承し、再監査や数値再集計をしていない。関連remote3枝は既存coverage/m9公開時のrefと一致を確認した。未列挙private archive・別サーバーはUNKNOWNであり、全履歴の不在証明とはしない。

実行入口でもmetadata差分を再確認する。発見した重複候補は自動再compileしない。現在は受入候補・保留0なので全6 cellのX/Yを取得する。

## 元経路と観測

`scripts/run_pf_m0_prices.py` のselectorだけが新規。Fullは元`full_trial`、p7は元`checked_trial`、両方とも固定native `af284d853890155bacd19e61850c196bd1d5c660` の`cost_trial`と測定wrapper / compilerを使用する。Stage 1/2・m9 caller、旧raw / STOP、凍結JSONは変更しない。

元のscalar・prepは公開済みStage 2 RESULTを固定commit `0a35101914c6833244b6e0bad7bc2767a49b5a47` で読み、実contextへ完全一致で照合する。Fullはc_H、p7はc_H + tail identity、signed pieceへ分配せずtotal Tに一回。保存入力receipt / snapshot bytesも照合する。

Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、symmetric_directional、各軸測定付きfull controlled Hadamard wrapper。元index0のlow/high-control、独立occupation作用、Hadamard両軸、compiled-vector両軸の7 gateと元許容差を保持する。新runの空cacheを使い、compiled-vectorが欠けるcache-only価格は受入しない。C guardとSOURCE-bound sidecarまで完了した価格だけ有効とする。

2 routeは別context / private / public namespace、6 cellを資源観測に応じてprocess並列化する。既存`accelerated_native`の同一signed部品再利用とcanonical v2高速fingerprintを、元trialの全区間へ適用する。各cell内でevolutionをX/Yに共有し、両軸を個別compileする。共有設定・他processに変更しない。

実行前からcontext / compose / fingerprint / wrapper / compile+archive / statevector gateをSTART/END/ERROR・所要時間・RSSで観測する。公開multiprocessing APIで各workerのexit code/signalを保存する。人工testはmetadata誤結合、実際のfork入口・元7 gate・高速化context・段階計測、異常終了時の部分保存 / 新投入停止 / retryなしを確認した。metadataとproviderを別processに分け、実科学前に検出したimport順問題を修正した。

## 入口・出力・STOP

```bash
python -B tests/check_pf_m0_prices.py
python -B scripts/run_pf_m0_prices.py --metadata-output <new-file.json>
# caller / tests / metadataをcommitで固定した後に一回だけ：
python -B scripts/run_pf_m0_prices.py --execute-approved-m0
```

route別出力は既存Full / p7 study配下の**新規**`m0_direct/run01`、privateも各routeの`m0_direct/run01`。既存directoryがあれば再実行不可。取得rawとimplementation proofをcommitし、そのcommitにsidecarを束縛して既存C guardへ渡す。今回の新規compile上限12、sampling / scientific retry / model予測 / fit / N/G=0。

取得・作用gate失敗は有効価格に昇格せずUNKNOWNとして部分raw/cache/progressを保存する。元gateを緩めず、片軸からの価格補完をしない。完了後commit/push・remote SHA/tree/主要成果物を照合してmandatory STOP。m1以降や別幅のmodel資格へ進まない。
