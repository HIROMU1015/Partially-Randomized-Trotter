# 出力・failure prefixの保全

元の `D.Stream` のcanonical encoder、chain/file SHA、unbuffered `xb`、16/axisのfull retention、以後compact、flush方針を使用する。各workerはphase/pass/arm/layerで一意なfileを持ち、親だけがsummary/control/statefileを書く。既存namespaceを拒否しappend/resumeしない。人工collision注入でも既存fileを上書きしない。

元の逐次R1の「完了pair bytesを次pairの残り上限へ反映」を、親のglobal ByteBudgetへ置換した。ReservedFileは原Stream.fileに被せるunbuffered proxyであり、write前に親からglobal result枠を取る。全worker・親summaryを64 GiBの同じ枠で管理する。固定per-worker配分はない。通常のshort writeでは未使用creditを同じfileで再利用し二重予約しない。不明writeのcreditは返却・他workerへ再配分しないので安全側に失敗し得る。reservedとACK observedを別記する。ACK不明時も未課金にしない。

logは256 MiBの同じglobal budget。24 taskのPLAN/start/end/failure/phase/receiptだけをJSONLへ記録し、trial状態は1 trial/byteのfixed-length fileに保存する。31,710,848 bytes = **30.24 MiB**。O(N) Python dictionaryや毎trialの巨大JSON journalを持たず、親のcurrent ticketsはO(workers)。statefile初期確保はlog枠へ一度課金し、上書きpwrite trafficはI/O費用として別計上する。この状態保存形式は旧R1へ無断追加できない契約差分。

retained log bytesの制限であり、累積pwrite trafficを256 MiBへ押し込む仕様ではない。並列policy・summary内の試行状態、追加marker/制御保存の範囲を新schemaで固定する必要がある。実scienceの最大failure evidenceを含む完全logサイズの保証は未接続であり、上限到達時はSTOPして保存不完全を報告する。

F01のcurrent attempt、last completed trial、in-flight event、candidate/committed trace、stream prefixをworker reportに保持する。sample結果前の失敗はcurrent eventをnullとし、前trial・別workerのeventを代入しない。flush失敗はstream完整行とengine commitが食い違い得るため両方を保存する。partial tailは完全chain/完了trialへ入れない。端数byteもraw archiveに残す。

成功時は全24 pairを固定identityで集約し、全state COMPLETED・trace・prefixを検査した後だけsummary/receiptを出す。receiptはsummary SHA/chainと全joinを記録する。summaryが記録する費用/output座標はsummary書込前のsnapshotで、返値のfinalization/outputとhost_outerはその後を含む。自己書込費用の循環を避けるため両者の観測範囲を区別する。失敗時に成功receiptを出さない。

全測定で各pairのchain/file SHAを再読照合する。異なる実行間では `generator_exclusive_ns` だけをtrial payloadから除いた正規化streamを比較し、event/evidence/angle・ID・分類・bit消費・retentionを維持する。raw file SHA/chainは時刻費用を含むため実行間一致を要求せず、各run内で原hashを照合する。raw summary/controlのPID/費用等も非決定metadataとして別管理する。

代表正常runと全fault runはraw tar.gz、全runの報告は圧縮JSONL、全pairの元SHA/chain/bytesと正規化SHAはruns.jsonに保存する。人工temporary scope/authority/SQLiteは測定後に削除し、原科学出力・grant・ledger・markerは開かず変更しない。主agent自身の検査であり別reviewerの独立レビューではない。
