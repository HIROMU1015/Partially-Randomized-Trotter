# 並列CPU/wallと費用の境界

parallel cost adapterはproducerの固定計測境界を読んで検査する。数値が実測真値であることを署名・OSログで独立attestするものではない。
host_outerがある場合、親CPU＋RUSAGE_CHILDREN差分の全worker CPUだけを同じ境界で加算する。coordinator snapshotとhost外側STOP込みを別表示する。
parent crash等で値が未保存ならnull/INSUFFICIENT_EVIDENCE。unknownを0にしない。

- Pair CPU/wallとcoarse・fine各partitionをown outer total/residualへ照合する。coarse＋fineは加算しない。
- Pair CPUはworker CPU内の内訳。全worker CPUへ再加算せず、差をworker未配分費用（喪失pair等を含む）として残す。
- Parent wall＋worker wallや複数pair wallを足してglobal wallと呼ばない。global wallは宣言されたhost外側wall。
- launchはparent CPUに内包、IPCはgenerator/save/outer等へ重なる診断座標、finalizationもparent内。いずれも再加算しない。
- X/Yは同じpair共有setupを一度だけ計上。Full prepareはconstructor/preflight内包。baseline/instrumentedと逐次/並列条件を分ける。
- generator/audit、angle/requery、evidence、serialize/hash/write/flush/finalize、memory/RSS/HWM、I/O、observer/residualを保存する。
- post-STOP worker CPUはproducerのtick量子化付き推定であり、resolutionとnullを保つ。
- statefile保持容量とpwrite累積trafficは別欄。RSS和は共有pageを含み、unique physical memoryや連続peakではない。

失敗prefixの費用も診断JSONへ保持するが、comparison_eligible=falseとし通常比較CSVには入れない。正常人工例だけを比較表へ出す。
Receipt自己書込時間の独立座標はunknown/null。既存outerへ含まれる処理を「無料」として追加0を置かない。

CachedObserverはprofiling bookkeeping cacheであり、law/angle/eventの科学cacheではない。旧observerとcall target/label/count/exclusive partitionを人工比較した。
実observer CPU時間の一致は要求しない。observer SOURCEが変わった費用を純observer overheadとして認定・差引きしない。

並列concurrent pair CPUを正式な科学主指標へ採択していない。逐次CPUは異なる実行条件であり、研究勝敗・一般優位・母平均confidence・量子費用換算を出さない。
[正常公開例](published_normal_example/cost_summary.json)と[独立人工例](independent_artificial_example/cost_summary.json)は、その入力の技術会計例である。
