# P5R技術修正と実行前条件

科学的sampling lawの変更なしに、旧fork poolを**1 sampler producer → durable Evidence → fresh exec logical providers**へ分離した。旧science SOURCEは編集しない。[bridge.py](bridge.py)は旧diagnostic/supportと固定P5Closedをそのまま使う。変更はこのnamespaceの追加ファイルのみ。

- ProducerはP5Closedのnormalizer/completion・104,977群を通常通り一度構築し、全4096 eventとangleを生成する。巨大stateは他processにfork継承もpickle転送もしない。表はproducerに一つだけ保持し、終了後に解放する。
- Group drawだけ、既存dyadic lawから作った累積整数表とbisect_rightに置換する。[selector.py](selector.py)はP5Closedを継承し、sampleの最初のdraw定義以外のASTが旧コードと一致することを検査する。completion、normalizer、event、child law、実proposal/weightは既存実装を呼ぶ。
- 既存dyadic_indexのcountはq_i 2^Hという整数で、累積端点e_iを一度保存する。bが入るhalf-open区間は同一、零mass端点はbisect_rightで飛ばす。これは既知のselector最適化であり新しい科学的アルゴリズム・定理ではない。全群を構築するsetup費用・memoryは引き続き払う。
- Producer単独所有は巨大mutable Python objectのCOW/複製とglobal monkeypatchを避ける設計選択。sampling/angle cacheを一つにまとめる。並列P5アルゴリズムは導入しない。重いlogical circuit部分は32固定batchを利用可能affinity内で並列execする。32はbatch数、64利用可能CPUに対して恣意的な低い上限ではない。producerとproviderは同時に走らない。
- `subprocess.Popen(start_new_session=True)`でfresh Pythonをexecし、科学module初期化済parentのstateを継承して計算しない。OSによる内部spawn実装のfork/vfork有無と、旧Python poolの継承state利用とは区別する。providerはbasis復元だけ行い、samplerを再構築しない。
- 小規模8qubit以下のbasis行列処理に各processが大量BLAS threadを作る費用を避けるため、今回の自分のchild環境だけBLAS/OMP/MKL threads=1。共有venv/cache/OS/cgroupを変更しない。32 process並列と合わせて記録し、旧実行とのruntime条件差を隠さない。

[supervisor.py](supervisor.py)はPID、task ID、UTC/monotonic start/exit、wait4のexit/signal・CPU/maxRSS、定期RSS/PSS/private/shared、cgroup memory.events前後差とfirst failureを記録する。native crash用faulthandlerは各job childで有効。原stdout/stderrは保持する。最初のfailureで新task dispatchを止め、自分が作ったactive process groupだけ終了・回収する。cleanupのTERM猶予5秒後は自分の残存childのみKILLする。科学時間/RAM上限ではない。host PSI上昇だけでは正常workerをkillしない。

各trialをpartial JSONLへflush/fsyncしてからCOMMITTEDを記録する。batch完了はdeterministic gzip roundtrip後。signalでも直前までのdurable prefixを残し、未完了をzero/acceptedにしない。markerはopen(x)で二度目を拒否、再submit/再試行なし。[job.py](job.py)のproviderはpath/event/angle SHA256、exact proposal×weight、bitstream再生成、word/phase/order、ratio、scalar/model/label/contract bindingを検査してから旧build_tailを呼ぶ。pathの意味論は旧schemaを保持する。

人工テストはH4 loaderを呼ばず、2/3 label有理入力と2qubit basisのみ。exit7・SIGTERM・launch failure、partial保全・marker拒否、全4096 bit整数のselector一致、全toy群CDF境界、128 toy eventのexact一致、順序を逆にした旧C1_5A_BITS_V1、8 toy pathの旧glue Evidence一致と直接小行列controlled operator照合を含む。[technical_tests.json](technical_tests.json)と原testログを参照。Run02の人工decoderで二重_unpackを検出し、科学実行前に修正した。Run03で14件PASS。その後のsetup観測guard・supervisor時間fieldを含む最終SOURCEのRun04を正式preflightとする。これらは本番安全性・科学的優位・canonical certificateの証明ではない。

科学固定値は[execution_contract.json](execution_contract.json)、依存固定はSOURCE/input/runtime manifests。開始前後・各child前後にSHA256照合。新authorizationだけをSOURCEの直接子commitにし、新markerを消費する。旧grant・markerは読み取りのみ。

比較上の交絡：旧Ordinary/Fullは21 fork worker＋3方法同時負荷、新P5はsolo・one cache・prefix selector・32 fresh provider・durable fsync/再読込。CPU/wallを同負荷・同最適化baselineの比較にしない。logical計数は同じ旧関数。未取得native/halves/stateprep/readout/NはUNKNOWN。
