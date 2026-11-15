# 稼働中の取得を保持した追加高速化

利用者の「できそうなら行って」に基づき、追加改善を調べた。
実行SOURCE9cbabc617の3 pathとX/Y並列は実稼働を確認済み。次の逐次部分はnative状態vector検査と回路走査。
固定nativeはQiskit Statevector.evolveを使い、m5 path0・m6 path0は短時間実測で各約1 CPU core。
現在の正常workerへbackend差替え・SOURCE変更・再起動を行わず、取得済み/進行中の計算を維持する。
CPU全体の空きだけを根拠に科学workerを再実行しない。

今回適用する追加策は公開待ち時間の削減。

1. 完了cellの公開準備を他cellの取得と重ねる。`publish_full_h_d7_holdout.py --stage-when-finished` は
   既存parentの元time終了記録と固定SOURCE/jobを確認してから、元receipt/compile記録をコピーする。
   科学jobの起動/停止/再開は行わない。終了済みなのにreceiptがなければ原因を保持してSTOP。
2. 既存 `READ_RECEIPT.archive_result` に任意のparallel_gzipを追加し、導入済み `/usr/bin/pigz` を使用。
   現affinityの64 CPUを使用し、元bytes/hash・gzip復号一致・64MiB以下parts・公開reader互換を維持。
   pigzがなければ従来Python gzipを使用する。圧縮途中の失敗はPASSにしない。
   元較正のraw/gzip/manifestと固定公開commitは変更しない。
3. 公開SHAからの2 cellのraw復号・元record/個別軸証拠の照合を2 processで並列化し、
   結果をcell順に集約してから従来どおり4行のexact比較を行う。予測再計算・fit・モデル判定なし。

追加人工3件：元bytes/SHA/gzip復号/part順、2 processのbarrier同時実行と4 exact行一致、
完了cellのbytes保存とreceipt欠落STOP。初回は人工recordのcompile_receipt項目不足を検出しfixtureのみ修正した。
[PUBLICATION_PERFORMANCE_TESTS.txt](PUBLICATION_PERFORMANCE_TESTS.txt)。約16MiBの人工bytesでparallel圧縮は約0.05秒、
hash/分割/復号確認込み約0.07秒。実H4の速度倍率・全体所要短縮はまだ測定していない。

変更は元D1公開worktreeだけにcommitし、実行checkoutのSOURCE9cbabc617・runtime・科学入力は保持する。
実行SOURCEと公開後処理SOURCEを区別して最終REPORTへ残す。引渡しの科学的範囲・最大6 compile・STOPは変更しない。
