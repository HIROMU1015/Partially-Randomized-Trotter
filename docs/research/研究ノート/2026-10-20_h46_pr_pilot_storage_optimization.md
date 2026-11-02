# 2026-10-20 H4/H6 PR pilot保存容量

基点`3e7365a606e18efa585252eef23f7b13264b0f38`から別branchで調査した。[報告書](../track_a_h46_pr_pilot_storage_optimization_20261020.md)を参照。

既存DET保存QPY/NPZ全276file（約10.36 GB）を別private領域へlossless圧縮し、全て原SHAへ復元できることを確認。圧縮後約0.635 GB。原本や契約は変更・削除していない。byte検証だけで科学値を解析していない。

33本案は維持。最大圧縮pair×4と16 MiB/trajectoryの未採択余裕を置くと、final約132.5 GB、各系1worker並行peak約137.0 GB、合計64workerのH6-heavy容量envelopeで約264.2 GB。現在空き約455.2 GBに収まる見込みだが、PR/rareサイズ上界ではない。9本案も別比較として保存する。

standalone stream archive/restore試作と人工18件。direct QPY dumpが通常とbyte一致する人工例も確認。旧科学runnerへ保存hookを接続していない。将来の新規raw一時fileを圧縮検証後に解放する扱いは新保存契約の採択が必要である。

次はユーザーが保存方式・nを判断し、Codexが新SOURCEでadapter/監査/復元とscratch予約を接続する。その後ユーザーがH4/H6各実行を別途明示承認する。n_E・正式shot政策・AD03は保留、今回の実科学計算0・mandatory STOP。
