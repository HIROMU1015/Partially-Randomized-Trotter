# D2共通境界相殺：一回限りの実行承認

2026-10-13 JST。利用者の原文：

> 次の作業に進んで大丈夫です

直前のCodex報告は、固定source `28fc3ca17ec51bb7656a498e3eea9b72966eb3fa`のD2準備完了と、そのsourceへの別途明示one-shot実行承認を次の作業として指定していた。この文脈で上記指示をD2一回実行の承認として記録する。利用者がsource SHAや個別条件を上記文中で再引用したとは表現しない。

Sの直接子authorization-only Aを専用実行branch/worktreeで作成・push後、固定contract/source/runtimeを変更せず外側observerを一回呼ぶ。r=2の六方式、positive ordered pair総219,303、旧law/N/bias/priceとzero-fill方針を保持。native compile、新合成、新angle、matrix、sampling、trajectory、LP、DF、NPZ、GPUは0。

runs=1、retry=0。exclusive新markerを消費した後はtechnical failureでも再実行しない。全outcomeでmandatory STOP。旧source/results/markers/authorization/STOPを変更せず、追加条件や次stageへの認可はない。結果と保存値・provenance監査に必要な資料のみcommit・pushする。

本Aで変更するpathは `artifacts/track_b_d2_boundary_source/2026-10-13/v1/authorization.json` と、このreceiptだけ。旧D1契約や新D2契約のsource preparation表示は履歴metadataとして維持し、別Aによって今回の限定実行を許可する。
