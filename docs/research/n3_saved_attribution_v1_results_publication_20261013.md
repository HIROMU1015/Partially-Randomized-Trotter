# N3保存帰属v1結果：GitHub公開とremote保存照合

2026-10-13 JST。結果payload `4b12efc6dc7086cfb4838adeb561167f4d03e230`、branch `n3-saved-attribution-v1-execution-20261013`。認可commit `0974fa114c319599e70988cc802cafdfedff3ea5`、source `a7a9f42d53d5efaca8715bee26edd9f1a0bad999`。
一回/retry0の数値事後評価と既存3証拠再認証を完了し、[receipt](../../artifacts/n3_saved_attribution_v1/2026-10-13/results_publication/receipt.json)を保存した。

## 固定commitの入口

- [結果と限界](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b12efc6dc7086cfb4838adeb561167f4d03e230/docs/research/n3_saved_attribution_v1_results_20261013.md)
- [GPT科学判断引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b12efc6dc7086cfb4838adeb561167f4d03e230/docs/research/n3_saved_attribution_v1_results_review_handoff_20261013.md)
- [一次結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b12efc6dc7086cfb4838adeb561167f4d03e230/artifacts/n3_saved_attribution_v1/2026-10-13/evaluation1/result.json)
- [選択保存3証拠の再認証結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b12efc6dc7086cfb4838adeb561167f4d03e230/artifacts/n3_saved_attribution_v1/2026-10-13/evaluation1/selected_existing_proof_recheck.json)
- [独立保存scalar/timing監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b12efc6dc7086cfb4838adeb561167f4d03e230/artifacts/n3_saved_attribution_v1/2026-10-13/evaluation1_audit/independent_saved_scalar_timing_audit.json)
- [実行source・plan](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b12efc6dc7086cfb4838adeb561167f4d03e230/docs/research/n3_saved_attribution_v1_execution_plan_20261013.json)

## remote確認範囲

GitHubから専用branchを独立bare cacheへfetchした。local object alternatesとconfigured remoteは使っていない。cacheは過去からshallowだが、今回必要なtree/blobとsource commitの9blobはすべて読み出してSHA照合した。
85 blobs / 39,106,692 bytesをローカルと一致確認し、fresh exportへ保存。登録40files、source9files、凍結plan/registry/準備4filesのidentity-only preflight PASS。
取得した監査scriptを使い、保存6Hのscalar/timer/コピー値の独立照合と集計を再抽出し、archiveとbyte一致した。
remoteで `--execute`、元science runner、solver/eigh、追加PSD再認証は呼んでいない。選択3証拠の再認証結果はcommitted blobとして一致確認した。
これはGitHub公開物の保存照合で、immutable CIや独立外部環境でのscience再現ではない。

最初のmetadata exportでgitの日本語path引用表記により停止し、NUL区切りpath取得へ修正した。一次結果/source/科学認可/一回制限に変更なし。[処理記録](../../artifacts/n3_saved_attribution_v1/2026-10-13/results_publication/export_path_handling_recovery.json)。
全体manifestの既存description欠落FAILを保持、個別証拠と再取得照合PASSへ混同しない。

## 研究判断と停止

非負density、full5/6→active4/5、ν2/3、一外部mode、候補2/H、元ε_proj=.01、geometry/basis/DF/L_D/PF窓なしの元6入力の事後development。
全696保存候補ROOT活性、OCC/同値SDPの12same-mode最終q差、SOS banded6/ext5のe寄与、保存phase会計を取得した。
未保存proposal/rと反証専用timerの完全因果帰属/同機能同品質費用は未確定。追加H縮約0の元結論を維持する。
認可消費済み、次はGPTによる継続/縮小/技術記録化とclaim範囲の判断開始認可。中心テーマnull・次段未認可・Mandatory STOP、N1停止/N2保留。
