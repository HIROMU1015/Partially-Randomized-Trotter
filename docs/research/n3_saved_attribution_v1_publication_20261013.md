# N3保存帰属v1：公開・取得確認・数値事後評価の認可待ち

2026-10-13 JST。完了済みpilot科学レビューの保存資料整理・source準備・合成検証・公開を一式完了した。
**次は固定planによる新しい数値事後評価と選択3証拠の再認証の明示認可。Mandatory STOP。**

## 固定取得資料

Repository：`HIROMU1015/Partially-Randomized-Trotter`。
branch：`n3-saved-attribution-v1-preparation-20261013`。
payload：`08ffe034ff693fa9797db2e9607ce9b876d0454f`。
posthoc source：`a7a9f42d53d5efaca8715bee26edd9f1a0bad999`。

- [準備内容と停止理由](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/08ffe034ff693fa9797db2e9607ce9b876d0454f/docs/research/n3_saved_attribution_v1_preparation_20261013.md)
- [出力義務・timer・未保存情報](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/08ffe034ff693fa9797db2e9607ce9b876d0454f/docs/research/n3_saved_attribution_v1_timer_and_output_map_20261013.md)
- [研究主張と観測可能性](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/08ffe034ff693fa9797db2e9607ce9b876d0454f/docs/research/n3_saved_attribution_v1_claim_and_observability_20261013.md)
- [新数値事後評価の限定scope](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/08ffe034ff693fa9797db2e9607ce9b876d0454f/docs/research/n3_saved_attribution_v1_posthoc_contract_20261013.md)、[固定plan JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/08ffe034ff693fa9797db2e9607ce9b876d0454f/docs/research/n3_saved_attribution_v1_execution_plan_20261013.json)
- [18 inaccurateの元pointer・保存status](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/08ffe034ff693fa9797db2e9607ce9b876d0454f/artifacts/n3_saved_attribution_v1/2026-10-13/preparation/inaccurate_event_pointers.json)
- [既存3証拠の完全なliteral・pointer接続](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/08ffe034ff693fa9797db2e9607ce9b876d0454f/artifacts/n3_saved_attribution_v1/2026-10-13/preparation/SOS_OCC_existing_proof_connection.json)
- [全216path・432event・696候補の索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/08ffe034ff693fa9797db2e9607ce9b876d0454f/artifacts/n3_saved_attribution_v1/2026-10-13/preparation/saved_literal_index.json)
- [保存40filesのSHA registry](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/08ffe034ff693fa9797db2e9607ce9b876d0454f/artifacts/n3_saved_attribution_v1/2026-10-13/preparation/saved_input_registry.json)
- [受領した完了済みGPT科学レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/08ffe034ff693fa9797db2e9607ce9b876d0454f/artifacts/n3_saved_attribution_v1/2026-10-13/preparation/review_intake.md)
- [posthoc module](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a7a9f42d53d5efaca8715bee26edd9f1a0bad999/src/trottertracks/representation_exploration/n3_saved_attribution_v1.py)、[runner](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a7a9f42d53d5efaca8715bee26edd9f1a0bad999/scripts/run_n3_saved_attribution_v1.py)

source fingerprint：`2ef6f58a0a3ac560b2e1aa2212dc7ee44a53bbb2cf7babf170b4dc11574d2d46`。
registry SHA：`d23d165ffebf1bb5bdfa542841cd678becc78dcae9b9e36bcd19733cd869023c`。
plan SHA：`a5f3fd39e47372e277365d9873f234cbd882ae1a84b6ca25a4f051e339f0c61f`。

## GitHubからの取得検証

別bare cacheへGitHubからfetchし、local alternatesを使わず新exportへ展開。
[81 blob・34,729,649 bytesの台帳](../../artifacts/n3_saved_attribution_v1/2026-10-13/publication/retrieval_inventory.json)がlocalと一致。
共有依存を含む9 source blobをsource commitと照合。浅い履歴の境界は残るが、必要なpayload/source blobを取得できた。
[再取得sourceのpreflight](../../artifacts/n3_saved_attribution_v1/2026-10-13/publication/remote_preflight.json)はidentity PASSで、40保存filesのSHAを検査した。
exportにGit metadataがないため、commit/blob関係は別途bare cacheで照合した。

[literal再抽出](../../artifacts/n3_saved_attribution_v1/2026-10-13/publication/remote_literal_copy_audit.json)は索引・inaccurate pointers・既存3証拠bundleの3filesがbyte一致。
新算術・PSD再認証関数に拒否guardを入れて確認し、NumPy/SciPy/CVXPY/SCSをimportしていない。
登録データの新数値事後評価、PSD再認証、新solver/producerは0。synthetic23の再実行や第三者科学再現・CIとしては扱わない。
[publication receipt](../../artifacts/n3_saved_attribution_v1/2026-10-13/publication/receipt.json)へ固定状態を保存した。

## 現在の明示認可地点

レビュー第12.6節は「保存結果の新たな数値的事後評価、証拠の再認証等を実行する場合は、個別の運用指示に応じた明示的な作業・実行承認を得る」と指定している。
準備・read-only literal抽出・source確認・synthetic・公開を終えたため、次は固定scopeの保存数値事後評価一回を実行する新指示を待つ。
対象は元の非負density6H、full5/6・ν2/3、一外部mode、候補2mode/H、元ε=.01。geometry/basis/DF/L_D/PF窓なし。
216保存path・696候補のphase/q/box/SOS残差整理と、同mode外部4のSOS正/OCC負、同H外部0のOCC正という既存3証拠の再認証まで。
2workers/thread1、Python3.12.3、AS1GiB/worker、aggregate RSS6GiB、input64MiB/output32MiB、wall/CPU capなし、attempt1/retry0。
元run1再実行、新κ/ε/solver/入力/証拠探索を含めない。元の消費済み認可を流用しない。
実行認可後も新事後結果と監査公開後にMandatory STOPし、重要な研究価値・次task・中心claimの科学判断をGPTへ戻す。
N1停止、N2保留、旧primary/gateと中心テーマnullを保持する。既存global manifest description欠落FAILは未解消のまま別記する。
