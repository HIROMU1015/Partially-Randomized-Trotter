# 固定S2比較の実行記録

branch `codex/pr-pf-s2-recomputed-theory-compiled-20261022`。実行単位は `units/system/candidate/stage`。全単位one-shot、失敗後retryなし。SOURCE/入力/契約/承認receipt、各grant/marker/result/STOPを保存した。

結果の入口は `docs/progress/track_a/figures/pr_pf_s2_recomputed_20261022/README.md`。context.pklは元bytesを保持したままcontext.pkl.gzへ可逆圧縮した。Gitには圧縮版を保存し、`context_compression_manifest.json`の元SHA-256で復元後を照合できる。`portable_input_map.json`に入力の相対パスを記録した。

SOURCE runnerを消費済みgrantで再実行しない。図の再生成は保存JSONだけを読む `scripts/build_recomputed_s2_figures.py`、検算は `scripts/verify_recomputed_s2_packet.py`。原科学結果を再評価しない。失敗した追加G診断をPASSへ変更せず、詳細科学レビューへ自動進行しない。
