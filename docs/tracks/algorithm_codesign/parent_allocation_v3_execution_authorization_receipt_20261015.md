# Parent allocation v3 one-shot承認記録

利用者原文：

> 次の作業に進んで大丈夫　修正が終わったらその先もできるなら行って

利用者はv2のCPU技術停止に対する保存処理の限定修正と、その後可能であれば進むことを明示承認した。93登録外tests・人工bytes同一性/性能比較・preflight・remote取得が通り、科学条件/capsがv2と同一なので、固定S3/A3でv3を一回だけ実行する。v1/v2 marker/結果を保持し、retry0、終了後mandatory STOP。

固定S3：`53a36adf9bdd84690e9cdf3e1f97f38a178441dc`。source公開確認HEAD `7c0fcf368e76a1dacb1ed160be72ff0d6f72be27`を実行sourceとして使用しない。
本A3はS3の直接子で、変更はv3 authorization JSONと本receiptのみ。
contract SHA256：`adf2a75ecaadbd012f1ec20bb6b741979e5542614f151da6988d26938dfd82ad`。16 KiB保存修正以外の科学条件・capsはv2同一。
science execution authorized=true、runs1、retries0、mandatory STOP=true。
旧v1/v2のsource、contract、承認、結果、marker、STOPを不変に保持し、新v3 namespaceで一回だけ実行する。
部分失敗・上限・数値/guard失敗でもretryしない。終了後は保存値とprovenanceのread-only監査・公開のみ。
科学的GO/STOP、新規性、研究継続、追加実験、GPTレビュー開始を自動採択しない。
