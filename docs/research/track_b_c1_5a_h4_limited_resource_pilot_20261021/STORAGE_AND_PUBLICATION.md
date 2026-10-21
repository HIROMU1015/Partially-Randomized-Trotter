# 原データの保全・公開形式

科学jobは技術的STOP。公開処理のため、methodごとの元raw hierarchyをdeterministic tar.gzへまとめ、4 MiB以下の`.partNNN`に分割した。これは転送単位であり、CPU/RAMや科学callの上限ではない。

[index.json](raw_archives/index.json)には各archive、part、全memberのbytes/SHA256と元の論理pathがある。Ordinary229 files、Full246 files、P5部分資料38 filesの計513 filesを保存した。全memberを元file bytesと照合した後、元filesを専用runtimeの`raw_preserved/result/`へ保全した。working treeには小さいsetup/summary/terminal/progressの同一bytesコピーを直接残している。

アーカイブには完走済みbatchだけでなく、P5の保存済み途中artifactも含む。保存されていることはtrial完走や科学的受理を意味しない。未保存・欠測を補完せず、source/terminalの分類を維持する。

このnamespaceをcurrent directoryとして、methodごとに次の形で復元できる。

```bash
cat raw_archives/ordinary.tar.gz.part* | tar -xz
cat raw_archives/full_return.tar.gz.part* | tar -xz
cat raw_archives/closed_P5_full.tar.gz.part* | tar -xz
```

各partはsuffixの数値順に連結する。archive pathは`result/<method>/...`で、凍結契約が指定した論理pathを保持する。batchはgzipを展開するとJSONL、event/angleはexact F3 typed JSONが得られる。hex整数/有理数の解釈は固定`g6_f3_contract.py`に従う。

tar/gzipの梱包・member bytes検証・公開SHA照合は科学STOP後の技術処理。source law、seed、precision、保存済み数値は変更していない。追加sampling、H4演算、native取得、N/CI、再実行は0。

公開commit・新file bytesは`publication_receipt.json`、receipt自身を含む最終HEAD再照合は最終チャットと専用runtimeの外部receiptに記録する。
