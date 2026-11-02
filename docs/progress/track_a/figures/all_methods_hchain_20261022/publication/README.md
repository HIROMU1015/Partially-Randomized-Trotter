# H-chain全方式総費用図：公開入口

2026-10-22 JST。ユーザーの「コミットプッシュして」に基づく公開。
ブランチ：`codex/pr-all-methods-hchain-figures-20261022`。

**[全図・データ・説明・検証記録を開く](../README.md)**

主図はε=0.05 / 0.005 / 0.0005。追加精度0.01 / 0.001と相対費用補助図も含め、
10図をPNG / SVG / PDFの30出力で収録した。全7系列の定義・欠測・条件は元packetのREADME・notesに記載する。

## 公開した範囲

- 図作成で保存・検証したpacket全89ファイルを同一bytesで収録。入力SOURCEの固定Git blobコピー、
  figure_data、生成script、caption、notes、provenance、表示・保全検証、STOPを含む。
- 上位READMEへ今回の公開入口リンクを1行追加。既存スライド構成案・図・主表・科学SOURCEは変更しない。
- [公開manifest](publication_manifest.json)に生成元packetの全SHA、公開基準commit、承認範囲を記録。
- [公開コピーの技術確認](verification.json)に転記、data、リンク、変更範囲と既存資料保全の結果を記録。

元packetのREADME・provenance・STOPにある「ローカル未commit・未公開」は**図生成を終了した時点の履歴**。
その記録のbytesとhashを保持し、今回の公開承認はこのpublication層へ追加した。
元preservation_checkは生成元checkoutでの確認であり、移設先checkoutに対する再判定として流用しない。
コミットとremote branchのSHAは今回のGit履歴および完了チャットから確認できる。

保存dataの照合は次で実施できる。再描画は行っていない。

```bash
python build_figures.py --check
```

作業対象は保存図の公開だけ。新科学計算・sampling・compile・fit・bootstrap・再会計・再ランキング・再最適化は0。
科学的な受入判定、最終evidence freeze、独立科学レビューへ進まない。
commit・pushとremote SHA照合後にmandatory STOP。追加の必須作業はない。
