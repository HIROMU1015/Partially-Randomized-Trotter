# 高次PRを含むH-chain比較図：公開入口

2026-10-22 JST。ユーザーの「コミットプッシュして」に基づく公開。
ブランチ：`codex/pr-higher-pr-comparison-figures-20261022`。

**[全図・データ・説明・検証記録を開く](../README.md)**

52図をPNG / SVG / PDFの156出力で収録。精度はε=0.05 / 0.01 / 0.005 / 0.001 / 0.0005。
図群Aは10図、Bは20図、Cは20図、Dは2図。方式・候補域・欠測・証拠水準は元packetのREADME・caption・notes・監査表に記載する。

## 比較範囲

- A：従来の全方式図とouter32 PR比較を別panelに配置。旧PFと高次PRの会計条件の同一性が確定していないため、8系列を直接重ねた図や両者の費用比は作成していない。相対図の左右panelは基準が異なる。
- B：H2/H4/H6のPR S2 / Yoshida4 / m5_best。通常のsigned評価とnoncancellation評価を分離。
- C：H4/H6の同じnative条件で保存されたm5決定論的PF部分集合との比較。noncancellation版のPF基準はfull m5のみ。旧図の強化S2/S4 PF全体に対する優位性を意味しない。
- D：保存結果の比較可能性と独立確認の準備状況。H8高次PRは欠測。独立確認未完了を完了扱いしない。

## 公開した範囲

生成済みpacket全276ファイルを同一bytesで収録。固定SOURCEコピー、figure_data、生成script、
caption、notes、provenance、表示・保全検証、STOPを含む。既存10図・既存スライド・主表・科学SOURCEは保全し、
上位READMEへ公開入口リンクを追加した。

- [公開manifest](publication_manifest.json)：全276ファイルのSHA、基準commit、承認範囲。
- [公開コピーの技術確認](verification.json)：転記、保存data、リンク、変更範囲、既存資料保全。

元packetのREADME・provenance・STOPにある「ローカル未commit・未公開」および公開停止は、**図生成を終了した時点の履歴**。
記録のbytesとhashを保持し、その後の今回の公開承認をpublication層へ追加した。
元preservation_checkは生成元checkoutでの確認であり、移設先checkoutの保全判定として流用しない。
commitとremote branchのSHAはGit履歴および完了チャットから確認できる。

保存dataの照合は、生成時と同じPython依存関係で次を実施できる。公開時の再描画は行っていない。

```bash
python build_figures.py --check
```

新科学計算・sampling・compile・fit・bootstrap・再会計・再ランキング・再最適化は0。
独立確認や新しい比較の成立判定は実行していない。
commit・pushとremote SHA照合後にmandatory STOP。追加の必須作業はない。
