# 文献更新の範囲・取得限界

調査基準日2026-10-24。旧調査501b4232…の39文献、取得済みPDF/text、metadata/bib、読取記録を再利用した。
今回の中心に直接対応するP01/P03/P08/P10/P21/P25/P26/P27/P32/P33の10件に限定して関連箇所を照合した。
取得済みPDFのSHAを旧catalogの取得記録と照合し、原cacheは変更・複製・公開していない。

## 更新検索

2026-10-21以後の更新を探すため、arXivを中心に以下のクエリ群を使った。

- `"partially randomized" "2026" after:2026-10-21 before:2026-10-25`
- `"Hamiltonian simulation" "product formulas" after:2026-10-21 before:2026-10-25`
- `"randomization" "higher-order" quantum chemistry 2026`
- `site:arxiv.org "partially randomized" "2026"`
- `site:arxiv.org "higher-order" "randomized" "Oct" "2026"`
- `site:arxiv.org "quantum chemistry" "product formulas" "2026"`

狭い日付指定では新たに採用する直接関連文献を確認できなかった。広めの補助検索では既登録P27等と、非関連の最適化理論・部分ランダム測定も返り、後者は追加しなかった。
これは検索索引・クエリに依存する限定的な結果で、10月21–24日の関連文献不存在や網羅性の証明ではない。
必要性のないcitation graph全探索は行わず、今回の比較に直接必要な本文読取を更新した。

P01/P25/P27/P32のarXiv landingを再確認し、使用版v2/v1/v1/v1を維持した。新しい雑誌版・全改訂の不存在までは主張しない。
P01 HTML v2は404だったためPDF v2と旧抽出textを使用。
他の対象arXiv HTML/PDFでは関連節を確認した。P27は今回、III.5の群内ordering randomizationまで確認範囲を広げた。

## 読取と未読の区別

P01のApp.A3/D1/E、P03 §5、P08 §3.3、P21 §IV、P25 §V.2、P26 II B4、P32 S4.2/S4.3、P33 III D等の関連箇所を確認した。
全定理の証明・全付録・公開codeの完全な再現監査はしていない。
P25 Eq.29のproxyと本研究のsignal誤差、P32のaccess modelと本研究のDF primitive、P33のenergy精度とε_signalを等置しない。

Yoshida/Suzukiの元出版社DOIは403で本文未取得。確認できた基礎構成の本文根拠はP21 §II、正式書誌は[参考文献](references.md)。
本文が未取得の箇所について、同等手法・対照が存在しないという結論を出していない。
今回の作業はCodexによる文献比較資料の準備であり、詳細GPT/Claude科学レビューは未開始。
