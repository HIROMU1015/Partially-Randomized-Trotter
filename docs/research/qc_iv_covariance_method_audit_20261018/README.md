# 研究④：DF共分散縮約候補の紙上方法差検証

2026-10-18 JST。**技術判定：`REDUCES_TO_KNOWN_METHOD`。指定された候補の範囲では、具体的な独立差は残らない。**

固定粒子数・DF構造を保ち、既知orbital-rotation shadowsの全energy共分散を展開前に縮約する処理は、Koizumiほか2606.30601v2 S8の既知class式へDF因子を代入し、通常の有限和の縮約・streaming・cacheを使うことで再現できる。同じ情報・同じ最適化を許した対照を[演算手順として具体化](method_comparison.md)した。著者CLIがDF全energyのAPIを提供している、実装済み・実測済み、という判断ではない。

| 成果物 | 内容 |
|---|---|
| [方法差検証報告](method_comparison.md) | 規約、normal-order、S8/SOURCEとの対応、既知法の具体手順、計算量・memory・相関情報 |
| [技術判定](technical_verdict.json) | 三分類中の判定、確信できる範囲、未確認事項、実行・採択境界 |
| [GPT・ユーザー向け引継ぎ](handoff.md) | この候補の探索を止める範囲、残る研究判断、次の担当 |
| [証拠索引](evidence_index.json) | 基点・必須資料・論文版・著者SOURCE固定commit、bytes/SHA、式・関数 |
| [静的・保全確認](verification.json) | 文書・JSON・リンク・identity・旧科学証拠の確認。数値検証ではない |
| [公開照合](publication_receipt.json) | remote HEAD・GitHub公開bytes/SHA256、旧成果保全 |

基点は調査最終commit `ab0c4d0ff0cd204ca7897459843da511109ad172`。著者SOURCEは `koizumiyuki/reprod_ORshadow_covariances@206aa441d6a08c1984975fddf14a23ccf054f370`。原論文・SOURCEは今回改めて取得し、前回索引とbytes/SHA256が一致した。重要GPT科学レビューや新数学保証の探索・証明は行わず、既知式の適用と許可された紙上の方法差・計算量比較に限定した。

同じ測定ensemble・estimatorを用いるので、縮約を速くしても測定の分散・shots・量子gate数は変わらない。古典費用の実装上の改善や将来の別推定法の可能性を一般に否定する結果でもない。新規性・研究GO/NO-GO・④全体の終了は決定しない。

**還元を確認した時点でこの候補の探索を終了し、実装・人工テスト・科学実行へ進まない。公開・保全確認後mandatory STOP。** A2現経路追加投資見送り、A2 `NOT_ADOPTED`、A1 `CLOSED`、K1-F未承認、他のSTOP/CLOSED/PAUSED/HOLDを維持する。
