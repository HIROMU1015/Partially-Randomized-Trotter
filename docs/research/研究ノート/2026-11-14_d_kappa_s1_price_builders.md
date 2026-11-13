# 2026-11-14 D：Full-H κ2 S-1/P-1準備とTruncated・Pauliの価格接続

S-1はR448と同じC数値SOURCEとreference readerを、byte照合したうえで使う。
固定κ2候補（26d4f23e）を再生成して照合し、logB≤(λT)²/s＋1 ULPのpost-gateを加えたcallerを `04bead11` に固定した。
toy Context・人工normalizerで、13 round・packet・STOP経路を確認した。実κ2の値は計算していない。

P-1の人工loaderで、元Full-H `setup()` がprovider root専用のmoduleを必要とすることが判明した。
そこで実行を子processへ分け、元の `install()` を最初に呼ぶ構成にした。
toy DFで、新旧ZZの全制御行列の一致、oracle、X/Y、scalarの一回会計、one-bodyの扱いを確認した。

Truncated p7は新しいbuilderを作らず、Full PFのlong-time native pathにB0I・p7のrowを渡す。
保持block、B0I tail identity、負時間pieceをtoyで確認し、T3の18 wrapperは候補として保持した。
PauliはA P0辞書と全round packetをSHA照合して読み、D2 tailとnative DF headを接続した。C guardのPauli routeは未登録。

人工・metadata 30 PASS。実H4 mean・sampling・build・compile・価格・N・Gは0。
[REPORT](../d_kappa_s1_price_builders_20261114/REPORT.md)。公開照合後mandatory STOP。
