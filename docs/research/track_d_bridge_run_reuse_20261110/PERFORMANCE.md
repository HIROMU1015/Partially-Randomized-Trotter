# 同条件性能比較と残る再走査

[SOURCE・対象](REPORT.md)。旧＝e6bd6c5のstream高速版、新＝run再利用版。固定native gzipのdecode＋engine＋最終T0を計時し、hash検証・import準備は計時前、結果検証／MPFR照合は計時後。peak RSSはimportや準備も含むprocess high-water mark。旧新16 fresh processの共通poolで一回計測、全体19.798秒。利用可能なCPU・十分な空きRAMとpressureのない状態を読取確認して投入した。共有設定や他jobに変更なし。内部ホスト・保存先・環境の具体値は公開しない。

raw読込／native lowering／全wrapper構築／P0の時間を含まない。前回のraw込みRSSと今回のnative-only RSSを直結して削減率を出さない。単回測定なので小さい差の統計保証はしない。

| Case | 旧秒 | 新秒 | 速度比 | 旧RSS MiB | 新RSS MiB |
|---|---:|---:|---:|---:|---:|
| q4 Reference X | 6.644 | 5.681 | 1.170 | 98.00 | 101.50 |
| q4 Reference Y | 6.641 | 5.757 | 1.154 | 95.00 | 98.50 |
| q4 Strong X | 6.447 | 5.456 | 1.182 | 96.00 | 99.00 |
| q4 Strong Y | 6.426 | 5.510 | 1.166 | 99.14 | 98.00 |
| q8 Reference X | 18.260 | 13.684 | 1.334 | 103.25 | 111.00 |
| q8 Reference Y | 18.261 | 13.799 | 1.323 | 103.26 | 108.50 |
| q8 Strong X | 18.115 | 13.677 | 1.324 | 103.12 | 107.26 |
| q8 Strong Y | 18.242 | 13.681 | 1.333 | 103.09 | 107.58 |

[正確な数値CSV](../../../artifacts/track_d_bridge_run_reuse/2026-11-10/PERFORMANCE.csv)。RSSはq4で−1.15%〜＋3.68%（増加を正）、q8で＋4.01〜7.51%。省メモリ化の追加達成とはしない。

## 何の再走査を省いたか

| 指標・X軸（Yも同傾向） | q4 Reference旧→新 | q8 Reference旧→新 | q8 Strong旧→新 |
|---|---:|---:|---:|
| 論理epoch | 10→10 | 20→20 | 20→20 |
| 旧1Q分類native訪問 | 631901 | 2164443 | 2189672 |
| 新native分類＋変更run内部訪問 | 274327＋255498 | 721977＋545681 | 750477＋541702 |
| 新unit metadata訪問（別の仕事） | 388833 | 1180011 | 1212248 |
| memoで省略したrun内命令数 | 294955 | 1306176 | 1303720 |
| CX feed | 474247→207211 | 1825588→792992 | 1853811→828591 |
| 実1Q synthesis | 2775→2775 | 2953→2957 | 1483→1487 |
| peak live node/unit | 60243→25859 | 127297→54273 | 128670→56030 |
| peak frontier参照 | 69794→35410 | 147250→74226 | 150207→77567 |
| peak cache slot（前＋現） | 2536→2472 | 2559→2495 | 1332→1225 |

[全8件の走査CSV](../../../artifacts/track_d_bridge_run_reuse/2026-11-10/SCAN_COUNTS.csv)。native分類はepochでopcodeを分類した訪問、変更run内部訪問は旧resolverへ展開した命令数である。新unit metadata訪問とは重複する仕事もあるので単純加算して「全Python参照数」としない。bit-keyや行列積内部の全参照を命令トレースした値ではない。native列のhash／最終展開／T0計数も必要である。

q8でnative分類＋変更run展開は旧分類訪問より約41%減り、CX feedは55.3〜56.6%減った。実合成は減っていないので、前回のcacheによる合成削減とは異なる効果を観測した。不変runはbit-key再生成も省ける。一方、全unit列は20回走査し、CX frontierも20回作り直す。一般worklist更新は0、全native反復が完全になくなったとは主張しない。

## 記憶とq依存性

q8 Reference Xのword/unit list容量peakは1,140,552→444,360 bytes、frontier list容量peakは1,252,880→645,968 bytes。ただしrun tupleは最大107,344 native参照を保持し、object／memo／cacheも増える。nodeの単位も命令からrunへ変わるためnode数だけでRAM削減とは判断できない。RSS増加の個別heap寄与は分解していない。全containerの同時合計容量と個別native object byte数は未計測、全体の実測はRSSが正本。

phaseの論理履歴は10／20 epoch分、各epoch wordそのものの履歴は保持しない。correctnessではword hashだけを記録した。frontierやrun保持量に固定小上限を設けず、合法な相殺候補を切り捨てない。

q4→q8で新時間は約5.5→13.7秒へ増え、native入力約2倍、epochは10→20。pathとcutも異なるためqだけの因果効果とはしない。O(L)記憶、最悪O(RL)時間は残る。raw QC／gate.definition cacheを含む入口も不変で、q1024〜4096の時間・RAM・一般P0資格は未確認。今回の改善率を長qへ保証しない。
