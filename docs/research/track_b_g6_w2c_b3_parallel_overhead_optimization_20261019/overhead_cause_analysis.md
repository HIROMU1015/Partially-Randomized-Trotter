# 管理CPU費用の原因分析

最終SOURCE `77f8667fc0898a3615a4efa834d971b3c339df71`。全functionを記録する測定同士を主比較とし、OFFは別の診断。各値は秒・3反復中央値。

| 負荷 | S0旧observer | S0 cached | P1共通Counter | observerだけ改善 | IPCだけ改善 | 両方改善 |
|---|---:|---:|---:|---:|---:|---:|
| light | 11.9682 | 3.9725 | 18.3768 | 7.1153 | 16.4110 | 6.1490 |
| mixed | 6.6512 | 2.5942 | 10.1431 | 4.3295 | 9.1472 | 3.7741 |
| heavy | 6.4287 | 4.2296 | 8.1445 | 5.2757 | 7.7019 | 4.9703 |

旧P1のCPU増には、科学操作自体より、管理pathまで観測するfine callbackと多量のIPCが強く寄与した。observerだけでCPUを 61.3%/57.3%/35.2%、IPCだけで 10.7%/9.8%/5.4% 削減。両方の効果には相互作用があるため、この2つを足して全体削減率にはしない。

Observer改善はPath/label・contextmanager・row dict生成・不要locals参照を削減し、同じ計測範囲を安く実装したもの。原observerと同じcall/return選択を確認し、外側の総CPUにも全callback費用を残した。scopeをtimer外へ移した改善ではない。

| 負荷 | S0 OFF CPU | 旧P1 OFF CPU | 改善P1 OFF CPU | 状態更新span CPU | 状態更新回数 |
|---|---:|---:|---:|---:|---:|
| light | 1.3540 | 3.3432 | 2.7797 | 0.0867 | 17408 |
| mixed | 1.1133 | 2.1798 | 1.8591 | 0.0485 | 9216 |
| heavy | 3.4008 | 4.1698 | 3.9372 | 0.0282 | 5120 |

状態pwriteのspanは約0.03–0.09 CPU秒で、旧P1の全CPU増を説明する主因ではなかった。診断timer自体の費用を含む外側CPUと、spanの値は異なる。状態保存は原方式を残した。OFF条件のCPU差を研究用測定の改善値として使わず、主表の保持observer条件を採用する。

| 負荷 | 旧P1 message数 | 改善P1 message数 | 削減 | 原状態更新bytes |
|---|---:|---:|---:|---:|
| light | 21784 | 13128 | 39.7% | 17408 |
| mixed | 11544 | 6984 | 39.5% | 9216 |
| heavy | 6424 | 3912 | 39.1% | 5120 |

通常5往復/trialを3往復＋axis末confirmへ削減。RESERVE/ENTERと正確なwrite前byte予約は省略しない。実際のwire RPC CPU/wall、parent CPU、worker CPU、launch、coarse/fineは全reportへ保存する。IPC座標は既にworker totalに含まれ、追加加算しない。

重要な対照：同じcached observerを使う逐次S0と比べると、並列管理の追加CPUはまだ残る。改善P1のCPU倍率はlight/mixed/heavyで 1.55/1.45/1.18倍。元S0よりCPUが減る結果を、並列化そのものが演算量を減らした証拠とは解釈しない。

第2案の6回probeではlightのCPU中央値 6.8791→6.0776秒。採用後の主結果はこのprobeを流用せず最終SOURCEで再測定した。
