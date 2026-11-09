# q8性能・q4との比較と限界

[固定科学条件](REPORT.md)。q4 phase4＋q8予測4を独立8 workerで処理し、全体wallは59.219072977秒。実空きCPU／RAM／loadを事前確認し、この8 taskに必要な並列度を採用した。共有設定・他processを変更せず、独自の小資源上限は設けていない。これはq8単独poolのwall timeではない。

## q8実測

単位：秒、RSSはMiB。worker全体はimport／原本I/O／native lowering／bridge／count-transfer／private保存を含む。最終JSON書込直前まで。sourceの時刻はperf_counter、RSSはLinux getrusageのprocess high-water mark。

| Wrapper | Load | Lowering | Bridge | Count/transfer | Worker全体 | Peak RSS | Native命令 | 反復 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Strong X | 0.194 | 19.023 | 33.524 | 1.631 | 55.207 | 491.238 | 429400 | 20 |
| Strong Y | 0.198 | 19.008 | 33.767 | 1.627 | 55.431 | 491.461 | 429405 | 20 |
| Reference X | 0.176 | 18.968 | 34.164 | 2.298 | 56.426 | 491.316 | 430996 | 20 |
| Reference Y | 0.174 | 19.059 | 34.297 | 2.294 | 56.634 | 494.023 | 431001 | 20 |

Bridge内の入力validation／countと出力countもBridge時間に含む。Count/transferは追加の独立検査であり、二つの時間を同じ計数として混ぜない。phase ledgerとfrontier観測のoverhead、保存raw QC・全native列・複数containerをRSSに含む。process peakの和を同時peak実測として扱わない。

## q4の旧価格予測との比較

Xを代表に表示するが、Yも個別記録に保存した。q4は旧4並列／旧binary64 phase集約、q8はphase ledger付きでq4位相調査と併走する8 task中の4件。private保存の内容も異なる。入力path／cutも異なり、qだけの因果効果や厳密speedup比ではない。

| 指標 | Strong q4→q8（比） | Reference q4→q8（比） |
|---|---|---|
| Native命令数 | 200882.000 → 429400.000 (2.138×) | 203178.000 → 430996.000 (2.121×) |
| Lowering秒 | 10.235 → 19.023 (1.859×) | 10.366 → 18.968 (1.830×) |
| Bridge秒 | 9.816 → 33.524 (3.415×) | 9.978 → 34.164 (3.424×) |
| Count/transfer秒 | 0.812 → 1.631 (2.010×) | 1.143 → 2.298 (2.010×) |
| Worker秒 | 22.508 → 55.207 (2.453×) | 23.190 → 56.426 (2.433×) |
| Peak RSS MiB | 278.555 → 491.238 (1.764×) | 280.805 → 491.316 (1.750×) |
| 反復数 | 10.000 → 20.000 (2.000×) | 10.000 → 20.000 (2.000×) |

## Frontierと保持量

| Wrapper | 最大live node | 最大retained slot | 最大frontier参照数 | 最大1wire stack |
|---|---:|---:|---:|---:|
| Strong X | 128670 | 129488 | 150207 | 22725 |
| Strong Y | 128670 | 129488 | 150207 | 22725 |
| Reference X | 127297 | 128123 | 147250 | 11407 |
| Reference Y | 127297 | 128123 | 147250 | 11407 |

frontier参照数はwire stackの参照和で、CXを二wireで数える。retained slotはtombstoneも含む。最大word containerはnative入力約43万命令で、全container合計ではない。RSSがQC／native列／一時listを含む実コストを捉える。固定小frontierを仮定しない。

q4 phase再検査は旧wordとの同一性確認のため追加listとI/Oを持つ。各phase workerのpeak RSSは 432.500, 432.566, 426.000, 427.027 MiBで、旧q4価格予測の記憶量を置き換える値ではない。

## ボトルネックと長q

q8の最大stageはBridge約33.5〜34.3秒、lowering約19秒。q4の約9.8〜10秒からBridgeが約3.4倍になり、反復は10→20。1Q再合成とCX相殺が互いに次の簡約を露出し、whole wordを繰り返し走査するコストが増えた。全qで反復がq比例だと認定する証拠ではない。

旧ruleはO(RL)時間・O(L)記憶、最悪O(L²)。T0直接計数はO(L)時間/O(W)記憶、transfer captureはO(LW)/O(W²)。frontier観測には各gate後にO(W)の参照数集計を追加する。固定H4幅での観測を、幅も変わる一般問題の無overhead性能へ移送しない。phase ledgerは固定160-bitの加算と局所phase項の保存を行い、局所合成精度は元binary64。

新whole-wrapper QC／full P0 DAG／CAR target／全入力simulationは作っていないが、保存raw QCと全native列は保持する。q1024〜4096には、繰返し再走査・可変履歴・raw QC保持・逆順取得・局所数値誤差・P0 domain保証の課題が残る。長q benchmark／線形外挿／新P0とのspeedup実測は0。既存A/F/Sを今回の構造価格で置き換えない。
