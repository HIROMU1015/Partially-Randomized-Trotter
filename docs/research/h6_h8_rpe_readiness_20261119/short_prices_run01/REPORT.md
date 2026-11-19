# H₆ DF19 Full PF／M5の短ラウンド実compiled価格

2026-11-20 JST、サーバー2。許可されたm0〜m2のX/Yを6件compileし、6軸の直接価格を取り込んだ。保存価格の再利用は今回の6軸では0件。通常の探索CLIで、元の8軸と今回の6軸を同じ候補へ結合した。残る価格待ちはm7〜m12の12軸。

| ラウンド | T・実ステップ幅 | qPF | physical ID | X RZ | Y RZ |
| --- | ---: | ---: | --- | ---: | ---: |
| m0 | 0.1 | 1 | `dec32b474f3fc33ddfab` | 406,671 | 406,671 |
| m1 | 0.2 | 1 | `7791db0dc2fab94ce734` | 408,590 | 408,590 |
| m2 | 0.4 | 1 | `e7200ae9e16ba879a78d` | 416,266 | 416,266 |

直線H₆、1.00 Å、STO-3G、保存tol-only DF19、Full PF、M5の符号付き分解、要求幅1.6、h=0.1。Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17。保存順の全DF fragment、一体項、総時間のscalar位相、symmetric_directional controlを使う。単価はX/Y別にcompileした測定付きfull Hadamard wrapper全体のRZ数。状態準備コストは除外する。

実行SOURCEは `e37339fcf0470afe19169c5c62b2d118fb9a92dd`、価格の公開SOURCEは `a427e0b858f5b766b2f5ba1cf7ad14af1249d370`。前回の照合以降の関連refと、このホストで稼働する対象ジョブを再確認してから起動した。入力ハッシュ、Full19 preparation、元候補・T・q・軸、6個の互いに異なる回路指紋、同一compiler hash、元軸receiptの一致と終了コード0を確認した。今回のcallerは保存native経路を再利用し、新しい独立statevector検証は実施していない。 [価格原本](AXIS_PRICES.json) · [入力拘束](INPUTS.json)

3つのnative構築と6つのX/Yの処理は独立に並列実行した。全体790.42秒、GNU timeの最大RSSは926,392 KiB。軸プロセスごとのpeak RSSは価格原本に記録した。

## 通常入口での取り込み

`prepare_h6_full_pf_optimizer.py --short-prices-commit a427e0b858f5b766b2f5ba1cf7ad14af1249d370` で新しい価格を固定する。保存13信号は従来の参照から読む。その `PROBLEM.json` と `ROUND.json` を `run_rpe_schedule_search.py --plan-only` に渡し、同じcanonical IDに14の価格が入り、不足がm7〜m12の12軸とQ3 templateであることを確認した。 [CLI検収結果](OPTIMIZER_CHECK.json) · [通常入口の使い方](../OPTIMIZER_PREVIEW.md)

長時間価格、Q3、新規信号、fit、総費用は保留。今回の実行範囲を完了し、STOPとした。
