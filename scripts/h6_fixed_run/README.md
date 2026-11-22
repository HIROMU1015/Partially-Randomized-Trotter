# H6の価格取得と方式別Q3

2026-11-22 JSTにH6専用の実行環境へ配置したコードを収録した。固定入力は直線H6、原子間距離1.00 Å、STO-3G、tol-only DF19、h=0.1、M=12。対象の3代表はFull PFの全19項・4次m5_best・要求幅1.6、B0IのS2・p15・要求幅0.2、PR-DFの4次m5_best・LD7・要求幅1.6・全ラウンドでR_outer=16、K=2。Q3は各方式の代表1候補で固定配分s=1を評価する。

## 実装

| 変更 | 入口と実装 |
| --- | --- |
| 回路部品の再利用と指紋計算の削減 | [Full PF](run_full_long.py) と [B0I・PR-DF](representative-prices-r2/run_prices.py) が [共通の価格計算](h6_price_acceleration.py) を使う。決定論回路の反復部品とPR-DFの前後の決定論部品を再利用する。元回路とX/Yの共通命令を一度走査し、X/Yは末尾だけを別に処理する。各軸のコンパイラ設定、コンパイル後の入力指紋照合、6指標、QPYの保存は同じ手順で行う。 |
| Q3の確率表の再利用 | [Q3評価器](q3-pilot/run_q3.py) が同じショット配分の部分評価・全体評価で厳密な各ラウンドの確率表を共有する。 |
| 方式別のQ3起動と回収 | [価格登録](finish_h6_registry.py)、[Q3実行](q3-pilot/run_h6_continuation.py)、[結果取り込み](ingest_h6_continuation.py) が `--method FULL_PF`、`TRUNCATED_PF`、`PR_DF` を受け取る。各方式のX/Y価格26軸が揃うと起動できる。完了済みの結果は再開時に回収し、起動記録があって結果がない場合は保留する。[PBSの方式別入口](q3-continuation.pbs) は `H6_Q3_METHOD` で対象方式を指定する。 |

既存の[正準バイト列ライター](../h46_b3_fast_fingerprint.py)に基づく。同じファイルを実行用ツリーの `optimizer-source/scripts/h46_b3_fast_fingerprint.py` から読み込む。

## 実行先で照合する入力

このディレクトリのコードは、既存のH6実行用ツリーに同じ相対パスで置く。保存入力と実行出力の保管先は、H6の作業領域にある。コードを同じ実行用ツリーに置くか、その入力を新しい実行用ツリーから読めるようにする。

| 入口 | 参照する固定入力 |
| --- | --- |
| Full PFとB0I・PR-DFの価格取得 | `payload-bound/` の固定入力・provider・carrier・依存ソースとmanifest、`optimizer-source/scripts/h46_b3_fast_fingerprint.py`、`representatives/INPUTS.json`。B0I・PR-DFはさらに `representative-prices-r2/PREPARATIONS.json` と保存信号を使う。 |
| 方式別Q3 | `q3-pilot/risk-source/src/` の固定評価器、構成定義、保存信号と対象方式の登録価格26軸。登録価格に記録された構成パスは実行先で解決できる必要がある。 |
| 結果取り込み | `optimizer-source/src/` の固定探索器、`initial-evidence/PROBLEM.json`、Q3開始時と同じ価格ファイル、方式別の出力ディレクトリ。 |

PBSのシェルスクリプトは予約R267805で使った設定を収録した。Linux環境とWSLからの照合には以下のPython入口を使う。価格入口はQiskit 1.3.0とNumPy 1.26.4を確認する。実行時のCPU数は、実行先の割当を `H6_ALLOCATED_CPUS` に渡す。以下の `$h6_run_tree` は配置先、`$h6_python` は固定依存を持つPythonの絶対パス。

```bash
"$h6_python" -B "$h6_run_tree/run_full_long.py" --m 5
"$h6_python" -B "$h6_run_tree/representative-prices-r2/run_prices.py" \
  --accelerated --first-round 5 --last-round 5 --signals "$h6_signal_file"
H6_ALLOCATED_CPUS=1 "$h6_python" -B "$h6_run_tree/q3-pilot/run_h6_continuation.py" \
  --method FULL_PF --registered-prices "$h6_price_registry" \
  --signals "$h6_signal_file" --output "$h6_dry_check_output" --dry-check
```

上記は入力照合用。価格取得には別途出力先の指定が必要で、Q3は `--dry-check` で照合結果だけを書く。現行の価格登録はPBSの元ジョブと出力を照合する。別ホストで新しく取得した価格を登録する経路と、H6の中央worker自動配分は今後の接続範囲。

## 確認記録

- RomuTailで、保存したFull PF m7の入力（T=12.8、q=8）から計算した元回路、X、Yの正準SHA-256が、PBSのジョブ `267857.CPUcluster-PRISM` の保存値とそれぞれ一致した。共通走査は88.89秒。従来の高速ライターで同じ入力のXとYを個別に走査すると83.45秒と84.06秒。計測区間は指紋計算部分。m7の再コンパイルは0件。
- [価格の人工回帰](test_h6_price_acceleration.py) 8件、[Q3配分・方式別実行の回帰](q3-pilot/test_h6_q3_method_dispatch.py) と [確率表の回帰](q3-pilot/test_h6_q3_table_reuse.py) 4件が通過した。人工回路で指紋と6種類のコンパイル指標が一致した。人工価格表と保存信号で方式別登録・照合、実行結果の回収、再開時の重複起動防止を確認した。
- PBSとRomuTailで実入力を照合した。サーバー2とWSLの新版配置・照合、およびM=12の実Q3は未検収。実Q3はサーバー2で別途検証する後半価格の登録後、方式ごとに進める。

回帰を再現する際は `recovered/SIGNALS.json` と `initial-evidence/` の3構成と問題定義も実行用ツリーから参照する。

```bash
"$h6_python" -B -m pytest -q -p no:cacheprovider \
  "$h6_run_tree/test_h6_price_acceleration.py"
PYTHONPATH="$h6_run_tree/q3-pilot/risk-source/src" "$h6_python" -B -m pytest -q \
  -p no:cacheprovider "$h6_run_tree/q3-pilot/test_h6_q3_table_reuse.py" \
  "$h6_run_tree/q3-pilot/test_h6_q3_method_dispatch.py"
```
