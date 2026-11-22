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

## HChainの通常探索との接続

共通入口 `scripts/run_rpe_schedule_search.py` は `PROBLEM.json` の `problem.system` で対象の原子数を受け取り、登録済み候補にも `n_atoms` を保持する。候補の単価・信号・Q3はその原子数、snapshot、Hamiltonian、状態に結び付く。通常入口は従来のQ3候補選択・予算・結果取り込みを使う。

H6 PR-DF用に通常の `ROUND.json` へ `hchain_price_preparation` を設定できる。値はH6実行領域の絶対パス `root`、`initial`、`signals`、対象の `method: PR_DF`、必要なら `pr_calibration_max_m` と `route`。対象の構成と範囲は `PROBLEM.json` の `methods.PR_DF` に指定する。Q3の `route` は `kind: h6_df19_fixed_q3`、`view_commit: cbc471c913072de45c25a0ed22d51dff8a9b7034`、実機の `interpreter` と固定評価器の `risk_source` を受け取る。

- 実行可能な通常開始では、保存済みの軸価格で不足があると [共通のPR-DF価格較正](../../src/trottertracks/pr_df_price_completion.py) を使う。較正上限の既定はm5。実ステップを分け、同じ構成・準備・コンパイラ内でX/Yを独立に較正する。固定外側イベント数では切片＋q、外側イベント数を変える方式では切片＋q＋qRを使う。既存の実測価格を優先し、不足軸にモデル予測を入れる。
- 各モデルの係数、較正点、較正に使わなかった保存点との記述的比較、軸別の実測／モデルの出所が `HCHAIN_REGISTERED_PRICES.json` に入る。モデル価格は条件付き総額に使う。外挿精度や全乱択分布の期待価格の資格は別に評価する。必要な実測が無い実ステップ群は、価格不足として扱う。
- 準備が済んだ同じrunの再開では、保存済みの価格表をSHA-256で照合する。準備済みの価格表から始める場合は、`hchain_price_registry` に `path`・`sha256`・`route` を指定する。`--plan-only` は新しい較正を開始しない。
- 人工価格と保存H6信号を用い、通常入口のモデル生成・全26軸の費用計算・Q3要求生成・固定評価器の入力照合まで確認した。同じrunの再開で価格表の再生成は0件。

## PBSでのH6範囲指定

[範囲ファイル](h6_scope_example.json)で方式、公式、要求幅、分割位置、PR-DFの絶対κ、RPEの最終ラウンドを指定する。例の `h=0.1`、`M=12` ではFull PF 12件、B0I 48件、PR-DF 70件を列挙する。候補のRPE信号と価格を揃え、配分の選択とQ3結果の採否にはH4と共通の探索器を使用する。

[全分割位置・幅h〜10hのH6設定](h6_scope_h_to_10h.json)では、3方式とも要求幅をh刻みの10点にする。B0IとPR-DFは全分割位置を使い、Full PF 30件、B0I 570件、PR-DF 1,330件を列挙する。PR-DFはS2、K=2、絶対κは1・2・4・8・16・32・64。

保存DF19では、B0Iの `p` とPR-DFの `L_D` に0〜18の任意の整数を指定できる。`"prefixes": [1]` は1だけ、`"prefixes": "all"` は0〜18を指定する。全19項を保持する点はFull PFで扱う。要求幅 `width_over_h` と絶対κは正の数を列挙し、`"3/2"` のような分数も使える。PR-DFの `K` は単一の非負の偶数かその一覧を指定できる。計画モードは0以上の最終ラウンド `M` を受け付ける。固定Q3評価器でMSEを評価できる範囲は `M=0..12`。

ショット数は各ラウンドで `N_X = N_Y = ceil(s * (11 + 4 * (M - m)))` とする。`scaled_shots.grid` が共通倍率 `s = 2^(k/divisor)` の範囲を決める。幅h〜10hの設定では `k=-12..16`、`divisor=4` なので、倍率は1/8〜16の29点。`compare_paper: false` のため、PR-DFの論文比較用倍率 `exp(2/κ)` は配分候補に加えない。

[PBS入口](h6_scope.pbs)に `H6_OPTIMIZER_CHECKOUT`、`H6_ASSETS`、`H6_PYTHON`、`H6_SCOPE_FILE`、`H6_SCOPE_OUTPUT` を絶対パスで渡す。`H6_ASSETS` は `payload-bound`、`representatives`、`optimizer-source`、`q3-pilot/risk-source` を含む既存のH6実行領域。既定は `--plan-only` で、候補表と固定入力の照合を行う。`H6_SCOPE_ACTION=--prepare-only` では信号・較正・価格表を準備し、`--execute` では共通のQ3探索まで進む。`H6_PREPARATION_WORKERS` と `H6_Q3_SLOTS` で割当内の同時実行数を調整できる。

準備枠は信号と価格較正で分ける。`H6_SIGNAL_WORKERS` と `H6_PRICE_WORKERS` の既定は割当内で使えるCPU数。従来の `H6_PREPARATION_WORKERS` を渡すと両方の既定値になる。価格較正は決定論回路の規模とPR-DFのイベント数から初期メモリを見積もり、同規模の処理で測ったピークRSSと実行中のRSSを次の割当に反映する。`H6_PRICE_MEMORY_GIB` の既定はPBSの350 GiB要求と一致し、観測できるcgroup上限が小さいときはそちらを使う。PBSのメモリ要求を変えた場合はこの値も合わせる。工程の件数・CPU時間・メモリは `SIGNAL_PREPARATION_METRICS.json` と `PRICE_PREPARATION_METRICS.json` に記録する。

PR-DFの価格計算では、同一の有限RTE分布と成分表から同じ成分列が出たとき、生成済みのイベントを再利用する。乱数生成器と乱数の消費順、各経路のイベント順、コンパイラ設定は固定したまま、乱択部分の回路を直接出力回路に合成する。大域位相は既存実装と同じ順に加算する。

実コンパイルは `calibration_max_m` のラウンド時間（例ではm5）以内に収める。価格モデルには、同じ実ステップ、公式、分割位置での観測を使い、X/Yを独立に較正する。同じ実ステップの短いラウンドで独立点が不足する場合は、同じ実ステップで1・2反復の較正専用回路を作る。その時間が上限を超える場合は、上限内のステップで較正し、対象ステップへの移送を条件とした価格を登録する。価格表に両方のステップと較正の根拠を残す。較正専用回路は探索候補に含めない。PR-DFで外側イベント数を変えるときは切片、`q`、`qR` を使う。得られる総額はモデルを仮定した条件付きの値。未完了の信号や価格は `HELD_CANDIDATES.json` で確認できる。

同じ範囲と固定入力で同じ出力先を指定すると、完了済みの信号・較正・Q3を再利用する。過去の信号束を指定する場合は `H6_SAVED_SIGNALS` で渡し、科学入力の来歴と物理スケジュールが一致した点を受け入れる。起動記録だけが残る処理は調査対象として記録する。

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
