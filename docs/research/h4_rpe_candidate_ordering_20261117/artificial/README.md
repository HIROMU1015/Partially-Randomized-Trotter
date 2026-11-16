# 人工入力による候補順の比較曲線

2026-11-17 JST、`saved_log_analysis` sub-agent作成・実行。作成者自身の人工確認であり、独立科学レビューではない。

共有の `ReplayClock` / `JointReplay.run()`、既存CPの13-round caps・重み・人工MSE式、`W.Family` の人工定数を再利用した。新schedulerは作っていない。scenario subclassは宣言済みの幅・準備状態・固定時間prior・初回配分価格をactionへ載せるだけで、選択・起動・完了・採択・予算の処理は共有controllerを使う。

各pairは同じ初期候補集合・初期知識・候補生成・caps・減算規則・精度閾値・時間仮定・予算で開始する。変更するのはA=`ROUND05`、B=`IMPROVEMENT_PER_TIME_V1` の選択policyだけ。slots=1、DP profile=8、lookahead=1、prefix再利用/振替なしは小さい人工caseの固定条件であり、実サーバーの資源上限ではない。

| 意図した人工case | Q3予算（両policy） | Aの到達 | Bの到達 | 予算時点の最安 A / B |
|---|---:|---|---|---:|
| FAST_REUSED_SHOT_FIRST | 1 | cost≤840へ100秒 | 同水準へ1秒 | 822 / 830.22 |
| CALIBRATION_DELAY_AND_OPAQUE_WIDTH | 3 | cost≤800へ63秒 | 同水準へ2秒 | 778 / 738 |
| EARLY_UNKNOWN_SCHEDULE_PAYS_OFF | 4 | cost≤700へ3秒 | 4秒・4起動で未到達 | 684 / 702 |

秒はすべて宣言済みの離散事象モデルで、実処理時間・CPU時間ではない。最初のcaseではBが早く同水準へ届く一方、同じQ3件数での最安はAの方が低い。Bが一般に優れることを示すcase集ではない。

1. **速い既知shotの例**：BASE/OTHERの初期適合点は両方公開済み。次配分のcostは822と830.22、宣言した構成classのQ3時間は100秒と1秒。Bは後者を先に調べる。各jobの未来の個別時間を調べて選んだものではなく、人工case設計時の固定class仮定である。
2. **幅・較正待ちの例**：binary64 `.2` は信号・価格を再利用できる準備済み候補。厳密`"1/5"`はこの人工interfaceに限って別IDの新候補として受理し、初回配分の既知価格580、較正60秒＋Q3 1秒と宣言した。初期traceで、新候補のscoreは580/183≈3.17、既存候補の822配分は32で、Bは再利用できるshotを優先する。Aは新候補の準備を先に行い60秒を支払う。人工新候補の答えはFAILと固定した。これは**実Pauli/DF mapperが厳密1/5を受理する証拠ではない**。
3. **Aが勝つ反例**：初期最安870に対して未知scheduleのcap配分は1204と高いが、capが人工PASSした後のm12 shot削減で684へ下がるように設計した。Aは未知scheduleを早く開き、既存V2の次配分へ進める。Bは即時改善のある既存shotを優先し、周期探索で3秒時に新scheduleを開くが、予算内では安い次配分へ届かない。futureの適合価格を選択器に渡していない。

人工MSEはCPの厳密有理数の重み付き逆shot式を用い、較正caseのOTHERだけ値2のFAILを固定した。これはcase-designの仮定であり、量子信号や実H4のMSE予測ではない。`ArtificialIndex` は選択されたlaunch key以外の読取と全答えの列挙を拒否し、数値は共有clockの完了時まで選択器へ公開しない。初期seedは宣言済み知識としてのみ使う。

成果物：

- `SCENARIOS.json.gz`：全宣言、初期pool、各選択時の理由・全score特徴、完了曲線、目標到達、予算検査。CLIのJSONをgzipで可逆圧縮して公開した。
- `CURVES.csv`：モデルwallとQ3起動件数に対する最安曲線。
- `CURVES.svg`：同じ曲線の標準matplotlib図（3case×wall/件数）。

再現：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/check_rpe_ordering_scenarios.py \
  --output /tmp/new_ordering_artificial_scenarios
```

`--plot` を付けるとmatplotlibがある環境でSVGも生成する。今回の図は既存の `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python` を読取利用した。環境・共有設定を変更していない。

すべてのpairでQ3起動上限、同じ初期pool、選択前の全archive列挙禁止を確認した。新規実信号・較正・compile・実Q3および既存H4 OUTCOMEの読取は0。CPU時間はnullで、モデル秒を総CPUと呼ばない。
