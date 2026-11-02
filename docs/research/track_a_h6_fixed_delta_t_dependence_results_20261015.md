# H6固定δ：T依存性の結果

2026-10-15。**登録された新72 cost group・144 full-wrapper compileとPR64 trajectoryを正常完了した。**
run `h6-fixed-delta-t-panel-20261015-run02` は23:01:49 JSTに終了し、native exit0、wall3117.158秒（約52分）。自動retry・次stageは開始していない。

要求精度0.05〜0.005では、PRのP=0・shot込みRZ費用は、各Tで適格な登録決定論的対照の最良値より約50〜55%低かった。
一方、厳しい精度ではS4が有利になる。T=.8/ε=.001ではPRがS4の約1.76倍、T=.4/ε=.0005では約6.87倍である。
T=1.6/ε=.001・.0005、およびT=.8/ε=.0005のPRは既存の精度適格性規約を満たさない。
これは固定した5構成・3つのTでの有限標本の費用比較であり、全候補最適性や母平均費用の保証ではない。

## 比較条件と出所

Linear H6、1.0 Å、STO-3G、保存Hamiltonian/DF rank19/state、12 system+ancilla、δ=T/q=.2。
B0 p15/S2、B0+I p15/S2、full PF p19/S2、full PF p19/global Yoshida S4、PR p5/r4/K2/S2を固定する。
要求精度は .05/.01/.005/.001/.0005。主指標は状態準備費用P=0の測定込みfull-wrapper RZ総費用である。
各Tで構成を選び直す探索は行っていない。

実行SOURCE `669842c85e2c04a90dcdcfb774b323a2737cb8ba`。
固定native provider `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`、旧science SOURCE `5a903aea2492c2d211c1e15e7737df0d50694c12`。
compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、symmetric_directionalを維持した。
T=.4/1.6は各32本の新しい独立標本、T=.8は旧独立確認32本を再利用した。旧探索8本とのpoolingは行っていない。
T=.8のsignal・PF・PR費用は旧公開結果commit `321cd8c3066194fe98741cbebbcdba1515d1deaa` とD0保存値、B0+Iは起点commit `34efc914b2f485ec694c7d6016025d5c76829be6` のp15/q4実測に結合した。

T=.8の旧B0とB0+Iは別の数値fingerprint系列である。compiler profile一致だけでcompiled bytesの同一性は主張せず、両者の微小な費用差・順位差をidentity位相だけの効果と解釈しない。
旧hostとserver compiler outputのbit完全同一性も未確認のままである。
B0+Iは保存scalar `C=1.1246847212849682e-7` によるexp(-iTC)を適用し、full-H referenceを回転しない。
H/DF/stateの再生成、compiler変更、H8/H4変更は行っていない。

## 精度と総RZ費用

表は **PR / 適格な決定論的対照の最小費用**。1未満ならPRが低費用、1超なら決定論的対照が低費用である。
経験的numerical allowanceの既定factor1を使用する。factor10の感度表も保存し、この表の適格性・主な傾向は変わらなかった。

| 要求精度ε | T=0.4/q2 | T=0.8/q4 | T=1.6/q8 |
|---:|---:|---:|---:|
| 0.05 | 0.453740 | 0.467465 | 0.497423 |
| 0.01 | 0.453702 | 0.467396 | 0.497245 |
| 0.005 | 0.453658 | 0.467275 | 0.496910 |
| 0.001 | 0.452914 | 1.756787 | 不適格 |
| 0.0005 | 6.869224 | 不適格 | 不適格 |

対照はT=.4ではε≥.001でB0、ε=.0005でfull S4、T=.8ではε≥.005でB0+I、より厳しい2精度でfull S4、T=1.6ではε≥.005でB0、より厳しい2精度でfull S4となった。
Full S4は全T・全5精度で適格だった。
「不適格」は既存の軸別ε/√2・bias・allowance・headroom規約で必要shotを決められない状態であり、費用0や全PR方式の不可能性を意味しない。

例としてT=.4/ε=.0005では、PRは1回路平均RZ約25,890と小さいが、必要総shotは約84.45億である。full S4は1回路RZ220,991、shot約1.44億で、総費用はPR約2.1865e14、S4約3.1830e13となる。
小さい回路だけでは、この精度でのshot増大を補えなかった。

[主表CSV](../../artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15/primary_RZ_P0_v1.csv)、[全6指標・allowance感度CSV](../../artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15/all_metric_comparisons_v1.csv)、[軸別shotとpaired費用・SE](../../artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15/accounting_v1.json)を保存した。
SEは同じtrajectoryのcos/sinを組にした標本標準誤差である。希少イベントを含む母平均の信頼上界ではない。
全回路のRZ/CX/RZ-depth/CX-depth/total-depth/sizeは[回路平均表](../../artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15/per_circuit_metrics_v1.csv)と[全216軸のprojection](../../artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15/wrapper_projection_v1.json)にある。

## T依存性の主な原因

full-Hに対する複素signalの絶対biasは次のとおり。

| 構成 | T=0.4 | T=0.8 | T=1.6 |
|---|---:|---:|---:|
| B0 p15/S2 | 0.000333758147 | 0.000667121294 | 0.00133244117 |
| B0+I p15/S2 | 0.000333803134 | 0.000667211268 | 0.00133262112 |
| Full PF S2 | 0.000333823538 | 0.000667252077 | 0.00133270275 |
| Full PF S4 | 7.57578874e-06 | 1.51515883e-05 | 3.03032311e-05 |
| PR p5/r4/K2/S2 | 0.000333280946 | 0.000666167823 | 0.00133053799 |

固定δでもPRのbiasは約3.33e-4→6.66e-4→1.33e-3と、今回の3点ではTにほぼ比例して増加した。
signed分解でのouter-PF成分は約3.33e-4→6.66e-4→1.33e-3、finite-RTE成分は約1.25e-8→2.49e-8→4.98e-8だった。
この構成ではouter-PF biasが主であり、finite-RTE打切り成分を小さくするだけでは厳しい精度でのheadroomは大きく回復しない。
これらは複素ベクトルの成分であり、絶対値を加えてtotal biasとすることはできない。

PRの局所時間δ/r=.05と1 occurrenceの分布は同じだが、全occurrence Rは8→16→32。
Bは1.016984→1.034256→1.069686、B²は約1.034→1.070→1.144へ増加する。
1軸平均RZも25,889.78125→51,234.15625→101,586.3125と増加する。
緩い精度ではPRと対照の回路費用がともに概ねqと増え、PRの利益は約55%から約50%へやや縮小した。
厳しい精度での逆転・不適格化には、Bの増加よりもbiasによる統計誤差予算の縮小が強く効いている。

B0+Iのdiscard成分は、T=.4/1.6で約2.04e-8/8.16e-8であり、B0の約6.54e-8/2.62e-7より小さい。
ただしこのp15 panelではPF成分がはるかに大きく、signed成分の相殺もあるため、identity補正がtotal biasを常に小さくするとはいえない。
状態準備費用Pの追加sweep、energy/QPE/RPE費用、別T・別構成への外挿は今回の集計対象にしない。

## 希少Taylor次数と主張の限界

各eventの次数2確率は `0.001051813380852956`。その確率自体は固定δ/rで変わらず、trajectory内の発生確率はRとともに増える。

| T | R=qr | 32本の次数0 / 次数2イベント数 | 1 trajectoryに次数2を含む確率 | 32本で次数2未観測の確率 |
|---:|---:|---:|---:|---:|
| 0.4 | 8 | 256 / 0 | 0.8384% | 76.38% |
| 0.8 | 16 | 511 / 1 | 1.6697% | 58.34% |
| 1.6 | 32 | 1024 / 0 | 3.3115% | 34.04% |

T=.4/1.6の32本では次数2を観測しなかったが、平均費用への寄与を0とは扱わない。
未観測回路の母平均費用上界は証明しておらず、表の費用削減率は経験的numerical allowanceと有限32本に基づく点比較である。
旧q8/r2の条件付きs2 pilotは今回のr4と異なり、その費用補正を流用しない。
この3点だけで連続したT領域の優位性や、全候補最適性を保証しない。追加sampling・新compileは実施していない。

## 実行・終了・会計の監査

- 新規signal jobs2、PR64 trajectory、72 cost group、144/144 full-wrapper compile完了。sampling API320、RTE events1280、reference matvec800、primitive probe750、control probe40。
- native wait exit0、全74 worker exit0・wait/reap、cleanup error0、first STOPなし。保存PIDの現在残存0を読取確認した。
- wall3117.158秒、signal158.787秒、heavy S4単独2178.304秒、normal最大8並列776.775秒。各数値role内部thread1。
- 観測stage aggregate RSSピーク：signal約1.05GiB、heavy約3.19GiB、normal約6.26GiB。これは単一workerのRSSではない。
- 最大観測間隔1.118秒、観測所要時間0.00757秒、host full PSI最大0.12%、OOM増分なし。5秒監視・既存資源上限内で完了した。
- 保存native control probe最大誤差1.49e-13未満、primitive native/expm最大3.43e-15未満、spectral/expm最大1.04e-13未満。形式的な全状態保証とは扱わない。
- 実行SOURCE/runtime対象filesとbound inputのstreaming SHA不変。provider/profile/environmentは起動時に固定照合済み。本集計ではvenv全34,273file inventoryを再走査していない。
- carry・未知の旧actual・失敗分の予約を返却/リセットせず、累積上限compile1910/trajectory916/sampling3584を保持した。上限値を累積actualの確定値と呼ばない。
- 今回の結果集計では科学array読込・新trajectory/reference/signal/compileを実行していない。H4/H8/既存venv・共有設定・他jobの変更0。

[完了監査](../../artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15/completion_audit_v1.json)、[受領bytes/SHA一覧](../../artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15/input_receipts_v1.json)、[新144軸の時間/RSS](../../artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15/execution_resources_v1.csv)、[SOURCE/profile出所](../../artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15/result_provenance_v1.json)を保存した。
軸時間はwrapper作成・fingerprint・transpile・metricsを含み、pure transpile時間ではない。軸RSSはその時点までのprocess高水位であり、軸専有量ではない。
実runtime/trace/cache/NPZはprivate homeに保全し、Gitには軽量projectionと監査のみを追加する。

## 集計の検証・再現・STOP

元の保存shot/費用314条件の一致、人工算術7検査、公開JSONだけから150会計行・900 paired metricの再計算に合格した。
この公開再計算は科学入力・Qiskitを必要としない。

```bash
/usr/bin/python3 -P -B scripts/summarize_h6_fixed_delta_v1.py --verify \
  artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15
```

[図](../../artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15/T_precision_cost_and_causes_v1.svg)は保存値だけから作成した。主担当による技術照合であり、サブエージェント・独立review・GPT詳細科学reviewは行っていない。
実行SOURCEと既存結果は保全した。追加研究・科学条件変更・新grant・次stageを開始せず、**mandatory STOP**とする。
