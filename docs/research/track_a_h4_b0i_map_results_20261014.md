# H4 B0+I 全240回路の実測と保存shot費用地図（2026-10-14）

全240回路が `MAP240_COMPLETE_STOP` で完了した。新規compile234件と、numeric fingerprintを照合した既存pilot6件の再利用である。12 workersで記録されたstage wallは193.990秒（約3分14秒）。全compute後に停止し、自動retry・次stageは行っていない。

実測B0+Iを追加した全1,812 geometry/precision点の費用点推定最小は、B0+Iが397点（21.91%）、B2が1,415点（78.09%）、同費用は0点だった。これは今回の登録domain・保存標本・compiler・状態準備cost P=0に限る事後比較である。科学的な主要claimの最終認定は別の結果reviewへ戻す。

## 条件と費用の定義

H4 linear chain、STO-3G、凍結DF rank12、8 system qubits＋ancilla1、T=0.8。距離は0.70/0.80/0.90/1.10/1.40/1.60 Å。B0+Iはprefix3/4/5/6/9、q=1/2/4/8、δ=T/q=0.8/0.4/0.2/0.1、二次PF、ordinary control、structured Gaussian loweringを用いる。120構成のcosine/sine両軸を実測した。Qiskit1.3.0、rz/sx/x/cx、optimization level1、seed17等の固定compiler optionsと既存private venvを維持した。

B0+Iはdiscardしたtailの既知identity係数を保持する決定論的対照。既存B0の保存signalを解析的な位相回転で変換したbiasを用い、新たなsignal simulationや量子backend測定は行わない。元B0/B1/B2/B3の登録集合・結果・原稿は履歴として保持する。

εはcomplex-signalの誤差要求であり、energy誤差Haではない。各軸のheadroomを `ε/√2 − bias` とし、正の場合だけ `N=ceil(2B² log(80)/headroom²)` を適用する。B0+IはB=1、各軸α=0.025。比較費用は `N_cos C_RZ,cos + N_sin C_RZ,sin`。状態準備P=0で、実機wallや全RPE総費用ではない。

## ε=0.05の実測費用比較

費用単位は百万RZ×shot。各方法で登録候補内の費用最小を取った。

| 距離 Å | B0+I prefix/q | B0+I費用 | B2費用点推定 | B0+IのB2比削減率 |
|---:|---:|---:|---:|---:|
| 0.70 | 5/1 | 152.87 | 206.51 | 25.97% |
| 0.80 | 5/1 | 152.91 | 171.34 | 10.75% |
| 0.90 | 5/1 | 148.13 | 134.06 | −10.50% |
| 1.10 | 5/1 | 144.71 | 112.91 | −28.17% |
| 1.40 | 3/1 | 98.91 | 102.93 | 3.91% |
| 1.60 | 4/1 | 118.88 | 97.28 | −22.20% |

正値はB0+Iが安く、負値はB0+Iが高い。例えば0.70 Å/p5/q1のcosine/sineはRZ8,454/8,453、shot10,092/7,992、合計152,874,144。1.40 Å/p3/q1はRZ5,218/5,217、shot9,077/9,880、合計98,907,746。

## 精度を変えた全地図

各距離302点、ε=0.005〜0.1のlog gridにε=0.05を追加した。同じ保存値の再会計で、1,812独立実験ではない。下表の範囲はB0+Iが低かったgrid点の最小・最大であり、連続精度上の厳密な交点ではない。

| 距離 Å | B0+I点最小 | B2点最小 | B0+Iが低かったgrid εの範囲 |
|---:|---:|---:|---:|
| 0.70 | 178 | 124 | 0.01725〜0.10000 |
| 0.80 | 103 | 199 | 0.02351〜0.06444 |
| 0.90 | 29 | 273 | 0.02676〜0.03540 |
| 1.10 | 0 | 302 | 該当なし |
| 1.40 | 87 | 215 | 0.04279〜0.10000 |
| 1.60 | 0 | 302 | 該当なし |

ε=0.005では6距離すべてでB2の費用点推定が低い。ε=0.1では0.70と1.40 ÅでB0+I、他4距離ではB2が低い。0.80/0.90 Åには中間精度だけB0+Iが低くなる領域がある。単純な距離順や精度順の一律優位とはならない。

全240軸で旧B0との差はRZ数0。さらにCX数、回路size、RZ/CX depth、total depthも全240軸で差0だった。これは位相を保持した回路の実compile結果であり、任意のcompilerで差0になる保証ではない。今回、追加identityによる回路規模増加は生じず、保存signalのbiasと必要shot数の変化が費用比較へ反映された。従来のΔRZ=0感度で報告した397点は、全候補の実費用取得後も同数となった。

元登録集合だけでは全1,812点でB2が点最小だったという歴史的結果は保持する。合理的なscalar保持対照を加えた今回の点比較では、B2の全面優位という表現は使えない。

## 監査・資源・会計

- SOURCE `924b779c2bc6f6e844d6283c355f91c73aed6445`、grant `15010795611b156656f943f761e47e99820765e6`、run `h4-b0i-map240-parallel-run02-20261014`。dataset identity SOURCEは `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`。
- source closure59件をactual SOURCEのGit blobと照合し、固定launch manifest・profile/保存input bindingを保持。全240結果は固定runnerのtask/source/plan/axis/metric/numeric fingerprint/reuse受理規則に合格した。
- 原本terminalのfirst STOP理由はnull、cleanup完了。12 workerのnative exitは全0、observerはcompute後に意図的SIGTERMで−15。専用14 identityの残存0を読取確認。driverの元tool sessionは取得不可のためnative exit codeはnullとし、terminal JSONから補完しない。
- observer191観測、最大interval1.01093秒、観測所要0.00997秒、staleness0.00997秒。保存観測の全memory PSI0、OOM counter増分0、監視違反なし。
- worker COMPLETE receiptのnative VmHWM最大326.57 MiB、observer sampled RSS最大21 MiB。VmHWMはそのworkerのreceipt時点までの生涯peakであり、task単独やcleanup後までの最終peakとは区別する。
- このrunのoutput/control/lock/driver log/launch receiptは論理内容1,114,523 bytes、directory込み実割当3,112,960 bytes（2.96875 MiB）、548 entries。凍結入力・旧run・worktree・今回の公開用解析資料を含まない。
- carry73,029件を保持し新規234件を加算、累積73,263/74,805件、残1,542件。観測実割当を加算した累積charge13,618,418,386 bytes、17GiB cap内。記録wallはcarry268,330.851435秒＋193.990122秒。失敗分の返却/resetはない。
- 事後解析はstdlibの単一process。全1,308旧候補の精度別最小を再計算し、全1,812点の選択済みB2費用・paired SEはFractionによる独立算術で照合。公開projectionのみの再現もPASS。新compile、NPZ/scientific array読込、input生成、GPU、共有環境/venv/H8/他job変更は0。

## 解釈の限界と資料入口

B2は同じ32 paired trajectoriesで選抜と費用推定を行っている。397点のうち374点でB0+IはB2点推定−2SEも下回るが、これは記述的な標本診断であり、同時信頼区間や母平均優劣の保証ではない。H4 event次数履歴のNOT_SAVED、172候補の丸められたnormalization、希少event・旧compiler output完全同一性の未確認は今回解消していない。H6/H8、state preparation P>0、別compiler/合成、noise、fault-tolerant cost、energy/RPE全体へ外挿しない。

[全結果bundle](../../artifacts/resource_applicability/track_a_h4_b0i_map_results/2026-10-14/README.md)に、240 wrapper metrics、120構成の費用入力、1,812点CSV、代表18点、native audit、byte manifest、公開再現scriptを保存した。実行用のSOURCE・plan・grant・入力・runtime原本は変更していない。これはlocal保存値照合で、独立科学review・外部再現・CIではない。主要claimや研究の着地点は結果reviewで判断し、追加科学計算には進まない。
