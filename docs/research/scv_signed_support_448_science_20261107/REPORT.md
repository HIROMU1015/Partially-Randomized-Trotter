# Signed Support 448-shot pilot：完了・追加point改善なし

2026-11-07 JST / SERVER1 GPUA100-gleap-01。

**448/448 physical shots、全bit/価格を保存し、学習・6 paired集計・終了保全を完了した。両軸とも追加の分散点改善を観測せず、学習費込みpoint scoreもB1-OL参照より大きかった。これは一pilotの探索的結果で、一般的無効性・厳密優劣・期待費用を証明するものではない。**

## 固定条件とSOURCE

保存linear H4／1.00 Å／STO-3G／legacy DF12／8 system qubits／全4電子sector、L_D3、m0、q1、δ=.02、r1、K2、coefficient_atol0。N1 actual finite-grid Q/全5正層/selected full、N2保存保証ground、認証済みexact Fraction engine、理想gate、X/Y、full no-prepを維持した。

科学SOURCEは **`f70e555fbfe08676f530d989192aade75a57ab1a`**。実装仕様138d0718、metadata結果b4512675、kernel b23b3c0、N1 744e1a0/N2 5e9bd08/engine fdf00ce、qualified managerを再利用した。[実行仕様](EXECUTION.md)と[準備SOURCE](../scv_signed_support_448_preparation_20261107/SCIENCE_SOURCE_PREFLIGHT.md)を参照する。元の201人工PASSは同SOURCEの既存証拠で、実科学後の再学習・テスト用実H4投入はない。

Campaign=`SCV-SIGNED-H4-H1-PAIRED-20261107-v1`、seed20261110、PCG64/SeedSequence、12 domain/24children。X train64→Y train64→X fit→Y fit→両軸凍結→X eval160→Y eval160を一回実行した。旧bit/係数/word/cacheは使わず、新cacheのみ。

## 学習と全5層paired診断

h1だけ軸別intercept OLS各1回、binary64 slope→exact Fraction freeze。他4層λ0。両軸rank2、exact signed幅≤6に合格した。下表は保存値の丸め表示で、exact係数・幅・統計値はraw/RESULT_SUMMARYを正本とする。Xの係数は小さい非零値のまま保持し、0への丸めやclipをしていない。

| 軸 | rank | λ1（表示近似） | 幅2+2abs(λ1)（表示近似） | exact幅6 guard |
|---|---:|---:|---:|---|
| X | 2 | -8.56211664316e-17 | 2 | PASS |
| Y | 2 | -0.0263157894737 | 2.05263157895 | PASS |

全5層の共通fresh quotaは各軸(8,128,8,8,8)。同じ評価recordでB1-OLとSCVを比較し、full＋trial ID固定half A/Bを各軸3、計6回だけ集計した。V̂は既存集計が返した推定平均のsample-based variance点指標で、真の母分散ではない。

| 軸 | B1-OL平均 | SCV平均 | V̂ B1-OL | V̂ SCV | δ=V̂ B1-OL−V̂ SCV | 保存flag |
|---|---:|---:|---:|---:|---:|---|
| X | 0.999946435558 | 0.999946435558 | 5.60860020945e-10 | 5.60860020945e-10 | -5.01919613261e-35 | INCONCLUSIVE_DEGENERATE_OR_SPARSE |
| Y | -0.0766145415653 | -0.0746420262728 | 0.00737790271621 | 0.00740702201167 | -2.91192954577e-05 | なし |

Xの追加差は極小で、active h1の退化/疎データflagを保持する。全体のV̂ B1-OLを0と扱っていない。YではV̂ SCVが増加し、half A/Bもδの向きは負だった。保存flagが空のYを、有意差ありと読み替えない。両軸とも `point_improvement=false`、`statistical_significance=NOT_ASSESSED`。Half spreadはCI/SEではなく、平均差のvarianceをvariance削減量のCIにも使わない。feature非定数性・rank/guard合格はCV改善の十分条件ではなかったが、全Signed Support手法の否定や正式GO/NO-GOは採択しない。

## 全段階の実現RZ

Qiskit1.3.0、basis=(rz,sx,x,cx)、opt2、seed_transpiler17、backend/coupling/layout/routing=None。状態準備だけを除くselected full価格を毎shot取得した。cache hitでも全RZを課金する。

| 軸 | train shots | 共通eval shots | train RZ | B1-OL参照：eval RZ | SCV参照：train+eval RZ |
|---|---:|---:|---:|---:|---:|
| X | 64 | 160 | 306,561 | 771,512 | 1,078,073 |
| Y | 64 | 160 | 306,789 | 771,943 | 1,078,732 |

全campaign physical RZは **2,156,805**（train **613,350**＋共通eval **1,543,455**）。参照費を足して二重課金しない。固定quotaのこの一campaignでは追加のSCV量子費用はtrain分であり、独立して2つのcampaignを走らせたものではない。

| 軸 | B1-OL：V̂×eval実現RZ | SCV：V̂×(train+eval)実現RZ |
|---|---:|---:|
| X | 0.000432710236479 | 0.00060464804536 |
| Y | 5695.32035646 | 7990.1916687 |

両軸とも学習費回収を支持するpoint材料は得られなかった。これらは保存済み探索的point scoreの引用で、保証付きshot数・期待単価・期待総RZ・最適B1-OLへの優越を新評価したものではない。精度・失敗確率・十分N・全RPEの総費用は未評価。

## 有限call・終了・実行負荷

実空きCPU/affinity・memory/pressureから **63 workers** を使用した。固定kernel/共有venv/共有設定・他jobを変更しない。GPU/float/O1/O2 kernel差替えなし。

- trial/path/probability/Born/bit threshold/price各448、selected full build896、compile154、OLS2、paired6。
- apply_exact351,388（上限427,840）、各probability最大888 gates（保存cap955）、新word947（上限29,120）。
- saved control復元・bundle decode・stream book各1。新signed moment有限和、N1再準備/CDF、N2 solve/spectral、engine再認証、科学retry・追加campaignは0。
- 全call RETURN、例外・in-flight 0。全448bitと448価格が一致し、bit-before-priceとSTOP後の新STARTなしを確認。学習/評価とscope/identity/quotaの照合もPASS。
- 親・manager・outerはexit0、63 workerも全exit0。consume-once、trigger撤去、Restart=no、COMPLETE/mandatory STOPを維持。追加shot・再投入はない。

親wallは **3965.35秒（約66.09分）**。測定worker user CPUの区間和169,324.41秒、system379.85秒、worker wall区間和180,068.08秒を別記する。これらを親elapsedや同時物理RAMと混同しない。親peak RSS1,301,532,672 bytes、最大worker RSS highwater475,209,728 bytes。PSS/RSS/cgroup/pressureとinclusive timerは[資源まとめ](CLASSICAL_RESOURCE_SUMMARY.json)および全rawに保存した。

compile154件、cache hit294件。bundle decode5.535秒、fit合計1.665秒、paired合計9.095秒は測定区間の引用。engine/build/Born/apply/price/compileはnested/inclusiveのため足し合わせない。純cache lookup等の未測定項はUNKNOWNで、旧TrialCostの0 placeholderを測定0にしない。worker初期化等を含む完全な古典金銭費用換算は未取得。

## 保存・照合・保全

[結果要約](RESULT_SUMMARY.json)、[保存記録integrity](SAVED_RECORD_INTEGRITY.json)、[古典資源](CLASSICAL_RESOURCE_SUMMARY.json)、[保全照合](PRESERVATION.json)を作成した。新しい数値fit/paired/Born/compileを実行せず、既存出力の引用・整合検査と資源会計だけを行った。

全raw472ファイル、137,145,580 bytesを **RAW_COMPLETE.tar.gz（12,636,380 bytes）** へ保存し、全memberの元bytes/hashを照合した。主要JSON110件は直接閲覧用にもコピーし、raw原本・cache・word・attempt・全journalは変更していない。

- [artifact入口](../../../artifacts/scv_signed_support_448_results/2026-11-07/run01/README.md)
- [raw manifest](../../../artifacts/scv_signed_support_448_results/2026-11-07/run01/RAW_MANIFEST.json)
- [complete raw archive](../../../artifacts/scv_signed_support_448_results/2026-11-07/run01/RAW_COMPLETE.tar.gz)

実SOURCE rootはf70e555のまま保全し、結果専用branch `codex/scv-signed-support-448-results-20261107` で公開する。元main/dirty/untracked、N1/N2/engine、旧224/752/metadata/中断結果・STOP/grant/ledger/marker・C1凍結を維持。並行O1/O2のworktree/process/cacheへ操作していない。

## 結論の範囲とmandatory STOP

この一条件・一campaignでは、Signed Supportのfresh学習はrank/guardを満たしたが、強いB1-OLに対する追加の分散点改善・学習費回収の兆候は得られなかった。相関や期待費用を厳密に判定したものではなく、科学的GO/NO-GO、control変更、追加測定、保証付きcalibration/productionを採択しない。

全結果・SOURCE・費用・STOPを公開し、remote SHA/本文/blob/tree/archive hashを照合して**mandatory STOP**。次の科学作業は、今回の結果を踏まえた有限scope/条件の判断と別途明示許可が必要。自動の再fit・追加campaign・詳細科学レビューは行わない。
