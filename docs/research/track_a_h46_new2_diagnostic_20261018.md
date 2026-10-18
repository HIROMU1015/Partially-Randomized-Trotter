# H4/H6：4th(new_2)の限定数値検証

利用者の「こちらを使って、H4とH6についてまずできる範囲でよいので、検証を行って。できるだけ並行して行って」を実行根拠とする独立診断。既存の科学契約・SOURCE・grant・ledger・STOPを書き換えない。

対象は保存済みlinear H4/H6、1Å、STO-3G、rank12/19、T=.8、保存状態とsigned generation順を保持する。H4 prefix3/9、H6 prefix5/15、q1/2/4/8。2次・Yoshida4次・new_2の全体PFを比較し、各outer stepのRTE総short-step数32を共通にする。signed係数の絶対値に比例配分し、有限paired Taylor cutoff K2/4を使う。new_2は既存8桁係数の実用4次候補であり、記号的な厳密4次証明とはしない。

既存のfull-Fock対角係数関数と有限RTE分布を再利用し、native basisを構築せず保存順prefixを保って、実際のordered DF fragmentを含む全体合成の期待signalを評価する。exact H_D/H_Rの2-blockだけの検証とは異なる。ランダム抽選なしで有限分布の期待演算子を計算するため、希少event欠測による標本誤差は入らない。ただし回路のnative意味論・compiled費用・量子shot実測は検証しない。

H4とH6を独立した2 processで同時実行し、各process内BLAS4 threads。同一systemの行列・固有分解・符号付きstep係数をcacheし、exact-tail／K2／K4／rawの5列をまとめて伝播する。q/cutoffごとのprocess再初期化や同じ固有分解の反復を避ける。現行共有資源を直前観測し、共有環境・他process・venvは変更しない。過去のwall/AS/RSS上限は新規診断に持ち込まない。

SOURCE固定前の人工5検査はPASS。合成係数、共通RTE予算、負時間、実用次数、scalar位相、raw/corrected normalizationを対象とし、新科学計算の前に実施した。原SOURCEの既存SyntaxWarningは改変しない。

[計画・入力binding・環境・provider closure](../../artifacts/resource_applicability/track_a_h46_new2_diagnostic/2026-10-18/README.md)。実行結果は別commitで追補する。実装author自身の検査であり、独立科学レビューではない。新しい科学的最終採択や原稿改訂は行わず、診断終了後mandatory STOP。

初回SOURCE `a9af63fe0ec70e5dffa36b1d80fa198192b4f341` のrun01は、数値診断に不要なnative basis生成を呼び、H4はregistry幅制限、H6は未導入qiskit-natureでSTOP。両worker native exit1・wait回収、wall8.008秒、有限RTE比較0。既存環境にはinstallせず、canonicalなfull-Fock I/Z/ZZ係数関数を直接利用するmatrix-only経路へ修正した。科学条件・閾値を変えず人工6件PASS。元STOPは別receiptとprivate原本に保持し、新SOURCE・新run02で再試行する。

## 実入力の検証結果

修正後SOURCE `7f4c9a171a1d4caeaeaee6c443217edec761f4c9`、run02をH4/H6並行で一回実行し、両worker native exit0・wait回収。各24、合計48のexact-tail PF比較と、K2/4の有限RTE96条件を完了。これは保存状態の有限時間complex signalの比較であり、energy biasやRPE全体費用の評価ではない。H4は保存sector36、H6はsector400。旧9/25のH4 sector70診断とは別のscope。

代表条件は、各1Å/STO-3G、H4 rank12/prefix3、H6 rank19/prefix5、T=.8、q1、RTE総short-step32、K4。以下はfull-H保存状態referenceに対するcomplex signalの絶対誤差とnormalization由来B²（必要shot数の全会計ではない）。

| 系 | 公式 | signal絶対誤差 | B²のみ | deterministic primitive actions |
|---|---|---:|---:|---:|
| H4 | 2nd | 0.00725185869 | 1.01369956 | 8 |
| H4 | 4th | 0.00224603257 | 1.30157667 | 24 |
| H4 | 4th(new_2) | 3.30853305e-05 | 1.07618994 | 40 |
| H6 | 2nd | 0.0112643066 | 1.03428994 | 12 |
| H6 | 4th | 0.00342240456 | 1.91749555 | 36 |
| H6 | 4th(new_2) | 5.10193773e-05 | 1.19937141 | 60 |

この代表条件ではnew_2のsignal誤差は2次・Yoshida4次より小さく、B²もYoshida4次より小さかった。一方、deterministic primitive actionsは2次の5倍、Yoshida4次の5/3倍である。primitive数はcompiled RZ数ではなく、同じqでの誤差差を同精度の総費用削減率へ置き換えない。

全条件の[CSV](../../artifacts/resource_applicability/track_a_h46_new2_diagnostic/2026-10-18/diagnostic_summary.csv)、[H4 exact結果](../../artifacts/resource_applicability/track_a_h46_new2_diagnostic/2026-10-18/H4_run02_results.json)、[H6 exact結果](../../artifacts/resource_applicability/track_a_h46_new2_diagnostic/2026-10-18/H6_run02_results.json)、[終了監査](../../artifacts/resource_applicability/track_a_h46_new2_diagnostic/2026-10-18/run02_completion_audit.json)を参照。

| 系 | 本体wall | 単一worker最大RSS | expm/eigh差 | matrix-free/occupation差 |
|---|---:|---:|---:|---:|
| H4 | 0.175216秒 | 493.062 MiB | 1.61e-15 | 1.36e-15 |
| H6 | 5.843670秒 | 676.305 MiB | 1.49e-15 | 2.01e-15 |

両workerを含むwitness wallは10.009秒。並行実行・5列まとめ伝播・行列/固有分解cacheを利用した実測であり、別campaignへの速度保証ではない。full-Fock scalar閉合、負時間paired Taylor、raw/corrected照合はいずれも事前固定1e-9以下。最終人工6件PASS。

## 限界・次の担当

新しい候補集合として計画を固定した数値診断であり、旧PR union/n32・旧r4等のcampaignへ結果を混ぜない。run02では新規trajectory、quantum sampling、native build/compile、費用model fit、SCF/DF/state生成、旧結果再会計は0。run01では不要なGaussian basis構築を試みてSTOPしたが、transpileは行っていない。full-Fock scalar・prefix・保存状態を保持した有限分布の期待値計算まで完了したが、8桁係数の記号的4次証明、compiled単価、bias headroomを含む実shot会計、同精度総費用、厳密bias/母平均winnerは未検証。

次の作業案はCodexがnew_2のnative adapterと代表条件の同精度compiled費用比較を準備すること。今回これらは実行せず、既存成果・旧SOURCE/grant/ledger/marker/STOPと他jobを保全してmandatory STOP。
