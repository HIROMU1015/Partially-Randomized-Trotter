# H4/H6：4th(new_2)の追加検証

三方式とも同じPR分割の比較であり、決定論B0/B0+Iとの比較ではない。前回の数値診断で不足したnative意味論、compile後の演算照合、同じsignal精度でのshot重み付き費用、8桁係数の次数を追加検証した。前回成果7198bbde27317fb0245bdaeeb6b0966a31096e6cを保全した独立branch。

対象はlinear H4/H6、1Å、STO-3G、保存rank12/19、保存状態。H4 prefix3、H6 prefix5、signed generation順、T=.8、finite RTE K4、各outerのtail32 short stepsを維持。S2 q4、Yoshida q2、new_2 q1は前回のq1/2/4/8格子から選び、今回の費用測定前に固定した代表点である。各公式の費用最適qや全geometryの探索ではない。

H4科学SOURCEは`687679a4ae20e4661bde1187dd8ab92ae176bd44`、H6並列executorのSOURCEは`f0dd2a9aa4e011b204c4ba6c51a050dff09bc471`、providerはc31ef424175c02379eafebe8fc6c7548c1d9aabe。8桁係数、科学条件、compiler basis rz/sx/x/cx、opt1、seed17は変更していない。旧provider/既存venvには書かず、専用process内でGaussian basisの明示的XY分解を選ぶ。ordinary controlの新費用系列であり、旧denseや別controlのcostへ混ぜない。

## 係数の次数

既存の5係数は[.40899193,.42008729,-.65815844,.42008729,.40899193]。8桁decimalの和は1、立方和は7.346136322761508e-9。一般の対称S2合成に必要な立方和0を厳密には満たさない。実際のbinary64値も有理数として和は1、立方和は約7.3461363139e-9であり、結論は同じ。

70桁の非可換toy(.73X-.42Z、T=.8)では、q4→8の観測次数4.0029、q32768では1.9951、q1048576では1.999995。中間では誤差項の相殺により非単調な観測次数があり、図は全点を示す。これはdecimal係数の誤差を分離した高精度toyであり、H4/H6 native binary64を巨大qまで実行した結果ではない。係数を補正せず、厳密4次の公式としては扱わない。

## Native・compiler検証

人工14件＋並列schedule4件、計18件はPASS。複素Gaussianのexterior-power minors全sector、12modeのvacuum/one-particle、負時間、controlled low/high、order0/2/4のproduct/relative phase、係数丸め、shot headroom、STOP条件を検査した。

実H4/H6では各群の代表1軌道を、回路を使わないladder-algebraのparticle-number全sector(H4 dim70/H6 dim924)と照合した。元の保存状態sector36/400から広げるのは、途中のspin-sector成分を捨てないためであり、初期状態・物理系は変えない。代表軌道のHadamard cosine/sineとcompile前後の全vectorを固定1e-9で照合した。両time符号のforced order0/2/4はnative回路と独立oracleを照合し、forced回路自体の追加compileはしていない。forced fixtureは費用標本へ入れない。

初回SOURCE8b0f6cf…のgeneric4x4 unitary経路ではcompile後にvector差1.32e-6を検出した。全6childを停止・wait回収し、過去STOPを保持。全体位相を除いても差が残った。許容差を緩めず、QRの各SU2を明示的XXPlusYY＋number Phaseへ分解する修正を行い、人工14件を再PASS。最終結果は修正後のH4 run02完了分とH6 run03並列分から作り、初回generic unitary系列は混ぜない。

## 同精度の費用

signal ε=.001、各軸ε/sqrt(2)、alpha=.025、empirical allowance1e-9、logB margin1e-12、P=0。前回固定した有限RTE期待signal/biasと今回の実compiled RZを結合。各群の8独立軌道でcosine/sineを同じevent列から生成し、Nreal×Ccos＋Nimag×Csinを軌道ごとに集計し、その平均・SEを報告する。量子measurement shot自体は実行しない。

| 系 | PR公式 / q | 必要shot合計 | RZ/軸の標本平均 | shot重み付きRZ平均 ± SE |
|---|---|---:|---:|---:|
| H4 | 2nd / 4 | 130,455,177 | 38,143.500 | 4.97602e+12 ± 7.88e+10 |
| H4 | 4th / 2 | 53,673,312 | 42,154.750 | 2.26259e+12 ± 3.45e+10 |
| H4 | 4th(new_2) / 1 | 39,887,751 | 33,123.125 | 1.32121e+12 ± 1.72e+10 |
| H6 | 2nd / 4 | 515,509,554 | 138,253.500 | 7.1271e+13 ± 1.41e+12 |
| H6 | 4th / 2 | 83,141,075 | 163,960.125 | 1.36318e+13 ± 7.68e+10 |
| H6 | 4th(new_2) / 1 | 46,572,752 | 128,686.000 | 5.99326e+12 ± 2.26e+10 |

| 系 | new_2の対S2 q4削減率 | 対Yoshida q2削減率 |
|---|---:|---:|
| H4 | 73.45% | 41.61% |
| H6 | 91.59% | 56.03% |

[6群の正確なJSON/CSV](../../artifacts/resource_applicability/track_a_h46_new2_native/2026-10-18/representative_cost_summary.json)、[費用図](../../artifacts/resource_applicability/track_a_h46_new2_native/2026-10-18/representative_cost.png)、[係数次数図](../../artifacts/resource_applicability/track_a_h46_new2_native/2026-10-18/coefficient_order.png)。削減率は同一PR分割の固定代表点の小標本比較であり、全候補の費用最良やPR対決定論の新しい優劣を認定しない。

## 範囲と残る検証

この比較は保存状態・単一T・代表prefix/qの有限標本診断。経験的allowanceは厳密bias保証ではなく、n8のSEは未観測の希少層の寄与を保証しない。forced rare-eventの回路意味論と、不偏なcompiled費用母平均の証明は別である。全公式の最適q、全geometry、RPE全体/energy費用、母平均winnerは認定しない。旧n32/r4のcampaignへ混ぜない。

採用96軸、数値検査222件、最大差1.5737951e-10で固定1e-9内。H4の3groupとH6の24shardはnative exit0・wait回収、全旧/新childの残存0を2回確認。H6を24軌道workerへ分け、同一seed_SOURCE687679a4…を保って引継ぎ、旧H6の正常な途中記録をPASSへ書き換えず手動handover状態で保全した。重複13軸はevent digest・回路fingerprint・metrics・compiler hashが完全一致。H4を再実行せず、旧H6の途中単価は平均へ混ぜていない。

並列H6のwitness wallは621.056秒（約10分21秒）。最大全worker同時観測RSSは27.058GiB、各worker観測thread1。これは実測であり、他campaignの速度保証ではない。source/input/provider closureのbyte同一性、旧完了repo clean、管理rootの既存958 staged削除とHEAD/status同一を確認。共有環境・既存venv・他job・GPU・SCF/DF/state生成の変更/呼出0。AI subagentなし、検査はauthor自身。

実compile entry累計121（初回10＋切分け1＋run02の62＋H6並列48）、採用96。失敗/途中のentryを返却・削除していない。元run02全体はhandover中断のまま、H4完了だけを独立に受け入れた。[終了監査](../../artifacts/resource_applicability/track_a_h46_new2_native/2026-10-18/final_completion_audit.json)、[handover受領](../../artifacts/resource_applicability/track_a_h46_new2_native/2026-10-18/parallel_handover_receipt.json)、[読取専用checker](../../artifacts/resource_applicability/track_a_h46_new2_native/2026-10-18/verify_saved_result.py)。

branchは`track-a-h46-new2-native-validation-20261018`。この追加検証はlocal commitへ固定し、pushは実行していない。旧結果・原稿・既存grant/ledger/marker/STOPを改変せず、次campaignを起動しない。

次の作業案：ユーザーが、今回の限定比較を踏まえて厳密4次係数版・費用最適q/より大きい独立標本へ進む必要性を判断する。Codexは追加指示後に必要な範囲を実装・検証する。自動で拡張せずSTOP。
