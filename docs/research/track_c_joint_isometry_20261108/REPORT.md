# Track C：same-basis runの共同isometry

基点 `10bf5a4813559cea807fcd8c50f3b966407747d1`。専用branch `codex/track-c-joint-isometry-20261108`。

**共同isometryによるPR内の資源改善を確認した。強Full PFの逆転には至らない。** 独立価格384標本のpaired比較で、RZ優先policyの総RZを3.355%、総CXを1.789%、sizeを3.190%削減した。一方、modular serial depth上界は0.696%増加した。固定shotでのbasis改善の約94.4%は複数applicationの共同共有に由来し、singletonのlayout変更だけの利益と分離できた。検証を完了しmandatory STOP。

## 固定条件と再利用

Linear H4、1.00 Å、STO-3G、保存DF12・generation order、Partial LD3/Full LD12、保存元Hのexact groundを利用可能とし、状態準備費用は除外。18round、δ0=.02、energy精度1.5936e-4 Ha、失敗上限.05、位相予算.4 rad。Track C compiler A、Qiskit1.3.0、rz/sx/x/cx、seed17 profileを維持した。Aはdefinition loweringと固定inverse cancellationを行い、微小角cutoff・近似交換最適化を使用しない。B2 P1の条件は混在させていない。

[直前のbottleneck研究](../track_c_bottleneck_20261108/REPORT.md)のsingleton builder、stable partial QR、compiler/source区間検証、価格MC、cost-sensitive α、RPE decoder、D価格・信号・資格を再利用した。既存D内部BCH係数と各roundの選択済みh/resolutionは保持し、Hamiltonian・state・DF・RTE sampling law・return aggregationを変更していない。元のeventオブジェクトを保持し、回路実装のsupportラベルだけを写す。

旧WP06a/bのsingleton効果、一般的interval-DPの負結果、always-partial QRが共有を壊して悪化した保存結果も参照・保全した。過去の価格proxyを新族へ無検証で外挿していない。新規性は一般的なQR/Givens/isometryの発明ではなく、このPR回路で共有runの列要求をまとめて資源へ接続した限定的な改善である。

## 構成：spanではなく各列のprojectorを保存

元の最大same-source-basis runに含まれるZ/ZZ supportの和集合をSとする。既存`partial_ops`で、元one-particle basis Uの列U[:,S]を指定した入力モードへ写す隣接Givens/RZ回路Vを構成する。全8モード上でunitary gateの積を生成するため、未指定列もunitary completionを持つ。非unitaryな長方形行列を量子gateとして挿入していない。

列順を昇順/降順、入力モードの配置を先頭側/末尾側とする**4つの事前指定layout**だけを比較した。無制限な順列探索、微小角の切捨て、係数の再fitは行っていない。mapping fに対し、各i∈Sのprojectorを保持するので、Fock空間で

`Γ(U) n_i Γ(U)† = Γ(V) n_f(i) Γ(V)†`

となり、個々のZ、ZZ、これらの積・rotationを保持する。列のspanが同じという条件だけでは足りない。run全体で同じVを1組だけ使い、各central applicationのsupportをfへ写す。ZZの2ラベルは既存APIに合わせて昇順へ正規化する。event順序・符号・確率・normalizer・相対位相は保持する。production builderのcentral-application処理を再利用し、event由来のancilla相対位相はoccurrence全体で集約して計上した。

前回の独立holdout384標本を、**今回のtraining**として再利用した。観測列集合と全singleton supportから639集合を登録し、元full-basis pairより実native RZが小さい625集合を採用した。rank1/2/3/4/5/6の採用数は28/190/146/172/59/30。rank7/8、未登録集合、高価な構成はfull共有へ戻す。さらにtrajectory全体について、従来full/singleton最小回路をfallbackとして保持し、実RZが小さい回路1本を選ぶ。CX/depth/sizeを別々の回路からminで拾っていない。

## 改善可能性と実価格

trainingの生標本には15443 run、56162 application、最長65 application/runがあった。rare層を意図的に多く含めるため、この生頻度を物理確率とはしない。層確率で重み付けしたm17 occurrence（t=.32、n=1024）の期待値は、約278.63 run、うちsingleton160.13、複数application118.50。共同部分basisを使える複数runは約68.44だった。残る長いrun・高rankの共有は維持する。独立holdoutで新たに現れた未登録集合は同cellで約1.94 run/occurrenceで、catalogueを追加せずfallbackした。

代表runのstandalone RZ価格を示す。同じrunの4指標をrawのABLATIONへ保存した。per-runの価格は比較診断であり、そのまま全RPE総Gへ加算していない。

| application数 / 要求列数 | 既存full共有 | 個別partial QR | 共同isometry |
|---|---:|---:|---:|
| 2 / 3 | 1020 | 976 | 686 |
| 3 / 4 | 1044 | 1472 | 856 |
| 7 / 6 | 1125 | 3441 | 1121 |

列数が少ないrunでは利益があり、6列の例では利益がほぼ消える。個別partial化の繰返しは共有を壊し、かえって重くなる。第1候補の昇順/先頭layoutだけでは支配cellのRTE RZ約4.6%減・DAG depth約2.2%増。4候補を与えたtrainingでは、m17 RTE RZ4.785%減、CX3.086%減、DAG depth1.278%増、size4.621%減だった。**これらはtraining上のoccurrence価格であり、独立holdoutの全RPE値と区別する。**

孤立basis templateによる理想的な削減可能性は[RUN_ANALYSIS.json](../../../artifacts/track_c_joint_isometry/2026-11-08/RUN_ANALYSIS.json)に保存した。実際の採用回路では、complete RTE occurrenceを直接compileしてfallback・central演算・相対位相・compiler残差まで計上した。basis単体の潜在削減を実現済み総資源と扱っていない。

## Fock・controlled・X/Y・native correctness

625採用集合について、全256次元Fock空間の各要求Z作用素とcompletionのunitarityを数値検証した。Z作用素のFrobenius誤差最大2.137e-14。ZZや積は検証した各Zの積として保持する。人工3件も、一般の複素列集合・反射layout・逆向きsupportのcontrolled ZZ product/rotationを確認した。

別に192-bit区間算術で元/新one-particle列の誤差を囲い、4電子sectorで`||ΔZ||≤4N_e||Δcolumn||`、ZZのtelescopingを用いた。involution誤差上界の最大は3.120e-14。native中間gateのsector射影は行わず、128 context・5380 unique loweringのfull local matrixを区間評価した。新basis pairの最大誤差上界は4.306e-14。従来singleton fallbackも含む最大値を採り、全trajectoryを覆う追加diskとした。

11種類の短runで、全512次元already-controlled operatorとX/Yを検証した。source旧/新のFro差最大1.830e-14、source/nativeは**joint全体の1個の共通global phase**を除いて最大1.896e-13、同じphaseを用いたinactive branchのIとの差最大1.396e-13。X/Y差最大1.166e-14。分枝ごとに別phaseをfitしていない。実際にnativeが共通−1を持つ例があり、absolute operator差を要求した初期テストはcompiler Aの既存契約より過剰だった。契約通りのjoint共通phaseと測定値で再確認した。

さらに元S2と最終D調整0321の1/2step・X/Y、計8件でwholeとmodular回路を比較した。RZ/CX/size差は全て0、1stepの旧/新native測定差最大5.130e-13。実DAG depthと部品depthのserial和は別々に記録した。whole一致を全長の最適性定理へ拡張していない。

## 全18roundの独立資源評価

catalogue・source点・α・整数Nを固定してから、使用する12 occurrence cell ×2層×16の**新しい独立価格384標本**を生成した。各標本で旧full/singletonと共同方式を同じtrajectoryで直接compileした。holdoutを使ったcatalogue拡張、候補選択、α再fitは行っていない。

下表はRZを目的とするpolicy。旧方式も今回の同じ独立trajectoryで再評価したpaired対照であり、前回の保存値1.7906e12/3.6485e11に対応する新標本推定である。括弧内は1 MC標準誤差。

| policy | shots | 総RZ | 総CX | serial depth上界合計 | size合計 |
|---|---:|---:|---:|---:|---:|
| 既存singleton方式 | 17054 | 1.79034e12 (1.65495e10) | 3.64837e11 (1.81472e9) | 8.83374e11 | 2.68522e12 |
| 共同isometry＋α再配分 | 17024 | 1.73026e12 (1.47894e10) | 3.58311e11 (1.62636e9) | 8.89520e11 | 2.59956e12 |
| 強Full PF | 14228 | 9.13529e11 | 2.87925e11 | 5.98001e11 | 1.43616e12 |

paired削減はRZ6.00742e10（差SE2.25175e9）、CX6.52632e9（差SE2.41336e8）。RZ3.355%減、CX1.789%減、size3.190%減に対し、serial depth上界は0.696%増加する。全指標で改善したとは主張しない。強Full比はRZ1.894倍、CX1.244倍で、差は縮まったが逆転していない。

CXを目的とする別policyも保持した。旧/新shots16534→16520、新総RZ1.73471e12、CX3.58170e11、serial depth8.89938e11、size2.60553e12。対応する強FullはRZ9.12635e11/CX2.87625e11。異なるpolicyから都合のよい列を組み合わせていない。

固定した新shotでのablationでは、singletonのlayoutだけによるRZ削減3.29588e9（差SE3.04848e8）、**複数applicationの共同共有による追加削減5.56399e10（差SE2.20545e9）**だった。後者がbasis改善の94.4%。共同共有単独のserial depth差は悪化方向であり、こちらもrawへ残した。このablationは凍結済みpolicyの診断であり、結果を使ってpolicyを選び直していない。

各round・各軸のNと4価格指標は[ROUND_RESOURCES.csv](../../../artifacts/track_c_joint_isometry/2026-11-08/ROUND_RESOURCES.csv)、共分散・paired差・decoderは[FINAL.json](../../../artifacts/track_c_joint_isometry/2026-11-08/FINAL.json)。共有cellの全round重みをまとめてから、独立cell/層ごとの共分散を合算した。legacy helperのpooled-index baseline SEはFINALでこの独立cell式へ訂正し、G・policy・Nは変更していない。MC SEとnative/source誤差、モデル条件を混同しない。

## RPE保証・公平性・限界

新しい一様置換diskは、既存の未supplement論理資格へ1回だけ追加した。最小rho下限.572700145、最小βstat下限.366215983、m17追加disk上界5.286e-6。36軸のα和≤.05をexact rationalで確認し、192-bit区間算術で整数Nを計算した。元のcanonical複素信号・normalization・branchを保持し、decoderへE0を渡さず平均energy誤差1.28402e-5 Ha。新実装の平均は追加disk内にある。

`3×.4<π`と`.4/(.02×2^17)=1.52587890625e-4 Ha < epsilon`から、保存Hermitian-H・exact ground・ideal native・fresh独立finite bitsモデルで全18roundの十分条件が閉じる。元連続積分Hの無条件包含、実機成功率、ノイズ、状態準備、T合成へは拡張しない。今回shot数の変化はα配分と資格予算によるものであり、PF biasやRTE normalizationを改善した結果ではない。

Fullにも13 blockの全8列に同じ4 QR layoutを提示したが、RZ改善0件だった。Fullの既存S2/Yoshida4・24 first-four順序・選択済みh・cost-sensitive αを保持した。global最適なFullや全13!順序探索とは主張しない。

資源は、compile済みD区間と**whole RTE occurrence**を連結する具体的modular実装の`Σ N E[C]`。RZ/CX/sizeはその回路で厳密に加算でき、m17 whole fitの外挿を必要としない。total depthはserial上界で、m17実DAG depthではない。m17巨大回路全体の直接compileは行わなかった。basis単位の予測と実occurrence MCは分離した。

## 成果保全と次の仮説

全raw・中間・初期layout・修正前後の価格・負結果を[RAW_DATA.tar.gz](../../../artifacts/track_c_joint_isometry/2026-11-08/RAW_DATA.tar.gz)に元byteで保存し、[RAW_INDEX.json](../../../artifacts/track_c_joint_isometry/2026-11-08/RAW_INDEX.json)で照合する。全1567ファイル。旧SOURCE/結果/grant/ledger/marker/STOPは変更していない。JSON型・support順・共通phaseテストの技術修正と、最初の201標本の不変hashも保存した。

実装は`src/trottertracks/track_c_joint_*.py`、人工testは`tests/test_track_c_joint_isometry.py`、保存値のみの集計は`scripts/summarize_track_c_joint_isometry.py`、読取検査は`python3 scripts/check_track_c_joint_isometry.py`。各moduleの`python -m`入口は固定cacheと前回archiveの読取コピーを使う。完全SOURCE SHA・固定入力hashはFINAL。private cacheは公開せず、以前の公開DF/state参照を継承した。

64 CPU affinityと約476 GiB以上のMemAvailableを確認し、独立worker64・BLAS1で計算した。共有設定や他ユーザーを変更していない。テスト3件・保存結果checkerはCodex自身のlocal検証であり、外部再現・CI・別reviewerの独立レビューではない。詳細GPT科学レビューは開始していない。

高rank runの共有とRTE normalizationが残るため、この有限候補だけでFullとの差を埋める見込みは小さい。次に価値があるTrack C仮説は、**same-basis runのcentral対角演算を厳密に合成し、単なる列和集合より小さい作用を構成できるか**、またはspin/局所性を意識したcompletionでdepth悪化を抑えられるかである。どちらも今回未検証。RTE sampling lawやreturn aggregationの変更は別担当範囲なので実施しない。

次の担当：**ユーザー**がこのPR内改善を採択し、次の研究仮説を選択する。**Codex**は新指示がある場合のみ次範囲を実行する。今回の成果を専用branchへ公開しremote SHA/tree/内容を照合後、mandatory STOP。別研究へ自動移行しない。
