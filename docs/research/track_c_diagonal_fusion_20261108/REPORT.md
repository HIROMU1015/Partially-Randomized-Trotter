# Track C：same-basis run内の対角gate融合

基点 `2745d57bea5c28604739bf9c01338e14e649c467`。専用branch `codex/track-c-diagonal-fusion-20261108`。

**RTE run融合のRZ改善余地は小さかった。** 既知の単一Pauli controlを共通基準にすると、RTE run内の角度集約・parity共有による全PR資源削減は、固定shotでRZ約0.135%、CX約1.67%。Dにも共通のparity共有を適用するとCX削減は合計約8.47%となるが、Fullも大きく改善し、PRとFullの差は縮まらなかった。合理的な限定検証を完了し、mandatory STOP。Track B・詳細GPT科学レビューは開始していない。

## 条件と再利用

Linear H4、1.00 Å、STO-3G、保存DF12・generation order、Partial LD3/Full LD12、保存元H exact groundを利用可能、状態準備費用除外。18round、δ0=.02、energy精度1.5936e-4 Ha、失敗上限.05、位相予算.4 rad。Qiskit1.3.0、Track C compiler A、rz/sx/x/cx、seed17 profileを保持した。D内部BCH係数・既存h/resolution・canonical信号は変更していない。

[共同isometry](../track_c_joint_isometry_20261108/REPORT.md)の639集合/625採用catalogue、singleton/full fallback、source/native区間証拠、価格MC、RPE decoder・cost-sensitive αを再利用した。C-N/C2とWP06a/bの既存価格・共有の履歴、always-partialの負結果も保全する。新しい長round whole-fitを移送していない。

既存`h4_d2_pauli_control._rotation`の汎用central-control部品と、インストール済みQiskitの`GraySynth`を再利用した。別Trackの実験・入力・実行契約を適用したものではない。既知のparity/CNOT合成を新規発明とは扱わない。[Amy–Azimzadeh–Mosca](https://arxiv.org/abs/1712.01859)。今回の研究結果は、固定PR演算子構造への接続と、利益・限界の定量的な切り分けである。

## 既存の融合と、今回追加したもの

| 処理 | 既存状態／今回の扱い |
|---|---|
| 同じbasisの隣接共有・共同isometry | 既に実現済み。referenceへ含め、その削減を再計上しない |
| RTE event由来の相対位相集約 | 既にoccurrence全体のPへ集約。符号・加算順・normalizerを保持 |
| compiler Aの固定inverse cancellation | referenceと全候補へ同じものを適用。既存のCX等の相殺後に価格比較 |
| DF対角primitiveの係数作成 | 既に各Z/ZZが集約され、exact zeroが除かれている。Dに新しい重複回転削減は0 |
| 単一Pauliのcentral control | 既知の関数が存在するが、従来C builderはgeneric controlled-RZZを使用。既知方式を強化共通baselineとして分離 |
| RTE run内の同じPauli角度集約・CZ積のparity化 | 今回追加。logical event列を保持し、同じsource basisの対角演算だけを代数的に再合成 |
| 複数parityの共同合成 | 既存GraySynthを利用し、PR/Fullの両方へ適用 |

候補は4方式に固定した。referenceは既存joint/full/singletonのRZ最小回路、term_parityは各回転を分離した既知control、mergeは同じPauliの角度集約とCZの偶奇相殺、networkはさらに共同parity合成である。各trajectory・D区間で実native RZ、CX、depth、sizeの辞書式順序により**1本の回路**を選ぶ。指標ごとに別回路のminを足していない。

単一controlled-ZZは、既存関数の`CX–CRZ–CX`で実装できる。従来のgeneric controlled-RZZのCCX経由loweringとの差は大きいが、これを新しいPRアルゴリズムの利益とは呼ばない。全18roundの主比較では、Fullにも同じ改善を与える。

## 対角融合の構成と相対位相

最大same-source-basis run内の各rotationについて、元のbinary64角度θとZ/ZZ support Sを読み取る。controlled evolutionの恒等式

`C[RZ_S(θ)] = RZ_S(θ/2) RZ_{anc∪S}(-θ/2)`

でnative parity多項式へ写す。θの和は`Fraction`によるexact rationalで求め、各最終係数を1回だけbinary64へ丸める。Pauli積は各wireのCZの偶奇へまとめる。負のproduct係数・Taylor eventの符号は元のancilla相対位相として残し、θのmoduloや微小角cutoffは行わない。

GraySynthはT placeholderでparity topologyだけを取得する。GF(2)のwire parityを追跡し、要求maskが各1回現れ、最後に線形変換がidentityへ戻ることを確認して、Tを元のRZ係数へ置換する。Qiskit1.3の数値角度経路にあるmod πを通さず、負角やcontrolled 2πの相対−1を保持した。共有package・compiler A自体は変更していない。

run境界の共同isometryは保持し、異なるbasisの演算子を可換扱いしない。論理eventの順序・sampling law・係数符号・確率・normalization・return aggregationは不変更。実装上並べ替えるのは、数学的に可換な同一basis内の対角gateだけである。古い回路をfallbackとして残した。

## 改善可能性：重複はあるが、残るRZの大半はbasis

前回の独立価格384標本を今回のtrainingと明記して再利用し、別seedの独立384標本で評価した。層確率で重み付けしたm17 occurrence（t=.32、n=1024）の独立holdoutは以下の通り。

| 方式 | RZ | CX | 実DAG depth | size |
|---|---:|---:|---:|---:|
| reference | 197582.16 | 33294.75 | 92233.26 | 291975.18 |
| term_parity | 175140.78 | 22988.25 | 68860.38 | 255147.05 |
| merge | 174800.41 | 22413.38 | 68512.32 | 254231.80 |
| network | 174800.41 | 22352.38 | 67977.07 | 254170.80 |

元rotation約1024個に対し、runごとの集約後supportは約853.81個、約16.6%の重複がある。しかし、既知control導入後は1回転が2 RZと軽く、集約後のrotation関連RZは約1708個、全RZ約174800の約0.98%にすぎない。この部分を全て無料にする仮想感度でも、このcellのRZ削減規模は約1%。これは実現可能な方式や一般下界ではない。

CZ productは物理確率で非常に稀で、同cellの元product wire数は約6.70e-5/occurrence、偶奇整理後は約5.42e-5だった。rare層を多くsampleした生頻度を物理頻度とはしない。残る大半のRZはbasis変換であり、今回の融合はnormalizationを変えない。[OPPORTUNITY.json](../../../artifacts/track_c_diagonal_fusion/2026-11-08/OPPORTUNITY.json)に全12cellを保存した。

## Correctnessと新しい誤差会計

同じ保存runを使った11代表×3方式、計33件で全512次元joint operatorとX/Yを比較した。source Frobenius差最大1.117e-14、nativeはjoint全体の共通global phaseだけを除いて最大1.732e-13、X/Y差最大9.881e-15。Fock空間全体の作用を確認しており、one-particle一致だけから推測したものではない。

one-body・DF blockの負時間を含む3条件×3方式、計9件も検証し、source差最大1.866e-14、native差最大1.602e-13。ここでも分枝ごとに別phaseをfitしていない。人工3件は負角・微小角・重複・controlled 2πの相対−1を含む。

新しいnative出力はRZ/CXと既存のCZ/P、basisは既存のRZ/XXPlusYYのままである。過去の128 contextのbasis/native/isometry区間証拠を再利用した。新しい角度和はexact rationalで、rotationごとに|θ|<π<4。binary64丸めunit roundoffをu=2^-53、RTE event総数をNとすると、追加作用素誤差を`2uN + N*2^-1074`で囲える。新raw複素信号disk最大5.269e-9をparent資格へ1回だけ追加した。29 D価格cellの全区間ではhalf角度係数の丸め誤差はexactに0だった。

既存source/native/finite-bit/normalization誤差を0扱いせず保持し、新しいdiskだけを追加する。全policyのrho・βstatは正、36軸のα和≤.05をexact rationalで確認し、192-bit区間算術で整数shotを計算した。PRの最小βstat約.366215976、canonical平均のenergy誤差1.28402e-5 Ha。E0をdecoderへ渡さずbranchが通った。

`.4/(.02×2^17)=1.52587890625e-4 Ha < epsilon`と既存alias条件により、**保存Hermitian-H、exact ground、ideal native、fresh独立finite bitsモデル**で十分条件が閉じる。元連続積分Hの無条件包含・実機成功率・T合成へは拡張しない。PF biasやRTE normalizationを改善したという結果でもない。

## 全18round：強化した両armを比較

候補・h/resolution・係数・α・整数Nを固定後、新しい12cell×2層×16、計384独立価格標本を生成した。holdoutで再選択・再fitしていない。下表のPR referenceは今回と同じtrajectoryによるpaired再評価であり、前回の1.7303e12と僅かなMC差がある。括弧内は1 MC標準誤差。

| RZ優先policy | shots | 総RZ | 総CX | serial depth上界 | size |
|---|---:|---:|---:|---:|---:|
| 既存共同isometry PR | 17024 | 1.74005e12 | 3.59371e11 | 8.94598e11 | 2.61365e12 |
| 既知control＋融合 PR | 17164 | 1.42343e12 (1.37133e10) | 1.97230e11 (1.57153e9) | 5.46130e11 | 2.07780e12 |
| 既存強Full | 14228 | 9.13529e11 | 2.87925e11 | 5.98001e11 | 1.43616e12 |
| 同じ既知control＋融合 Full | 14232 | 5.66091e11 | 1.01463e11 | 2.09332e11 | 8.39091e11 |

PRの全変更を含む減少はRZ18.196%、CX45.118%、depth上界38.952%、size20.502%。paired差のMC SEはRZ1.37368e8、CX1.07064e8で、各armの独立SEを足していない。ただし、この大部分は既知の単一control導入による。新しいFull比はRZ2.514倍、CX1.944倍。両armの差は縮まらず、比較基準を強化するとPRの相対的な不利が大きくなった。

固定した新shotでのablationは次の通り。

| 追加処理 | PR総RZ削減（差SE） | PR総CX削減（差SE） |
|---|---:|---:|
| reference→既知term_parity | 3.16449e11 (5.49939e7) | 1.44449e11 (2.91441e7) |
| term→同じPauli集約・CZ偶奇 | 1.91824e9 (4.06832e7) | 3.22761e9 (8.00356e7) |
| merge→共同parity network | 0 | 1.49900e10 (4.69616e7) |

最後のnetwork CX利益のうち、Dが1.46161e10、RTEが3.73875e8。したがってRTE run固有の追加効果は、既知term基準の全PRに対してRZ約0.135%、CX約1.67%。Dも含む共通parity共有全体ではCX約8.47%。この区別をせず「新しいPR融合でRZ18%減」とは主張しない。FullのDF primitiveは元からuniqueなので、angle mergeだけのRZ/CX削減は0。mergeの順序変更だけではFull depthが悪化する例も保存し、network/fallbackで扱った。

CXを目的とする別α policyも保存した。新PRは16760shots、RZ1.42448e12/CX1.97283e11、対応Fullは14278shots、RZ5.66137e11/CX1.01468e11。異なるpolicyから都合のよい指標を拾っていない。各roundの全4指標・N・資格は[ROUND_RESOURCES.csv](../../../artifacts/track_c_diagonal_fusion/2026-11-08/ROUND_RESOURCES.csv)、全共分散・ablation・decoderは[FINAL.json](../../../artifacts/track_c_diagonal_fusion/2026-11-08/FINAL.json)。

## 加算性・未確定事項・STOP

価格対象はcompiler Aで個別にlowerしたD区間・whole RTE occurrenceを連結する具体的modular回路。RZ/CX/sizeは厳密に加算できる。PR/Fullの1/2step・X/Yの8件でnative wholeとmodularのcount差は全て0、1stepの測定差最大5.593e-13。実DAG depthとserial和は別に記録し、m17実DAG depthやglobal最適whole価格へ外挿していない。m17巨大回路は直接compileしていない。価格MC統計誤差、source/native丸めdisk、物理モデルの条件は別である。

両armの既存step/orderと科学条件は保持し、同じ対角合成・fallbackとα最適化を与えた。新価格で全Full順列/hを再探索した最適性主張ではない。この実行可能なFullが安いという結論は、Full側に追加機会が残ることによって覆らない。

全raw・候補・正負のablationを[RAW_DATA.tar.gz](../../../artifacts/track_c_diagonal_fusion/2026-11-08/RAW_DATA.tar.gz)に元byteで保存し、[RAW_INDEX.json](../../../artifacts/track_c_diagonal_fusion/2026-11-08/RAW_INDEX.json)から照合する。sourceは`track_c_diagonal_fusion.py`/`track_c_diagonal_resources.py`、人工testは`test_track_c_diagonal_fusion.py`。保存値集計は`summarize_track_c_diagonal_fusion.py`、読取検査は`python3 scripts/check_track_c_diagonal_fusion.py`。固定入力hashと完全SOURCE SHAはFINALへ記録した。旧SOURCE/raw/result/grant/ledger/marker/STOPは変更していない。

開始時にCPU affinity64・MemAvailable約443GiBを確認し、独立worker64・BLAS1で並列化した。共有設定・他ユーザーは変更していない。検証はCodex自身のlocal検証で、外部再現・CI・詳細GPT科学レビューではない。

結論として、同じPauliの重複はあっても、効率的なcontrolを使った後の対角RZ費は小さい。既存basis共有を保持する範囲の局所融合をさらに探索してPR優位を得る根拠は弱い。次段階では**RTE normalization・samplingとbasis費を合わせた設計をTrack Bと接続すること**が有力な課題になるが、改善可能性は未検証である。

次の担当：**ユーザー**が既知control/共通対角合成の採択と、Track Bとの共同課題へ進むかを判断する。**Codex**は新しい明示指示があった場合のみ次範囲を実行する。今回の研究は成果公開・remote SHA/tree/内容照合後mandatory STOP。
