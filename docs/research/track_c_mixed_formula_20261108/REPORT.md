# Track C：混合時間発展公式の検証（2026-11-08）

今回の5段対称分割族では、**作用素誤差の低減は確認したが、基底energy bias・必要shot数・観測した回路価格の改善には結び付かなかった**。a=1/4は既存S2の半刻み2回と有限RTEまで同一であり、新アルゴリズムの改善には数えない。2MNと交換子一方を消す係数は、その同値対照に対して全18roundの同条件比較でshot数を下回らず、96 paired標本の回路価格も同じだった。この族を追加調整して救済せず、負結果を保存してmandatory STOPする。

これは固定H4・検討した係数/刻み/RTE分割での結論であり、PR全般の不可能性や強PFの総価格優位を証明しない。新公式の18round native誤差保証と絶対総Gは未資格である。

## 固定条件と再利用

Track Cのlinear H4、1.00 Å、STO-3G、DF12生成順、元Hamiltonianの保存基底状態、8 spin orbital・4電子sector70を維持した。partialはLD3（D4 block、R218 component・7 bases）、強対照はtail-free LD12（13 block）。delta0=.02、m=0..17、epsilon_E=1.5936e-4 Ha、alpha総和=.05、位相予算=.4 rad、状態準備費除外。B2 P1の科学条件は使用していない。

基点はC2-A `ce84fa8e2d2f1d487df3f05e63d519ebb03f83ad`。保存common.pklのSHA256は`68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2`。新SCF・DF fit・physical ground solveは0。aggregate Dの固有分解のみexact-D診断用に追加した。

Python3.12.3 / NumPy1.26.4 / SciPy1.14.1 / Qiskit1.3.0。Track Cのreference compiler Aを使用（rz/sx/x/cx、定義ベース、数値最適化・angle cutoffなし、seed17、確率的passなし）。B2のopt1/seed7や旧Cの不適格opt2は使用しない。制御はlowering前に付け、joint全体の共通global phaseだけを許容する。

既存の`track_c_h4`のblock spectra・tail_operator・P3平均と正規化、`track_c_h4_price`のconditional_flags・X/Y wrapper、DFPartialS2の_append_block、DFRTE event builder、compiler A、既存RPE decoderを再利用した。新規実装は係数族の短いsequence、三次log、比較・会計・測定意味の検査に限定した。旧SOURCE/raw/grant/ledger/marker/STOPは変更していない。

## 先行結果からの出発点

| 既存結果 | 今回への含意・保存元 |
|---|---|
| C-N・compiler A価格 | S2の2304 paired標本でfull=強加法vector。短価格q1/2→q8モデルは最大6.09%残差、m17移送未資格。`720416c5b88b227cf76ff7aa729d91a0a98ed58b` |
| C-S | 選択した残差方式はlog range約386、shot負担>10^338。一般的no-goとはしない。`a78054dc`系列 |
| C-P | ほぼidentity、signal差約7.66e-15、shot改善未観測。同上 |
| C2-B | 約.514%は強baselineの最適性未確定のため優位性未確定。`602ca4b67b1037e221e910a2733444630c5add62` |
| compiler A18roundと保証A | 局所変換誤差・event phase・SOURCE入力誤差を分けた条件付き固定行列モデルの保証。別角度へそのまま移さない。`3258a89f225d9530fba38e9d2343bc0165dd6d7d`、`720416c5…` |
| C2-AとB2 P1・長roundモデル | C2-Aの有限RTE信号/shot処理を再利用。B2は技術的な低メモリ化・価格モデルの参考に限定し、入力を移送しない。既存モデルを今回の回路族の絶対価格に流用しない |

## BCHと改善機構

A=-iH_D、B=-iH_Rとして、既知の対称5段族

`U_a(h)=D(ah) R(h/2) D((1-2a)h) R(h/2) D(ah)`

の自由非可換語を有理数・記号aのまま三次まで展開し、全三次語を検算した。対称性からlogは奇数次で、

`log U = h(A+B)+h^3{α[A,[A,B]]+β[B,[A,B]]}+O(h^5)`

`α=(1-6a+6a²)/12`, `β=(1-6a)/24`。

`C_D=[H_D,[H_D,H_R]]`, `C_R=[H_R,[H_D,H_R]]`なら、`H_eff=H-h²(αC_D+βC_R)+O(h⁴)`。a=1/2はD/2–R–D/2、a=1/4はS2半刻み2回、a=(3−sqrt3)/6はα=0、a=1/6はβ=0。2MNのa=.1931833275037836は既知係数であり、新発明ではない。[Omelyan–Mryglod–Folk](https://arxiv.org/abs/cond-mat/0110438)、[Takaishi–de Forcrand](https://arxiv.org/abs/hep-lat/0505020)を参照。

exact groundでは`⟨C_D⟩+⟨C_R⟩=⟨[H,[H_D,H_R]]⟩=0`、かつ`<C_D>=-2Σ_n(E_n−E0)|<n|H_D|0>|²≤0`。従って主要mixed energy係数は`−(α−β)<C_D>`、

`α−β = 1/96 + (a−1/4)²/2 > 0`。

**この5段族では負のaを含めても、一般のground leading mixed biasを消せず、exact-Dでの最小はa=1/4である。** 作用素ノルム最小化と基底位相最小化は異なる。これはD内部PFのある全公式の最適性定理ではない。

実装では既存片側D sweepをF、逆sweepをBとして、

`F(ah) R(h/2) B((1−2a)h/2) F((1−2a)h/2) R(h/2) B(ah)`

を用いた。ゼロ段を削除し、隣接する同じD block/Rを結合してからRTEを接続する。中央をB/Fとすることでa=1/4の半刻み同値を保ち、a=1/2では中央ゼロによりR二段が元の一段へ戻る。scalarは全体時間に対してancilla相対位相として一回計上した。

## 固定H4での主要誤差

`<C_D>=−.07277680776710108`、`<C_R>=+.07277680776709988`、和は約−1.18e-15。保存状態のmodel residualは3.89e-14、modelと保存physical_HのFro差は3.15e-13。交換子の相殺は単なる零への丸めではない。

下表は実際のmultiblock sequenceの三次logから得たH_effのh²係数。D内部はRを代数的にゼロにした同じsequenceの寄与、追加mixedは残差である。物理Hamiltonianを変更した科学runではない。

| 公式 | ground H2全体 | 外側D/R | D内部 | 追加mixed内部 | H2作用素ノルム |
|---|---:|---:|---:|---:|---:|
| 元S2 | .01334023 | .00303237 | .01194830 | −.00164044 | .06240266 |
| 2MN | .00385183 | .00087556 | .00322115 | −.00024488 | .00685863 |
| α=0 | .00357450 | .00081252 | .00305502 | −.00029304 | .00719389 |
| β=0 | .00444674 | .00101079 | .00361823 | −.00018227 | .01294404 |
| a=1/4・半刻み対照 | .00333506 | .00075809 | .00298708 | −.00041011 | .01560067 |

2MNの作用素係数は同コスト半刻み対照より約56%小さいが、ground係数は約15.5%大きい。α=0もground係数は約7.18%大きい。元S2からの見かけの低減には追加D/R段の価格が伴う。D内部PFは大きく、外側交換子だけを最適化しても全体energy誤差を説明できない。h=.02/.01/.005の位相差と作用素差、exact-D診断、人工3×3行列の収束で検算した。

## 有限RTE・全18round・RPE

初期m=0,4,8で684行を計算し、作用素誤差改善の残った2MN/α=0だけを新係数候補として全18roundへ進めた。β=0は追加調整せず打切り。全比較は同じh menu=.01,.02,.04,.08,.16,.32、RTE resolution=1,2,4,8,16,32を使用。半刻み最小h=.01の対照を含めるため、元S2だけh=.005も追加した。

各R occurrenceのevent数は`max(1,ceil(|t_R| resolution/.02))`。resolutionはevent総数ではない。K2のP3平均を積み、normalizer B、有限打切り残差、conditional float64/input diskを分離した。RTE occurrence増加と負時間をabs時間で全て計上し、追加samplingを無料にしていない。

全18roundは3564行、実条件重複除外後3244条件。2MN/α=0はそれぞれ588実条件で比較し、双方feasibleの576条件中、shot数を半刻み対照より減らした条件は0。2MNは167条件で増、409で同数。α=0は145で増、431で同数。candidateだけがfeasibleになる条件も0。全588条件でPF位相も対照を下回らなかった。これらは同一h/rでの比較であり、全パラメータの最適性を主張しない。

初期SCREENのprimitive数はliteralな各stepの数だった。後続FULLでは全族に刻み境界の同D結合を適用し、a=1/4と旧h/2のD呼出数を一致させた。初期rawは修正せず保存した。`shots×(D指数+RTE event)`は候補選択用surrogateであり、異種native価格の合算ではない。

次はそのsurrogateで各roundのmenu内選択をした**条件付きsource診断**。合計shot数最小化やnative G最小化の解ではない。端数ceilとsequenceごとの浮動小数diskも含む。

| 公式 | m17 h / resolution | m17 PF位相rad | m17 rho下限 | m17 shots/軸 | 全round X+Y shots |
|---|---|---:|---:|---:|---:|
| 元S2 | .02 / 32 | .01398844 | .57270135 | 626 | 11064 |
| 半刻み対照 | .04 / 32 | .01398844 | .57270135 | 626 | 10788 |
| 2MN | .04 / 32 | .01615616 | .57270135 | 633 | 10894 |
| α=0 | .04 / 32 | .01499286 | .57270135 | 629 | 10836 |
| partial Y4 | .08 / 32 | .00081557 | .08591307 | 26090 | 68398 |
| Full S2 LD12 | .04 / 0 | .05606639 | .99999989 | 256 | 8560 |
| Full Y4 LD12 | .16 / 0 | .01306066 | .99999992 | 205 | 7458 |

正係数の新族と半刻み対照は同条件で同一のRTE時間・normalizer・event数を持つ。m17選択点のlogB=.557390826、RTE event数4194304。partial Y4は負時間に伴い18481152 event、logB=2.45441923となり、位相改善が大きなshot増に転じた。Full PFにはRTE減衰がないが、これだけでnative価格の勝者は決まらない。

既存のnearest-branch decoderへ有限RTE複素信号の位相と2^mだけを渡した。E0はdecode後の誤差判定にのみ使用。選択した7族は全round正しいbranchで、平均信号の最終energy誤差は全てepsilon_E内だった。2MNは6.1631e-6 Ha、α=0は5.7193e-6、半刻み対照は5.3362e-6。原Hamiltonianの連続的入力誤差を新たに保証したわけではない。

条件付き計数式は`N=ceil(4 log(1440)/(rho² sin² beta_stat))`、各36軸へのalpha=.05/36。全signal diskが保証され、fresh独立shotが実装されれば、3β<πのbranch条件とβ/T17=.000152587890625 Haからunion-bound成功率≥.95となる。**今回の新角度族はfloat64 source診断であり、native成功率≥.95の新定理ではない。** finite-bit CDFの既存proposal budgetも記録したが、新しい区間証明へ未統合。量子shot実行・成功率MCは行っていない。

## 回路価格と総資源

h=.04、outer step n=1/2/8、resolution=1/32、通常/少なくとも1個order2の層をそれぞれ8標本、合計96 paired標本。3族でflags・成分indicesを共有し、正しい層確率を掛けた。X/Yは別々にwhole compileし、measurementを保持した。6価格セル、576 measured wrappersで**RZ・CX・depth・sizeが全paired標本で一致し、nativeのgate名・wire列も一致**した。paired差のmeanとSEは0。期待価格自体のMC SEはゼロではなく、全24次元共分散を保存している。

例：n=1/r=1/Xの推定期待価格はRZ22892.91、CX6554.02、depth13809.42、size35551.98。n=8/r=32/XはRZ313755.75、CX72245.50、depth155497.13、size475760.75。いずれも3族共通。RZ/CX/depthを互いに加算していない。T合成は未実施。

局所correctnessは24 X/Y source/native検査でprojective state差最大2.38e-13、測定差最大4.46e-13。さらに元D spectraとsigned DF component/event_unitaryから独立に組んだtrajectory演算子を、full controlled sourceのX/Yと比較した（SOURCE_SEMANTICS.json、24件の最大差2.89e-15）。共通global phaseの許容によってcontrolled相対位相の誤りを隠していない。

n1/2でaffine期待価格を仮置きして独立n8を観測すると、Xの相対残差はr1でRZ .370%、CX .128%、depth .0035%、r32でRZ3.366%、CX1.550%、depth1.793%。MC共分散から伝播したcombined SEとの比は全て約1.4以下で、低標本数では系統誤差との分離ができない。事前許容幅・長roundの誤差envelopeも未設定であり、このfitをm17へ移送しない。

`G_j=Σ_{m,b} N_{m,b} E[C_{j,m,b}]`のj=RZ/CX/depth/sizeを個別に考える。同じh/rで長回路価格まで同一なら、各cellのNが対照以上なので候補のGも対照以上となる。短価格では価格同一を観測したが、**長回路全体の期待価格・系統誤差は未資格なので、絶対Gや厳密な総価格支配は数値として確定していない**。今回の候補に科学的改善が見られないため、全m17直接compile・追加の大規模価格MC・新角度の完全interval保証へは進めなかった。旧compilerの省略誤差をゼロ扱いして総Gを決める処理は行っていない。

## 成果物とSOURCE

[機械可読最終結果](../../../artifacts/track_c_mixed_formula/2026-11-08/FINAL.json)、[全round CSV](../../../artifacts/track_c_mixed_formula/2026-11-08/FULL_ROUNDS.csv)、[全round選択・SOURCE](../../../artifacts/track_c_mixed_formula/2026-11-08/FULL_SUMMARY.json)、[paired価格と共分散](../../../artifacts/track_c_mixed_formula/2026-11-08/PAIRED_PRICE_SUMMARY.json)、[RPE branch](../../../artifacts/track_c_mixed_formula/2026-11-08/RPE_CHECKS.json)を参照。

[raw bundle](../../../artifacts/track_c_mixed_formula/2026-11-08/RAW_DATA.tar.gz)に、初期screen・全round・paired全標本・BCH演算子NPZ・集計JSONを元のbyte列のまま保存した。RAW_INDEX.jsonのSHA256で各展開ファイルを照合できる。非公開common.pklは含めず、科学入力identityと既存保存NPZの参照を保持した。

| 段階 | 公開scientific SOURCE |
|---|---|
| BCH・初期screen | `ba62ea5556bab33eb2fc58b4417603ac5d15f6da` |
| 全18round・境界結合数 | `42d8c95b26d74629a455611aef0d8e587a501c6b` |
| paired価格・RPE診断 | `ce2d440f4f8199d5b35cce3426bd93c60b5beaaa` |
| 独立trajectory測定意味 | `3971408636dc31a9063cee4c0dd13f490eb4f73e` |

64 CPU affinityと約473 GiBの空きRAMを観測し、独立taskに応じ初期48・全round43・価格64 workerを使用した。各workerのBLASは1 threadとし二重並列を防止。価格wall136.14秒、観測worker peak RSS最大495200 KiB。共有設定・他利用者プロセスへの操作なし。実測値は今回の環境の技術値であり、アルゴリズムの一般的速度保証ではない。

## 再現・読取検査

保存結果の検査は`python3 scripts/check_track_c_mixed_formula.py`で完結し、H4の再計算を行わない。rawは`tar -xzf RAW_DATA.tar.gz -C <空の展開先>`で展開し、RAW_INDEX.jsonと照合できる。4件の人工テストもPASSした。

科学実行の各SOURCEでは、専用の`TRACK_C_H4_COMMON_CACHE`を指定し、`PYTHONPATH=src`、BLAS/OMP/MKL各1 threadで、初期`python scripts/run_track_c_mixed_formula.py`、後続`python -m trottertracks.track_c_mixed_rounds`、`...track_c_mixed_price`、`...track_c_mixed_account`、`...track_c_mixed_semantics`を順に実行した。旧出力の再開・上書きを行わず、再実行には別の新出力先と適切な実行許可を用いる。

## 次の研究判断とSTOP

今回の明確な成果は、ground-state mixed commutatorの相殺関係、既知5段族のground leading biasの下限、D内部誤差の大きさ、有限RTEを接続すると作用素改善が資源改善へ残らない固定H4の負結果である。

次に有望なのは、ユーザーが研究方向を判断した後に、(i) 外側係数より支配的なD内部PFを、ground位相を目的として改善する構造、(ii) 残る交換子補正をRTEの追加samplingなしで実装できる構造、(iii) 負時間によるnormalization増を避ける高次構成を検討すること。ただし実装価格やHermitian補正の生成費まで含める必要があり、今回その新仮説の計算は開始していない。

担当：ユーザーが次の主要仮説を選択し、必要な固定条件と新規検証範囲を承認する。Codexはその承認後に限定機構検証を実施する。今回の結果を改善成功として採択せず、詳細GPT科学レビュー・別研究計算へ自動進行せずmandatory STOP。
