# H4 B0定数項保持対照：保存値診断と科学判断待ちSTOP（2026-10-13 JST）

今回の追加解析指示の停止条件「B0+Iの結果によって主要な科学的結論が変わる可能性がある」に該当する参考感度が得られたため、最優先Bの証拠整理で停止する。原稿の主要結論や比較classを自動変更しない。

6距離×302精度点のうち397点で、B0+Iの追加1-shot compiled RZ費用をゼロと置いた参考値が登録B2の点最小を下回った。距離別は0.70 Å 178点、0.80 Å 103点、0.90 Å 29点、1.10 Å 0点、1.40 Å 87点、1.60 Å 0点。これはB0+Iの実測優位ではなく、未測定の比較対照が主要な解釈を変え得ることを示す条件付き感度である。

元の登録B0/B1/B2/B3集合でB2が全1,812点の点最小である結果は変更しない。科学SOURCE、raw、ledger/checkpoint、旧grant、消費/予約は不変。H4解析専用branch `track-a-h4-geometry-analysis-20261013` は公開結果commit `7c4483378047b57f514c47dad98905e3f2a9a756` から独立している。

実験条件はH4 linear/STO-3G、固定DF12、8 system qubits+ancilla1、二次DF-prefix PF/canonical finite RTE、T=0.8、L_D=0/3/4/5/6/9/12、q=1/2/4/8、delta=T/q。指標はcomplex-signal ε、α_axis=0.025、状態準備P=0の解析shot×full-wrapper compiled RZ。H6/H8・S4・energy/RPE全体に一般化しない。

元source `_prepare` ではB0のconstantはnuclearのまま、保持prefixのone/DF blockは各I/Z/ZZ係数を含めて作用する。B2/B3は捨てるtail blockのI係数をconstantへ加え、非identity項だけをRTEへ渡す。したがってB0+Iは保持済みblockの定数を二重加算する操作ではなく、tailだけのIを追加する。B0自体は登録したdiscard仕様を実装しており、今回これをbugと認定しない。

`diagonal_coefficients` の各DFブロックのI係数は λ/2 Ση_i² + λ/2 Σ_{i<j}η_iη_j = λ/4[(Ση_i)²+Ση_i²]。保存Gのlower triangleからsourceの `numpy.linalg.eigh(..., UPLO=L)` が用いる対称行列G_Lを定義すると、λ/4[(Tr G_L)²+Tr(G_L²)]で取得できる。保存係数768 float/geometryと12 λ、energy scalarだけを選択decodeし、新しいeigh・DF分解・Hamiltonian構築・回路作用は行わない。全12片を保持し、near-null片も閾値pruneしない。

最初の限定診断では元sourceにない全G symmetry条件を追加して停止した。保存near-null片にはupper/lower差があるが、元sourceはlower triangleだけを使う。調査したλ重み付き非対称差は最大約4.74e-16。余分な仮定を除き、sourceと同じlower規約を用いて再集計した。科学入力やsourceは変更していない。実数算術上のtrace identityであり、元eigensolverの丸めのbitwise再現・厳密数値certificateとはしない。

回路sourceはancillaにp(-delta*constant)を各q stepで入れ、signal sourceはexp(-i delta*constant)を乗じる。delta=T/qなのでtail位相の合計は -T c_tail。wrapperはsine軸にS†を入れ、保存値の虚部と対応する。targetは凍結energy Eのexp(-iET)。元B0保存complex signalと保存biasを照合した上で、z_I=exp(-iT c_tail) z_B0のみを算術的に適用した。modulusとB=1は保持し、cosine/sineの各bias・headroom・整数shotを別表で再会計する。

ε=0.05の代表結果。G単位は百万RZ×shot。Δは各軸に同じ追加RZ/shotを仮定した場合の交点であり、実際の追加compile費用は未測定である。

| 距離 Å | B0+I参考点最小prefix/q | B2 G | B0+I G（Δ=0参考） | 共通追加RZ/shotの交点Δ* |
|---:|---|---:|---:|---:|
| 0.70 | p5/q1 | 206.510 | 152.874 | 2965.95 |
| 0.80 | p5/q1 | 171.336 | 152.913 | 1018.96 |
| 0.90 | p5/q1 | 134.063 | 148.133 | -780.72 |
| 1.10 | p5/q1 | 112.907 | 144.709 | -1823.28 |
| 1.40 | p3/q1 | 102.934 | 98.908 | 212.41 |
| 1.60 | p4/q1 | 97.277 | 118.877 | -1244.03 |

0.70/0.80/1.40 ÅではΔ*=2965.95/1018.96/212.41 RZ/shotが正であり、共通追加費用0≤Δ<Δ*という参考モデルではB0+Iが低い。0.90/1.10/1.60 ÅはΔ=0でも今回のB2点最小より高い。負のΔ*は「非負の共通追加費用では逆転しない」という意味であり、実compiler差が必ず非負になる証明ではない。

一般の軸別差では G_I=N_I,c(C_old,c+Δ_c)+N_I,s(C_old,s+Δ_s)。低費用となる条件はN_I,c Δ_c+N_I,s Δ_s < G_B2-G_I(0)。実際に必要なcompile後のC_I,c/C_I,sは今回取得していない。既存B0に追加scalarの角度変更を吸収できる可能性があっても、compilerのgate削除・融合結果やRZ数を同一と認定しない。

算術上、位相保持はaxis biasとshot余裕を変える。0.80 Åの参考B0+Iは18,080 shots、B2は28,102 shotsで、既存B0より高い1-shot費用にもかかわらずreference Gが低くなる。1.40 ÅではB0+I参考点最小がp3/q1へ変わり、既存B0の点最小p5/q1とは比較構成も異なる。scalar保持が全候補・全精度で必ずbiasやshotsを減らすとは主張しない。分子のcorrelation/DF spectrum/軌道回転の因果説明は調べていない。

基準データは固定Git結果と元raw。GPTレビューの18行のG/SE/対照G/B/36軸shotは元summaryと一致した。レビューZIP SHAを受領時に確認し、外側review MarkdownとZIP内の同文書のbytesが一致した。GPTレビューは全395,016点・全raw再実行の証拠ではなく、代表18点の独立スカラー検算と科学的判断として区別する。

全6凍結NPZのstreaming byte SHAをplan期待値と照合し、G.npy/lambdas.npy/energy.npyの小memberだけをdecodeした。H.npy、blocks.npy、state、MOs、integrals等は科学arrayとして読み込んでいない。全120 B0/PM1-B0 signalのbytes/SHAを公開projectionと照合し、元target/bias、identity、B=1とmodulusを確認した。13人工スカラー検査PASS。新SCF/DF/state/signal simulation/trajectory/eigh/build/compile/transpile/GPUは0。解析は1 process/内部thread1、AS256MiB、CPU45s/wall60s、I/O128MiB/実行・output8MiB制限。Linux getrusageのru_maxrssはpre-execのpeakを含み得るため、収録値を今回解析のnative VmHWMとは扱わない。

B0+Iの未知costをB2の32 paired trajectoryのSEに吸収しない。397点は同じ保存値の感度gridであり独立科学実験ではない。B2点最小は同じ32本で選抜したposthoc minimumで、±2SE・母平均winner・希少event未観測を厳密保証に格上げしない。

停止後は機構分析Aの完全な切替地図・Taylor event観測C・H4/H6/H8条件対応表と原稿本体統合Dを実行していない。H4のevent観測を今回未取得のまま保持し、H6 pilotの0.00734%補正を流用しない。H8のworktree/source/runner/plan/grant、他job・CPU配置・共有venv/global Git設定は変更0。

GPTへ戻す論点は、既知scalar保持対照を比較classへ正式に追加するか、登録集合内という従来の限定claimを維持して感度を原稿へ記すかである。対照強化を選ぶ場合の最小確認候補はε=0.05の0.70/p5/q1、0.80/p5/q1、1.40/p3/q1の各cosine/sine、計6 full-wrapper compile calls。これは費用取得の候補案のみで、新source/plan/grant・fresh resource確認と明示実行認可が別に必要。旧grantを再利用しない。新科学signalや一律sampling増量は今回開始しない。

[結果bundle・source/入力provenance・全1,812点感度CSV・公開値照合script](../../artifacts/resource_applicability/track_a_h4_geometry_scalar_diagnostic/2026-10-13/README.md)。元[登録H4 geometry結果](track_a_h4_geometry_results_20261013.md)はそのまま保持する。
