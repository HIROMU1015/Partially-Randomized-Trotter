# N5-B0：m17の誤差項と不足証拠

2026-11-08 JST。[REPORT](REPORT.md)。保存fieldの正本は[M17_INPUTS](../../../artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json)。新しい誤差和・RPE評価は行っていない。

## 1. 帰属表

| 項 | 既存SOURCE／証拠 | 今回の取扱い |
|---|---|---|
| Binary64 target生成・SOURCE→native | Compiler A、旧B′ ERROR_RPE／Reference STAGE3_ALL_SUPPORT、CT-0 | 各roundの旧raw disk。Actual literalが合成targetで、理想角へsnapしない |
| Dyadic量子化 | N3-B0 budget、provider.round_index／provide | θとn2^-36の差。新しいbや量子化上界を計算しない |
| Basic word近似 | 保存43 guard。Exact_word_matrixはW込みのcyclotomic matrix | Logicalなstrict Frobenius誤差。新guard／再合成なし |
| C0 word近似 | 保存299 strict guard、up_to_phase=False | C0では量子化でなく保存logical guardが直接対応 |
| Wの具象化 | N3-B1aの8状態補題、N3-B1b-0 runtime条件 | Exact Wとfloat Pのscalar差。既存emitter項の内部であり、別に再加算しない |
| Local inverse／compose／正規化 | 旧lower_native、固定inplace経路 | C0の64u、辞書の保存emitter項の内部。Wだけを精密にしてもjoinを省略しない |
| Outer shell | N4-B consumer、N4-C／N5-AのS<=1構造 | 初期phase設定＋segment→outer join。RZ内部emitterとは別のwhole項 |
| その他のSOURCE／finite-RTE／finite-bit proposal／vacuum算術 | 旧raw disk | 同じraw単位で一度だけ合算。Common-joint-scalar議論でSOURCE vacuum角の誤差を消さない |

旧guardは**数学的logical wordとactual binary64 RZ target**の比較で、float具象化前のWを含む。Guard内のWと、具象化時のW誤差は比較対象が違う。両方を保持することは二重計上ではないが、emitterのW subtermを保存64uへもう一度足すのは二重計上になる。

## 2. Local・joint・rawを混同しない

辞書missでは保存規約が

`e_RZ(theta) < q(theta) + sum_(selected basic) g_j_saved + e_emit`

C0 hitでは `e_RZ(theta) < g_C0_saved + e_emit_C0` である。この1-qubit Frobenius上界をspectralへ移し、物理位置のtelescopingと別のshell項でjointを囲う。

`epsilon_joint <= sum_i e_RZ,i + epsilon_shell`

`d_syn <= 2sqrt(2) epsilon_pair`, `d_total = d_old + d_syn`

複素X/Yには両軸を覆うepsilon_pairを用いる。N4-Cのmax-axis規約と旧disk一度だけ加算を維持する。Frame価格取消し、signed価格補正、期待RZ数を誤差の省略へ使わない。

**Shellはwhole当たりの項で、gate capと直接比較してFAILにしない。** N4-Cの保存shell値5/562949953421312を、m17のper-gate capへ入れなければならないという規約ではない。逆に、その値が小さいだけで全RZ和が予算内とも言えない。今回新しい和は作っていない。

### ShellのS<=1を移送できる条件

[N4-CのSOURCE証明](../pr_reference_n4c_accounting_20261108/REPORT.md)は、Referenceの先頭reset群（B′にはなし）、途中にreset／measure／barrierのないunitary body、最後のreadoutという構造を使う。Q／rはbody内のunitary反復数を変えるが、このfence配置を変えない。Compiler条件Cも新たなfenceを作らず、consumerは同じinplace経路で各nonempty segmentを一度だけouterへjoinする。この条件をm17で保つ限り、segment数S<=1、outer初期phase設定1回という既存の記号的証明を移送できる。

新しいbody fence、別routing／compiler、QPY／to_instruction／copy経路を導入するなら、この移送は自動ではない。保存8 wholeの観測だけを全supportの根拠にせず、今回m17回路をbuildしてSを観測したとも主張しない。上記SOURCE／compiler条件付きの構造対応であり、新しいshell上界の数値評価ではない。

## 3. 保存m17の条件

| 保存field | B′ | Reference |
|---|---:|---:|
| Shots/axis | 626 | 630 |
| Worst-case RZ X | 13770556061 | 13714327029 |
| Worst-case RZ Y | 13770556062 | 13714327030 |
| Gate十分条件 X | 6.902168834414281e-16 | 2.5648979455928107e-15 |
| Gate十分条件 Y | 6.902168833913054e-16 | 2.5648979454057876e-15 |
| d_old | 3.6648120444602677e-6 | 0.000457439007713711 |
| Nominal raw radius | 0.5726909945383861 | 0.5726909945383861 |
| 旧raw radius下端 | 0.5726873297263416 | 0.5722335555306725 |
| 旧statistical phase下端 | 0.38598699711850315 | 0.38519464269240195 |
| 追加raw disk保持余地の保存下端 | 2.6883267253440152e-5 | 9.949232607511949e-5 |
| Joint synthesis保持余地の保存値 | 9.50467028767889e-6 | 3.5175849221870066e-5 |

Q131072／r32／88 bits、PF_phase_upper=0.01400660359797486、B_lower=0.38599339640202507も保存値である。これらはCT誤差を新しく加えたradius／phase／Nの結果ではない。Gate十分条件は、旧worst-case一様誤差会計で保存Nを保つための安全な割当で、真に必要な最適precisionではない。

## 4. 各証明の制約

### C0

保存certificateは `g_saved + 64u`、u=2^-53。64uは保存値1/140737488355328に対応し、m17両armのgate capより大きい。従って**現certificateをそのまま当該一様gate条件へ入れる資格はない**。

保存logical guardだけを比較すると、C0 library299のうち11 wordが両armY cap内、guard＋保存emitterでは両armとも0 wordとなる。これはmetadataの大小照合で、実誤差の再評価ではない。11はlibrary件数であり、m17の必要位置や物理多重度ではない。Emitting誤差が本質的下限だという結論にもならない。

### 固定辞書

N3-B0の保存量子化上界は約5.1448789686e-12。Basic割当は各10^-13、既存の実guard和・allocated uniform bound・emitter項は[M17_INPUTS](../../../artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json)へそのまま保存した。ここで再加算していない。

Actual guardだけを比較した43 basicの件数は、B′ Y cap内0、Reference Y cap内1。後者はj=-28の保存recordである。したがってラベル10^-13だけで全wordを一律に評価せず、実guardを参照する必要がある。ただしsingle basicがcap内でも、選択chunk和、量子化、emitter、域を含む最終資格にはならない。

固定b36のgridと固定word選択を保つ限り、basic guardだけを改善しても量子化cell幅は変わらない。同じcellの非C0 targetへ同じwordを返す構造にはuniform分解能の制約がある。一方、実m17が必要とする有限target集合がcell内のどこにあるかは未確認なので、これをm17の不可能性証明へ移さない。新b／指数範囲／精度は選んでいない。

### Emitter／shell

辞書の保存emitter項45/140737488355328は、高々43 carrier等の経路を覆う保守的十分上界。単一のcomplete word経路では未使用の余裕があり得るが、その余裕を新しいgate budgetへ数値配分し直していない。

64u・上記辞書項・shellは実装の観測誤差でも数学的下限でもない。同じ実wordで評価を改善する可能性は[EMITTER_BOUND_OPTIONS](EMITTER_BOUND_OPTIONS.md)へ分離する。改善後のm17成功や新規合成の必要性は、今回確定していない。

## 5. T-countへ残る条件

M17の保存Nは626／630で異なる。1-shotでframeの価格が共通でも、shot重み付き差には絶対価格項が残り得る。m17の共通policy／域／precisionと保存Nの十分性を資格し、そのpolicyのD／CSWAP／scalar／中央／frame／vacuumの価格を同じlawへ対応させる必要がある。

M7のK=160222等やB_P=6072をm17へ代入しない。新しいT価格和、期待値、改善率、総Tを計算していない。**m17 CT RPEとT優越は未資格、mandatory STOP。**
