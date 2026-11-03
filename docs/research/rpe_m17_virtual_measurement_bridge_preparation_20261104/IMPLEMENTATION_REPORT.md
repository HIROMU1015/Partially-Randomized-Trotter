# m17仮想測定bridge adapterの技術準備

2026-11-04 JST。**adapterと人工検査を実装した。実H4を評価できる科学実行SOURCEは、BLASの丸め上界が未確立のため未確定。**
状態は`TECHNICAL_ADAPTER_PREPARED_KERNEL_UNRESOLVED`、`scientific_execution_ready=false`。
今回の「実行してよい」「続きを行って」は直前の提案である技術準備へ適用し、固定SOURCE後の新科学実行とは区別した。
実H4の新しい作用・平均・norm・CDF・sampling・包含・予算評価は0。

前段：[bridge調査と選定](../rpe_m17_virtual_measurement_bridge_preflight_20261104/NEXT_STAGE_DECISION.md)、
[固定入力・最小計画](../rpe_m17_virtual_measurement_bridge_preflight_20261104/MINIMAL_VALIDATION_PLAN.md)。
前段の「kernelの根拠を確定できなければ実H4数値評価の前に不足を報告する」を維持した。

## 実装した範囲

| ファイル / 処理 | 再利用と差分 |
|---|---|
| [virtual_bridge_adapter.py](virtual_bridge_adapter.py)：numeric_basis / numeric_components | latest exportの保存local binary64 matrixをQiskit Operatorで数値compose。旧Aのrestricted_basisと同じfull空間→指定sector順へのslice。新しいGaussian gateやcontrolled回路を生成しない。成分のsign/supportを含むP_i64を構成 |
| 同：matrix_residual / component_bounds / decode_matrix_ball | 既存ball_adapterのexact dyadic/Acb、global radiusを再利用。ideal basisはmean_adapterの解析式。保存centerを低精度で読む際の丸め半径も保持。11 basis相当を共有する設計 |
| 同：rotation_growth / action_growth | ideal unitaryとの誤差、math.cos/sinの保存出力との差、matvec/scale/subtractの丸めを結合する。係数は導出済みのkernel boundが入力される場合だけ使用し、実環境の値を捏造しない |
| 同：Growth / trajectory_growth / signal_error | 平均と全supportのaffine majorantを分ける。加法誤差を保持してunderflow項を無言で0にしない。raw state normと最後のvdotを一回ずつ計上 |
| 同：guard_completion / classify_axes | floatで作られる実際の範囲閾値、complex abs評価の誤差、有限性を別条件として扱う。完了または算術根拠が未確立ならModel II上界はnull。strict比較でも固定PCG64のU_execはnull |
| [run_virtual_bridge.py](run_virtual_bridge.py) | default/--describeは6原本と4 helperのbyte照合のみ。科学ロードは含めない。--runもkernel未確立を返してexit2、H4処理を開始しない。これは完成した科学runnerではない |
| [test_virtual_bridge.py](test_virtual_bridge.py) | 人工2×2行列・rational lawで検証。旧Aの実関数を参照し、保存local matrixからの数値basis/sector順を比較。実H4入力は数値評価しない |

旧A runnerはm0/q1・m3/q8/r1用なので、そのexecute/prepare_contextをm17として再利用しない。
既存Acb meanのq乗はfloat trajectoryの平均を表さないため、majorantだけを伝播する小さいadapterを追加した。
元module・実行契約・結果は変更していない。承認証明・grant・管理基盤は追加していない。

## 数学的な伝播条件

各primitiveのideal対象はunitaryで、実際の浮動小数点処理fについて

    ||f(v)−Uv|| ≤ (g−1)||v|| + a,  g≥1, a≥0

が全到達状態で成り立つと仮定する。ノルムmajorantはH(x)=gx+aで、
合成はH_after∘H_before、binary powerでq/rの長列を避ける。
誤差はH_chain(||ψ||)−||ψ||以下。これはideal側のunitarityに依存し、任意の非unitary参照へは使えない。
IID orderの平均はΣp_n H_n、uniform上界は正のmassを持つorderの係数ごとの最大値を使う。
これは非線形なFP programを平均してから実行する処理ではない。

KernelBoundsのmatvec項はκ_mv ||P64||_F ||v||+a_mv、scale/subtract/dotも明示的なノルム式。
rotationのboundは2経路 `c64*v` と `(-i*s64)*(P64@v)` の丸めを分離してからsubtractへ渡す。
有限値に対するevent.phase=±1および−iの作用はsign/swapとして扱う。
overflowがないこととabs評価の根拠は、全supportの完了判定に別途必要。

最後のsignal boundはraw ψのn²区間から、|n²−1|、state-action誤差、vdot誤差を足す。
clippingはprojectionの非拡大性を使い、action誤差へ二重加算しない。
受理後の確率/Bernoulliの解析3·2^-53は最後に一回だけ加える。
logical intervalを共通参照とするため、既に保存されたbuilder π64 transferを往復加算しない。
U_catからphase上端を引いてU_logicalを作る処理もない。

## 確認できたbackendと未確立部分

既存の隔離環境をread-onlyで確認：Python3.11.0rc1、NumPy1.26.4、
OpenBLAS0.3.23.dev / USE64BITINT / DYNAMIC_ARCH / NO_AFFINITY。
実際の`openblas_get_corename64_()`は**Prescott**を返した。
build configurationにあるHASWELLの文字列を実行kernelと同一視しない。
ローカルbinaryにはzgemv_t_PRESCOTT、zgemv_n_PRESCOTT、zdotc_k_PRESCOTTのsymbolがあり、
先頭disassemblyの確認まで行った。kernel全経路・thread reductionの丸め深さは確定していない。

NumPyのSOURCEではcontiguous complex行列×vectorがCBLAS GEMVを使い、
contiguous complex vdotもCBLAS DOTへ渡す経路がある。
従ってNumPy側の単純なfallback loopだけを読んで既存実行全体へγ_dを当てることはできない。
[NumPy1.26.4 matmul SOURCE](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/umath/matmul.c.src)、
[vdot SOURCE](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/multiarray/vdot.c)。

OpenBLAS v0.3.23の[KERNEL.PRESCOTT](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.23/kernel/x86_64/KERNEL.PRESCOTT)では
ZGEMVNKERNEL=zgemv_n_dup.S、ZGEMVTKERNEL=zgemv_t.Sと指定されることも確認した。
このrelease版の指定だけでは、現在のdev buildのbinaryとの完全な対応は確定しなかった。
参照を試みた一部のOpenBLAS SOURCE URLは取得不能で、内容を推測で補っていない。
残る事項はGEMV/DOTの加算深さ・FMA/分岐・thread合算・subnormalの扱い、
NumPyのscalar/vector操作とcomplex absに対する上界、およびoverflowしない範囲の確認。
version・symbolの存在・人工テストPASSからこれらを保証したとは扱わない。

`installed_kernel_contract()`はこの不足をUNRESOLVEDで返す。
外部ファイルやCLIから任意のγ係数を入れて実環境を認定する経路は作っていない。
KernelBoundsは人工数学テスト用の入力interfaceでもあり、承認やbackend保証の証明書ではない。

## 検証と到達点

新人工16件、以前のbridge人工15件、再利用する包含helperの限定5件、計36件PASS。
実行記録・最終件数は[VALIDATION.txt](VALIDATION.txt)を参照。
default CLIでは最新包含4原本、B raw state、元m17 operatorの6件をbyte/hash照合し、
`new_real_numerical_calls=0`、`U_exec_upper=null`を確認した。
これはLOCAL技術検証であり、独立科学レビュー・CI・実H4上界評価ではない。

対象条件はlinear H4/1.00 Å/STO-3G/保存legacy DF12/B state/4電子sector70、
L_D3/単一δ=.02/m17 q131072/r32 K2、extract_identity_phase、cutoff1e-12。
保存N411/axis、α=.016957089147912167、ε=.15244325851620388とPF/RTE/stat予算・βを保全する。
最新ideal categorical局所十分条件、全体UNDETERMINED/false、旧INCONSISTENT/u_num=nullを変更しない。
原本・SOURCE・grant/ledger/marker・STOP、dirty main・他worktree・SERVER2 H8・共有環境を保全した。

## 次の担当と範囲

**Codexの次段案**は、既存GEMV/DOTの対応SOURCE/算術条件を絞って確認し、
係数とguard前提を導出できれば、このadapterへ接続して実行SOURCEを完成すること。
現時点で「科学実行準備完了」や「実行すれば必ず数値上界を得られる」とは報告しない。

既存kernelの保証を置けない場合、**ユーザーの研究判断**として、明示した算術仮定の下の条件付き結果を目標にするか、
演算順を固定した別kernelを使う新Model IIを定義するかを選ぶ必要がある。
後者は既存NumPy仮想測定との同一性を失うため、今回無断で採用していない。
新科学実行は、kernel前提を含む具体的SOURCEと[実行条件案](SCIENTIFIC_EXECUTION_PLAN.md)を固定した後に別途許可する。
既存のユーザー許可そのものを再証明する仕組みは不要。
今回の公開・remote照合後にmandatory STOPする。
