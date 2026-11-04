# RPE m17仮想測定bridge：固定科学実行計画

2026-11-04 JST。**実装・人工検証まで完了した新SOURCE。実H4の科学実行は未実施。**
このfolderを含む公開commitをSOURCEとして、ユーザーが確認後に別途明示許可した場合だけ--runを実行する。
SOURCE commitは最終チャットとGit履歴で固定する。承認証明書・grant・承認原文の保存は不要。

## 固定対象と入力

| 条件 | 固定値 |
|---|---|
| 系・Hamiltonian | linear H4、隣接1.00 Å、STO-3G、保存legacy DF12 |
| state / sector | Bの保存raw physical state、8 qubit・4電子sector70 |
| PF/RTE | Partial L_D3、単一δ=.02、m17/q131072/r32/K2 |
| 位相・cutoff | extract_identity_phase、coefficient_atol=1e-12、既存signed event/F→events→R/Re/+Im |
| 保存統計入力 | N411/axis、α=.016957089147912167、ε=.15244325851620388 |
| β | PF=.014050012886294481、RTE=9.899584668737594e-8、stat=.38594988811785885 |
| 確率モデル | 数学的IID uniform53 categoricalと独立uniform53 BernoulliのModel II。固定PCG64のIID保証は対象外 |

μ0は保存binary64、εは保存decimal文字列のexact rational、sは既存の包含区間を再利用する。
N/α/β/εの再配分、旧ideal categorical mean/CDF/sの再計算は行わない。

固定入力は旧[run_virtual_bridge.INPUTS](../rpe_m17_virtual_measurement_bridge_preparation_20261104/run_virtual_bridge.py)の6件。
各path・SHA256・元commitをそのまま使用し、workerが数値loadの直前にbyte/hash照合する。

| 入力群 | 固定commit / 用途 |
|---|---|
| latest RESULT、source_replay、categorical_laws、enclosure_operators.zip | `34840ce9bf81d985b32764d9bc9d77645ed52e59`。保存logical interval/norm²/s、local matrix、law、F/R包含 |
| B physical_state.npz | `e22d7718c47358ff650dbc04e8e72f6ab1c8bf41`。raw ψ、exportとsigned-zeroを含めbyte対応 |
| 元m17 operators.npz | `cd749066904d3f29b28b03c19ab69370372b157f`。保存numeric F/R64 |

科学helperは728c90a2d…のvirtual_bridge_adapter、18fffce48…のball/mean adapter、
e2fd35f1d…のkernel_boundsをbyte固定して再利用する。旧SOURCEには手を加えない。
過去run02の未保存payloadではなく、保存local matrixから一度構成する新numeric P_i64 contextを対象とする。

## Runtime条件と実行順序

1. fresh出力を作り、supervisor/workerの両方で新SOURCEがHEADと一致することを確認する。
   Python3.11.0rc1、NumPy1.26.4、Qiskit1.3.0、python-flint0.9.0/FLINT3.6.0を固定する。
2. 新規の自分のworkerをgdbで起動する。実際にmapされたNumPy/OpenBLAS/libmのhashとPrescott coreを、
   [kernel profile](../rpe_m17_openblas_kernel_bound_preflight_20261104/kernel_bounds.py)へ照合する。
3. **同じworker内**で人工70次元C行列のGEMV/DOT/complex scale/subtract/absを各1回使う。
   実CBLAS引数(alpha=1,beta=0、ColMajor/Trans、70×70)、Prescott GEMVTの全出力partition合計70、
   DOT、NumPy AVX2/FMA、_Py_c_abs/hypotの使用を確認する。
4. 全threadのMXCSRをcheckpointで読み、critical kernel/ufuncへ入る各threadのmodeも継続検査する。
   RC=nearest-even、FTZ=0、DAZ=0、例外maskが必要。新threadも次checkpoint/critical entryで検査する。
   検査不能・不一致・想定外stopではworkerを停止し、未判定を保存する。
5. 保存入力をloadし、ψ/F/Rのcomplex128、shape、C layout/stride、有限性、ψ/exportのbyte対応、
   固定tuple・inventory・保存law対応を確認する。layout変換で不一致を隠さない。
6. local matrixのfull-space numerical compositionでbasis11件を共有し、P_i64を218件構成する。
   全Pを検査し、保存θ2個に対するcos/sin64を一度取得。contextを新NPZへ保存する。
7. 96 bitのbound passを1回実行。ideal basisとnumeric P/F/Rの残差、KernelBounds、
   平均/全supportのaffine growth、raw norm・vdotを結合する。
8. 全state boundaryのuniform上端とmatrix norm/scalarのfinite envelopeを確認する。
   これは「最初のoverflowがある」と仮定した場合に、その直前までの上界から次intermediateもfiniteと示すscreen。
   平均normで代用しない。hypot range≤2^400とcomplex abs誤差も別途検査する。
9. clipping前のcoordinate/magnitude guardを全supportで示せた場合だけ、下記strict条件を判定する。
   pass前後、最終結果の前にも全threadを再検査し、監視成功前に候補PASSを公開しない。
10. 成立・未判定・中断のいずれでも結果を残してmandatory STOP。自動retry/resume・精度追加はしない。

runtime_request/runtime.jsonはFP条件のcheckpoint通信と数値provenanceであり、ユーザー許可を認証する仕組みではない。
新processのtraceにptraceが許される実行環境が必要。共有設定変更や既存jobへのattachは行わない。
検査不能ならSTOPし、無監視の計算へfallbackしない。

## 精度・科学call範囲

この新SOURCEの最初の固定範囲は**96 bit、bound pass 1回**。
以前の最大3精度の計画案は履歴として保全する。今回は保存96 bit包含へ接続する1 passを固定し、
未判定時の追加精度や別上界方式への切替は計画外とする。

最大：保存数値load1、numeric basis/P context構成1（11 basis/218 P、F/Rは保存値）、
保存θに対するcos/sin出力各2、complex bound pass1、両軸のstrict比較各1。
component残差/norm、有限性screen、U_logical/平均上界の算出はこのpass内に含む。
新SOURCE replay、SCF/DF因子化、H生成、state solve、CDF生成、ideal mean q乗、
sampling/RNG replay/仮想shot、回路build/compile、他round、成功率・費用評価は0。

dominant処理は70次元component残差。既存basis cacheとFLINT内部threadを使い、q/rはscalar affine binary power。
FLINT threadsはCPU affinityから選び、NumPy構成phaseとAcb評価phaseを順に実行する。
実際の空きCPU/RAM/GPU・共有負荷/quotaを確認し、自jobの設定だけを調整する。
人工CPU/RAM/worker/wall capは設けない。終了回収用10秒timeoutは科学runのwall capではない。

## 判定と出力

全supportの完了と算術条件が揃った場合のみ、保存logical intervalを共通参照に

    U_virtual,IID,a ≤ U_logical,a + η_action/raw,a + 3·2^-53,
    upper(U_virtual,IID,a + saved_s_a) < lower(saved_ε_a)

を両軸で判定する。π64 builder transferを往復計上せず、U_catから上端を引く処理もしない。
strict境界一致は不認定。Model IIの十分条件確認でも、U_exec（固定PCG64）=null、全体RPEは未認定。

| 局所結果 | 意味 |
|---|---|
| CONDITIONAL_VIRTUAL_IID_SUFFICIENT_CONDITION_CONFIRMED | 両軸のstrict条件成立、runtime/finite/guardを確認 |
| ARITHMETIC_BOUND_OBTAINED_BUT_COMPLETION_UNRESOLVED | finiteな作用上界は得たがguard/rangeが未確立。U_virtualはnull |
| UNDETERMINED_FINITE_ENVELOPE | no-overflow screenを示せない。候補誤差は条件付きであり保証にしない |
| NOT_CERTIFIED_BY_THIS_BOUND | 有効な保守的上界でstrict条件を示せない。RPE失敗の証明ではない |
| INTERRUPTED_OR_INVALID_MANDATORY_STOP | SOURCE/input/runtime不一致・例外・中断。partial原本を残し停止 |

出力例：`artifacts/rpe_m17_virtual_measurement_bridge/2026-11-04/run01/`（今回未作成）。
RESULT.jsonにSOURCE/入力/段階call数/区間上端/判定/部分結果、numeric_context.npzに新P/F/R/ψ、
runtime.jsonにdispatch・全thread検査とcheckpoints、observer.logに診断末尾最大64KiBを保存する。
NPZは約17MiB規模の固定inventory。ログは集約し、各critical entryのmode検査を間引かない。
同じdirectoryを上書きしない。親は自分のsession/workerだけを回収し、原研究のSTOPや他jobを変更しない。

```bash
# ここから先は固定SOURCE確認後の別の科学実行許可が必要。
PYTHONWARNINGS=ignore MPLCONFIGDIR=<owned-temp-dir> python docs/research/rpe_m17_virtual_bridge_scientific_runner_preparation_20261104/run_scientific_bridge.py --run --output-dir artifacts/rpe_m17_virtual_measurement_bridge/2026-11-04/run01
```

次の担当：ユーザーが公開されたこのSOURCEとscopeを確認し、必要なら科学実行を明示許可する。
Codexはその許可後だけ上記単一jobを実行し、結果を保存・公開してSTOPする。
