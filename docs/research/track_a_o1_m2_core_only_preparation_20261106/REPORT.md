# O1 M2：core-only省略誤差・model normの限定評価準備

2026-11-06、SERVER 2（GPUA100-gleap02）。起点 `f891f6e3265e4f0124c598fddfb7775bc1eddd1b`。[静的survey][S]を受けた**最小実装・人工テストまでの準備**。実H4の入力decode、core/operator再計算、保存値再解析は未実行。科学実行許可はまだなく、公開確認後mandatory STOPする。

## 再利用と最小差分

追加は[係数上界module](../../../src/trottertracks/o1_m2_core_only_bound.py)、[明示path runner](../../../scripts/run_o1_m2_core_only_bound.py)、[人工test](../../../tests/management/test_o1_m2_core_only_bound.py)、本報告の4ファイルだけ。既存B adapter／保存update005 loader／`SpatialIntegrals`検証を再利用し、旧SOURCEを変更していない。CAR・微小係数保持mapperは既存fixtureとstrict mapperを人工確認に再利用した。旧[B operator PASS][B]を作り直すための全H4行列再構成は行わず、今回のnorm上界はPauli mapperや大きい全H toleranceに依存しない。

runnerは入力load前に再利用B／adapterのbytesを起点`f891f6e3265e4f0124c598fddfb7775bc1eddd1b`と照合する。結果には実行SOURCE commit・関係4 SOURCEのhash、Python／host／NumPy・SciPy・OpenFermion・Qiskit版を記録する。これは科学SOURCEとruntimeの来歴であり、ユーザー承認の認証や証明ではない。

既存`packet_scale`の根拠を、そのまま正確な係数和へ限定拡張した。空間積分packetのoperatorに対して

`||O(c,h,g)||₂ <= |c| + 2 Σ|h_pq| + 2 Σ|g_pqrs|`

を用いる。hにはspin個別のCAR monomialが2個、gには4個とprefactor1/2があり、各個別monomialのnormが1以下であることによる。spin-summed E自体をnorm1と仮定していない。核scalar、一体項、gの符号・微小項を残したまま、入力floatを`Fraction.from_float`で正確な有理数へ写して係数差・絶対和を計算する。出力は正確な有理数と、それを下回らない上向き丸めfloatの両方。精密なspectral norm値や固有値を計算する方法ではなく、保守的なCAR係数上界である。

## 承認後に計算する対象を先に区別する

| 対象／出力 | 意味と固定された範囲 |
|---|---|
| 数学的B参照 | raw H0/C0の保存binary64 bit値をexact real係数と解釈し、Fractionで`H_H^ref=P(H0_bits)`、`C_H^ref=P(C0_bits)`、`R_H^ref=H_H^ref−C_H^ref`を扱う。ここでは恒等式を正確に保つ |
| 既存adapter形成packet | 既存のbinary64 Pと差形成によるH_H/C_H/R_Hをそのまま保持する。形成R、形成H−C、形成H−C−R、それぞれと数学的B参照との差を別の係数上界として返す。小残差の形成損失を零化して隠さない |
| B-relative core-only energy | `|E0_S(H_H^ref)−E0_S(C_H^ref)| <= bound(R_H^ref)`。形成済みH_H/C_H packetについても別の上界を返す。両者を同じ結果として混ぜない |
| full／sector | 係数上界はfull Fock全体で通用し、compression normにも通用する。固定空間spin liftはNα/Nβを代数的に保存するため、同じ保存sectorでHermitian operatorの最低固有値差へ接続できる。予定sectorはNα=Nβ=2。native回路の保存則・相対位相資格を認定するものではない |
| raw H0→P(H0) norm | raw非Hermitian operatorから数学的P参照への変更上界だけを返す。Hermitian元物理Hのenergy誤差ではない。raw H0の固有値を物理基底energyと読み替えない |
| 条件付き省略予算 | `--hh-omission-budget`は明示指定のみ。未指定なら`DEFERRED`、上界が指定予算内ならB参照内だけの`WITHIN_DECLARED_B_REFERENCE_BUDGET`。上界が大きい場合は`BOUND_INCONCLUSIVE`であり、真のnorm超過を証明したFAILではない |

数学的B参照もadapter形成packetも、**既にbinary64で形成されたraw core packet**を基点とする。J/U縮約そのものを高精度で再実行した結果や、SCF／積分生成のenclosureではない。従って元H_phys・正確なJ/U core・native coreとの対応は別の問題として保留する。係数上界を物理accuracy certificateへ拡張しない。

## 引用／計算／保留の境界

- **引用のみ**：既存[B]の限定B operator PASS、旧FAILの保存、scalar／電子identity・一体補正・全8 frame／sector対応。研究Cの既存NO-GOは[S]の引用のまま保持する。旧λ・価格・shotや既存C0の0.1 Haを新M2予算へ移さない。
- **別途承認後に計算可能**：固定inputの再load→既存raw core形成→既存B adapter→上記のexact係数上界。全H4行列・eigensolve・state・PF/RTE・回路build/compile・sampling・価格・shotはこのrunnerにない。今回それらの科学callは0。
- **DEFERRED／UNASSESSED**：元Hermitian H_physの定義・保存値との対応、参照／生成精度、物理model変更の許容上限、M2の目標energy予算、元H stateのbind、全RPEのerror／failure／branch条件、exact-J/U／native対応。H0やH_Hを元H_physへ正式再定義しない。`physical_energy_bound=None`を保持する。

M2 core-onlyの妥当性を元Hに対して判断するには、ユーザーがこの不足条件を決める必要がある。一方、H_H内の限定省略上界を算出するだけの承認範囲は明確化でき、元H_physが未定義でも数値を物理PASSへ昇格しない条件で実行できる。単位半径やH_H基底stateは仮定せず、元H厳密基底state／prep費用除外の研究条件も変更しない。

## 人工テスト・確認結果

**49 passed in 2.63 s**（新13件＋既存B／微小係数／残差adapter36件）。人工2 spatial orbitals・2 signed framesだけを用い、既存独立CARのoperator2 normと最低固有値差が係数上界内にあること、`1e-10/1e-22`の保持、巨大な共通scalarの正確な相殺、上向きfloat化、数学的Pと形成済みBの桁落ち差、raw anti-Hermitian差の非物理energy扱い、未指定／不正budgetを確認した。SOURCE差を入力decode前に拒否することと、人工runner出力のruntime記録／budget None維持も確認した。新testでは`np.load`と回路生成を禁止した。新runnerのhelpも入力decodeなしで確認した。

初回は人工anti-Hermitian fixtureのH/core核scalarが不一致だった1件のみ失敗し、共有scalarにそろえて再確認した。科学条件・判定式・旧許容値は変更していない。旧H4結果の再解析や人工PASSの科学性能への転用はない。

runtimeは既存専用Python 3.12／NumPy 2.5.4／SciPy 1.18.1／OpenFermion 1.8.1／Qiskit 2.6.2をread-onlyで再利用した。cacheは専用`/tmp/o1_m2_core_only_prep_cache_20261106`、bytecode／pytest cacheは無効。実行したのは`tests/management/test_o1_m2_core_only_bound.py`と既存`test_o1_m2_hermitian_b.py`、`test_o1_m2_small_coefficients.py`、`test_o1_m2_residual_adapter.py`の4ファイルだけ。

## 固定入力・実行承認範囲・STOP

今後の対象は同じ保存update005全8 frameと元h/g/scalar。input commit `aced1647b81e8d28091058cdc7b6e3e1ee54896a`、core path `artifacts/research_c_pr_integrated/2026-11-02/run01/update005_candidate.npz`（SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`）、元H path `artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz`（SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`）。生成SOURCE `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4`と今回の準備SOURCEを区別する。今回NPZは取得・decodeしていない。

次は**ユーザー**が「この準備SOURCE・固定2入力でB-relative core-only norm／energy上界だけを計算し、物理判定／未指定budgetは保留する」範囲を明示承認する。**Codex**は承認後に準備commitから別science worktree／branchを作り、fresh結果へ限定計算しmandatory STOPする。物理Hへの判断は不足条件の採択後のみ。新fit・rank探索・別Hamiltonian、PF/RTE、状態solve、回路、shot／総費用はその限定承認にも含めない。

今回は新科学の実行許可がないため**科学STOP**。準備4ファイルを専用branchへcommit/API公開し、remote SHA・本文とSOURCEの全文一致・local cleanを確認後**mandatory STOP**する。旧SOURCE・input・raw・FAIL・許容値・grant・marker・STOPは変更しない。承認証明・新grant・管理frameworkは追加しない。

[S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f891f6e3265e4f0124c598fddfb7775bc1eddd1b/docs/research/track_a_o1_m2_improvement_survey_20261106/REPORT.md
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68ef4feee3835ed7135d254ac879e8d88c69c895/docs/research/track_a_o1_m2_hermitian_b_science_20261106/REPORT.md
