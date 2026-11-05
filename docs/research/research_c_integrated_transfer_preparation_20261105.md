# 研究C：H4・1.15 Å移送の統合準備

2026-11-05 JST。技術準備であり、新SCF・積分生成・DF fit・保存実配列の再解析・PF/RTE・shots・native build/compile・実H4統計は実行していない。以下は未承認の統合実験案の移送部分であり、研究方針の正式採択や科学的GOではない。

## 再利用する固定資料とコード

| 資産 | 固定した出所・確認内容 |
|---|---|
| 既存移送準備 | `f2a9baea08eeeeb2af7b4a87445acdc61932c911`、実装 `bd38fbbfa2c5fa79d4fb416aa4ecc7769ee8e5ee`。[旧準備報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2a9baea08eeeeb2af7b4a87445acdc61932c911/docs/research/research_c_near_objective_transfer_readiness_20261103.md)。旧科学契約は `UNAPPROVED_DESIGN_NOT_EXECUTABLE`、target/input/rank/LD/N/calls が未確定だった |
| `c_near_transfer.py` | 同SOURCEの `System`、`validate_arrays`、`objective_specs`、`recipe_arm`、`PhysicalContext`、`downstream_handoff`。新しいoptimizer・物理再構成・shots solverを実装しない |
| 元入力engine | `c1_molecule_extension.integrals` のPySCF RHF/ao2mo。調査base `546bca7ed2cece59b714b986a264d7b0b0871a01` と旧移送branchでblob `701b7065ba2924a6ba983fac2a38788bb1e6657d` が一致した |
| 中立初期化・restore | `c1_molecule_extension.factor` の既存explicit Cholesky XDF、`c1_real_orthogonal_codec.roundtrip`、`c_common_convergence.fit` と `c_pr_aware_df.fit` のγ=0。物理・codec・独立value/gradientの判定と有限fit終了状態を分ける |
| 有限PR/ENTRY | `c_near_practical`、`c_near_stable`、`c_u_step_control`。旧H4固定wrapperの入力・scale・予算組立だけを統合adapterで分離する |
| 下流 | `beb8de5cd8b71f80bb69556f92299435ff431c3e` の候補固有ground/alias、General-K/CAR、PF/RTE、RPE、P0/native、shell込みserial会計。旧候補の数値価格は新候補へ移さない |

mainの収集版には本文と一部SOURCEがそろっていても、旧移送module・finite wrapper等のroot pathがそろわない。統合SOURCEは既存科学SOURCE `beb8de5…` と旧移送adapter原本を起点に閉じる。収集commitを科学SOURCEそのものと呼び替えない。旧SOURCE manifestのhashを変更しない。

## 移送先の独立性と入力

第一対象は直鎖H4、隣接1.15 Å、STO-3G、charge0/spin0、Nα=Nβ=2、sector36、system256/controlled512、通常number CDF8、元順LD2である。

旧P-Cには同じ距離のPF記録がある（`e098c54c78f589055082f9cfc2b13de50c90ca94:artifacts/research_direction_geometry_energy_difference_pilot/2026-09-25/inputs/h4_sto3g_d115_rank12_ld3_v5.json`）。これはrank12/LD3・総電子sector70のPF結果metadataであり、研究C用の元h/g/scalar packet・中立CDF8 J/U/x0ではない。今回用packetの存在やDF適格性をこの記録から推定しない。未使用とはPR/ENTRY有限recipe比較に未使用という意味であり、全研究でblindな距離とは呼ばない。旧P-CのSTOPを再開しない。

元h/g/scalar packetをこの用途で固定できる保存資料は、今回確認した固定tree・既知資料からは見つからなかった。全ディスク・全履歴に存在しないとは主張しない。新契約では新SCF/積分生成を明示的に含め、一つの共通packetを両目的・全startで共有する。別の旧DF候補や旧PF入力を代用品にしない。

新input adapterはcanonical RHF、`conv_tol=1e-12`、chemist `(pq|rs)`、元の一電子h、核反発scalarを既存engineで生成する。距離はÅで原点からz方向に `0,1.15,2.30,3.45` の生成規則。実装は `i*distance` とし、binary64の演算規則をそのまま保存する。MO直交性allowance `1e-10` とRHF収束判定を変更しない。SCF/MO/積分のpackage版、生成SOURCE、geometry、dtype/shape、元packet bytesのidentity、MO係数のprovenanceを保存する。

`PhysicalContext`へ渡すpacketは厳密に `h/g/constant`。生成engineが返す `mo_coeff` は来歴として別保存し、物理packetへ余分なキーを混ぜない。κは元gから計算、核scalarは一回、spinはinterleaved α/β、既存OpenFermion→Qiskit bit reverse、signed J/identity位相と元frame順を維持する。

## 移すrecipeと移さない数値

developmentで固定したアルゴリズム、初期化生成規則、正則化選択規則、finite-budget、実行政策選定規則、精度・compiler・統計政策を移す。1.00 Åの最適J/U、h/g、ground、tail λ、scaleの数値、shots/W/native価格を移さない。

targetは自身の元gからrank8 explicit XDFを一回作り、修正real-Schur codecと既存tensor-only fitでC0を作る。C0から二つの事前固定seedによる同じU摂動政策でP1/P2を生成し、各一回のtensor-only restoreを行う。developmentの数値U行列をtargetへコピーしない。三startは各PR/ENTRYへ完全に同じJ/U/x0を渡す。fitに失敗・DF不適格となったstartは欠測状態として残し、代替seed/rank/LDや反対側geometryを追加しない。

新targetの共通tensor scaleはC0から一度、各目的の初期mass scaleは各paired restored startから一度決める。旧H4のf*/M*/B*を流用しない。正則化候補のrelative-strength政策・smoothing政策自体はdevelopmentと共通にし、method別strengthをdevelopment training費用の登録規則で一度選ぶ。targetの結果を見て変更しない。初期penalty寄与の正規化は、勾配・曲率・最適強度の同等性の証明ではない。

targetは三start×二armの6軌道、登録した二budgetで最大12endpointを保存する。developmentで選んだmethod別strengthのみを適用し、targetで3strengthを再探索しない。有限未収束・U一次未達の候補を消さず、J/U残差・raw optimizer status・物理DF適格性を報告する。exact tail λのみで成功判定しない。

同一rank/LDのDF予算 `CA/8=.00019875 Ha`、sector operator norm＋既存engineering guard、tensor/General-K/raw-CARの判定を使う。rank8で予算が成立するかは未検証。成立しなければ登録した結果として保存し、rank ladderへ自動拡張しない。


移送C0のrestoreに必要なscaleと、C0完成後のPR/ENTRY用scaleを区別する。C0 fit前は、既存 `c1_h6_initialization.tensor_metrics` のraw XDF8 tensor objectiveを一度取り、正ならC0 gamma0 fitのloss_scaleとして使う。これはexplicit initializer1unitに含めて記録する。C0 fit後に同じ既存helperで正のtensor lossを求め、登録したneutral_scale1unitとして外側PR/ENTRYおよびP1/P2 restoreに固定する。rawとrestoredの二scaleを保存し、未完成C0の値を初期fitへ要求する循環を作らない。zero/nonfiniteをfloorで補わない。開発P1/P2 restoreは保存neutralの固定f*を用いる。科学実行前の条件明確化であり、追加call・計算・結果変更は0。

## 下流へ渡す条件

`input identity → candidate identity → physical DF → own tail law → own ground/alias → PF/RTE → attenuation/phase → RPE shots/W → new native価格 → shell込み条件付き総資源` の依存順を守る。

既存 `downstream_handoff` はinput/candidate/rank/DF budgetを結び、旧ground・PF・shots・W・価格をnullにしたまま新ground段へ渡せる。targetでも候補固有13round、h=.1、s1/K0、登録r集合、同じβ/α/成功確率、P0 Qiskit1.3.0 opt1 seed17 rz/sx/x/cx、MPFR160 declared phase・phase fitなし、fenced serialとhead/readout shellを使う。固定r32結果と双方retuning結果は別に保存する。

targetでtrainingを行うのは、developmentで固定したr探索・candidate選択手順の適用である。正則化recipeの再開発を許さない。final native cohortはtrainingと独立、使ったデータ・seed family・候補identityを区別する。全law synthesis/implemented-law signal/actual CAの未知量はnull、ideal/exact-nativeとfinite-bit価格proxyの資格を保持する。一条件での成功・不成功を全系の一般的GO/NO-GOへ格上げしない。

保存CDF8/XDF8/同J/U強Dは入力・representation・accuracy lane・compiler・会計が一致した範囲だけ比較する。旧H4 1.00 Åのbaseline価格をtarget baselineへ流用しない。targetの同J/U強Dを候補固有に評価する。

## 最小実装差分と人工検証

1. `c1_molecule_extension.integrals_from_specification`：既存RHF/ao2mo本体を抽出。旧 `integrals(2/6)` は元 `specification(n)` を渡して既存条件を保持する。旧 `integrals(4)` の拒否も保持する。
2. `c_integrated_transfer_input.specification/generate`：固定H4・1.15 Å metadataから上記既存engineへ委譲する。科学実行CLIは追加しない。
3. `test_research_c_integrated_transfer_input.py`：完全mockのPySCF/ao2mo dispatch、元h/g/scalarのexport、passive import、旧entryとfactory委譲、無効metadataのchemistry前拒否を確認する。

```bash
/tmp/prt-representation-venv/bin/python -B -m pytest -q \
  tests/management/test_research_c_integrated_transfer_input.py \
  tests/management/test_research_c_c1_h2_h6.py \
  -k 'integrated_transfer_input or mocked_PySCF_generation'
```

実施結果：**10 passed、13 deselected、0.58 s**。内訳は新規9・既存PySCF mock回帰1。`numpy.load`禁止下で新dispatchを試験し、実PySCF/RHF/積分生成・保存実配列読込みは0。mockで生成仕様を確認したことは実H4の物理予算やSCF収束の証拠ではない。既存移送準備の72 unique PASSは過去の報告であり、この10件へ加算しない。

## Calls・資源・実行依存

新input生成の科学logical callsは、既存instrumentationどおり **SCF1＋spatial_integrals1＝2**。この段階で新SCF/積分を行った実績は0。

追加初期化には、共通sector context/元物理Hamiltonian、XDF initializer、codec、neutral tensor-only fit、二摂動＋各restore、独立value/gradient・tensor/物理/representationを含める。最適化器内部VG/Hessian数はlogical fitter呼出し数と別に実測する。統合契約のcall表で各処理を一回だけ積算し、下流の共通ground/contextを二重計上しない。新input生成2を、旧F16の1629や旧R32の1231へ無断に含めない。旧未使用callの移譲はない。

必要な依存は、developmentのstrength/recipe選定完了→選定の凍結→target input生成・neutral資格→paired optimizer→対象候補固有のPF/RPE/nativeである。固定条件内で独立start・arm・policy・cohortを並列化し、line search/J-refinement/fit process-local bindingは依存順を保持する。修正ffsim bridgeがprocess内の一時bindingを使うため、独立fitは専用processで分離する。共有venvやglobal moduleを変更しない。

H4 sector36/full512の既存matrix/worker/cacheを再利用する。CPU/RAM/worker/wallの人工capは設けず、実行直前にaffinity・空きCPU/RAM・GPU・quota・圧力を確認する。old H4 F16は全体wall92.7661544336 s、親CPU134.805493673 sだったという既存報告があるが、新neutral/restoreや64update・複数startのtarget所要時間を保証しない。未実行targetのruntime実測やspeedupを主張しない。

## 保全・残る判断

旧移送契約・SOURCE・raw・旧P-C・旧PR/ENTRY・update064系列・grant/ledger/marker/STOPは変更しない。今回の差分は新commitに置き、旧固定commitから独立させる。target科学出力は統合契約のfresh namespaceへ保存し、旧runや旧未使用callをretry/resumeしない。

技術準備後はMANDATORY STOP。次担当のユーザーが固定した統合範囲を別途明示許可した場合だけ、Codexが新H4 input生成・初期化・下流を実行できる。1.15 Åの科学実行、H6等の拡張、重要GPTレビュー、論文化GOの自動開始はない。実行承認原文の保存、証明書、新grantを新規要件にしない。

## 追加の静的適合性確認：保存CDF8/XDF8と環境

2026-11-05 JST、親統合担当から依頼された範囲のSOURCE/JSON metadata・報告本文だけを確認した。保存NPZ/QPYのdecode、候補のnorm・価格・shots・統計の再計算は0。これは既存資産の適合性確認であり、新しい重要科学レビューではない。

強baseline監査の実体は `codex/research-c-strong-xdf-cdf-comparison-audit-20261103`、固定commit **`8375433869999c2e10501b1983c11b2b22c5d423`** の [research_c_strong_xdf_cdf_comparison_audit_20261103.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8375433869999c2e10501b1983c11b2b22c5d423/docs/research/research_c_strong_xdf_cdf_comparison_audit_20261103.md)。確認した `origin/main=546bca7…` には同pathがないため、名前の似た別監査で代用しない。

後続の共通ideal laneは [結果commit `39ee9705b7a459c40c205c542d660b5d2ab90d33`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39ee9705b7a459c40c205c542d660b5d2ab90d33/docs/research/research_c_xdf_cdf_common_ideal_lane_execution_20261103.md)、科学SOURCE **`2d20ad196a406752de8e4c0ce9516b101e49d927`**。`run01/SOURCE.json` は同execution_commitを記録し、STOPは61 scientific_units、failed=false、retry=resume0、mandatory_stop=true。旧契約の歴史的status文字列を今回変更しない。この共通ideal結果は確認したmainにもbeb8de5にもないので、固定URL/commitから参照する。

| 比較条件 | 保存CDF8/XDF8と今回developmentの対応 |
|---|---|
| 元入力 | 共通元snapshot `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。PRの `prepare/input.npz` は元snapshotの `h=one_mo,g=eri_chemist,constant=nuclear_repulsion` を保存したpacketでSHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。容器hash差は保存SOURCEの直接コピー来歴で説明される。今回member配列を再比較していない |
| モデル/sector | 直鎖H4・1.00 Å・STO-3G・charge0/singlet・Nα=Nβ2・sector36・system256/controlled512で一致 |
| DF identity | CDF8はB2_CDF_relative、fit SOURCE `da2edb465d5ae6432439fbd1df8e57512ee59c11`、export `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08`。XDF8は元OF1.6.1 signed λ/G・rank10 snapshotの元順prefix8。互いも新PR/ENTRY endpointも同じJ/U・同じ近似Hamiltonianではない |
| Split/PF | 保存baselineは全8frame＋κのordered symmetric S2、s1、h=.1、m0..12。決定論なのでPR split LD/r/Kは非適用（旧r0/K0表示をPR r0と解釈しない）。新PR/ENTRYのpartial側はCDF8/元順LD2 |
| 精度/成功確率 | CA=.00159 Ha、DF reserve=.00019875 Ha、βtotalπ/6、βstatπ/12、候補failure.01、αaxis.01/26、own-DF ground oracle、状態準備除外で一致。達成DF誤差自体は異なる |
| Compiler/precision | P0 Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation_degree1、qubits_initially_zero=false。`nature_jw_exact_zero_lapack_atan2_v1`、MPFR160 declared phase、global phase fitなしで対応 |
| 合法な実装差 | 旧CDFのκは8mode直接eigh。保存XDF B3はspatial eigh→exact spin lift・relative Gaussian fusion、最新PR/同J/U Dはκspin対応。これらの合法route差は保存し、未測定節約の補正や純粋な目的/Uだけの因果帰属を行わない |
| 会計 | head/kernel/readoutをall-wire fenceで分離、shell込みserial kernel composition、scheduled serial depth、長roundのcompile省略によるserial合成。比較の費用scopeが一致 |
| 資格lane | 旧numerical synthesis laneだけでなく、後続で保存候補固有DF/PFを保持した `ideal_exact_native` が両方に存在する。合成誤差0は反実仮定、actual CAはnull。PR側と同じ限定ideal問いへ接続できる |

**再利用の判断**：1.00 Å developmentでは保存CDF8/XDF8の共通ideal値を、共通元task・共通精度/会計における**固定s1の別DF baseline**として再利用できる。保存総RZはCDF8 `102056444569`、XDF8 `100267421030`（既存報告の転記）。新差・比・検定は今回算定していない。異なる近似Hamiltonian/ground/DF誤差と合法native route差があるため、同J/U対照や全方法を最適化した決定論baselineとは呼ばない。新endpointによる公平retuning結果と比べても、旧baselineへ新探索機会を与えたという意味にはならない。

1.15 Å targetには物理入力が一致しないため、この二価格をbaselineとして流用できない。1.00 Åの参照列に限定し、targetでは新候補固有の同J/U強Dを使う。保存baselineを再計算して新候補へ見せたり、別geometry/rank/精度へ価格を移送したりしない。旧update064/r32の同J/U結果と、near-practical PR/ENTRY候補の結果も区別する。

環境のread-only確認では、既存F16契約の実行ファイルが両方存在し、各pythonの `-B -c` で `importlib.metadata.version` のみを問い合わせて以下が完全一致した。科学module・NumPy・JAX・Qiskit・PySCF自体はこの確認でimportしていない。

| 役割 | python・metadata一致 |
|---|---|
| Fit | `/tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python`。ffsim0.0.83、NumPy1.26.4、SciPy1.14.1、JAX/jaxlib0.4.38、Qiskit2.4.2、PySCF2.14.0の7 pins一致 |
| Eval | `/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python`。NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、Qiskit-Nature0.7.2、OF1.6.1、OF-PySCF0.5、PySCF2.7.0、threadpoolctl3.6.0、SymEngine0.13.0、Qiskit-Aer0.15.1の10 pins一致 |

Package metadataの一致はJAX x64/GPU動作やnative backend・新inputの物理適格性の証拠ではない。これらは将来の許可済み実行前の実資源・backend確認で行う。今回、共有venv/package/cache/OS/他jobを変更していない。
