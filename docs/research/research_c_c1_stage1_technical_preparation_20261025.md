# C1 Stage 1 技術実装・人工検証（2026-10-25）

**Stage 1の最小接続処理、独立環境、限定runner、人工検証を完成した。実H4の科学実行は0。** 起点は `826ea4d02407501ee3252d65e5662af18cc49192`、実装SOURCEは `394fcf3a445caf0a5f206b7bbcf4a8c88f971504`。専用branch `codex/research-c-c1-stage1-implementation-20261024`、worktree `/tmp/prt-c1-stage1-implementation-20261024` を使用した。準備は10月24日に開始し、25日JSTに完了したため技術artifactのdirectory日付は開始日のままとした。

ユーザーは前回案の初回条件を採択し、技術実装・人工検証を承認した。科学実行は承認していない。[未承認科学契約](research_c_c1_stage1_science_contract_unapproved_20261025.json)は `execution_authorized=false, grant_issued=false`。残る判断は、下記の工学的数値allowanceを限定pilotに採択するか、および固定SOURCE/input/契約/call/STOPを指定する新たな科学実行承認である。Stage 1の科学的PASSや資源優位は得ていない。

## 1. 採択済み科学条件と境界

保存P-H4（直鎖H4、隣接1.00 Å、STO-3G、charge 0、spin 0、4空間/8 spin軌道、保存canonical RHF MO）、Nα=Nβ=2の36次元sectorを対象とする。ffsim v0.0.83通常C-DF、number表現、元gから補正したκ保持、`max_vecs=8, tol=1e-8, L-BFGS-B` の**1 fit**。一般Kは元の1 frameを維持し、同一入力の保存実rank10 X-DFを対照とする。

物理DF誤差はCDF/XDFを別々のsector spectral normで測定・報告する。絶対DF許容値を設けず、`CDF≤XDF`を必須gateにしない。adapterはsector合計とfull256のbase/各frameで同値性を判定する。full physical/XDF norm、energy、回路、Stage 2/3は対象外。`L_D`・delta窓・PF/RTE/sampling/shot条件は該当しない。今回実施したのは、この将来経路の人工検証である。

## 2. SOURCEの再利用と最小実装

| 実装 | 接続した処理・再利用元 |
|---|---|
| [c1_stage1.py](../../src/trottertracks/c1_stage1.py) | lazy importの数値glue。既存ffsim lower-levelを1回呼ぶfit/export準備、独立reference経路、既存adapter取込、保存XDF sector処理、norm/allowance/構造診断 |
| [c1_stage1_execution.py](../../src/trottertracks/c1_stage1_execution.py) | stdlibのSOURCE/input/承認gate、独立fit subprocess、NPZ+JSON受渡し、call journal、partial出力、STOP。grantは作らない |
| [限定runner](../../scripts/run_research_c_c1_stage1.py) | 既定はbyte/hash preflightのみ。科学実行entryは新たな明示承認packetとfresh outputを要求。hidden fit workerも同じ承認を照合し、exclusive `fit.claim` で再fitを拒否 |

新fitter・JW algebra・Gaussian合成器・linear algebraを作っていない。次をそのまま再利用した。

- 既存 [`c1_general_k_adapter.py`](../../src/trottertracks/c1_general_k_adapter.py)：明示number/κ role、J→Kの1/2、U→interleaved V、元frame順序、I/Z/ZZ係数。ファイルは変更していない。
- 固定ffsim `558d3517385778c733f1aa226bb61b8f30f4b0f3` の `linalg.double_factorized(...return_optimize_result=True)`。factoryは使わず、元gから `κ=h−0.5 einsum("prrq",g)` を計算して元cと一緒にexportする。status取得のための二重fitはない。
- [`reference.car_annihilator`](../../src/trottertracks/pr_dictionary_validation/reference.py) のgeneric bit作用。8-modeにも適合し、NumPy allocation contextとsparse bilinear cacheを薄く接続した。4-mode保存reference・square fixture・旧PASSは流用しない。
- 元P-H4生成commit `0c329b30a81accb35c212be8defa66fee0dd0f12` の純粋 `spin_lift_unthresholded` だけを移植した。元moduleのimportには不要な科学pipeline依存があるため、SCF/生成関数を含めずこの小helperだけを複製した。係数配置・出所をコード内に記録した。
- [`df_hamiltonian.py`](../../src/trotterlib/df_hamiltonian.py) のDFHamiltonian/PhysicalSector/df_linear_operatorと既存basis-column dense化。既存square kernelを保存XDFに使用し、新分解・再hermitization・ground-state solveをしない。`trotterlib.__init__`の広い回路importを避けるprivate namespaceに**実moduleと実relative imports**をロードする。mock計算への置換ではない。
- OpenFermion1.6.1の既存JW sparse evaluator、SciPy1.14.1のSVD、NumPyの収縮・積。OpenFermion symbolic `+=` の `EQ_TOLERANCE` による係数削除を避け、unique term mapを直接渡す。

`df_trotter/`、`df_partial_s2.py`、`df_rte_tail.py`、`df_rte_qiskit.py`は前回静的調査の通り後段資産として維持し、今回のrunnerは呼ばない。basis registry、Gaussian/controlled/adjoint/compile、tail lawをStage 1へ埋め込まない。

## 3. 独立Hamiltonian経路とscalar/ordering

| 経路 | 入力と構成 |
|---|---|
| 物理H | 元h/g/c → unthresholded spin lift → InteractionOperatorの係数 → 無閾値FermionOperator term map → 既存JW sparse → sector36 |
| raw CDF | exportのraw κ/J/U/c → generic CAR `a†a` → `n_iσ^U=Σ_pq U_pi U*_qi a†_pσ a_qσ` → `0.5 Σ_lijστ J_lij n_iσ n_jτ` |
| K candidate | 既存adapterのK/VとI/Z/ZZ係数 → 独立OpenFermion bilinearから `Z_a^V=I−2n_a^V` → 行列上の積和 |
| 保存XDF | 同packetのκ/λ/G/c → 既存square sector matrix-free → 36本のbasis action |

raw referenceはadapterのK/V/Z係数を使わない。candidateはraw CAR assembly helperを使わない。入力と数値libraryの共有はあるが、density-productとI/Z/ZZの変換経路を分ける。これは代数的U/V方向の検証であり、実Gaussianゲートの規約検証ではない。

spin modeは `a=2p+σ`、共通行列はmode aをbit aに配置する。OpenFermionとPhysicalSectorの最上位bit規約からは一度だけbit reversalし、同じsector順序へ写す。κと元nuclear cはmodel levelで各1回、各frameのI項はframe内に残す。保存 `spin_two` をffsimのspatial gに渡さず、truncation_valueをscalarへ足さない。

同じN=4 sectorだけでは `α(N−4)` のbase誤りを隠せるのでfull256のbase/各frameも比較する。frame間の誤りの相殺も避ける。保存XDF κと元gからのκの数値一致は将来の構造gateで確認し、不一致ならSTOPする。過去のenergyやPASSを新norm値として再利用しない。

## 4. 独立環境と依存lock

| 環境 | 準備・確認済み |
|---|---|
| fit専用 `.venv-fit` | CPython3.11.1、ffsim0.0.83、NumPy1.26.4、SciPy1.14.1、JAX/jaxlib0.4.38、PySCF2.14.0、Qiskit2.4.2。22 distributionsをexact版・wheel SHA256で固定 |
| 評価専用 `.venv-eval` | CPython3.11.1、NumPy1.26.4、SciPy1.14.1、OpenFermion1.6.1、OpenFermion-PySCF0.5、PySCF2.7.0、threadpoolctl3.6.0等。41 distributionsを同様に固定。Qiskit/ffsim/numba不要 |
| 検査 | 両venvの `pip check` PASS。fitは固定commitのPythonファイル102件をinstall済みとbyte照合し差0、JAX x64=true、CPU deviceを確認。実fitterはmockで置換したtestだけ |

[全lock・wheel inventory・実runtime](../../artifacts/research_c_c1_stage1_technical/2026-10-24/environment/)を保存した。wheel本体・venv・cacheはtask領域のGit管理外に置く。共有venv（旧Qiskit1.3.0/PySCF2.7.0）をupgradeしていない。ffsimが要求するQiskit≥2/PySCF≥2.12との衝突はprocess/venv境界で避けた。全Python SOURCEの照合と配布wheel hashを記録したが、Rust native extensionのsource再buildまでは要求していない。

配列境界はpickleなしNPZとJSON。κ[4,4]、J/U[R,4,4]、c[]、optimizer x/jacはfloat64で保持し、評価operatorはcomplex128。input/member/output hash、shape/dtype、number flag、one-body role、SOURCE/環境identity、実R、frame順序、OptimizeResult、最終tensor residualをexportする。元MO h/g/cはload whitelistで取り込み、保存state/matrix/energyはdecodeしない。今回そのwhitelistを使った実P-H4 decodeは0。

再作成時はPython3.11.1の新venvに `pip install --require-hashes -r <role>-requirements.lock` を使う。保存wheelhouseを使うなら `--no-index --find-links .wheelhouse/<role>` を追加できる。これは環境手順であり、科学実行commandではない。lockはLinux x86_64/CPython3.11向けで、別platformへ無断に読み替えない。

## 5. 人工検証結果

**新規32件（数値/runner27、fit環境5）＋既存adapter13件＝45件PASS、加えて既存29 subtests PASS。fail/skip 0。** [test summary](../../artifacts/research_c_c1_stage1_technical/2026-10-24/test_summary.json)とJUnit原本を保存した。ローカル人工検証の証拠であり、分子科学結果やimmutable CI再現ではない。

| tests | 確認した内容 |
|---|---|
| [数値接続tests](../../tests/management/test_research_c_c1_stage1.py) | 非対角/対角/負J、実非対称U・複素U、κ/c、spin/bit parity、full256/sector36、複数frame、微小係数保持、誤ったI係数の検出、構造FAIL時停止、非ゼロ人工gの1/2/κ/signed-square整合、1回のmock fit |
| [実行境界tests](../../tests/management/test_research_c_c1_stage1_execution.py) | preflightで数値decodeなし、hash不一致、未承認拒否、call超過拒否、load whitelist、mock環境間export/照合/operator保存/STOP、worker失敗と親中断でretryなし |
| [fit環境tests](../../tests/management/test_research_c_c1_stage1_fit_environment.py) | 実API signature、102 SOURCE照合、JAX x64 JITの人工scalar勾配、mockした実ffsim入口を1回だけ使用、既存fit.claimで再fit拒否、raw出力/metadata保全 |
| [既存adapter回帰](../../tests/management/test_research_c_c1_general_k_adapter.py) | 前回のデータ/係数/mock bridge仕様を保持。実回路buildは行わない |

実施コマンドは `.venv-eval/bin/python -m pytest -q tests/management/test_research_c_c1_stage1.py tests/management/test_research_c_c1_stage1_execution.py`、`.venv-fit/bin/python -m pytest -q tests/management/test_research_c_c1_stage1_fit_environment.py`、評価venvの既存adapter testである。cache/Matplotlib/bytecode出力先はtask専用に指定した。

新規科学callは0、実ffsim fitは人工入力に対しても0。実H4に対する行列、sector norm、tensor residual、energy、保存科学値の再解析、実回路/compileは0。P-H4はGit blobのSHA256を再照合し、[stdlibのみのpreflight](../../artifacts/research_c_c1_stage1_technical/2026-10-24/preflight.json)を行った。npz decodeも数値module importもない。

## 6. 数値allowance：実装した係数と採択に残す判断

版 `binary64_operation_scale_guard8_v1`。**限定pilot向けの工学的roundoff allowance案**であり、厳密interval certificateではない。前回draftでnullだった演算長・guard・norm係数を具体化し、人工の同値例が通り、意図的な誤りが落ちることを確認した。実H4結果を見て調整した値ではない。

`u=2^-53`、`γ_m=m u/(1−m u)` とし、spectral比較は

`τ=8[γ_mL B_L+γ_mR B_R+32 d u ‖Δ‖_F]`。

Frobenius構造診断はこれを `sqrt(d)` 倍する。dはその診断行列の寸法（κ/Uは4、spin κは8、sector36、full256）。Bは相殺前の係数絶対和から計算する。最後の項は差行列のnorm評価roundoffに対する固定規則であり、観測残差を使って係数・閾値を調整する仕組みではない。

| 演算経路 | mの規則（N=空間軌道、M=2N、D=2^M） | H4のm |
|---|---|---:|
| raw frame | `8D+8N²+32N²+32` | 2720 |
| K frame | `8D+8M²+8[1+M+M(M−1)/2]+2M²+32` | 3016 |
| base | `8M²+32` | 544 |
| physical sparse | `8(M⁴+M²+1)` | 33288 |
| saved XDF | `16M²+8R_XDF+32`、保存R_XDF=10 | 1136 |
| U unitary | `8N+8N²+16` | 176 |

複素multiply/addを高々8実演算分、GEMM dot長D、density/係数の和の長さ、I/Z/ZZ項数、scalar操作を計上した保守的な演算経路長である。raw/Kの合計には `8R` を追加する。全FLOP数をそのままγの添字に使っていない。

B_rawは `2Σ_ij |Jij|s_i²s_j²`（`s_i=Σ_p|U_pi|`）、B_Kは係数絶対値と `b_a=1+2(Σ_p|V_pa|)²` の積和、baseは `|c|+2Σ|κ|`。physical/XDF側の係数scaleも契約へ固定した。unitarityやHermiticityを黙ってprojectせず、構造gateを先に確認する。旧C4の1e-10をコピーしていない。

guard8とnorm係数32dは、使用library全経路の厳密な最大誤差定理から導いた定数ではない。SVDのbackward stabilityと演算長・scaleを踏まえたpilot用の余裕であり、広い条件数域について認証したものではない。この精度水準をpilotに採択するか、より強いreference/誤差保証を求めるかは**統括・ユーザーの判断事項**とした。runnerは別承認の `numerical_allowance_accepted=true` とこの版を要求する。絶対物理DF誤差thresholdは追加しない。数値floor近傍のCDF/XDF差から優劣を認定しない。

## 7. 科学call・出力・STOP

正常終了時のR依存規則を実装と契約で照合する。Rは1 fitが返す正の実rank≤8で、今は未測定。callは論理job/診断の種類を区別し、内部のNumPy積やhashを独立のfitとして数えない。

| 種類 | 正常経路の回数 |
|---|---:|
| input decode / export decode | fit input1、評価input1、export1 |
| fit / fit資格 / tensor再構成 / maxabs残差 / F残差 | 各1 |
| adapter / Hphys / 保存XDF sector | 各1、XDFは別途36 basis actionを記録 |
| full base / frame / raw-K full合計 / sector制限 | 2 / 2R / 2 / 2 |
| spectral norm | R+4（base1、framesR、adapter sector1、CDF/XDF物理差各1） |
| 構造Frobenius | 2R+11（κ/correction3、J/U各R、4 Hamiltonian Hermiticity、Nα/Nβ commutator4） |
| scale/allowance | 3R+15、各上記normに1回。内部の絶対和・F normを含む |

全counter単位を足すと正常時 `83+8R`、R=8で147。これはnested basis actionやallowanceも含む帳簿上の合計で、独立した重い科学runの件数ではない。Hamiltonian構成jobは `6+2R`。solverのnit/nfev/njevは固定SciPy既定に従い別記録する。fit初期Choleskyはその1 fitの内部1回であり追加fitではない。

実装は科学job開始前にattemptをledgerへ記録する。最初の構造/同値性FAILで停止し、並列投入済みのframe対は消費したprefixとして残す。生のJ/U/x/jac/funとstatusを資格判定前に保存し、未収束もpartial exportだけを残してoperatorへ進まない。すでに構成したoperatorは再計算なしで `8+2R` 個のNPZへ保存し、hash/shape/dtypeのinventoryを残す。

hash/schema/補正/status/非有限/call/承認欠落は技術的STOP、未収束・規約/sector不整合・adapter/構造FAIL・XDF不適格は科学的STOPまたは保留。retry/resume、rank/tol/seed/optimizer/reference precisionの自動変更は0。SIGINT/SIGTERMではこのtaskが新規作成したfit子process groupだけを終了してpartial/STOPを保全する。SIGKILL等でSTOPを書けない中断も、fresh outputとfit.claimによって再利用を拒否し、ユーザーへ戻す。

CPU/RAM/worker/phase/total wall-timeの恣意的な上限はない。確認時は32 CPU、利用可能RAM約52 GiB、memory pressureなし、GPU driver利用不可。raw CAR bilinearを共有cacheし、frame対を `min(R,使用可能CPU)` worker、BLASは使用可能CPUをworkerで割ったthread数へ調整する。実run前に空き資源・他job負荷を再確認する。256×256 complex128は約1 MiB/行列で、bilinear cache等のaggregate RAMを考慮する。速度向上倍率は主張していない。

## 8. 固定情報、保全、次の担当

[SOURCE manifest](../../artifacts/research_c_c1_stage1_technical/2026-10-24/SOURCE.json)は実装commitの22 source/環境ファイルのSHA256とffsim Python102ファイルを固定する。実行承認はさらに公開commit・契約hash・SOURCE manifest hash・input hash・fresh output・one-fit・allowance版を結び付ける。manifest自身を同じcommitへ自己参照させず、実装commitと公開契約commitを分けた。

固定P-H4 packet SHA256は `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。入力はGit commit `86b8b002704aced8ca239414080f53793de81658` の原本を保全する。元workspaceのtracked diffと保護文書hashを照合し、旧SOURCE/結果/grant/ledger/marker/STOP、C4終了、Gaussian singleton保留、共有環境を変更していない。

- **統括・ユーザーの次判断**：工学的allowance案をpilotに採択するか判断する。より強い精度保証が必要なら、実H4を走らせる前に必要差分を指定する。
- **ユーザーの次承認**：採択後、固定SOURCE/input/未承認契約/call/STOPとfresh outputを指定し、Stage 1科学実行を新たに明示承認する。
- **Codexの次作業（その承認後）**：1 fitと登録済み比較だけを実行し、CDF/XDF物理誤差とadapter資格を別報告してSTOPする。Stage 2/3へ自動的に進まない。

今回の完了範囲は技術実装・人工検証・契約公開まで。公開とremote照合後はmandatory STOPし、科学実行を開始しない。
