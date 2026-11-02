# 研究C・C1：元Hamiltonian入力回収・ffsim通常C-DF接口・初回検証準備

2026-10-24 JST。技術状態：**C1_INPUT_AND_INTERFACE_READY_FOR_IMPLEMENTATION_PROPOSAL**。

関連する固定branchから、元MO一電子・二電子積分、MO係数、核反発項と生成来歴を持つH4 packetを回収できた。ffsim通常C-DFへ接続する設計を具体化できる。ただし、この状態は**実装提案を判断できる**という意味であり、科学的GO、対象分子・sector・rank・誤差条件の採択、実装や科学実行の許可ではない。

今回の変更は本書1件だけ。SOURCE・共通研究文書・manifest・過去の研究記録・科学結果を変更しない。新規adapter、環境構築、fit、積分生成、Hamiltonian再構成、回路build/compile、Operator/Statevector、norm/energy/残差評価、保存数値の再解析は実施していない。

## 1. 目的、固定資料と方法

C1の目的は同じ元物理Hamiltonianを起点に、圧縮DFとnative PR回路費用を共同で改善できるか調べること。通常C-DFは初期接口確認の優先候補であり、最良方式としての正式採択ではない。[前回C1調査][C1]と[研究C初期調査][INITIAL]を再利用し、[C4一時終了][C4]とGaussian singletonの停止を維持する。

- 作業起点：`ec27625839564dc22db2728923120f497b7cadee`。今回の独立branch：`codex/research-c-c1-input-ffsim-readiness-20261024`。
- 外部SOURCE：ffsim **v0.0.83**、`558d3517385778c733f1aa226bb61b8f30f4b0f3`。前回取得済みの固定SOURCEを再利用し、周辺MolecularHamiltonianと依存関係も確認した。
- 正式共通指示：`research_project_instructions_unified_2026-10-23.md`、SHA256 `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f`。§8.4・§10・§10.3に従い、必要な入力同定と接続設計を優先し、既存契約・共有環境を保護した。
- 方法：固定Git tree/objects、既存SOURCE・tests・生成receipt/contractの静的読取り。NPZはZIP/NPY headerからmember名・shape・dtypeを確認し、文字列metadataとbyte hashを読んだ。科学arrayの値はdecodeせず、モジュールをimportしなかった。GitHubでは候補ディレクトリのblob/sizeとH4 receiptも確認した。

## 2. 探索範囲と入力分類

関連性の高い次の固定treeを探索した。branch間をmergeせず、保存Git objectをread-onlyで参照した。全repository履歴・全サーバーを網羅した不存在証明ではない。

| 固定tree | 探索理由と結果 |
|---|---|
| 起点 `ec27625839564dc22db2728923120f497b7cadee` | 既知H4/H5はsquare-DF packetのみ |
| 元workspace HEAD `e098c54c78f589055082f9cfc2b13de50c90ca94`、PR2-v4 `b2e1bf65e21893b6c617223b42313623d3186f12` | H4 development/held-out snapshotを確認。保存DF表現で、元integralsは含まれない |
| Hamiltonian construction feasibility `ae7434452450e80ec007d65417335e18a063fa1e` | 設計・診断・SOURCEを確認。元分子積分packetを追加で回収できなかった |
| H4 input generation/storage review `2094c07eb445463ea2dbcd574a6124a1d90f86a8` | 生成計画・承認準備の存在と実データ保存を区別。調べたtreeには元積分packetがない |
| H4 exploratory readiness `868ee7ced6818b2531a3aa9c12498f02bce55dc3` | 既知DF snapshot等を確認。新しい元H4積分をこのtreeでは確認できなかった |
| 非DF H4 reference evidence **`86b8b002704aced8ca239414080f53793de81658`** | 元積分入りのH4/H2 packetを発見。以下の主な回収元。既存runの状態や科学結果を変更しない |

| 候補 | 区分 | 取得できたもの・不足 | 初回への位置付け |
|---|---|---|---|
| **P-H4：非DF H4 pilot input** | **A：元物理入力と来歴を確認** | 元MO h/g、C_MO、核反発、spin tensor、保存X-DF、生成SOURCE/契約/runtime、receipt hash一致。AO h/ERI/overlapの独立arrayはsnapshotにない | 本命候補。小分子で一般Jの接口とPR下流を調べられる。採択は統括へ戻す |
| **P-H2：非DF H2 pilot input** | **A：元物理入力と来歴を確認** | 同種の元MO packet、生成SOURCEとmarker。今回byte hashを同定。H4のようなreceipt内snapshot hashは確認できなかった | 小さい追加smoke候補。圧縮による研究上の利益の判別力はH4より限定的。追加実行を自動提案しない |
| H8 saved integrals | **B：確認部分に不足** | spatial/spin integral・canonical orbitals・constantのheaderを確認したが、生成bindingと全規約metadataの照合は未完 | A候補が得られたため深掘りを止めた。初回pilotには推さない |
| 既知H4 rank12、H4 1.30Å、H5 rank9、別研究のH4 tol-only snapshot | **C：近似DF表現のみ** | constant/one-body/lambdas/G等。元pre-DF積分はそのpacketに含まれない | 保存近似Hの再表現pilotとしてなら別候補。元物理Hへのfitと呼ばない |
| 生成SOURCE・計画だけの経路 | **D：使用可能な元入力を未確認** | PySCF呼出しやsession cacheはあるが、生成済みtensor保存の証拠にならない | 今回起動しない。A候補の代わりに再生成する必要はない |

Aは文書・schema・byte identity上の分類であり、今回operator同値性・積分品質を数値的に再検証した意味ではない。前回の「起点treeの既知packetだけでは元inputが不足」は当時の探索範囲の結果として正しい。今回、関連branchの別packetで不足を補った。

## 3. P-H4：元物理Hamiltonianのidentityと証拠

固定回収元は `86b8b002704aced8ca239414080f53793de81658`、pathは

`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`。

[保存packet所在][H4DIR]、[receipt][H4RECEIPT]、[contract][H4CONTRACT]、[生成SOURCE binding][H4SOURCE]を併読する。科学配列を新しいbranchへ複製せず、固定位置を参照する。

| 同定項目 | 確認結果 |
|---|---|
| byte identity | 2157448 byte、SHA256 **`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`**。receiptの `snapshot_sha256` と一致。Git blob **`5828b6b225bb4990a46934edba152b82ec97782f`**とsizeをGitHubにも照合 |
| 物理条件 | linear H4、H位置 `(0,0,0),(0,0,1),(0,0,2),(0,0,3)` Å、STO-3G、charge0、PySCF spin_2S0、symmetry=false。active_space/frozen_coreはnull、全4空間軌道・8 spin orbital |
| 電子数とsector | 元生成SOURCEはNα=Nβ=2のspin sectorを使用。保存Qiskit sector indicesのshapeは(36,)。これは保存入力の来歴で、C1の対象sectorは未採択 |
| AO/MO | canonical real RHF MO。SOURCEは `h_MO=Cᵀ h_AO C`、PySCF ao2moとrestore(1)でchemist gを保存。C_MOも保存済み。AO h/ERI/Sそのものはsnapshot memberにないが、保存MO Hamiltonianを定義するh/gは揃う。AOからの再生成は不要 |
| constant | `nuclear_repulsion` scalar arrayを保存。SOURCEは同じ `enuc` を元HとDFへ渡す。値の再計算なし。別のfrozen-core補正は採用していない |
| spin lift | interleaved index `2p+σ`。`spin_one[a::2,a::2]=h`、`spin_two[a::2,b::2,b::2,a::2]=g.transpose(0,2,3,1)/2`。保存spin_twoは既に1/2込み |
| one-body correction | 保存 `df_one_body` は元spin hにOpenFermionのnormal-order補正を加えたDF用one-body。独立補正arrayはない。差分を今回計算していない。元hは `one_mo`／`spin_one` に別保存 |
| X-DF保存条件 | receipt/SOURCEは `df_rank=None, df_tol=1e-8`、保存df_lambdasのshape(10,)。これは旧rank12とは別packet。C1 baseline採択や新しい誤差保証を意味しない |
| 生成版 | 当時のSOURCE **`0c329b30a81accb35c212be8defa66fee0dd0f12`**。[h4_molecular_inputs.py][H4GEN]のGit blob `105c41dbd51e34d74026cfa5f7a382d76678a1f7`。SOURCE.jsonのファイルSHA256 `fb53bf44459dc6ef18223d55594511c03766c53537f6c445dad7a0b7a42225ea` と一致し、回収元treeでも同一blob |
| 生成runtime | SOURCE記録：NumPy1.26.4、SciPy1.14.1、OpenFermion1.6.1、openfermionpyscf0.5、PySCF2.7.0、Qiskit1.3.0、Nature0.7.2。RHF conv_tol1e-12/minao/max_cycle100は**当時の生成条件**で、今回呼出しなし |

NPY header照合結果：

| member | shape | dtype |
|---|---|---|
| `mo_coeff` | (4,4)、Fortran order | `<f8` |
| `one_mo` | (4,4) | `<f8` |
| `eri_chemist` | (4,4,4,4) | `<f8` |
| `nuclear_repulsion` | scalar | `<f8` |
| `spin_one`／`spin_two` | (8,8)／(8,8,8,8) | `<f8` |
| `df_one_body`／`df_lambdas`／`df_g_matrices` | (8,8)／(10,)／(10,8,8) | `<c16`／`<f8`／`<c16` |

同packetのstate・DF_state・non_DF_matrix・DF_matrix・energy等はheaderの存在だけ確認し、値を読まなかった。旧保存matrixを新C-DFのreferenceと無条件に扱わない。`scf.chk`も固定ディレクトリに存在するが内部datasetは未検査。必要な元MO payloadはNPZ側にある。

保存receiptの旧PASSは再検証していない。SOURCEの生成worker binding、receipt hash、file schemaを技術照合したもので、当時のgrant/STOPをC1へ流用するものではない。

## 4. 他候補のidentityとpilot提案

**P-H2**は同回収元commitの `artifacts/non_df_h2_pilot/2026-10-22/run01/inputs/snapshot.npz`。[所在とreceipt][H2DIR]。16784 byte、SHA256 **`fb5ca11d7f0be73abd063ae7b67c6b72399c5a2336635f3764af1a7316e81175`**、Git blob `ce6741e371927d23a6cf69e243251565e085d3e5`を照合した。MO C/hは(2,2)、gは(2,2,2,2)、nuclear scalar、すべてfloat64。保存条件はH2 1Å・STO-3G、charge0・spin0、全軌道、Nα=Nβ=1。生成SOURCEはmarkerの `b5ed2c6e2399b55a0c24eb8437f842892a6703ef`、[h2_non_df_pilot.py][H2GEN]のblob `c28d8dc18f3e6b8f0de04d5d657817ff31a3830f` は回収元と同一。元MOから利用すれば、旧spinorb_from_spatialの閾値処理を新C-DF入力へ引き継ぐ必要はない。

**H8（区分B）**は同回収元の `artifacts/resource_applicability/pr_pf_s2_recomputed/2026-10-22/units/H8/SYSTEM/input_preparation/integrals.npz`。SHA256 `72a800938a9329a422774526a4086887735e0b490beb8349cf6046910be88612`。spatial h(8,8)/g(8,8,8,8)/canonical_orbitals(8,8)、spin h(16,16)/g(16,16,16,16)、constantのheaderは確認済み。生成binding・規約の全照合は今回の初回候補選別に不要なため保留した。

**区分Cの例**は起点treeの既知connected-cluster H4 packetで、schema `connected_cluster_df_hamiltonian_snapshot_v1`、SHA256 `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。metadataのrank12は元gを持つことを意味しない。PR2-v4のH4 1.30Å snapshotもconstant/one_body/lambdas/G等で、元gはない。これらを再構成して比較する場合は「保存近似Hの再表現」であり、新しい元物理H比較とは分ける。今回再構成していない。

**提案：P-H4を最初の本命inputとして統括へ戻す。** 来歴と元h/gが揃い、保存X-DFとの同一input関係を追えるためである。P-H2は必要なら小さい接続確認の選択肢とし、両方を自動的に実行対象へ追加しない。元packet回収は完了したが、P-H4の採択、保存X-DF使用か新しい共通X-DF作成か、C-DF rank、sector、DF誤差基準は未決定。

## 5. ffsim固定SOURCEと入出力契約

固定commitの[Hamiltonian][FFH]、[fit][FFFIT]、[MolecularHamiltonian][FFMOL]、[Trotter][FFTROT]、[pyproject][FFPROJECT]を静的確認した。Git blobは順に `fd87cc70f1a3b3d37349ab0bc6790b68166e9f45`、`2eac9e1690ddf5fb7779136aa7a94b04b02de526`、`7a5a9dd647026b7db4ed1068184d4eec3df9dbfe`、`ad91992a7f9973f3d16641e03af5f6f61b3cf536`、`eb3f1ad809393f3f58c285ba0c20889809650bb0`。

### 5.1 元MO入力と出力schema

ffsim `MolecularHamiltonian(h,g,c)`のsecond-quantized形は

`H_phys=c+Σ_pqσ h_pq a†_pσ a_qσ + (1/2)Σ_pqrsστ g_pqrs a†_pσ a†_rτ a_sτ a_qσ`。

これは保存 `one_mo,eri_chemist,nuclear_repulsion` に対応し、元spatial gには1/2を前掛けしない。保存spin_twoやdf_one_bodyをこのspatial入口へ直接渡さない。

`DoubleFactorizedHamiltonian.from_molecular_hamiltonian` のnumber表現出力は

`H_CDF=c+Σ_pqσ κ_pq a†_pσ a_qσ + (1/2)Σ_lijστ J_lij n^l_iσ n^l_jτ`、

`n^l_iσ=Σ_pq U_lpi a†_pσ a_qσ U*_lqi`。

| field | schema・意味 |
|---|---|
| `one_body_tensor` | (N,N)、κ。元hとは別で、下記normal-order補正後 |
| `diag_coulomb_mats` | (R,N,N)、一般実対称J。rank1/PSD/疎を保証しない |
| `orbital_rotations` | (R,N,N)、U。固定fitは実gのみを受け、実回転パラメータで最適化 |
| `constant` | scalar。number表現では元cを継承 |
| `z_representation` | number/Zの属性。falseがAPI既定。Z表現ではκとcも変わる |

`from_molecular_hamiltonian` は `κ_pq=h_pq−(1/2)Σ_r g_prrq` を**元g**から計算し、gだけをfitする。`to_molecular_hamiltonian` は `g_fit=Σ U U* J U U*` を再構成し、`h_return=κ+(1/2)Σ_r(g_fit)_prrq` を返す。

したがって近似fitでは、roundtripのone-bodyは一般に元hと同じとは限らない。これはSOURCEの静的な式の帰結であり、今回差を数値評価していない。既定のκ固定のdensity-product近似を比較するか、normal-ordered hを固定する別政策を用いるかは科学条件として明記する。後者を無断で実装せず、まずAPIの出力H_CDFを忠実に保持する接口を提案する。新旧表現を比較する際に、別々のone-body政策を混ぜない。

`to_z_representation` はJのrow sumをUで戻したone-body補正と `1/4Σ_li J_lii−1/2Σ_lij J_lij` のconstant補正を加える。`to_number_representation`はそれらを戻す。Zフラグだけ変更したり、number→Z展開とこの補正を二重適用したりしない。初回接口は**number表現を候補**とするが、正式採択ではない。

### 5.2 通常C-DFのfit挙動

| 項目 | 固定SOURCEから分かること／将来記録すること |
|---|---|
| 入口 | `from_molecular_hamiltonian(...,optimize=True)` → `linalg.double_factorized`。新しいfitterを作る必要はない |
| 目的 | 実tensorの `1/2||g−g_fit||_F²`。係数正則化・SCDF shiftなし。別のt2振幅regularizationとは区別 |
| 初期化 | modified Choleskyで初期J/Uを作る。`optimize=True`では `cholesky=False` を指定しても初期化をeighへ切り替えられない |
| rank | `max_vecs`はouter frame数の上限。初期Choleskyがtolで先に止まれば実Rは小さい。最適化は得られたRのJ/Uを更新し、固定の指定rankを必ず作る関数ではない |
| tol | APIはtensor max-absolute誤差として説明。SOURCEの初期化はCholesky残差を用いる。rank上限で先に止まればtolを超え得る。非線形optimizerの収束条件や最終tensor誤差の認証ではない |
| optimizer | SciPy minimize、既定L-BFGS-B、jac=True。JAX JIT/value_and_grad、構造別cache。SOURCEはjax_enable_x64=Trueを設定 |
| J support | `diag_coulomb_indices`でupper-triangle supportを制限できるが、別の近似政策になるため初回条件として未採択 |
| 終了・残差 | lower-level `double_factorized(...,return_optimize_result=True)`はOptimizeResultも返せる。Hamiltonian factoryはそれを返さず、SOURCEはsuccessを科学gateとして強制していない。将来success/message/nit/nfev/実Rと最終残差を保存して判断する |

将来のfit出力に「tol指定済み」と書くだけでは不十分である。tensor残差のnormとoperator/sector誤差の評価は別の承認済み科学callにし、最終誤差・optimizer statusを確認してから回路価格を比較する。今回どのfit入口も呼び出していない。

## 6. 環境分離と将来exportの案

固定pyprojectはPython≥3.11、**Qiskit≥2.0・PySCF≥2.12**、JAX/NumPy/SciPy等を要求する。元入力の生成runtimeはQiskit1.3.0・PySCF2.7.0であり、共有venvをupgradeして合わせない。

提案経路は、**既存の元MO packet読取り → 独立fit専用環境のffsim → 規約付きarray export → 既存Qiskit1.3.0のDF/PR環境**。ffsimから既存環境へPython object、pickle、QPYを運ぶ必要はなく、NumPy arrayとJSON metadataを境界にする。fit側でPySCFが依存にあっても、元integralsを再生成せず保存h/gを使う。初回fit環境のpackage版・JAX device/64bit・実resourceは将来の実装準備時に記録する。

export schemaの最小案（**未生成**）は、`κ[N,N],J[R,N,N],U[R,N,N],constant` と、source packet SHA256/member、元h/g規約、元geometry/unit/sector来歴、ffsim commit/実dependency、number/Z属性、one-body政策、fit設定/結果、実R、各配列shape/dtype/hash。元MO Cと入力h/gを参照可能に保ち、field名だけでnormal-order済みか推測しない。将来float64/complex128を使う場合も実dtypeを記録し、必要精度を無断で落とさない。

今回venv・package・cache・環境変数を変更せず、環境作成・install・fit・array exportは0。

## 7. 一般対角fragmentの最小設計

### 7.1 JからK、符号・prefactor・位相

既存SOURCEの `λ(a†Ga)²` はsquare型であり、一般JをGへ代入すると別のHになる。新しい一般対角frameを1 frameのまま保持する案では、全prefactor込みの実対称Kを使い

`D_l(K)=Σ_ab K^l_ab n_a n_b`（対角と両順序を含む）と定義する。

ffsim spatial Jからinterleaved spinへは、**`K_(2i+σ),(2j+τ)=J_ij/2`**。spin-block表記を介すなら `b=i+σN → a=2i+σ` のpermutationを使う。Gaussianは `V_(2p+σ),(2i+τ)=δ_στ U_pi`。この配列配置とQiskit little-endianのstate index変換は別で、bit reversalを二重に行わない。

`n=(I−Z)/2`から得る係数は次の通り（代数式だけで、今回は評価していない）。

- `c_l=(1ᵀK_l1+tr K_l)/4`
- `z_la=−(Σ_b K_lab)/2`
- `z_lab=K_lab/2`（a<b）

`exp(−iτD_l)`は `DiagonalEvolutionPrimitives(global_phase=−τc_l, rz=(a,2τz_la), rzz=(a,b,2τz_lab))` に対応する。対角K_aaはn_a²=n_aとしてscalarとZへ入る。同じ空間軌道でも異なるspin a≠bは正当なZZを持つ。`K=ληηᵀ`への退化で、現行 `df_squared_diagonal_primitives`／`exact_df_diagonal_coefficients`の符号・係数と式上整合する。数値的な退化testは将来の人工検証で行う。

既存 `append_diagonal_primitives` はRZ/RZZとglobal phaseを受け、controlledではphaseをancilla PhaseGateとして保持する。外側の元constant c、各frameのc_l、one-body diagonalのscalar、RTE identityはpolicyに従い各1回だけ入れる。uncontrolledのglobal phaseを消してからcontrolしてはならない。

Gaussianは既存 `U_to_qiskit_ops_jw_givens(V)`を再利用し、既存helper内部の `BogoliubovTransform(V.T)` 規約を維持する。現行partial-S2の適用順序はbasis inverse、中心、basis forward。一般frameでの目標作用は行列として `Γ(V) exp(−iτD) Γ(V)†` であり、helperとffsim Uの定義がこの作用に適合するか独立CAR referenceで照合する。V、V.T、V†を見かけで交換しない。

### 7.2 2案の比較と提案する最小差分

| 案 | 再利用と追加 | 科学的意味・注意 |
|---|---|---|
| **一般Kを1 frameで保持**（接口案として推奨、未採択） | frame/U/K/group/provenanceの薄い型、K→既存primitive、general deterministic spec、既存tail component型への係数入口を追加。Gaussian/emitter/RTE/wrapper/compilerは再利用 | 元outer frame数を保持し、同frame内の係数を合成してからtailを作れる。現行square preparationを無変更で通るわけではない |
| 符号付きsquare化 | `K=Σ_a κ_a v_av_aᵀ`、`G_a=V diag(v_a)V†`で既存signed-square DFHamiltonian/kernelへ写す。これも将来の数値処理・adapterが必要 | auxiliary square数はΣrank(K_l)で元Rと別。同frame groupを保持しないsplit/rankingや別atomの絶対値和はPF/tail lawを変え得る。変換しただけで公平なC-DFになるとしない |

最小record案は `frame_id,group_id,U,K,representation,source_hash`。Hamiltonian側には独立したκ/constant、partition側にはgroup保持・rank/weight policy、tail側にはbasis_id/support/coefficient/identity policyを持たせる。現行DFHamiltonianの意味を変えず、新一般型から既存下流型へ写す限定adapterを設ける。名前・配置も未実装の提案であり、どちらかを今回正式採択しない。

## 8. 下流への接続点と再利用地図

固定ローカルSOURCEは起点commit。入力生成元の別branch SOURCEは来歴照合だけに使い、既存PR環境を無断で入れ替えない。

| 接続点／SOURCE | 現行の型・再利用できるもの | 最小追加metadata/adapterと独立確認 |
|---|---|---|
| 元h/g → ffsim | `MolecularHamiltonian`→`DoubleFactorizedHamiltonian`。P-H4の元MO payloadが対応 | read-only import/export、spatial chemist・1/2なし・constant、元hash、補正政策。保存df_one_bodyを元hと誤用しない |
| [`df_hamiltonian.py`](../../src/trotterlib/df_hamiltonian.py) | `DFHamiltonian(constant,one_body,lambdas,g_matrices,metadata)`、`PhysicalSector`、NumPy/Numba matrix-free G² | 一般Kの型/作用入口がない。signed-squareなら既存kernelを使えるがgroupと元Rを保持。general frame referenceは別に必要 |
| [`df_trotter/`](../../src/trotterlib/df_trotter/) | DFModel/DFBlock、one-body Gaussian、JW Givens、`DiagonalEvolutionPrimitives`→emitter | square eta/lamと一般Kを区別。κのspin lift、U向きとnormal-order補正を照合 |
| [`df_partial_randomized_pf.py`](../../src/trotterlib/df_partial_randomized_pf.py) | `rank_df_fragments`→`DFFragmentPartition`、明示prefix/hash | 現重み `absλ||G||_F²`等を一般Jへそのまま使えない。frame/group選択と新weightの意味を研究判断で固定。ranking proxy≠実λ_R |
| [`df_partial_s2.py`](../../src/trotterlib/df_partial_s2.py) | `prepare_df_partial_s2`→`DFPartialS2Preparation`、one-body/fragment spec、ordered S2、scalar control | fragment specのeta/lamをgeneral diagonal specへ限定拡張。既存preparation生成入口もsquare前提。新H/partition/preparation hashを作る必要 |
| [`df_rte_tail.py`](../../src/trotterlib/df_rte_tail.py) | `DFBasisRegistry`、`DFDiagonalPauliComponent`、`DFTailExtraction`、identity/threshold/normalization記録 | KからI/Z/ZZを作る係数入口を追加し、後半の成分処理を再利用。component_id/frame/support/basis hash、同frame集約、drop normを記録。旧square extractorへJを押し込まない |
| [`df_rte_circuit.py`](../../src/trotterlib/df_rte_circuit.py)／[`df_rte_qiskit.py`](../../src/trotterlib/df_rte_qiskit.py) | `DFRTEEventPreparation`／component spec→`QiskitDFRTEEventCircuitBuilder`、順序付きevent列、basis共有・中心control・identity phase | general係数を既存FRAME_Z成分へ適合させれば回路部を再利用。新λ_R/新finite targetの正しさは別gate |
| [`df_partial_s2_repeated.py`](../../src/trotterlib/df_partial_s2_repeated.py) | ordered repeated builder、同basis D0 half境界融合、phase集約、attenuation metadata | general deterministic specのdispatchだけ限定拡張する案。元frame/groupと実隣接性を保持し、任意frameの融合を仮定しない |
| [`pr_dictionary_validation/`](../../src/trottertracks/pr_dictionary_validation/) | CAR/exterior reference、control0/1、absolute operator、actual adjoint、Hadamard X/Y、compiler検証、C4 B2 | 分子8-modeに適合する独立targetとregister/state schemaを先に確認。C4の4-mode reference/PASS/grantを流用せず、検証関数単位で使う |

全経路は、**保存元integrals → 通常C-DF → general frame → Gaussian/中央回転 → group単位D/R → partial-S2/RTE → controlled wrapper → native compiler**。Gaussian synthesisやcompilerを新設する提案ではない。主な不足は取込/係数/spec/tail入口である。

関連tests `test_df_hamiltonian.py`、`test_df_partial_s2.py`、`test_df_partial_s2_repeated.py`、`test_df_rte_tail.py`、`test_df_rte_qiskit_builder.py`、C4管理testsを静的再利用候補とした。既存のrank1・identity・phase・順序・boundary・adjoint・X/Y coverageは活用できるが、新一般Kのcoverageは未作成。`scripts/run_df_ground_state.py`、`run_rte_connected_cluster_cost_validation.py`、`run_research_c_rustiq_limited.py`等は設計参照のみで、旧契約付きCLIをC1として起動しない。

## 9. 強いX-DF baselineと公平な比較

同じP-H4を採用しても、保存10-square X-DF、ffsimの非圧縮Cholesky、ffsimのeigh X-DFは**別の分解/順序**になり得る。比較は次の選択を明示してから行う。

ffsim Trotter SOURCEには `current_basis`を持ち越し、相対回転と終端復元を使う既知実装がある。現行PRの同basis共有・境界融合とは対応する範囲を区別し、外部Trotter gateをPRの代替実行器としてそのまま採択しない。より一般のrelative-basis最適化を導入する場合も、両分解へ同じ機会を与え、追加差分を記録する。

1. 保存X-DFを既存PR baselineとして使うか、同じ元h/gから共通のX-DFを作り直すか。保存値を使う場合は入力・補正・誤差policyを照合し、新しい科学的検証の範囲を固定する。
2. 純粋な非線形fitの寄与を見たい場合、同じCholesky初期分解を非最適化対照にする案もある。これを「既存eigh X-DFと同じ」とは呼ばない。初回から対照を無断で追加しない。
3. 両方式に同じJW Gaussian、relative-basis機会、同frame boundary融合、合法なB2、native compiler/profile/seed、full measured X/Y wrapperを与える。C4 B2の適用はtyped generator/run/frame/phaseを保持できる場合に限り、general Jへ自動適用済みとはしない。
4. Rだけでなく中央Z/ZZ support、回転往復、tail λ_R、1-shot RZ/SX/X/CX/total depth/CX depth等、fit/生成/compile費を記録する。削除閾値・回転順・basis reuse・splitを片側だけ変えて圧縮効果と呼ばない。

旧DF screeningはinvalidated、旧UWCはraw欠損という前回の区分を維持する。旧順位・C・C_use・L_Dやprose改善率を採用せず、保存結果の再集計をしていない。C4/Gaussianの限定結果も分子C1の優位性の証拠にしない。

## 10. 初回検証案：targetを分けた3段階（全て未承認・未実行）

**Stage 1：表現とoperator。** まず元MO h/g/cから独立CAR referenceのH_physを定義する。candidate J/U/κ/cから作るH_CDFは、取込adapterとは独立の式で照合する。一般Kのrank1退化・diagonal/offdiagonal・負固有値・spin permutation・one-body/scalarを人工入力で確認し、各unique frameの中心operatorとGaussian向きを検証する。H_phys−H_CDFの近似誤差は独立したDF誤差gateであり、近似候補をH_physにmachine-epsilon一致させるgateとはしない。

**Stage 2：接続とnative費用。** 同じ候補H_CDFの正確なframe発展／ordered finite PF productと、raw circuitをabsoluteに照合する。controlled control0=I/control1=対象U、完成回路のactual adjoint、実Hadamard X/Y、raw/compiled両方、scalarの相対位相を確認する。コンパイル回路の正しさtargetと、元物理Hへの近似誤差を混ぜない。両baselineに同じ機会を与え、1-shot価格を記録する。

**Stage 3：有限algorithmと総資源。** 新しいD/R splitとtail atomsのfinite meanを独立referenceへ照合し、研究BのPF bias、研究Aのlaw/shot/confidenceと共同で同精度比較を設計する。同じ物理Hでも別DF・別atom列のfinite targetは自動一致しない。DF近似、PF、finite-RTE/threshold、compiler精度、統計誤差を分離する。最終総資源評価は未実施。

Hamiltonianの数保存sectorでの一致だけでは、full controlled位相同値を保証しない。Stage1のsector誤差とStage2の候補演算子full-space検証を分ける。通常C-DFにSCDF shiftは導入しないが、将来SCDFを使う場合のsector scalar復元・atomごとのsector保存問題は未解決として残す。

正確なfit/reference/build/transpile/call数は、入力採択・実R・frame/atom数・reference方法が決まってから設計する。今は架空のcall数や実行可能なgrantを作らない。初回は1入力・1候補条件を中心とし、大きなrank/seed sweepやend-to-end全gridを自動追加しない。

## 11. 条件の決定状態とblocker

| 条件 | 現在の区分 | 次の担当・必要判断 |
|---|---|---|
| 起点commit、ffsim固定版、今回read-only/1文書/STOP | **今回固定済み** | Codexはその範囲だけ実施 |
| P-H4/P-H2の保存identity・生成geometry/sector/rank来歴 | **技術的に同定済み** | 将来科学条件への採用はユーザー・研究C・統括 |
| P-H4とP-H2のどちらを初回に使うか、full/sector評価 | **研究C・統括判断** | 元物理Hと主張scopeを選ぶ。旧sectorを自動採択しない |
| number表現、一般K保持かgroup付きsquare、export形式 | **技術的に選択可能・提案のみ** | ユーザー・研究Cが最小実装案を採択後にCodexへ依頼 |
| X-DF保存再利用／fresh分解／共通Cholesky対照 | **研究C・統括判断** | 科学的寄与とinput/one-body/誤差policyを揃える |
| C-DF max_vecs/tol、optimizer停止条件、実R、J support、DF誤差norm/許容 | **未決定の科学条件** | 研究C・統括。tolや旧rank12を正式値にしない |
| κ固定／normal-ordered h固定、scalarの配置、groupとranking | **科学条件と接口の共同判断** | 研究C・統括。既存補正との二重適用を避ける |
| L_D、delta窓、r/K、PF target/誤差係数・予算 | **研究Bとの共同判断** | 新表現のPF評価を設計。旧C_use移送なし |
| tail atom/norm、identity policy、law/shot/confidence | **研究Aとの共同判断** | 新分布・finite mean・統計負担を定義 |
| 独立reference、absolute tol/precision、native指標/profile | **技術案と研究判断が必要** | Codexが選定scopeに十分な案を準備し、科学契約で固定。C4 tolを無条件継承しない |
| STOP/retry/結果保全・実行許可 | **今回STOPは固定、将来契約は未作成** | 技術gate失敗時の証拠保存と停止を設計。再試行・条件変更・科学実行は別承認 |

**実装採択前に解消すべき本質的blocker**は、初回input/sectorの採択、2案のfragment semanticsとone-body政策、X-DF対照とDF誤差基準である。元物理入力そのものの不在はP-H4について解消した。ffsim APIは接続可能な構造を持ち、現時点で `C1_INTERFACE_BLOCKED` とする理由はない。ただしgeneral-K/spec/tail adapterは未実装であり、そのまま実行可能という意味ではない。

## 12. 次の作業・承認・保全

1. **ユーザー・研究C・統括：** P-H4を初回候補として採用するか、sector、general-K案、one-body政策、X-DF対照、DF誤差方針を決定する。研究A/Bとの分担を上表で合意する。これは今回まだ採択していない。
2. **Codex（次の別依頼で提案）：** 採択済み範囲の独立fit環境準備、元packet reader、最小general-K/spec/tail adapter、人工unit/mockと独立reference接続、未承認科学契約を準備する。元Gaussian/emitter/wrapper/compilerを再利用する。今回この実装承認は受けていない。
3. **ユーザー：** 準備成果を受けて、必要なfit・operator/reference・build/compile等の限定科学実行を別に明示承認する。環境構築やmockの承認とscience grantを分ける。RC-DF/SCDFへは自動拡張しない。

RC-DFは忠実なregularization objective/fitterの確保、SCDFはoptimizer/input・shift/sector/controlled位相の未解決事項を[前回調査][C1]から引き継ぐ。通常C-DFを改造して代替fitterを作らない。

将来の重いfit/operator/compileはSOURCEとtimingから支配的処理を特定し、利用可能なCPU/GPU/RAMに応じてJAX・matrix-free・Numba・cache・独立case並列を適用する。process/threadと共有immutable入力を協調させ、科学条件・精度・seed・provenanceを変えない。恣意的なCPU/RAM/worker/phase・total wall上限は置かず、科学call・候補・出力/log・STOP境界は保持する。実測のないspeedup倍率は主張しない。

今回の技術確認はNPZ schema/shape/dtype、元入力byte hashとreceipt/GitHub blob、生成SOURCE hashとversion、ffsim固定式/API/依存、既存SOURCE/testsの再利用照合まで。**新科学call・数値再解析・積分生成・DF再構成/fit・build/compile・Operator/Statevector・新adapter/optimizer・環境構築・export・grant発行はいずれも0**。

原workspace、他worktree、既存venv/cache、他job、C4/Gaussian一時終了状態、旧SOURCE・raw・結果・grant・ledger・marker・STOPを変更しない。本書だけを専用branchへcommit/pushし、remote SHA・本文blob・変更ファイルを照合した時点で**mandatory STOP**する。新実装や科学実行には進まない。

[C1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ec27625839564dc22db2728923120f497b7cadee/docs/research/research_c_compressed_df_interface_survey_20261024.md
[C4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ec27625839564dc22db2728923120f497b7cadee/docs/research/research_c_c4_closeout_handoff_20261024.md
[INITIAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0924b1fdcc3a28d9a65062b4feaf9d0db944328/docs/research/research_c_hamiltonian_circuit_survey_20261024.md
[H4DIR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/86b8b002704aced8ca239414080f53793de81658/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data
[H4RECEIPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86b8b002704aced8ca239414080f53793de81658/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/receipt.json
[H4CONTRACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86b8b002704aced8ca239414080f53793de81658/artifacts/non_df_h4_pilot/2026-10-23/run01/contract.json
[H4SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86b8b002704aced8ca239414080f53793de81658/artifacts/non_df_h4_pilot/2026-10-23/run01/SOURCE.json
[H4GEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/h4_molecular_inputs.py
[H2DIR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/86b8b002704aced8ca239414080f53793de81658/artifacts/non_df_h2_pilot/2026-10-22/run01/inputs
[H2GEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b5ed2c6e2399b55a0c24eb8437f842892a6703ef/src/trotterlib/h2_non_df_pilot.py
[FFH]: https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/hamiltonians/double_factorized_hamiltonian.py
[FFFIT]: https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/linalg/double_factorized_decomposition.py
[FFMOL]: https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/hamiltonians/molecular_hamiltonian.py
[FFTROT]: https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/qiskit/gates/double_factorized_trotter.py
[FFPROJECT]: https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/pyproject.toml
