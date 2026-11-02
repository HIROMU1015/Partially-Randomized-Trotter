# 研究C・C1：圧縮DFとnative PR回路の接口・再利用調査

2026-10-24 JST。状態：**C1_INTERFACE_SURVEY_COMPLETE_METHOD_NOT_ADOPTED_NO_SCIENCE**。

本書は、[研究C初期調査][INITIAL]のB1・C・D1・Eと既存コードを再利用したread-only調査である。[C4の一時終了](research_c_c4_closeout_handoff_20261024.md)はユーザー採択済みだが、C1の具体的方式・実装・科学実行は未採択。旧Track Bの「C1」pilot／「C4」importance samplingとは別の候補番号である。

結論は、**通常C-DFをffsimから接続する最小接口確認を第一候補、忠実な目的関数の実装を確保したRC-DFを条件付き第二候補**とすること。SCDFは配布データの取得だけでは新しい分子のfitへ進めず、sector shiftの扱いも追加で必要なため、初回候補には推さない。これは性能順位でも研究機構の正式採択でもない。

## 1. 目的と調査範囲

目的は、同じ元の物理Hamiltonianに対するDF近似の自由度を使い、Gaussian往復・中央Z/ZZ・partial-S2／RTEを含む実native費用を下げられるか、具体的に検証できる接口を選ぶこと。outer rankが小さくても、各対角行列の密度、basisの列、D/R split、tail norm、必要shot、古典fit費用が増えれば総費用は減らない。

ローカル静的調査の固定treeは `c3511345ab24bb19a77fb9f17a8675e4e805a8a3`。Pythonモジュールをimportせず、SOURCE・tests・scripts・既存報告・manifestを読んだ。外部SOURCEも実行・installしない。保存NPZはZIP member名のみ、著者notebookはJSON内のSOURCEのみを確認し、配列読込み・pickle復元・数値再解析をしなかった。公開archiveのbyte/hash照合は資料同定であり科学計算ではない。

## 2. 表現、rankと改善機構

実数・8-fold対称なspatial chemist積分を `g_pqrs`、空間軌道数をNとする。比較の共通形を

`g_pqrs ≈ Σ_lij U^l_pi U^l_qi J^l_ij U^l_rj U^l_sj`

と置く。積分の添字規約とHamiltonianの1/2は別に固定する。ここでouter rank Rは回転frame数であり、`rank(J_l)`、有効軌道数Ξ_l、実装で分割したsquare数、乱択atom数とは異なる。

| 方式 | Jとrankの意味 | 近似・古典fit | native接続上の特徴 |
|---|---|---|---|
| 既存X-DF | `J_l=λ_l η_lη_lᵀ`。1 frameにつき符号付きsquare、Jのrank≤1。inner truncationはηの有効成分削減 | 積分行列の固有分解／Choleskyと各one-body因子の対角化。outer/inner切捨て誤差を分ける | 現行square型へ直接対応。Ξが大きければ中央ZZは密で、rank1だから回転が1個になるわけではない |
| 通常C-DF（追加対照） | Uと一般実対称Jを最適化。Rを減らしてもJはfull rankになり得る | 積分tensor残差を最小化する非線形fit。初期化・収束と前処理費が必要 | 同一frame内の一般密度積を一括実装できる。既存square型への無変換格納は不可 |
| RC-DF | 同じ一般実対称J。正定値・rank1の保証はない | tensor残差に係数のelementwise正則化を追加。R、正則化係数・次数、残差、収束を独立記録 | 小さな係数normは動機になるが、PR tail normやcompiled費用を直接最適化したものではない |
| SCDF | fit前の `J_l=w_lw_lᵀ` は半正定値rank≤1。shift後 `J_l−α_l11ᵀ` はrank≤2 | 未shift tensor残差とshift後係数のL1 penaltyを共同最適化。反復fit・閾値管理が必要 | frame削減と追加signed-square数を別計上。shiftを捨てるとcontrolled位相が変わる |

RC-DFの論文目的は概ね `1/2||g−g_fit||_F²+Σ_lij ρ_lij|J_lij|^γ`（γ=1または2）。SCDFはrank1制約と `Σ_lij|w_li w_lj−α_l|` を組み合わせる。いずれも論文のqubitization／measurement用λをPRのλ_Rやnative gate価格と同一視しない。[RC-DF論文][RCDF]、[SCDF論文§2・付録B][SCDF]。

各frameの回転は数保存Gaussianで、D_lはそのbasisの密度積。回転間の相対basis、one-body専用frame、終端復元が費用に入る。RC-DFの一般Jの平方根を形式的に取ると複素因子になる場合がある。HermitianなPRへは一般対角blockを保つか、実固有値の符号を保持したsquareへ変換する必要がある。次節の接口案は代数的な提案であり、今回対角化・実装・検証していない。

## 3. 外部実装：取得できたものと不足

### 3.1 ffsim：通常C-DFの有力な再利用先

公式APIと、tag **v0.0.83**のcommit **`558d3517385778c733f1aa226bb61b8f30f4b0f3`**を照合した。[API][FFAPI]、[Hamiltonian SOURCE][FFH]、[fit SOURCE][FFFIT]。

- `DoubleFactorizedHamiltonian`は `one_body_tensor, diag_coulomb_mats, orbital_rotations, constant, z_representation`を保持する。数表現は `H=c+Σκ_pq a†_pσ a_qσ+1/2Σ_lijστ J_lij n^l_iσ n^l_jτ`。一般対称Jを直接保持できる。
- `from_molecular_hamiltonian`の `optimize=True` が通常の圧縮fit。`tol`はtensorのmax-absolute許容でありoperator誤差ではない。`max_vecs`を指定するとtolとは別のrank指定になる。`diag_coulomb_indices`は対角Coulomb行列の係数support制限に使える。
- 固定fit SOURCEはJAXの微分・JITとSciPy最適化、Cholesky等の初期値を使い、目的はtensor残差。RC-DFの係数正則化やSCDFのjoint symmetry目的はここにはない。別機能の振幅regularizationをHamiltonian RC-DFと取り違えない。
- one-body correctionとnumber/Z表現間変換はSOURCE内でone-body・constantまで扱う。外部の補正をさらに重ねない。`to_molecular_hamiltonian`は将来の再構成確認に利用できるが、同じ変換だけを独立referenceにしない。
- sector LinearOperatorと既知のbasis持越しTrotter回路もある。[回路SOURCE][FFTROT]の `current_basis` と相対回転・終端復元は強い既知baselineの参考になるが、PRのfinite targetやcontrolの検証を代替しない。

固定[pyproject][FFPROJECT]はPython≥3.11・**Qiskit≥2.0**を要求する。既存C4のQiskit1.3.0環境へ直接installする経路は採らない。将来使うなら、独立したfit用環境から型・単位・hash付きarrayをexportし、既存compiler環境へ読み込む案が最小。今回環境作成・install・exportはしていない。

確認したGit blob：Hamiltonian `fd87cc70f1a3b3d37349ab0bc6790b68166e9f45`、fit `2eac9e1690ddf5fb7779136aa7a94b04b02de526`、Trotter `ad91992a7f9973f3d16641e03af5f6f61b3cf536`、pyproject `eb3f1ad809393f3f58c285ba0c20889809650bb0`。通常C-DFを追加候補にする理由は、一般Jのfitと入出力が取得でき、RC-DFの正則化の寄与を分離する強い対照になるためである。

### 3.2 RC-DF著者資料とPennyLaneの候補実装

[RC-DF著者データ][RCDATA]はCC BY4.0、`rc-df-data.tar.gz`、公開サイズ665070260 byte、公開MD5 `c89fa730990c85ce93e89bb94602cdd6`。本調査では先頭73400320 byteだけを取得し、その範囲のloader notebookを静的確認した。**全archive未取得・全体hash未照合であり、全体にfitterがないとは結論しない。**

確認した `hydrogen_chains/load_hydrogen_chain_data.ipynb` は保存pickleの結果を読む例で、汎用fit入口はこのloaderにはない。SOURCE中で `Zs=data["Us"][idx]` と読む行があり、Us/Zsのschemaやtensor転置を無確認で流用できない。notebook SHA256 `e0937a4e6642bf9f2d5e1e2eac6aadf80258bbf8882302f856f1702fdd2273bd`。pickleは復元していない。新しいH4に使える著者互換fitterは、今回の確認範囲では確保できていない。

追加候補として、[PennyLane factorize API][PLAPI]と **v0.45.1**／commit **`5f61ce25df3cc28a1ac785d20e47d70761202ed2`**の[固定SOURCE][PLSOURCE]（blob `dcc6f0f4e16f7deecce54a563f6060464aeb040b`）を読んだ。`compressed=True`、`regularization="L1"/"L2"`、JAX・Optaxによるgreedy fit、cores/leavesがあるため再利用調査の価値がある。ただし、次の**目的関数契約の不一致**がある。

固定 `_compressed_cost_fn` は残差に `prefactor*jnp.linalg.norm(Zs,ord=norm_order,axis=(1,2))[0]` を加える。SOURCE上は最初のcoreのmatrix normを選び、[JAX normの定義][JAXNORM]ではord1/2はそれぞれ誘導1-norm／spectral normである。論文の全core・全entryに対するL1／二乗L2 penaltyと同じ式とは確認できない。これは静的な式の照合結果で、数値bug検証はしていない。**regularization引数が存在するだけで忠実なRC-DFと呼ばない。** 著者確認・別版の修正履歴または適合する固定実装の確保が先決で、今回新fitterや修正で穴を埋めない。別APIの `symmetry_shift` も、それだけでSCDFのjoint fitにならない。

### 3.3 SCDF著者archive

[著者配布][SCDATA]はCC BY4.0、`scdf.tar`、42657280 byte。全体を取得し、公開MD5 **`85fad564f3487c90b4fe849f42a52d97`**と一致。SHA256 **`a60f80d5e74fe5525a47ea9fba6a1e59df0324ad8538d85cd553d715412d2861`**。静的tar inventoryは33 entryで、実質的なSOURCEは `SCDF_tensor_factorization/load_SCDF.py`、残りは主に保存NPZと付随metadata。**この完全archiveには汎用optimizer SOURCEはない。** 別の公開場所の不存在までは主張しない。

loaderのSHA256は `bcb9b71c6ae2a2ce9179dfdacc4891ac598813bf274567a3f2dab7092315005d`。`ndf,Wt_jax,Gtflat_jax,alpha`を読み、上三角から反対称行列を作りexpmで回転を復元する例がある。収録はFeMoco/P450とH10以上の鎖の結果で、H4用fit入口はない。H10のNPZ member名はこの4種類のみで、元integrals・one-body・constant・geometry・sectorの完全packetではない。配列値は読んでいない。このデータを既存H4へ転用したり、論文のfitを再現済みと扱ったりしない。

### 3.4 OpenFermion：X-DFと規約変換の再利用

[公式API][OFAPI]と[v1.6.1 SOURCE][OFSOURCE]（Git blob `330c4556ddd36b8885ceafce73afe8b456a1bd0f`）を確認した。`low_rank_two_body_decomposition`はX-DFの入口で、one-body correctionを返す。4番目の戻り値 `truncation_value` は切捨て係数和のboundであり**Hamiltonian scalarではない**。`final_rank`は閾値より優先する。

SOURCEの並べ替え重みは `|λ|*(Σ|G_pq|)^2`で、プロジェクトの `lambda_frobenius_squared` や単なる固有値順とは異なる。`spin_basis`、chemistへの並べ替え、入力に含まれる1/2を固定せず、出力配列だけを交換しない。既存X-DF・規約確認には使えるが、一般RC-DF／SCDFのfitterではない。

## 4. 既存SOURCEと検証資産の再利用地図

以下は上記固定treeの静的評価。リンク先はリポジトリ内のSOURCEであり、今回そのコードを変更・再実行していない。

| SOURCE | 再利用できる処理 | そのまま使えない理由・最小接口・整合条件 |
|---|---|---|
| [`df_hamiltonian.py`](../../src/trotterlib/df_hamiltonian.py) | `DFHamiltonian`のconstant/one-body/signed square、`PhysicalSector`、sector NumPy/Numba matrix-free `G²`、入力hash | 型は `Σλ(a†Ga)²`。一般JをGへ格納すると二乗する別Hになる。group付きsigned-square変換なら既存kernelを再利用可能。一般Jを保持するなら別diagonal作用接口が必要。PySCF生成は今回は使わない |
| [`df_trotter/`](../../src/trotterlib/df_trotter/)のmodel/decompose/two_body | 積分規約、DFModel/DFBlock、spin拡張、再構成 | DFBlockのeta/lamはsquare用。decomposeの1/2とOpenFermionの補正を保持。general blockの係数型が必要で、外部constantとの二重加算を防ぐ |
| 同ops.py | `U_to_qiskit_ops_jw_givens`、Gaussian actual inverse、`append_diagonal_primitives`のphase/RZ/RZZとancilla phase | `df_squared_diagonal_primitives`の入口だけはsquare限定。一般Kから同じprimitive型へ写すadapterが最小。Uの向き、spin順、angleの2倍を独立確認 |
| [`df_partial_s2.py`](../../src/trotterlib/df_partial_s2.py) | 決定論one-body、ordered S2、D/R prefix、basis/circuit組立、scalar位相 | prepare/specは既存DFHamiltonianとsquare列を前提。一般block specかgroup付きsquareが必要。新しいfragment identity・L_Dと旧prefixが同じとは限らない |
| [`df_partial_s2_repeated.py`](../../src/trotterlib/df_partial_s2_repeated.py) | 同じbasisのD0 half境界融合、scalar集約、順序付きtrajectory、attenuation metadata | 同じframe/実演算の隣接性が条件。任意U同士は相殺しない。group境界とidentityの意味をadapterで保持する |
| [`df_rte_tail.py`](../../src/trotterlib/df_rte_tail.py) | frame registry、identity/phase policy、符号付き成分・truncation記録、係数からinvolutionへの正規化 | `exact_df_diagonal_coefficients(eta,lam)`はsquare専用。一般KのI/Z/ZZ入口が必要。共通frame内の係数を合成してからλ_Rを取るか、別atom化するかを明記 |
| [`df_rte_qiskit.py`](../../src/trotterlib/df_rte_qiskit.py) | event列の実順序、basis持越し・復元、中心だけcontrol、ancilla相対位相 | 新一般係数をregistry/component型へ接続してから再利用。旧RTE mean/referenceやλ_Rは移送不可。full builderと研究C4 fixture builderの型も別 |
| [`df_partial_randomized_pf.py`](../../src/trotterlib/df_partial_randomized_pf.py) | partition/rank metadata、入力同定、cache、明示的prefix、ground-state/evolution補助 | `absλ||G||_F²`等のranking proxyは一般Jの物理的重みでも厳密tail λ_Rでもない。新weightとgroup/split意味を決める必要。PF surrogateはrigorous boundではない |
| [`pr_dictionary_validation/`](../../src/trottertracks/pr_dictionary_validation/) | 独立CAR/reference、absolute control0/1、actual adjoint、full Hadamard X/Y、compiler/phase handling、B2 | [C4引継ぎ](research_c_c4_closeout_handoff_20261024.md)の資産。新H用の独立target・adapterが必要。旧grantやfixtureのPASSを新分解へ流用しない |

**接続の最小案：** 元integralsと規約/補正 → 外部既存fit → `frame_id,U,K,scalar,one_body,provenance` → 一般対角係数 → 既存Gaussian＋中心primitive → 新しいpartition/tail登録 → 既存partial-S2／RTE → actual controlled wrapper。欠けているのは主に外部出力の規約付き取込み、general block/spec、partition重みとtail登録であり、Gaussianやcontrolled compilerを全面的に作り直す必要はない。これは実装計画候補で、コードは追加していない。

静的に確認したtestsは `test_df_hamiltonian.py`、`test_df_partial_s2.py`、`test_df_partial_s2_repeated.py`、`test_df_rte_tail.py`、`test_df_rte_qiskit_builder.py`、`test_uwc_preprocessor.py` とC4管理tests。square係数、切捨て値とscalarの区別、絶対位相、境界融合、順序、control0=I/control1=Uの検証を再利用できる。ただし新Kのcoverageはない。`scripts/run_df_ground_state.py`、`run_rte_connected_cluster_cost_validation.py`、`run_grouped_uwc_pf_qpe.py`等は将来の下流接続の参考で、今回起動していない。新しいvalidation pathを作る段階でのみlibrary/runner/test/artifactと各索引を揃える。

## 5. 一般対角fragmentの接口案（未実装）

### 5.1 prefactorを吸収したK

spin-orbital添字i,jと**全prefactorを吸収した実対称K**を使い、`D(K)=Σ_ij K_ij n_i n_j`（対角と両順序を含む）を接口の定義とする。以下は `n=(I−Z)/2` による代数的帰結：

- scalar `c_K=(1ᵀK1+tr K)/4`
- Z係数 `z_i=−(Σ_j K_ij)/2`
- ZZ係数 `z_ij=K_ij/2`（i<j）
- `exp(−iτD)`はphase `−τc_K`、RZ角 `2τz_i`、RZZ角 `2τz_ij`。

`K=ληηᵀ`なら既存square式になる。ffsimのspatial `1/2Σ_ijστ J_ij n_iσ n_jτ`からはspin展開したKにJ/2を入れるが、block alpha/betaとプロジェクトの `2i+σ` 順序のpermutationを明示する。外部Z表現を使う場合はこのnumber→Z変換を二重適用しない。全spaceで消せるglobal phaseも、controlled evolutionではcontrol1の相対位相として保持する。

推奨接口はこのKを1 frameのまま扱い、既存 `append_diagonal_primitives` とframe registryへ渡す案。別案は `K=Σ_a κ_a v_av_aᵀ` の符号付き実square化である。後者なら既存G²型を流用できる一方、補助square数をouter fragment数と誤認せず、元frame/group IDを保持してD/R splitをgroup単位で定義する必要がある。同じframeの全係数を合成すれば残る相殺も、別atomの絶対値和では失われ得る。変換前後でtail lawやPF productが自動的に同じになるわけではない。

### 5.2 必須の取込metadata

元tensor/geometry/basis/unit/電子数/spin/constant、chemist・physicist規約、spin順、1/2とnormal-order補正、実装version/objective/seed/initialization/停止理由、R・各rank/Ξ/support閾値、Uとframe/group、fit残差のnorm、取り込む表現のnumber/Z属性を持たせる。hashは保存inputと出力の両方に付ける。これは将来のschema要件で、今回packetやadapterを生成していない。

## 6. 近似誤差、scalar、sector、finite target

| 層 | 比較で固定／検証するもの | 自動的には成立しないこと |
|---|---|---|
| 元物理H | 同じ核配置・AO/active basis・元integrals・電子数/spin・mapping・核/one-body項 | 同じ分子名や保存DFファイル名だけでは同じ元Hといえない |
| DF近似 | tensorのFrobenius/maxabs、切捨て別誤差と `||H−H_fit||` または指定sector norm | fit residualやCCSD(T) energy誤差がoperator normを認証するわけではない |
| deterministic PF | 新表現のordered fragment・L_D・step・時間窓とPF bias | 旧X-DFの係数C、C_use、最適L_Dを新表現へ移せない。C_useは実行窓の経験包絡 |
| finite RTE | 新しいtail atoms、符号・λ_R・identity・K・r、有限mean/補正の独立target | 同じ近似Hでも分割・atom化が変われば有限targetは異なる |
| circuit/compiler | Gaussian向き、scalarとcontrol相対位相、actual adjoint、raw/compiled operatorとX/Y、bit精度 | sector energy一致、uncontrolledでphase一致、compiler同一のいずれもfull controlled正しさを単独保証しない |
| 統計・総費用 | 研究Aのlaw/必要shot/失敗確率、研究BのPF bias配分、古典fit/生成費 | native1-shotや係数λだけの低下を同精度end-to-end改善へ外挿できない |

一般に共通不変sectorまたはfull spaceでのoperator差εが確定すれば、unitary evolutionの差はDuhamelにより `≤|t|ε` と評価できる。tensor誤差からこのεへの変換には軌道数・規約・対象spaceを明記する。状態依存energy誤差とoperator誤差は別である。将来の独立referenceは元のsecond-quantized CARから作り、candidate adapterと同じ変換だけで自己照合しない。

SCDFについて、shift後を `H_s=H_fit−a1 N̂−(A/2)N̂²`、`A=Σ_l α_l`と定義すると、粒子数ν sectorでは `H_fit=H_s+s_ν I`、`s_ν=a1ν+Aν²/2`。元H_fitの発展へ戻すには `exp(−it s_ν)` を掛け、controlledではancilla相対phaseに入れる。これは本書のsector接口の代数で、full-space同一性ではない。one-body補正・全体scalarと併せて1回だけ適用する。

また、最終Hが数保存でも各finite atomが同じsectorを保存するとは限らない。Gaussian＋FRAME_Zの各Z/ZZは数保存だが、physical X/Yを含むPauli分解では個別atomがsector外へ出得る。sector上のshift同値だけでPF／RTE productの同値を主張せず、個々の因子と実finite mean・full X/Yを別に確認する。研究Cだけで新しいlaw・shot数・PF誤差予算を採択しない。

## 7. 強いbaselineと既存証拠の区分

比較は同じ元inputから作るX-DF、通常C-DF、候補正則化方式へ、同じGaussian synthesis、relative basis持越し、同frame境界融合、合法なB2角度結合、同じnative compiler/profile/seedとfull measured wrapperを与える。B2は既知baselineでありC1固有の新規性に数えない。回転順の変更、tensor cut、中央係数cut、sampling law変更を同時に入れる場合は寄与を分ける。

1. **表現比較：** 同じ誤差基準下のR・inner support・J密度・fit費用と1-shot費用を調べる。physical precisionが揃っていなければ性能順位を出さない。
2. **有限algorithm比較：** D/R split・r/K・atom化を固定する比較か、各方式を同じerror予算で再選択する比較かを宣言する。旧イベント列を同じラベルで使うだけでは同じ分布にならない。
3. **end-to-end比較：** PF・DF・finite-RTE・compiler誤差と研究Aの統計負担を合わせる。これは未実施で、初回の接口確認だけでは結論しない。

既存証拠は[VALIDATION_STATUS](../../VALIDATION_STATUS.md)と[manifest](../../artifacts/validation_manifest.json)のstatusに従う。

| 既存資料・処理 | 本調査での扱い |
|---|---|
| C4 B0/B1/R1/B2とGaussian singletonの固定報告 | 限定合成fixtureでの位相/回路検証・既知baseline資産。分子C1の優位性の証拠にはしない。数値は[C4引継ぎ](research_c_c4_closeout_handoff_20261024.md)に完全scope付きで引用 |
| 旧DF screening | `invalidated_pending_regeneration`。旧ランキング・C・L_D推奨を使わず、再集計もしない |
| 旧grouped UWC | `prose_only_missing_raw_artifacts`。raw欠損のprose数値をBLISSや新DFの実証としない |
| UWC/BLISS関連SOURCE | number-sectorのN/N² shift等の実装資産。一般BLISS・SCDF fitや同精度総費用評価を完了した意味ではない |
| grouping/RTE/relative-basis/P-A系の設計とSOURCE | 順序・frame共有・既知境界融合を対照へ引き継ぐ。旧Track Bの停止・特定条件の非改善を保全し、新方式の効果へ転用しない |

保存H4 packet `artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz` とH5 snapshotのZIP member名を確認した。`schema_version,hamiltonian_hash,constant,one_body,lambdas,g_matrices,metadata_json`であり、元のpre-DF二電子積分arrayは含まれない。この固定treeで確認したpacketだけでは「同じ元物理integralsをfitする」準備が揃わない。元inputが別資料に存在しないとまでは断定しない。

将来、保存square-DFからtensorを再構成して圧縮するなら「保存近似Hの再表現pilot」であり、元物理積分へのfitとは明記して分ける。今回再構成・array decode・残差計算はしていない。元inputの来歴・hashを回収できないことは科学比較のSTOP条件で、無承認の積分再生成で補わない。

## 8. 最初の限定検証候補

| 優先 | 候補と得られる情報 | 追加作業・費用・条件 |
|---|---|---|
| 1（推奨、未採択） | **ffsim通常C-DF＋一般対角block接口**。取得できるfitterでR削減と一般J化のnative費用を分離し、強い圧縮baselineを得る。失敗しても、R削減をGaussian/ZZ/λ_R増加が相殺する機構を特定できる | isolated fit環境、外部出力取込、一般K/spec/tail adapterが必要。既存Gaussian・phase・wrapperを再利用。fit反復費を計上。固有の研究寄与は通常C-DFの再実装ではなく、PRの同精度共同評価で示す必要がある |
| 2（条件付き、未採択） | **RC-DF、まずγ=2等の1目的を固定**。通常C-DFを対照に正則化のPR tail/native効果を分離する。悪化や無改善でも係数normと実費用の乖離が分かる | 忠実な外部fitterの確保が前提。現在のPennyLane固定版を無検証で使わない。将来1正則化設定・1rank/誤差policyに限定した契約を準備し、fitは別承認。今回ρやrankを採択しない |

SCDFは次候補として保持する。著者データは取得可能だが新H用optimizer/inputが不足し、rank1前提の資産再利用だけでshift後rank・sector/finite-target問題は解決しない。これらが解決し、shiftを復元した同精度PRでの利益を具体化できれば再検討する。RC-DFとSCDFのどちらが最良かは未確定。

## 9. 次段階の最小実装・科学契約の案

以下は**提案であり、実装依頼・実行grantではない**。

1. **Codexの資料準備（次依頼が必要）：** 同じ元integral packetとconstant/one-bodyを回収・hash固定。ffsimまたは忠実なRC-DF実装を1版に絞り、objectiveとschemaを確定する。見つからない場合は不足を統括へ戻す。
2. **別途承認する実装準備：** 最小general-K取込/spec/tail adapterと人工unit/mock、独立CARとの検証設計、未承認の科学契約を作る。rank1への退化、一般対称・負固有値、diagonal項、spin permutation、scalar、control0/1、actual adjointとX/Yのcoverageを含める。新fitterの独自設計を開始しない。
3. **別明示承認する初回科学pilot：** 例えば元のlinear H4・1Å・STO-3Gの1条件を候補とし、同一inputのX-DFと候補1方式を比較する。ただし元input・X-DF rank12という旧ラベルの適合性も未確認であり、採択値ではない。sector、外側rank／inner政策、L_D/group split、delta窓、r/K、誤差・referenceを新契約で固定してから実行する。
4. **初回call scopeの設計：** 利用できる保存分解のidentityが一致すればreuseし、なければ1入力・1候補設定の外部fitと必要なX-DF分解だけを別承認する。すべてのunique frameに必要なlocal/phase/adjoint検証と、少なくとも各方式の1 ordered full controlled case・X/Y wrapperを含める。正確なbuild/reference/transpile数はRとframe/support決定後に列挙し、今は架空の固定call数やgrantを発行しない。条件変更・fit再試行・seed sweepは自動追加しない。
5. **その後の共同判断：** 初回接口がPASSしても、研究Bが新表現のPF誤差、研究Aが新tailのlaw/shot/認証条件を判断する。同精度end-to-end比較へ進むかは統括・ユーザーへ戻す。

将来の重い実行では、既存timingとSOURCEから支配的処理を特定し、CPU/GPU/RAMを確認してJAX・matrix-free・Numba・cache・独立fit/compileの並列化を使う。process数とBLAS/OpenMP/Rayon等を協調させ、必要なreference/mock確認後にrunを固定する。恣意的な低CPU/RAM/worker制限やphase/total wall capは設けない。input/output/call/log/STOP制限は維持し、未測定のspeedup倍率は主張しない。今回計算backendや共有設定を変更していない。

## 10. GO／STOPと統括へ戻す判断

**次の実装準備へ進める条件：** 入手可能な外部実装のobjectiveと入力が一致し、one-body/scalar・spin/1/2・general block/group semanticsを説明でき、強いX-DF/C-DF対照と独立referenceを用意できること。GOは性能優位や科学実行承認を意味しない。

**科学比較へ進めない条件：** 元input未同定、regularization式の不一致未解消、general Jをsquareに誤格納、sector shiftをphase抜きで使用、旧λ_R/C/STOP/grantの流用、未規定のfit再試行、近似誤差を揃えず価格だけ順位付け、独立reference不在。これらは当該計画の技術的STOPであり圧縮DF全般のNO-GOではない。

**ユーザー・研究C・統括の判断事項：** 通常C-DFから接口確認するか／忠実なRC-DF実装確保を先にするか、元物理Hとsector、許容DF誤差、groupとsplit政策、独立寄与の判定指標。研究Aとはtail law・shot／confidence、研究Bとは新PF target・bias配分を共同決定する。古典fit費と同精度総費用を含める段階も別に採択する。

今回実施したのは固定証拠引用、静的SOURCE/API/配布物確認、2件のMarkdown、最小研究metadata・索引の追記と公開照合だけ。**新科学call・保存数値の再解析・Hamiltonian生成・fit・build/compile・Operator/signal・新adapter/optimizer実装は0**。旧SOURCE・raw・grant・ledger・marker・STOP、他研究状態と原workspaceを保全する。公開・remote照合で**mandatory STOP**し、次の実装／科学実行はユーザーの別依頼・別承認へ戻す。

[INITIAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0924b1fdcc3a28d9a65062b4feaf9d0db944328/docs/research/research_c_hamiltonian_circuit_survey_20261024.md
[RCDF]: https://arxiv.org/html/2212.07957v3
[SCDF]: https://arxiv.org/html/2403.03502v1
[RCDATA]: https://zenodo.org/records/7866658
[SCDATA]: https://zenodo.org/records/10836682
[FFAPI]: https://qiskit-community.github.io/ffsim/api/stubs/ffsim.DoubleFactorizedHamiltonian.html
[FFH]: https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/hamiltonians/double_factorized_hamiltonian.py
[FFFIT]: https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/linalg/double_factorized_decomposition.py
[FFTROT]: https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/qiskit/gates/double_factorized_trotter.py
[FFPROJECT]: https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/pyproject.toml
[PLAPI]: https://docs.pennylane.ai/en/stable/code/api/pennylane.qchem.factorize.html
[PLSOURCE]: https://github.com/PennyLaneAI/pennylane/blob/5f61ce25df3cc28a1ac785d20e47d70761202ed2/pennylane/qchem/factorization.py#L491
[JAXNORM]: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.norm.html
[OFAPI]: https://quantumai.google/reference/python/openfermion/circuits/low_rank_two_body_decomposition
[OFSOURCE]: https://github.com/quantumlib/OpenFermion/blob/v1.6.1/src/openfermion/circuits/low_rank.py
