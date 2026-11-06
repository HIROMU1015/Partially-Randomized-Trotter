# SERVER 1：旧Track A・B2元Hamiltonian入力の回収調査

2026-11-06 JST。既存保存物のread-only調査。**既知SHAの旧S0 HDF5はNOT_FOUND。MO・積分を含む同名候補はFOUNDだが、旧DF12との対応はIDENTITY_UNVERIFIED。旧DF12へ結合できる元H ground stateもNOT_FOUNDであり、科学実行のSTOPを維持する。** NOT_FOUNDは下記探索範囲内の結果で、原本の不存在を意味しない。

固定参照：[前回REPORT（2796d26270cf39e46b40accc4f898fe29d459819）][PRIOR]、DF12 snapshot commit `bf5d3b2405b2bec4f88dd0589b75dd737b13e052`、生成SOURCE `c644925b50587072784846df09bf02c39e8453e1`。対象はlinear H4、1 Å、STO-3G、charge 0、singlet、旧DF rank 12、採択B2のgeneration `L_D=3`。本調査ではdelta窓を使用せず、科学的数値評価を行っていない。

## 1. 探索範囲・方法

- SERVER 1の自分のホーム内の研究用保存先：主repositoryと登録worktree、旧DF研究・高次PF研究repository、研究用data/archives、研究台帳、既存backup。自分が所有する通常ファイルを対象とし、directory symlinkは追跡しなかった。
- OpenFermion保存先は、共通Python環境およびWP11環境のpackage設定・保存先を確認した。以下の`OF_DATA`は共通環境の`openfermion/testing/data`を指す。私用の絶対パス・host名は公開しない。
- ホーム側の候補台帳はNPZ 2,316件を含む。package fixtureを除く2,288件でZIP member／NPY header／文字列metadataを確認した。別研究のpytest一時領域にある非ZIPの`.npz` 3件は読取失敗として記録した。積分・MO・stateの数値配列はdecodeしていない。
- 7個の既存Git object storeで到達ref・reflogを確認し、tar/zip 32件のmember一覧とGit bundle 230件のrefを確認した。通常の履歴にないheadを含む13組のbundleは非公開の一時bare repositoryで調査した。incremental bundleの不足親は既存object storeを読取参照し、全13組で履歴列挙を完了した。その他の利用可能なbundle headは初回の到達object集合に含まれていた。調べたGit/archive member名にHDF5/H5原本はなかった。
- 既知の自分所有の一時研究root 5件も確認し、NPZ/HDF5/H5候補なし。共有一時領域全体・他ユーザーのホームは探索していない。認証領域、agent履歴、一般cache、外部backup、削除済みblock、全unreachable/unlabelled blobの総当たりは対象外。

ファイルhash、size、mtime、探索エラー、保存場所の対応表、header、manifest参照は非公開ローカル記録に保持した。元ファイルを移動・上書きせず、既存SOURCE・raw・result・grant・marker・STOPを変更していない。HDF5 2件は調査前後のSHA256・size・mtime_ns一致を確認した。

## 2. 主要な発見ファイル

`S0_COPY`は既存PR-boundary worktree内のS0保存物、`H246`は2026-10-22の再生成系列の保存先を表す。mtimeは各例示コピーのUTC値であり、生成日時や来歴の証明ではない。Git blobにはfilesystem mtimeを付与しない。

| ID・保存場所／ファイル | bytes | mtime UTC | SHA256 |
|---|---:|---|---|
| H1：`OF_DATA/H4_sto-3g_singlet_df_d100.hdf5` | 27016 | 2026-07-21 17:45:02 | `ba225d17b51d4076a4a8d3a4769f2991a2b969ada1d083664ca54f9343803d6a` |
| H2：`OF_DATA/H4_sto-3g_singlet_distance_100_charge_0.hdf5` | 27016 | 2026-09-08 05:05:38 | `64a4e509bd8f3aab6e57e74c889929e8b877790139c6f44043b20ca100be397b` |
| D12：`S0_COPY/h4_1p00_rank12_development_v1.npz` | 8652 | 2026-10-27 03:15:58 | `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a` |
| I22：`H246/generation/H4/run01/integrals.npz` | 70434 | 2026-10-21 16:49:01 | `346e2be530bd665678428203c3ec73363d10de24b130e3df8dead76de23ff908` |
| D22：同generationの`tol_only_snapshot.npz` | 18126 | 2026-10-21 16:49:02 | `ed9b5a136e54a47047b6458ebb7ccb6c9b9ef699cdd534ae1f9f18da5310a520` |
| R22：`H246/forward/H4/run01/reference.npz` | 5732 | 2026-10-21 16:53:36 | `8c7fd3c6a48fefc1fe370defcdaaefe902637dd225625df693190c5ecaf769a7` |
| CC：既存connected-cluster H4の`.hamiltonian.npz` | 3589 | 2026-08-28 08:35:39 | `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b` |
| M1：[別系列の`input_generation/data/snapshot.npz`][M1]（固定Git・backup blob） | 2157448 | 対象外：Git blob | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |

最優先の既知SHA **`a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958`** と一致するHDF5は見つからなかった。[旧V0監査][V0]は2026-09-28旧S0実行時の同名HDF5を記録するが、H1はそのfile bytesと異なる。同名・同geometry・近い保存energyやmtimeから同一入力と認定しない。file hash不一致だけでoperatorが異なると認定することもしない。

## 3. HDF5に保存されていたもの

H1/H2とも、保存geometryはz座標`[-1.5,-0.5,0.5,1.5]` Åの直線H4、STO-3G、charge 0、multiplicity 1。`n_electrons=4`、`n_orbitals=4`、`n_qubits=8`。次はdataset名・shape・dtypeと既存package SOURCEの静的読取による分類で、配列の数値的一致は検査していない。

| 保存項目 | H1/H2のheader／metadata | 判定 |
|---|---|---|
| MO係数 | `canonical_orbitals (4,4) float64` | FOUND |
| MO積分h/g | `one_body_integrals (4,4)`、`two_body_integrals (4,4,4,4)`、ともにfloat64 | FOUND。OpenFermion-PySCFのMO積分名・規約。旧S0のh/gとの対応は未確認 |
| AO関連 | `overlap_integrals (4,4) float64` | AO overlap FOUND。AO h_core／AO ERIの独立datasetはNOT_FOUND |
| 核scalar | `nuclear_repulsion () float64`、保存値`2.29310124732` | FOUND（保存値の引用のみ） |
| SCF関連 | `hf_energy () float64`、`orbital_energies (4,) float64` | FOUND。SCF object／checkpoint・実行SOURCE識別子はない |
| FCI関連 | `fci_energy`、`fci_one_rdm`、`fci_two_rdm`はすべて`() bool=False` | 未計算placeholder。FCI結果として扱わない |
| ground state／CI vector | 対応datasetなし。`general_calculations_keys/values`もbool=False | NOT_FOUND |
| provenance | root/dataset attrsは空。元h/g・MO hash、DF12 hash、生成SOURCE、stateのbasis/bit-order記録なし | 旧DF12へのbindingはIDENTITY_UNVERIFIED |

MolecularDataの保存SOURCEは未設定値をFalseとして保存する。HDF5に`fci_*`という名前があることはstate保存の証拠にならない。積分規約もheaderだけから推定せず、既存package SOURCEを読んで分類した。当時のdependency全bytes一致は認定していない。

## 4. DF12・別系列stateとの結合判定

| 保存物 | 保存内容・関連SOURCE／manifest | 旧B2への判定 |
|---|---|---|
| D12 | 固定snapshotとfile SHA一致。`one_body (8,8)`、`lambdas (12,)`、`g_matrices (12,8,8)`、full state `(256,)`、sector state `(36,)`、text metadata | **DF12本体FOUND**。bare h/g・MO・元H stateはmemberにない。保存stateはDF groundであり転用不可 |
| I22/D22 | I22はspatial h/g・spin h/g・MO係数を保存。`integrals_receipt.json`と`tol_only_receipt.json`あり。[再生成SOURCE][MIG]から旧recipeを呼び、SCFを新規実行した系列。D22のλは10個で、stateはそのDF10のsector最低固有vector | 入力・stateともFOUNDだが、旧S0元入力とのbindingなし。旧DF12の元H stateとして不採用 |
| R22 | `reference (70,) complex128`、`qstate (256,) complex128`。[SOURCE][FWD]はD22のDF stateを読み、`reference`へDF時間発展を適用して保存する | receiptの`ORIGINAL_FULL_H_DF_OCCUPATION_REFERENCE_CONSTANT_ONCE`という語は元分子H groundを意味しない。元H groundとして不採用 |
| M1 | 元h/g、MO、`state (256,)`、別の`DF_state (256,)`、`fci_vector (6,6)`を保存。生成SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12`の[`build_h4_inputs`][M1GEN]は`canonical_ground(full,indices)`とDF側のsolveを分ける | **別系列の元H ground state FOUND**。新SCF・tol-only DF10の元入力であり、旧B2への対応はIDENTITY_UNVERIFIED |
| CC／その他 | CCは別DF Hamiltonian hashの一体補正後snapshotでstateなし。6入力freezeとそのbackup tarには0.70/0.80/0.90/1.10/1.40/1.60 Åのみ。GitのFCI付きtechnical snapshotは固定test SOURCEでSCF/FCI/DFをmockしたfixture | 1 Åの旧S0元入力・stateの供給根拠にはならない |

M1のtargetは保存元積分のunthresholded spin liftによる元H。Nα=Nβ=2、sector 36、spatial MOの`2p=α,2p+1=β`。SOURCEはOpenFermionからQiskitへのbit reversal後のindicesを使い、そのQiskit full-vector順でstateを保存する。これはM1自身の来歴であり、旧SOURCE `c644925…`とのMO・軌道順序・h/g対応を示さない。M1 stateやDF stateを旧B2へ代入しない。

ローカルmanifestのSHA256：I22 `integrals_receipt.json`は`dab4a34900a5b8ded94b0239b5e1f2d04cb8174f86e9b6463ea9928710a60dca`、D22 `tol_only_receipt.json`は`623ac3c34fbb22ab364f8c637377b70cdd475abb73d2969f3a9acd86293c817b`、R22 `receipt.json`は`7fa702116769244bf6ffa463e151e2fc4ddac95c3b9bb476bf89c962c0499939`。原本・私用パスは公開しない。

旧生成SOURCEの2ファイルはD12 metadataのsource_hashesとbyte hashが一致した：`pr2_s0_s1_validation.py`は`1ccf6505d2933699504db187e3849251d1fef7bc58d8dca0b5f19871896ca6fe`、`df_hamiltonian.py`は`7d711e1c3c093079ae890ffc1907813d55d211a4fb622e4f36edee05fdea0c52`。その経路はSCF→InteractionOperator→DF12→DF ground solveであり、snapshot metadataに生成元HDF5のSHAやbare h/g・MO hashは保存されていない。**SOURCE一致とDF12 file一致は確認できても、候補HDF5との完全な数値対応は未確認である。**

## 5. 不足証拠・次の承認・終了

1. 第一に、既知HDF5 SHAの原本、または同じ旧S0実行に結合されたbare h/g・MOとその生成receiptが必要。SHA一致は同じファイルの証拠であり、それだけでDF12との完全数値対応を認定しない。保存時receiptにより、元積分・MO・scalar、旧SOURCE・dependency、DF12 snapshotを結ぶ証拠が必要である。
2. 保存証拠だけで不足が残る場合は、比較対象と許容誤差を固定した**別承認の数値対応検証**が必要。H_physとOpenFermion cutoff後H_srcを区別し、MO順序・ERI規約・spin/JW順序・DF補正/Hermitian化・scalarの扱いを明示する。本調査ではoperator再構成・数値比較を行っていない。
3. **元H state solveは、同定された元h/g・MOが回収でき、そのHへ結合された保存ground stateが得られない場合に必要。現在の未同定候補をsolveする段階にはない。** 将来の明示承認では対象H、sector、軌道順序、bit-order、生成SOURCE、保存provenanceを固定する。新SCF・積分生成・DF再生成を代替手段として自動実施しない。

今回の実行はファイル列挙・hash・header/metadata・既存manifest/SOURCEの読取とREPORT作成だけである。新SCF、積分/DF生成・fit、state solve、Hamiltonian/operator数値再構成、PF/RTE、sampling、compile、shot/費用算出、追加科学検証は0件。研究上の採択・旧STOP・検証statusは変更しない。

公開対象はこのREPORT一つ。専用branchは`codex/track-a-b2-server1-input-recovery-20261106`、親は固定調査commit `2796d26270cf39e46b40accc4f898fe29d459819`。公開完了時にremote SHA・親・単一ファイル差分・本文bytesの一致を確認し、**mandatory STOP。新規生成・state solve・追加科学検証へ進まない。**

[PRIOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2796d26270cf39e46b40accc4f898fe29d459819/docs/research/track_a_b2_original_state_spec_20261106/REPORT.md
[V0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2796d26270cf39e46b40accc4f898fe29d459819/docs/research/pr2_v0_input_recovery_audit_bf5d3b2_20260928.md
[MIG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/88cd0907e00151268e3f9b41a350af3efa2f1bb8/src/trotterlib/h246_migration.py
[FWD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb4a241116ba1a5fcc88e2c9e7c91911a7da63d1/src/trotterlib/h246_reconstruction_forward.py
[M1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz
[M1GEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/h4_molecular_inputs.py
