# LOCAL：Track A・B2 元Hamiltonian入力の回収調査

2026-11-06 JST / LOCAL / Codex。起点：`2796d26270cf39e46b40accc4f898fe29d459819`。

**既知原本HDF5を発見し、既知SHA256との完全一致を確認した。元h/g積分・MO係数・AO overlapは保存されている。対応する元H ground stateは今回確認した入力候補からは `NOT_FOUND`。DF12との生成来歴は整合するが、数値的なbinding・operator比較は未実行である。** SERVER 1へ接続せず、その調査結果を前提にしていない。原本は元の場所でread-only確認し、複製・移動・公開していない。

## 1. 探索対象と範囲

[旧V0監査][V0]を先に読み、9月27日旧pilot入力の回収失敗と、9月28日S0生成HDF5の観測を区別した。旧監査はOpenFermionの「既定保存先」と記すが、絶対pathは記載していない。今回、LOCALに残る環境の実際の保存先を解決した。

| 範囲 | 今回の確認 |
|---|---|
| `/home/abe` | hidden・ignoredを含むファイル名探索。研究projects、別worktree、`myproject`、OpenFermionの各環境、download・保存archiveを含む。`.git`内部と`node_modules`はこの探索から除外し、Gitは下記で別確認。directory symlinkの再帰追跡なし |
| `/media/abe/T7` | 利用者所有の既知SSD保存先。`myproject`、`internal-archive-20261105`、`research-c-integrated-20261105`、coldstore、worktrees25-copy等の展開済みファイル名、関連manifest/indexを確認 |
| 後続archiveの内容目録 | coldstore `P0044/P0045/P0114/P0119/P0120`の5本、`tmp-archive-20261105/*pr2*.tar.zst`の41本、LOCALのH4 geometry review ZIP 1本。計47 archive・18,228 member名をread-only列挙。exact対象HDF5はなし。P0119には既知DF12 NPZ名あり。payload全配列の読取り・復元なし |
| 後続の保存入力候補 | 上記home/SSDのNPZから名称がground/state/H4/snapshot/integral/molecular、またはground-state cacheの候補を選び、held-outを除外。2,563 pathのZIP/NPY headerと文字列metadataを確認、読取失敗0。member名・CRC・sizeによる1,833組への整理は探索の重複除去だけで、科学的同一性判定ではない |
| Git | 現在LOCALで到達可能な全refのobject名、exact HDF5 path履歴、固定SOURCE・保存S0 JSON・DF12 blobを確認。対象HDF5名のGit objectは見つからず、stashは空 |
| `/tmp` | 他ユーザーを横断せず、利用者所有の研究prefix directoryだけを確認。現在該当したのは今回の隔離worktree 1個、候補形式ファイル2個。coldstoreへ退避された旧`/tmp`は上記archiveとして確認 |

全backup payload、分割archive、削除済みblock、未提示cloud/別端末、他ユーザー領域、SERVER 1/2の現在のfilesystemは未調査。archiveのmember目録だけから内部stateの科学的同定はしていない。`NOT_FOUND`は上記の確認範囲に限定する。

## 2. 回収したHDF5と保存内容

原本の絶対path：

```text
/home/abe/Project/Partially Randomized Trotter/.venv311/lib/python3.11/site-packages/openfermion/testing/data/H4_sto-3g_singlet_df_d100.hdf5
```

| 識別情報 | LOCALの観測 |
|---|---|
| SHA256 | `a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958`。依頼文・旧V0監査の既知原本と完全一致 |
| size / owner | 27,016 bytes / uid 1001 (`abe`) |
| birth | `2026-09-28T06:38:21.155379417+09:00` |
| mtime / ctime | ともに `2026-09-28T06:38:21.159379401+09:00`。mtime単独を来歴の証明にしていない |
| 分子 | H4 linear、z座標 `[-1.5,-0.5,0.5,1.5]` Å、全x=y=0。STO-3G、charge 0、multiplicity 1、4 electrons、4 spatial orbitals、8 spin orbitals |
| sector | HDF5自体にはsector basisなし。固定SOURCEとDF12 metadataは `Nα=Nβ=2, Sz=0`、sector dimension 36 |
| scalar | 保存`nuclear_repulsion=2.29310124732` Ha、hex `0x1.25845777d3feap+1` |
| description / name | `df_d100` / `H4_sto-3g_singlet_df_d100`。root/dataset attributesは空。SOURCE commitやDF12 hashを記した専用sidecarはこのHDF5にない |

HDF5は`h5py.File(path,'r')`で開き、dataset名・shape・dtype・storage metadataと小さい分子metadataだけを読んだ。以下の積分・軌道配列の数値decode・比較はしていない。

| dataset | shape / dtype | 保存判定 |
|---|---|---|
| `one_body_integrals` | `(4,4)` / float64 | 元のspatial MO hあり。gzip storage 93 bytes |
| `two_body_integrals` | `(4,4,4,4)` / float64 | 元のspatial MO gあり。gzip storage 705 bytes |
| `canonical_orbitals` | `(4,4)` / float64 | MO係数あり。gzip storage 106 bytes |
| `overlap_integrals` | `(4,4)` / float64 | AO overlapあり。gzip storage 51 bytes |
| `orbital_energies` | `(4,)` / float64 | 保存あり |
| 元H state / FCI vector | 該当datasetなし | **保存なし**。`fci_energy/fci_one_rdm/fci_two_rdm`はbool scalar `False`（欠損sentinel）。CISD/CCSD関連field、`general_calculations_keys/values`もFalse。RDMもstateの代替にしない |

## 3. 元Hの規約とDF12への来歴

固定生成SOURCEは `c644925b50587072784846df09bf02c39e8453e1`。その`pr2_s0_s1_validation.py` SHA256は `1ccf6505d2933699504db187e3849251d1fef7bc58d8dca0b5f19871896ca6fe`、`df_hamiltonian.py`は `7d711e1c3c093079ae890ffc1907813d55d211a4fb622e4f36edee05fdea0c52`。LOCALのGit objectから再hashし、保存DF12 metadataのSOURCE hashと一致した。[S0 SOURCE][GEN]、[DF SOURCE][DFGEN]

SOURCEは `MolecularData(...,description='df_d100')` → `run_pyscf(run_scf=1,run_fci=0)` → `get_molecular_hamiltonian()` → rank12 DF → **DF** ground solve → NPZ保存である。LOCALのOpenFermion-PySCF保存実装も、SCFの`mo_coeff`を`canonical_orbitals`に保存し、hを `Cᵀ h_core C`、gをchemist ERIから `transpose(0,2,3,1)`、すなわち `g[p,q,r,s]=(ps|qr)`として保存する。核反発は独立scalar。spin liftは`2p=α,2p+1=β`、InteractionOperatorのtwo-body係数には1/2が入る。LOCALのOpenFermion `EQ_TOLERANCE=1e-8`未満のspin tensor成分は0化される。

従って、保存MO積分が定める `H_phys` と、cutoff後の `H_src`、DF/Hermitian化後の `H_DF` は区別する。DF NPZの`one_body`はbare hではなくone-body correction後である。DF分解第4戻り値を核scalarへ加算しない。これらはSOURCEの静的確認であり、今回operatorを構成して検証した結果ではない。

| 来歴の結合点 | 今回独立に確認した証拠 |
|---|---|
| 9月28日のS0 HDF5そのもの | 旧V0の完全file SHA256と現在のHDF5 bytesが一致。birth/mtimeも旧監査記録と整合 |
| 目的のDF12 snapshot | `bf5d3b2405b2bec4f88dd0589b75dd737b13e052`の保存NPZをhashし、SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`と一致。Git blob `39770ad86433a029d2f5157a90384f5ee37b12ce` |
| 分子・scalar・SCF記録 | HDF5のgeometry/basis/charge/multiplicityとSOURCE・DF12 metadataが整合。保存`hf_energy=-2.0985459369977173`と核scalar hexも一致。これは保存metadataの照合で、新energy評価ではない |
| Track Aへの接続 | 固定起点treeの`track_a_ax2b_h4_pilot_v5/.../input_reference.json`も同じDF12 snapshotを参照する。[AX入力記録][AX] |
| 保存stateの役割 | NPZの`state_vector(256)`、`sector_state_vector(36)`は生成SOURCE上 **DF ground**。元H stateではない。元h/g・MOはNPZにない |

**判定：既知S0原本の回収と、採択DF12への文書・SOURCE上の生成系列は支持される。** ただしS0 NPZ/manifestには元h/g/MOやHDF5の直接hash結合がなく、数値再構成を行っていないため、元H対DF12のoperator一致・model誤差、軌道配列の数値bindingをPASSとしない。9月27日旧pilot hash `d8b4aaf21afcc3935d5b5aa4d0805b358c5ec670d8104d25807c7cd0620a3dc3`の回収とは別であり、旧 `STOP_INPUT_REPRODUCTION_MISMATCH` を解除しない。

## 4. 別入力・state候補の識別

以下は発見・照合した関連候補であり、目的の元H stateへの代入候補として採用していない。repo以下のpathは上記LOCAL repositoryを基準とする。

| 保存物・LOCAL所在 | file SHA256 / 来歴・採否 |
|---|---|
| S0 DF12 `artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz`（LOCAL archive/worktreeにも複数copy） | `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。目的DF12本体。元H stateはなし |
| `.worktrees/rpe-controlled-iid-success-energy-preparation-20261105/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz` | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。M1 H4-A、DF10。h/g/MO、元H `state(256)`、`DF_state(256)`、FCI vector `(6,6)`あり。新SCF系列であり旧DF12とのbindingなし |
| `.worktrees/research-c-integrated-preparation-20261105/artifacts/research_c_pr_aware_df_kappa_diagnostic/2026-10-27/inputs/P_H4_original_snapshot.npz` | 上のM1と**file SHA256完全一致**。`original`という名前でも9月28日S0原本ではない |
| `.worktrees/rpe-controlled-iid-success-energy-preparation-20261105/artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz` | `793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4`。state `(70,)`。保存Hamiltonian hashは `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`。別RPE DF入力 |
| `/media/abe/T7/research-c-integrated-20261105/artifacts/research_c_integrated/2026-11-05/development/run01/downstream/common/physical_ground/ground_physical.npz` | `1af275a131463aecd4070a1e941d0e61ddfcd11b7e8fb1fcd2473cadcad842a7`。`energy/sector_state(36)`のみ、旧HDF5へのhash結合なし。寸法だけで同定不可 |

同じ `H4_sto-3g_singlet_distance_100_charge_0.hdf5` 名の別ファイルも確認した。下表のprefix以下は全て `lib/python3.11/site-packages/openfermion/testing/data/`。いずれも対象SHAと不一致で、同じgeometryだけでは採用しない。不一致はfile bytesの不一致であり、積分の数値差を評価したものではない。

| 環境prefix | 別HDF5のSHA256 |
|---|---|
| repoの`.venv311/` | `9c2782f1fcc8fed865a0a5c1e86875edfe39e0b7e94b1b43b0135157e5f7d189` |
| `/home/abe/myproject/Evaluation_numGate_highorder/venv/` | `fb2a52615acd741ab177ad0290301b84233fb336dc73faad27cc3edd4de7dfe8` |
| `/home/abe/myproject/Evaluation_numGate_highorder/.venv-req/` | `0395819ff0a01c49d9bbbfed23d6e4f755794153ba2fe0f956ddf2f10d07edea` |
| `/home/abe/myproject/.venv/` | `2fe48df90a3a8403b723b54d15b2d53daf0367f269f51f5670b39f8df9230c04` |
| `/home/abe/Project/Evaluation_numGate_DF/.venv/` | `93c13dbeefe3522e9a66b28702e24575526f9753da75df5dcbfb148f1ed9415a` |

また `/home/abe/myproject/.venv/.../H4_sto-3g_singlet_1.0.hdf5` はSHA256 `777438bed6bfe4040c00ea7f828415398c434a4f30de649b5e8ddea55386e044`、原子列のz順が `[0.5,-0.5,1.5,-1.5]`で異なる。これら6候補にもstate/vector名のdatasetは見つからなかった。

## 5. SERVER 1との後日照合キー

SERVER 1の結果は未受領・未評価。比較にはファイル名だけでなく、§2の完全HDF5 SHA256・size・geometry・dataset layout、§3のSOURCE commit・hashと次の固定識別子を使う。

- DF12 NPZ SHA256：`3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。
- 保存DF Hamiltonian hash：`de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。
- 保存DF state-vector hash：`c9aca811b5c023772d148d0331c82958bac6824b593a367cb65a5f89937f4f63`。
- LOCAL読取package SOURCE SHA256：`openfermion/chem/molecular_data.py` = `f6489483082534354131466f92c43eb726a39082bc514977475d36ddb7e57e38`、`openfermionpyscf/_run_pyscf.py` = `f092eed5b9792bf5ba54b78fe8614a5d261d1eba700dec33f3923f4a81e89add`。当時の実行dependency全bytes一致を保証するものではない。

後二つのDF hashは保存metadataの引用で、今回配列から再計算していない。異なるstate-hash関数間で直接同一性を判定しない。時刻・inodeは補助情報で、別環境との同一性は完全file SHAを優先する。

## 6. 解決したこと・残ること

元積分・MOの具体的回収先不明という問題は解決した。既知S0 HDF5の実体と保存fieldを確認でき、M1 DF10や別RPE snapshotを流用する必要はない。一方、同じ元Hの保存ground stateは確認範囲で `NOT_FOUND`。H_phys/H_srcのtarget定義、元H対DF12の数値binding・model差、元H stateの取得は未解決である。全環境での不存在とは結論しない。

科学数値callは0。SCF、DF生成、state solve、operator再計算・比較、PF/RTE、sampling、回路build/compile、費用評価、重要科学レビューは実行していない。元SOURCE/input/raw/grant/marker/STOPを変更せず、既存dirty checkoutも維持した。公開対象はこのREPORTだけであり、積分・MO・stateの実データは含めない。

## 7. 次に必要な範囲とmandatory STOP

1. **ユーザー／SERVER 1担当**：独立REPORTを突き合わせ、上記HDF5完全SHAとlayout、保存state候補の来歴を照合する。LOCALから自動接続・転送しない。
2. **ユーザー**：必要なら、回収済みh/g/MOだけからH_physまたはH_srcを明示して構築し、固定DF12との対応・model差を評価する限定科学範囲を承認する。新SCF・新DF生成やM1への入力変更を含めない。
3. **Codex（別途科学実行承認後のみ）**：上の限定比較を行い、必要なら**その同定済み元Hのみ**のground stateを `Nα=Nβ=2` sectorでsolveし、精度・basis/JW順序・位相規約・provenanceを保存する。既存のintegral loader、sector solver、独立occupation参照を適合確認して再利用する。`build_df_h_d_from_molecule`はSCFを呼ぶため、保存入力の読取代わりに起動しない。

このREPORTの専用branch公開・remote SHAと本文のreadback確認で **mandatory STOP**。上記は次回の提案範囲であり、今回の実行許可ではない。追加の元H生成・state solve・B2/RPE科学実行へ自動進行しない。

[V0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2796d26270cf39e46b40accc4f898fe29d459819/docs/research/pr2_v0_input_recovery_audit_bf5d3b2_20260928.md
[GEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c644925b50587072784846df09bf02c39e8453e1/src/trotterlib/pr2_s0_s1_validation.py
[DFGEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c644925b50587072784846df09bf02c39e8453e1/src/trotterlib/df_hamiltonian.py
[AX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2796d26270cf39e46b40accc4f898fe29d459819/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/input_reference.json
