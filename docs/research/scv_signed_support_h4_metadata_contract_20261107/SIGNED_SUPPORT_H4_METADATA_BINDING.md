# Signed support実H4 metadata-only：固定入力と最小接続

2026-11-07 JST / SERVER1（GPUA100-gleap-01）。**静的設計。実H4の新特徴・moment・非定数性は未評価。接続SOURCEも未実装。**

## 1. 再利用できる範囲

固定実装 `b23b3c059dd1fdcc86178659311983048dea2257` の [scv_signed_support.py](../../../src/trottertracks/algorithm_codesign/scv_signed_support.py) を変更せず再利用する。既存 [SOURCE記録](../scv_signed_support_preparation_20261107/SOURCE.json)、[実装](../scv_signed_support_preparation_20261107/SIGNED_SUPPORT_IMPLEMENTATION.md)、[correctness](../scv_signed_support_preparation_20261107/SIGNED_SUPPORT_CORRECTNESS.md)、[次回scope](../scv_signed_support_preparation_20261107/NEXT_SCIENCE_SCOPE_DRAFT.md)、[REPORT](../scv_signed_support_preparation_20261107/REPORT.md)を確認した。

人工106件PASSは既報の参照で、今回の再実行・実科学評価ではない。核となるfeature・moment処理に追加修正が必要な証拠は静的確認では見つかっていない。ただし、実H4へ接続した成功を先取りしない。

既存基盤 `560c6092bfb5e71af8eedc4a4be5416922764f98` の `scv_scalar.py`、`a1_control_variates.py`、`a1_n3_connection.py` はb23b3c0とbyte一致。再利用対象はsaved bytes binder、exact normalized categorical law、固定5層の検査、fingerprint、SignedControlのexact有限和。group選択、OLS、range計算、paired評価は今回のmetadata-only接続から呼ばない。

## 2. 入力の正体：catalogはPREPARATION.json

N1結果commit `744e1a09661ebba6841fe5faf456f88c41c47780` の以下2ファイルが科学的な直接入力である。別のcatalogを生成・抽出し直さず、**PREPARATION.json全体の元bytes**をcatalog_bytesとして渡す。

共通relative directory：`artifacts/o2_a1_new_target/2026-11-06/n1/`。

| 直接入力 | 保存schemaと使用field |
|---|---|
| [LAW.json][law] | uniform_grid_size、application_label_order、orders、order_masses[].exact、label_masses[].label/exact。外側の整合確認にはfive_patternsとfive_strata[].exactも使用 |
| [PREPARATION.json][catalog] | hamiltonian_hash、preparation_hash、components[].component_id/df_fragment_id/coefficient_sign/diagonal_pauli_support |

PREPARATIONにはD block・basis操作・係数hex等も保存されているが、binderはそれらから行列・rotation・stateを復元しない。係数絶対値、probability、rotation角度から新しいlawやfeatureを作らない。zero-grid labelも全行のまま渡す。

以下は元commitのGit blobを取得して算出したbyte SHA256であり、N1 RESULTの既報hashとも一致する。新しい物理数値の算出ではない。今回のworktreeでも元blobとのbyte一致を確認した。

| ファイル | bytes | SHA256 | Git blob（N1 commit） |
|---|---:|---|---|
| `LAW.json` | 66825 | `b77980d198cc9edc620daec9bf87df02b66e96575fd85327aed0793f3e86ecef` | `15b40241ba648c0f8750e087425d3baeb053730f` |
| `PREPARATION.json` | 356218 | `ca897db7be52ad30dcfab0512d7530188fc13a9d8ca0c78430809492d01f738b` | `d7cf105c31b30100ee16dc4fefe41999cd5c009f` |

補助provenanceは同じN1 commitの保存記録。実行時にこれらを数値ロードする必要はなく、下記の固定identityを実行設定へ保持する。必要なら元blobとのbyte照合だけを行う。

| 補助ファイル | SHA256 | 用途 |
|---|---|---|
| `CONTROLS.json` | `4d2ac7e2679484e8ed9a8d3abd2c4e2347e892c4505881acb9d2825efe0b1b17` | 固定a,bの来歴 |
| `POLICY.json` | `cb2500f894de6230aedbd0bc43e1ca3b6d458b69b40c55714b61baeaea6a624d` | selected policyの来歴。操作行列は復元しない |
| `RESULT.json` | `8f370a5ec4d4d3084c85f9263923dbd336e1f70db66d00da5d72665a54b4932c` | N1 identityと直接入力hashの既報 |
| `SOURCE.json` | `4b16410a1de32bb8fd6b010522bb2090245257f8fc009e881df0648955d36896` | 元N1条件・SOURCEの来歴 |

## 3. 固定identityと契約上の意味

| identity/設定 | 固定値 |
|---|---|
| Hamiltonian hash | `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5` |
| partition hash | `900342ad6262021ff9b919c36ee182a8ed83914a697524e3fae07248d0d082f8` |
| preparation hash | `f02980a5cbec9478aa1f4e6dd010f117d152f89770c40389544cf876dd197299` |
| tail hash | `890442900e605a262201338f19afdad638302d670c75a9d95371c10b83ea38d8` |
| selected policy fingerprint | `f6eba8633890dbf09b5388928a369650251bf53567d51ab4179282c7c239b2cc` |
| 固定a | `df-fragment-3:Z0Z1` |
| 固定b | `df-fragment-6:Z0` |
| system_qubits | 8。N1 SOURCEの保存scopeから固定。metadata-onlyではsector vectorを作らない |
| feature version | `rotation_coefficient_sign_times_diagonal_support_size_over_2_v1` |
| feature box | [−1,1]。I/Z/ZZのsupportサイズ0/1/2と整数sign±1による |
| 5層pattern順 | `((1,1,0,0),(1,0,0,0),(0,0,0,0),(0,0,1,0),(0,0,1,1))` |
| 位置規約 | products→rotation。n0のL0、n2のL2がrotation |
| LAW identity | LAW.json全bytesのSHA256。ConditionalSamplerのlaw_idと一致させる |
| catalog identity | PREPARATION.json全bytesのSHA256。subsetや整形済みJSONのhashへ変更しない |

Hamiltonian/preparationのfield比較は既存scalar binderが行う。partition/tail/policyは上記固定N1 provenanceへ紐付け、元bytesの一致を用いる。元Hamiltonian・preparationやbasisを再計算してhashを再発行する作業は不要。

N2 `5e9bd08068738cdf1070bdc43bee8194b113ae33` の保証stateとengine認証 `fdf00cee3c87d804845497afc89ac6a71f3b3f59` は**参照commitだけ保持**する。state vector・spectral witness・gate証明の数値ロード、engineサービスの初期化は行わない。selected policyも固定identityの参照であり、full回路を組み立てない。

## 4. 将来の一方向接続：非実行の呼出し仕様

以下は次回用の設計であり、この順序を今回実行していない。

1. 専用のcleanな実行SOURCE commit、依存環境、fresh output、N1 input commit/path/hashを確認する。read_bytesはLAW/catalogそれぞれ1回で、その同じbytesをhash照合とbinderに使う。
2. `scv_signed_support.metadata_from_bytes(law_bytes, catalog_bytes, system_qubits=8, law_sha256=固定値, catalog_sha256=固定値, hamiltonian_hash=固定値, preparation_hash=固定値, a=固定a, b=固定b)` を1回呼ぶ。
3. 内部で `scv_scalar.metadata_from_bytes` → `ConditionalSampler.from_record` → exact CategoricalLaw/OccurrenceLawと固定control banks → scalar.Metadata → signed Component群とsigned Metadataが構築される。ここでは新しいsigned momentをまだ計算しない。
4. 外側では元LAW bytesを標準jsonで1回解析し、固定5pattern・保存five_strataとの照合用recordを得る。`metadata.sampler.cells('B1-OL')` を1回だけ取得して保存値とexact比較し、全5正層・質量和1を検査する。labelの順序も元LAWとcatalog由来metadataで一致させる。
5. `SignedControl(metadata)` を1回だけ構築し、既存のmeans/second_moments/variancesとsnapshotを取得する。momentを別の式・engine・全path列挙で再計算しない。
6. 保存した全5層momentの整合を確認し、exactなΣ1の符号で非定数性を一度判定する。結果・identity・call記録を保存し、成否にかかわらずSTOPする。

同じcomponent行からsign/supportを取り、f(l)=s_l|S_l|/2。固定Qの一位置有限和でk=1,2のm_k=Σ_l ρ(l)f(l)^kを使い、conditional momentは `(f(a)^k, (m_k−ρ(a)f(a)^k)/(1−ρ(a)), m_k, m_k, m_k)`。ν=m₁、Σ=m₂−ν²。既存SignedControlの処理そのものを使い、global平均でh1平均を代用しない。

snapshotの `science_ready=false` はそのまま保持する。別に `metadata_structural_eligibility` を保存し、構造的適格性を測定開始可能性・統計的有効性へ読み替えない。

## 5. 最小の追加SOURCEと既存処理の制約

将来追加するのは、上記2ファイルのI/O・固定設定・軽い外側の整合検査・call記録・exact結果の保存・STOPを行う**薄い入口だけ**。科学核の変更、新sampler/OLS/engine/worker/managerは不要。今回はその入口も実装していない。

既存 `scv224_activation.py` のmetadata部分はbinder接続の参考になるが、親入口・select_fragment_control・stateサービス・PRNG・managerは今回の目的に適さないため呼ばない。`a1_n3_campaign.save/append` のatomic保存/fsyncの方式は参考にできるが、campaign moduleをimportしてengine依存を引き込む必要はない。次回の小さい標準ライブラリI/O接続で同じ保存順序を保てばよい。

既存signed sourceはH/preparation/LAW/catalog/featureをidentityへbindする一方、policy/state/engineまで検証するscientific loaderではない。これらは固定参照・実行設定に残す。metadata-onlyの適格性にstate/engineの再認証を条件付けない。

## 6. SOURCE・依存環境の固定範囲

核の基準commitはb23b3c0。次回の入口追加・人工テスト後、実科学入力を投入する前に**新しい実行SOURCE SHA**を公開・照合する。この草案の文書commitを、まだ存在しないrunnerの実行SOURCE SHAと偽って記録しない。

下表はb23b3c0のmodule-level importをASTで静的に追跡した14ファイル。関数内の遅延engine/価格importは実行経路へ入れない。SOURCE全体はcommitで固定し、表は最小経路を説明するhash一覧とする。

| b23b3c0から再利用するrelative SOURCE path | SHA256 |
|---|---|
| `src/trottertracks/__init__.py` | `2d51bfa08b2b1e000b86ec53bd463e786529307a7270078600e3b03995e90285` |
| `src/trottertracks/algorithm_codesign/__init__.py` | `b6503716cfd51d52ca9038e6de5a411eb78380d30451f81df9b81c31b1be090f` |
| `src/trottertracks/algorithm_codesign/a1_connections.py` | `7feeac68acab48712d8b43edc0db6d07ef90fc1503413feb99b5ac3fac08c6a5` |
| `src/trottertracks/algorithm_codesign/a1_control_variates.py` | `22c8bea657f52b74ffb40e3e74b4f953fb67fee312cb801bccc3f7f369ec1960` |
| `src/trottertracks/algorithm_codesign/a1_n3_connection.py` | `5d2ed2985cbf015665ddb01e77a9c5d947b7aeed38d470429993e722448a6acf` |
| `src/trottertracks/algorithm_codesign/a1_n3_pilot_design.py` | `c4cfb615957cff52495ca7389cd3ac20880a69b5f40766e581719d0a429e7d1b` |
| `src/trottertracks/algorithm_codesign/df_correction_comparison/__init__.py` | `068d65713e74559764c504304b1f3c1640977b68c2f2193e65371c84f08f8ef3` |
| `src/trottertracks/algorithm_codesign/df_correction_comparison/intervals.py` | `aec65b0a7581655b93f0895ae40014f24932e3f3584f3afbdcf15576b2959726` |
| `src/trottertracks/algorithm_codesign/g7_generator.py` | `bb4177d26f1cee9721c2b083de01f4aa309f3ad39f22597cfae3b662881269fb` |
| `src/trottertracks/algorithm_codesign/g9_p5.py` | `322c4d340303e581aaefff1c3a6f58d019caf9e3f60beadc901425e766524d70` |
| `src/trottertracks/algorithm_codesign/parent_allocation.py` | `6ebecaf81a446dcdeaafa89b9bd0018ac8d11d23695da57d58b5683b99a4709d` |
| `src/trottertracks/algorithm_codesign/return_aggregation.py` | `f716d3c69fc1063e47b7618022662715f6b34a9c1f35cc3e7ed26813f5ac69f0` |
| `src/trottertracks/algorithm_codesign/scv_scalar.py` | `8c7739015fe447a940ea1373bdec16f986ad221a14c3aa717ee13a5af3de0b5d` |
| `src/trottertracks/algorithm_codesign/scv_signed_support.py` | `00c6b99e9b706af7e4b3264796e99f7947b262165776bba580f9612f9ba38dd7` |

Python候補は既存read-onlyの `/home/AbeHiromu/projects/partially-randomized-trotter-wp11/.venv-server-opt2/bin/python`、Python3.12.3、mpmath1.3.0。今回、version metadataだけを確認した。mpmathは既存n3 moduleのimport依存だが、本処理でinterval計算・configureは呼ばない。NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0も同venvに存在するが、metadata-onlyの数値経路では不要である。

既存pilotのBUDGET定数がimportされても、新しいshot予算や取得処理を採用しない。実行時はPython実体・version、mpmath version、追加入口と上表SOURCEのhashを記録し、依存差が科学的な意味へ影響する場合は実入力前に解決する。共有venv/設定を変更せず、bytecodeとcacheは自分の領域に分離する。既存engine高速化SOURCEへ置換しない。

## 7. 今回実施した静的確認と未実施の処理

LAW/catalogの元Git blob取得、byte hash、JSONのfield名・保存identityの参照は行った。**実LAW/catalogをmetadata_from_bytes・Metadata・SignedControlへ投入していない。** Fractionによる新しいfeature/moment計算、h1正質量値の集計・非定数性判定、CDF再生成、全path列挙は行っていない。

NPZ、N2 state、spectral witness、exact engineのgate/state payloadは数値ロードしていない。旧224/752のbit・係数・cacheの再解析、新しい相関・分散/RZ比較も行っていない。今回SOURCE・テストの変更と新テスト実行は0。実入力の構造的適格性は次回の一回評価までUNKNOWN。

[law]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/744e1a09661ebba6841fe5faf456f88c41c47780/artifacts/o2_a1_new_target/2026-11-06/n1/LAW.json
[catalog]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/744e1a09661ebba6841fe5faf456f88c41c47780/artifacts/o2_a1_new_target/2026-11-06/n1/PREPARATION.json
