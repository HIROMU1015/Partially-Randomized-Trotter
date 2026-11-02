# C1-1-R1：新partition来歴のH4 basis・G6 exact model

**`R1_LIMITED_REANALYSIS_COMPLETE`。D0 partitionをbytes/hash両方で再現し、13 basis、324 label、新exact λ/pを取得した。保存DFとの相対spectral残差は `4.609337135740557e-13`（同じ比較の既存条件 `1e-10` 内）。厳密なmodel bindingは`MODEL_BINDING_UNCLOSED`、basisは旧実装と区別した新来歴。実行1回・retry 0。公開・保全確認後mandatory STOP。**

これは承認された入力・basis・係数・限定operator再解析まで。旧PR-2/C1-1/D0結果を変更せず、G6–DF adapter、新sampler、native価格、sampling、十分N、性能比較を実施していない。科学的優位・独立原著価値・投稿GOを認定しない。

## 1. Gate別の判定とidentity

| 対象 | R1の確認範囲・状態 |
|---|---|
| 保存snapshot・全必要配列・state/sector digest | `VERIFIED`。保存V1 record/layoutと一致、既存state/Hermiticity条件内。厳密ground state・物理準備可能性は未認定 |
| D0 partition payload/hash | `VERIFIED`。canonical 1,834 bytes一致。旧PR-2 hashとの不一致は保持 |
| Basis操作・registry | 13定義を保存、`BASIS_RECONSTRUCTED_PROVENANCE_DISTINCT`。descriptor/局所行列SHA/canonical basis hash一致。旧basis列のbyte-exact一致は未確認 |
| Component対応 | 324 ID・順序・support・fragment・identity/sign対応を保存。同じID順、sign差0、係数hex差135。旧実行入力と同一とは呼ばない |
| 新exact係数/λ/p | 正p、sum1、label一対一、λp sign=cをexact arithmeticで確認、`VERIFIED` |
| 保存DFとの数値binding | `MODEL_BINDING_NUMERICALLY_SUPPORTED`、以下の限定した全operator比較に対する判定 |
| 元DFとのrigorous binding | `MODEL_BINDING_UNCLOSED`。新norm certificateなし。係数のexact性と別問題 |
| 候補short-step | T=4/5、q=1、r=4、m=5ならx≤1をexact確認。実C1 schedule未固定、全occurrenceは`CONDITIONAL` |
| Adapter / provider・誤差・費用接続 | `IMPLEMENTATION_REQUIRED` / 未完・別承認対象 |

| Identity | 値 |
|---|---|
| 固定snapshot | `artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz` |
| Snapshot bytes / SHA256 | 8,652 / `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a` |
| 保存Hamiltonian hash | `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`、R1も一致 |
| 歴史的旧PR-2 partition hash | `722ec0077b4505d963c03bccdd6311f9dd8b3d3a5ed1b9117f7a6f8fa85a7450`、不変 |
| D0 / R1 partition hash | `16b35c1dde1a6807d39c733defd87e315e4ac7317a831b240d0f7214e744a834`、bytesも一致 |
| 新preparation hash | `1750dbcc44f387e0eac0ae33a2eea84f9aabecc3f05be16d1360578cf2dde25f` |
| 新tail hash | `57e4bf36084310c1ce279a4a01936896fd0e8d7edacbc0463520d4299ccab219` |
| 新G6 model SHA256 | `e96ea0faea31bc6a784dfb8dc6f7cf4856789b0256e4a1c53b63ce087c86dc49` |

モデルはlinear H4、1.00 Å、STO-3G、DF rank12、8 system qubit、生成順prefix3。det `[0,1,2]`、tail `[3,4,5,6,7,8,9,10,11]`、`explicit_ordered_partition`を固定。[入力記録](result/input_identity.json)、[state検査](result/state_checks.json)、[partition比較](result/partition_identity.json)、[canonical payload](result/partition_payload.canonical.json)。D0の新来歴をR1の条件とする方針は今回のユーザー明示承認による。旧PR-2完全再現の認定ではない。

## 2. Basisとcomponent

[Registry index](result/basis_registry_index.json)から13個のbasisファイルへ辿れる。各ファイルにbasis ID/hash、original fragment、対応component、time-order操作列、gate class、descriptor、qubit、parameter、局所行列のreal/imag float hexとSHAを保存した。deterministic blockのdiagonal値等と最初の全component checkpointは[preparation checkpoint](result/preparation_checkpoint.json)。

既存のGaussian/JW/registry経路だけを1回呼んだ。元のserialized registryは[以前の限定探索](../track_b_c1_1_input_basis_exact_model_20261020/registry_search.json)で未取得であり、旧C1-1も保存前に停止、D0はbasisを作っていない。したがって旧操作列と同じ実装・native費用とは認定しない。固有vector位相・縮退自由度・runtime依存を無視していない。

Bはtime-orderのlast…first、個別operatorは `B Z_support B†`、実時間順はB†→Z→B。actual adjointは逆gate順と共役転置で定義する。[既存operator SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_rte_tail.py#L907)。basisのunitarity Frobenius値も保存したが、診断値だけで個別basisに新たなPASS閾値を設けていない。

[Component比較](result/component_comparison.json)は旧保存行とのID対応と全係数hex差を保存する。今回は旧/new ID列が一致、old-only 0、sign差0、135/324係数がbitwise不一致。小数表示や旧float λの一致で、同一vector/同一入力とは扱わない。新しい保持集合・順序は既存抽出SOURCEの出力そのもので、追加sort/mergeなし。threshold0で333 componentを保持し、identity9をscalar phaseへ移し、非零random324 labelを残した。threshold drop0。[既存抽出SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_rte_tail.py#L560)。

## 3. 新G6 exact model packet

[Model core](result/exact_model_core.json)、[model identity](result/model_identity.json)、core内のordered label chunk一覧が確定した新C1入力。9 chunkは抽出順に36 labelずつを保存する。各行にlabel/component/fragment/support、sign、新coefficient binary64/hex/exact rational、p exact、basis ID/hashを結び付けた。同basis/同係数/同supportという理由で別labelを統合していない。

採用したのは**今回再抽出した有限binary64 c_iのexact dyadic像**。旧float λや旧probabilityではない。新normalizerは

`3749117741741871730460790747818994150798451967898295072210771/6427752177035961102167848369364650410088811975131171341205504`。

λ=Σ|c_i|、p_i=|c_i|/λをexactに計算し、全p_i>0、Σp_i=1、λp_i sign_i=c_iを確認した。これらの係数関係に限ってexact arithmeticで成立している。モデル全体を`MODEL_BINDING_EXACTLY_ESTABLISHED`とは呼ばない。model SHAはcoreのcompact sorted UTF-8 bytesをhashし、coreのordered chunk hashesとtail/basis来歴がlabel bytesを結ぶ。coreファイル自体のSHAとは別定義。

Signed involutionは `Q_i=sign_i B_ideal Z_support B_ideal†`。ideal unitary gate semanticsと、保存した浮動小数点局所行列を区別する。語Q(i1)…Q(il)はoperator左から右、実時間は逆。G6 parent位相 `(-iσ)^len(u)` と `exp(-iσφQ_i)` を保持し、signをさらに二重適用しない。[固定v1.1 §§1–2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md)。今回はevent・ReturnKernel・回転angle・providerを生成していないため、その実装接続の動的検証は別段階。

constantと抽出tail identityはlabel外の `exp(-it c)` 位相として保存した。tail identityは旧値と同じbinary64 hex `0x1.7b5a0a49eddedp-4`。これらもadopted binary64のexact rational像をcoreに記録したが、元DF Hamiltonianの厳密分解を証明したものではない。

## 4. 数値bindingとshort-step

[数値検査](result/model_binding.json)の対象は、保存full DFに対する「同じconstant/one-bodyを含むselected deterministic DF＋新tail（抽出scalar identity含む）」である。既存 [PR-2比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/pr2_new_series_validation.py#L489) と同じ `||new_tail-(full-deterministic)||₂/max(1,||full||₂)` を評価した。

- 新model数値評価：`4.609337135740557e-13`、既存 `RECONSTRUCTION_TOLERANCE=1e-10` 内、`MODEL_BINDING_NUMERICALLY_SUPPORTED`。
- 旧係数を**R1 basisへ載せたhybrid**：`4.609338560771245e-13`、診断値のみ。別のPASS閾値を流用せず、元PR-2 operatorの再現とは呼ばない。
- 旧PR-2の本来の抽出operatorとの完全比較：旧basis操作列がなく`UNRESOLVED`。
- rigorous operator-norm certificate：未取得、`MODEL_BINDING_UNCLOSED`。係数の有理化やfloat残差をstrict certificateへ格上げしない。

共通basis対角和とBをcacheして重複dense積を避けた。これは監査の加算順を変えるが、label lawを変えない。各exact dyadicは採用binary64へ正確に戻る一方、行列積/和/norm自体は浮動小数点演算である。各basis単位や全coherent-signal誤差/confidenceへこの判定を無条件に拡張しない。

[Short-step記録](result/short_step.json)：未採択候補T=4/5、q=1、r=4、m=5ならδ_tail=4/5、x=λ/5≤1がexactに成立し、binary64 time像でも成立した。既存SOURCEはdeterministic半stepの間にfull step_timeのtailを置き、そのtailをr回分割する仕様。[schedule参照](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_partial_s2.py#L712)。このfactoryはsamplingを含むため呼んでいない。実C1 schedule/time/signは未固定であり、全occurrence成立はCONDITIONAL。時間/r/m/splitを変更していない。

## 5. 実行・SOURCE・保全

| 項目 | 値 |
|---|---|
| Branch | `track-b-c1-1-r1-new-partition-exact-model-20261021` |
| Base | `8188c8a50f79ea9d17fddb987c651650b58aa22c` |
| 科学SOURCE参照 | `243eea8e840f3c472753f7935e510391626fd325` |
| R1実行SOURCE commit | `418a76987782ce50be670b95b8f998043b3f40f5` |
| 新authorization commit | `3fb433dcd1879e43fe444795e65dbd711ee5e1b4`（SOURCEの直接子、receiptのみ追加） |
| Run ID | `track_b_c1_1_r1_h4_new_partition_basis_exact_model_20261021_one_shot_v1` |
| Run/retry/exit | `1 / 0 / 0`、旧authorization再利用なし |
| Runner wall / own CPU / peak RSS | `94.76896238699555 s / 267.16813899 s / 383624 KiB` |

[契約](execution_contract.json)、[SOURCE監査](SOURCE_AUDIT.md)、[manifest](source_manifest.json)、[新authorization](authorization_receipt.json)、[run identity](result/run_identity.json)、[コマンド](result/launch.json)、[terminal](result/terminal.json)、[outer exit](result/process_exit.json)、[実行結果照合](execution_outcome.json)を保存。wall/CPU/RSSはrunner self、CPUはそのthreadsを含み、outer launcher/別Git helper/child hierarchyの総量ではない。量子資源・速度比較benchmarkではない。

同じ既存隔離runtimeを読取使用し、Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、qiskit-nature0.7.2等を固定。[環境identity](environment_identity.json)、[重要runtime SOURCE SHA](critical_runtime_files.json)。元PR-2の全binary/操作列が同一とは言わない。package追加・version/BLAS探索なし。専用cache、既存BLAS既定値、CPU/RAM/workerの独自上限なし。実行前affinity64、MemAvailable `451956416 kB`、storage free `430352510976 bytes`。[資源観測](result/prelaunch_resources.json)。

[呼出し観測](result/call_observations.json)はgeneration1、prepare1、partition helper2（内側含む）、dense reference2。これは指定Python関数のprofileで全C call証明ではない。許可された既存basis抽出は内部BogoliubovTransformのbasis用オブジェクトを使用した。新event/wrapper build、compile/synthesis、SCF/DF生成、sampling/native/Nは実行していない。

[人工logger4ケース](artificial_logger_checks.json)を科学実行前に確認。実行中は各Gate判定前にcheckpointを保存した。[stdout](result/raw_stdout.log)にmandatory STOP、[stderr](result/raw_stderr.log)は0 bytes、failure/secondary failureなし。成功のため架空のtraceback/failureは作成しない。実行後のSOURCE修正・再試行なし。

[保全検証](preservation_verification.json)で232固定ファイル不変、旧C1-1/D0の実worktree61ファイルもbytes/SHA一致。旧C1-0/方法ノート等は基点に対し不変、旧raw/auth/grant/claim/ledger/marker/STOPを変更していない。非公開レビュー原本2件も不変・未公開。全変更はR1 namespace追加のみ。共有環境・他worktree・他processへの変更なし。

[Run output manifest](run_output_manifest.json)が原ログ等を含む44ファイルのbytes/SHAを記録する。runner inventoryはouter log等copy前の時点をそのまま保持。最終公開HEAD・全公開bytesの照合はpublication receiptと最終確認で記録する。

## 6. 引継ぎ

新partition来歴の入力/basis/exact係数packetは取得できた。旧C1-1/D0を成功に書き換えず、新たな実行証拠として扱う。元basis完全再現、rigorous DF binding、strict controlled provider、実schedule、coherent bias/confidence/認証費用、adapterは未閉鎖。

次はユーザー・GPTがこの技術結果を評価して、どの仕様・保証を先に閉じるか判断する。Codexは別の準備・実装指示と必要な明示承認がある場合だけ後続へ進む。[handoff](handoff.md)。主担当Codex本人の静的・人工確認と一回再解析であり、別agent/reviewerの独立再現ではない。**公開・保全確認後mandatory STOP。**
