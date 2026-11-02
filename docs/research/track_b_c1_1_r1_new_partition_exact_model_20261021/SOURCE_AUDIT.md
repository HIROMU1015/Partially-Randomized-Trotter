# R1 実行前SOURCE監査・受理仕様

対象は保存linear H4 1.00 Å / STO-3G / DF12 / 8 system qubit / generation prefix3。基点 `8188c8a50f79ea9d17fddb987c651650b58aa22c`。今回のユーザーR1指示を根拠とし、旧C1-1/D0のSOURCE・結果・auth・STOPは変更しない。新SOURCE・契約・新authorization-only直接子commitを固定し、実科学1回・retry 0とする。以下は実行前の仕様であり、成功結果の先取りではない。

## 新partitionの意味

歴史的旧PR-2 hash `722ec0077b4505d963c03bccdd6311f9dd8b3d3a5ed1b9117f7a6f8fa85a7450` は保持する。新C1参照はD0 canonical payload（1,834 bytes）とSHA256 `16b35c1dde1a6807d39c733defd87e315e4ac7317a831b240d0f7214e744a834`。ユーザーがこの新来歴を採用し、R1ではD0とbyte/hash両方の一致を要求する。旧hash判定を黙って無効化したのではない。旧不一致分類や旧結果をPASSに変更しない。

NPZは1回だけ読み、Hamiltonian memberはD0と同じ `np.array(copy=True)` とg-stack tuple viewで具現化する。full/sector state等も同じarchiveから読む。既存 `_digest_record`、layoutと既存state/Hermiticity許容差を確認する。新SOURCEで独立の分子・DF・stateを生成しない。

既存 `generation_partition` と `_partition_hash` を使う。payloadは固定helperの最初の代入AST式をそのまま評価し、sorted compact UTF-8 JSON（末尾改行なし）のhashとhelper返値を照合。実値/bytes/比較結果を保存してからD0一致を判定し、不一致ならbasis前にSTOPする。

## 固定SOURCEとの対応

科学参照 `243eea8e840f3c472753f7935e510391626fd325` の既存 `src/trotterlib` bytesを保持する。[manifest](source_manifest.json)で新実行SOURCE、入力、C1-0/C1-1/D0 packet、重要runtime files等232ファイルを固定。旧結果に記録された主要SOURCE SHAが一致することは[D0静的記録](../track_b_c1_1_d0_partition_hash_20261021/static_reference_audit.json)にもある。

- [generation partition](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/pr2_s0_s1_validation.py#L269)：index順を保つ。ld3、det `[0,1,2]`、tail `[3..11]`。
- [prepare](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_partial_s2.py#L328)：同じexplicit ordered policy、descending_abs、Gaussian/JW、identity extract、threshold0。1回だけ呼ぶ。内側でpartition hashを再算出するので外側の1回と合計2回がSOURCE上の予定。
- [basis fingerprint](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_rte_tail.py#L389)：ordered descriptor、qubit、parameter文字列、局所complex128行列SHAとcanonical basis hashを保存・照合する。新しいbasis IDを任意で付けない。
- [component抽出](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_rte_tail.py#L560)：既存sort/retentionをそのまま使用する。R1で別のsort・集約・新しい閾値を加えない。
- [operator規約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_rte_tail.py#L907)：`B Z_support B†`。回路時系列はB†→Z→B。
- [既存再構成比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/pr2_new_series_validation.py#L489)：`tail - (full - deterministic)` のspectral norm / max(1, full spectral norm) ≤ `1e-10`。R1は同じDF/split/operatorを対象としてこの条件を使う。

完全なserialized registryはC1-1の[限定探索](../track_b_c1_1_input_basis_exact_model_20261020/registry_search.json)で未取得。前回C1-1はserialize前で停止、D0はbasis処理0。新たな原registry提供はないため、固定SOURCEで新来歴として再構成する。全保存領域に存在しないとは断定しない。

## Basis実装と数値検査

既存 `U_to_qiskit_ops_jw` はqiskit-nature `BogoliubovTransform(U.T)` からbasis操作を抽出する。これは承認された既存basis用処理で、内部的にはQiskitのbasis回路オブジェクトを使う。新event/wrapper build、transpile、compile、synthesisは呼ばない。追加の `basis_change_unitary` 回路構築も避け、既存操作の局所行列を `Operator.compose` で時系列に積む。

固定runtimeのOperator SOURCEを静的確認した。default `front=False` はother×self、subsystemではqubit0がtensorの右端となる。従って時系列にcomposeすると B=last…first。actual adjointは逆順と各局所行列の共役転置で定義する。`gate_class`、descriptor、real/imag float hexも保存する。[重要runtime SHA](critical_runtime_files.json)。

Basis descriptorとactual matrixのSHA/canonical hash一致、component→basis参照をGate 3条件とする。旧操作列がないため`BASIS_BYTE_EXACT_VERIFIED`は使わず、再構成後は`BASIS_RECONSTRUCTED_PROVENANCE_DISTINCT`。unitarity Frobenius残差は診断だけで、新しいPASS許容差は設けない。全DF operatorの数値再構成はGate 5で別判定する。

## 新componentとexact modelの採用

**新しく再抽出したbinary64係数**を採用する。旧float λ/probabilityをexact値として流用しない。旧componentとのID/fragment/support/identityと共通labelの相対順を比較し、旧係数hex・sign・保持集合の差も保存する。保持集合や符号が異なっても、新しい既存threshold-zero抽出の結果として明示する。旧component listと同一入力とは呼ばない。新labelは抽出順の非零retained componentsへ一対一で割り当て、同basis・同係数・同supportでも統合しない。

共通componentの構造/相対順、全新componentの合法fragment/support、ID重複なし/非零、係数に対するsign保持はhard Gate。旧係数自体とのbit差は独立新モデルの差として記録し、旧再現PASSの条件にすり替えない。固定SOURCEにないdrop/threshold/sortの変更はしない。

c_iを `Fraction.from_float` でexact dyadicに写し、λ=Σ|c_i|、p_i=|c_i|/λをexactに定義。positive p、sum1、λ p_i sign_i=c_i、label/component一対一を確認し、ordered chunkファイルに保存する。model coreにordered chunk hashes、source/input/environment/policy/basis来歴を結び、compact JSONのSHAで新model identityを定義する。

Q_i=sign_i B_ideal Z B_ideal†をsigned involutionの定義とする。ここでunitarityは既存gateのideal semanticsに基づく。保存した有限精度行列自体のexact unitarityや元DFモデルとの厳密同一性を、係数の有理化だけで認定しない。constant/抽出identityはrandom labelから分離し、既存 `exp(-it c)` 位相を保持する。

[G6 v1.1 §§1–2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md) の Q(i1)…Q(il) はoperator左から右、回路時間順は逆。parent位相 `(-iσ)^len(u)` と単一signed Q_i回転 `exp(-iσφQ_i)` を維持する。signをさらに外側で二重適用しない。今回はG6 event/ReturnKernel/sampler/angle/native providerを作らず、この接続規約と入力定義だけを保存する。

## Gate 5と許容差の範囲

full DFとselected deterministicを既存dense referenceで各1回だけ構成し、新tailは共通basisごとの対角和とBをcacheする。label自体は統合しない。この線形監査の加算順は既存component単位加算とは違うことを記録する。scalar identityを含め、上記の同じfull/selected DF比較にだけ `RECONSTRUCTION_TOLERANCE=1e-10` を適用する。

state/Hermiticityは既存 `1e-12`。個別basis unitarityや旧係数を新basisへ載せたhybrid診断には同じ許容差を流用しない。hybridは旧/new ID列が同じ場合だけ計算し、元PR-2 operator再現とは呼ばない。旧basis列が欠けるため、本来の旧抽出operator全体との比較は未確定。

new dyadic c_iはbinary64へ正確に戻るので、その数値行列評価を使う。行列積/和/normはfloatであり、diagnosticが通れば`MODEL_BINDING_NUMERICALLY_SUPPORTED`という限定判定。rigorous operator-norm certificateや元電子Hamiltonianとの厳密bindingは`MODEL_BINDING_UNCLOSED`を維持する。

T=4/5、q=1、r=4、m=5は未採択候補のまま、x=λ|T|/4をexactに確認し、binary64 time像も記録する。既存SOURCEのpartial-S2はdeterministic half→tail full step_time→reverse half、tail r repetitionsという仕様。実C1 schedule/time/signが未固定なので全occurrence成立はCONDITIONAL。request factoryはsamplingを含むため呼ばず、静的参照だけにする。

## 保全・logger・資源

[人工logger4ケース](artificial_logger_checks.json)はstdlibのみで成功。stage衝突なし、一次例外/tracebackを保存、二次logger failureは別記録。Gate判定前のcheckpointを先に保存する。SOURCE/import経路は本人が静的確認し、科学入力を使った試走は行わない。

同じ既存D0 runtimeを読取使用し、package追加・BLAS/version探索なし。環境metadata/実体と重要runtime実装のSHAを固定するが、元PR-2の全binaryと同一とは言わない。CPU/RAM/workerに独自上限を設けず、既存BLAS既定値、依存Gate順とfail-fast、共通basis cacheで実施する。cacheは専用runtimeのみ。共有環境・他worktree・他processに変更しない。

科学実行は1回、retry0。失敗後SOURCE/input/閾値を修正しない。成功・失敗どちらも証拠を公開・照合してmandatory STOP。別agentの独立レビューではなく、主担当Codex本人による静的確認と人工確認である。共通正式指示2026-10-17と両AGENTS.mdを適用。ユーザーの新namespace追加限定指示を優先し、既存索引/claim/ledger/STOPは編集しない。
