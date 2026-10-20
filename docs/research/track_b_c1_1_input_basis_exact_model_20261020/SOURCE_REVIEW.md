# C1-1 実行前SOURCE確認

C1-0 `a48b3c8244fb5d932e0d6bf4a5de6bb573ce35ed` の専用branch。ユーザーの今回のC1-1指示だけを実行根拠とし、旧grantは再利用しない。新runnerは入力同一性、既存basis再構成、exact係数sidecar、限定行列比較を行う。G6–DF adapterではない。

- 固定参照 `243eea8e840f3c472753f7935e510391626fd325` の `src/trotterlib` blobと作業領域の全対象ファイルを照合。既存科学SOURCE変更なし。
- `_load_snapshot_once` のkey/layout/embedded digest検査と `_digest_record` を使用。full PR-2 runner、held-out入力、SCF、state再生成は呼ばない。
- `generation_partition(h,3)` → `prepare_df_partial_s2` を1回。既存の `descending_abs`、`fermionic_gaussian_jw`、`extract_identity_phase`、threshold 0、explicit ordered partitionを維持。
- 既存basis経路は内部で `BogoliubovTransform(U.T)` を使う。これは許可されたGaussian/JW操作抽出のための既存basis用オブジェクト。新event/wrapper回路、transpile、compile、synthesisは呼ばない。`basis_change_unitary` の追加回路構築も避け、局所gateの `Operator` を順に行列合成する。
- 既存 `component_dense_operator` の規約は `B Z B†`。signed labelは `sign(c) B_ideal Z B_ideal†`、時系列では右端を先に適用。保存binary64行列そのものの厳密unitarityやモデルbindingは認定しない。
- 係数は保存JSONの順を保持し、`Fraction.from_float` でexact dyadicへ写す。λを再度exact和で定義し、pのpositivity/和1/λp sign=cを検査。別operatorを集約しない。
- `H_D` は既存 `h.select_blocks([0,1,2])` と既存dense DF reference。tailはbasis別に対角係数をcacheして和を取る。これは監査の加算順を変えるだけで、label/lawを変更しない。
- 既存V1–V3のstate/Hermiticity `1e-12` と、同じDF再構成に使ったrelative spectral `1e-10` を使用。unitarityには新閾値を設けず診断値のみ。数値残差をrigorous norm certificateと呼ばない。
- 実行前構文確認は `ast.parse` のみ。科学モジュールのimport、NPZ開封、runner実行はSOURCE/auth commit固定後の1回に限定。科学test・試走なし。
- 新しい解析の主担当Codex本人による静的確認であり、別reviewerによる独立レビューではない。

環境は既存の隔離 `20261012/native-dependency-v2` を読取利用し、版やpackageを変更しない。元S0環境に記録された主要7 package版と一致するが、当時のqiskit-nature版は未記録のため完全な環境再現とは扱わない。qiskit-natureのない既存環境は、8-qubit fallbackが固定registryの4-qubit gate制限に適合しないため、科学実行前のmetadata調査段階で除外した。結果に合わせた版の探索はしていない。

`execution_contract.json`、`environment_identity.json`、`source_manifest.json`、`registry_search.json` が実行範囲と固定値の機械可読記録。authorization commitはSOURCE commitの直後でreceiptだけを追加する。実行は1回・retry 0。不一致/失敗は原ログを保存してSTOP。成功時も公開・保全確認後にmandatory STOP。
