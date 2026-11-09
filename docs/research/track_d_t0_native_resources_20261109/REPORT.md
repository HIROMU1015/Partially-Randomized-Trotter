# Track D T0：Streaming資源計数・max-plusの最小実装

2026-11-09 JST。**確定native命令列を一度だけ読む計数器と、三種類のwire transfer・連結・embeddingを実装した。人工／回帰29件PASS。** full QuantumCircuit、P0 compile、state simulationを必要としない。出力は常に `NATIVE_EXACT` と `P0_UNQUALIFIED` を併記し、未compile PR入力のP0後価格を保証しない。

実装は [track_d_native_resources.py](../../../src/trottertracks/track_d_native_resources.py)、試験は [check_track_d_native_resources.py](../../../tests/check_track_d_native_resources.py)。[仕様](IMPLEMENTATION.md)／[検証](TEST_RESULTS.md)／[T1未実行案](NEXT_EXECUTION_PROPOSAL.md)／[STOP](STOP.md)。

## 実装範囲と再利用

同じリソース評価系列のbranch `codex/resource-eval-track-d-longq-structure-20261109` を継続利用し、[固定設計a2b7099d](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/a2b7099d5847ad1790a5f077aa45e9e8da2f42e1/docs/research/track_d_longq_structural_resource_design_20261109)の次段階を新namespaceへ追加した。設計文書・旧STOPは変更しない。適用AGENTS、PROJECT_MAP、正式共通指示2026-11-09の§8.1/8.4/10.1/10.3/10.4と既存固定契約を確認した。承認証明機構や新grantは作っていない。

| 再利用SOURCE | 判断・差分 |
|---|---|
| C7-2 `weighted_depth` | filtered-out操作もwireを同期するrecurrenceを再利用・拡張。旧関数は零入力の一指標計数でQiskit等をimportするため、薄いadapterだけでは一回限りのstream・非一様arrival・transfer保持を満たさない。元SOURCEを変更せず、小さい共通row更新へ適応した |
| 設計 `check_maxplus_mock.py` | 既存の整数row更新・max-plus積を移植し、native検査・三指標・count・register embeddingを追加した。元mock7件もそのまま回帰実行 |
| 既存 `track_d_minimal_reference.metrics` | 新しい人工native回路をQiskit1.3で計数し、関数本体をそのままoracleとして利用。transpileは呼ばない |
| C7-4 `event_features` | 定義を読解し変更しない。既存のpathwise lane-max回帰methodを限定抽出して実行。feature抽出やfitを再実装しない |
| C3／StrongBuilder／Reference adapter | head・relative Gaussian・tail共有・step融合・actual inverse・CSWAP・χの接続点を静的確認。科学builderは呼び出さず変更しない。追加interfaceは確定native列の受取・remap・既存native circuitの読取だけ |

本体はstdlibのみの `trottertracks` moduleに置いた。`trotterlib.__init__`の広い科学依存を読み込まない。既存counterの関数本体をASTで取り出すのは**テストoracleだけ**で、本番コードではSOURCE抽出・exec・科学runner importをしない。

## 対応する価格と依存関係

- 入力はrz/sx/x/cx/measureのbound native命令。RZ角は有限実数、qubit/classical indexとregister幅は明示する。未展開CRZ/RZZ/Gaussian/CSWAP、directive、条件、symbolic parameter、不明なcustom gateを拒否する。
- RZ/CX count、各depth、total depth、size、logical widthを返す。measurementもcount/size/totalに含め、filtered depthの同期とclassical wire依存を残す。idle qubitも宣言幅に含む。
- 非一様な整数arrivalを指標別に受け取り、三種類のexitを返す。空template、laneの独立性、shared ancilla後続gate、classical mappingを扱う。
- `then`は時間順の合成、`embed`は全wireの依存関係を保持した埋込み。gate countは別に加算し、同じwordをchunk化しても相殺・再合成しない。
- actual A†にはcallerが供給する実際の逆native列を使う。transferの逆行列や自動gadget生成は実装しない。

global phaseは通常gateとして加算しない。明示された位相をmetadataとして保持し、template連結時はbinary64の診断的和を扱う。量子作用やχの正しさ、位相のrigorous enclosureを保証する仕組みではない。scalar/χがnative RZ等として既に現れる場合は、その命令を通常どおり数える。

## 人工・回帰結果

stdlib試験18件、Qiskit1.3 reader／既存metrics照合4件、旧設計mock7件、合計29件PASS。18件には旧C7-4 feature回帰1件が含まれ、重複して30件とは数えない。85種類の長さ0〜3のnative wordについて全313 cutを検査し、一括stream／template合成／非一様arrivalが一致した。X/Yやwhole-pathの科学標本数とは無関係である。

Qiskit試験は小さい人工native circuitのみで、transpile呼出しを禁止するpatchの下で既存metricsと比較した。simulation・新P0・H4入力は使っていない。stdlib coreのimport時にQiskit/numpyが読み込まれないことも別に確認した。試験は小さく、独立な三suiteを並行実行した。資源上限・共有環境の変更や新benchmarkはない。

## P0・長qに残る条件

確定wordのcount/depthは整数として厳密で、計数器は命令を保持しない。通常計数は有界arityで時間O(L)、作業memory O(W)。transfer captureはO(LW)/O(W²)、cached transfer適用はO(W²)、一般合成はO(W³)。Wはqubit＋classical wire、価格指標は3固定である。実際のPR descriptorからnative wordを作る仕事やcache storageは別に残る。

未解決なのはP0前後のGaussian/parity/CX/1q再合成/CSWAP界面、有限frontier閉包、長q絶対価格と全law期待値である。native列をchunkごとに独立compileした価格の和と、全P0価格の一致は今回保証しない。208−224qの208／全残差0の未証明部分、主A・比較F/S、N、全13-round会計は変更していない。

参照した分子研究は固定の統合9d12d7c6、[q64最終fe361b20](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe361b206a6572142775ac01972545d312e5fb14/docs/research/track_d_c8_1_q64_20261108/COMPLETION_RESULT.md)、[C7-2 709d6ed9](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/709d6ed9c234bf6d5a8848d3a30be69d88fd6476/docs/research/track_d_c7_2_cx_structure_20261107/REPORT.md)、[C7-4 64072ba8](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64072ba81bb70db1df1b34e94adb08e78926f24b/docs/research/track_d_c7_4_model_stability_20261107/MODEL_REPORT.md)。それらの数値を再計算せず、raw/QPY/NPZを開いていない。T1対象を具体化するためC7-2**人工**traceのID／schemaだけを読んだが、新counterを適用するT1比較はまだ行っていない。

## 保全・終了

既存SOURCE、科学builder、旧tests、入力・raw・結果・grant・ledger・marker・STOP、q64結果、統合枝は不変。追加は本体1module・人工test1file、新しい技術ログ、今回の文書とnavigationだけである。非公開payload・内部環境情報を追加公開しない。

**commit/push、remote SHA・tree・公開bytes照合後mandatory STOP。** 次の担当はユーザー/GPT：まずT1の保存人工trace限定比較を採択するか判断する。CodexはT1、新P0、H0、長q実測、N・全RPE費用、詳細科学レビューへ自動進行しない。
