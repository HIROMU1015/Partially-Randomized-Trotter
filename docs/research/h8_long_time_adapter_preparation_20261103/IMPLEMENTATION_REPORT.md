# SERVER2 H8長時間・高次PR adapterの実装準備

2026-11-03。**実装・人工検証まで完了。Stage A/B/Cは未実行、科学実行はHOLD。**
実H8の信号、reference、sampling、回路build/compile、価格、順位の新結果はない。
検査は主担当Codexが行ったもので、別担当者による独立科学レビューではない。

第一計画資料は固定commit
[`d829ead865e7ff5defe38ba9d97a5662c75441aa`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/d829ead865e7ff5defe38ba9d97a5662c75441aa/docs/research/h8_p8_long_time_preflight_20261103)
のP8原因報告、長時間計画、次段階判断、SOURCE_REUSE.csv、VALIDATION.json。
現在の共通AGENTS、実際のworktreeのAGENTS/PROJECT_MAP、共通運用指示2026-10-25の§8.1/8.4/10を適用した。

## SERVER2で確認したもの

保存H8 native configが指す既知の封印storeだけを確認した。NPZの数値配列はdecodeしていない。
snapshot、各receipt、preparation export/identity、関係するSOURCEのファイル存在・サイズ・SHA256を確認した。
pathを除いたidentityと検査記録は[VALIDATION.json](VALIDATION.json)に保存した。

| 科学入力 | 固定identity |
|---|---|
| Linear H8、1.00 Å、STO-3G | 保存tol-only1e-8 DF rank28、16 qubit、spin sector4900 |
| Hamiltonian | `1a5ebefa18fc1c9f01ad644b81d13f49c7bf944bccfe3837d459c35d3325dbe2` |
| state | `50a8fcb4cd29d60349db8e8fe728cb609eaaa903d9531833ef63e2711fa363fb` |
| snapshot | `d4c4346127e97c241439ae1c4ae010962b0456b19f461cfd3d9b06e7d4a50955` |
| DF receipt | `4911d1f5faab5fe81bfb70c6c2ef01265e3085107a513d4c990e1f39b44a7e0f` |
| Full preparation export | `771137828552b858361b6e2b25d2b675fb83f908aad6209f757e46464b817420` |
| compiler | `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` |

既存専用venvのdistribution metadataでQiskit1.3.0、NumPy1.26.4、SciPy1.14.1、Numba0.60.0、pytest8.3.5を確認した。
実H8での速度・メモリ・GPU適合性は未測定。
既存providerにOccupationCSR、matrix-free DF operator、spin-factorized primitive、sparse exponential/reference pairがあることを静的に確認した。
4900-square dense行列・全固有スペクトルを作るH46 contextは使わない。

prefix15/16は保存Full28の先頭16/17ブロック（one-body込み）として取り出せる。
DBA保存bindingにはB0_p15とB0I_p16があり、後者にはexact binary64 tail identity receiptがある。
この対応だけをmetadataで確認した。新adapterでの作用一致・p16 scalarの再照合は未実行。
p15のB0Iには新scopeでexact scalarを評価する必要がある。保存receiptがあると偽らない。

## 新しい実装

- [h8_long_time_adapter.py](../../../scripts/h8_long_time_adapter.py)：metadata preflight、候補identity、固定schedule、finite多項式、reference identity、paired G、CLI。
- [h8_long_time_runtime.py](../../../scripts/h8_long_time_runtime.py)：lazy H8 loader、Full view、sparse Stage A、再利用native pipelineによるStage B/C。
- [proposed_scope.json](../../../configs/h8_long_time/proposed_scope.json)：第一科学候補の提案domain。実行する候補一覧とは区別する。
- [test_h8_long_time_adapter.py](../../../tests/test_h8_long_time_adapter.py)：人工小行列・mockの検査。

H8 loaderの保存Full-block restoreをそのまま使い、prefix7/8/15/16の切り分けを接続する。
Full28だけは空tailの抽出を避け、既にrestoreしたFullブロックへのviewを加えた。
このためloaderのprefix関数を呼出中だけ観測し、finallyで元へ戻す。原ファイルは変更しない。
原scalarとPRのextracted identity、B0Iのexact stored-binary64 scalarを区別する。

S2/Y4/m5のbinary64係数・piece順序・整数配分は既存の純粋schedule関数を再利用。
H8ではsparse tail作用のforward/Horner多項式を使い、Kはeven event cutoff、多項式最高次数はK+1。
rawとcorrectedを別に伝播し、normalization closure、exact-tail PF、full-H referenceとのbiasを保存する。
canonical tauは既存samplerと同じsymbolic lambda・演算順を使う。
shot政策は既存qualify_recordを再利用し、empirical allowance floor、Decimal照合、1% headroom条件を緩めない。

nativeのcompose/sample_requests、actual gate adjoint、symmetric-directional control、measurement付き両軸wrapper、compiler/cache、rarity診断を再利用する。
旧event_oracleにはT=.8固定scalarがあるため直接呼ばず、既存block/event作用を新Tに接続する短いwrapperを設けた。
scalarはsigned schedule全体の後に一回だけ適用する。
同じtrajectoryからcos/sin価格を得て、Stage Aで固定した別々のNでGを会計する。
確認精度設計もunweighted単価和ではなくpaired Gの分散を使う。

## 実行入口・現在の状態

CLIのdefault `--stage inspect` はファイルhash確認だけで、科学moduleはimportしない。
Stage A/B/Cは`--execute --scope ... --output ...`が必要。switchは誤起動防止であり、利用者の実行承認を代替しない。
SOURCE/input/scope不一致、旧時間reference、非適格候補、cohort混合、既存outputへの上書きを拒否する。
入力・内部path・seed実値を持つconfigは専用cloneのGit管理領域だけに保管し、公開していない。
新結果の保存先もこの作業専用run領域に限定する。資格情報や承認証明packetは作らない。

実行用Stage実装は用意したが、実物理入力との接続・数値精度・compiler挙動は今後のStage自身の検査対象。
人工PASSからH8の適格性・科学的正しさ・PRの資源優位を結論しない。

## 原本とSTOP

元SOURCE、入力、結果、QPY、grant、ledger、marker、旧false、one-shot/retry条件は変更していない。
[`862d732fba0a5f353c522a9219dec78ae2f21a4b`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md)
の希少RTE正式終了を維持する。新実験は旧runの再開ではない。
G04補完の固定記録は
[`37694741ee356fd09b5dfcb56130c55ab381044c`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103)。

**mandatory STOP：今回の成果物公開後、科学実行は開始しない。**
