# Track A AX-2B：H4限定検証の実行前契約・metadata preflight v3

2026-10-10 JST。利用者の「作業を進めて」は、直前に提示したH4の入力・coverage・実行条件を固定する**準備**へ適用する。
[GPT独立レビュー](track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md) §21および[接続準備 v2](track_a_ax2b_bound_ports_preparation_v2.md)を具体化した。
新しい分子計算、native分子準備、signal、sampling、回路構築・compileは実行しない。
`H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。

## 固定できたものと残るもの

[metadata preflight](../../artifacts/resource_applicability/track_a_ax2b_h4_prelaunch_preparation/2026-10-10/metadata_preflight_v3.json)にinput/source/旧結果hashと全scheduleを記録した。
入力はlinear H4 1.00 Å、STO-3G、legacy DF rank12、generation-prefix、8 system qubits、Nα=Nβ=2のsector36、T=0.8。
targetは保存binary64 DF係数の数学的Hamiltonianと、元の保存vectorを数学的に正規化した指定state。真の基底状態や元積分Hamiltonianへ置き換えない。

| 項目 | 固定・観測した内容 | 状態 |
|---|---|---|
| 保存入力 | [既存snapshot](../../artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz)、8,652 bytes、SHA-256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a` | localとsource commit blob一致。NPY header/Unicode metadataだけをdecode |
| scientific source | `8281c2a59e7d3ea0c77ae03a4fe0227361165c1d`のrunner closure167件 | local/Git bytes一致。既存preparation freeze178件も不変 |
| 旧v5証拠 | result保存commit `aa9b4768819680600d99ceb962b22aec16b99fb0`、実行base `b2e1bf65e21893b6c617223b42313623d3186f12` | input/reference、8 correctness、保存監査、terminalの11記録をcommit blobまで照合。旧実行を新source実行に読み替えない |
| 検証coverage | 179組のprimitive ID・実binary64時刻、各3 probeで537作用。H4-Eの追加half/undo ±0.05も含む | 全集合を保存。vector作用や分子の正しさは未検証 |
| H4-E | B2 K2/B3 K6各order0/2の4代表event群、100 control probe計画 | generation順first nonidentityの具体的component選択はfuture入力準備時。sampling0 |
| native instruction上界 | 各blockのbasis operation列とtail component specが必要 | **未固定**。旧28 wrapperの実測サイズは最大Taylor order等の構造上界を代用しない |
| 環境 | Python3.11.0rc1、numpy1.26.4、scipy1.14.1、mpmath1.3.0、qiskit1.3.0、openfermion1.6.1 | 現hostでmetadata観測した提案。旧v5のPython3.11.1記録を保存 |
| CPU/output | CPU3がaffinity内、下記relative output案は不存在 | CPU予約・割当なし、output directory作成なし |
| 科学実行 | source/inputを部分的に結合したmanifest | coverage.sealed=false、execution_plan_sealed=false、別grantなし。launcherは拒否する |

各probeは保存state、first sector column、last sector column。179はprimitiveと時刻の組数で、異なる時刻の個数そのものではない。
scheduleは実sourceのmerged ordinary、unmerged directional、negative S4、half/undoを列挙する。共通PF iteratorを使うため、独立PF schedule oracleと呼ばない。
MPの独立性はoccupation Hamiltonian構成と別精度・Taylor前進和にあり、prepared orbital decomposition全体の独立証明は含まない。

## 結果前の8 cellと記録量

| cell | method/PF | prefix | q | R/r/K | native決定論作用 | native tail作用（raw+corrected） | native / MP各精度 stage記録 |
|---|---|---:|---:|---|---:|---:|---:|
| H4_B1_S2_q1 | B1/S2 | 12 | 1 | — | 25 | 0 | 26 / 26 |
| H4_B1_S2_q4 | B1/S2 | 12 | 4 | — | 100 | 0 | 104 / 104 |
| H4_B0_q4 | B0/S2 | 6 | 4 | — | 52 | 0 | 56 / 56 |
| H4_B2_K2 | B2/S2 | 6 | 4 | 8/2/2 | 112 | 48 | 192 / 204 |
| H4_B2_K4 | B2/S2 | 6 | 4 | 8/2/4 | 112 | 80 | 224 / 204 |
| H4_B3_K6 | B3/S2 | 0 | 4 | 8/2/6 | 16 | 112 | 160 / 60 |
| H4_B1_S4_q1 | B1/global S4 | 12 | 1 | — | 73 | 0 | 74 / 74 |
| H4_B1_S4_q4 | B1/global S4 | 12 | 4 | — | 292 | 0 | 296 / 296 |

表は固定scheduleから数えた**予定量**で、実行測定値ではない。nativeはHorner内部matvecを含み、MPはrandom cellのcorrected/raw/exact_tail各pathを含む。
MPは80/120桁各一回、8 cell×2精度。中間stateの再正規化を行わない。referenceは36次元sector一枚、独立occupation全36列照合、solver0・状態再生成0。
H4-N/Aはreference差、stage state/norm、raw/corrected/logB、signed discard/PFまたはfinite/outer-PF分解を保存する。
H4-Eのphase/basis/register/ordinary・directional両branch/両axis検査は代表eventだけであり、全分子event平均の列挙ではない。
H4-Mは既存synthetic会計準備を参照し、今回の経験的差をu_boundへ昇格させない。N/G=null、accuracy UNDETERMINED、総allowance未認定を維持する。

## 固定する実行条件の案

既存H4 v2 planの上限は変更しない。資源は**提案**であり、実際のCPU割当やgrantは未確定。

| 項目 | 提案 |
|---|---|
| CPU/BLAS/worker | CPU3候補、各1。実割当・使用直前のavailability照合は別 |
| phase wall | input_reference900秒、correctness1800秒、explicit estimator（phase名wrapper_cost）300秒 |
| total / AS | 3000秒 / 8 GiB。ASはaddress-space上限で、RSS実測やメモリ予約ではない |
| output/log/diagnostics | 512 MiB / 64 KiB / 1024、terminal領域を予約 |
| reference / deterministic / tail | matvec合計10000、per-action20000、deterministic100000/cell、tail896/cell |
| primitive / control / stages | primitive2000、control200、stage4096/関数呼出し（全path合計） |
| instructions | untranspiled1,000,000。構造上界はexpanded gate数・compiler RAM・物理資源上界ではない |
| sampling / compile / solver / quantum shots | すべて0。代表eventの明示回路構築とstatevector検査はfuture H4-Eに限り別認可が必要 |
| future output | `artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v1`。今回未作成 |

retry/resumeなし。source/input/env/coverage不一致、primitive/instruction cap超過、wall/log/output/AS超過、非finite・Hermiticity/sector/phase/register不整合でSTOP。
coverage超過時はprobeを結果依存に間引かない。部分成果と失敗・未完了terminalを保存し、上限を自動増量しない。
完了は登録範囲の技術・経験的照合のみで、精度適格性・総u・shot/費用順位やH6 GOの認定ではない。全terminal後mandatory STOP。

## instruction receiptを埋める次の狭い準備scope

現在の[科学launcher](../../scripts/resource_applicability/run_track_a_ax2b_bound_v2.py)はexact `expected_bounds`との一致を要求する。
旧保存記録にruntime basis operation列とmaximum-order tail specの十分な情報はなく、metadataだけではこのfieldを埋められない。
そのため、形式だけsealed=trueへ変えたり、旧cost recordを転用したりしない。

次の明示指示に載せるscope案は**H4-P：入力をloadしnative準備のreceiptだけを作る一回作業**である。
保存入力一回load、同じ8 cellに対する既存 `_prepare` / `_prepare_discard`、`actual_bounds` / `check_bounds`を使う。
prepared basis operations・symbolic tailの構築は発生するため、今回のmetadata-only権限へ含めない。
reference/signal/MP、trajectory、wrapper build、transpile/compile、solver、Hamiltonian/state生成はこのH4-Pでも対象外。
案はCPU/BLAS/worker1、wall900秒、AS8GiB、output16MiB/log64KiB、preparation8 call、retry/resumeなし。
各blockのoperation数・数値basis識別、tail spec digest、native bounds、179組/537 probesとsource/input/envをreceiptへ結合し、生成後STOPする。
H4-Pの別grant・budget enforcement付き入口は未作成であり、現launcherのmanifest sealを迂回してH4-Pを実行するコマンドは今回提供しない。

H4-P後、receiptとCPU/outputを結合してscience manifestをsealし、**さらに別のH4限定検証launch認可**へ進む。
これは2段の未確定依存関係の提示で、いずれの認可も作っていない。H6入力生成・pilot、本検証、H8/GPUへ自動移行しない。
target/state、metric、PF/control意味論、独立性を変更する必要や正しさの矛盾が出た場合はGPTへ戻す。通常の保全的実装・schema修正はCodex範囲。

## 検査・保全・公開

[metadata専用runner](../../scripts/resource_applicability/prepare_track_a_ax2b_h4_prelaunch_v3.py)はstdlib-onlyで、`--execute`を持たない。
[専用tests](../../tests/tracks/resource_applicability/test_ax2b_h4_metadata_preflight_v3.py)12件は合成NPZ header、duplicate/object/shape/length拒否、runtime scheduleとの静的対応、認可前I/O拒否、未seal拒否、既存output保護を検査する。
numerical importを禁止し、実分子I/O・signal・sampling・回路構築はtestに含めない。local engineering evidenceであり、CIや科学的証拠ではない。
今回の実入力照合はtest結果とは別のmetadata auditである。

[別inventory](../../artifacts/resource_applicability/track_a_ax2b_h4_prelaunch_preparation/2026-10-10/preparation_inventory_v3.json)へsource、契約、metadata、12 tests receipt、保全照合を登録する。
旧178 source/freeze・科学結果/科学manifest・Track Bと既存dirty差分を保全し、対象だけをcommit/pushする。
現在は `H4_METADATA_FIXED_NATIVE_COVERAGE_UNSEALED`。準備の進展をlaunch-readyまたは科学GOへ読み替えない。
