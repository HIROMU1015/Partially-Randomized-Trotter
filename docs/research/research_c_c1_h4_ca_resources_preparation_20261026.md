# H4 CDF8／保存順XDF8：共通CA・RPE shots・条件付き総資源の技術準備

2026-10-26 JST。起点 `f4de9f62c2ff1cbaaa75b6cdaa07217d00dfd957`、独立branch `codex/research-c-c1-h4-ca-resources-preparation-20261026`。

**保存full512合成profileから既存RPE／Hoeffding／energy／serial費用評価へ接続する経路を準備した。新規21＋既存回帰67＝88 unique tests PASS。実H4の新しい数値join・shots・CA・総費用評価は0。SOURCE24件・保存JSON12件・依存9版のpassive preflight PASS。最大119 logical unitsの未承認科学契約を固定し、公開照合後mandatory STOPする。**

- [H4専用の薄い接続](../../src/trottertracks/c1_h4_ca_resources.py)／[passive既定CLI](../../scripts/run_research_c_c1_h4_ca_resources.py)／[人工test](../../tests/management/test_research_c_c1_h4_ca_resources.py)
- [固定未承認119-call契約](research_c_c1_h4_ca_resources_contract_unapproved_20261026.json)／[SOURCE・人工/回帰原本・passive照合](../../artifacts/research_c_c1_h4_ca_resources/2026-10-26/technical/)
- [前段native/full512結果・数値資格の制約](research_c_c1_h4_native_fair_execution_20261026.md)

## 対象と固定条件

直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2、sector36。通常number CDF rank8と、保存OpenFermion1.6.1 eigh元順λ/G prefix8 XDF。元g補正κ・核scalar・spin/bit/frame順は旧native/PF契約のまま。全frame ordered symmetric S2、s=1／4、δ=.1、m=0..12、PR splitなし。4候補の順序はCDF:s1、CDF:s4、XDF:s1、XDF:s4。s1を主対象、s4を対照とし、cross-s探索や結果を見た条件変更は行わない。

共通CA=.00159 Ha、DF reserve CA/8=.00019875 Ha、β_total=π/6、β_stat=π/12、deterministic phase room=π/12、candidate failure=.01、axis failure=.01/26を維持。各DFのground-state oracleとalias確認済みbranch、ideal IID Born測定を仮定し、状態準備費用・実機雑音を除外する。4候補同時・事後選択した勝者の成功確率保証ではなく、登録候補ごとのfailure配分である。

native条件は両arm共通のH6 Gaussian precision／MPFR160 P0、Qiskit1.3.0・opt1・seed17・rz/sx/x/cx・approximation_degree1・qubits_initially_zero=false、既存の完全同一basis共有／fsum角度結合。今回は保存metricsを利用するだけで、compileや新Gaussian融合は行わない。head／1 h-step kernel／X/Y readoutを別compileし、all-wire fenceでserialに反復する既存policyを費用モデルに用いる。

## 再利用と必要最小限の差分

| 既存実装 | 再利用内容 |
|---|---|
| `c1_h4_native_fair` | 条件・P0・precision identity、305-call終了・full512全入出力coverageの確認。旧runnerを呼ばない |
| `c1_h6_ca_resources.metric_record` | kernel/head/readoutの非負整数metricsとmeasurement scope |
| 同 `assess_saved_round` | 既存RoundEvidence、Hoeffding solver、固定N margin検査を無変更で呼ぶ。PF・合成を一回ずつ入力 |
| 同 `serial_round_cost` | `compose_serial_cost`によるn=s2^m反復とhead/X/Y費用、両axisの費用を一単位として計上 |
| 同 `aggregate_candidate` | 既存totals、numeric capacity、候補分類、null/未資格の扱い |
| `c1_stage3_accuracy` | `AccuracyTarget`／`assess_round`／`assess_energy`／`aggregate_totals`／`compare_totals` |
| `c1_stage3_deterministic_accuracy.assess_fixed_round` | 求めた正整数shotsを変更せず、既存Hoeffding式の統計marginを別経路で確認 |
| `c1_stage3_integrated` | call journal／partial progress／STOP／資源からのworker選択 |

新規コードはH4の保存JSON schema・SOURCE・DF/PF/full512 profileの対応付け、専用passive entry、出力・並列taskの回収だけである。既存H6の入力loaderはsector400/rank14/literal-shared aliasを前提とするため再利用せず、数式・solver・serial費用関数のみ再利用した。新しいfitter、simulator、compiler、shots式・集計器は作っていない。旧SOURCE・関数・tests・科学契約への変更は0。

## 誤差・shots・CAの接続

1. 保存済みの各armのDF **norm＋元guardの合計**をHaで再利用し、native identityと元の保存recordとの一致を確認する。両armの値は異なる。新DF normやguardは計算しない。
2. 保存52 profileからPF operator誤差と合成complex signal上界pilotを各一回入力する。`fsum(PF,synthesis)==saved combined`、`asin(combined)==saved phase`を既存helperで検査する。combinedをPFとして渡して合成を再加算しない。
3. 決定論モデルではattenuation1、RTE補正誤差0、law discrepancy0。これは登録した決定論lawの定義であり、未知の合成・reference丸めを0と認定する処理ではない。
4. 既存solverはρ_lower=1−combined、β_det=asin(combined)、coordinate=ρ_lower sin(β_stat)/√2からHoeffding十分shotsを求める。各roundでX/Y対称のNを保存し、固定N marginも検査する。新precisionを反映しない旧CDF shots・旧XDF10 shotsは読み込まない。
5. 全13 round、failure配分、DF reserveが適格なときだけ、既存`assess_energy`が`DF_norm_plus_guard + β_total/(δ2^M)`を共通CAと比較する。DF誤差はHaへ変換した後に一度だけ加える。PF/合成signalをHaに直接足さない。

結果の`NUMERICAL_PILOT_CA_PASS`は、保存level B誤差を前提とする**条件付き数値pilot**である。full512全基底coverageは有限probeの代用ではないが、CAR/expm・Operator復元・norm・旧PF誤差の丸めenclosureは未資格のまま。厳密CA保証へ昇格しない。

境界的なshots数をfloatから黙ってclipしない。既存方針通り、overflow・coordinate-square underflow・N≥2^53・固定N marginの未資格は`UNRESOLVED`、可能なら未資格Nのdecimalを保持する。Nの事後増加・再配分・別solver fallbackは行わない。

## 条件付き費用とsupport

全13 roundのenergy条件を満たした候補だけ、各axisで

`cost(m,axis) = (s2^m) × saved_kernel_metrics + saved_head_metrics + saved_axis_readout_metrics`

を既存関数へ渡し、`Σ_m [N_X cost_X + N_Y cost_Y]`を集計する。主指標は条件付き期待総RZ数、副指標はRZ depth・CX count/depth・total depth・circuit size・executions・measurements・shots。readoutのsize/depthは既に測定を含むため再加算しない。

決定論の固定回路を明示したserial policyで反復するため、このモデル内の費用supportはsingleton（lower=mean=upper）である。これは回路countの算術的な確定性であり、精度enclosureや実機・全体再compile最小depthの保証ではない。depthは`scheduled_serial_sum`、全体を新たにcompileした`circuit.depth()`ではない。s4のkernelがs1と同countでも反復数は4倍であり、反復を落とさない。

既存binary64集計は、finiteかつ整数値で2^53未満という数値資格を要求する。overflow・coverage不足なら`UNRESOLVED`としてtotals=null。未知誤差や未取得費用を0にしない。CDF/XDFとも同じsで比較し、単一kernelの約11.99%削減からshots加重総費用の優位を先に結論しない。

## 固定入力・SOURCE・実行順序

科学用SOURCE manifestは24件、読み込む保存証拠は11 JSON＋旧SOURCE manifestの12件。旧native公開commitは`f4de9f62c2ff1cbaaa75b6cdaa07217d00dfd957`、旧科学SOURCEは`862a97d4213121749448ca716cde820a44b99772`、その契約SHA256は`4a99e76ea879226e077376dd69f12b529f9eb1a48e98dd048143892a59b0e578`。原P-H4 SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`、CDF fit SOURCE・export、元順XDF8のDF/PF SOURCEも旧固定契約から独立に継承し、旧入力・行列・QPYはdecodeしない。

- 新SOURCE manifest SHA256：`fc03dae7a2db264d4aa4a92abf0ff0e6c812dd6b25879afd8cddfd56d299d521`
- 未承認119-call契約SHA256：`7540af39daec4e1584f2805a8338c4ab91d2042923d83830dcaef183f5fb4edb`

| 科学call分類 | 最大 |
|---|---:|
| 保存bundle読込・終了状態/SOURCE/alias/horizon確認 | 1 |
| 4候補の証拠attachment | 4 |
| 4×13 round accuracy・shots・固定N確認（X/Y一組） | 52 |
| 候補別energy判定 | 4 |
| 適格候補の13 round serial費用（X/Y一組） | 52 |
| 候補別totals/classification | 4 |
| 同s CDF/XDF比較 | 2 |
| **合計** | **119** |

SOURCE/input/environmentのpassive照合→保存305-call完了/full512 coverageとDF/PF/ground来歴確認→4 packet→候補ごとに13 round／energy／適格時だけserial費用／totals→同s比較2件→証拠公開・remote照合→mandatory STOP。

最大52 accuracy record、X/Y shots104値、適格時のper-shot cost104行、候補4件・同s比較2件を得る。不適格候補では13 cost unitsをskipし、残枠を別処理へ転用しない。`INFEASIBLE_REGISTERED_SUFFICIENT_CONDITION`は登録十分条件の未達であり、物理的不可能性を意味しない。未知bound／数値範囲の未資格は`UNRESOLVED`で、他の登録候補は継続する。

SOURCE・入力・schema・非finite・candidate/round・PF/合成の二重計上・alias・metric/readoutのcorrectness不一致では新しい依存callを停止し、開始済みtaskを回収して部分証拠・first traceback・STOPを保存する。fresh `<execution-date>/run01`一回のみ、retry/resume0、条件変更・追加候補・fallback禁止。preflight不一致は科学call前に停止し、fresh science outputも作らない。

## 人工・回帰検証と環境

**新21＋既存67＝88 unique tests PASS**。新21は人工JSONのみのend-to-end、成功119 units、不適格/未資格106 units・null totals・他候補継続、SOURCE/alias/metric/DF/coverage/PF/signal不一致でSTOP、no-call-after-FAIL、fresh再利用拒否、反復16384とaxis/head算定、solver/energy/集計の整合、数値範囲、実byte preflightのpassive性を確認した。旧native/旧H6 runnerはmockで遮断した。

既存回帰67はH6 scalar CA/resources、決定論fixed-N accuracy、Stage3 accuracy/serial接続。sourceは無変更で、既存式・未資格/overflow処理・候補別failure配分を再利用する検証である。人工PASSを実H4のCA達成や費用結果とは扱わない。

初回人工batchは17 FAIL／4 PASS。Git metadata用の広い`subprocess.check_output` mockがMatplotlibのfont確認にも干渉したことと、異常contract fixtureが既存のexclusive-saveへ再書込しようとしたことが原因。task-localなGit metadata helperだけのmockと人工fixtureの書込へ修正し、新21件は全PASS。初回原本はgzipで保全した。科学的FAILや旧SOURCEの修正ではない。

旧protected62件・raw109件をbyte/hashだけで再照合し不変。独立worktree、read-only既存Python3.11.1環境のliteral path `/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python`と旧依存lock／固定9版を再利用。actual passive preflightはNumPy/SciPy/Qiskit/ffsim未import、科学call0、science run01なし。Python bytecode出力を無効にし、実行時cache/tempは当task内へ分離する。

scalar JSON／log／ceil／整数算術のみで、行列・compile負荷はない。資源snapshotはaffinity32 CPU／空きRAM約52.8 GiB。4候補を既存resource_plan＋ThreadPoolExecutorで独立実行し、科学実行時に実資源を再確認する。scalar経路にBLAS数値kernelはなく、BLAS上限は設定しない。worker4は候補数に由来し、CPU/RAM/wallの恣意的capではない。共有環境・venv/cache・他ユーザーjobは変更しない。

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python \
  scripts/run_research_c_c1_h4_ca_resources.py \
  --contract docs/research/research_c_c1_h4_ca_resources_contract_unapproved_20261026.json
```

これはpassive確認のみ。別明示科学許可を受けた場合だけ`--execute --output artifacts/research_c_c1_h4_ca_resources/<execution-date>/run01`を追加する。承認証明書・承認原文保存・新grantは不要。

## 未資格事項・次の担当・mandatory STOP

今回の技術接続に残るblockerはない。実H4での新shots・CA適格性・総費用・同s比較の結論は未取得。丸めenclosure・strict CA、実機雑音、状態準備込み・無条件の資源優位、4候補同時／事後選択保証は未資格のままである。

次は**ユーザー／研究C**が固定公開SOURCEと119-call契約を確認し、保存値の限定数値再解析を別途明示許可するか判断する。許可後に**Codex**がSOURCE/input/環境を再照合し、fresh run01で一度実行し、全候補・shots/energy/費用・call/STOPを公開する。詳細科学レビューは今回実施せず、必要なら別許可とする。

**今回の実H4科学call・fit・operator/PF/合成再計算・build/compile・shots算定・CA・総費用の新評価は0。commit・push・remote照合後mandatory STOPし、新しい科学計算へ自動進行しない。**
