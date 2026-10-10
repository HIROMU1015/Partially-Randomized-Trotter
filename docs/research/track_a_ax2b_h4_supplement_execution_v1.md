# Track A AX-2B：H4補完一回実行・保存監査 v1

2026-10-10 JST。`EVENT_CONTROL`と`S4_MP`は、それぞれ別grant・exclusive outputで一回実行し、
両parent/worker terminalが`H4_SUPPLEMENT_COMPLETE`となった。
保存監査は`SAVED_EXECUTION_CONSISTENCY_PASS`。**H4限定の技術的照合であり、科学的GO/STOPの最終承認ではない。**
mandatory STOP、`H6_NOT_AUTHORIZED`、`DRAFT_NOT_AUTHORIZATION`。

## 対象・実行契約・来歴

linear H4 / 1.00 Å / STO-3G / legacy DF rank12 / generation prefix 0,6,12。
8 system qubits、Nα=Nβ=2、sector dimension36、T=.8、q1/q4、δ=.8/.2。
指定saved binary64 H_DFと数学的に正規化した指定saved stateを対象とする。
新S4はB1 / prefix12、explicitはB2 / prefix6 / K2とB3 / prefix0 / K6、q4・R8・r2。
Hamiltonian再生成やground-state認定、chemical accuracyのenergy estimationを追加していない。

ユーザーの直前のH4補完手順に対する「作業を進めて」を、今回のbounded一回実行指示として記録した。
GPTレビューや準備manifestから認可を推論しない。
[GPTレビュー原文](track_a_ax2b_h4_limited_stop_independent_review_2026-10-10.md)、
[準備・synthetic検証](track_a_ax2b_h4_supplement_preparation_v1.md)、
[実行前seal/別grant索引](track_a_ax2b_h4_supplement_execution_seal_v1.md)を参照。

| 記録 | commit / repository path |
|---|---|
| 旧science source | `b228f2307f5fea77f066ee13e11ba6d2d8b9bea7` |
| 旧6 correctness / 12 MP / STOP | `a87cf25548a3b93262027ce780317872d7c4e883` / [旧保存監査](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/saved_execution_audit_v2.json) |
| 今回の実行source | `67aa6bb54dd5385eb3c56def1b6052da12643447` / [180 science・2 validation freeze](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_preparation_v1/2026-10-10/source_freeze_v1.json) |
| 実行前seal・2 grant | `5bbebb4d562ba9812eaf0838ba6419518c21d529` / [metadata preflight](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_execution_v1/2026-10-10/seal_preflight_v1.json) |
| 実行前remote bytes照合 | [prelaunch receipt](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_execution_v1/2026-10-10/prelaunch_remote_verification_v1.json) |
| 今回の結果保存 | `1c2ad5a2c0d704e4b4758fd4c07672d9f3fb3be8` / この文書・新raw・保存監査・union |
| 公開後のremote照合 | [remote receipt](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_execution_v1/2026-10-10/remote_verification_v1.json) / 新raw1212件、science180・validation2、旧science173・prep195・raw30、主要34リンクを照合 |

source/予算/精度/stage/probe条件の変更、retry/resumeは0。
CPU1・worker1・BLAS1、AS8GiB、output128MiB、log64KiB。
EVENT_CONTROL total900秒・validation300秒、S4 total2400秒・validation1800秒。
[資源観測](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_execution_v1/2026-10-10/resource_assignment_v1.json)はCPU専有やcgroup上限の保証ではない。
これらは評価の古典計算予算であり、量子回路費用C・shot N・総資源Gと区別する。

## 保存された事実

| 単位 | correctness | MP80/120 | explicit群 | parent wall | 最後のsnapshotでの観測peak RSS |
|---|---:|---:|---:|---:|---:|
| EVENT_CONTROL | 0 | 0 | 4 | 42.070871秒 | 551596032 bytes |
| S4_MP | 2 | 4 | 0 | 172.173061秒 | 421392384 bytes |

各単位でreference matvec36、179 primitive/time組×3 probe=537を完了。
全8 cellのcoverage canonical SHA-256は両単位とも
`eacf9dec340e9b356090527597cfc1a277775350f050d5b50e2bb26e3c5e9609`で、sealと一致。
independent occupation全36列差は1.832288481395828e-15、primitive最大差は1.7636654707726433e-15。
旧保存値と同じであり、総数値誤差uの証明ではない。

| S4 cell | deterministic作用 | 保存stage比較数 / precision | 最大保存state差（MP120） | corrected信号のMP80/120差 |
|---|---:|---:|---:|---:|
| q1 | 73 | 74 | 約1.55007e-14 | 約4.75965e-81 |
| q4 | 292 | 296 | 約7.42947e-14 | 約2.60546e-80 |

各cell・precisionで指数生成27（PF26＋reference1）、polynomial生成0。
hit47 / 266、lookup74 / 293を保存した。logical作用・state matrix products・scalar stageは別に記録する。
binary64 referenceとMP referenceの差は、両cell・両precisionで2.0014830212433605e-16。
これらは保存済み比較値の要約であり、新しいfit・再最適化・u認定ではない。
cache前の同一36次元S4完走値は存在しないため、実測speedupや旧完走時間との倍率比較は主張しない。

explicit 4群はordinary/directional両branch、X/Y wrapper、saved stateとsector端列の3 probeを照合。
control state actionは計100。最大wrapper/control差4.107825191113079e-15、
最大signed algebraic event作用差5.560962556441864e-16。
order0のphase +1、order2のphase -1を保存した。
両control構成とalgebraic event照合には共有basis/loweringが残る。
全分子event平均、独立の全control lowering、fresh-shot IIDの実証とは区別する。

## 一次artifact・監査入口

[保存監査](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_execution_v1/2026-10-10/saved_execution_audit_v1.json)に、
unit別terminal、calls、reference/primitive、全新raw1212件のSHA-256、欠測、stage/precision/cache統計を保存した。
rawはEVENT_CONTROL 590件・669798 bytes、S4 622件・10129042 bytes。進捗は572 / 602件。
強制STOPは発生していない。RSSは保存境界の観測値であり、独立に監視した全lifetime peakとは呼ばない。
[保存専用audit helper](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_execution_v1/2026-10-10/verify_saved_execution_v1.py)はstdlib JSON/hashのみを用いた。

| 内容 | 一次保存先 |
|---|---|
| EVENT_CONTROL parent / worker | [parent](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/event_control_launch_v1/terminal_status.json) / [worker](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/event_control_launch_v1/worker_terminal.json) |
| S4 parent / worker | [parent](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/s4_mp_launch_v1/terminal_status.json) / [worker](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/s4_mp_launch_v1/worker_terminal.json) |
| S4 q1 correctness / MP | [correctness](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/s4_mp_launch_v1/H4_B1_S4_q1_correctness.json) / [MP80](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/s4_mp_launch_v1/H4_B1_S4_q1_mp80.json) / [MP120](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/s4_mp_launch_v1/H4_B1_S4_q1_mp120.json) |
| S4 q4 correctness / MP | [correctness](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/s4_mp_launch_v1/H4_B1_S4_q4_correctness.json) / [MP80](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/s4_mp_launch_v1/H4_B1_S4_q4_mp80.json) / [MP120](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/s4_mp_launch_v1/H4_B1_S4_q4_mp120.json) |
| B2 K2 explicit | [order0](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/event_control_launch_v1/H4_B2_K2_explicit_order0.json) / [order2](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/event_control_launch_v1/H4_B2_K2_explicit_order2.json) |
| B3 K6 explicit | [order0](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/event_control_launch_v1/H4_B3_K6_explicit_order0.json) / [order2](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_validation/2026-10-10/event_control_launch_v1/H4_B3_K6_explicit_order2.json) |

[coverage union](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_execution_v1/2026-10-10/coverage_union_v1.json)は
旧6 correctness / 12 MP＋新2 correctness / 4 MP＋explicit 4群を、record path/hash/run出典で結ぶ。
欠測は今回の補完義務について0。**一回の8/8実行と記載せず、旧`H4_LIMITED_STOP`と過去の欠測記録を保持する。**
旧30 raw・今回前の2765ファイル・rootレビュー原文・source180件の保全を確認した。
他研究のdirty差分やuntrackedはstage対象に含めない。

## 残る範囲と次段階

trajectory/occurrence sampling、transpile、compileはすべて0。代表event/control回路のbuild/state actionは今回の限定認可範囲。
測定shotを実施せず、`N/G=null`、`accuracy_eligibility=UNDETERMINED`、`numerical_allowance_certified=false`。
経験的照合の完成と総u certificate、PR資源winner、rare-event期待cost、RQ-P2達成を区別する。
RQ-R主軸・RQ-P1補助・RQ-P2未達を維持する。

H6は[既存準備契約](track_a_ax2b_h6_pilot_preparation_contract_v2.md)のtol-only / 7 cell / 36 wrapper案を保持する。
backend機能の存在と実入力検証・性能確認・科学的承認を区別する。
別入力生成のscope/budget/grant、actual rank/sector/DF provenance、actual probe/instruction bounds、
source/env/CPU・H6 launch grantは、別指示で固定すべき未解決事項。
H6入力生成・pilot・本検証・H8は開始していない。別系列server run10やTrack Bに触れていない。

ここでmandatory STOP。H6の実行や科学的GO/STOPを自動認可せず、次はユーザーの指示を受ける。
GPTの判断に必要なsource・旧結果・今回結果・失敗履歴は上のcommitと相対リンクから追跡できる。
