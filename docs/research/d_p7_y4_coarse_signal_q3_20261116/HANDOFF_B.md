# Bへ：粗幅p7/Yoshida4の信号完成、Q3はSOURCE guardでUNKNOWN STOP

**元configuration `dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0`、PF要求幅32h/5について、新規6＋保存7の全13round信号が完成した。Q3はworker1回起動、科学API呼出し0件でSOURCE guard停止。粗幅MSE・総費用は未取得/null。追加投入せずSTOP。**

同じX/Y配分は `[59,55,51,47,43,39,35,31,27,23,19,7,2]`。

| 幅 | MSE下界 Ha²（保存表示） | MSE上界 Ha²（保存表示） | 判定 |
|---|---|---|---|
| 現幅16h/3 | 2.418371981954278179229700e−6 | 2.418371981954278179229706e−6 | 保存適合 |
| 粗幅32h/5 | null | null | **UNKNOWN：科学Q3 API前のSOURCE guard停止** |

現幅locator：`b4296b110b129d226dbef2743c14f725a27c3bf6:docs/research/pf6_cost_budget_q3_results_20261115/SUMMARY.json#rows[15]`。exact τ、Z1＋T1 total decoder、保存solver proxy、IID deterministic raw、全事象平方lossは元契約を引き継いだ。新信号の数値閉包やMSEはまだ検証されていない。粗幅のFAIL、不可能性、shot増による救済可否は判定していない。

## 次に再利用できる固定入力

- Configuration：`3179680ed9b4cff251b3401cabea2f6100cf47fb:docs/research/d_p7_y4_coarse_signal_q3_20261116/SIGNAL_INPUTS.json#configuration`。
- **Bの厳密intakeを通過したbound packet**：`f23c3f10ed56d0c218e49efbdbfb31d12ffa8a53:docs/research/d_p7_y4_coarse_signal_q3_20261116/BOUND_SIGNAL_INPUT.json`。
- 原13round packetと各元record：`fff936fa0cddc1ff13cb1edcf38fdde1b4a1a8b2:docs/research/d_p7_y4_coarse_signal_q3_20261116/signal_run01/`。共通binding SOURCEは51aba420b…、original numerical SOURCEは各provider_binding.input_sourceとBINDING_PROVENANCEに保持。値・normalization・bias・uncertainty・gateは不変。
- q/T/physical ID・元record locatorは [SIGNAL_PACKET_INDEX](SIGNAL_PACKET_INDEX.json)、取得/停止内訳は [RESULT_SUMMARY](RESULT_SUMMARY.json)。
- Q3凍結入力：`278fb5214f1091efba2a05b305bc05190cd18ff6:docs/research/d_p7_y4_coarse_signal_q3_20261116/Q3_INPUTS.json`。
- Q3停止結果：`a4da9fb1fc7c6de65b81d81229ae42d7cf59104c:docs/research/d_p7_y4_coarse_signal_q3_20261116/q3_run01/`。

total view `97ffe0a66f2dc14fa4b3264e885428216a2716c7a8a288085ddfe3b8a0e622ec`、binding `ef7912d2ede9f55de01c4bcf91c5c3adb935874370aaed84a9c6d48b7e9445ba` を元IDと別に保持する。

保存linear H4 / 1Å / STO-3G / tol-only DF10、B0I p7・一体項＋fragment0..6保存順、標準Y4 signed順、h_RPE=.1、P0除外。新規m6〜m11のq=[10,20,40,80,160,320]、保存stepを共有し、scalar c_Hとtail identityは元kernel順序で総Tに一度ずつ。新step/固有系/reference・sampling・回路build/compile・価格・fit・別配分は0。signal全体wall3.726370 s、peak RSS505,316 KiB。

## 停止原因・事前testの不足

Accelerator.__init__ → _kernelで `Reference total DP SOURCE changed; do not silently adapt`。科学API・数値閉包・DPは未開始で、`WORKER_PERSISTENCE_ERROR.scientific_evaluation_calls=0`。8 DP workerは設定値で、実際のDP worker起動は0。外側worker終了code0でも論理状態はUNKNOWN。

Python3.12.3のast.dumpが空のFunctionDef.type_paramsを追加するため、実AST hashはeb3150a5…、既存guardは3d043e71…となる。元710aa69bと実関数は同じPython上でAST一致し、診断コピーから空fieldを除くと期待hashに一致する。runtimeやguardは変更していない。詳細[Q3_STOP_DIAGNOSTIC](Q3_STOP_DIAGNOSTIC.json)。

module/SOURCEのpassive事前testだけではコンストラクタのAST checkを検出できていなかった。次回は実workerが使うAcceleratorコンストラクタまで事前検査する必要がある。原kernelを緩めず、互換性修正・人工同値性検査・新SOURCEと新出力先を固定した後、**別の明示的な復旧実行承認**が必要。今回Q3の未使用上限を自動転用しない。

## 費用・ロードマップ・STOP

粗幅総費用は未評価。将来適合点が得られたときの比較基準は、同公式現幅 **435,641,568 RZ** と、計画時点のp7最安 **435,481,090 RZ** を別々に保持する。後続B93ab9dbbの現幅S2別配分の395,962,678 RZは参考として別記し、今回の配分・二基準を置換しない。粗幅へ旧価格係数を流用していない。

統合担当への差分：**粗幅p7/Y4の不足6 signalは解消し、13roundのsource-bound packetが利用可能になった。新Q3適合点・価格資格・総費用は0件、Q3はUNKNOWNのまま。** ロードマップ本体は統合担当に任せる。

原結果・SOURCE・旧STOPを保全し、commit/push・remote SHA/tree/成果物照合後 **mandatory STOP**。新しい信号再取得、Q3再試行、compile、別配分、価格モデル、N/G/winnerへ進まない。
