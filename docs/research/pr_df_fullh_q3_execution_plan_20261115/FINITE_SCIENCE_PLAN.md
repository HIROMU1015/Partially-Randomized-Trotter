# 次回指示へ転記できる有限実行案（今回は未実行）

この文書は計画であり、科学実行の許可ではない。[REPORT](REPORT.md)の3候補表、[固定入力索引](CANDIDATE_INPUT_INDEX.json)と併用する。

## 対象・最大件数

- 対象：PR-DF LD1、PR-DF LD2、Full-H pure RTE κ2だけ。元/total configuration・全physical ID・A packetとnumerical SOURCEは固定入力索引の各行を正本とする。Full-H R448、Pauli、PF/p7は含めない。
- 各候補1 whole-shot vector：m0〜m12、X=`[64,60,56,52,47,43,38,34,30,25,21,17,12]`、Y=`[64,60,56,52,47,43,38,34,30,25,21,17,12]`。USER_SUPPLIED control。予定allocation IDは索引に固定し、元physical/canonical IDを変更しない。
- **新規Q3科学callは最大3件、各候補最大1回。** 直前の§10.5照合で同等結果をSOURCE付きで受理できる候補は0新規callへ減らす。既取得か判断不能な候補は保留し、上限3を埋める目的で再評価しない。
- signal/normalization/scalarの再取得・再計算、旧partial-state DP、追加vector/shot探索、MC、価格/E[C]/G/winnerは禁止。PF6の欠測資源記録・最終API returnを補うretryも含めない。

## 採択済み契約

推定器`consecutive_dyadic_nearest_branch_principal_energy_total_v1`（Z1＋T1、zero HOLD、tie決定論的argmin、元binary64演算順・principal guard）。明示targetは保存full DF10 solver energy proxy −2.1663874486347634 Ha、全事象loss `(Ehat−Eproxy)^2`。model閾値は十進εE=`0.00159360010199040 Ha`のexact二乗。表示floatを正本にしない。

保存ideal finite-K rawに条件付くshots/X/Y/round間IID二項測定、確率則`EXACT_AFFINE_OF_BINARY64_RAW`。PR-DFとFull-Hは同じmodel名でもsplit/準備/κ/event lawが異なるので、元law_bindingを全roundで保持する。correctedへ置換・normalization再乗算・scalar再適用をしない。元targetの未認証状態を変えず、別のevaluation_targetでproxy採用を明示する。

主判定はQ3直接モデルMSE。bias²/Varは診断。rawの直接riskには物理近似の影響も含み得るためδE/δE²を加算しない。旧Minkowskiは独立参照のまま。条件付きモデル結果をground認証、実sampler移送、数値certificate、物理的十分N/global RMSEへ昇格しない。

## 再利用ルートと実行前の技術準備

PR-DF：固定A bundleを`rpe_signal_statistics_caller.accept_saved`でSOURCE付き受入→元configuration→`T.create(original_source=...)`→`T.bind_signal`→`T.evaluate`。9c0e12d87の`A_FIXED_GRID_510FBACD_TOTAL_DECODER_VIEW_V1`を使い、元spec・fixed-grid・exact κ/R/signed allocation・normalizationを保持。Pauli intakeやgeneric ceil-gridの入口へ付け替えない。

Full-H κ2：保存PLAN configurationとB_SIGNAL_INPUTを読む→`T.create`→`T.bind_signal(original_packet_source=...)`→`T.evaluate`。元q/grouping/R/event数、one-body込みlawとscalarを保持。R448 readerという履歴上の参照名を理由にR448 packetを使用しない。

既存kernel710aa69b、accelerator9e294778、C c628e9e8、現B8dba096dの固定bridge/monitor修正を優先利用する。専用callerが必要なら3候補の固定locatorを明示的に消費する薄い接続だけにする。今回はcallerを実装・起動していない。

実行許可後、科学開始前に最低限の人工検証を行う：PR-DF/Full-Hの入力route、target/law/ID混在拒否、whole-vector保持、Future→候補名監視、親の監視例外時に未開始callを止めつつ開始済み結果を保存できること。**候補worker自身で原API return・partials・CPU/peak RSSを保存し、親のresource monitorが失敗しても原結果の保存に到達できる構成**を確認する。8dbaの監視修正・既存fixtureを再利用し、科学6件をテスト代わりに再実行しない。

新caller・INPUTS・実際に消費するSOURCEを科学実行前commitで固定する。今回の計画commitは次回の科学実行SOURCEではない。新namespace・候補別出力と一回性を維持し、元packet・旧結果・STOPを変更しない。

## 数値追跡範囲の提案と未確認事項

| 項目 | 提案値 | 今回の確認範囲 |
|---|---:|---|
| phase安全半径ρ_m | 全13round 1.0 rad | 過去controlと同じ値の提案。3候補の閉包を新たに検査していない |
| maximum_round_outcomes | 4,225 | 同一整数vectorに対する既存計画のcount support上限を再利用。新しい確率計算なし |
| maximum_verification_pairs | 55,043,173 | 旧count-only有限計画から転記。候補別の実際のgood支持・binary64閉包は未確認 |
| maximum_frontier_states | 非終端8,192 | 既存total DPの有限追跡案。終端状態数とは区別 |
| maximum_transition_pairs | 各構成141,559,036 | 既存同vectorの保守的有限計画値。新total状態分布の実評価・認定はしていない |

正本：4139bed3c3364f4cf2afa3451d4a956961319060の`rpe_dp_finite_scope_preparation_20261114/FINITE_SCOPE.json`、及び元control Q3 INPUTS・旧DF DP INPUTS。これらは同vectorの**範囲設定の出所**で、3候補の成功証明ではない。

次回実行前に、型・整数vector・h/M/target・既存count-only条件との一致、SOURCE/packet/bridgeの同一性を確認する。保存証拠から全countの包含又は閉包条件の同等性を立証できなければ、未確認のまま有限実行内のgateで検査する。計画だけを根拠に閉包PASSを設定しない。必要範囲が提案上限に収まると示せない場合は該当候補を開始せずSTOP。CPU/RAM/workerの恒常的上限としてこれらの科学scopeを扱わない。

Full-Hの保存減衰はPR-DFと違うため、同vectorでモデル精度を達成するとは予測しない。必要なら将来の結果を材料に別の有限配分案を作るが、今回の最大3callに追加しない。

## 数値gate・判定・保存

SOURCE/元・新ID/physical schedule/law/target/全13mean一致、finite binary64 rawの範囲、explicit evaluation_target、全roundの正確整数Nを先に検査する。Q3内ではexact二項質量、finite閉包、transition allowance、全round質量保存、principal guardと残余loss、追跡範囲、上下界整合を既存実装で検査する。旧partialの閉包診断を基礎に使っても、旧partial DPのMSEをtotalへ転用しない。

| 判定 | 条件 |
|---|---|
| MODEL_BOUND_MEETS_ADOPTED_THRESHOLD | 有効MSE上界≤τ |
| MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD | 有効MSE下界>τ |
| INCONCLUSIVE | 有効上下界がτをまたぐ。上界超過だけを物理的失敗としない |
| UNKNOWN | SOURCE/閉包/数値/質量/範囲/残余会計/保存等のgate不成立 |

coarse・first-unsafe lemma・total DPの元出力、全13round zero/tie・first-unsafe・残余mass/loss、MSE上下界、bias²/Var/moments、valid-upper min、数値allowanceを保存。有効上界は既存規則でminを取り、異なる上界を加算しない。残余を0にせず、total decoderの全eventを保持する。

実入力・canonical bridge・allocation・law・評価sourceと、完全なAPI return、worker/child CPU、peak RSSのscope・wall/stageを候補別に保存する。観測欠測はUNKNOWN/nullとし、別jobの記録から補わない。価格側へは元configuration/physical ID/total ID/m/axis/allocationを分けたSOURCE付き結果索引を渡す。N資格producerやG会計は起動しない。

## 並列化・STOP

対象候補は独立processで並列実行できる。現serverの空き資源とD/D1等の実行を観測し、既存高速backendを適切なworker/batchで利用する。過去の8workerを恒常上限にせず、共有設定・他process・worktree・venvを変更しない。D1の価格較正完了は待たない。

重大gate不成立・資源不足・保存不能を検知したら新science call投入を止め、すでに開始済みの独立jobの証拠を安全に保存する。無断retry、vector/ρ/範囲/backend変更、許容誤差緩和は禁止。数値と運用の終了statusを別々に記録する。

不足分の評価最大3件が終了、又は重大異常でSTOP。結果が適合でも追加shot配分・N探索・他方式MSE・価格/G/winnerへ進まない。`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`を維持する。

**次回指示の承認単位**は、この3候補のうち同等結果がないものについて、上記vector各1本・提案finite scope・採択済み契約によるQ3最大3callと閾値判定。今回の計画公開ではその科学callを承認・開始していない。計画公開後mandatory STOP。
