# 有限V2 menu → 不足証拠一覧：A/Cへの引渡し

基点B`6a9d7fd36b687e58aecbed1278c6d7927d6827d6`。新APIは`pr_rpe_rmse_evidence.plan`、CLIは`plan_pr_rpe_rmse_evidence.py`。既存generate/assess/selectを再利用し、N/C provider、cost、実最適化は呼ばない。

## 入力

将来ユーザーが物理条件と有限menuを固定した後、既存`B.generate`又は`B.generate(I.a_spec(...))`が返す**V2 configuration全体**を渡す。planner自身はM/κ/幅/λ/予算を選ばず、configを新規生成しない。118旧候補やV1 packetを全scheduleとして受け付けない。[MENU_TEMPLATE](MENU_TEMPLATE.json)は実候補が未選択のためentries空であり、そのままでは実行できない。

```python
from trottertracks.pr_rpe_rmse_evidence import plan

entry = dict(
    configuration=fixed_V2_configuration,
    N_packet=supplied_A_packet_or_None,
    prices=supplied_C_records,
    cohorts=[dict(candidate_id=cid, axis=axis, cohort=selected_cohort)],
    source_requirements=[dict(m=m, axis=axis, N=expected_A_source, C=expected_C_source)],
    diagnostic_references=source_only_diagnostic_references,
)
report = plan(user_entries, price_policy=user_price_policy,
              required_methods=user_required_methods,
              formula_menu=user_formula_menu,
              unexplored_scope=user_unexplored_scope)
```

`cohorts`は各candidate/axisで明示選択し、重複・外国candidate/axisを拒否する。`source_requirements`は任意のper-round/axis追加拘束で、SOURCE objectを完全一致比較する。省略したSOURCEはNOT_SPECIFIEDのまま。A/Cの数値module・結果SOURCEを未決定なのにBが補完しない。source constraintとは独立に、既存assessのfull identity/law/measurement/profile/source/model/RMSE検査は必ず行う。

```bash
python3 -B scripts/plan_pr_rpe_rmse_evidence.py --menu USER_FIXED_MENU.json --out NEW_PLAN.json
```

CLIに科学実行flag・default候補・provider/cache miss・計算fallbackはない。出力はfresh pathのみ。

## 全round/axis出力の読み方

| 出力 | 意味と次の担当 |
|---|---|
| configuration_id、M/common_kappa/schedule_times | 全scheduleの拘束。Bがκの異なるroundを混ぜない |
| identity、physical_candidate、source_candidate_id、physical_schedule_mapping | A由来physical IDとB requestを区別。mappingの旧grouping等も保持。完全一致が必要 |
| N/C.availability/value_status/qualification/source/reasons | 既存assessの状態。record存在と値存在、科学資格を分離 |
| required_source/source_constraint、source_issues | 追加のSOURCE pin。元assessがAVAILABLEでもこれが不一致ならready=false。Nのsource_candidate_idが外側に重複宣言されていればnested identityと照合 |
| normalization、error_contract | finite-K/signedκ・元policy/SOURCE・boundの存在・全round coverage。Bはnormalizationを新評価せず、独立資格を付けない |
| reuse_conditions、unmatched_prices | full V2/law/source/measurement/cohort等の保存再利用条件。外国recordをsource付きで列挙しrelabelしない |
| whole_schedule_blockers、cost_evaluation_possible | 1roundだけのPASSで全Gへ進めない。全体のN/C/SOURCE資格が必要。可能表示もG実行ではない |
| next_work | Aの保存mean/reference/全体RMSE確認、Cの保存full-wrapper価格/law/domain確認。自動起動なし。今回のtaskでは科学実行未承認 |
| diagnostic_references | 診断の出典だけ。N/Cへ昇格しない |

`evidence_menu_resolved`は指定候補がready又は適切な証拠でINFEASIBLEになったか、`requested_methods_have_ready_evidence`は要求4方式等にready候補があるかを別に示す。部分coverageは`AVAILABLE_QUALIFIED_SUBSET`、未探索範囲は入力のまま。全候補の証拠が揃っても`comparison_complete=false`、`cost_selection_scope=NOT_EVALUATED`、G/winner=nullで、費用順位は出さない。厳密に出典が対応したINFEASIBLEには不要なC取得作業を提案しない。外国N SOURCEのINFEASIBLEで候補を除外しない。

`next_work`は各軸の証拠不足であり、科学call数ではない。Aは同一物理roundのmean/referenceから両軸を供給できる場合に共有し、Cは同じpath/cohortのX/Yを対応付ける。これを別々のsampling jobへ機械的に重複変換しない。古典cohort標本数やnormalizationをshot/価格へ二重乗算しない。plannerは実行許可の認証・発行・判定を行わず、ユーザーの明示許可はチャットのscopeで扱う。

## Aの現在成果と次の受入条件

A最新`ee15bbd29ec571dd2526feecdb81ed757f783557`の二時点準備はB`f09af824...`/V1をASTで参照し、実条件未選択・全十分N/bound=null。今回そのmoduleをBへmerge/改変せず参照した。**Aの次の技術作業**はV2 assembly/a_spec/identityへpinを追従し、元physical candidateとlawを照合すること。単にSCHEMA文字列をV2に変更して過去IDを移植しない。

将来のA packetはconfiguration ID、M/共通κを含む全identity、source_candidate_id、元SOURCE/law/state/prior/decoder、対象roundを保持する。二時点診断では対象外roundのN_X/N_Yと全schedule RMSE boundはnullのまま。nominal Nや旧HP Nを十分Nにしない。新plannerはこのpacketを未資格Nとして表示できるが、Aの数値解析を重複実装しない。

## Cの現在成果と次の受入条件

C公開`a11efd130e9dfe7086b88386570594ebe54d46af`、実行SOURCE`9516351e639278349acf55334e838520a7b3833e`のS0a run02は保存4path×XY8件の構造S parity。元REPORTを参照し、今回BでSを再計数・比較し直していない。Cのmean_compiled_price/physical_state_closure/full_rpe_resourcesはnull、qualified_for_candidate=falseのまま。

**Cの次の技術作業**はV2 request/whole-schedule contextに対応した価格packetと、長q/大Rの有限検証案。受入には同じcandidate、axis、law、compiler/control/wrapper/P0、SOURCE/cohort、DIRECT/MODEL、finite cohort/全law、domain資格、不確かさが必要。S0aをDIRECT/MODEL Cへ変換しない。C1 MODEL false、既知のpaired calibration group、未知joint/cross-configuration covariance/CIをそのまま保持する。

## 実Gへ進むための依存関係（未承認案）

1. **ユーザー**：元H/sector/state/snapshot、target RMSE、h/prior/decoder、誤差契約、各方式の有限formula/幅/M/κ候補、価格policy、SOURCE/profileと未探索範囲を決める。Full PFを適切に含め、H4 B0をFullとしない。8次や純RTEの新物理仮定を技術実装で採択しない。
2. **A/C/Bの技術接続**：選んだSOURCEでV2の全physical requestを固定。Full-H RTEの一体項/λ/normalization/law/event grouping、PF高次signed scheduleのbridgeを[物理SOURCE契約](SOURCE_CONTRACTS.md)に従い確認。Bはmenu→本plannerで全round不足を出す。この段階に新scienceは不要。
3. **Aの科学scope**：まず既存二時点案を用い、完全一致した保存mean/referenceなら許可付き再解析だけに限定。不足時だけその固定2時点のmean/内部対照を取得し、必要pairだけreferenceを追加する。A既存案の上限はmean2/内部対照2/reference action最大4であり、これを新しい全round計算許可としない。局所pilot後の十分N・全schedule契約は別の有限scope。
4. **Cの科学scope**：同じ選択physical候補で保存full-wrapper direct/cohort/modelを先に照合。同一source/domainの適格保存値があればcompile省略。欠測のdomain/長q/大R/law平均・希少event・C−S残差/独立validationだけをCが具体的な有限セル/標本数で計画する。Bは標本数や新しい平均モデルを先回り採択しない。S0aは完了済みで再実行不要。
5. **A/C**：選択全roundの十分N・元H/decoderを含む全体RMSE証拠、適格full-wrapper Cを供給。未知共分散等は不明のまま、点費用/CIのどちらを主張できるか分ける。**ユーザー**が具体的SOURCE・全候補/軸・有限計算数と実G/比較の範囲を明示許可した後にのみ、**B**が既存cost/selectへ進む。

数値menuが未確定なので、今回Bは実験候補数・実M/κ/RMSE予算/PF幅、必要compile総数を数値確定していない。上記技術追従と科学的判断・科学実行承認は別であり、承認証明書/新grant機構は不要。**今回commit/push・remote照合後mandatory STOP。**
