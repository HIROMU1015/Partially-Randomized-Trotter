# 4方式RMSE比較：有限menuの不足証拠を特定する実行準備

2026-11-13。B基点`6a9d7fd36b687e58aecbed1278c6d7927d6827d6`の既存branchを継続。**ユーザーが将来固定する有限V2 menuから、各configuration/round/axisのN/C/SOURCE・誤差契約の不足とA/Cの担当を列挙する薄いplannerを追加した。新13＋関連回帰43＝56技術test PASS。実menuは未確定、実N/C/G・数値探索は0。**

## 前回からの差分

前回118候補/236軸の保存inventoryは[既存coverage](../pr_rpe_method_coverage_20261113/COVERAGE.json)に保全した。今回はこれを再列挙したり新共通κ scheduleへ認定したりせず、**新たに指定されるwhole-schedule V2 configuration**を入力とする。全体RMSE契約・同一source/law/price policyが足りない箇所を、次のA/C処理へ対応付ける。

既存`pr_rpe_rmse_controller.generate/assess/coverage/select`、A固定grid adapter、Cの受動price record形式を再利用。新module`pr_rpe_rmse_evidence`とJSON CLIだけを追加し、controller/既存provider/科学SOURCEは変更していない。`select`へ渡すassessmentのcostは必ずnullで、比較条件の検査だけに使う。N/C計算、G積和、費用最小候補選択、cache-miss起動はない。

## 4方式の必要証拠

| 方式 | 全roundに必要なA側証拠 | 全round/axisに必要なC側証拠 | Bが保持する区別 |
|---|---|---|---|
| Full PF | 元Hの全保持、formula/要求幅/実幅/q、state/prior/decoder、PF/統計を含む全RMSE契約と十分N | 同formula・時間・全保持Hのcontrolled full wrapper価格、SOURCE/profile。保存direct適合なら再compile不要 | κ/Kなし。H4 B0/p7とは別identity。正式baseline探索範囲はユーザー判断 |
| Truncated PF | 保持Hと元Hとの差を含む全体RMSE資格、十分N | 保持H/phase/scalar・formula/q・wrapperの価格 | 元H誤差をBで生成しない。旧B0のHP NだけでRMSE PASSにしない |
| PR | 共通κからの全schedule、H6早期override、finite-K/signed allocation/law/normalizationと全round RMSE十分N | 同一物理requestの平均compiled C、cohort又は全lawの意味、domain/長q/大R資格 | 異κのround混成、C1 falseの昇格、小q→長q補完をしない |
| Full-H RTE | 全非scalar H・一体項・scalar・λ/law/normalization/event単位のsource bridge、Aのmean/N/RMSE評価 | 同source/preparation/event groupingのfull-wrapper価格・法則/平均/domain証拠 | DF B3は別方式。旧qRの信号同値性から価格同値性を仮定しない |

共通条件は元H/sector/state/snapshot/decomposition/SOURCE、RMSE target/policy、h/prior/decoder、compiler/control/measurement/full-wrapper価格単位/P0。方式ごとの設計変数の個数は揃えない。Mが異なるなら各全scheduleのRMSE証拠が必要。`formula_menu`等の既存明示機構を使い、正式探索範囲をBが採択しない。

## Full PFと純RTEの具体的SOURCE調査

[SOURCE_CONTRACTS](SOURCE_CONTRACTS.md)に関数・意味・欠測・担当を整理した。

- S2、標準Yoshida4、m5_bestの既存H4/H6 Full PF作用/controlled wrapper経路を特定。8次Morales/Yoshidaの係数・展開・DF/Pauli state-action/局所costも存在するが、現B V2では8次や標準4thを生成しない。高次PFの存在を同条件のRMSE/全wrapper資格とはしない。
- DF Full-H RTEの既存`prepare_full_h`は一体項Z成分をtailへ追加し、deterministic blocksを空にする。元scalar/λ/law/preparationと保存stateを固定する必要がある。Pauli L_D0は全非identity項で、constant保持。両表現・旧stateは相互alias不可。
- T2.0後続SOURCEにあるs=qRのmean共有は価格同値の主張ではない。新plannerはsource-bound mapping全体を出し、同event数でも旧groupingが違うmock requestを区別した。物理event/N/normalization/C評価器をBで重複実装していない。

## 新plannerの出力と利用条件

各round/axisに、physical候補とB全identity、configuration/source_candidate ID、law/mapping、N/C存在・値存在・資格・source、要求SOURCE拘束、normalization/RMSE契約、保存再利用条件、全schedule障害、A/C next_workを出す。詳しいI/Oは[HANDOFF](HANDOFF.md)、未選択templateは[MENU_TEMPLATE](MENU_TEMPLATE.json)。

保存N/Cがあっても完全V2 identityとlaw/sourceが異なれば不適合。外国price recordは出典と理由を残し、自動relabelしない。追加指定SOURCEと不一致なら全体ready=false。局所二時点packetやnominal Nは未評価round/null boundのまま、構造Sも価格へ昇格しない。

`evidence_menu_resolved`、要求方式ごとのready、有効な部分集合と未探索範囲を分ける。適格なINFEASIBLE候補へ不要なC取得を提案せず、N SOURCEが不一致のINFEASIBLEでは除外しない。証拠が全て揃った人工menuでもG/winner=null・比較未実行のまま。これは全探索済みの実最適値ではない。

## 最新A/Cの参照結果

| 固定成果 | 現在の意味 | 次の必要作業 |
|---|---|---|
| A `ee15bbd29ec571dd2526feecdb81ed757f783557` | 二時点pilotの技術準備。B f09/V1参照、実M/κ/予算未選択、十分N/全体bound=null | AがV2 identityへ追従。ユーザーが固定条件/誤差契約を判断後、保存mean/referenceを優先する有限pilot案へ。局所結果は全round証拠にしない |
| C `a11efd130e9dfe7086b88386570594ebe54d46af` | 公開REPORTでS0a run02の保存4path×XY8件parity完了。mean compiled C等null、モデル資格false | CがV2価格packet、長q/大R/law平均・境界残差の有限検証案。BはS再計数/残差再解析せず、S0aを平均Cにしない |

A/Cの新しい科学pipeline、branch/worktree/process、SOURCE/input/raw/cache/results/STOPへ介入せず、公開commitのcode/reportを読取参照した。無条件mergeは不要で、B旧A/C moduleも不変。

## 技術テストと限界

[TESTS.json](TESTS.json)／各log、[ARTIFICIAL_SUMMARY](ARTIFICIAL_SUMMARY.json)。新13件は4方式、Full/truncated、B3、κ/H6初期例外、同一H/sector/state/RMSE/profile、V1拒否、局所二時点、存在/資格、部分集合、SOURCE、不適格/欠測、旧grouping、科学非起動・fresh CLIを検査。関連回帰は既存method coverage14、controller17、A/B/C integration12の43件、計56 PASS。過去99件を再実装していない。

これは人工/metadata技術検証であり、科学的なRMSE成立や価格精度・優位性の証拠ではない。実Hamiltonian新signal/N/RMSE、保存科学値の再解析、S/C計数、sampling、build/compile/transpile、fit、実G/最適化、実M/κ/幅のmenu生成は0。

## 何が決まれば実Gへ進めるか

**科学的判断**：ユーザーによる元H/stateとRMSE/error契約、比較方式・有限PF formula/幅/分割/M/共通κと価格policy/SOURCEの固定。

**技術接続**：A/CのV2 packet/物理request bridge追従、保存値の完全identity/source/law/domain照合。ここで新科学計算を増やさず、省略可能なmean/reference/compileを先に特定する。

**追加科学証拠**：Aの局所pilotと、必要なら選択全roundのRMSE十分N・全体decoder/error資格。Cの不足domainだけのfull-wrapper平均価格/全law・長q/大R資格と必要な不確かさ。これらの対象・SOURCE・有限上限と実G/比較範囲が明示承認された後、B既存cost/selectを使用できる。現段階では数値menu未確定のため新しい実験条件やcompile数を推定していない。

担当・依存関係・有限科学scope案は[HANDOFF](HANDOFF.md)。commit/push・remote SHA/主要成果物を照合後、**mandatory STOP**。新science、A pilot/C次価格検証、実G最適化へ自動進行しない。
