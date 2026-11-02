# H2/H4/H6高次PF：保存値統計・総費用評価の実行前STOP

2026-10-21。**`STOP_BEFORE_SCIENCE / SHOT_POLICY_NOT_ADOPTED_FOR_OUTER32`**。今回の利用者指示による保存値数値解析のone-shot実行承認は受領済みである。ただし同じ指示の「必要な規則が未確定なら独断で決めずSTOP」に該当したため、科学数値解析を開始していない。実行承認がないことを停止理由にしているのではない。

## 1. 固定証拠と照合結果

| 系 | 固定結果commit | 科学SOURCE | 探索trajectory/軸 | 今回のmetadata照合 |
| --- | --- | --- | ---: | --- |
| H2 | `a05f4ed19e87e1eb65f7fc346385b5b4eae69845` | `4ec26921c6872ed6e8667f167074e1e405cf1a3f` | 525 / 1050 | PASS |
| H4 | `605ca07a653241db49348fbd7f4e091f9f18ced7` | `5a448d94ea389ada6e2b2755de6a215ed2034679` | 793 / 1586 | PASS |
| H6 | 同上 | 同上 | 607 / 1214 | PASS |

[remote refs](../../artifacts/resource_applicability/track_a_h246_saved_evaluation_preflight/2026-10-21/remote_refs.json)で両公開branchのHEADが指定commitと一致することを読取確認した。Git objectから固定summary・SOURCE blobを取り出してprivate原本へ結合した。全3系とも12群、S2/Yoshida4/m5_best×q1/2/4/8、K4/outer32、cosine/sine paired、EXPLORATION stageである。H2 rank3/prefix1、H4 rank10/prefix3、H6 rank19/prefix5、直鎖1 Å/STO-3G/tol-only1e-8、T=.8/P=0を変更しない。

H2 70/H4 72/H6 72のSOURCE Git blob、各2012/2146/2146の入力・runtime・SOURCE seal、525/793/607の保存result JSON SHA、plan・終了witness・終了監査SHA・native exit0/worker reapを照合した。系ごとのmetadata照合を3 threadで同時に実行し、同じfileのhash結果はcacheで共有した。原結果の数値fieldは評価せず、JSONのbyte SHAだけを確認した。QPY/NPZはdecode・simulation・再compileしていない。archive復元はSHA結合した既存終了監査PASSを参照しており、今回archive全体を再復元したとは主張しない。

H2 design pilot396本、H4/H6の既存pilot、旧探索・確認標本は今回の探索標本へ混合しない。旧STOPもそのまま保持する。

## 2. STOPに至った規則

[費用取得準備§7](track_a_h46_m5_cost_sampling_readiness_20261020.md)は「将来の会計前に、軸別ε/√2、α、normalization、bias・u・log margin、headroom guard、shot整数式、経験的allowanceの扱いを原SOURCEと照合して別途正式採択」と明示している。

[付属プロトコル§5/9](track_a_h46_higher_pf_supplementary_protocol_v1.md)では保存allowanceを経験的適格性の診断とし、正式shot政策は未採択。[探索取得契約](track_a_h46_pr_exploration_execution_20261021.md)も正式shot政策を取得範囲へ含めていない。H4/H6の封印config/planには`decisions.formal_shot_policy_adopted=false`、H2の保存signalにも`formal_shot_policy_adopted=false`がある。

[共通正式v2§3/6](track_a_pr_resource_evaluation_protocol_v2.md)の旧α/shot規則は当該保存会計SOURCEに結合して継承されている。新outer32系列へ自動移送したり、今回の解析承認だけで値が選択されたことにしない。v2の高次PF付属参照も、経験的allowanceを正式shot政策へ変更しないと明記する。AD03の主比較採用保留とは別の判断であり、AD03保留それ自体を今回の探索費用評価の禁止理由にはしない。

不足は新しいsignalや回路ではなく、**保存経験的u・log marginと凍結shot算術を、この新系列の条件付き費用診断に採用するという判断**である。

## 3. 採択案（未採択・未実行）

既存値と実装を最小限再利用する案を[採択判断JSON](../../artifacts/resource_applicability/track_a_h246_saved_evaluation_preflight/2026-10-21/policy_adoption_required.json)へ固定した。

- 精度は保存5点 .05/.01/.005/.001/.0005、軸精度ε/√2、α_axis=.025。
- 主会計は通常signed biasの軸別絶対値、保存経験的uの倍率1、保存log_Bに保存log_B_marginを加えた経験的normalization診断。
- h=ε/√2−|bias|−u、両軸h>0かつu/h≤.01を要求する。INELIGIBLE/UNDETERMINEDや欠測を0費用へ変換しない。
- `d0_frozen_shot_rule_v1.py:u_aware_shots`の凍結整数算術と演算順を使用する。式は `max(1, ceil(2 B²/h² log(2/α_axis)))`。安全な整数域を外れる場合は元SOURCEのLOG_DOMAIN_ONLYとnullを保持する。
- 参照誤差を0と認定せず、strict bias/reference certificateはnull。compiled状態差の1e-7受入閾値をshot allowanceへ流用しない。
- 非相殺は必要な保存成分とclosureが揃う場合の別診断とし、成分絶対値和をbiasへ使う。同じ宣言済み算術を使う案であり、不足ならnullのままにする。

これは科学条件の自動採択ではない。利用者が採択または代替規則を指定した後、今回受領済みの解析承認範囲を維持した新契約・SOURCE・入力seal・科学call予算・有効one-shot grantを作る。[解析契約DRAFT](../../artifacts/resource_applicability/track_a_h246_saved_evaluation_preflight/2026-10-21/analysis_contract_DRAFT.json)と[grant未発行記録](../../artifacts/resource_applicability/track_a_h246_saved_evaluation_preflight/2026-10-21/grant_NOT_ISSUED.json)は実行不能な記録である。今回有効な科学grant・ledger・消費markerは発行/作成していないため、実科学解析の一回枠は消費していない。

## 4. 今回の実行範囲と検査

実施したのは固定Git/metadata/SHA調査、read-only preflight、人工検査、STOP記録と文書整理である。preflight SOURCEは`835ef47c4230c29cb48b13d3497ee3339666a89f`。新sampling/signal/simulation/回路生成/compile/統計/shot/総費用/順位は全て0件。有効な数値解析grantは未発行で、使用済みgrantの再利用もない。主要結果の独立数値照合は`NOT_EXECUTED_PRE_SCIENCE_STOP`であり、作者自身のmetadata照合を独立数値検証と呼ばない。

人工5件がPASS。未採択・不完全なpolicy、旧系列だけのpolicy、pilot混入、不完全な候補集合、片軸・旧law混入を拒否する。[preflight receipt](../../artifacts/resource_applicability/track_a_h246_saved_evaluation_preflight/2026-10-21/preflight.json)は全metadata PASS後のpolicy不成立をexit2で記録している。科学計算失敗や数値FAILによるSTOPではない。

## 5. 総費用比較の現状と次の担当

| 系 | S2/Yoshida4/m5の総費用・改善率・最良候補 | 独立確認の推奨対象 |
| --- | --- | --- |
| H2 | 未評価、null | 未選定、null |
| H4 | 未評価、null | 未選定、null |
| H6 | 未評価、null | 未選定、null |

[evaluation_status_3.csv](../../artifacts/resource_applicability/track_a_h246_saved_evaluation_preflight/2026-10-21/evaluation_status_3.csv)も未評価をnullで保持する。4次PFが改善した/しなかった条件を、未計算のまま断定しない。単価だけの探索順位・旧pilotの順位を、新標本のshot込み順位の代わりに使わない。

**次の担当：ユーザー**が§3のshot/allowance政策をこのouter32保存値解析に限って条件付き採択するか判断する。解析の実行承認自体は既に受領済みで、同じ一回実行承認を重ねて要求しているのではない。

**次の担当：Codex**は政策判断後、その規則に結合した実行準備・独立数値経路の人工検証・新one-shot解析を行う。今回の制約に従い、統計評価、paired SE、shot/費用分解、固定集合内の比較、有限標本・希少層・対照identityの制限を記録する。新sampling/compile/独立確認P/候補拡張/AD03採択へ進まない。

ローカルcommit後、**mandatory STOP**。公開pushは行わない。
