# 初回2評価の条件一覧・未採択案

2026-11-13 JST。[REPORT](REPORT.md)／[固定SOURCE・各roundの参照](SOURCE_AND_CONDITIONS.json)。本書は科学実行の準備であり、下記の推奨値・契約は今回採択していない。元A bundleのtarget・law・資格flagを変更しない。

## 固定条件は再利用する

- 対象はA91349のPR-LD1/LD2、元H4 DF10・保存H/state/snapshot、full DF-H energy task、全13round。S2/元ΔPF・h=.1/K2/M12/各LD・κ・signed pieces・event数・prior/decoder/alias/shift0は元canonical SOURCEどおり。
- 元Minkowski budgetとεEは保全する。保存policyの科学資格がPROPOSED_NOT_ADOPTEDであることと、元条件を固定していることを区別する。資格flagを書換えてPASSを作らない。
- RPE Monte Carloは主評価から省く。PF Trotter誤差の主評価にH_eff固有値シフトを使う利用者方針を保持する。これらを再度の選択事項としない。
- P=0、full Hadamard wrapper/測定軸・compiler/profile、価格種別/law/cohortの扱いは既存どおり。今回Gは対象外。

## 未採択の条件と推奨案

| 判断事項 | 推奨する初回診断案 | 意味・制約 |
|---|---|---|
| evaluation_target | 保存`reference_solver_energy_ha=-2.1663874486347634 Ha`を**条件付きproxy**として別入力に採用 | A bundleの引用。bba150834ca30bdb186fbfd58c55a8d51177b855:PHYSICAL_INPUTS.jsonのtarget、元truth_proxy_sourceへ追跡できる。厳密ground包含・資格は未取得 |
| riskの利用範囲 | `RAW_MODEL_DIAGNOSTIC_ONLY`。MSE上界と評価したNを報告し、精度達成/十分NのPASS/FAILを付けない | full-H targetに対するraw riskは純shot成分ではない。元Minkowskiを直接総risk条件へ置換せず、δEを加算しない |
| 未解決loss | BOUNDED_INTERVAL `[0,L_U]`、下記の主値範囲式からL_Uを決めるloss family | 未解決energyを補完せず、単一推定器のRMSEとは区別。L_Uの数値は未計算 |
| 二項確率rule | EXACT_AFFINE_OF_BINARY64_RAW | 与えたbinary64 rawを有理数としてp=(1+raw)/2をexactに計算。旧MCの丸め済みpとの相違を明記。ruleは未選択 |
| IID scope | 元ideal_finite_RTE_lawに条件付くfresh IID shots/axes/rounds | 保存lawの定義とbindingを再利用する。今回のjoint count modelへの適用を明示し、actual samplerの資格へ昇格しない |
| whole-shot vector | 下記の保存nominal形を**新しい共通control入力**として選ぶ案 | AのLD1/LD2にはshotsがない。旧配分の科学的資格・旧S3 RMSEは移送しない。別vector指定も可能 |
| phase安全集合 | 初回候補として全round `ρ_m=1.0 rad` | 2ρ_prev+ρ_current<πという理想則には余裕があるが、元binary64/tie規則の有限検査は別に実施。1.0は最適値でも採択済み値でもなく、π/3へ自動拡張しない |
| count/pair範囲 | 選んだNが含む全count domainと、その隣接組の下記の有限上限 | CPU/RAM capではない。SOURCE/対象/上限を実行前に固定し、UNKNOWN時に無断拡張しない |

旧nominal形のcontrol案（X/Y各軸）は、

`64,60,56,52,47,43,38,34,30,25,21,17,12`

のコピーである。SOURCEは`2084791b1b09cff65e07fa8566bb734e1e6bff8e:docs/research/rpe_s3_final_plan_20261113/run01/EXECUTION.json`のnominal_N_XY。元configurationは今回の両IDと異なるため、N packetとして受け入れず、ユーザーが採用する場合だけ新allocation ID/SOURCEを付ける。これを十分NやLD2向けに検証済みの配分と呼ばない。

loss案の式：元canonical hをfloatへ戻し、`b=nextafter(pi/float(h), +infinity)`、

`L_U=max((b−E_proxy)^2, (−b−E_proxy)^2)`。

二乗差は既存exact scalarと同じ解釈で有理数計算する。これは「仮に未解決energyを同じ主値範囲で評価するloss family」の上限案であり、実decoderのNone出力を置換しない。元priorはclipに使用されないため、prior幅を上界にしない。今回は保存solver値をこの式へ代入していない。

検査範囲案：`C_m=(N_mX+1)(N_mY+1)`として、各評価の

`maximum_round_outcomes=max_m C_m`

`maximum_verification_pairs=Σ_(m=1..12) C_(m−1) C_m`。

これは選んだ整数vectorの組合せ上限で、実rawの分類・確率・MSEから調整しない。degenerate lawや重複位相除去で実際の検査量は小さくなり得る。Nが未選択なので上限の実数値も未選択。既存d717の検査アルゴリズムを使い、新しいdecoder/risk controllerは作らない。

## 明示許可後にだけ行う範囲

基本案は**LD1/LD2各1whole-vector、計2回のD.evaluate**。独立した2結果として、元K.accept_saved/K.prepareとCの固定metadata契約を使用する。hは元binary64由来のexact値で、decimal 1/10へ置換しない。risk/loss・target・上限は新runの別入力へ記録し、原A設定を更新しない。

1. 固定SOURCEからrawと来歴を読む。対象・raw/target/整数Nの不一致は数値処理前に拒否。
2. 指定rule/IIDモデルで二項確率をexactに評価する。corrected/Bを掛け直さない。
3. 全roundのzero/unsafe/reference-oracle-tieと、元decoderのgood遷移条件を評価する。
4. 検証成立時、全zero確率、未解決確率範囲、初回unsafe分割、指定lossでのモデルMSE上界を返す。
5. 元入力N_X/N_YとSOURCE、モデル条件、数値形式、UNKNOWN理由を保持する。出力は独立の新結果先へ保存し、原raw/STOP/ledger等を上書きしない。

`reference_oracle_tie`は全RPEの実tie確率ではない。元decoderでは全zeroが優先するため、zeroなしの全体tieには、検証成立時の保守的範囲`[0,P(any unsafe)−P(any zero)]`を併記できる。これは既存出力からの差であり、全履歴tie確率の厳密列挙ではない。差の数値計算も今回未実施で、実施する場合は上記の有限適用範囲に含める。

## UNKNOWN・判定範囲とSTOP

loss/target/IID未選択、raw不足、有限検査範囲外、good遷移の閉包不成立、数値domain外ではUNKNOWN又は明示エラー。ρやN、対象構成、最大検査量を自動で変更しない。未確認結果をPASSにせず、MCへfallbackしない。

1配分ずつの評価から得られるのは**各1つのNとrisk上界の対応**だけである。最小/必要/最適N、配置族全体、LDの正式な優劣は決まらない。より多い配分を試す場合は、既存許可の対象・本数・探索範囲内か確認し、範囲外なら別明示許可を受ける。

MSE上界が保存εE²等を超えても、保守的上界だけからそのNや方式の不可能性を認定しない。逆に小さくても、正式な元Minkowski適合、target/state/実law/数値誤差の資格、十分N/global boundへ昇格しない。今回の初回案ではD.compare/W.compare/B.cost/B.selectによる実G/winnerは計算しない。

正式Gには、採択済みのN評価水準と、同identity/law/compiler/lowering/full wrapper/P0の適格E[C]が必要。人工価格・構造proxy・未資格model/cohortは代用しない。C/Dへの価格要求を今回新規生成する必要はない。

**本準備はここでSTOP。** ユーザーが上記の条件を選び、有限2評価を明示許可した後だけBが実行する。チャットでの許可で足り、承認証明書・grant発行・別の認証手続きは不要。
