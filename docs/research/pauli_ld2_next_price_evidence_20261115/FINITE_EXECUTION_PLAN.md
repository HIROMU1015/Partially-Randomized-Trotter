# 有限実行案：LD2 m9長roundの独立normal 2 path

**提案・未採択・未実行。推奨範囲はm9だけ、2 fresh path×X/Y、最大4 compile。**
計画の公開・採択は科学実行許可ではない。新しい予測値、seed表、科学実行SOURCEは今回作成していない。

## 1. 固定する対象

| 項目 | 提案する値・固定参照 |
|---|---|
| m / q / R / T / K | 9 / 512 / 526 / 51.2 / 2。T正本は下記packetの`3602879701896397/70368744177664` |
| canonical physical ID | `RMSE:978923356c4e83c897d03442fb488997d32047237444f95757658f61321b2729` |
| feature map / 順序 | `PR_STEP_V1` / `[one,pf_step_count,event_count]`、値`[1,512,269312]` |
| configuration | `d7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3`、prefix2 |
| 物理条件 | H4 linear1Å/STO-3G/tol-only DF10、LD2 head one-body+fragment0,1、S2、step0.1、K2 normal |
| law / 辞書 | 元PCG64/NumPy float64 CDF。P0保存LD2辞書`dd986e52057aa01a330a83dd147316696dbae669dc8a51828ee2a54734016ea7` |
| lowering / scalar | `PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`。c_H＋保存s_Rを総Tで一回、元binding維持 |
| 価格・compiler | 測定付きfull controlled Hadamard wrapper、状態準備費除外、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/P0/symmetric_directional |
| cohort（提案名） | `D_PAULI_LD2_M9_LONG_NORMAL_N2_V1`、path index0/1、X/Yは同一pathの別軸 |
| 観測 | 同じ軸の2 path算術平均。2比較単位・4個別receipt。path1欠測ならn1平均へ変更しない |
| compile上限 | 最大4。CAL/B0/W/旧holdoutの再compile0。追加seed/path、m10以後の自動取得0 |

元packetは`714247619afb4714fd663eb15c0470ef32bdc7af`の
`docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json`、
`packets`からprefix=2を選び、`rounds`をm=9で選択する。
model正本は`b74e4e536392c2bbe94cd371436615fc56c0564f`の
`docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json`。
X/Y別の既存係数・domain・5 CAL cellだけを再利用し、再fitしない。

## 2. 実行指示前に具体化する差分

固定D2 SOURCE `13ab1acfaebe5f33958e1db67862f5c45725ab91` の
`scripts/run_pauli_b1_prices.py`を静的確認した。
`validate`は元7-cell設計とn_pathsを固定比較し、`request_for`のcanonical round分岐はM0/M5/M6のみ。
**M9_ROUNDを旧jobへ足してそのまま投入することはできない。**
候補Aのm6 n2も元m6 n1の検査を通らない。旧計画・旧validator・旧結果を黙って書き換えない。

後続の限定実装では、新しいm9有限driverから以下の実経路を結ぶ。

1. 元CAL用planと`frozen_model_gate`でb74eのモデルを確認する。m9取得計画は別定義とし、
   CAL record集合・fit SOURCE・元designを変更しない。validatorを任意cell受理へ緩和しない。
2. 既存`pauli_tail_price_connection.round_request`とCのcanonical price contextでm9を解決する。
   identity/law/dictionary/profileを照合し、旧m6や非configuration IDに偽装しない。
3. 元`seed(candidate_id,cohort,index,outer,piece)`規則で各path512 stepの表をmetadataとして固定する。
   旧CAL/holdout/Wのseed表と、新cohort内の重複を確認する。今回はseedを生成していない。
4. 後続の明示許可範囲で、元b74e係数からm9-X/Yの**2 exact予測**を観測前に別fileへcommit/pushする。
   係数・元4予測を変更せず、physical ID/特徴量/axis/cohort/seedと予測locatorを結ぶ。
   今回の計画には予測数値はない。予測凍結前にsampling/価格取得・開封しない。
5. 新driverのm9 request→既存`acquire_path`/normal sampler/native/D2/wrapper/compiler→
   `compile_receipts`→canonical `price_records`/`check_round_records`相当のguard→記述比較を接続する。
   旧`records`も元validate/request_forに依存するので、その経路全体を新対象へ正しく束縛する。
   旧ADMIT/COMPAREは705fの保存手順を再利用できる部分だけ使い、m5/m6限定の選択や固定path数をm9へ流用しない。
6. metadata・人工試験で正しい対象、wrong-axis/SOURCE/seed拒否、2 path欠測、O=0、
   通常入口のgate/個別receipt/保存再利用経路を確認して実行SOURCEを固定する。実H4 testを無許可で起動しない。

採択する比較は元の記述式をこの新cohort2単位に明示的に適用する案である。
`O_a=(C_a,path0+C_a,path1)/2`、`d_a=P_a−O_a`、`abs(d_a)`、`d_a/O_a`、`abs(d_a)/O_a`。
exact正本、%は表示のみ丸め、O=0の相対はnull。入力gate不成立/receipt欠落は無効・保留であり精度FAILではない。
数値閾値・95%・モデルPASS/FAILは採択しない。m5/m6で採択した規則を、新cohortの実行許可へ拡張しない。

## 3. 資源・順序・出力・STOP

独立path/軸の並列化、同じ入力・compiled cache再利用、重複fingerprintの削減を初期から検討する。
既存索引`c511f6b4f521f0e1dd7853bea02e0b0058272640:docs/PERFORMANCE_METHODS.md`は参照したが、
Full-H用helperがPauliで動くことは未確認。対象・SOURCE・位相・cache key・実呼出しcontextを照合した
既存手法だけを使い、必要な新SOURCE/人工検証を科学投入前に具体化する。今回は性能コードを変更していない。

長roundでは入力/context、sampling、構築、path別/軸別compile、gate、receipt公開の開始・終了と実測時間を
段階logへ記録し、親processが進捗と子exit code/signalを観測する。CPU生存だけで前進やPASSとしない。
Python3.11.0rc1の非同期faulthandler定期dumpは使わない。独自の低いCPU/RAM capを課さず、
共有環境の制御を変更しない。既存m6のwallからm9のETA・RSSを比例外挿しない。

提案public：`docs/research/d_pauli_ld2_m9_longpoint_results_20261115/run01/`。
提案private：`/home/AbeHiromu/projects/pauli-ld2-m9-longpoint-20261115-private/`。
確認時点では既知D2 Pauli worktreeのpublicと上記privateは未存在。実行時に対象worktreeと保存先を再確認する。

順序：本案採択＋対象・SOURCE・予測計算と取得/比較の明示許可→限定技術準備・metadata重複照合→
予測2件とcohort/seedの固定公開→normal2 path/最大4 compile→gate・4個別receiptを固定公開→
新cohortだけの有限平均・2記述差→引渡し・STOP。
科学実行許可は適切な範囲で一度に与えられればよく、各段階で再承認を要求する仕組みは追加しない。

raw RESULT・元compile record/QPY所在hash・実行SOURCE・公開SOURCE・個別receipt・gate・
比較結果と親旧成果へのlocatorを残す。無断retry、path追加、別roundへの代替、閾値変更・再fitをしない。
資源・入力・SOURCE・必要gateが不成立なら部分証拠を保存してSTOP。完了後もmandatory STOP。
新cohortを旧m5/m6データへ加算して既存独立比較を更新しない。

Aを後で採る場合の別候補はm6、cohort提案名`D_PAULI_LD2_M6_REPLICATION_N2_V1`、fresh index0/1、
X/Y最大4 compile。物理IDは旧m6のcanonical ID、Pはb74eの保存m6予測をそのまま使う。
public候補は`docs/research/d_pauli_ld2_m6_replication_results_20261115/`、private候補は
`/home/AbeHiromu/projects/pauli-ld2-m6-replication-20261115-private/`（確認時未存在）。
今回Bと同時に許可・投入する計画ではない。

希少イベント調査・強制order2・追加samplingは保留。自然発生したeventとgateの保存はするが、
未観測order2/非自明位相を既存toyや別辞書で補完しない。
LD1/Full-H移送、新CI、正式E[C]・N/G・winnerには進まない。
