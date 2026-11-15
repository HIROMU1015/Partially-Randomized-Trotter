# B・D・統合担当へ：未観測PF価格60軸のC採否案

**Cの推奨：60軸すべて条件付き採用。ユーザー未採択、価格値は未算出。**

[REPORT](REPORT.md)の冒頭/§3/§5が判断と利用条件、[RECOMMENDATION.json](RECOMMENDATION.json)が12 group/60軸の固定locator、[CHECKS.json](CHECKS.json)がmetadata照合結果。`rows[*].price=null`、`user_adopted=false`、元qualification/STOPは不変。

## 対象と根拠

- Full10/B0I p7 × S2 16h/7・標準Yoshida4 16h/3・M5 16h、m7/m8/m10/m11/m12×X/Yの60軸。固定H4/1Å/STO-3G/tol-only DF10、元H/state/snapshot・piece/fragment順・scalar一回、symmetric-directional、full measured controlled wrapper/P0、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17・coupling mapなし。
- 対象q：S2=56,112,448,896,1792、Y4=24,48,192,384,768、M5=8,16,64,128,256。m7/m8も未観測であり、単なる補間採用ではない。M5/m4 q1はQPY監査点ではない。
- alignment：`0b4b2d941cd669aaa45231e1236bc2d3158a9d12:docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json`。
- 12モデル：`12372fa6b53c878dc5dd124e036572ba0cd82061:docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json`。
- 実装論証：`dd435d07e3f0490165ff49889be9c71f573bd5db:docs/research/d_pf_unobserved_round_closure_20261115/REPORT.md`と保存schedule540check。
- 観測前監査規則：`d3338c7ef87a1da1f02690627555bbb221fbdea4`、監査結果：`606881b042ed69f9a1e7f00311417f2b65855d7b:docs/research/d_pf_qpy_closure_audit_20261115/`。

44 compiled/12 wrapper QPYで12 groupのA1〜A5 PASS、実周期RZ=b、端+終端=a+b。未観測60点の延長終端角は保存MARGIN_OK。Cは結果を再実行せず読取・SOURCE/identity照合した。

## ユーザーに採択してほしい条件

固定入力族とcompiler経路に限定し、障壁が局所最適化の全反復で持続するというDの実装論証、installed Rustと参照tagの対応、未調査の規模分岐が意味を変えないこと、終端ZSX枝の延長を明示前提として、元凍結式の導出価格を固定shot配分の全13 round比較へ利用する。

A5単独は最終監査回路のCX/motifを確認しただけであり、中間passの不変性証明ではない。CX非新設だけでは外側相殺の伝播を排除できないため、この部分は条件付きモデル前提として残す。無条件の全q閉包・2%誤差保証を付けない。m7/m8とm10〜m12に共通の根拠を使うため、現証拠で一部だけを採否別にする理由はない。

終端直接compileを必須とする現固定契約は確認した資料群にない。旧NEXT_VALIDATION_PLANのm12は条件付き選択肢。追加を新たな必須条件にしない。もし実装前提を受け入れず直接反例探索を選ぶなら、p7 S2 m12 X/Yはその2軸にしか効かず、他route/公式の資格を代替しない。

## B/Dへ

- 直接価格**96/156**は不変、導出対象**60/156**を分離。m0〜m3の別幅を式で置換しない。m6/m9の2%規則はその独立holdoutだけに保持。
- 採択後の価格算出は別指示。元JSONは改変せず、各行へ元model/receipt/closure source・axis・physical/configuration・actual幅・compilerと採択policyを付ける。既存`MODEL_PREDICTION`と条件付き導出の出所を使い、`DIRECT_COMPILED_DETERMINISTIC`として登録しない。
- 今回の推奨では資格flagを昇格しない。現C tierは`UNQUALIFIED_MODEL`。後続はユーザーの有限domain policyの下で、B既存model/scenario経路へ接続する。新provider/optimizerは不要。
- 比較できるのは、同じ宣言前提を置いた6候補の固定shot配分での条件付き総コスト。N十分性・RPE成功・他方式価格・厳密winnerは別。未知systematic/価格区間はnull、2%や独立CIで補わない。
- 前提差/反例が出たdomainはUNKNOWNとして保留し、自動refit/compileをしない。

**ロードマップ差分：直接価格96/156は不変。未観測60/156についてCが全軸条件付き採用を推奨。ユーザー採択・60価格算出・固定shot全round会計は未実施。** 本体更新は統合担当。

新科学計算0、QPY読取/監査再実行0、価格算出0、fit0、N/G0。他担当のbranch/privateは不変。commit/push・remote照合後**mandatory STOP**。
