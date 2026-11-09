# 新A Q3-A：Prefix受理・feedback誤差の保証接続

2026-11-10 JST／SERVER 2。**Q1の全対象履歴での受理判定は`CONDITIONAL_ACCEPTANCE`。** 成功event上の非ゼロ性・branch一意性・固定aliasの余裕は論証できたが、旧binary64 rankingと192-bit区間検査の一様誤差／幅が未資格である。Feedback算術はper-callの誤差包含を持つ一方、全対象角・native context込みの共通cap保証は未完了。**340 shotsの実装成功保証・全18round削減は認定していない。**

基準はQ2完了`b6b74a3f0a73bb5984d8ee978705eb0e498b86c2`。同じQ研究branch `codex/algo-improve-new-a-total-cost-20261109`を継続した。今回は静的解析・数学的条件整理・人工5テストのみ。Q0／Q2 campaign、保存科学count／入力の新規数値再解析、科学build／compile／simulation／sampling、RPE／価格／Clifford+T評価はいずれも0件。

## 1. 確認SOURCEと再利用

固定科学条件はLinear H4／1.00 Å／STO-3G／DF12、元H exact ground供給・準備費除外、Partial S2／LD3／h=.02／K2、canonical independent finite bits、18round。m17 q131072/r32/bits88、energy精度1.5936e−4 Ha、failure .05、phase cap .4。全て維持した。

- Q0 `diagnose_pr_reference_q0.py`／`check_pr_reference_q0.py`、固定SOURCE／INPUTS／CHECKS／圧縮RESULTSの保存status・minimum・feedback bracketをread-onlyで確認。Q0の全n／全count診断を再実行していない。
- Q1 `prefix_from_counts`／`feedback_from_prefix`／`feedback_readout`／`decode_counts`、旧branch decoder、既存人工fixture、契約・SOURCE・TESTS・STOPを確認・再利用。
- Q2 REPORT／RESULTS／CHECKS／SOURCE／STOPを確認。有限16構成のPASSと価格は保存結果として引き継ぎ、回路・QPY simulationを再実行していない。
- 旧`error_and_shots`、座標Hoeffding、Gershgorin alias、Compiler A、Qiskit1.3 parameter／等価library／BasisTranslatorの接口、mpmath1.3 atan2のSOURCEを確認した。既存V6-Qのreset／supplier条件とm17価格関数も静的に参照した。

出典・hashと実施境界は[SOURCE.json](../../../artifacts/pr_reference_feedback_q3a/2026-11-10/SOURCE.json)。詳細な前提と論証は[GUARANTEE_CONNECTION.md](GUARANTEE_CONNECTION.md)。旧Q1のproduction実装・precision・成功event・受理条件は変更していない。固定SOURCE上の不要な拒否や実装バグを実証したわけではなく、保証の不足を理由に別branch decoderを重複実装しなかった。

## 2. 今回閉じた部分と未資格

| 論点 | 今回の到達点 | 残る条件 |
|---|---|---|
| Prefix vector非ゼロ | 旧座標成功eventと正半径・β<π/2から証明 | 旧SOURCE／law・統計条件への適合 |
| 正しいbranch | S上で一意。全m0〜16の距離gapは粗くても2⁻¹⁵ radより大きい | 旧float rankingが同じbranchを選ぶ一様数値契約 |
| Alias | 保存spectral upper<4、h<1/32から真のbase位相<1/8。正しいm16中心はguardから十分離れる | 数値中心の包含・period代表の適合 |
| 192-bit受理 | binary距離誤差η・区間端点slackνが各gapの半分未満なら受理を証明 | 全対象countについてη・νを囲む証拠 |
| Feedback算術 | count argとの整数period関係と、各戻り値の誤差包含を確認 | 全対象countの区間幅とcap内包含 |
| Parameter／symbolic P→RZ | float受渡し・符号反転は追加角度丸めなし。指定等価経路は全λで成立 | 全m17 contextでの実経路・core誤差保証の保持 |
| Q0→energy hull | 成功確率の条件付き接続を明示。有限precisionの余裕条件を追加で特定 | 中心区間幅・hull丸めを保存余裕内へ移す資格 |

**全targetについて`PROVED_ACCEPTANCE`でも、実`COUNTEREXAMPLE`発見でもない。** ライブラリへの故障注入testを実際の固定runtimeでの拒否例とは呼ばない。既存のintervalモデルは維持し、それを新しい形式検証済みlibraryと認定してもいない。

## 3. 人工検証

[新test](../../../tests/test_pr_reference_feedback_q3a.py)は既存Q1人工fixtureをimportして再利用し、旧18テストを再実行しなかった。[log](../../../artifacts/pr_reference_feedback_q3a/2026-11-10/TESTS.txt)の5件がPASS。

1. 位相誤差を人工的にcap近くへ寄せた非ゼロcount履歴の受理。
2. m6で負の実軸を通る正負の全prefixにおけるperiod／branch対応。
3. 誤った旧branchを注入した場合の拒否。
4. argを含むだけの広い区間を注入した場合の拒否（包含だけでは全受理の証明にならない）。
5. 人工count端点・負の実軸近傍で、返された算術包含と384-bit点oracleを比較。

5は人工参照だけ384-bit、production mp／ivは192-bitのまま。途中で保存科学値を評価していない。testのPASSは一様η・ν・全support・RPE成功のcertificateではない。著者Codexによる技術検査であり、独立レビューではない。

## 4. 340 shotsの保証へ残るもの

Q0の共通安全側条件`.000748774257 rad`は維持する。追加角度誤差だけをradで加算し、prefix不確かさはuへ、旧diskは保存B・ρLへ既に含まれる。一般operator誤差をradへ置き換えない。symbolic P→RZの条件が実contextで保持されれば純粋角度として追加0にできる部分はあるが、現状で未知のnative全体を0扱いしない。

新たに明示した実装残余は、Q1のenergy中心が点ではなく包含区間であること。Q0のphase径≤.8に、その中心幅と外向き丸めを無料で加えられない。保存δloと共通δsafeの差を余裕として使う数学的案を整理したが、新しい差・丸め予算の数値評価は行っていない。

Prefix受理・feedback信号の条件が全historyで成立すれば、historyで条件付けたm17 Hoeffdingとunion boundにより`34/720+1/360=.05`へ接続できる。追加の受理／実装失敗を確率αextraとして残すなら`.05+αextra`であり、そのまま既存予算を満たさない。failure再配分やpost-selectionは採用しない。

## 5. 次の判断・mandatory STOP

[NEXT_VALIDATION_PLAN.md](NEXT_VALIDATION_PLAN.md)に、先に判断すべき数値branch保証のrouteと、保存Nによる有限count-domain監査、保存bracketの最小再解析を分けて準備した。実m17回路・価格へはまだ進まない。

**GPT／ユーザー**が、旧binary経路の一様数値契約を確立するか、数学的に同じnearest branchをinterval主体で選ぶ最小改修を検討するか判断する。重要な新規保証の採否もGPT／ユーザーへ戻す。保存科学入力を用いる次の数値診断は、固定scopeを指定した別途実行承認が必要。

**Codexは成果公開・remote照合後mandatory STOP。** Q0／Q1／Q2の旧SOURCE・raw・result・STOP、他Track・共有環境は保全した。[Q3-A STOP](../../../artifacts/pr_reference_feedback_q3a/2026-11-10/STOP.json)を維持し、科学レビュー・新しい科学campaign・m17価格・全18roundへ自動進行しない。
