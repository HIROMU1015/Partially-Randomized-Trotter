# 保存共分散の伝播と未取得部分

本書は一campaignの保存出力を整理する。新fit/bootstrap/価格再解析は行わない。[固定規則](PLAN.md)、各scenarioのresult.json `uncertainty`を参照。

出力vectorは方式別6資源とpaired差6資源の18量。実際の順序は各componentの`output_labels`に保存した。対応する線形作用Lを保存mean-price/係数共分散Σに掛け、LΣLᵀを保存した。method・X/Y・資源のoff-diagonalを保持し、共有係数をq別に独立にしていない。これは**期待価格推定量の名目的な変動**であって、量子測定分散や実験run全体の物理的支出分散ではない。

|component|取得できたjoint情報|資格と限界|
|---|---|---|
|direct_q2/4/8/16/32|C7-4保存36量sample covariance/n、元平均と全量一致|各q内の共有path/方式/XY/資源を保持。保存pseudo-IID解釈に条件付き|
|A/S long_q_shared_model|保存72/108係数bootstrap covarianceを全7長qのshot重みへ伝播|256既存stratified bootstrap、同じpath indexを全量共有。新bootstrapなし。独立長q検証やmodel biasは含まない|
|F long_q_shared_model|保存48係数covariance、元q2/q4平均からの線形変換|元のcross-q=0というpseudo-stream仮定を継承するだけで、joint-IID証明ではない|
|F linked_q2_q4_and_long_q|同じβによるq2/q4直接価格と長q予測のjoint線形伝播|直接/fit共有を含む。このcomponentはdirect_q2/4とlong componentを置換する候補で、重ねて加算しない|

A/Sの係数はq2..32の直接価格と同じ標本から得られた。保存係数covarianceだけでは直接平均とのcross-covarianceを回収できない。bootstrap個別実現値・joint統計の追加処理を勝手に再解析せず、unknownとした。Fは保存βとq2/q4平均の可逆関係だけを使ってこの共有を扱えるが、他cohortとの依存までは得られない。

## 長q blockの名目的paired SD

以下は総ΔGのSDではなく、**長qモデル係数blockだけ**の条件付きSD。丸め値、CIなし。

|資源|A|F|S|
|---|---:|---:|---:|
|RZ count|1.51005e6|5.43997e6|5.43243e6|
|CX count|5.87e-9|0|4.40e-9|
|RZ depth|6.98780e7|2.70035e8|2.59238e8|
|CX depth|3.80855e7|1.46246e8|1.41605e8|
|Total depth|1.52993e8|5.92274e8|5.69508e8|
|Circuit size|1.51005e6|5.43997e6|5.43243e6|

CXのほぼ0は保存モデルのpaired直線と保存再標本化の名目的性質。全supportの価格分散0、外挿bias0、母平均の完全な把握を意味しない。F joint component等の全18×18値はJSONへ保存した。実行中、負の伝播分散を丸めるcaseは0だった。

## 全体をnullとした理由

- q1には保存されたfull joint mean-price covarianceがない。保存paired cohort SDを無断で母平均のSEへ再資格付けしない。
- direct-q間、A/Sの直接平均とfit、scenario間のjoint covarianceが不足する。未知off-diagonalを0で埋めない。
- model bias、長q外挿誤差、finite-bit lawとideal lawの距離・joint-IID資格、actual実装誤差は未取得。
- 既知componentの単純和は全体の不確かさでも下界でもない。未知のcross項は正負どちらにも寄与し得る。

したがって**全scenarioのtotal covariance、total SD、total CIはnull**。`UNCERTAINTY_PARTIALLY_QUALIFIED`を保つ。scenario間の点推定差は探索的感度であり、統計的confidence intervalやbias boundではない。追加joint処理・bootstrap・新fitが必要なら次の別承認scopeへ戻す。

理想RPE共通Nとactual成功保証の区別も維持する。この不確かさの不足は、明示採択された条件付き点推定を定義することの妨げにはならない。
