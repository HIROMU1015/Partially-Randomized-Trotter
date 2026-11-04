# 2026-11-05 Controlled IID測定と数学的energyの条件付き95%接続

ユーザーの実行許可後、固定SOURCE ffa1e22b3001ad1c4eb275ff1121ed17111452ecを無変更で
専用branch/worktreeにてArb160・1 pass実行した。
Linear H4/1Å/STO-3G/保存legacy DF12、Partial L_D3/δ=.02/K2/cutoff1e-12、
保存physical normalized ray・8 system qubit・4電子sector70、m0..17の保存q/r/N/α/β/ε/μ₀/E_ref、
uniform53 lawとrun_le_1 support policyを維持。

決定：全36軸のt=ε−(U_cat+Δ_mean)>0と直接Hoeffding上界を構成し、exact総和≤1/20を確認した。
失敗上界の上向き表示0.0496929886343339467、成功下界の下向き表示0.950307011365666053。
同じGへの旧Dの17 strict branch・alias/liftと旧Cのreference込みenergy証拠を再利用し、
CONDITIONAL_IDEAL_CONTROLLED_IID_MATHEMATICAL_ENERGY_95_SUFFICIENT。
G上のenergy上端は旧値0.000138689200360119854 Ha、保存目標binary64 .00015936001019904 Ha。
Δ_meanをGの確率へ一度だけ計上し、位相/energyへ再加算しなかった。

新36 tailセルと1 exact集約のみ。native exit0、runner .0212秒、process wall.64秒、RSS107,780 KiB。
新H/state/DF/mean/support/geometry/reference、Float64 decoder、sampling/build/compile/cost/重要レビュー0。
原RESULT/stdout/stderr/resourcesの4原本、全36軸転記表、技術照合を
[実行報告](../rpe_controlled_iid_success_energy_execution_20261105/REPORT.md)へ公開する。
追加precision・科学retry/resume0。旧A成功下界・旧Σα UNKNOWN・旧一様18/36判定を上書きしない。

理由：今回変わったのはGの確率を生む平均モデルで、G/中心/ε/q/referenceは同一。
新たな幾何・固有値計算なしに、既存の数学的含意と新しい直接Hoeffdingを接続できた。
保証対象は指定ideal IID測定・数学的decoder・alias条件下の保存DF sector ground energyに限る。
PCG64/Float64/compiled/noisy/実状態準備/真の分子energy/総費用を認定しない。
RPE全体false/UNDETERMINED・両U nullを維持。
次はユーザーが結果の扱いと残る保証の優先順位を判断し、Codexは別指示の範囲だけ技術準備する。
公開照合後mandatory STOP。
