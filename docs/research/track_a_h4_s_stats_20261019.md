# Track A H4 S_STATS：保存統計によるPの標本数設計

状態：`COMPLETE_MANDATORY_STOP`。承認された数値設計1 pass・84 scenario・14基準方向を完了。
native exit0を親witnessが回収し、終了監査もPASS。Pは未承認・未実行。
S_STATS実装SOURCE：`83bba491ff15ad5c735281833a579d7330a63bd3`。
結果SHA：`77231c5d7d04c3fc0030cff0a43bb2680cae2478f32abb51e32f0a0b40239774`。
固定D結果commitは`4e8ad1936363266d1b55bb689c36846e6c3dfcdb`。
元`S_STATS_contract_DRAFT.json`と[事前設計](track_a_h4_dp_phase0_20261019/h4_pr_confirmation_design_DRAFT.md)を変更しない。

## 結果：事前規則による推奨n

f=4・η=.02の固定規則を適用し、9候補で合計4,185の新しい無条件trajectoryを提案する。
旧n8/n32とのpoolingや、計算量を理由としたnのcap・候補削減は行っていない。

|候補|役割|自身のendpoint最大n|推奨n|
|---|---|---:|---:|
|B2_p2_q4_2nd_r4_K2|主|901|901|
|B2_p2_q8_2nd_r4_K2|主|115|115|
|B2_p2_q8_2nd_r4_K4|診断|27|115|
|B2_p3_q1_2nd_r2_K2|主|277|277|
|B2_p3_q1_2nd_r4_K2|主|2119|2119|
|B2_p3_q2_2nd_r4_K2|主|459|459|
|B2_p3_q4_2nd_r4_K4|主|145|145|
|B2_p3_q8_2nd_r2_K2|主|26|26|
|B2_p3_q8_2nd_r2_K4|診断|28|28|

p2/q8/K4診断候補は、自身のSEからの27を対応K2主候補の115まで引き上げた。
p3/q8/K4は自身の28がK2主候補の26より大きいため28を採用した。
p2/q4主候補はε=.01/.005を共用し、他のq8候補はε=.001/.0005を共用する。
複数εを独立標本として加算していない。

|SD倍率f|目標差η|提案trajectory合計|測定compile軸|
|---:|---:|---:|---:|
|1|.02|267|534|
|1|.01|1050|2100|
|2|.02|1050|2100|
|2|.01|4185|8370|
|4|.02|4185|8370|
|4|.01|16729|33458|

上表の小さい感度scenarioへ推奨を差し替えていない。
fは保存SE由来のSDを拡大する係数であり、式にはf²として効く。
normal quantileはz_(1−.05/10)=2.5758293035489、z_.8=.8416212335729144。

固定14比較はすべて`expected_direction=+1`（保存PR平均が完成DのPF費用より小さい）。
exact tieは0件。主10・診断4を区別して保存した。
この方向は探索/旧標本による基準であり、独立確認の成功・母平均優位・全PR候補最良の認定ではない。

## Pの必要科学call予算（未実行）

|項目|推奨scenarioの予算|
|---|---:|
|新無条件trajectory / native evolution|各4185|
|測定wrapper compile|8370（各軌道cosine/sineの2軸）|
|occurrence Σq_j n_j|9770|
|RTE event Σq_j r_j n_j|37662|
|native control probe|最大36|
|独立event oracle|最大9|
|endpoint会計|1 pass、14固定比較|
|master/派生step seedの予約slot|11559（未生成）|
|新PF compile / signal / SCF / DF / state / reference / shot再算出 / fit|0|

これはPの提案予算であり、S_STATSの実行実績ではない。今回のsampling/compileは各0。

## 対象と計算規則

H4/1.00 Å/STO-3G/legacy固定DF12/8 system qubits/T=.8/P=0、通常signed・allowance倍率1。
事前登録9候補（主7＋診断K4相方2）、主10＋診断4の14固定比較。
保存mean exact・paired SE・n・PR shot、完成Dの各εの固定PF exact G/shotだけを使う。
入力は元S_STATS契約、P_registration、D_result、D終了監査、D後P契約、事前設計文書の6保存file。
すべて固定結果commitのGit blob SHAを検証し、private snapshotへコピーする。
raw標本の読込/再集計、bootstrap、候補再選択、新sampling/compile、元結果/grant/ledger/STOPの変更は0。

`s_plan=sqrt(n0)*SE_saved`、family α=.05、m=10、planning power=.8。
`n=max(2,ceil[((z_(1−.05/10)+z_.8)*f*s_plan/(η*D))²])`。
f=1/2/4、η=.02/.01、14×6＝84 scenarioを1回計算する。
候補別推奨nは主endpointでf=4/η=.02の最大。
診断候補は自身のendpoint最大と対応K2主候補nの最大を取る。恣意的capを置かない。
同一候補の複数εには同じn軌道を共用し、Pのtrajectory/compileを二重計数しない。

基準方向は`PF_G_exact−saved_PR_mean_exact`を有理数で算出する。
正はPRが安い、負はPFが安い、0はexact tieである。新標本による再現性・母平均保証ではない。
希少層の候補固有p2/p4は承認済みsummaryにないため、visibilityと未知tail費用はnullに保つ。
別の条件付きpilotから確率を移送せず、lawも再構成しない。

これは選択済みn8/n32の保存SEによる近似設計である。
f=4やnormal planning powerはtail分散・母平均coverageの保証ではない。CI/p値は計算しない。

## 実装・保全・終了

`scripts/h4_s_stats.py`はstdlibだけを使う数値kernelと拒否gate、
`prepare_h4_s_stats.py`は数値計算を呼ばないseal、`h4_s_stats_witness.py`は実child exitの回収を担当する。
元`h4_dp_contract.normal_n_formula`を再利用する。
人工14件は既知normal値とceil、exact方向、候補共用、K4下限、one-shot/失敗予約/承認拒否を確認した。
作者による技術確認であり、独立reviewとは呼ばない。

実装SOURCE、6入力、CPython/実stdlib/runtime closure、承認receipt、spec digestを固定し、
新しいprivate grant/ledger/lock/outputを使う。既存Dの57＋297記録はSHAとcopyを検証して保全する。
84 scalar scenarioは小規模なので1 processで処理する。計算資源・memory・wallに恣意的上限を置かない。
共有設定・affinity・他jobへ変更しない。
数値設計1 pass・84 scenario・14方向の予約を先に記録する。失敗時はSTOPし、自動retryしない。
終了監査は保存済み数値結果とSHAを確認し、数値設計を再実行しなかった。
SOURCE 4 file、固定入力6 file、実際に使うCPython/stdlib/runtime 136 fileを前後照合した。
親witness wallは.2651786431670189秒、解析＋final照合区間は.09966849163174629秒。
1 processで完了し、retryは0。旧354記録とcopyのSHAを照合し、新23記録もSHA storeへ保全した。
作者のmetadata監査は`scripts/audit_h4_s_stats.py`。新旧raw標本への統計演算はない。

## Pの契約に残る情報

9候補のn、14方向、PF/PR shot、6感度scenarioのP科学call予算を保存した。
P code SOURCE候補は固定commit `4e8ad1936363266d1b55bb689c36846e6c3dfcdb`。
そのP kernelはDで使ったSOURCE8a82b927…の該当fileと同じbyteである。
P最終SOURCEの受入、科学input/runtime/preparation/law照合、全旧系列のseed除外inventory、
新namespaceによるseed一覧・collision receipt、実行資源のfresh照合はPのsealに残る。
S_STATSで新seedやP trajectoryは生成しない。P grantは無効のままである。

現在はmandatory STOP。P templateはn・PF・shot・基準方向・予算が固定されたDRAFTであり、
seed一覧/衝突照合等が揃った実行用sealではない。S_STATS grantをPに流用しない。

次の段階（未実行）：CodexがPのSOURCE/実行closureと全旧seed inventoryのmetadata確認、
private seed一覧・衝突照合・最終sealを準備する。ユーザーがその具体的契約を確認してP実行を判断する。

## 保存成果物

- [1回計算の全結果](../../artifacts/resource_applicability/track_a_h4_s_stats/2026-10-19/result.json)
- [84 scenario CSV](../../artifacts/resource_applicability/track_a_h4_s_stats/2026-10-19/normal_scenarios.csv)
- [候補別推奨n CSV](../../artifacts/resource_applicability/track_a_h4_s_stats/2026-10-19/candidate_n.csv)
- [P契約DRAFT：n・14方向・SOURCE候補・予算・seed未確定事項](../../artifacts/resource_applicability/track_a_h4_s_stats/2026-10-19/P_contract_after_S_STATS_DRAFT.json)
- [終了監査](../../artifacts/resource_applicability/track_a_h4_s_stats/2026-10-19/completion_audit.json)
- [SOURCE・6入力のSHA binding](../../artifacts/resource_applicability/track_a_h4_s_stats/2026-10-19/source_input_binding.json)

公開projectionはprivate絶対path、receipt/grant原本、seed一覧を含まない。
成果物はlocal commitまで保存し、remote pushは行っていない。
