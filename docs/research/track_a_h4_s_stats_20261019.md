# Track A H4 S_STATS：保存統計によるPの標本数設計

状態：ユーザーが数値解析1回を承認。人工14件PASS、SOURCE固定・seal後にone-shot実行する。
固定D結果commitは`4e8ad1936363266d1b55bb689c36846e6c3dfcdb`。
元`S_STATS_contract_DRAFT.json`と[事前設計](track_a_h4_dp_phase0_20261019/h4_pr_confirmation_design_DRAFT.md)を変更しない。

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
終了監査は保存済み数値結果とSHAを確認し、数値設計を再実行しない。

## Pの契約に残る情報

S_STATS後、9候補のn、14方向、PF/PR shot、6感度scenarioのP科学call予算を保存する。
P code SOURCE候補は固定commit `4e8ad1936363266d1b55bb689c36846e6c3dfcdb`。
そのP kernelはDで使ったSOURCE8a82b927…の該当fileと同じbyteである。
P最終SOURCEの受入、科学input/runtime/preparation/law照合、全旧系列のseed除外inventory、
新namespaceによるseed一覧・collision receipt、実行資源のfresh照合はPのsealに残る。
S_STATSで新seedやP trajectoryは生成しない。P grantは無効のままである。

次の担当：Codexが承認済みS_STATSを1回実行・保存監査してmandatory STOPする。
ユーザーが具体的n・SOURCE・seed・予算の揃ったP最終契約を確認し、P科学実行を別途判断する。
