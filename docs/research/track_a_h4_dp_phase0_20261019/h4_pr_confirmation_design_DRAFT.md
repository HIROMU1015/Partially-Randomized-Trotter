# H4 PR独立確認の事前設計 DRAFT

状態：`DRAFT_UNAUTHORIZED`。PFとnは`PENDING_D_RESULT`、nの数値計算はさらに`PENDING_S_STATS_APPROVAL`。
Phase 0では保存済みmean/paired SE/n/shotを転記しただけで、raw再集計・bootstrap・power/nの実数値計算は0。

## 事前登録集合

64c85f7a…の通常signed・倍率1・強化決定論subsetに対する保存winner IDを引用した。
旧n32はfixed anchorの選択・推定標本、新A n8は探索標本であり、独立確認ではない。

|対象ε|旧PR代表|新A代表|
|---|---|---|
|.05|B2_p3_q1_2nd_r4_K2|B2_p3_q1_2nd_r2_K2|
|.01|B2_p3_q2_2nd_r4_K2|B2_p2_q4_2nd_r4_K2|
|.005|B2_p3_q4_2nd_r4_K4|B2_p2_q4_2nd_r4_K2|
|.001/.0005|B2_p3_q8_2nd_r2_K2|B2_p2_q8_2nd_r4_K2|

主検証は重複を除く7候補、10 candidate×ε比較。
.005の旧/新の僅差は双方を含めたまま検証し、探索点で小さい方へ絞らない。
q/r schedule上端にある旧p3/q8/r2および新p2/q8/r4について、K4相方2候補を
.001/.0005の診断用として追加する。合計9候補・14 endpoint（主10＋記述的診断4）。
これは構造に基づく固定選定であり、新しい費用ソートをしていない。
残る候補は探索結果として保持する。確認を実施しない候補の劣位や全PR grid最良は主張しない。

機械可読登録はP_registration.json、転記統計はP_saved_statistics.csv。
候補追加・ε追加・確認結果による再選択・探索n8/旧n32とのpoolingを禁止する。
同じ候補の複数εでは1組のn軌道を共用し、独立実験として重複計数しない。

## D後のPF固定規則

Dの受入済み全104候補会計から、各εの通常signed・倍率1の最小Gを持つPFを採用する。
同費用はcandidate ID辞書順で代表を固定し、tiesも記録する。
このPFをそのεの旧PR・新A・診断候補の全てに共通使用する。旧16 subset winnerを優先しない。
D未完了・欠測・不適格なendpointでは比較を未確定にし、別PFを都合よく選ばない。

探索meanと完成D費用の符号も、後述S_STATSの限定された14固定pair比較の承認内でP前に固定する。
この新しい基準方向の数値比較もPhase 0では実行していない。新標本の符号が同じなら点順位再現、
逆なら非再現、exact tieならINCONCLUSIVE。完成Dが探索PRより安くても候補を削除しない。
`REPRODUCED`と「PRがPFより安い」は別fieldとし、PF優位の再現も方向を明記する。
経験的非相殺を将来併記する場合は同じ標本・同じPFの保存shotを使う記述的副評価とし、
新たな確認実験・主10比較の同時保証へ数えない。今回の実行予算案は通常14 endpointに限定する。

## 標本数設計：数式まで、数値計算は未実行

候補j/endpoint eの保存標本数n0とpaired SEを入力とし、
`s_plan(j,e)=sqrt(n0)*SE_saved(j,e)`と定義する。これは式であり、Phase 0では評価していない。
新標本は候補間で異なるmaster/step streamを使い、共通乱数は使わない。
したがって候補間のraw共分散の再計算は、この設計の必須入力ではない。

目標は固定PF費用D_eの2%の差。family α_F=.05、主比較m=10、計画power=.8。
normal plug-inシナリオは

`n(j,e;f,η)=max(2,ceil[(z_(1−α_F/m)+z_.8)^2 * f²*n0*SE_saved² / (η*D_e)²])`。

f=1/2/4、η=.02（主）/.01（感度）の表を、別承認S_STATSで1回だけ計算する。
推奨n_jは当該候補の主endpointにおけるf=4、η=.02のnの最大。
診断K4相方のnは対応K2主候補のn以上とし、同じ式を自身の保存SEにも適用した最大にする。
現時点の9候補のnはすべてnull。H8のn48やH4旧n32を機械的に採用しない。
計算後の値が大きいことを理由に無断でcap・標本削減をしない。P承認前に具体的予算を提出する。

これは選択された小さいn8/n32由来の分散を使う近似設計であり、f=4はtail分散の保証ではない。
正規性・CLT・選択後coverageを仮定して母平均優位の検定を認定しない。
本案ではCI/p値を出さず、主10比較の多重性は計画上のBonferroni配分として明記する。
記述的な点再現10件を「10件の母平均保証」と数えない。各候補の途中平均を見たn変更は禁止。

## 希少eventと相関

各候補の元finite-RTE分布における1 occurrenceのp2/p4を使う場合、
`π_nonzero=1−(1−p2−p4)^R`、`π4=1−(1−p4)^R`、
`visibility(n)=1−(1−π)^n`、`n_vis(γ)=ceil[log(1−γ)/log(1−π)]`。
これもPhase 0では式のみ。候補固有の保存law metadataが不足すれば数値はnullにする。
別q/r/prefixの条件付きpilotから確率・費用を移送しない。lawの新しい科学的再構成は自動実行しない。

πが小さい／未観測層の費用が大きい場合、通常SDだけで必要nを保証できない。
感度式`μ=(1−π)μ_observed+π μ_tail`で未知μ_tailを未確定に保ち、
πや費用上界がないときは母平均上界をnullにする。visibilityのnは母平均精度のnではない。
観測層はS0_ONLY/S2_SINGLE/S2_MULTI/S4_PRESENTとして全標本を保持する。
条件付きn8のs2_multi/s4を新しい無条件標本へ混ぜない。未観測を確率0・費用0にしない。

各trajectoryで`G_e=N_cos,e*C_cos+N_sin,e*C_sin`を作り、そのSD/SEを将来P承認内で計算する。
cos/sinを独立標本として扱わない。同じ候補を複数εで使う共分散は新標本から保存し、
候補ごとの新nと同一標本の共用を明記する。候補間の独立streamは数学的独立性の証明とは呼ばない。

## 判定状態と追加承認

- REPRODUCED：全固定nが技術受入され、新点順位の方向がP前に固定した方向と一致。
- NOT_REPRODUCED：同条件で方向が逆。悪いtrajectoryを除外せず報告する。
- INCONCLUSIVE：exact tie、基準自体がtie、固定比較不能。母平均については常に別fieldで未確定。
- TECHNICAL_STOP：SOURCE/input/law/paired/control/compiler/native/IO等の不合格。科学的非再現と区別。

追加S_STATS承認案：転記済み14 endpointのmean/SE/nと完成D費用だけを入力とする数値設計1 pass。
f×ηの最大84シナリオ・候補別n、および14固定pairの基準方向だけを計算する。
これは固定pair比較の明示的な追加承認も含む案であり、単なる読取許可から拡張解釈しない。
raw trajectoryの平均/分散/共分散再集計、bootstrap、
探索順位の変更、実H4 sampling/compileは含めない。必要ならraw解析は別S_RAW契約とする。
現在、raw再集計を必須とする根拠はないためS_RAWの実行を要求していない。

次の担当：D完了後、ユーザーが必要なS_STATSと、最終n・固定PF・SOURCE等を含むP科学実行を別々に承認する。
