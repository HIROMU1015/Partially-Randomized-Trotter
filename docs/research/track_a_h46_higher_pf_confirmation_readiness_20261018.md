# H4/H6高次PF：独立確認の標本計画・契約準備

2026-10-18。**最小3候補と拡張7候補の技術計画を作成。新sampling・compileは0。adopted_n=null、native adapter未完成のため実行readyではない。mandatory STOP。**

## 1. 固定証拠と範囲

比較結果commit `1fc58aa3f5568887e0678a9be939551b0ed6523d` と、その未加工result SHA256 `ef6adede7b64a91b672df0f94e7a183da71421a0eeac2b7633b97aaa9bcd612e` を入力にした。旧n8の平均・分散・SEをそのまま読み、実験設計の算術、保存event次数のmetadata集計、保存worker所要時間の資源試算だけを行った。新signal・Hamiltonian/SCF/DF/state・random draw・回路build/compile・費用ランキング再選択は0。探索n8はCの独立標本へ混ぜない。

条件はlinear H4/H6・隣接1Å・STO-3G・DF rank12/19・prefix3/5・T=.8・K4・outerごと32 RTE short steps。ε=.001、各軸ε/√2、α_axis=.025、経験的allowance1e-9、logB margin1e-12、P=0。ordinary control、explicit XXPlusYY/number Phase、Qiskit1.3.0/rz,sx,x,cx・opt1・seed17・backend/couplingなしを維持する。new_2は元8桁版で、厳密4次公式とは扱わない。

## 2. H6の保存n8と希少event

| 候補 | RZ費用標本平均 | 不偏標本分散 | 標本SE | 保存event次数0/2/4 |
|---|---|---|---|---|
| H6_S2_q8 | 1.40962200e+13 | 1.73643834e+23 | 1.473278e+11 | 2048/0/0 |
| H6_Y4_q2 | 1.36318210e+13 | 4.71458284e+22 | 7.676737e+10 | 511/1/0 |
| H6_Y4_q4 | 1.37646431e+13 | 1.63069403e+22 | 4.514828e+10 | 1022/2/0 |

各n8で4次eventと複数2次eventのtrajectoryは全て0観測。S2/q8は全8本がs0、Y4/q2はs0が7本・s2_singleが1本、Y4/q4はs0が6本・s2_singleが2本だった。forced correctness fixtureは集計へ入れない。新取得側はprivate replayのevent digestを公開済み費用recordへ照合し、旧Y4/q2は公開shardのSHAとaggregateを使用した。

保存lambda・signed係数・32-step配分から、既存provider `c31ef424…:src/trotterlib/rte.py::_paired_order_weight` と同じK4の重み式を使って、理想的な独立step法則の層確率も設計用に算出した。logBは保存値と絶対1e-12以内で一致した。これはbinary64 PRNG法則や物理入力誤差の認証ではない。

H6 Y4/q2のs4確率は1 trajectory当たり約1.74e-5、s2_multiは約0.00311。n35でもs4を1回以上観測する理想法則上の確率は約0.061%に過ぎない。したがって、nを数十〜数百へ増やすだけでは希少層の費用・分散を把握したとは言えない。未知層費用はnullのままで、0補完しない。母分散の数値上界は未取得。

## 3. 比較差と多重比較を分けた設計

最小3候補はH6 S2/q8、Y4/q2、Y4/q4。primary familyは次の2比較に事前固定する案である。

1. S2/q8 − Y4/q2：保存差4.6439897031e11 RZ、S2基準で約3.29449%。
2. Y4/q4 − Y4/q2：保存差1.3282205141e11 RZ、q2基準で約0.974353%。

拡張7候補ではH6 new_2/q1、H4 S2/q8・Y4/q2・new_2/q1を追加し、両系の3方式全pairとH6のq差を合わせたfamily7とする。比較同士はY4/q2を共有するため独立ではないが、Bonferroniのfamily構成は比較間の独立性を必要としない。一方、差の分散を`s_a²/n_a+s_b²/n_b`と置くことには**候補ごとに新しい独立乱数streamを使うという計画上の仮定**が必要で、namespaceの違いだけで証明されたとは言わない。CRNによる共分散削減は仮定しない。

このfamily alpha=.05はcompiled費用のMC比較に関する提案値であり、保存signalのshot会計に使ったalpha_axis=.025とは別である。保存shot数・誤差配分は変更しない。

再利用した計画式は`higher_pf_readiness.planning_equal_n_difference`：

`n = ceil((z_(1-alpha/(2m))+z_power)^2 * (s_a²+s_b²) / Delta_plan²)`。

mはfamily2または7、両側・名目family alpha=.05。観測差は計画effectの例で、未知母平均差の保証ではない。power80/90%、差が観測値の半分、分散2/4倍、差の区間半幅を観測差の1/2または1/4とする案を分離した。分散2/4倍は感度条件であり、上界や保証ではない。

pairごとの同数nを出し、候補ごとに関与するpairの最大nを採る配分と、全候補を同じnにする配分を併記した。前者も計画上の分散条件を満たす保守的な配分であり、計算時間の最適配分を証明したものではない。

## 4. 最小3候補の具体案

nの順序はS2/q8、Y4/q2、Y4/q4。時間は32workerで保存所要時間をLPT配置した**代理試算**で、実行時間保証ではない。

| 案 | nの配分 | trajectory | compiled軸 | worker時間代理 h | wall代理 分 |
|---|---|---|---|---|---|
| observed_gap_power80 | 10/35/35 | 80 | 160 | 19.37 | 38.3 |
| observed_gap_power90 | 13/45/45 | 103 | 206 | 24.94 | 54.8 |
| variance_x2_power80 | 20/69/69 | 158 | 316 | 38.26 | 74.1 |
| variance_x4_power80 | 39/137/137 | 313 | 626 | 75.76 | 148.2 |
| half_observed_gap_power80 | 39/137/137 | 313 | 626 | 75.76 | 148.2 |

名目80%案10/35/35は合計80本・160軸。全候補を同数35に揃える案なら105本・210軸になる。名目90%案13/45/45は103本・206軸。出発点としては後者を検討できるが、**どちらも未採用**で、n8分散・normal近似・観測effectに条件付く。厳密な検出力・coverageや必要十分なnを示したものではない。希少層の不確かさを重く見る場合に分散感度案も併せて判断する。

差の区間半幅を目標にした案もJSONに保存した。単独平均の相対MCSE1%だけを、0.97%差の識別設計へ読み替えない。

## 5. 拡張7候補の案

nの順序はH6 S2/q8、Y4/q2、Y4/q4、new_2/q1、H4 S2/q8、Y4/q2、new_2/q1。

純粋な差のnormal式だけでは、大差のpairにn=2等が出る。これをそのまま実行推奨にせず、**独立標本も最低限元pilotのn8を再現し、さらに各候補の相対MCSE1%のプラグイン目標も満たす**未採用の報告用条件を別labelで加えた。このfloorは提案する技術方針で、n8が統計的に十分という定理ではない。元の低n試算と全候補同数案も残している。

| 案 | nの配分（報告用条件案） | trajectory | compiled軸 | worker時間代理 h | wall代理 分 |
|---|---|---|---|---|---|
| observed_gap_power80 | 13/45/45/8/8/19/14 | 152 | 304 | 27.57 | 54.8 |
| observed_gap_power90 | 17/57/57/8/8/19/14 | 180 | 360 | 34.38 | 65.2 |
| variance_x2_power80 | 26/90/90/8/15/38/28 | 295 | 590 | 54.18 | 102.1 |
| variance_x4_power80 | 52/180/180/8/29/75/55 | 579 | 1158 | 107.32 | 202.2 |
| half_observed_gap_power80 | 52/180/180/8/8/19/14 | 461 | 922 | 102.39 | 194.1 |

7候補で確認しても、除いた適格q全てを含む母平均での全格子最適性は証明しない。H6 q2/q4の対抗比較を追加した新しい確認scopeであり、旧6点selection digestを書き換えない。

## 6. 資源・並列化と再利用

現在の観測はCPU affinity64論理CPU、短時間idle約61.09core相当、MemAvailable約468.3GiB。実行時には再観測する。16/32/64workerを比較例としたが恒常上限・予約ではない。32workerで過去最大worker RSSを足した代理値は約51.4GiBで、共有Python・cache・初期化・希少eventの未知増加を厳密に含めた上界ではない。

H6の保存1軌道平均wall代理はS2/q8約1078秒、Y4/q2約535秒、Y4/q4約1149秒。実測CPU時間そのものではないため、合計をCPU時間の実測値とは呼ばない。旧H4のY4/new_2はserial group全wall/8の償却代理で、軌道別時間の実測は欠測と明記した。slot配置・完全な線形加速・今回のidle状態を将来へ保証しない。

実装時は候補・trajectoryをprocess並列にし、process内でpreparation/oracleをcache、同一trajectory内のcosine/sineのeventを共有する。Qiskit内部のnum_processes=1は既存compiler profileの再現条件として保ち、独立trajectory側で並列化する。prototypeのn8限定launcherをそのまま起動しない。

## 7. 実行契約・SOURCE・seed・STOP

公開bundleに最小3候補/拡張7候補のDRAFT契約とINVALID grantを保存した。SOURCEは**将来のnative C adapterをsealするまでnull**、n_C・case・command・worker割当も未採用。現在の準備codeはsampling/compile能力を持たず、`--execute`は入力を読む前にexit2で拒否する。

既存`task_id`はsample<8、launcherは固定12群を検証する実装なので、Cには候補集合とn_Cを明示した別adapterが必要である。数学・係数・native kernelを変えず、CONFIRMATION stage/selection digest、fresh seed registry、exclusive grantと候補/replica/axis ledger、正常終了と全worker回収証拠を接続する。その未実装部分を今回「完成」とは扱わない。

seed namespaceはSOURCE・承認plan digest・CONFIRMATION・candidate・replica・outer・piece・input SHAを含める案。実seedはまだ生成0。旧成功/失敗native namespace、追加探索1824 seed、既知legacy/reference1505値と、新C registry内部の衝突を起動前に再検査する。過去の照合件数だけで将来の新registryをPASSにしない。

budgetは`sum(n_C)` trajectory・`2*sum(n_C)` cost-wrapper軸、通常trajectoryのfull-wrapper同一軸でcontrolを行う案。追加の科学compile/controlが必要なら、事前に別登録して承認範囲へ含める。control基準は保存1e-9等を維持し、未知のmodel誤差2%基準をこの標本確認へ流用しない。

SOURCE/input/compiler/runtime/event lawの不一致、実OOM・自身process異常・nonfinite・軸/seed/budget違反ではSTOP。上位cgroupの一時PSI・CPU競合・他worktreeの通常更新だけでは強制停止しない。登録n_Cを終えた時点でmandatory STOP。差が識別できなければINCONCLUSIVEとし、結果を見てn・候補を自動追加しない。再fit・retry・q再選択は禁止。

最終評価手順も採用時に固定する。差の点推定と独立候補SEを基礎に、Welch型区間などを採用する場合は近似として明示し、希少tailを含む厳密coverageとは呼ばない。正常終了と統計的有意性は別の判定である。

## 8. 技術確認と次の作業

新15人工テスト・既存32人工、合計47件PASS。小さい希少確率の保持、差の区間半幅式、分散感度、pair配分条件、資源計数、未承認/未実装の起動拒否を検査した。metadata71fileを固定し、private replayの内容は公開せずSHA/集計だけを公開する。AI subagent・独立reviewは0。これは作者自身の技術確認である。

[公開bundle](../../artifacts/resource_applicability/track_a_h46_higher_pf_confirmation_readiness/2026-10-18/README.md)。既存SOURCE/結果/grant/ledger/STOP・取得/統合branch・他worktreeは保全した。新C sampling/compileを起動せずmandatory STOP。

**ユーザー**：最小3候補か拡張7候補か、観測差/半差など検出したいeffect、名目power・区間半幅・感度条件と配分案を選ぶ。標本数はまだ採用していない。

**Codex**：選ばれた条件を基にC native adapterと最終SOURCE/入力seal・固定seed registry・実行契約を完成させる。具体的な実行準備報告の後、ユーザーの別途明示承認を受けた場合に限って新sampling/compileを一回起動する。
