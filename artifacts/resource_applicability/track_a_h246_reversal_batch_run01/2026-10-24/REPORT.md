# H2/H4/H6 同一q対照・H2精度補完：承認済み一括解析 run01

**固定83イベントを一回正常終了した。同一qではH4/H6ともq4はFull/PR両方Yoshida4、q8は両方S2が低費用で、方向逆転はない。異なるqの主比較では逆転が残り、H2は今回補完した他4精度を含む登録5精度すべてでFull S2・PR Y4が低費用だった。** この結果は固定構成の費用方向に関するもので、qだけの因果効果や母平均優位を証明しない。

## 固定条件・実行証拠

linear H2/H4/H6、隣接1 Å、STO-3G、tol-only DF1e-8、rank3/10/19、T=.8、P=0、α_axis=.025、USUAL_SIGNED。Full prefix3/10/19、PR prefix1/3/5、K4・ABS_TIME_PER_OUTER32。主比較はH2 q4/q2、H4/H6 q8/q4。補助同一qはH4/H6 q4/q8、ε=.001。symmetric-directional、状態準備を除くfull Hadamard cosine/sine wrapperのcompiled RZ費用。候補を各精度で選び直していない。

| 実行識別子 | 値 |
|---|---|
| SOURCE | `1c5bb8a65a7aa7c90707b783eb5d6cf97f4d0bfe` |
| 契約SHA256 | `bae3aa50a3bda959c94c0149d3abb7f8d0cbbdc491cf0800e62d155622dfd2e5` |
| 入力SHA256 | `a8b55834c76ab24ea4efec81da5c957059c5ef81dd0dba43ff1bf20683770d83` |
| 元科学結果SHA256 | `e60cbce4933fda17dba5c8477f34cfb19559bec241e2a4297e249a480bb7bc94` |
| 実行 | native exit0/reaped、1.469秒、attempt1/retry0 |
| 実ledger | bootstrap1/load1/A分解8/B会計16/内部shot16/内部headroom32/B分解8/統合1＝83 |

[ユーザー承認](approval_receipt.json)、[直前preflight](fresh_preflight.json)、[終了](native_completion.json)、[台帳receipt](execution_ledger_receipt.json)、[固定契約・人工検査](../../track_a_h246_reversal_batch_ready/2026-10-24/REPORT.md)。新namespaceのgrantを承認後に一回消費し、旧grantは再使用していない。SOURCE/input/runtime1,007 fileと既知1,792 fileを保全。新sampling/build/compile/simulation/allowance/reference/Hamiltonian・標本統計・再ランキング0。旧strict COMPAREはBLOCKED、mandatory STOP。

## C：異なるqの主比較・3系5精度

比はYoshida4/S2。表示は保存されたexact比の小数6桁表記であり、exact値は[15セルCSV](primary_directions_15.csv)、C/N分解と差は[30比較CSV](primary_comparisons_30.csv)、単価・shot・bias/log_B/u/headroom・n/保存SEは[60 endpoint CSV](primary_endpoints_60.csv)。元22比較は引用し、H2他4精度の8比較だけを追加した。

| 系 | ε | Full低費用 | PR低費用 | Full G比 | PR G比 | 点推定の逆転 |
|---|---:|---|---|---:|---:|---|
| H2 | 0.05 | S2 | Y4 | 1.483250 | 0.773727 | あり |
| H2 | 0.01 | S2 | Y4 | 1.463405 | 0.763357 | あり |
| H2 | 0.005 | S2 | Y4 | 1.438694 | 0.750478 | あり |
| H2 | 0.001 | S2 | Y4 | 1.244859 | 0.649450 | あり |
| H2 | 0.0005 | S2 | Y4 | 1.013341 | 0.528773 | あり |
| H4 | 0.05 | S2 | S2 | 1.494288 | 1.027663 | なし |
| H4 | 0.01 | S2 | S2 | 1.475295 | 1.014677 | なし |
| H4 | 0.005 | S2 | Y4 | 1.451308 | 0.998208 | あり（弱い順位） |
| H4 | 0.001 | S2 | Y4 | 1.252223 | 0.861549 | あり |
| H4 | 0.0005 | Y4 | Y4 | 0.986417 | 0.679116 | なし |
| H6 | 0.05 | S2 | S2 | 1.498718 | 1.182401 | なし |
| H6 | 0.01 | S2 | S2 | 1.463410 | 1.154574 | なし |
| H6 | 0.005 | S2 | S2 | 1.419625 | 1.120083 | なし |
| H6 | 0.001 | S2 | Y4 | 1.084086 | 0.855743 | あり |
| H6 | 0.0005 | Y4 | Y4 | 0.704263 | 0.556473 | なし |

H2 ε=.0005のFull G比は1に近い保存会計であり、PRの有限標本SEと同じ不確かさでは扱わない。H4 ε=.005のPR点差は各保存SEより小さく、頑健な順位例にはしない。H4/H6 ε=.0005ではFull/PRともY4が低費用であり、逆転の有無を系の名前だけで説明できない。登録点間の境界は補間しない。

H2の新16 endpointはすべて宣言allowance下で適格だった。[新16行](H2_new_endpoints_16.csv)。受入済みDIAGNOSTICSの通常signed bias/reference/allowance/log_B_marginと元compiled単価を使い、共通PF表の別政策を代入していない。旧ε=.0005は再計算0。H2 PRの単価標本はS2 n33/Y4 n34で固定、新GのSEは未計算/null。単価の保存SEは引用するが、旧主精度のG SEを他精度へ流用しない。

## A：同一qの補助対照・ε=.001

| 系 | q | Full低費用 | PR低費用 | Full G比 | PR G比 | 逆転 |
|---|---:|---|---|---:|---:|---|
| H4 | 4 | Y4 | Y4 | 0.813321 | 0.557320 | なし |
| H4 | 8 | S2 | S2 | 2.460747 | 1.634392 | なし |
| H6 | 4 | Y4 | Y4 | 0.209934 | 0.168232 | なし |
| H6 | 8 | S2 | S2 | 2.094025 | 1.526373 | なし |

| 系 | 方法 | q | C_eff比 | N_total比 | G比 |
|---|---|---:|---:|---:|---:|
| H4 | FULL | 4 | 2.983524 | 0.272604 | 0.813321 |
| H4 | FULL | 8 | 2.990187 | 0.822941 | 2.460747 |
| H4 | PR | 4 | 1.910861 | 0.291659 | 0.557320 |
| H4 | PR | 8 | 1.923968 | 0.849490 | 1.634392 |
| H6 | FULL | 4 | 3.006764 | 0.069820 | 0.209934 |
| H6 | FULL | 8 | 2.994411 | 0.699311 | 2.094025 |
| H6 | PR | 4 | 2.039118 | 0.082502 | 0.168232 |
| H6 | PR | 8 | 2.018374 | 0.756239 | 1.526373 |

全8比較でexact軸別積とC/N/crossの加法恒等式が成立した。[8比較の差・寄与・n/保存SE](same_q_comparisons_8.csv)、[4系qセル](same_q_directions_4.csv)。同一qではY4の単価増加をq4のN削減が上回る一方、q8では上回らなかった。Full/PRの両方でこの方向が揃った。元のS2/q8対Y4/q4では、異なるqでのN/Cを比べるため方向が異なる。

したがって、今回の逆転は「S2/q8とY4/q4を採った固定構成の資源比較」として成立し、同一qでFull/PRが逆方向になる現象は今回の保存範囲では支持されなかった。q/RTE構成の影響を限定する対照であり、signed piece・RTE配分・normalization・deterministic/random partitionも関わるため、qのみを原因と断定しない。主比較を同一q結果で置き換えない。

## C/N/Gと保存診断の対応

費用の正本は`G=Σ_a N_a C_a`。新16比較のexact分解は既存2e59の関数をそのまま使用した。H4/H6主比較のPR C比は元解析でH4が1未満、H6が1超だが、ε=.001では両者ともN削減によりY4 Gが低い。同一qではC比もN比も主比較とは異なり、Cだけの説明は成立しない。

Fullのbase log_B=0、PRはY4で保存normalizationが増える。保存bias/uとheadroomの改善、normalizationの増加を同時に含む元shot会計を維持した。新H2のheadroom/Nは各εの実評価出力から引用する。旧代表RTE6軌道では主比較S2→Y4でevent数が半減し、Y4各outerは10/12/10配分だったが、同一qはsigned piece構成が異なり、event数だけをcompiled RZへ換算しない。[元構成・診断](../../track_a_h246_order_cost_analysis_run01/2026-10-24/REPORT.md)。新しいsignal・回路・operation寄与の計算は行っていない。

## 証拠の強さ・Dの独立確認準備

主表は探索標本を選択後に固定した点推定である。H2の別確認には同じ標準Y4 candidate-lawの支持があるが、現DIAGNOSTICSの各endpointと同一policyの独立再現と認定していない。H4/H6の既存高次確認はm5_bestであり、今回の標準Y4/q4へ転用しない。PRの保存G SEは回路費用のMonte Carlo誤差であり、shot誤差・PF bias・母平均の保証ではない。新CI/検定/差SEを算出せず、候補間共分散を仮定せず、未観測stratumの費用や母確率を0にしない。

[D準備](independent_confirmation_plan.json)。既存独立S2はH4 n25/H6 n23、law/input/preparation/native数値SOURCE/compilerが静的対応したので、再取得を既定としない。H4のNPZ容器SHA差は科学配列一致と区別する。新ε=.001のS2 G/SEへ再会計する場合は別契約が必要。標準Y4を独立確認するなら最小新取得候補はH4/H6のq4の2群、必要軸は`2×(n_H4+n_H6)`。nは未決定で、求めるMC精度・統計目的・tail限界・cohort/seed除外範囲・非適応STOPを先に固定する。

今回の限定された主張を記述するために新規取得を必須とはしない。独立した標準Y4方向再現を主張に含めるかをユーザー・GPTが判断する。同一qで逆転が消えた結果に対し、標準Y4の独立標本追加だけで次数単独の因果効果が証明できるとはしない。新独立sampling/compileや詳細レビューは開始していない。

## 保全・公開範囲・次の担当

[検証](result_verification.json)、[保全監査](preservation_audit.json)、[file目録](file_inventory.json)。元科学結果はprivate領域でbyte保全し、[公開JSON](analysis.json)は軌道ごとのprivate単価配列/内部入力を省いたprojectionとした。元resultのH2行が保持する説明fieldはε=.0005の保存元contextであるため、公開側では`saved_base_explanation_0005`と明記した。新精度のheadroom/log boundは`accounting.shots.axes`が正本で、CSVもその値を引用する。科学値は変更せず、SOURCE/元結果/台帳/旧STOPを保全した。

Full/PRの保存reference末尾差、empirical allowance、有限標本・事後選択、H6旧中断stageの元parent exit/reap不明を保持する。新規結果を外部再現・独立担当の科学レビュー・厳密certificateへ格上げしない。論文主表/Figure2・共有環境・他ユーザーprocessを変更していない。

次：**ユーザー・GPT**が固定構成とq対照を踏まえた主張範囲、および標準Y4独立確認の必要性を判断する。**Codex**は追加実行を開始せず、今回のA/B/C・D準備を完了し[STOP](STOP.json)。独立確認を選ぶ場合のみ、別SOURCE/input/call/標本設計を固定して承認を受ける。
