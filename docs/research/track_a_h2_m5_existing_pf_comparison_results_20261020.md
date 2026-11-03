# H2 m5：既存PF条件での統合比較完了

統合SOURCE `3bdef997838c40201170d82b26b1cda83df4283e`、取得SOURCE `5d0be1699dcef8f09f3bdcd8bb00511640571809`。旧H2 D70/A66と、追加m5 D35/PR66を5精度で統合した。**段階6まで完了、mandatory STOP。**

H2直鎖1 Å・STO-3G・tol-only 1e-8/rank3・4 qubits/sector4、T=.8/P=0。旧入力・状態・reference・native preparationを再利用し、通常signed bias・経験的allowance倍率1・保存analytic shot式で評価した。H4/H6 outer32/n33は輸入していない。

| ε | 固定集合内の決定論最良 | PR探索点最良 | PR/決定論費用比 |
| --- | --- | --- | ---: |
| 0.05 | B1_p3_q1_2nd | B2_p1_q1_2nd_r2_K4 | 0.864678 |
| 0.01 | B1_p3_q1_2nd | B2_p1_q1_2nd_r2_K4 | 0.864442 |
| 0.005 | B1_p3_q2_2nd | B2_p1_q1_2nd_r2_K4 | 0.888195 |
| 0.001 | B1_p3_q4_2nd | B2_p1_q4_2nd_r2_K4 | 0.813584 |
| 0.0005 | B1_p3_q4_2nd | B2_p1_q4_2nd_r2_K4 | 0.813412 |

PR比は1未満ならPRの標本点費用が低い。これは候補選択を含む探索n8の結果であり、母平均順位・独立確認・全格子最適性を保証しない。両軸をtrajectory単位でpaired会計し、標本SEも保存している。未観測rare-event費用は補完していない。

| 公式 | ε | 決定論の点最良費用RZ | PRのn8点最良費用RZ ± SE |
| --- | ---: | ---: | ---: |
| 2nd | 0.05 | 5533984 | 4785111.5 ± 31086 |
| 4th | 0.05 | 14681766 | 未取得（既存H2にないYoshida4 PR） |
| 4th(m5_best) | 0.05 | 51860752 | 41799384 ± 1.6604e+05 |
| 2nd | 0.01 | 1.8815619e+08 | 1.6265006e+08 ± 1.0567e+06 |
| 4th | 0.01 | 3.8602839e+08 | 未取得（既存H2にないYoshida4 PR） |
| 4th(m5_best) | 0.01 | 1.2964042e+09 | 1.0448906e+09 ± 4.1506e+06 |
| 2nd | 0.005 | 1.1705347e+09 | 1.0396633e+09 ± 6.7541e+06 |
| 4th | 0.005 | 1.6478827e+09 | 未取得（既存H2にないYoshida4 PR） |
| 4th(m5_best) | 0.005 | 5.1856795e+09 | 4.1796212e+09 ± 1.6602e+07 |
| 2nd | 0.001 | 5.9673629e+10 | 4.8549488e+10 ± 1.2893e+08 |
| 4th | 0.001 | 7.4234357e+10 | 未取得（既存H2にないYoshida4 PR） |
| 4th(m5_best) | 0.001 | 1.29658e+11 | 1.0450263e+11 ± 4.1511e+08 |
| 2nd | 0.0005 | 3.0569508e+11 | 2.4865593e+11 ± 6.6036e+08 |
| 4th | 0.0005 | 3.0956125e+11 | 未取得（既存H2にないYoshida4 PR） |
| 4th(m5_best) | 0.0005 | 5.1871249e+11 | 4.1807158e+11 ± 1.6607e+09 |

新m5のrはsigned S2 piece単位で、総event数11qr。旧S2はqrであり、同じ総event予算の比較ではない。旧決定論S4はYoshida3段合成で、Yoshida4 PRは今回追加していない。

取得はm5決定論15pair＋PR528 trajectory＝543pair/1086軸、失敗0。保存signal101はSHA/生成関数ASTの同一性で再利用し、新signal計算0。旧探索・確認・前停止runのseedを除外し、新cohortを旧標本とpoolしていない。

fresh資源から60 workerを使用。取得runのwall 527.654秒、取得workerは全exit0/reap。同run全体はFP単価の結合漏れでSTOP/native exit1を保持し、別の統合専用契約が保存全543pairを受け入れた。統合は新sampling/compile0・native exit0/reap、wall 3.811秒。全3258 archiveで元byte SHA復元PASS、元byte 3,313,827,986、圧縮byte 201,733,166。RAMは[終了監査](../../artifacts/resource_applicability/track_a_h2_m5_existing_pf_comparison/2026-10-20/end_audit.json)の観測値で、厳密なピーク上界ではない。

[比較JSON](../../artifacts/resource_applicability/track_a_h2_m5_existing_pf_comparison/2026-10-20/comparison.json)、[候補別会計CSV](../../artifacts/resource_applicability/track_a_h2_m5_existing_pf_comparison/2026-10-20/accounting.csv)は1185行。旧D/Aの保存exact平均とSEが全行再現するgateを通過した。ineligibleと欠測は区別し、欠測を0補完していない。

[位相保持修正](track_a_h2_m5_phase_guard_20261020.md)は、固定Qiskit1.3.0の各翻訳passで元のgate/angle/wire列と一致することを確認しながらscalarを保持するprocess私有処理である。追加policyを旧compiler設定hashとは別に記録した。raw1e-7・他1e-9未満は不変、状態からの位相fit・postcompile補正はない。旧H2費用のnative受入範囲を後から新しい保証へ変更していない。初期化・compileの旧2件と取得run03の統合STOPはそのまま保持した。

[統合のみの別受入契約](track_a_h2_m5_saved_integration_contract_20261020.md)で旧Dと同じFP決定論6pairの参照を復元した。FP PR n1は統合せず、preparation一致・保存signal全記録一致・重複拒否を適用した。人工・回帰26件と追加認可2件がPASSし、元Dに再記録のないidentity/overlay aliasへの修正は会計回帰12件がPASSした。統合run01のmetadata STOPも保持し、別の統合run02が正常終了した。科学取得の再実行は行っていない。

H4/H6 DESIGN_PILOTとは契約・SOURCE・標本を分離している。H4完了/H6 cutoff STOPを本H2完了によってPASSへ変更しない。AD03、正式shot政策の新採択、詳細科学レビュー、最終freezeは未実施。

**次の担当：ユーザー。** この有限標本比較を確認し、段階7の事前固定候補・新cohortによる独立確認を実施するか判断する。Codexは自動的に追加sampling/compileへ進まない。
