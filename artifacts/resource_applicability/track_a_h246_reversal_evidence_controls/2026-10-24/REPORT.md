# H2/H4/H6 次数依存の費用逆転：証拠範囲と最小追加検証

2026-10-24。**H4/H6・ε_sig=.001では、固定保存値上のFull S2低費用・PR標準Yoshida4低費用という逆転を確認できる。ただし、今回の標準Yoshida4/q4の独立確認と母平均順位の保証は未取得。** 同一qの補助対照はq4/q8の計8比較で保存値が揃う。H2の他4精度も入力は揃うが、受入済みDIAGNOSTICS政策の16 endpointは新規会計が必要。今回の新科学call・grant発行は0、旧strict COMPAREはBLOCKED、mandatory STOP。

## 固定範囲と引用元

linear H2/H4/H6、隣接1.0 Å、STO-3G、tol-only DF1e-8、rank3/10/19、T=.8、P=0、α_axis=.025、USUAL_SIGNED。Full prefix3/10/19、PR prefix1/3/5、PR K4・ABS_TIME_PER_OUTER32。symmetric-directional、状態準備を除くfull cosine/sine Hadamard wrapperのcompiled RZ費用。主検証はH4/H6 ε=.001、補助H2 ε=.0005、弱いH4 ε=.005を分離する。DF返却順・signed weights・元入力・候補・cohort・empirical allowanceを維持した。

- 最新解析は`d2536c4f959bfb4edd7d2deeceb155585fb53eb0`。[元REPORT](../../track_a_h246_order_cost_analysis_run01/2026-10-24/REPORT.md)の科学SOURCE `2e59a55beec8159c7832ff4ff02513b8976c0eb9`、契約SHA `2478152414dce3c468e48c17e04d0addf3143794190be686593bc94eae1708e8`、結果SHA `3614edf7afb06891428b48a0a8969a56b7bf699b3647cf7a1cdb3cd0df3ab6c0`を照合。
- 過去4方式はH2 `9e5ea6b235d7beecc02651b1393b561948e85789`、H4 `868ee7ced6818b2531a3aa9c12498f02bce55dc3`、H6 `77ea542152de1c9ff796aefaf9be91c68c7ff1cd`。これらと最新解析で封印済みの入力から引用し、旧結果を再計算していない。
- Full588候補`f75a7c25d584b69d90de454731a7f343a421e059`、高次PR保存費用`8613b3ecda2030a677b1690fe7aa3da23dec4873`、独立確認`3eca46148070ba4d4b0f5815ef45943407cc959e`、旧H6 `3cccfcb2051dacf29e91473e2e1152acef1574bd`、3環境索引`6dafed86cc82149d55b4e6395382953d5f3631d7`を限定照合。[全出典とSHA](source_catalog.json)。
- 共通運用指示2026-10-23の§8.1/8.4/10.1/12を確認。指定の§10.3は提供された正式版に存在せず未取得。前回GPTレビュー本文の別ファイルは未取得であり、今回の利用者指示にある要約を前提とした。詳細レビューは開始していない。

## A：H4/H6・ε=.001の保存証拠

| 系 | 方法 | 公式 / q | prefix | n | 保存C_cos=C_sin | 保存N_cos / N_sin | 保存G | 保存paired G SE |
|---|---|---|---:|---:|---:|---|---:|---:|
| H4 | FULL | S2/8 | 10 | 1 | 62571 | 24225252 / 18415797 | 2668093076979 | 非該当 |
| H4 | FULL | Yoshida4/4 | 10 | 1 | 93799 | 18013027 / 17606209 | 3341048717564 | 非該当 |
| H4 | PR | S2/8 | 3 | 48 | 51251.0 | 24249726 / 18445335 | 2188164571311.0 | 15629532287.03437 |
| H4 | PR | Yoshida4/4 | 3 | 33 | 49547.666666666664 | 19240638 / 18807773 | 1885209985424.3333 | 14504412056.459614 |
| H6 | FULL | S2/8 | 19 | 1 | 311350 | 22822005 / 27400849 | 15636885592900 | 非該当 |
| H6 | FULL | Yoshida4/4 | 19 | 1 | 469386 | 17929288 / 18185400 | 16951728941568 | 非該当 |
| H6 | PR | S2/8 | 5 | 33 | 184315.9696969697 | 22907931 / 27494282 | 9289932763968.213 | 76211309054.89842 |
| H6 | PR | Yoshida4/4 | 5 | 33 | 186918.57575757575 | 21115055 / 21415717 | 7949791328110.182 | 41565857890.619064 |

候補ID・Hamiltonian/state/snapshot SHA・compiler/native・reference・bias・log_B・margin・u・分散/共分散・stratum・出典を[候補別JSON](primary_and_weak_evidence.json)に対応付けた。[8候補CSV](primary_candidates_8.csv)は保存fieldの転記。Fullは確定した回路費用のn=1であり、PRのMonte Carlo SEをFullへ付け替えない。analytic shot数も実測shotではない。

保存解析のΔG=Yoshida4−S2は、H4 Full `672955640585`、PR `-908863757660/3`、H6 Full `1314843348668`、PR `-44224667383315/33`。PRの点差は各保存SEと比べて明瞭に離れるが、これは有限標本の記述であり、新しい差SE・有意性検定・CIは算出していない。候補間共分散を仮定せず、母平均優位を認定しない。Fullの近接する保存順位の感度とPR標本誤差は別問題である。

同じ系ではHamiltonian・状態・rank政策・T/P・compiler fingerprint `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`・native `M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1`・費用単位は対応する。FullとPRではprefixと近似/normalizationが異なる。各方法内のS2/Y4でprefixは固定だがqは8→4、signed piece/RTE配分も変わる。次数単独の因果効果にはしない。

Fullのbase log_Bは0。H4 PR S2/Y4の保存log_Bは`0.0008505094007942772 / 0.03301794834187473`、H6は`0.0021078034526976377 / 0.08181069434428587`。uは全8候補で`1e-9`。H4/H6のFull・PR間でreferenceのbinary64末尾差が残る（例：H4 real `-0.16160185891555842 / -0.16160185891555753`、H6 real `-0.8510893550745258 / -0.8510893550745398`）。方法内では一致を検査した。新しいreferenceや誤差boundを計算せず、条件付きの保存政策を維持する。

元cohortはEXPLORATIONで、探索結果から選択後に今回の解析候補を固定した。独立な事前登録試験とは呼ばない。保存stratumは4 PR群ともS0のみ（n48/33/33/33）で、S2_single/S2_multi/S4の観測数0は母確率や費用0の証明ではない。未観測層の費用/母分散boundはnullのまま。[元代表RTE表](../../track_a_h246_order_cost_analysis_run01/2026-10-24/representative_RTE_6.csv)ではH4/H6 S2は256 event、Y4は128、Y4の各outer配分10/12/10。最初の登録軌道だけの観測で、全軌道stream検証やoperation別RZ寄与の確定ではない。

H4 ε=.005のPRはS2 G=`74588809113.0`、SE=`532769891.0619943`、Y4 G=`74455179604.66667`、SE=`572842608.0242287`。保存ΔG=`-400888525/3`は各保存SEより小さい。**点順位だけを記載し、頑健な逆転例には数えない。** H6 Fullの旧中断stageからのcomplete pair受入と、元parent exit/reap不明も未解決のまま保持した。

## 独立確認の実際の範囲

[照合JSON](independent_confirmation_scope.json)はcandidate-law一致と同一endpoint政策を分離している。

| 固定資料 | 保存された独立PR群 | 今回への適用 |
|---|---|---|
| 3eca・H2 | S2/q4 n24、標準Yoshida4/q2 n28、ε=.0005 | 同じ候補lawの補助的な独立方向再現。現DIAGNOSTICSとは凍結N/reference政策の同一性を認定しない |
| 3eca・H4 | S2/q8 n25、m5_best/q1 n9、ε=.0005 | S2の候補lawは対応。標準Y4/q4・ε=.001の確認にはならない |
| 3eca・H6 | S2/q8 n23、m5_best/q1 n9、ε=.0005 | 同上 |
| 旧H4主表・R0124 | S2 prefix3/q1/K2/r4 n1108、prefix2/q8/K4/r2 n78 | 今回の標準Y4とprefix/K/RTE政策が異なる |
| 旧H6確認・3ccc/R0139 | S2 prefix4/q2・q8/K2/r4 n48/104、B0I PF | 高次PF baselineは存在するがPRはS2。今回のPR標準Y4ではない |

3eca結果は新seed namespaceによる同じ一次担当のlocal独立標本であり、外部再現ではない。H2/H4の再構築NPZ容器SHAは旧と異なるが、保存科学配列/Hamiltonian/stateのhash一致報告がある。旧全seedとの完全非衝突、希少tail・母分散boundは未保証。旧正規近似区間をε=.001の標準Y4へ転用しない。

旧確認commit `0d3578a4584a924ed10bdd124db6093e02d90c94`はlocal未取得/GitHub422だったが、統合索引R0017に基づき`b8bd64339169ecdf61b90fb7b575a3937fdc0c8c`の保存結果写しをSHA `d0ab64c57b33adff384e2f1599881259875d9b38b689ad0a5bf62861181a92d1`で照合回収した。そこでもH4高次はm5、H6は元SIGBUS/NOT_EVALUATED。後の再構築完了結果と混ぜない。**調査した固定結果・既知索引には今回のH4/H6標準Y4の独立確認を発見できなかった。全保存環境で存在しないという証明ではない。**

## B：同一q補助対照

[16比較の利用可能性](same_q_availability_16.csv)、[保存値と条件](same_q_controls.json)。H4/H6×Full/PRの4系列それぞれでq1/2/4/8のS2/Y4が存在。q4/q8の8比較は、同一prefix/rank/入力/方法内reference/u/cohort・ε=.001、PR K4/ABS_TIME_PER_OUTER32を維持し、保存signal・bias・log_B・N・compiled C・G・適格性が揃う。**新規取得は不要。** 適格PR8候補の既存308 paired trajectory receiptは元SHA・SOURCE/input・無条件cohort・両軸compilerを直接照合した。数値平均/分散を再計算せず、単価のliteral paired入力をprivate manifestへ準備した。 q1は両公式不適格、q2はS2不適格・Y4適格で、欠測0補完や代替cohortで比較しない。

同一qでもsigned piece数・符号・RTE配分・normalizationは異なる。同一q対照はq交絡を限定する補助であり、次数だけの因果効果を完全には識別しない。主比較S2/q8対Y4/q4を維持し、今回新しい対照G比・差・順位を計算していない。公式間の標本は同じ探索campaign内でもtrajectory対応を仮定しない。

## C：H2の未評価4精度

[4候補の入力一覧](H2_missing_precision_inputs.json)。受入済みDIAGNOSTICSのsignal/bias・empirical allowance/log_B_margin・保存reference/normalization、単価/paired価格、主ε=.0005のN/G、元`u_aware_shots` kernelが揃う。主endpointを別政策へ置換せず、ε=.05/.01/.005/.001の16 endpointはN/G=nullの未計算として保持。Hamiltonian/reference再計算・新sampling・compileは不要だが、**新しいshot/G評価は別承認が必要**。既存共通PF表の別allowance結果を同一条件のendpointとして採用していない。

## D：追加検証の選択肢と推奨

| 案 | 確認すること・変わり得る判断 | 追加処理とSOURCE/入力 | 別承認・負担 |
|---|---|---|---|
| 1 既存証拠で整理 | 固定構成・条件付きの点推定逆転。母平均や次数単独の主張はしない | 今回の引用/照合のみ | 科学call0、追加実験不要 |
| 2a 同一q保存対照 | q差を限定しても会計方向が残るか。機構説明を補強/修正し得る | 上記8比較、元2e59の`h246_order_cost_analysis_backend.decompose`を再利用 | bootstrap1/load1/分解8。新shot/統計/sampling/compile0。最小の推奨補助解析 |
| 2b H2他4精度 | H2の精度依存性。H4/H6の独立再現性は変わらない | 現DIAGNOSTICS固定4候補×4精度、元`Backend.account`/`u_aware_shots`再利用 | bootstrap1/load1/account16/u_aware_shots16/axis_headroom32/分解8案。新sampling/compile0。補助目的を選んだ場合のみ |
| 3 独立確認 | 標準Y4/q4の独立平均でも方向が再現するか。探索選択の影響を減らす | H4 `M5_MIN_ab57ad5bd40dc583`・H6 `M5_MIN_37ff3428bb58f5cd`、prefix3/5、K4/outer32、ε=.001。取得SOURCE c028・元8825のpipelineを候補にする | 新独立sampling/build/compile/費用統計が必要。n・seed政策・実SOURCE/入力/正確なcall予算は未固定、現在は実行可能契約ではない |

**当面は案1で主張の範囲を限定し、機構説明を一段強めるなら案2aだけを先に選ぶのが最小。** H2拡張をH4/H6の主張の必須条件にせず、案2bは独立に選択できる。新規独立確認は「標準Y4の独立標本で再現した」と主張する場合に限って必要であり、現時点で全群の再sampling/compileを要求しない。

案3では、既存独立S2（H4 n25/H6 n23）の保存単価を、新ε=.001の固定Nへ結合できるか先に検証する。Hamiltonian/preparation/compiler/native/lawの条件が成立すれば、新取得の最小候補は上記標準Y4の2群のみ。旧ε=.0005のGをε=.001の結果として転用せず、S2再会計を含む新しい解析承認を必要とする。不一致なら影響群だけ追加S2取得を検討する。標本数は希望する主張/精度に基づきGPT・ユーザーが固定し、過去nや未観測tail=0から恣意的に決めない。Fullは対応する保存単価/shotの決定論的会計を用い、PRの標本SEを同じ不確かさとして課さない。

[固定範囲・予算DRAFT](additional_validation_DRAFT.json)は案2a/2bを分離し、数値production SOURCE=null・grant0の**非実行契約**とした。既存`Backend.account`はε=.0005固定なので、ε引数と候補+精度の出力名を明示する最小adapterが必要。選択後にCodexがSOURCE/input/runtime・内部call・受入/STOPを封印し、実行直前に一度だけ承認を求める。今回は実行承認を求めず、grantも発行していない。

## 実装・検査・保全と次の担当

新metadata collectorのSOURCEは`bb626fe8ed341d3ba4b7a46bf0b0f07c2bcd16c5`。既存`h246_order_cost_analysis`の固定参照・hash・Git回収処理を再利用し、同一q/公式/law照合と固定field引用だけを追加した。元科学kernelを変更せず、`--execute`は入力を読む前に拒否する。一次担当の人工6件PASS（不足/不適格を0にしない、law/reference/cohort/q差を拒否、m5を標準Y4にしない、未承認実行拒否）。[技術検査](technical_checks.json)、[metadata-only preflight](metadata_preflight.json)。

[終了保全監査](preservation_audit.json)でSOURCE/input/runtime・旧結果・grant/ledger/marker/STOP・既知主表/Figure2を含む1,731保護fileをSHA照合。別worktreeの通常作業やHEADは凍結しない。private raw/event/seed/内部絶対path・有効grantは公開対象に含めない。共有環境・他者process・共有venvの変更0。既存文書とmanifestの履歴は保持して今回の入口だけを追記した。

次の担当：**ユーザー・GPT**が案1の条件付き整理で十分か、案2aの補助解析や標準Y4の独立確認が必要かを判断する。**Codex**は選択された範囲だけ実行契約を完成し、別承認後に実行する。今回は[STOP](STOP.json)で終了し、新数値解析・sampling・compile・詳細レビュー・最終freeze・旧strict解除を開始しない。
