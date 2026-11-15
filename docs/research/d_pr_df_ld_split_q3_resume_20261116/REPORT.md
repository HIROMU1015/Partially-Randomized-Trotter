# PR-DF LD1/LD2：固定4配分のQ3再開結果

**新規Q3を4件、各候補・配分につき1回実行し、全4件で採択済み閾値へのモデル適合を確認した。** 保存V0を加えた各LDの3配分の範囲では、両LDともB_PRICEが最小適合モデル費用。LD1は189,232,849,610.910739 RZ、LD2は46,341,978,336.212733 RZである。価格は既存の探索モデル参照で、正式E[C]・G・方式winnerは認定していない。

保存H4 linear 1.00 Å／STO-3G／tol-only DF10、PR-DF L_D=1/2（one-bodyは本数に含めない）、元fragment順、S2、Δ=0.1、K2、元κ・候補固有schedule/normalization/signalを対象とした。固定価格・整数配分は4ef70274、Q3入力は4d3aa8a6を直接再利用した。新sampling・build/compile・fit・価格再予測・配分再生成・長round取得・LD3は全て0。

## 6配分の結果

| LD | 配分 | 実際の固定モデル費用 RZ | Q3 MSE下界 Ha² | Q3 MSE上界 Ha² | 判定／出所 |
|---|---|---:|---:|---:|---|
| 1 | V0 | 261,314,830,475.559718 | 4.72462588934653801951643E-7 | 4.72462588934653960729534E-7 | 適合／保存V0再利用 |
| 1 | A_SHAPE | 193,529,481,006.191629 | 6.25468375335249038887699E-7 | 6.25468375335290388084314E-7 | 適合／新規1 call |
| 1 | B_PRICE | 189,232,849,610.910739 | 7.67458737806743819998359E-7 | 7.67458737806743978776249E-7 | 適合／新規1 call |
| 2 | V0 | 63,914,076,114.980849 | 4.72409731948051054908399E-7 | 4.72409731948051194818804E-7 | 適合／保存V0再利用 |
| 2 | A_SHAPE | 47,334,382,243.300078 | 6.25404466306816472494629E-7 | 6.25404466306840669256650E-7 | 適合／新規1 call |
| 2 | B_PRICE | 46,341,978,336.212733 | 7.67401652027366652784195E-7 | 7.67401652027366792694600E-7 | 適合／新規1 call |

全6件の判定は`MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`。不適合0、INCONCLUSIVE0、UNKNOWN0、欠測0。[SUMMARY.json](SUMMARY.json)にexact有理数の費用・MSE上下界と固定locator、[MENU.csv](MENU.csv)に比較表を保存した。数値表示は丸めたが、整数RZへ丸めて費用を作り直していない。費用欄は3/4予算額ではなく、その整数列の実際の保存モデル費用である。

判定条件は元のまま、MSE上界≤τで適合、下界>τでモデル不適合、跨げばINCONCLUSIVE。τ=15136964828156312001/5960464477539062500000000 Ha²（ε=0.00159360010199040 Ha）。total decoder `consecutive_dyadic_nearest_branch_principal_energy_total_v1`、保存DF10 solver proxy −2.1663874486347634 Ha、全事象平方energy loss、ideal finite-RTE rawのIID shots/axes/rounds、EXACT_AFFINE_OF_BINARY64_RAWを維持した。旧δEやδE²を追加せず、同名のLDでもPR-Pauliとは混ぜていない。

V0の各軸列 = `[64,60,56,52,47,43,38,34,30,25,21,17,12]`。
A_SHAPEの各軸列 = `[48,45,42,39,35,32,29,25,22,19,16,12,9]`。
B_PRICEの各軸列 = `[64,60,56,52,47,43,38,34,30,25,21,17,7]`。
X/YおよびLD間で同じ整数列になった固定入力を再生成せず使用した。LD1とLD2は各自のconfiguration・保存信号・lawに対して別callを実施した。各LDにV0/A/B以外の配分を加えていない。

## 各LDの有限集合内最小と長round依存

| LD | 選ばれた配分 | m0..m6モデル費用 RZ | m7..m12モデル費用 RZ | m7..m12 / 総額 |
|---|---|---:|---:|---:|
| 1 | B_PRICE | 198,979,534.093441 | 189,033,870,076.817298 | 99.8948493697037429356922% |
| 2 | B_PRICE | 66,223,444.937371 | 46,275,754,891.275362 | 99.8570983645607075586689% |

B_PRICEの選択X/Y列は両LDで上記の同じ13整数。最小費用のexact値はLD1 **34135453817125413895385/180388626432 RZ**、LD2 **29844234048521/644 RZ**。各LDで確認した3配分内の最小であり、全配分・全L_Dの最適性や単調性ではない。

m0..6/m7..12の加重項・費用は前回e0b3cad2の保存済み内訳をそのまま接続し、占比だけをexact有理数で算出した。新しい単価・総額・係数を生成していない。LD1は保存E_I[S]＋較正残差モデル、LD2は自身のfull-wrapper Cの3点モデル（offset0）という根拠の差がある。LD2 m7..12は未検証の外挿であり、Q3適合はこの価格を検証するものではない。両LDとも最小モデル費用の99.85%以上が長round側なので、前回の「LD3追加よりLD2長round価格の確認を優先」という研究上の推奨は維持できる。今回その取得には進んでいない。

## 実行環境と固定AST照合

既存の成功済みB2 package **738df226** の`TESTS.txt`から、Python **3.11.0rc1**、実行ファイル `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python`を特定した。同packageのEXECUTION_LOGは2 scientific callの結果とexit0を記録する。旧scientific RESULT自体にはsys.versionの別欄がないため、過去の実行環境の参照はこのpackage内検証記録に基づく。今回のINPUTS/EXECUTIONには実際のversion・executable・platformを直接保存した。

この既存環境はプロジェクト内の専用Python3.11.0rc1に結ばれたvenvで、そのまま利用できた。新規installや共有設定変更は0。kernel bytesは固定710aa69bと一致し、actual AST hashは元の期待値
`3d043e716246e57a8b05784b0d5c0f6777aaf43410fed8c4dd8b68df6dd590b8`
に一致。accelerator9e294778、既存route27e70efe、C c628e9e8を変更せず、期待hashの置換・type_paramsの除去・guard無効化・照合規則変更は全て0。[ENVIRONMENT.json](ENVIRONMENT.json)に出所と実確認を保存した。

新callerは前回e0b3cad2のworkerを固定Git blobから読み、同じSOURCE viewを使用する。相違は互換runtimeの指定、旧出力と分離した出力先、元固定入力の再利用、親supervisorがowned workerの終了code/signalを記録する部分のみ。数値kernel・価格・配分・信号・判定条件は変更していない。

## 科学call前の人工検証

[TECHNICAL_TESTS](TECHNICAL_TESTS.txt)は2件PASS。1件目は実runtime/constructorの固定AST gateと本番4件のmetadata対応。2件目は**本番と同じProcessPool→U.worker→P.worker→実Accelerator constructor→T.evaluate→API/RESULT/RESOURCE/PERSISTED保存**を、既存の2-round人工raw信号fixture・2子processで検査した。metadata intakeだけを人工configuration/viewへ差替え、constructor・数値評価・保存本体はmockしなかった。元fixture名はPauliを含むが保存Pauli/H4科学データを使用しない共通total-Q3の人工例である。

最初はfixtureに必要なconfiguration/signal_packet参照欄がなく、人工検査1件が失敗した。欄を補った後に2件を再確認し、初回logは[TECHNICAL_TESTS_INITIAL](TECHNICAL_TESTS_INITIAL.txt)で保全。人工検査を実H4のQ3実行や価格検証の証拠へ昇格していない。保存H4 P0演算子の回帰suiteは起動していない。Python3.11.0rc1のperiodic faulthandlerは使わず、協調的stage・round進捗を観測した。

## §10.5差分照合・取得状態

[DELTA_CHECK](DELTA_CHECK.json)は前回e0b3cad2 STOP以降のremote/ref・同じ候補/配分・job/output差分に限定した。新たなFull-H m0価格scenarioとp7 width-h Q3は別候補。今回の4件と同等の完了receipt・進行中job・未解決出所は0。旧2回の技術投入は科学call0として元出力に保全し、今回のみ再開許可の下で4件を各1回実行した。再利用V0は2件で、新しいV0 Q3は0。

| 固定case | 新規call | closure | 検査pair数 | total DP遷移数 | 取得・保存 |
|---|---:|---|---:|---:|---|
| LD1 A_SHAPE | 1 | VERIFIED_FINITE_BINARY64_SUPPORT | 907914 | 51433304 | API/RESULT/資源保存完了 |
| LD1 B_PRICE | 1 | VERIFIED_FINITE_BINARY64_SUPPORT | 2317210 | 93680072 | API/RESULT/資源保存完了 |
| LD2 A_SHAPE | 1 | VERIFIED_FINITE_BINARY64_SUPPORT | 907914 | 51433304 | API/RESULT/資源保存完了 |
| LD2 B_PRICE | 1 | VERIFIED_FINITE_BINARY64_SUPPORT | 2317210 | 93680072 | API/RESULT/資源保存完了 |

count-onlyは前回の固定結果を再利用し、新列のclosureは今回の各callで確認した。DPの遷移は固定上限141,559,036以内、非終端frontierは8192以下、元の質量/残余loss会計を保持した。zero/tieを含む全事象loss、元binary64 engineering allowanceを維持。MSE区間はこの条件付きモデル計算の上下界であり、価格のCIや物理精度の数値certificateではない。

各workerが原API_RETURN、PARTIAL_COARSE/LEMMA/TOTAL_DP、RESULT、RESOURCE_FINAL、PERSISTEDを保存し、親がSUPERVISOR_RECEIPTを保存した。API_RETURN[0]とRESULT.recordは4件とも完全一致。4つのouter workerのexitcodeは全て0、signalなし、errors/monitor_errorsとも0。個別DP childのOS exitcodeは既存acceleratorの記録範囲ではなく、今回追加したexitcode欄はouter Q3 workerのもの。[COMPLETION](run01/COMPLETION.json)の`process_exitcodes`はこのouter worker4件を記録している。

## 資源

4独立caseを並列、各8 DP worker・batch8という既存の性能検証済み設定を再利用した。これは恒常的な資源上限ではない。全体wall232.919794秒。以下はAPI/RESULT保存までのworker値。

| case | wall秒 | worker CPU秒 | child CPU秒 | process-tree CPU秒 | worker peak RSS KiB | 観測tree RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| LD1_A_SHAPE | 155.191 | 146.648 | 100.233 | 246.881 | 581632 | 4131292 |
| LD1_B_PRICE | 232.836 | 218.738 | 196.494 | 415.232 | 793500 | 5117604 |
| LD2_A_SHAPE | 150.864 | 144.540 | 100.020 | 244.559 | 571400 | 3713592 |
| LD2_B_PRICE | 227.617 | 210.156 | 195.856 | 406.012 | 824664 | 5363640 |

RSSのtree値は定期観測で、厳密な同時peakではない。`max_single_completed_child_peak_RSS_KiB`は個別childの最大で、tree合計ではない。処理中のcode/sourceは変更せず、他担当jobも停止・変更していない。

## SOURCEと引渡し

- 実行code: `005dc8c019061c07ba2ac8da389af0117f7aa27b` / `scripts/run_pr_df_ld_split_q3_resume.py`
- 再開INPUTS: `edf78cf5d43f707c59d7931d271a8d78163cac12` / `docs/research/d_pr_df_ld_split_q3_resume_20261116/INPUTS.json`
- raw公開: `75f416de6623a21a3b705e3f770a7c148325cb20` / `docs/research/d_pr_df_ld_split_q3_resume_20261116/run01`
- 保存結果join: `0c0395609a48b268b37c2f02412ff7188177e15a` / `scripts/summarize_pr_df_ld_split_q3_resume.py`

価格・配分の正本は4ef70274261cf18bd9356a15d780eb5b6b61c60cのALLOCATION_INPUTS、元Q3入力は4d3aa8a666e153188addfb3a4f2b8ef10733a3ddのQ3_CALL_INPUTS。固定locatorを[SUMMARY](SUMMARY.json)と[HANDOFF](HANDOFF.md)へ結んだ。原価格モデル・LD1 m3/m4独立holdout・旧raw・旧技術STOPは無変更。

## ロードマップ差分・mandatory STOP

統合担当は「LD1/LD2 A/Bの4件はAST gateで未開始」から「同じ固定4件をPython3.11環境で各1回完了、全適合、各LDの3点有限menu内最小はB_PRICE」へ更新できる。元STOPは履歴として残す。実行環境の待ち依存は解消したが、長round価格のモデル根拠、全law E[C]、CI、実PCG64 joint-law移送、正式G、最適L_D・winnerは未認定。

今回の新規Q3は4件、再sampling/compile/fit/単価予測/配分生成は0、科学retry0。公開・remote SHA/主要artifact一致を確認後mandatory STOP。追加Q3・長round取得・LD3や別の配分へ進まない。ロードマップ本体は統合担当が更新する。
