# H4固定h境界：保存データ再監査v2完了

ユーザー「次は、計算完了まで実行して。必要なら修正して。承認の確認はいらない。承認した」に基づき、準備済み固定checker v2の保存データ監査を一回実行した。**INDEPENDENT_BOUNDARY_JW_PADE_PHASE_COST_POLICY_AUDIT_PASS**。この実行中の技術修正・数値再試行は不要だった。旧v1のNUMERICAL_ALLOWANCE停止・source・一次データ・認可・消費markerは変更せず、新scopeの監査結果を追加した。

## 固定source・入力・比較条件

準備基準commit `d96c9c6cb654058201cc15688aa1422beedefac5`、checker-v2固定source `3953db1205f65ae6eda3e07b718c452f6e3d2f12`。元producer source `5e3341c12bd84f30f19072cefa590f4499c36c38`。v2 scope `QC_ONEBODY_REPARTITION_H_BOUNDARY_REAUDIT_V2_20261015`。新認可は今回のユーザー指示をsource/input/contract SHA・limitsへ結び付け、旧認可を流用していない。

linear H4、1.00 Å、STO-3G、保存DF rank12・generation順全12density。8spin/4spatial、Nα=Nβ=2/36dim、先頭一体/density pairの固定X、v1 state/seal/frame/leading不変。Hamiltonianはconstant I+dΓ(onebody)+ΣλdΓ(g)^2。PR L_D/delta scanではない。共通比較h=0.002745119306432635..1 Ha^-1、半刻みsupport最小0.0013725596532163176。既存762追跡点/381共通比較点を使い、h・再配分係数・候補を追加していない。

同じoriginal/response_selected/known_linear_extraction、S2/CS2/S4/EXT2（Richardson）対照、target10^-4/10^-5Ha、予算(det/stat/synth/info)=.5/.25/.125/.125、overlap≥.99、numericfloor10^-10Ha、controlled価格P1=(4,1,2)/P2=(8,1,4)、uniform4context10%gateを保持。E0は保存biasの監査だけに使用し、selector/aliasへ渡していない。

## 独立監査と実行費用

| 項目 | 実績 | 登録上限 |
|---|---:|---:|
| 独立PF再構成/完了cell | 4572 | 4572 |
| Padé expm/完了expm | 91401 | 571500 |
| cell/projector array loads | 9294 | 13716 |
| method records確認 | 4572 | 元保存scope4572 |
| cost rows確認 | 18288 | 元保存scope18288 |
| peak RSS | 265728000bytes（253.4MiB） | 512MiB |
| wall/CPU time | 31.49s / user31.22s+system.27s | capなし |

1worker/1thread、CPU SciPy、36dim保存sector、512entry factor-LRUを使用。32CPU/約64GBを確認、GPU driverは利用不可。固定条件を守り、speedup倍率は主張しない。新PFデータ収集・Schur・Hamiltonian eig・RDM・leading/selector/optimizer・分子生成・compile/T合成0。独立PF再構成は新しいデータ収集と分けて上記に計数する。v2 attempt1/retry0。

JW Hamiltonian再構成、保存H eigenpair・固定X、Padé PF照合、円上cluster/phase/alias/projector接続、保存行列の残差/unitarity/orthogonality/allowance自己整合、CS2/EXT2幅伝播、共通h/refinement、controlled primitive/query/repeat/information/synthesis会計、全対照の選択、10%gate、局所bracketまで完了した。

最大Padé対保存行列差は2.609927742554705e-13で、旧10^-9一致条件内。両行列から計算するallowanceの差の最大は2.476166831781919e-10で、旧同値比較10^-10を超える。保存行列自身の自己整合と独立照合を分離したv2では全cellが通過した。この観測は前回の静的なcross-matrix比較不整合を支持する。旧停止cellは旧handlerが記録していないため復元しておらず、特定cellについて旧エラー発生順序を実測したとは主張しない。旧checkerを再実行していない。

## 保存された比較結果の監査確認

新規最適化や再選択は行わず、元の保存比較結果を独立に確認した。全4contextで候補はCS2:response_selected、最強対照はCS2:known_linear_extraction。

| target (Ha) | 価格仮説 | 再配分候補h | 強対照h | 候補費用 / 対照費用 | 登録10%gate |
|---|---|---:|---:|---:|---|
| 1e-4 | P1 | .6018810386505993 | .5852343720673502 | .9709028655599986 | 未達 |
| 1e-4 | P2 | .6018810386505993 | .5852343720673502 | .9715851441831697 | 未達 |
| 1e-5 | P1 | .3384276539623714 | .3302093037931624 | .9742728053826825 | 未達 |
| 1e-5 | P2 | .3384276539623714 | .3302093037931624 | .9749574508296341 | 未達 |

離散grid・境界分類はともに **NO_UNIFORM_REGISTERED_COMPONENT_GAIN**。候補/対照の局所境界bracketは保存した基準で4/4context解決、次のsampleがBIAS_TOO_LARGE、相対幅≤1%。再配分候補が選択できるhは対照より大きいが、登録10%費用改善には至らなかった。これらはsampled local bracketsで、連続領域のglobal optimum・単調性・不存在証明ではない。

価格比はfixed-N scalar/basis共有/fusionを戻したcontrolled primitive/Heisenberg proxy。追加γ・leading係数の量子取得費用、状態準備・実compile/T・QPE/RPE全ground-task費用は未確定で、理想情報を無条件にゼロ費用としない。h選択は保存gridの事後oracleであり、真値なしの運用可能なprecision selectorではない。一分子development入力だけの局所結果で、新規性・他分子移送・独立生成speedup・研究採択/終了を確定しない。

## 証拠・状態

[今回の認可](../../artifacts/qc_onebody_repartition_h_boundary_reaudit_v2_execution/2026-10-15/authorization.json)、[prelaunch GitHub照合](../../artifacts/qc_onebody_repartition_h_boundary_reaudit_v2_execution/2026-10-15/prelaunch_remote_receipt.json)、[固定契約](../../artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15/contract.json)、[source freeze](../../artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15/source_manifest.json)、[result identity](../../artifacts/qc_onebody_repartition_h_boundary_reaudit_v2_execution/2026-10-15/result_identity.json)、[v2独立監査](../../artifacts/qc_onebody_repartition_h_boundary_reaudit_v2/2026-10-15/attempt_1/independent_audit.json)、[実測resource](../../artifacts/qc_onebody_repartition_h_boundary_reaudit_v2_execution/2026-10-15/checker_resource.txt)、[保存summary](../../artifacts/qc_onebody_repartition_h_boundary_reaudit_v2_execution/2026-10-15/saved_results_summary.json)、[実行・保全監査](../../artifacts/qc_onebody_repartition_h_boundary_reaudit_v2_execution/2026-10-15/execution_identity_caps_audit.json)、[旧一次JSON](../../artifacts/qc_onebody_repartition_h_boundary_v1/2026-10-15/attempt_1/raw.json)、[旧失敗記録](../../artifacts/qc_onebody_repartition_h_boundary_v1/2026-10-15/attempt_1/independent_audit.json)。

旧失敗時の文書・機械記録UNVALIDATED_PRODUCER_ONLYは履歴として保持。今回の固定保存scopeについて別記録でINDEPENDENT_V2_NUMERICAL_AUDIT_PASSへ進んだ。immutable source CI・外部科学再現とは呼ばない。旧5140protected files、160旧manifest prefix/non-result/global description FAIL、root434pathsを保全。

N1STOP/N2HOLD/N3CLOSED/THRIFT現template終了、TrackA/B・C1–C4・第二候補を維持。今回の計算は完了、GitHub公開・再取得後Mandatory STOP。次は結果の研究上の意味・第一候補の継続/終了/第二候補への移行をGPTが別開始承認で判断する段階であり、Codexはその判断を代行しない。追加計算や同一技術監査を自動で起動しない。

親rootのbaseline434pathのstatus/bytesは保持。監査中、対象外の別worktree directoryとTrack Bレビューが追加され全statusは436pathとなった。これらへ書込み・stage・commitせず、対象worktreeの固定source/inputだけを検査した。元baselineを新snapshotで置換していない。詳細は実行保全監査のroot_added_paths_outside_this_scope。
