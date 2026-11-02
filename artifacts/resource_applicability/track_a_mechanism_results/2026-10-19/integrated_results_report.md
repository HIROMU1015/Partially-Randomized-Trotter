# Track A Mechanism Analysis：固定32比較のone-shot結果

2026-10-19 JST。**正確な費用coreは32/32完了。normalization／headroomの説明は完了0、未解決28比較、入力欠測4比較。mandatory STOP。**

ユーザーの明示承認に基づき、SOURCE `5c256067db29a326bcaa0a0e244b38f6b87bd138` と準備時に固定したprivate契約・入力bindingのまま、保存値解析を1回実行した。launcher／workerはnative exit0・wait/reap済み。残り212比較はmetadataの保全のみ。今回の結果はローカルの保存値解析証拠で、immutable CIや別担当者による再現ではない。

## 条件と読み方

線形H4/H6/H8、隣接距離1 Å、STO-3G、T=.8、P=0、通常signed bias、allowance倍率1。H4 legacy固定DF12・実測PF subset16、H6 tol-only DF19・強化PF136、H8 tol-only DF28・強化PF184。3系を同一DF政策として扱わない。分割prefix・outer PF・q/r/Kは[固定候補IDとcohort](comparison_metadata.csv)の登録スケジュールを保持し、新しいdelta窓・分割・候補探索は実施していない。

単価はfull Hadamard wrapperのcompiled RZ、費用は保存解析shotとの積。取得compiler条件はQiskit 1.3.0、rz/sx/x/cx、optimization_level=1、seed_transpiler=17、backend/coupling=None。H8の元profile bytesと旧出力の完全同一性が未結合という既知の制約は保持する。

新Aは探索n8。H4旧n32は選択・推定兼用fixed anchorで`H4_PRE_D_P_VALIDATION`、H6/H8旧n32はconfirmation union内での選択を含む。H8独立n48の2比較は固定対照との主評価で、n8／n32と合算しない。既存の再利用replicaは元の48軌道内で一度だけ数える。H4の別D/P作業のSOURCE・結果・予定jobは変更も統合もしていない。

## 正確なcore

`G_m=Σ_a N_m,a C_m,a`、`R_G=G_PR/G_PF`、`R_N=Ntot_PR/Ntot_PF`、`Ceff_m=G_m/Ntot_m`、`R_Ceff=Ceff_PR/Ceff_PF`。

固定SOURCE内で、同一保存cohortのpaired軌道から軸単価平均とGを再構成し、保存exact G・保存削減率との完全一致、`R_G=R_N×R_Ceff`、`R_G=Σ_a w_PF,a R_N,a R_C,a`の有理数での完全一致を32件すべて確認した。64軸行は比較ごとのcosine/sineであり、独立標本数ではない。

[exact比率CSV](mechanism_ratios.csv)のfractionが本projectionの数値表現の正本。以下の小数はその表示丸め。削減率は保存された`100(1−R_G)`の出力で、正ならPR点費用が低い。旧／新の候補再選択・ランキングは行っていない。

| 比較ID | 系 | ε | cohort | R_G | R_N | R_Ceff | 削減率 |
|---|---|---:|---|---:|---:|---:|---:|
| A2_NEW_PR | H4 | 0.05 | 新A n8 | 0.839780 | 0.847097 | 0.991362 | 16.0220% |
| A2_OLD_PR | H4 | 0.05 | 旧PR n32 | 0.806538 | 0.764253 | 1.055328 | 19.3462% |
| A6_NEW_PR | H4 | 0.01 | 新A n8 | 0.743042 | 0.935003 | 0.794695 | 25.6958% |
| A6_OLD_PR | H4 | 0.01 | 旧PR n32 | 0.519082 | 1.054935 | 0.492051 | 48.0918% |
| A10_NEW_PR | H4 | 0.005 | 新A n8 | 0.493100 | 1.227186 | 0.401813 | 50.6900% |
| A10_OLD_PR | H4 | 0.005 | 旧PR n32 | 0.499015 | 1.027261 | 0.485772 | 50.0985% |
| A14_NEW_PR | H4 | 0.001 | 新A n8 | 0.534926 | 1.002989 | 0.533332 | 46.5074% |
| A14_OLD_PR | H4 | 0.001 | 旧PR n32 | 0.562605 | 0.929756 | 0.605111 | 43.7395% |
| A18_NEW_PR | H4 | 0.0005 | 新A n8 | 0.450598 | 1.683394 | 0.267672 | 54.9402% |
| A18_OLD_PR | H4 | 0.0005 | 旧PR n32 | 0.473666 | 1.559663 | 0.303698 | 52.6334% |
| A42_NEW_PR | H6 | 0.05 | 新A n8 | 1.768195 | 1.117329 | 1.582520 | -76.8195% |
| A42_OLD_PR | H6 | 0.05 | 旧PR n32 | 1.132901 | 1.237181 | 0.915712 | -13.2901% |
| A46_NEW_PR | H6 | 0.01 | 新A n8 | 0.845237 | 0.825738 | 1.023615 | 15.4763% |
| A46_OLD_PR | H6 | 0.01 | 旧PR n32 | 0.691095 | 1.162617 | 0.594430 | 30.8905% |
| A50_NEW_PR | H6 | 0.005 | 新A n8 | 0.707204 | 1.369187 | 0.516514 | 29.2796% |
| A50_OLD_PR | H6 | 0.005 | 旧PR n32 | 0.650347 | 1.100073 | 0.591186 | 34.9653% |
| A54_NEW_PR | H6 | 0.001 | 新A n8 | 0.691020 | 1.002024 | 0.689625 | 30.8980% |
| A54_OLD_PR | H6 | 0.001 | 旧PR n32 | 0.688323 | 0.928421 | 0.741391 | 31.1677% |
| A58_NEW_PR | H6 | 0.0005 | 新A n8 | 0.900234 | 2.611278 | 0.344749 | 9.9766% |
| A58_OLD_PR | H6 | 0.0005 | 旧PR n32 | 0.896743 | 2.419530 | 0.370627 | 10.3257% |
| A82_NEW_PR | H8 | 0.05 | 新A n8 | 0.968916 | 0.550012 | 1.761628 | 3.1084% |
| A82_OLD_PR | H8 | 0.05 | 旧PR n32 | 1.080230 | 0.707101 | 1.527689 | -8.0230% |
| A86_NEW_PR | H8 | 0.01 | 新A n8 | 0.681613 | 1.034495 | 0.658885 | 31.8387% |
| A86_OLD_PR | H8 | 0.01 | 旧PR n32 | 0.667627 | 1.174092 | 0.568633 | 33.2373% |
| A90_NEW_PR | H8 | 0.005 | 新A n8 | 0.700074 | 1.062514 | 0.658885 | 29.9926% |
| A90_OLD_PR | H8 | 0.005 | 旧PR n32 | 0.685451 | 1.205437 | 0.568633 | 31.4549% |
| A94_NEW_PR | H8 | 0.001 | 新A n8 | 0.820481 | 1.603745 | 0.511603 | 17.9519% |
| A94_OLD_PR | H8 | 0.001 | 旧PR n32 | 0.768718 | 1.355974 | 0.566912 | 23.1282% |
| A98_NEW_PR | H8 | 0.0005 | 新A n8 | 0.968822 | 1.276307 | 0.759082 | 3.1178% |
| A98_OLD_PR | H8 | 0.0005 | 旧PR n32 | 1.508125 | 3.985626 | 0.378391 | -50.8125% |
| H8_CONFIRM_1 | H8 | 0.05 | 独立n48 | 0.970641 | 0.550012 | 1.764764 | 2.9359% |
| H8_CONFIRM_3 | H8 | 0.0005 | 独立n48 | 0.966974 | 1.276307 | 0.757634 | 3.3026% |

## H8独立n48の2比較

| ε | PR | 固定PF | R_N | R_Ceff | R_G | 削減率 |
|---:|---|---|---:|---:|---:|---:|
| 0.05 | B2_p8_q2_2nd_r4_K4 | B0I_p11_q1_2nd | 0.550012 | 1.764764 | 0.970641 | 2.935924% |
| 0.0005 | B2_p7_q16_2nd_r4_K4 | B0I_p15_q4_4th | 1.276307 | 0.757634 | 0.966974 | 3.302597% |

ε=.05では保存shot合計の比が1未満、有効単価比が1を超える。ε=.0005では保存shot合計の比が1を超え、有効単価比が1未満となる。それぞれの積が観測費用比を正確に説明する。Ceffは軸別shot構成にも依存する量であり、一般の単純平均回路比へ置き換えない。normalization／bias／headroomがこのshot差を生むという説明は、下記guardを通過していないため確定しない。

## 説明側は未解決

| 比較status | 件数 |
|---|---:|
| EXPLANATION_COMPLETE | 0 |
| UNRESOLVED | 28 |
| EXPLANATORY_INPUT_MISSING | 4 |

64の方法sideでは、PR28 sideに`NORMALIZATION_LOG_BINDING`、PF32 sideに`LOG_BOUND_MISMATCH`、PR4 sideに入力欠測が記録された。前者は固定SOURCEのsaved B／log_B一致guard、後者は説明式と保存log shot boundの一致guardでの拒否理由である。原因を数値的に追加調査したり、guardを緩めたり、欠測を逆算・補完したりしていない。最初に失敗したguardの記録であり、後段のheadroom／整数丸めの一致も未認定。

準備時の「60 sideに必要fieldが存在」は数値説明PASSを意味しなかった。今回その60 sideすべてが説明guardで未解決になった。core32件の受入と説明側の認定は別statusとして保持し、normalization²／headroom²／整数丸め因子の科学的分解は本結果から主張しない。

欠測は`A18_OLD_PR`（H4 ε=.0005）、`A58_OLD_PR`（H6 ε=.0005）、`A82_OLD_PR`（H8 ε=.05）、`A98_OLD_PR`（H8 ε=.0005）のPR側。詳細は[説明status CSV](explanation_status.csv)。欠測0埋め・saved shot置換・再丸めは0。

## 受入・制約・STOP

実行前後で固定SOURCE、42入力role、Pythonと66 runtime file、grant／ledger／marker、結果SHA、32 IDs、出力schemaを照合した。保存SEは文字列として元入力と同一で、新SE／CI／p値／bootstrap／選択補正は0。新sampling／signal／compile／ランキングも0。旧STOP・無効grant案・元結果は保全し、再試行やgrant返却は行わない。以前の準備文書の未承認statusは履歴としてそのまま保持する。

単価にmodelを含む行は`VALIDATED_MODEL_ESTIMATE`を保持し、直接実測へ読み替えない。H4は暫定系列、全PF領域の最小値を意味しない。H6/H8のmodel条件、探索・選択効果、有限標本、高次event未観測の制約を保持する。母平均優位性・rare tail保証・因果効果・3点scaling・FT／非Clifford資源・最終全RPE総費用には拡張しない。

承認receipt、契約/input hash、private入力・軌道・raw shot、grant／ledger／marker、native receipt、正本`results.json`／`mechanism_axes.csv`／`mechanism_totals.csv`はprivate専用run領域に保存した。本branchは比率・公開可能な科学metadata・statusの軽量projectionのみで、公開pushは未実施。

次担当：**ユーザー**。説明側のSOURCE／入力binding診断と修正準備を別途指示するか判断する。Codexの追加実行・修正・再解析は停止中。再実行する場合は、今回の原結果・消費済みone-shotを保持し、新SOURCE／契約／対象と別の実行承認を必要とする。212補足・H4 D/P・新統計・公開pushの自動承認はない。
