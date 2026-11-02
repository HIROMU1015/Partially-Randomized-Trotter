# WP05-a full controlled interrogation接続

## 目的

WP06-bで固定した`support_run_le_1`を、中央RTEだけでなくcomplete controlled partial-$S_2$、
反復回路、cosine/sine Hadamard wrapper、ancilla測定まで伝播し、完全wrapperを直接transpileした。
同一finite-RTE trajectoryのfull basisと選択policyを対応付け、中央RTE差を加えるだけの旧bridgeが
wrapper境界を含めても5%以内か、$q=1,2$較正が未使用$q=4$を予測できるかを検査した。

これは状態準備なしの1 interrogation cost検証である。backend実行、量子shot、$alpha$・shot数の
再最適化、$q>4$の直接回路、最終総costは含まない。

## 固定条件

| 項目 | 条件 |
|---|---|
| 物理系 | H4直鎖、1.0 Å、STO-3G、8 qubit |
| DF・候補 | rank 12、$L_D=3$とtail-free $L_D=12$ |
| RTE | $delta=0.02$、$K=2$、$r=1,2,4,8,16,32$ |
| policy | WP06-bでholdout済みの`support_run_le_1` |
| calibration / holdout | $q=1,2$ / 独立seedの$q=4$ |
| 標本 | $L_D=3$は各$(r,q)$ 8 trajectory、full/policyを対応付け |
| wrapper | controlled partial-$S_2$、cosine/sine、ancilla Z測定、状態準備なし |
| compiler | Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなし |

$6\times3\times8\times2\times2=576$本のrandomized measurement-bearing wrapperと、
$L_D=12$の6本をtranspileした。深く入れ子になったQiskit定義の再帰serializationを避けるため、
cost測定時の回路identityには既に全下位semanticsを束縛するcompiler-independent fingerprintを渡した。
これはtranspile設定や得られるmetricを変えない。

## production経路の接続

`DFPartialS2StepRequest`と`DFPartialS2RepeatedRequest`へ明示的なRTE basis planを追加し、
`QiskitDFPartialS2CircuitBuilder`からRTE builderへ伝播するようにした。planなしの既定経路はfull basisの
ままであり、既存fingerprint payloadも変更しない。planありの場合だけpolicy IDとplan fingerprintを
partial-$S_2$ semanticsへ加える。

専用2-qubit testではcomplete controlled repeated partial-$S_2$のfull/policy operatorを全行列で比較した。
物理H4成果物では計算量を抑えた独立診断として、1本の固定複素ランダム状態へcontrolled evolutionと
cosine/sine wrapperを作用させ、最大差$1.08\times10^{-16}$を得た。relative ancilla phaseは一致し、
wrapperは追加controlを適用せず、cosineが実部、sineが虚部を読む規約も保持した。

## $q=4$ holdout

$q=1,2$の平均からaxis・metric別のaffine式を固定し、未使用$q=4$を予測した。

| 対象 | 最大絶対相対誤差 |
|---|---:|
| 選択policy、RZ count、全$r$・両軸 | 2.288% |
| 選択policy、全6 metric、全$r$・両軸 | 2.431% |
| full basis、RZ count、全$r$・両軸 | 4.234% |
| tail-free $L_D=12$、RZ count | 0% |

従って事前の5%基準を全て通過した。完全wrapper上の選択policyの直接RZ変化は、評価した$(r,q)$で
full basis比0.99--8.76%減だった。cosine/sineはこのcompiler条件では同じcostとなったが、回路は両軸を
別々に構築・transpileしている。

## additive bridgeとwrapper境界

WP06-bの独立schedule-transfer streamで測った中央RTEの平均差を$q$倍し、今回の完全wrapperで直接得た
full/policy差と比較した。差の絶対値をfull-wrapper平均で規格化した最大RZ残差は2.625%で、5%基準を
通過した。境界残差は非零なので完全な加法性ではないが、今回のH4、$q\leq4$、1 compiler条件では
追加の境界項を必須とする大きさではなかった。

## 固定WP04条件への順位感度

今回の$q=1,2$完全wrapper較正を、WP04のround、shot、$alpha$配分を変えずに長$q$へaffine外挿した。

| 候補 | RZ点推定 | local 5% + calibration区間 |
|---|---:|---:|
| $L_D=3$、full basis直接再較正 | $1.7692\times10^{12}$ | $[1.4426,2.0958]\times10^{12}$ |
| $L_D=3$、`support_run_le_1` | $1.5933\times10^{12}$ | $[1.3489,1.8377]\times10^{12}$ |
| $L_D=12$、tail-free | $1.6963\times10^{12}$ | $[1.6115,1.7811]\times10^{12}$ |

選択policyの$L_D=3$点推定は直接再較正full basisより9.94%低く、$L_D=12$より6.07%低い。
一方で区間は重なる。さらに$q>4$を直接transpileせず、$alpha$・shot・scheduleを再最適化していない。
従ってこれは非decision-gradeな順位感度であり、部分ランダム化の科学的優位性や最終総costではない。

## 判断と次の検証

`support_run_le_1`は完全wrapperへの接続、5% additive-bridge基準、独立$q=4$ holdoutを通過した。
production既定値は引き続きfull basisとする。次はWP05-bとして、同じ明示policyを未使用$q=8$と
比較対照$delta=0.01$へ広げる。そこまで通過した後にだけWP01-D/C07で$alpha$・shot数を再最適化し、
候補区間を再判定する。

## 成果物と再生成

- artifact：`artifacts/research_direction_full_scope/2026-09-22/wp05a_full_controlled_interrogation_connection_v1.json`
- runner：`scripts/run_research_direction_full_scope.py`
- test：`tests/test_research_direction_full_scope.py`

```bash
.venv311/bin/python scripts/run_research_direction_full_scope.py
.venv311/bin/python -m pytest -q tests/test_research_direction_full_scope.py
```

専用testは`4 passed`、変更後のlocal全suiteは`524 passed, 4 warnings`だった。warningは既存の
grouped-UWC testにおけるcomplex-to-real cast由来である。

成果物fingerprintは
`d8196ef1d8a576b7b7ab443c8613d2bd70b7a4fa57f43ac495be62bf2748f512`である。
これはsourceとupstream hashを記録したlocal dirty-worktree evidenceであり、immutable CIまたは外部再現結果ではない。

## 後続検証による更新（2026-09-22）

この文書の「次の検証」は実行済みである。WP05-bで$q=8$と$\delta=0.01$へ拡張し、初回に
$\delta=0.02,r=32,q=8$が5%を超えたため、独立32 trajectoryのWP05-bRを追加した。再検証では
選択policyのRZ誤差0.516%、全metric最大0.537%、full basis RZ誤差0.829%となり、全checkを通過した。
詳細は[WP05-b/R拡張・再検証](research_direction_full_scope_extension.md)を参照する。

続くWP01-D/C07の候補別再最適化では、$L_D=3$の点推定が$L_D=12$より13.92%低く、5% local
model区間は僅かに分離した。ただし25%移送区間は重なるため、頑健な方向判断は未確定である。
次はM08/G08として、支配的な後半roundの$q>8$ proxy精度を直接定量化する。詳細は
[WP01-D/C07再最適化](research_direction_decision_cost.md)を参照する。
