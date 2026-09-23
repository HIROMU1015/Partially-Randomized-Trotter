# WP06-a DF回路構造pilot

## 目的と判定規則

Gate S1で最大の残存不確かさをfull controlled interrogationの回路scope・構造と判定したため、
大規模なRPE回路へ進む前に代表的なDF Z/ZZ作用だけで構造改善の可能性を調べた。
事前固定したresearch-routing triggerは、代表RZ costの相対変化5%以上、候補順位反転、
$q$方向の傾き・proxy domainの変更、またはcontrolled relative phase補償の欠落である。

これはH4鎖、1.0 Å、STO-3G、8 qubit、DF rank 12、$L_D=3$、$\delta=0.02$の固定snapshotと、
Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなしのlocal pilotである。
新しい構造候補を既存proxyへまだ組み込まず、WP05へ進む前に再較正が必要かだけを判定する。

## support限定Gaussian completion

既存eventはfull Gaussian basis $B$を用いて$B Z_i B^\dagger$または
$B Z_iZ_j B^\dagger$を実装する。前者は$B$の第$i$列、後者は第$i,j$列だけで決まるため、必要列を
保持し、残りを標準基底から決定論的に直交補完したunitaryを構成した。このcompletionで得た回路は
既存回路と小行列で比較し、controlled回路ではglobal phase同値でなく枝間relative phaseを含む
直接同値性を要求した。

物理snapshotから係数絶対値が最大のZとZZを各一つ選んだ結果は次の通りである。

| event | support | full / support basis演算数 | controlled RZ | RZ変化 | controlled CX | depth変化 |
|---|---|---:|---:|---:|---:|---:|
| `df-fragment-6:Z0` | Z0 | 34 / 14 | 204 → 78 | -61.76% | 58 → 26 | 123 → 103 |
| `df-fragment-3:Z0Z1` | Z0Z1 | 35 / 21 | 311 → 189 | -39.23% | 98 → 66 | 162 → 162 |

uncontrolled回路でもRZはZで62.69%、ZZで41.08%減った。全同値性検査を含む最大operator残差は
$4.73\times10^{-15}$で、$10^{-10}$の判定値を十分下回った。従って5%の事前triggerは発火した。

## 短い列とbasis共有の競合

support限定構成は単発では安いが、同じDF fragmentに属する異なるZ/ZZ supportごとに別basisを
持つため、full basisの共有を失う。最大係数側の同一fragmentから異なるZZ supportを取り、
controlled列を比較した。

| 列長 | full basis共有RZ | support別RZ | 相対変化 | support別depth変化 | RZで選ぶ構造 |
|---:|---:|---:|---:|---:|---|
| 1 | 311 | 189 | -39.23% | 0% | support限定 |
| 2 | 332 | 267 | -19.58% | +45.26% | support限定 |
| 3 | 353 | 406 | +15.01% | +86.64% | full basis共有 |

全列のoperator残差は$1.24\times10^{-15}$以下だった。列長2ではsupport限定がRZを減らす一方で
depthを増やし、列長3ではRZでもfull basis共有に逆転する。したがってsupport限定basisを全eventへ
一律採用せず、event列のfragment・support runを見てfull共有とsupport限定を選ぶ必要がある。

同一full basisの長さ2列について明示的なbasis融合を無効／有効で比較すると、未transpile sizeは
48.95%、depthは47.37%減った。現compilerは隣接する逆演算を自動相殺したため、transpile後の6指標は
同じだった。明示融合はbuilder負荷を減らす既存方針として維持するが、この1 compilerだけから
compiled-cost改善を一般化しない。

## controlとscalar phase

full event全体をcontrolする素朴な構成と、basis変換を非制御のまま中央Z/ZZだけをcontrolする現構成を
比較した。relative phase補償を含めたoperator差は$4.73\times10^{-15}$である。現構成はRZを
2,567から311へ87.88%、CXを1,834から98へ94.66%、total depthを4,570から162へ96.46%減らした。
この最適化はすでに現行builderへ入っているため、新しいproxy変更理由にはしない。

固定stepのconstantと抽出identityを合わせたscalar phaseは$-0.0477143$ radだった。非制御回路では
global phaseだが、controlled回路で補償を省くとoperator差は0.0477098となった。PhaseGate補償を
入れるとwhole-control参照と一致し、二つのphaseを個別適用しても集約してもoperator差は
$2.23\times10^{-16}$だった。現行のrelative-phase保持と反復時のphase集約方針を維持する。

## サイズ方向の構造診断

物理snapshotとは別に、seed固定の実直交basisで$n=4,6,8$のcontrolled ZZ単発を診断した。

| $n$ | full / support basis演算数 | full / support RZ | RZ変化 |
|---:|---:|---:|---:|
| 4 | 10 / 9 | 98 / 84 | -14.29% |
| 6 | 21 / 15 | 198 / 134 | -32.32% |
| 8 | 36 / 21 | 324 / 180 | -44.44% |

この表は構造的なサイズ依存の診断で、分子系サイズまたはchemistry evidenceではない。3点から一般の
漸近則を主張せず、少なくとも代表サイズで差が消えないことだけを確認した。

## 判断

事前の5% triggerは発火し、既存のfull-basis専用proxyをそのまま高統計化してWP05へ進む方針を
撤回する。一方、短列で構造順位が反転したため、support限定への一律置換も採用しない。

次は限定的なWP06-bとして、次を行う。

1. production event builderへ、full basis共有とsupport限定を明示選択できるpolicyを入れる。
2. 同じtrajectory上でfragment・support runごとに候補を比較するsequence-aware選択を作る。
3. 物理event分布でRZ、CX、depthを同時に較正し、未使用短列holdoutで検証する。
4. その更新後にだけWP05のfull controlled interrogation接続へ進む。

T4は高優先を維持し、T6はこのfocused follow-upに限って継続する。アルゴリズム候補
$L_D=3$対12の順位、RPE反復数$q$方向の傾き、full partial-$S_2$、Hadamard wrapper、状態準備、
実backend、noise、最終総costは今回評価していない。

## 後続結果

WP06-bでは、同一元basisのsingleton runだけsupport限定にする`support_run_le_1`を独立trainingで
固定した。未使用列長3,6のholdoutはRZ -10.67%、CX -8.13%、total depth -2.25%で、最大controlled
operator残差は$1.34\times10^{-15}$だった。additive proxy bridgeのRZ $q$ slopeは最大8.00%変化し、
$L_D=3/12$の点順位が反転したが区間は重なった。後続WP05-aではcomplete Hadamard wrapperの
$q=4$ RZ holdout最大2.29%、additive bridge残差最大2.63%で5%基準を通過した。さらにWP05-b/Rで
$q=8$と$\delta=0.01$、WP01-D/C07で候補別再最適化まで完了した。5% local model区間は
$L_D=3$側へ僅かに分離したが、25%移送区間は重なるため頑健な方向判断は未確定である。詳細は
[WP06-b sequence policy](research_direction_sequence_policy.md)、
[WP05-b/R](research_direction_full_scope_extension.md)、
[WP01-D/C07](research_direction_decision_cost.md)を参照する。

## 成果物と再生成

- artifact：`artifacts/research_direction_structure_pilot/2026-09-22/wp06a_circuit_structure_pilot_v1.json`
- runner：`scripts/run_research_direction_structure_pilot.py`
- test：`tests/test_research_direction_structure_pilot.py`

```bash
.venv311/bin/python scripts/run_research_direction_structure_pilot.py
.venv311/bin/python -m pytest -q tests/test_research_direction_structure_pilot.py
```

専用testは`4 passed`、変更後のlocal全suiteは`515 passed, 4 warnings`だった。warningは既存の
grouped-UWC test由来である。artifactはsnapshot SHA-256、Gate-S1 fingerprint、source hash、compiler条件を
記録するlocal dirty-worktree evidenceであり、immutable CIまたは外部再現結果ではない。
