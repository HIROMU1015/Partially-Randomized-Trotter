# 保存stateによる積型ノルム証明限界：限定診断SOURCE

2026-11-04 JST。**技術準備済み、科学実行は未実行・未許可。**
公開commitの本runnerをSOURCEとして固定し、ユーザーが確認後に別途明示承認した場合だけ`--run`する。
承認文書・証明書・grantの新規発行は不要。旧grant・ledger・marker・STOPは変更しない。

## 対象と意味

linear H4 / 隣接1.00 Å / STO-3G / 保存legacy DF12 / B raw state / 8 qubit・4電子sector70、
Partial L_D3 / δ=.02 / m17 q131072 r32 K2、extract_identity_phase / coefficient_atol=1e-12。
N411/axis、α=.016957089147912167、ε=.15244325851620388、
β_PF=.014050012886294481、β_RTE=9.899584668737594e-8、β_stat=.38594988811785885を保存literalのまま保持する。
Model IIの数学的IID uniform53と、固定PCG64の非保証を区別する。

目的は[前案の絶対欠損・積型証明の限界](PROOF_LIMIT.md)を保存raw stateへ適用することだけ。
float64仮想測定の実行・母平均・abort確率・RPE十分条件・費用の評価ではない。
既存のideal categorical包含・平均作用誤差・旧判定は再計算も変更もしない。

## 固定SOURCEと入力

[runner](run_norm_proof_diagnostic.py)自身を公開HEADとbyte照合し、同fileの`HELPERS`にある
e2fd35f1dのkernel係数、728c90a2dの旧bridge/helper、18fffce48のball/mean、
c3a5cc387のarray検査/atomic JSONを各固定commitへ照合する。
SOURCE commit SHAは公開後の最終チャットとGit履歴で特定する。

入力commitは2件とも`81fa6813194e4e14189092626f66dd10793225e1`、
prefixは`artifacts/rpe_m17_virtual_measurement_bridge/2026-11-04/run01/`。

| path末尾 | SHA256 | 用途 |
|---|---|---|
| numeric_context.npz | `f8c1f3185423f479d82ba0bc1eb94bfb8fd9d0cafa487c618964f3bd22bddbae` | 保存context。inventoryを確認し、ψだけ数値ロード |
| RESULT.json | `d2989174012c41e6908d21ccc65d894820e763749e922835f9c5e191eb3c8735` | 科学SOURCE c3a5cc387、固定tuple/Nと保存α/β/ε等のprovenance |

両入力はhashと固定Git blobのbytesへ照合する。原本を書き換えず、再構成/再保存しない。
同じbytesは元SOURCE/実行報告によりB raw stateへ接続済みであり、今回新しいstate solve/normalizationはしない。
過去のrun02の未保存payloadへ読み替えない。

## 精度・順序・call範囲

1. fresh出力directoryを作成。SOURCE/helper/input bytesを照合する。
   不足object・不一致・破損は数値ロード前にSTOPする。
2. 元RESULTのSOURCE、正常終了、m17固定tuple/Nを確認し、α/β/εを再配分せず出力へ保持する。
3. contextの221-member inventory、ψのshape70・C-contiguous/aligned complex128・stride・有限性を確認する。
   他の220行列memberはdecodeしない。layout変換・castで不一致を隠さない。
4. 一回だけψの実/虚binary64成分をexact dyadicとして扱い、140平方項のrational和でnorm²を求める。
   arbitrary-precision integer/Fractionを使い、丸め誤差・underflow floorの消失は0。
   これは証明用の精度であり、現行NumPy/OpenBLASの仮想作用を別算術モデルへ置換しない。
5. 固定kernel係数とqからΛをexact rationalで計算し、`n²Λ > C`および`n²Λ > C_abs`を判定する。
   等号では下限のみで断定しない。floorが小さくてもguardやmeanを認定しない。
6. 結果を保存し、成否・未判定・中断のいずれでもmandatory STOP。

最大callは保存context load開始1、exact norm pass開始1・完了1、二つの限界比較各1。
実行passは1回だけで、追加precision・retry/resume・自動fallbackは行わない。
P/F/R行列演算・norm/欠損、rotation/trig再生成、ideal mean/CDF/s、統計予算、
sampling/RNG/trajectory/仮想shot、回路build/compile、他round、費用・独立詳細科学レビューは0。

数値処理は70複素成分のexact和と少数のrational演算だけで、並列起動の対象となる重い行列評価を含めない。
巨大なrational乗算のq乗はBernoulli下限で避ける。恣意的なCPU/RAM/worker/wall上限は設けない。
実際のquota・空き資源と他利用者への非干渉を守り、自job以外の設定を変更しない。
将来別途準備する重い包含評価では行列ごとの並列化・共有immutable context・BLAS/FLINT協調を行う。

## 判定・保証境界・出力

| method_status | 意味 |
|---|---|
| NORM_PROOF_OBSTRUCTION_CONFIRMED | この絶対欠損・独立積型majorantの下限だけで少なくとも一つのguard閾値を超える |
| FLOOR_INCONCLUSIVE | この下限では証明限界を断定できない。guardの成立・不成立は未判定 |

どちらも実signalの超過やabort trajectoryを証明しない。
`U_virtual_IID_upper=null`、`U_exec_upper=null`、`RPE_tolerance_certified=false`、
`scientific_status=UNDETERMINED`、旧`INCONSISTENT`/`u_num=null`を常に保持する。
finite_intermediates_proved/all_support_completion_provedはfalse（今回未証明）。
旧finite結果を取り消すものではない。

本runnerは実BLAS/scale/subtract/dot/hypotを行わないため、そのdispatch/全thread FP modeを新しく認定しない。
係数は固定theoremの入力。完全なbridge認定を将来再開する場合は、旧runtime検査、
全support有限性、complex abs上界、両座標guard、平均作用誤差、strict十分条件がすべて必要なまま。
本診断をそれらの代替にしない。

出力はfresh `artifacts/rpe_m17_norm_proof_feasibility/2026-11-04/run01/RESULT.json`。
SOURCE/input hash、固定literal、段階call、exact norm²/Λ/閾値との関係、診断/全体status、STOPを保存する。
SOURCE/input不一致・例外・中断は`INTERRUPTED_OR_INVALID_MANDATORY_STOP`として部分記録を残す。
既存directoryへの再実行は拒否し、他の保存先へ自動retryしない。

```bash
# 公開固定SOURCE確認後の別途科学実行承認が必要。今回は実行しない。
python docs/research/rpe_m17_norm_only_adapter_preparation_20261104/run_norm_proof_diagnostic.py --run --output-dir artifacts/rpe_m17_norm_proof_feasibility/2026-11-04/run01
```

次の担当はユーザー。診断の必要性と公開固定SOURCEを確認し、実行するならこの範囲を別途明示承認する。
Codexは許可後だけ一回実行し、raw保全・公開照合の後にSTOPする。
