# guard証明改善：次の最小技術検証案

2026-11-04 JST。実データでの認定可能性は未確定。
本計画は次段案であり、新科学実行SOURCEや実行許可ではない。

## 推奨する順序

1. **保存norm²を再利用する。** SOURCE18fffce48/結果34840ce9の96 bit区間は、B raw stateと
   context81fa6813のψに適用できる。同一性の根拠は[調査結果](TECHNICAL_FINDINGS.md)へ。
   exact norm²診断6d4b1434を先に実行しない。将来区間が閾値をまたぐ等、具体的必要性が生じたときだけ再検討する。
2. **人工系のrow graph→signed energy接続を準備する。** 既存forestの実row積和/underflowを再利用し、
   zero/sign/swapのexact性を保つ最小adapterを作る。現在のprototypeは入力されたboundsを合成するだけなので、
   符号付きradial項を実graph/到達集合から得る処理が次に必要。
3. **未知を残す。** 根拠が得られないrow/radial項を0や負にせず、UNKNOWN/STOPを返す。
   genericなnorm-only項へ戻す場合は、その積み重ねでも閾値へ届くかを人工的に確認する。
   保守性を取り除けない場合、不要な実H4全行列評価を開始せず、障害を報告する。

## 次のadapterで扱う最小の範囲

| 部分 | 再利用と追加する処理 |
|---|---|
| kernel row graph | e2fd35f1のbinary/forest条件・γ_D・Jη項を再利用。短いdepthを使うなら同じbinaryのgraph根拠を追加 |
| 状態とmetric | まずH=Iと有限な人工reachable setから開始。必要な場合だけ正定値Hとobservableのdual boundを導入 |
| energy | signed drift/radial/error²を全到達stateで覆う。経験的な符号・独立性を仮定しない |
| rotation | cI−isPの非Hermitian項と二経路scale/subtractの丸めを保持 |
| 合成 | F→product→rotation→phase→Rの実行順序を固定。収縮を許すnorm型と全prefix包絡を使う |
| 平均 | 旧作用誤差・ideal θとのtrig差を保持。norm gainをsignal_errorへ流用しない |
| 最終gate | 実vdotの丸め、全prefix有限性、complex abs上界、両座標guard、strict判定を保持 |

metricを広い全supportで共通に使えなければ、reachable-set上の条件付き包絡を検討する。
長い非自明係数のpowerは、既存Arbの上向き端点を再利用して扱い、巨大なFractionの分子分母を直接生成しない。
今回のexact-rational prototypeは人工検証用である。
supportをsamplingで選別せず、正のmassを持つ成分・orderを除外しない。
全trajectoryの列挙は標準案にせず、共通basis・symbolic構造・不変集合の証明を優先する。
実H4にその構造があることは現時点ではUNKNOWN。

人工検証は、exact signed permutation orbit、非unitary/非Hermitian行列、
外向きのcoherent丸めモデル、F/R間に非可換eventがある例、
contracting block内の一時増幅、underflow、metricのobservable損失、strict guard境界、
未確立radial/runtime/finite条件でのSTOPを含める。
既存のball_adapter、KernelBounds、runtime/guard/axisテストを適合する範囲で再利用する。
今回の人工37件を再現するだけで実H4の保証へ格上げしない。

## 実データを扱う前に固定するもの

技術的に有利な証明を構成できた場合に限り、既存保存contextから必要なmatrix metadata/
drift/radial/metricを評価する最小範囲を定め、科学SOURCEとして公開する。
固定対象はlinear H4/1Å/STO-3G/保存DF12/B raw state/sector70、
L_D3/δ=.02/m17 q131072/r32 K2、保存N/α/β/ε・現行kernel/profileとguard閾値。

入力は結果81fa6813のcontext/RESULT、包含結果34840ce9のnorm²/logical interval/laws等を必要最小限で固定する。
path/hash・SOURCE identity、精度・pass/call範囲、全supportの量化、runtime条件、
出力、strict判定、UNKNOWN/STOP、retry禁止を新SOURCEで明示する。
旧SOURCE・計画・raw・grant/ledger/marker/STOPは変更しない。
数値が保存されているだけで、新しいnorm/metric/残差/guard評価の実行許可へ読み替えない。

独立のmatrix/row証明が重くなれば、共通immutable入力・basis cache・並列化を準備し、
BLAS/FLINT workerを協調させる。利用可能資源から構成を選び、恣意的なCPU/RAM/worker/wall上限は置かない。
科学的意味と実kernelの演算順を保持し、自job以外の環境やprocessを変更しない。

## 担当とSTOP

次はユーザーがこの技術準備への進行を依頼するか判断する。
Codexは依頼後に上記の人工・SOURCE準備を進める。
実H4の新しい評価は、固定SOURCE公開後のユーザーの別途明示承認が必要。
重要な独立科学レビューも別の許可範囲であり、今回未実施。
現段階では両U=null・全体UNDETERMINEDのまま、公開照合後にmandatory STOPする。
