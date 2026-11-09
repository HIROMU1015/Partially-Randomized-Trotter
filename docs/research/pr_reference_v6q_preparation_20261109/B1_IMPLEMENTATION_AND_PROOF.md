# B1：実装対応と条件付き証明

[REPORT](REPORT.md)／[Q2計画](Q2_VALIDATION_PLAN.md)。本書は新moduleの静的論証と人工テストの記録であり、Q2科学検証・m17全support資格・正式な新保証採択ではない。

## 1. Exact binary入力と実装

対象は既存SOURCEが既に生成したF/G。g_F、g_Gと各RZ/RZZのθを、**保存されたbinary64値そのもの**の実数として解釈する。

\[
S_F=g_F-\sum_{j\in F}\theta_j/2,\quad
S_G=g_G-\sum_{j\in G}\theta_j/2,\quad
\Phi=\operatorname{RN}_{64}(S_F+S_G).
\]

元Hamiltonianから理想角度を再計算せず、E0／gap／ground記述を受け取らない。元SOURCEのglobal_phase集約後の値を用い、その上流誤差をexact和で消去したとは扱わない。

| [新SOURCE](../../../src/trottertracks/pr_reference_v6q.py)の関数 | 数学的対応／前提 |
|---|---|
| `_binary_input` | floatの内部値を`float.__float__`で取得し、`as_integer_ratio`からFractionへ。float subclassの独自変換を使わない。builtin intは変換後のFractionが元整数と等しい場合のみ受入れ、暗黙の丸めを拒否 |
| `_lane_sum` | SOURCE順に読み、RZ/RZZのhalfはFractionの2除算。非対角basisのXX+YY等は旧SOURCEで真空rayを保つ前提を使用。gate名・width・param数・classical条件を確認 |
| `exact_binary_phase` | `_lane_sum(F)+_lane_sum(G)`もFractionのまま。一回だけ`_round_once`へ渡す |
| `_round_once` | binary64 runtime条件、有限normal出力域を確認し、**唯一の合計float変換**を行う。τへの折返しは追加しない |
| `_verify_nearest_even` | 返却floatと両隣floatのexactな中点でRNを検査。tieでは格納significandの偶奇を使う。誤った値を補正して採用せず停止 |

F/Gごとにfloatへ戻す経路はない。半角のための中間floatもない。`Fraction(str(theta))`、decimal再解釈、fsum、別precision、解析Hamiltonian位相への置換は行わない。

支持するgate名は旧vacuum providerと同じrz／rzz／xx_plus_yy／z／p／id。真の固定Qiskit gate familyという前提であり、同名の異義custom gateを名前だけで資格しない。symbolic ParameterExpression、Decimal、Fraction parameter、complex、bool、非有限値、形状不一致、clbit／classical条件は拒否する。数値numpy scalarでもfloat継承型なら格納値を取れるが、それ以外の型は自動変換しない。**実SOURCEで出現する全型の適合はQ2／Q3待ち。**

## 2. Runtime丸めの検査

使用した既存project runtimeはCPython3.12.3、radix2、mantissa53、max_exp1024、double8byte、float_info.rounds1。標準`Fraction.__float__`は分子・分母のinteger true divisionを使う。既存V6調査で確認した[CPython3.12.3 numbers SOURCE](https://github.com/python/cpython/blob/v3.12.3/Lib/numbers.py)／[integer division SOURCE](https://github.com/python/cpython/blob/v3.12.3/Objects/longobject.c)と同じ経路である。build情報だけで任意入力の丸めを形式認定せず、返却時にも以下を確認する。

finiteな返却値xと隣接値x_−、x_＋について、Sが二つの中点の間にあり、中点一致ならxのsignificandが偶数であることがRN-evenの必要十分条件になる。中点はfloat演算ではなくFractionで作る。max finiteの外側隣接値が∞の場合は、既に|S|≤max finiteを要求しているので、その側の範囲はこの事前条件で閉じる。

この検査が返却値のRN性を保証するため、拡張精度や別丸めmode等で最終変換が異なる場合に、未評価の誤差を黙って加えることはない。検査失敗時はArithmeticErrorで停止する。隣接binary64、格納format、Python整数／有理数のexact性はruntime前提であり、wheel・OS全体を形式検証したとの主張ではない。

overflow域は拒否。S=0はexactな0を返す。**0<|S|<min normalは拒否**し、旧相対丸めモデルへ新しいsubnormal絶対誤差allowanceを混ぜない。入力の小さいhalfはFractionとしてexactに扱えるが、最終exact和がこの未資格域なら結果を返さない。この制限を将来変更することは今回の実装範囲に含めない。

人工テストは独立の整数common-grid和を参照し、正負のmidpoint、片側ずつの丸めが残差を失う例、誤った候補floatを中点検査が拒否する例を含む。実H4数値をこのproviderへ通したものではない。

## 3. 位相の意味と旧SOURCEへの接続

旧[`reference_core`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/pr_reference_interference_pilot.py)を再利用する。F|0⟩=e^{iS_F}|0⟩、G|0⟩=e^{iS_G}|0⟩、旧actual SOURCE inverseの関係の下では、

\[
z_{\rm Ref}=e^{i(\Phi-S_F-S_G)}\langle\psi|FG|\psi\rangle.
\]

従って変えるのはancillaのrelative P角だけであり、common joint scalarの不変性を理由にPを除去しない。F/Gの順序、F.inverse()、basis共有、D融合、signed event、scalar、二つのCSWAP cascade、X/Y readoutは旧関数のまま。source inverseを独立合成wordのinverseや別のgate列に置換しない。

新`phase_for_round`はdefaultで旧transferのF/G別逐次和と最終加算を使う。`mode='exact_binary_m17'`かつ整数m=17の場合だけB1を選ぶ。m0〜m16への新mode指定は拒否する。`reference_for_round`はこのphaseを旧gadgetへ渡す新callsiteで、旧`assemble`・`trajectory`やrunnerを上書きしない。

呼出側はmを実際の保存scheduleに結び付ける必要がある。m引数だけでq/r/bitsや全supportを検証したことにはならない。Q2のq1/q4は局所correctness fixtureとしてdirect providerを検査するもので、m17 whole実行を偽装しない。m17 opt-in選択自体はQ1のmockと静的構造で検査する。

## 4. 保存Mと全support：証明できた命題と不足

旧[`phase_bound`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/pr_reference_transfer_error.py)のMは、**同じbinary SOURCE lane**のexact prefix絶対値を覆う。S_F、S_Gはその最終prefixなので、|S_F|,|S_G|≤M。従って|S_F+S_G|≤2Mである。数学的相殺を新たに仮定してMを縮めていない。

入力が受理され、exact和が0または許容normal範囲なら、最終RNにより

\[
|\Phi-(S_F+S_G)|\le u|S_F+S_G|\le2uM.
\]

global_phase取得前の集約・Qiskitのfloatτ剰余・controlled SOURCEとの対応について、旧B_normが同じbuilder／normalizationへ適用されるなら、

\[
\delta_{\rm B1}\le B_{\rm norm}+2uM.
\]

この**受理入力クラスの条件付き命題**はSOURCEに対応する。次の表が本番全supportへ広げる際の状態である。

| 証明義務 | 今回の状態 |
|---|---|
| 同じF/Gとbinary入力 | 読取実装・旧builder非変更は静的確認。実Qiskit入力との照合は未実行 |
| 変換・half・和がexact | Fraction／ratioの実装と人工20検査で確認。unsupported型は拒否 |
| 合計一回RN | 静的に唯一の最終変換、各返却値の中点検査。人工tie検査PASS |
| 片側Mから2M | 同じSOURCEへの旧prefix保証を前提とする三角不等式。旧Mは再算定しない |
| overflow／subnormal／型rejectが全supportで生じない | **未証明**。有限4例のPASSでも閉じない |
| B_normの継承 | builder・global_phase生成不変更なので適用条件を保持する設計。Qiskit取得・copy／compose／P経路の実接続は未検証 |
| signed event・scalar・actual inverse・relative信号 | 旧関数を再利用する静的支持。実H4 coherenceはQ2待ち |
| m17だけ切替、他17roundは旧保証 | 新wrapperのdefault／選択規則は確認。実際のschedule callerへの運用接続は未資格 |

受入れ範囲を全supportへ示すための具体的な追加証拠候補は、SOURCE由来のfinite literal型とdyadic latticeの閉包である。例えば全入力とhalf／global-phase更新後の値が2⁻ℓの格子に入り、ℓがnormal出力を保証する範囲なら、非零exact和の下限を示せる。ただしactual global_phaseの剰余／丸めを含む閉包証明と有限literal域の確認が必要で、**今回ℓを保存値から新計算していない**。これは新precision選択・量子化policyやA1の追加候補ではなく、このB1のreject域を覆うための証明義務である。必要な数値監査をする場合は別途固定scopeと実行承認を要する。

## 5. 固定compilerに対する価格不変補題

旧[STRUCTURAL_APPENDIX](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/docs/research/pr_reference_bprime_20261107/STRUCTURAL_APPENDIX.md)と[compiler A](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/track_c_compiler_a.py)を再利用する。

前提は、genuineな同じgate family／parameter形状・context、全結合・Target／calibrationなし、equivalence追加なし、controlはlower前、同じpass順、角度依存最適化なしである。新wrapperが同じF/Gを旧gadgetへ渡しPのfloat角だけを交換するなら、

1. SOURCEのgate名・wire・順序・register・reset・CSWAPは同じで、Pは角0でも省略しない。
2. 初段の固定inverse cancellationで新P角を理由にRZを削除するpathはない。basis共有・D融合を生成し直さない。
3. post-unroll family/contextは同じ。既存symbolic P→RZテンプレートへ新floatを束縛するのでPのRZ数は1。
4. 最終CX→X取消passはRZを削除しない。新角を理由とするrotation cutoff／1qubit synthesis／反復最適化は存在しない。

以上の前提の下で、whole回路のRZ数は単に局所Pの個数を観測するだけでなく、**同じcompiler contextと全passの適用範囲**から保持される。旧RZ価格の上下界・canonical lawも変更しないので、V6-Pの価格不変条件へ接続できる構造的理由になる。

ただしactual callsiteでP以外を変えていないか、P parameterがそのまま保存されるか、consumerが追加compile/controlを行わないかはQ2／Q3で照合する。別contextが必要と判明した場合は旧資格を転用せずSTOPし、最小局所compileの固定scopeへ戻す。今回compile／価格取得は0件。

## 6. V6-Pとの関係

V6-Pの式が仮定するexact sum＋一回RNを新moduleで実現する設計には到達した。B_norm、その他disk、PF、法則、B′、他roundの誤差／N／価格を変更しない。しかし全support受入れ・実Qiskit／P経路・固定consumerへの接続が未資格なので、**条件付き利益を実装済みの全18round利益へ昇格しない**。

native RZの個数不変はCT wordのT価格不変を意味しない。新しいP角には別word／domain／strict保証が必要となり得るが、今回はその評価・合成を行わず旧CT系列STOPを維持する。Fractionの長round走査・整数費用も未測定である。
