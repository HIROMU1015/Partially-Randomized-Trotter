# 将来の最小識別計画：F1を先に、F2は条件付き

**全て提案。今回の実装・科学実行・回路build/compile・sampling・証明実行は0件。** 本書はgrantや実行契約ではない。数値入力、call inventory、SOURCE、精度、停止点を固定し、別途明示承認を得る必要がある。現状は既知法適用候補なので、先にユーザーが「独立新法の検討を続けるか、既知法のPR接続を評価するか」を判断する。

## 1. どの判断を変えるか

F1は「固定fragmentの辞書を選ぶ具体手順が、現行のuniform armと異なる制御列を作り、同じH／meanを保ち、近接既知構成との方法差を説明できるか」を判定する。mixedが出たことだけでは独立新規性としない。

F2はF1に意味のある構成差が残った場合だけ、「既知共有後の回路単価差がB/N・prep/readout・古典費で相殺されないか」を調べる。F1が既知法の同じ生成手順への還元を示した場合、独立新法としてのF2は提案しない。応用評価へ進むには、その位置付けを別途採択する。

## 2. 最初の入力候補と理由

第一候補は分子の再生成ではなく、**4 fermionic mode、2粒子の初期状態、固定one-body deterministic blockが1個、残差density-square fragmentが2個**の小さいsynthetic fixture。geometry・AO basis・DF rankは非該当、fragment数2をDF rank2の分子結果と呼ばない。`L_D=1`はこのfixtureのone-body blockだけで、残差の2fragmentは動かさない。

保存factorを模した構造族として `F_l=B_l(Σ_p d_lp n_p)^2 B_l†` を使う案を置く。一方のframeは少数modeに局在、他方は4modeへ広がるGivens列とし、係数・frameはexact rationalで記述できるものを実行前に一つ固定する。ここで数値行列・Pauli係数を生成していない。mixedが有利になるよう結果を見てfixtureを追加探索しない。両fragmentが同じ辞書を選べばその結果を残す。

4modeにするのは、全Fock registerが小さく、2粒子でdensity-squareを自明な1粒子作用だけに潰さず、個別Pauliのsector外作用も調べられるため。full system 16次元、ancilla込み32次元の将来小系参照を用いる設計である。これらは空間サイズの指定であり実測結果ではない。初期stateはregister順を固定した `X_0 X_1 |0000>` とし、そのprepを全armで含める。exact ground state、E0、gapは使わない。

分子移送は最初の必須作業ではない。F1/F2後に保存molecular snapshotへ進むなら、同じ物理splitでS/P選択を作れるか、係数とrecipeのidentityを先に確認し、geometry・basis・rank policy・`L_D`を別契約で固定する。H2/H4を自動採用したり、新基底関数を同一Hの改善に混ぜたりしない。

## 3. F1：構成・意味論・取得手順を識別する

1. 一つのfixtureを凍結し、S/Pの係数map、scalarとzero policy、full register、precisionを照合する。二fragmentの全四bit列を同じ既存typed Primitive構成でも組み立てる。未実装production routerの移植だけで差が出たとはしない。
2. 主案の二端点一pass selector、全S、全P、同じ四候補の有限列挙対照を比較する。全対照へ同じ係数map・cache・literal recipeを渡す。selection workとcache初回費を別に記録する。実装予定の具体差は[方法差表](METHOD_DELTA_AND_PRIOR_ART.md)の範囲に限定する。
3. 意味論診断の候補設定は `T=.2, q=2, δ=.1, r=2, K=2`。q/rが2なのはmicrostepとD境界の両方を含め、K2はproductとchildの順序・paired phaseを確認するため。これらは最適化条件ではなく未採択のprobe設定である。
4. 各bit列のHとfinite corrected meanを既存moment参照で確認する。full registerのoperatorと初期stateのX/Y信号を分ける。degree-K+1 polynomialとの照合だけではactual circuitを認証しないので、n0/n2、負係数、I、異basis、同basis run、D境界を含む固定motifのcontrolled operator probeを追加する。motif一覧と数は生成前にSOURCE/inputと一緒に固定し、希少eventを恣意的に平均へ混ぜない。
5. 価格状態遷移は、物理fixtureと別の小さい4-label recipe catalogで有限列挙と照合する。K2/r1ならn0の4通りとn2の64通りを含む。同じ小catalogで、各eventのoperatorを独立に構成したB加重meanもpolynomial参照へ照合する。moment関数へ同じRを入力しただけの一致をsampler検証と呼ばない。価格の照合は定義したmodelの検査であり、実compiler予測精度の測定ではない。

F1はMC trajectory抽選を必要としない。小さい候補列挙・moment参照・固定motifで構成差を識別する。全RTE pathや全native synthesisの再現は要求しない。compiler候補は `Qiskit 1.3.0 / optimization_level=1 / seed_transpiler=17 / basis=[rz,sx,x,cx] / topology-free`、ordinary ancilla control、同じX/Y wrapper。既存共有・相対Gaussian簡約等を利用する場合は同じ版・同じ適用範囲を両armへ与える。

PO-01～09の全てを一度に一般証明する契約ではない。小系検査を越える保証が必要になれば、その課題を残して停止し、別作業を提案する。

## 4. F2：最小の回路単価と統計交換

F1が通り、継続が採択された場合の最初の候補は**同じfixtureに固定した `T=.2,q=1,r=2,K=0`**。全armで同一設定とし、F1/K2の結果を費用平均へ流用しない。これはorder0のchildが二回あり、同basis runの共有とsingletonの両方を既存法に与えられる最小の構成である。高次数の性能や長時間RPEについて結論しない。

canonical lawの有限trajectoryは各armでcomponent数をLとするとL²通り。**全L²通りを確率重み付きで扱う案**を第一候補とする。恣意的に64 trajectory等を固定せず、係数catalogを得た後、正確なarm別call数と出力件数を契約へ記載する。build/compileはX/Yを含むfull wrapper。topology-free・compiler設定・control・precision・scalar・prep/readoutを共通にする。L²が実務的でない場合、勝手に少数samplingへ替えず、設計を戻して検証範囲を再承認する。

統計の候補は複素signalの `η_stat=.025, α=.05` とし、主案と同じbounded-range N policyを使う。真のsignalへのbias allowance候補も`.025`だが、共通の既存boundまたは小系参照で適格性を確認する必要がある。予算を満たさなければ同精度費用比較は不適格であり、r/Kや精度を後から都合よく変更しない。量子measurement shotの実抽選は不要で、Nは指定policyの必要回数として報告する。

報告対象は、λ、B、N、1-shotのactual RZ/CX/depth、N加重費、prep/readout、係数map・選択・sampling law構築の古典時間／memory。T/native・全RPEはこの最小段階では取得しない。候補選択用literal scoreとactual順位が違えば、その差を記録する。normやΦだけの成功を総資源改善にしない。

## 5. GO／縮小／STOPの判定

| 判定 | 条件と次の扱い |
|---|---|
| F1構成継続候補 | 同じ物理targetと正しい制御mean、既存共有後の具体差、取得可能な情報、比較手順が残る。独立新規性・性能GOではない |
| 応用へ縮小 | 同じmixed生成が既知構成の組合せで得られ、独立方法差を説明できない。実装価値を評価するならユーザーが別採択 |
| F1 STOP | phase/order/scalar破壊、sector射影の不正使用、情報優遇、価格oracle、全arm同一生成で追加差なし、取得workが明白に過大 |
| F2継続候補 | 同精度・同条件でactual N加重の指定資源に意味のある差。量子／古典の交換と適用範囲を明示。閾値は目的に応じ実行前に固定 |
| F2 STOP | 実compiler後に差が消失、N/prep/古典費で相殺、共通bias条件不適格。条件変更で成功例を追わない |

将来の重い処理は入力とrecipeを共有し、独立arm／compileを適切に並列化する。実際のCPU/GPU/memoryと非干渉を確認し、dominant workloadをreferenceと照合してから本runへ進む。人工的なCPU・memory・wall capは設けない。科学call・候補・log/output範囲とone-shot/STOPは契約で固定する。今回の文書公開後のmandatory STOPは、この計画によって解除されない。
