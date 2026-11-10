# R可変compiled RZモデルの設計案

設計段階。係数の校正、descriptorの実データ再計数、価格予測は未実行。H4/H6の係数を共有せず、既存の凍結Mq/Affineを変更しない。以下の式は将来の検証対象で、採用済み価格式ではない。

## 対象と依存関係

Linear H4/H6・1.00 Å・STO-3G、保存tol-only DF10/DF19・weighted completion、元fragment順、元H/state/snapshot/preparationを固定する。H4はS2/δ=.1/PR prefix1/K2、PF対照B0/p7。H6は4次m5_best/δ=1.6/PR p7/K2、PF対照Full19。compilerはQiskit1.3.0/rz,sx,x,cx/opt1/seed17、hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、symmetric_directional・測定付きfull Hadamard wrapper、P=0。実hardwareやstate準備費のモデルではない。

固定δではT=qPF δであり、T,qPF,δを独立な回帰変数にしない。q_RPEとqPFも区別する。R変更時にもH4/H6の科学bindingを保持し、別DF rank、別prefix、H6早期PR p5/R32、別δへ移送しない。

H4は1 outerにR個のevent、全体L=qPF R。H6では元順の11 signed weight w_jに対して

\[
 r_j=1+\lfloor (R-11)|w_j|/\sum_k|w_k|\rfloor+\text{largest-remainder correction},\quad
 \sum_j r_j=R,\qquad \tau_j=\lambda_R\delta w_j/r_j.
\]

実行可能条件は**整数R≥11**。11の倍数・偶数・2冪である必要はない。余りのtieは元piece indexで決める元allocationをそのまま使う。signed weightの順、負のpiece、binary64の丸めを保ち、対称化や新しい配分を導入しない。全event数はqPF Rで、11qPF Rではない。H4のRは正整数。metadata判定を信号/価格の科学評価と混同しない。

## Rを変えると何が変わるか

| 要因 | compiled価格への影響・扱い |
|---|---|
| event数 | L=qRは増えるが、eventは同じ価格の独立箱ではない |
| finite K2 law | τ_jでorder確率・rotation angle・relative phaseが変わる。R増加を同じevent分布の延長と扱わない |
| basis共有 | 同basisの連続application、identity、products→rotationの順で変わる。head/tailの境界も含める |
| compiler | adjacent inverse basis、wire-local Z foldだけでなく1Q再合成/CX界面等がある。解析native countをP0価格へ昇格しない |
| PF反復境界 | qを増やすとpiece/outer境界が増え、wrapper端部とscalar-onceは別。固定Rのq Affineは有限観測であり一般則ではない |
| H6 allocation | Rでr_jが階段状に変わり、同じR増分でもτ_jと境界classの変化が異なる |
| order2 | 2 products＋1 rotationを持つlegal event。追加basis/identity/phaseの寄与があり、未観測は価格0ではない |

## 第一候補：構造価格＋少数のcompiled残差補正

既存Accounting kernelによるtrajectory ωの全wrapper構造価格をS_a(q,R;ω)とする。literal basis cancellation・Z fold・単一scalar・head/tail・signed pieceを含むが、実compilerを再実装したものではない。将来、新Rでのscope拡張を別SOURCEとして行い、元gate会計を保つ。今回その実データ計数は行わない。

最初の候補は、systemとaxisごとに

\[
 \widehat C_a(q,R;\omega)=S_a(q,R;\omega)+\theta_{0,a}+\theta_{q,a}q+\theta_{L,a}(qR).
\]

校正する量は同一trajectoryのpaired C−Sで、CとSの別標本平均を独立扱いしない。3係数の**残差モデル**であり、全価格をevent数に比例させる式ではない。Sが角度・path・basis境界の依存性を担い、残差のq/qR依存は反証可能な仮説として扱う。X/Yは同じpathの2応答、同一係数とは強制しない。負値をclampして隠さず、不成立として止める。

固定Rだけではq列とqR列が従属し、θqとθLを識別できない。異なるqとRを組み合わせた矩形が必要。既存MqのbまたはAffineのa,bを新Rへ固定移植しない。新モデルは独立の名前・校正SOURCE・domainを持たせ、既存モデルを比較基準として残す。

初期校正は(qL,R0),(qH,R0),(qL,R1)の3セル、未使用(qH,R1)を独立q×R検証へ予約する。3セル平均の3係数は飽和しており、校正一致をモデル成功やsystematic=0の証拠としない。既知R0の過去holdoutを新モデルの未使用holdoutへ再ラベルしない。R2や長qの未取得領域は別の検証が必要。

cohort平均の価格式は **同一cohortでのS平均＋補正** とする。cheap descriptor sampling等で全lawのE[S]を求める段階は別の科学実行である。有限cohort平均、個別path、モデル平均、全law期待価格は別単位で保持する。

## 必要な場合だけ追加する局所補正

既存のconnected incrementは、κ1(e)=C(e)、κ2(e1,e2)=C(e1e2)−C(e1)−C(e2)、κ3(e1,e2,e3)=C(e1e2e3)−C(e1e2)−C(e2e3)+C(e2)。これとorder/basis条件付きのpair/triple stratificationを**方法として**再利用できる。

piece内IIDのwindow期待多重度は(r_j−k+1)_+ と各order確率の積。ただしH6のpiece間はτ_jが異なり、headが挟まる。全qR eventを一つのIID列へ平坦化しない。piece内、head↔tail、piece境界、outer境界を区別し、controlled構成と元wrapperで必要なclassだけを追加する。以前のuncontrolled中央RTEのK1–K3係数や独立SEの合成仮定を移植しない。

第一候補のholdoutが不適合で、残差がbasis/order classに対応する場合に限り、追加のcompiled局所校正を提案する。最初から全2/3-event組合せ、全signed piece×全Rのcompileを要求しない。compiler全体の再実装や多項式の次数追加を自動開始しない。

## rare event・不確かさ・費用への接続

全lawを主張するにはorder2と複数rare windowの質量・条件付き価格が必要。無条件n8 pilotがorder2を観測しなくても、その寄与を0にしない。legal forced-order対を別診断として取り、元CDF lawの層重みで扱う。強制pathを無条件cohortへpoolしない。未取得層の寄与はnull、または科学的に正当化されたcompiled価格上界×未取得質量として別に扱う。有限標本の最大価格は上界ではない。

trajectory価格の分散、有限平均のSE、係数不確かさ、モデルのR/q systematic、実装lawとideal lawの差を分離する。単純なpath RMSEを平均価格の誤差へ読み替えない。θの共有により予測はround/R/X/Y間で相関する。nominalなJ Σθ Jᵀだけで全体CIは完成しない。未知のjoint covariance/order2/長q systematicはnullのまま。

将来Nを固定する比較では、|ΔG|≤Σ N|ΔC|が価格誤差予算の基本。価格だけの共分散が資格されればVar(G|N)=NᵀΣC N。N/Cの資格や依存が不足する場合、この式だけで全不確かさを表したことにしない。Normalization BをCへ再乗算しない。

全RPEに必要な価格精度は一律の相対%ではなく、AのN(T,R)と候補間費用差に応じて決める。Bは、支配寄与の価格区間とsystematicを含めても候補順位を区別できるか確認し、区別できないセルだけを追加検証する。今回はNの重み付け・価格精度の実数値算定・G比較をしていない。RPEのαtotal=.05は価格モデルの信頼水準ではない。
