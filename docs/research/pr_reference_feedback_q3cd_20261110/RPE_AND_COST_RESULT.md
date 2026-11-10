# Q3-D：成功保証・費用境界の接続

固定条件・証拠・実行範囲は[REPORT](REPORT.md)、nativeの構造証明は[NATIVE_GUARANTEE](NATIVE_GUARANTEE.md)。数値の正本は[COST.json](../../../artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/COST.json)と[D.json](../../../artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/D.json)。研究上の正式採択と実機保証は別に残る。

## 1. 340 shotsの前提と結論

Sを保存m0–16の全34座標の成功eventとする。Q3-A/Bの証明によりS上で実装は必ず正しいbranchを一様受理し、aliasを保ち、真のenergyを含むpriorを返す。feedbackはその履歴だけから決まり、同じbinary64 parameterを全m17 trialへ使う。

| 保証材料 | 今回接続したもの |
|---|---|
| Q0 | 未知ρ、bias、Hoeffding、n340の全count幅、共通δsafe=.000748774257 rad |
| Q1 | count型、feedback生成、P(−φ)→H、confidence decoder |
| Q2 | 保存16 raw/native correctness・signed readout trace |
| Q3-A/B | S上のbranch/alias、一様数値受理、δarith、正のhull丸め余裕 |
| Q3-C | 全contextの同一core lowering、Pのangle保存、virtual-stateによる旧raw disk移送、fresh consumer |

Q3-Cは各固定履歴のphysical meanを `ρ sin(s+b+eφ)` のQ0許容集合へ入れる。native誤差一般をradに足さず、旧raw diskからのρ下界とbias上界を使う。B′はρL=.5726873297263416、bias上界=.4−.38598699711850315。Reference-B1はρL=.5726872813784947、bias上界=.4−.3859869126962799。bias計算では保存binary値をexact Fractionへ変換する。

新たな純粋角度chargeはδarithだけで約2.22034e−16 rad以下、P lowering自体は0 radの解析恒等式。旧H/coreのnative誤差、finite-bit、SOURCE、B1 normalization等は消していない。Q3-Bの正のκ/ωを使う

`(κ+ωE)t17 + ωQ0 < δlo−δsafe`

をexactに再確認した。Referenceで丸め後余裕≈8.614430033512188e−13 rad。未知のnative/runtime誤差をこの小さい余裕へ無償で追加できるという意味ではない。

各S上の履歴で、canonical producerからの340 fresh independent trialを使うとQ0/Hoeffdingの条件付き失敗確率≤1/360。その履歴について積分し、`Pr(fail)≤34/720+1/360=.05`。受理失敗のpost-selectionはしていない。confidence hullの非空性・幅はcount成功event上に成立し、実装が不整合countを返すS外はfailureに含める。

最終推定をexact rational energy-hull midpointとすると半幅≤`.4/(131072·F.from_float(.02))` Ha≈.000152587890625 Ha<.00015936 Ha。入力stateの用意は仮定し、その古典記述・E0・gapを新しくfeedbackへ渡していない。

判定：**`GUARANTEE_CLOSED_IN_FIXED_MODEL`**（両arm）。ideal native/reset/measurement、元のSOURCE/Hamiltonian表現、binary64/192-bit primitiveモデル、IID bitsという旧前提を継承する。実機／任意library／任意RNGへの無条件保証ではない。

## 2. 価格移送とbreak-even

全contextで旧YとQは同じnative gate名・wire列、1 RZのparameterだけが異なるので、任意角・全trajectoryについて資源jごとに `CQ,j=CY,j`。angle cutoff／numeric optimizationはない。RZ(0)も残り、depthのDAG構造も同じである。

RZ/sizeは旧Xに対してQが+1、CXは同じ、peak qubitsは同じ。新RZの合成T数が一定という意味ではない。

`C17,XY,j =626 μX,j+626 μY,j`

`C17,Q,j =340 μY,j`

`saving17,j =626 μX,j+286 μY,j >0`。

追加P費はμYに含まれる。回路外にquantum feedback operationを追加しないので、別建てA_native=0はSOURCEから従う。古典prefix・angle生成は1 RPE attemptあたり1回、backendとの通信・校正・回路生成CPU/RAMは未価格。これらをnative量の0埋めとして扱わず、総native削減と実行時間削減を区別する。

m0–16のN／単価を固定するため全18roundの節約量はm17節約量と等しい。全計は各armの35 interrogation type（prefix34軸＋末尾Q）をshot加重する。旧方式は36軸、全shots8,598。新方式はprefix7,346＋Q340=7,686。Referenceの比較元はB1の626/axisであり、旧naive630/axisからの改善と混同しない。

## 3. 全18round境界と相対比較

概数を以下に示す。判定は機械可読結果のexact Fractionで行い、概数は外向き端点ではない。率の下限は小数6桁へ切捨て。

| 資源 | B′ Q方式 [下界,上界] | Reference Q方式 [下界,上界] | ReferenceのB′ Q比・削減率下限 |
|---|---:|---:|---:|
| native RZ | 約1.772330242424e12 | 約[1.767228305686,1.767228306987]e12 | 0.287865% |
| native CX | 約[2.508915762394,2.581457462545]e11 | 約[2.306330605224,2.378882143454]e11 | 5.182861% |
| gate size | 約[2.591669745107,2.598923915122]e12 | 約[2.566311174722,2.573566329846]e12 | 0.698523% |
| shot加重DAG depth | 約[4.580868289117,5.696842102002]e11 | 約[2.069991668956,2.686606806406]e11 | 41.351581% |

| m17 RZ | 従来626 X＋626 Y | 新340 Q |
|---|---:|---:|
| B′ | 約2.470545679180e12 | 約6.709149609698e11 |
| Reference-B1 | 約[2.463326250634,2.463326252027]e12 | 約[6.689544132813,6.689544136598]e11 |

両armの従来X/Y比全18roundRZ削減率下限は50.382149%／50.381058%。これはPR内部でのreadout改善であり、Reference固有の機構ではない。Reference固有の差は、**両armをQ化した後の**上表の比較に分離した。

CX下界はReferenceの保護されたD/frameだけを残し、rotation/CSWAP CXは捨てた解析下界。sizeはRZ/CXの境界とSXの厳密期待数・測定/resetを加算する。SXは旧fused D・basis run式・products・CSWAP・Hから数え、消去対象でない。4つの自由symbol templateで最終pass前からX-freeを確認したため、未取得Xを0にしたものではない。F†側の保存inverse-D 20 recordでもSX/CX/RZ多重度は一致した。

depthは旧V5の安全側上下界と、その実SOURCE inverse／非X鎖の既存補強を維持する。期待depthの最適値・実機時間・routing後depthではない。上下界の端点を実現可能な最適policyや独立統計信頼区間とも見なさない。

判定：資源別 **`TOTAL_COST_REDUCTION_ESTABLISHED_IN_FIXED_MODEL`**。追加qubitsはQで0、既存のB′9／Reference17は維持。CX・RZ・size・depth・qubits・classical費を独自換算して合算していない。

## 4. 検算・未資格・引渡し

保存V6-PのB′／Reference-B1全18roundRZ総計をexact再現し、今回の全和／差／負のshot差の端点相関を検算した。Q0やQ3-Bの科学campaignは再実行せず、m17の新bit列／whole circuit／実測は生成しなかった。定義済み物理shot要求のSOURCE対応と数学的成功保証であり、340-shot実機campaignの観測結果ではない。

T-count／T-depthは未資格。任意feedback RZをClifford+Tへ移すにはword、strict relative-phase／誤差budget、precision／全support policy、T価格を別途資格化する必要がある。今回その合成・guard・価格計算を行わず、旧CTの負結果・STOPを維持する。

強Full PF／Truncated PFにも同じ測定改善機会を与えるべきで、今回の数値をそれらへのPR優越へ移さない。実RNG、backend noise、classical時間／memory、正式な保証採否はGPT／ユーザーの次の判断である。Codexは公開・remote照合後mandatory STOPし、後続実装・科学計算を自動開始しない。
