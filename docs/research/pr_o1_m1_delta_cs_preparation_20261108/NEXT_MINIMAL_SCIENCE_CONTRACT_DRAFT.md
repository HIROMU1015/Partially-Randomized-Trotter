# 次の限定科学契約草案：C-Sを最小単位で判別する

2026-11-08 JST。**DRAFT／未採択・未許可・未実行。今回のSOURCEは技術plannerであり、H4科学runnerの完成版ではない。**

## 1. 問いと今回の停止点

M1でのglobal collect・既知selector・matched splitを再実験しない。C-Sでは、**同位置のcoarse＋canonical operator residualが、strict合成の安さとnormalization/rangeの負担を交換して、通常S0・一様S1に対する追加利益を持ち得るか**を問う。最初から全18round RPEや多くの角度/maskを評価しない。

現SOURCEで完了したのは順序・role・phaseを保つ接口と人工9test。実H4のslot抽出、角度provider、strict合成・誤差bindingは未接続である。以下のV0を先に行う提案とし、V1以降はその結果後の別許可とする。

## 2. 入力候補と固定すべき条件

候補は**A1 N1の保存linear H4/1.00Å/STO-3G/legacy DF12/8 system qubits/全4電子sector/L_D3/m0,q1,δ=.02,r1,K2/coefficient_atol0**。N1 `744e1a09661ebba6841fe5faf456f88c41c47780`のH/partition/preparation/tail/actual Q/selected full、N2 `5e9bd08068738cdf1070bdc43bee8194b113ae33`の元H保証groundを参照する案。

H_D/H_RとPF順序・scalar/one-body/JW orderingは全arm共通。SCF/DF/CDF/preparation/state solveを再実行しない。P3のrank10/T_eff-only split、F2の4mode占有fixtureを代わりの同条件にしない。

旧engine認証`fdf00cee3c87d804845497afc89ac6a71f3b3f59`は**旧selected列の証拠**であり、C-Sの変更後のgate列・実装誤差を覆うと仮定しない。V0ではstate/Bornを一切使わない。既存RPEの一般正当性・energy bridgeは再検証しない。

## 3. 次の最小許可案 V0：保存primitiveのmetadata＋strict局所資格

| 項目 | 有限scope案 |
|---|---|
| 目的 | 既存値と異なる実H4角度がstrict providerへ接続でき、residualの局所mean/errorと単価交換を扱えるか。full量子費用の改善判定ではない |
| 入力固定 | 上記保存N1 full payload/role付き操作記述をbytes/hash固定。元H/state identityは参照のみ |
| 対象数 | eligible D diagonal classとR diagonal classから**各1、最大2つの局所回転spec**。事前canonical slot順のfirstを使い、signal/価格から選ばない。選択roleが保存記録から確定できなければSTOP |
| coarse/residual | 一つの事前固定coarse政策、各slot1coarse。探索なし。exact Angleからresidualを記述し、πやscalar位相を捨てない |
| finite policy | canonical ordinary paired residualの**一つのr/K・数値精度policy**を事前固定。適用域外ならr/K探索で救済しない |
| science call上限案 | 保存input/binding1回、2slot記述1組、局所target/coarse/residual operatorの比較2組、有限residual mean/error2組。実H4 full-state作用/Born/RNG/量子shotは0 |
| strict合成上限案 | K_res=2を採択した場合、通常2＋coarse2＋2orders×2residual=**最大8 positive angle/precision/tool keys**、各1 acquisition/guard。exact fast path/重複keyは再取得しない。負角はactual adjoint。採択Kが異なる場合、この8を流用せず先に有限数を確定 |
| 価格 | T/T†・CX・Clifford・depthを別保存。局所priceだけでfull改善としない。RZ-onlyへ勝手に変更しない |
| 出力 | 実slot/role/generator/角度/coarse/residual、finite law・coefficient/proposal/weight、normalization/range、strict誤差・tool/profile、価格/古典費、call/例外/STOP |
| 禁止 | full H生成、N1/N2再計算、full-state evolution/Born、sampling、full interrogation build/compile、N/G/variance/ランキング、全round、新selector/Parent探索、旧cacheの無照合使用 |

**実行前にまだ採択が必要**：対象保存gate/roleの正確なpath・slot規則、coarse政策、K_res/r_res、strict合成precision/toolとerror予算。最大8keyはK_res2の場合の有限scope案で、今回の正式条件ではない。新native providerが対応できない角度表現の場合は、人工correctnessを先に追加して新SOURCEを固定し、未対応のまま科学実行しない。

### 再利用と不足

`wrapper_sequence.Angle`と今回のslot planner、既存canonical paired law/weight・error_accounting、actual adjoint、strict primitive guardのcoreを優先。atan/pi限定の旧native AngleとD1/D3登録cacheを、任意radian/π混合specへ偽装しない。新しい汎用synthesizerやsamplerは不要だが、角度/key/providerの限定adapterと実inputのhash bindingは必要で、現在の科学入口には未実装。

### V0で判断が変わる条件

- strict coherent targetを扱えない、coarse誤差やresidual rangeが未保証、同位置bindingが不成立なら、この接続のままV1へ進めない。
- 通常合成に対する局所priceの差がなく負担だけ増える場合、full pilotを行う積極的材料は得られない。局所結果だけからSCV/PR全体やC-S全般のNO-GOは採択しない。
- strict対応と局所tradeoffが得られた場合だけ、元の位置と元tailを併せたV1の価値を判断する。局所tradeoffを期待量子費用改善とは呼ばない。

成功・不成功いずれも結果公開後mandatory STOP。自動V1移行・精度変更・合成retryなし。技術的な実装debugと、失敗した科学acquisitionの再実行は区別し、後者は別許可とする。

## 4. V1以降で初めて比較できる量（今回は設計のみ）

V0後に継続すると判断した場合、元S2とH_D/H_Rを保つ**一つのshort q1条件、S0/S1/S2×X/Y**に限定した次契約を別途固定する案。

| 必須項目 | 比較規約・未解決事項 |
|---|---|
| 強baseline | S0通常strict合成、S1同じcoarse/residualの一様適用。融合/Clifford fast path・precision・入力情報/価格機会を共通化 |
| original/residual law | 元N1 actual Qのpathと追加residualを順序付きで結合。独立性がある場合だけfactorization。実proposalからWを導き、補正meanを追跡 |
| signal/bias/半径 | 同元H state上のideal PF→finite residual→native mean。X=Re、Y=Im、bit0→+1、Sdg→H、controlled diag(I,U)、scalar/identity/event phaseを保持 |
| 許容誤差 | finite/residual truncation、coarse synthesis、係数/角度/finite-bit/native差、元PFとの差を分離。既存engine allowanceを変更列へ無断流用しない |
| 必要shot | 共通target・ε/α下で、既存bounded confidence/会計coreにactual range/半径と誤差headroomを入力。Nは未資格ならnull。B補正とraw減衰を二重計上しない |
| 価格 | 同native T profileのfull no-prep価格。TとCX/depth/古典費を別vector。旧RZ価格・F2/P3単価の転用なし |
| 不確かさ | 原tailとresidualの希少branch、価格母平均、cal/price情報取得費、返却/STOPを含む。少数path compileを期待価格certificateにしない |
| 有限実行 | source/input・exact class/coarse/residual policy、mean call上限、価格program/cohort数、STOPを実装依存で事前固定。全18roundを含めない |
| 判別 | 同じ精度/成功条件でN×full単価の交換がS0/S1後に残るか。未取得項を0補完しないと比較できなければUNKNOWN/STOP |

V1のactual finite mean・law・実state作用・full compile・価格・N/Gは新科学計算。今回の指示では未許可。単なる通常RPEの再検証、長round価格modelやenergy bridgeの再実行を必須工程にしない。

## 5. 承認と保全

次担当はSERVER1 Codex。まず上記未確定provider/bindingを限定技術作業として完成し、必要な人工差分だけを試験して科学SOURCEを固定する。そのSOURCEと有限V0条件へのユーザーの明示許可後に初めて実H4を投入する。承認証明書/grant発行を新要件にしない。

元main/dirty/untracked、F1/F2/PS48、P3、SCV224/448、N1/N2/engine、研究C NO-GO/C1凍結、旧grant/ledger/marker/STOPと他jobを保全。共有環境を変更せず、実空き資源で効率化する。今回の技術SOURCE公開照合後mandatory STOP。
