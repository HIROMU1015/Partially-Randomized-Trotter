# 量子化学ground-energy研究：C1–C4・既存成果のGPT判断用統合資料

2026-10-14 JST。Codexの技術資料。重要科学レビューは開始していない。候補の優先順位、正式採択、研究継続・終了、論文化方針は判断しない。ユーザー指定のC2/C3一回監査と、固定済み証拠の読取り・統合を完了した段階でMandatory STOPする。

今回の更新は、C2の二点外挿とC3のSchur-rootが既知手順に対応し、現案に独立した低費用の情報取得／回路生成手順を特定できなかったこと。新しい性能結果はない。C1の固定oracle診断、C4の完了監査、Track A/Bの保存成果を再実行・再集計していない。[C2/C3監査全文](qc_c2_c3_final_technical_audit_v1_20261014.md)

## 1. この資料の読み方

目的は元量子化学Hamiltonianのground energyを同じ精度・成功確率で低費用化すること。ただし保存証拠は、その目的の異なる部分を検証している。complex signal、operator approximation、eigenphase bias、shot-weighted compiled RZ、Bernstein十分T予算、active-energy保証幅を同一指標としない。異なる実験の倍率で総合順位を作らない。

下表のidentity IDは末尾の固定本文表および[25本文のcommit/SHA/取得記録](../../artifacts/qc_c2_c3_final_technical_audit_v1/2026-10-14/reference_identity.json)へ接続する。本文をGitHubから実際に取得してlocal fixed blobと照合した。元raw結果の数値再認証・集計ではない。latest Track Bは固定branchのPROJECT_MAPと証拠索引から確認し、IDE ActiveFileは参照先・認可の根拠にしていない。

## 2. 共通評価軸による比較表

| 研究系列・候補 | 元の目的 | 肯定的な証拠 | 否定的な証拠・停止条件 | 既知法との関係 | 実用的な改善機構 | 評価済み指標 | 未評価の費用 | 残る技術的障害 | 固定identity・scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 固定macrocycle | ground-connected PF位相を安い固定Uで推定 | 固定pilot・選択seal・保存oracle診断とcheckerが完了 | 通常16contextでは未選択macroを含むoracleにも登録5%費用優位なし。bad16はeligible不足で未判定 | 現factoryは既知compositionに対応。独立新規性を認定しない | 異符号主要誤差の相殺、共有、finite-h利得を狙うが現catalogで強対照を超える証拠なし | 固定固有位相、overlap/branch/precision、条件付きHeisenberg cost proxy、method regret | 真値なしprecision/selector、state取得、実QPE/RPE、分子、全準備/測定 | catalog/grid外は未判定。regretは選択method内hの事後oracle込み | E02–E03、scope S1。結果5ff55e74、4非分子H、24macro/6h、target1e-5/1e-6、価格3/6 |
| C2 branch確認PF外挿 | PF ground-energy biasを複数幅で除去 | 同branch/剰余仮定のRichardson式と誤差伝播を静的確認 | 新性能証拠なし。独立した安い情報取得法を特定できず実装停止 | 二点外挿・gap/branch・精度配分は既知範囲。RWWは準備も扱う | 二次bias相殺で許容stepを大きくし得るが取得費用が未解決 | 今回は係数未代入の数式・access条件のみ | 複数幅controlled費用、証拠、overlap、alpha、state準備、全energy task | 安いground同定・remainder certificate・同情報強対照差 | E01、今回監査。元fc063696、分子/precision/費用pilot未固定・未認可 |
| C3 self-energy active space | virtual couplingを保持しmode削減 | E<cのSchur-root、単調性、resolvent誤差/scale条件を静的確認 | 新性能証拠なし。安い具体的accessがなく実装停止 | Feshbach/Schur、Neumann、QSVT/LCUに対応。DUCC/SW/trace-downfoldingとの違いは既知 | 小active空間を狙うがdegree/normalization/body-rank/準備が利益を失わせ得る | 今回は静的root/誤差/計算量のみ | K-ground準備、制御access、ancilla、root各点推定、preprocessing、全gate | U<c、一般二体coupling、term増大、α_K。N3条件を一般分子へ移送不可 | E01、今回監査。元fc063696。実分子/precision/費用pilot未認可 |
| C4 finite-RTE cost-aware proposal | 固定meanのsampling負担とwhole-path費用を調整 | 完了報告はsigned mean意味論とD1接続点を条件付き整理 | 現law/whole-path weighting/backward DPの独立生成差は未特定、Phase B停止 | q∝\|a\|/√g、rejection/zero-fill、weighted automataは既知 | 価格に応じproposal変更。効率は費用access・有限状態・実確率評価次第 | 完了資料の技術照合のみ、科学benchmarkなし | range/tail/ceil、Bernstein十分shots、online古典、setup/probability/weight、全ground task | 一般inverse stackの圧縮、finite bits、tail。moment×平均価格だけでは総費用を示さない | E04–E05、C4監査bd512d34。元fc063696/D1 2dcd5673を固定参照 |
| Track A H4 geometry | DF-prefix PRの測定込み回路費用を評価 | 保存geometry比較では部分ランダム化が登録集合内の点最小 | 有限32標本、工程interval重複、P=0。一般energy利益ではない | 既存PR/PF/DFを同wrapperで比較した局所実証 | deterministic prefixとfinite RTEによる短回路・normalizationの交換 | complex-signal bias、解析shot予算×compiled RZ、native指標 | state prep、rare-tail母平均、実shots、長RPE、FT費用 | geometry/compiler/分解を越える移送、同情報S4等の強比較 | E12、S2。7c448337、linear H4/STO-3G、DF12、6距離、T.8/L_D7値/q1,2,4,8 |
| Track A H6独立確認 | 探索後に固定したPR構成の別seed確認 | 8構成×32独立trajectoryでfull PFに対するRZ点推定の改善方向を支持 | 未観測Taylor次数、母平均winner未認定、残64構成未確認 | PR/PF/DFの局所native-cost比較 | prefix5とfinite RTEを同confidence signal taskへ接続 | complex signal、shot-weighted RZ/CX/depth/size | tail cost certificate、state prep、full ground task、外部再現 | rare event、経験的numerical allowance、precision/r条件の移送 | E11、S3。0dd5c3d2、H6 1Å/STO-3G、DFtol1e-8/rank19、sector400、T.8/q1..8 |
| Track B D1 finite-RTE | 同finite targetの表現・native費用を比較 | 12行/全support会計、共通bias/moment/range、strict-phase実sequence価格を取得 | known development provider。全ground優位・online古典費用未取得 | PR/SCU/CTS/return集約と既知law、重要先行claimは既存資料へ対応 | representationのweight/range・acceptanceとnative価格の交換 | Bernstein十分N、T/CX/1Q affine forecast、acceptance/tail | online生成/確率、分子/DF scaling、実shots、ground task | access modelの公平性、価格/保証/準備感度。Tを1Qへ重複加算しない | E13–E14、S4。2dcd5673、3qubit explicit involution/P5/r1,2/εaxis1/200 |
| Track B D2 boundary | 同native隣接inverse規則をwhole trialへ適用 | 保存pair会計で費用・CX最小領域の変化を支持 | 全方式共通の合法最適化。相殺だけで独立算法利益としない | 既知inverse cancellationの具体的実装・会計 | 連続境界のexact inverse除去 | 同law/bias/N下のT/CX/1Q、h感度、quantum Pareto | whole-circuit最適compiler、online古典、全FT/energy task | 指標ごとの勝者とprimary/all-arm比較を区別 | E16–E17、S4。latest固定本文d3a232a4、D2 result d1c59b28、r2同6arms |
| Track B D3 guarantee/price | 保証と取得合成価格の寄与を分離 | 四ledgerで保証によるr選択とpriceによるT順位変化を保存分析 | 結果は十分forecast。小差の頑健性・新規性・全費用未認定。旧v1型不備FAILは残す | 誤差配分・費用交換は既知。先行13claim照合は独立新規性の採否ではない | 共通bias上界が残余幅/Nへ、実priceがnative切片へ影響 | 48保存行、G0/G1×P0/P1、T/CX/1Q、key寄与・h感度 | online古典/DF、実prep/readout単価、地上task、実shots | strict1e-7価格でもnative誤差creditは1e-6固定。primary/all-armを区別 | E15/E18–E19、S4。本文d3a232a4、result f0df9a3b、source dceccaa7 |
| N1 representation selector | 近似biasと回路/shotを同時に減らす | 保存decode修正、全context一回結果、exact fallbackを検証 | fixed selector全3context×RZ/CXでexact選択、certified objective改善なし | basis co-design/既知回路を費用込みで評価した局所結果 | basis節約を元H精度予算へ戻す | fixed complex-signal certified shot×native RZ/CX | 一般分子/FT/最終energy task、他catalog | 近似許容幅がshot負担を上回る条件は未支持 | E08、S5。4c719a72、JW4/ν2/signedDF2/T.6/q1..8、N1 STOP |
| N2 signed charge/L1 fit | 低rank charge表現の精度と算術費用 | 固定Sのweighted entry-L1修正、primal/dual検証 | 旧charge回路はdirectより高価。fit修正はgate利得を示さない | HWP/signed算術と既知LP | 圧縮表現をcompute/phase/uncomputeへ接続 | 固定fitの許容誤差、旧限定回路費用 | 安いadder/FT/スケーラビリティ、全energy task | fit品質とbody/bit-width/ancilla/phase費用の両立 | E07/E23、S6。一件fixed4×2 S、εH.0021、追加scanなし、N2 HOLD |
| N3 occupation保証 | active-energy縮約誤差を構造から認証 | 限定classの証拠・旧族分離・理想族包含、保存幅改善を保持 | 元ε_projでは追加H縮約なし。強い既存族・同品質費用に独立task差不足 | Schur/Fantope/SDP/SOSとの理論的位置が既に確定 | occupancy/coupling方向を残したlower certificate | active-energy保証幅、rational PSD、限定prototype費用 | ground solver/PF/readout/最終energy資源、一般二体 | 有用な狭幅task、一般filling/scaling、未保存因果情報 | E09–E10、S7。ccc575ee、one external/nonnegative density。N3 CLOSED |
| THRIFT現mixed template | mixed relative evolutionのstep節約で元taskを安くする | H4のideal PF評価でstep削減を保存 | 登録価格modelでremaining budget不足、詳細backend投資見送り | 現constructorは既知手順。THRIFT一般の不能性ではない | step節約で追加関数算術/制御を回収する構成 | operator-error step frontier、登録費用予算 | reversible関数実単価、prep/QPE/RPE全ground cost | 追加算術を回収する余白。template以外への再開は別判断 | E06、S8。3f0fbbef、H4linear1Å/STO-3G/DF16全保持、T1,4/εU.001 |
| 過去PF固有値・QPE分枝 | energy bias surrogateとtask接続の妥当性 | 保存windowではPF位相/摂動傾向・dominant branchを確認 | survival phaseをQPE RMSEへ置く条件は不成立、C_useは経験的envelope | PF摂動・有限branch統計は既知。局所fitを保証へ格上げしない | 安価なscreeningの適用窓と失敗箇所を特定 | phase bias、survival、branch weight、QPE RMSE | 地上状態情報の取得、strict bounds、全測定/準備cost | rare branch、conditioning、gap/alias、window外移送 | E24、S9。basebd512d34の保存文書、H4/STO-3G/DF12/全L_D/6δ |
| 過去RPE horizon | 目標energy精度と長q必要範囲を照合 | 目標精度からround範囲を定義、後続round別scheduleを保存 | 4短roundの成功では長qを保証しない。固定r,K外挿は予算/attenuation不適格 | 既知RPE/QPE誤差とfinite RTE会計 | roundごとのPF/r/K配分が必要 | 理想小行列の位相/半径、target horizon | 長q compiled wrapper、fresh shots、最終総費用 | CA/10は暫定、matrix schedule通過と実全費用を区別 | E25、S10。basebd512d34保存文書、H4 1Å/STO-3G/DF12/L_D3/δ.1 |

この表は各成果が支持する範囲を並べたもの。C2/C3以外の新しい技術監査・先行性判定をしていない。C1/C4の否定的所見からTrack A/Bの結果を取り消さず、Track A/Bの局所肯定例からC2/C3/C4の独立新規性を証明しない。

## 3. 保存条件のscope cards

- **S1 C1**：非分子2qubit Pauli H4件、三fragment、macro24、single/subdivision12・S4三・C2三の保存対照、h=1/(8Λ2^s),s=0…5、Λ=fragment係数L1。|00>通常16、|11>bad16、energy target10^-5/10^-6 model units、controlled価格3/6。geometry/basis/DF/L_D/固定粒子sectorは適用外。消費済み940eig pilotと後の8,064行保存診断を区別する。method regretは選択method内のhを事後oracleで選んだ限定指標。今回どちらも再実行0。
- **S2 Track A H4**：linear H4、距離.70/.80/.90/1.10/1.40/1.60Å、STO-3G、8 system+1ancilla、DF fragment12、L_D=0/3/4/5/6/9/12、T=.8、q1/2/4/8、δ=.8/.4/.2/.1。B2/B3は32paired trajectory。complex ε=.005… .1（.05を含む）。structured Gaussian Givens・full measured wrapper、state prep P=0、axisα=.025。εはHaではなくsignal。geometry結果の固定source d0726e94、dataset5aa6c368、review artifact81c78b36を保持。
- **S3 Track A H6**：linear1Å/STO-3G、12modes、alpha3/beta3 sector400、DFtol-only1e-8/rank19/cutoff0、weighted Hermitian projection。固定prefix5の8構成、T=.8、q1/2/4/8、δ=.8/.4/.2/.1、r4/4/4/2、K2/K4、32新trajectory。complex ε=.05/.01/.005/.001、axisα=.025、P=0、Qiskit1.3.0/optimization1/seed17/rz,sx,x,cx/no routing。order4未観測等がありpopulation certificateなし。本文の当時の「remote未主張」を今回の取得性確認と区別し、外部科学再現にしない。
- **S4 Track B D1–D3**：既知developmentのexplicit3qubit involution、p=(1/5,3/10,1/2)、τ=5/7、σ=+、state000、Q0=Z0、V1=RXX01(π/4)、V2=RXX12(π/4)RZZ01(π/4)、Qi=Vi†ZiVi。finite P5(-iτR/r)^r、r1/2、ordinary/partial/P3+tail/local-full/P5/matched-CTS。分子geometry/basis/DF/L_D非該当。同r同finite mean、r間は異なるfinite approximationで同physical target。εaxis1/200、αaxis49/24000、resource failure1/12000/row、family1/20、proposal160/root256bit、primitive native accounting strict1e-6。hはaccepted whole-callの準備/読出し価格でh=0は仮定。D3四ledgerは代替forecastで96axisの一つの同時family campaignではない。P1はstrict1e-7新価格だがnative誤差creditは1e-6に固定。Nは十分forecast、実shotでも必要最小Nでもない。T⊂1Q、classical online NOT_ACQUIRED。最新prior-art13claim/8paperの報告は読取りのみ、再調査しない。
- **S5 N1**：synthetic one-body+signedDF2(weights .7,−.4)、JW4modes/ν2/sector6、scalar .11、化学geometry/basisなし、L_Dなし、deterministic S2/T=.6/q1,2,4,8/δ=.6,.3,.15,.075、model budget.02、εcomplex.04/XYfailure.05/normalization1。44factor/178pair、保存X0X1準備をwrapper費用に含む。source3fe779f8、plan89a40b39、元packSHA70b2631b…は原結果本文へ接続。
- **S6 N2**：旧signed-charge検査はsynthetic JW4で化学geometry/basis/DF/L_Dなし、direct/knownHWPとの固定回路比較。後の補助修正はS=[[1,0],[1,1],[0,1],[-1,0]]、K0=[[.3,−.1],[−.1,.2]]、J=SK0Sᵀの03/30へ+.001、εH=.0021の一件。weighted symmetric entry-L1、intended rationalとstored binary64を区別。旧Frobenius不一致の修正であり、新gate/性能scanなし。
- **S7 N3**：実対称一体+非負unordered density、固定粒子、one external mode。元幅pilotは3stress family×full5/6→active4/5、ν2/3、候補2/H、ε_proj=.01。geometry/basis/DF/L_D/PF窓なし。旧族分離S1 star(scale1.25,full10→9,ν4,external8,δgoal.005)とSOS/OCC特定query分離(two_external_stress_5,full5→4,ν2,external4,δ1/400)は別の保存条件。6H prototypeの幅・費用、理想OCC=SDP、SOS包含、元目標追加縮約0を混同しない。
- **S8 THRIFT**：linear H4/1Å/STO-3G/realRHF8modes/ν4、DF16全保持（rank10は1e-12 reportingのみ）、L_D/PRなし。T1,4、r倍増1…1024、εU=.001、7methods、registered template κ3/6/12。元inputSHA f345af12281836731c3122bb2a4c0c2b389d7c90ded7ba16976d3cfeeb7bd951。step利得と予算不足を区別。現template CLOSED、THRIFT一般の不能性なし。
- **S9 過去PF/QPE**：固定linear H4/1Å/STO-3G/8qubits/DF12 snapshot、全L_D=0…11、δ=.0125/.025/.05/.1/.2/.4、q1/2/4、中央tailをsector exact exponentialで作用。D6 conditioningから小δ二点を除外したfit窓と、全phase/branch比較窓を区別。survival phase、peak bias、QPE RMSEは異なる。経験的C_use/envelopeをrigorous upperへ変換しない。過去local数値照合を現在immutable CIや新認証にしない。
- **S10 過去RPE**：H4鎖1Å/STO-3G/DF12/L_D3/δ.1、保存snapshot、r4/K2、PF/RTE/stat位相予算.02/.02/.36rad、βRPE.4rad。CA/10は暫定。q8と長q4096/32768の診断は小sector matrix参照。後続δ/round別scheduleが存在しても、長qの実compiled wrapper・fresh shots・全task評価完了ではない。

## 4. 現時点で主張を混同しないための整理

| 証拠区分 | 記述可能な内容 | この資料では認定しないこと |
| --- | --- | --- |
| 既知理論 | PF外挿・Feshbach・QSVT/LCUなどの入力条件付き構成 | 本project発の原理・独立新規性 |
| 保存局所結果 | 指定model・入力・価格・精度・windowでの肯定/否定観測 | 他分子、全precision、全compiler、全ground taskへの一般化 |
| 実装成立・技術監査 | decode/証拠検査/IR/strict-phase価格/保存会計の完了範囲 | 安価な本番scalable implementation、科学的採択 |
| 未評価 | classical online、overlap/state prep、tail、全RPE/QPE、強対照同情報比較 | 0費用、oracleアクセス無料、総資源優位 |

C1のoracle未達はfixed catalog/grid内の説明を絞る。C1全般の不能定理ではない。C4の既知手順還元は現構成の技術所見で、正式研究終了は未判定。C2/C3に未解決costが残ることは継続価値の証明ではない。N3の保存理論やTHRIFT step改善は保持するが、終了済み系列の再開根拠として無断使用しない。

Track AにはPRの費用改善を支持する限定証拠が存在する。一方、global manifestの既存FAIL、rare tails、状態情報・準備、全round/全taskの不足を残す。Track Bの保証/価格/representation/resource mapは別の肯定的技術証拠であり、既知lawの新規性や一般量子化学優位を意味しない。全地上energyの最終total-cost評価を完了したというclaimはない。

## 5. GPTへ残す六つの判断と必要情報

| 科学的判断 | 固定証拠から渡せる材料 | 判断を変え得る未取得情報／承認境界 |
| --- | --- | --- |
| C1を正式終了するか | 通常16のoracle未達、known composition、制限付きregret | 現catalog外の具体的機構が必要という科学判断。Codexが新探索を起動しない |
| C4への独立算法投資を見送るか | 既知optimal law/DP/rejectionへの還元、圧縮/全費用未知 | 同情報既知法を超える具体的生成規則・tail/古典費用証拠。未知であるだけでは差にならない |
| C2/C3を継続するか | 今回の入力・誤差・access・cost監査、独立生成差未特定 | C2の安いbranch/remainder取得、C3の安いencoding/window/ground solver。productionやpilotは別認可 |
| Track Aを発展させるか | H4 geometry、H6固定独立標本のsignal資源所見 | tail母平均、状態準備、同情報強対照、全ground taskへつながる最小証拠。新計算は別承認 |
| 四候補を見送り課題を再設計するか | 既存結果が残すtask/access/costのgap | 目的を維持して何を解くかの重要科学判断。第五候補は今回生成しない |
| 論文化claimと最小証拠は何か | 既知理論/局所肯定/局所否定/未評価を分けた表 | 先行性、外部再現、taskの意義、一般性、同条件総費用の科学的評価 |

この表は選定・推薦順位ではない。C2/C3pilot契約・新science認可を発行しない。GPTレビュー代行・C1終了・C4採否・主研究採択も行わない。次の担当は、ユーザーの**別明示開始承認を受けたGPTの重要科学判断**。

## 6. 固定本文identity一覧

以下の表は取得本文のmetadataから生成する索引で、性能値の再計算ではない。関連source/input/authorization/raw-resultの詳細identityは各固定本文のリンクから取得できる。今回の25本文照合を、それらの全raw証拠の新しい再認証へ格上げしない。

| ID | 固定commit | 取得本文 | SHA256 |
| --- | --- | --- | --- |
| E01 | `fc063696026b2500d9608abe447849fd76274ed5` | [docs/research/qc_algorithm_candidates_technical_prevalidation_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc063696026b2500d9608abe447849fd76274ed5/docs/research/qc_algorithm_candidates_technical_prevalidation_20261013.md) | `4fb5f542a921c8859aa37989d141d27abf80054901b8e04b3409a62ac4a83ff3` |
| E02 | `5ff55e749ceb33e54da0b3c70ac6ac08599ebb82` | [docs/research/qc_c1_saved_oracle_diagnostic_v1_results_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5ff55e749ceb33e54da0b3c70ac6ac08599ebb82/docs/research/qc_c1_saved_oracle_diagnostic_v1_results_20261014.md) | `5c9f9135cc50bde4171f7a1a5a7837c7bf564962071485978e1eb476b4749fe9` |
| E03 | `5ff55e749ceb33e54da0b3c70ac6ac08599ebb82` | [docs/research/qc_c1_saved_oracle_diagnostic_review_handoff_v1_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5ff55e749ceb33e54da0b3c70ac6ac08599ebb82/docs/research/qc_c1_saved_oracle_diagnostic_review_handoff_v1_20261014.md) | `91ae779917d65c68c819672a8c791cb32f32d76a8fdeabad522ae9f5701acbf7` |
| E04 | `bd512d346cbd65e1ee29a5ce3209b6b889fa8e38` | [docs/research/qc_c4_finite_rte_method_efficiency_technical_audit_v1_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bd512d346cbd65e1ee29a5ce3209b6b889fa8e38/docs/research/qc_c4_finite_rte_method_efficiency_technical_audit_v1_20261014.md) | `5752758949ee12c027b21476545688db4b097d413eb9bc25dd214fdb8a0de2de` |
| E05 | `bd512d346cbd65e1ee29a5ce3209b6b889fa8e38` | [docs/research/qc_c4_finite_rte_gpt_handoff_v1_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bd512d346cbd65e1ee29a5ce3209b6b889fa8e38/docs/research/qc_c4_finite_rte_gpt_handoff_v1_20261014.md) | `b62c540150d5e1c5f27bcb436255f7afb692b84aae4d38b55f64193efa2a0e5e` |
| E06 | `3f0fbbefa8f395a59f8a380691100dc0a8f656ae` | [docs/research/qc_mixed_access_feasibility_v1_closure_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3f0fbbefa8f395a59f8a380691100dc0a8f656ae/docs/research/qc_mixed_access_feasibility_v1_closure_20261013.md) | `85ca959c90cd816c7c54d85ab171c7a8278ef723669782e1b52e3da996ec20c4` |
| E07 | `3f0fbbefa8f395a59f8a380691100dc0a8f656ae` | [docs/research/hamiltonian_algorithm_design_results_20261011.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3f0fbbefa8f395a59f8a380691100dc0a8f656ae/docs/research/hamiltonian_algorithm_design_results_20261011.md) | `d3659a1c698f8fc963f2f4e6c40d8f5311a8dc938d11ddb0c6bee808453088a0` |
| E08 | `4c719a72eb8caae2f77fc746bd4672d610d97610` | [docs/research/n1_decode_recovery_v1_results_20261011.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c719a72eb8caae2f77fc746bd4672d610d97610/docs/research/n1_decode_recovery_v1_results_20261011.md) | `7877d64a9261d0bbd3aefffc768ab13853b1d2f2cf10534a560e6c8b47d6d153` |
| E09 | `ccc575ee339be091fd4031272c38bee13866f097` | [docs/research/n3_saved_attribution_v1_claim_and_observability_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ccc575ee339be091fd4031272c38bee13866f097/docs/research/n3_saved_attribution_v1_claim_and_observability_20261013.md) | `f9f0278121444ea15dc07a2ff5677d682cff28732b182d9db8adc04eeeb2b3e9` |
| E10 | `ccc575ee339be091fd4031272c38bee13866f097` | [artifacts/n3_achievements_closure/2026-10-13/research_decision.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ccc575ee339be091fd4031272c38bee13866f097/artifacts/n3_achievements_closure/2026-10-13/research_decision.json) | `8f2d81e4c1f7f0f567d9a0657b3d65ed591281d05e09c604740aad4661ad0624` |
| E11 | `0dd5c3d204dab9b731268a1aa2add621080f3d20` | [docs/research/track_a_h6_independent_confirmation_result_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/docs/research/track_a_h6_independent_confirmation_result_v1.md) | `6f66db8e5c3fcf14b50e2480d2d1404c58834215a3bf631362de0dabc6b400a7` |
| E12 | `7c4483378047b57f514c47dad98905e3f2a9a756` | [docs/research/track_a_h4_geometry_results_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/docs/research/track_a_h4_geometry_results_20261013.md) | `a6034da2108ee6be4730989c1d77b02619c4c603f3f5813d8664ae993527186c` |
| E13 | `2dcd56737be35526f6436a52835f59d817adf30c` | [docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md) | `9589c8796bad54128e714d693669353eb1f588774f05dbce5e1035831d6ac9e2` |
| E14 | `2dcd56737be35526f6436a52835f59d817adf30c` | [docs/tracks/algorithm_codesign/d1_saved_resource_analysis_and_gpt_handoff_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/docs/tracks/algorithm_codesign/d1_saved_resource_analysis_and_gpt_handoff_20261013.md) | `9b425dd545b84694e689b42d47be025b70683fc1193b00ffacd0012a9c5d52b3` |
| E15 | `d3a232a446062e6baadef2b69eab87faa0285363` | [docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md) | `bdedcd66c0ce4e160fc40a0a176ecdfa9716aac85b3895f7acb0653efcc0152f` |
| E16 | `d3a232a446062e6baadef2b69eab87faa0285363` | [docs/tracks/algorithm_codesign/d2_boundary_results_and_gpt_handoff_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d2_boundary_results_and_gpt_handoff_20261013.md) | `4328f109bc1a4c560bbc908edaaeacc078702acca4c34bf876736e09d47f550d` |
| E17 | `d3a232a446062e6baadef2b69eab87faa0285363` | [docs/tracks/algorithm_codesign/d2_final_saved_integration_and_gpt_handoff_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d2_final_saved_integration_and_gpt_handoff_20261013.md) | `04ba522cb15f8239ddcf6bd8415b9ccecb9b3987286f4b44401fa0091a42d811` |
| E18 | `d3a232a446062e6baadef2b69eab87faa0285363` | [docs/tracks/algorithm_codesign/d3_final_saved_mechanisms_and_gpt_handoff_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d3_final_saved_mechanisms_and_gpt_handoff_20261014.md) | `e941f594a6602900815dcb21aee8de3434f0fb1f581f64873121e6244482a0e3` |
| E19 | `d3a232a446062e6baadef2b69eab87faa0285363` | [docs/tracks/algorithm_codesign/d3_v2_results_and_gpt_handoff_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d3_v2_results_and_gpt_handoff_20261014.md) | `92e3befb31efbb723b53e6fdde248f42a55de25fff534d82dfd84cd0978a5732` |
| E20 | `bd512d346cbd65e1ee29a5ce3209b6b889fa8e38` | [docs/research/研究概要・現状.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bd512d346cbd65e1ee29a5ce3209b6b889fa8e38/docs/research/研究概要・現状.md) | `af41a1726a03b79f26fc5129b4fa330b78c2e75b5234550ced348542298e72d6` |
| E21 | `bd512d346cbd65e1ee29a5ce3209b6b889fa8e38` | [docs/research/研究目的・研究課題.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bd512d346cbd65e1ee29a5ce3209b6b889fa8e38/docs/research/研究目的・研究課題.md) | `08f9af0ab09747c2d9b4497c6ec94587b4f42c147efbf42a564f785bddf2fc26` |
| E22 | `bd512d346cbd65e1ee29a5ce3209b6b889fa8e38` | [VALIDATION_STATUS.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bd512d346cbd65e1ee29a5ce3209b6b889fa8e38/VALIDATION_STATUS.md) | `87e425ac9a4734cfc13644275aa246b7181df178a60203c074e2077b2a495d93` |
| E23 | `bd512d346cbd65e1ee29a5ce3209b6b889fa8e38` | [docs/research/n13_redesign_results_20261011.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bd512d346cbd65e1ee29a5ce3209b6b889fa8e38/docs/research/n13_redesign_results_20261011.md) | `69312ddd372577af93fde4447954ae7e89385953d895368d810e85c59502a444` |
| E24 | `bd512d346cbd65e1ee29a5ce3209b6b889fa8e38` | [docs/pf_delta_validation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bd512d346cbd65e1ee29a5ce3209b6b889fa8e38/docs/pf_delta_validation.md) | `3a33518a7b5053f29de0861361e064365898e99cbaf0b8e25f36a3da6202c2cb` |
| E25 | `bd512d346cbd65e1ee29a5ce3209b6b889fa8e38` | [docs/rpe_target_round_horizon_validation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bd512d346cbd65e1ee29a5ce3209b6b889fa8e38/docs/rpe_target_round_horizon_validation.md) | `7b2a8956fdfc1125d914b4fae131cc6fd80b9d8d71679bf54243bf308e08447c` |

## 7. 保全と最終STOP

[scope](../../artifacts/qc_c2_c3_final_technical_audit_v1/2026-10-14/scope.json)、[status](../../artifacts/qc_c2_c3_final_technical_audit_v1/2026-10-14/method_status.json)、[既存374files](../../artifacts/qc_c2_c3_final_technical_audit_v1/2026-10-14/preservation_index.json)。旧144manifest entries/non-result fields、既存global `relevant_commits[7].description` 欠落FAILは保持する。今回の局所document/hash検査はglobal validation PASS、immutable CI、外部科学再現ではない。

N1 STOP、N2 HOLD、N3 CLOSED、THRIFT現template CLOSED、C1/C4旧結果・source・認可・STOPを保持。最新Track Bの固定本文をread-onlyで参照するだけで、別branchを変更・mergeしない。終了済み実行認可を流用しない。

独立branch `qc-c2-c3-final-audit-integration-20261014` に二資料とmetadataをcommit/pushし、[GitHub再取得receipt](../../artifacts/qc_c2_c3_final_technical_audit_v1/2026-10-14/publication/receipt.json)でbytes/SHAを確認後Mandatory STOP。新数値・新production・新候補・C1/C4再調査を開始しない。今回の限定監査を反復せず、重要科学判断の別開始承認を待つ。

## GitHub公開確認・最終STOP

[再取得receipt](../../artifacts/qc_c2_c3_final_technical_audit_v1/2026-10-14/publication/receipt.json)：payload `d91991038b1a2b97fac52a00d8ed4ebb413e2214` の22変更blobs、374既存保護files、25固定参照本文をGitHubの独立cache（local alternatesなし）からbyte/SHA照合した。receiptと本追補は後続metadataとして公開し、最終取得で全リンクを照合する。新科学計算・保存数値事後評価・数学プログラム・C1/C4再監査・production・GPT科学レビューは0。正式採択/優先順位/継続/終了/論文化を決定せず、Mandatory STOP。
