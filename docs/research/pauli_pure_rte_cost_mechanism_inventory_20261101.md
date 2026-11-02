# Pauli-JW純RTEの低費用：既存証拠と未分離の原因

2026-11-01。既存証拠の技術調査。**現在説明できるのは、候補固有のbias・normalizationから決まるshot負担と、compiled回路価格との競合である。決定論的prefixだけの因果効果や全分子に共通する単一原因は未実証。**

新しい科学計算、数値再解析、統計量・倍率・Gの再計算、再ランキング、sampling、build/compile、Hamiltonian/state/reference生成は実施していない。以下の候補・順位は既存報告で既に選抜されたものを引用した。独立した詳細科学レビュー、主要claimの正式採択ではない。

## 1. 固定資料・科学条件の対応

| 証拠 | 結果を読む固定commit | 科学SOURCE・対象 |
|---|---|---|
| [H6 Hybrid追加検証][h6report] | `00958ae1945c780718b2deef07d5dab1096a5cd8` | `887b69e449cb3c378286f09d0d7626599850fb67`、解析 `af891fd45805454c6592bbee5441266f14235ffd`。p78/q8/r16,32,64/K2・p684/q1/r16/K4の固定4候補 |
| [H4/H6純RTE event拡張][evreport] | `10cb3e5cca7834ebf75301bd91bd093ca51d4a6b` | `e755ecf581600192fd2745d7066e410cebd933d3`。L_D=0/q8、新候補8。旧s64とHybridの価格は引用 |
| [H2追加独立確認][h2report] | `4aea054b3369709bc0d35ff081071e62b5f77cbe` | `bf5f2339f723f4edb7bf91a8ca88f33251bbef57`。Phase Aは既存8候補の確認価格追加、保存target/Nを再利用。Phase Bはs64/128の探索3候補 |
| [Pauli-JW統合資料][synthesis]・[latest15.csv][latest15] | `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b` | 10/26までの保存14/15とH4例外の案内。10/27のH6 Hybrid追加検証は別の更新範囲 |
| [構造調査][structure]・[後続D1][d1]・[D2][d2] | 本調査では上記 `b6c89969…` の保存版を使用 | D1科学SOURCE `9843b803878db6c3aee7ab5e7c6bb4818572c944`、D2 `74515d8eb4f073c40fc783681a3775dbca89e9b1`。固定H4/prefix27/q8/r4/K4の別対照 |

指定のH6 REPORT、CANDIDATE/COMPARISON CSV、sampling_design、price_statistics、SOURCE、verificationをGitHubから実取得した。event_cost_trends、selected_cost_breakdown、H2資料、統合資料とlatest15、構造/D1/D2も固定refで取得し、主要18ファイルのremote Git blobとlocal Git objectの一致を確認した。H2ディレクトリに `REPORT.md` はなく、正しい本文は `docs/h2_pauli_rte_validation_20261026.md`。event拡張のSOURCE入口は [CURRENT_SOURCE.json][evcurrent] → [SOURCE_e755…json][evsource] であり、存在しない汎用 `SOURCE.json` を確認済みとは扱わない。

共通条件はlinear H2/H4/H6、隣接1.00 Å、STO-3G、中性singlet、保存Pauli-JW非identity項M=14/184/918＋constant、保存raw ray/full-H reference・共通exact norm、S2、T=.8、P=0。DF分解は本比較に使わない。constantはL_D=0でも保持する。central-rotation control、full Hadamard cosine/+Im sine、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、追加topology/routing指定なし。価格は状態準備を除くRZ、GはRZ×shot。

主会計は**仮想compiled残差ρ=0**。H4は既存FLINT包含と候補別precision schedule、H2/H6の追加targetは既存tight L2（Arb scalar128bit・vector中心53bit）。理想law信号の包含を実compiledρの保証へ転用しない。結果公開commit、科学SOURCE、旧値の出所を区別する。[event報告][evreport]、[H6 SOURCE][h6source]、[H2 SOURCE][h2source]。

## 2. 代表比較：BだけでもCだけでも総費用は説明できない

R/Hは純RTE/Hybrid。粗=.05、厳=.0005。s=q*rはevent数で、Kは最大偶数event次数（多項式次数K+1）、S2の次数とは別。全行は確認cohortの引用で、nは古典価格標本数、Nはshot十分条件の回数である。

| 略号 | ε_sig | 保存candidate ID | L_D | q/r/K | s | 確認n・出典 |
|---|---|---|---|---|---|---|
| H2-R | 0.05 | `PR_p0000_q1_r4_K2` | 0 | 1/4/2 | 4 | 384 / [h2csv][h2csv] |
| H2-H | 0.05 | `PR_p0001_q1_r4_K4` | 1 | 1/4/4 | 4 | 96 / [oldcost][oldcost] |
| H4粗-R | 0.05 | `p000_q8_r8_K2` | 0 | 8/8/2 | 64 | 384 / [evbreak][evbreak] |
| H4粗-H | 0.05 | `p036_q2_r8_K4` | 36 | 2/8/4 | 16 | 48 / [oldcost][oldcost] |
| H4厳-R | 0.0005 | `p000_q8_r8_K4` | 0 | 8/8/4 | 64 | 384 / [evbreak][evbreak] |
| H4厳-H | 0.0005 | `p001_q8_r8_K4` | 1 | 8/8/4 | 64 | 48 / [oldcost][oldcost] |
| H6粗-R | 0.05 | `H6_p000_q8_r64_K2` | 0 | 8/64/2 | 512 | 48 / [evbreak][evbreak] |
| H6粗-H | 0.05 | `PR_p0078_q8_r32_K2` | 78 | 8/32/2 | 256 | 64 / [h6csv][h6csv] |
| H6厳-R | 0.0005 | `H6_p000_q8_r64_K2` | 0 | 8/64/2 | 512 | 48 / [evbreak][evbreak] |
| H6厳-H | 0.0005 | `PR_p0078_q8_r32_K2` | 78 | 8/32/2 | 256 | 64 / [h6csv][h6csv] |

費用会計は元と同じ `G=N_cos*C_cos+N_sin*C_sin`。分数は保存文字列のまま、decimal欄も保存値のままで、新しい丸め・倍率・平均は作っていない。

| 略号 | B（保存exact又は保存decimal） | N_cos / N_sin | C_cos = C_sin | G exact | G SE |
|---|---|---|---|---|---|
| H2-R | 1638430308470357/1125899906842624 | 14906 / 14903 | 907/48 | 27036763/48 | 11489.641344756476 |
| H2-H | 6113487734987227/4503599627370496 | 12920 / 12920 | 775/32 | 1251625/2 | 20808.97393554041 |
| H4粗-R | 1.6620900065204434 | 19369 / 19370 | 41639/128 | 1613053221/128 | 44580.37483807345 |
| H4粗-H | 1.3837493166001436 | 14384 / 13570 | 10291/24 | 47945769/4 | 37995.654220550095 |
| H4厳-R | 1.662369839186228 | 193753833 / 193753845 | 124901/384 | 8066682748313/64 | 486679288.01218957 |
| H4厳-H | 1.5871499573638734 | 183683129 / 177738535 | 17797/48 | 134004611546 | 1163065295.2161603 |
| H6粗-R | 1661454695514047/1125899906842624 | 15268 / 15268 | 149519/48 | 570714023/6 | 355807.7258949337 |
| H6粗-H | 5747355790514877/4503599627370496 | 11465 / 11493 | 299249/64 | 3435079271/32 | 138534.1239994281 |
| H6厳-R | 1661454695514047/1125899906842624 | 152683318 / 152680785 | 149519/48 | 45657735316457/48 | 3558125067.408183 |
| H6厳-H | 5747355790514877/4503599627370496 | 177727546 / 247950732 | 299249/64 | 63691899506611/32 | 2568645672.459057 |

- **H2粗**：純RTEはHybridよりB・Nが大きいが、低いCと合わせた保存Gは低い。H2の低Gを「Bが小さいから」とは説明できない。Kも2対4で異なる。
- **H4粗**：純RTEのCは低いが、Hybridの低いB・Nを打ち消せず、Hybrid低Gとなる例外である。q/r/K・sも異なる。
- **H4厳**：同じq8/r8/K4でも、純RTEはB・Nが大きくCが低く、保存Gは純RTEが低い。prefixだけの回路費用効果を測った対照ではない。
- **H6厳**：純RTEのBはHybridより大きいのに、N_cos・N_sin・Cのすべてが小さい。下記bias/headroomが、Bだけでは説明できないNの差を会計上説明する。
- **H6粗**：純RTEはNの両軸ともHybridより大きいが、Cが低くGは低い。「H6は常にNとCの両方が小さい」とはしない。

### bias・数値allowance・headroom

次は代表候補の保存signed biasとu。H6の粗/厳は同じcandidateの同じ信号・uを使うため重複表示しない。H2の細かい精度では保存minがq2/r2/K4へ変わるので、この粗精度の信号値を転用しない。

| 略号 | signed bias_cos | signed bias_sin | u_cos | u_sin |
|---|---|---|---|---|
| H2-R | -2559240299578501/36893488147419103232 | -38697810722130223/590295810358705651712 | 338859119/2361183241434822606848 | 338859119/2361183241434822606848 |
| H2-H | 1626568371/9007199254740992 | 713768359/4503599627370496 | 1471/9007199254740992 | 735/4503599627370496 |
| H4粗-R | 1024019526173458153/9671406556917033397649408 | -55185856738246641/75557863725914323419136 | 318052407/9671406556917033397649408 | 346614523/9671406556917033397649408 |
| H4粗-H | 11315390587805419645/9444732965739290427392 | 445273709885561351/2361183241434822606848 | 404353/9444732965739290427392 | 855751/18889465931478580854784 |
| H4厳-R | -33898117/18446744073709551616 | 1889810199/147573952589676412928 | 11641/295147905179352825856 | 6263/147573952589676412928 |
| H4厳-H | 518913863963502451/75557863725914323419136 | 5405705078094428739/4835703278458516698824704 | 51216207/1208925819614629174706176 | 437086691/9671406556917033397649408 |
| H6厳-R | 1117751031/144115188075855872 | -1390296821/288230376151711744 | 5626317/144115188075855872 | 11252627/288230376151711744 |
| H6厳-H | 1234339285/17592186044416 | 1998945861/17592186044416 | 533/17592186044416 | 533/17592186044416 |

既存 [shot実装][shotcode] は `h=epsilon_axis−|bias|−u−ρ`、`epsilon_axis=元binary64 epsilon_sig/sqrt(2)`、`N=ceil(A_upper/h^2)`、`(u+ρ)/h<=.01` を使用する。`A_upper` は保存Bとlog包含から得る `2*B_up^2*ln(2/.025)` の上端であり、Bを信号へ一回適用することとは別である。uは数値allowance、ρは実装残差であり、今回ρを測定したわけではない。

H6厳精度の保存 `axes.*.rho_zero.headroom_exact` / `u_over_h_exact` を引用する（[純RTE会計][evjson]、[Hybrid会計][h6json]）。他代表行のuは上表、hは同じ候補・精度の保存会計を参照し、本調査で再評価しない。

| 略号 | h_cos exact | h_sin exact | u_cos/h_cos exact | u_sin/h_sin exact |
|---|---|---|---|---|
| H6厳-R | 6521765120365847/18446744073709551616 | 6521819213501719/18446744073709551616 | 720168576/6521765120365847 | 720168128/6521819213501719 |
| H6厳-H | 5227609803667223/18446744073709551616 | 4425861698631447/18446744073709551616 | 558891008/5227609803667223 | 558891008/4425861698631447 |

**H6厳精度では、純RTEの小さいbiasが広いhに対応し、Bが大きくても必要Nは小さくなることを、保存会計が直接示している。** 両者のu/hは元guardを満たす保存行であり、数値allowanceの失格が順位を作った例ではない。Hybridのbiasがどの交換子・保持項・tail近似に何割由来するかは未分離である。上記は会計上の説明であり、異なるlawを仮想交換した新しい因果寄与率ではない。

## 3. Event数：既存の最小点をどこまで説明できるか

### 純RTE（q8/K2、ε=.0005）

次表は [10/26報告のB/N/C表][evreport] の既存表示を引用。全5精度・K4を含む元値は [event_cost_trends.csv][evtrends]。比較用に新しいGを計算していない。

| 分子 | s | cohort / n | B | 保存N_total | C_cos=C_sin | G | G SE |
|---|---|---|---|---|---|---|---|
| H4 | 64 | confirmation / 384 | 1.66209 | 388296014 | 325.3047 | 1.263145e+11 | 4.468464e+08 |
| H4 | 128 | confirmation / 48 | 1.290387 | 233557853 | 631.4167 | 1.474723e+11 | 1.073676e+09 |
| H4 | 256 | exploration / 8 | 1.136081 | 180992303 | 1268.75 | 2.29634e+11 | 3.17433e+09 |
| H6 | 64 | confirmation / 48 | 20.6408 | 60852202283 | 424.1042 | 2.580767e+13 | 3.140839e+11 |
| H6 | 128 | exploration / 8 | 4.692424 | 3094651181 | 793.125 | 2.454445e+12 | 5.557905e+10 |
| H6 | 256 | exploration / 8 | 2.175295 | 663720708 | 1580.625 | 1.049094e+12 | 1.58689e+10 |
| H6 | 512 | confirmation / 48 | 1.475668 | 305364103 | 3114.979 | 9.512028e+11 | 3.558125e+09 |
| H6 | 1024 | exploration / 8 | 1.214851 | 206953734 | 6239.5 | 1.291288e+12 | 4.538681e+09 |

H6はs64→512でshot節約が単回C増加を上回り、s1024ではB・Nがさらに下がってもC増加が勝つという既存の費用分解がある。H4ではs128からC増加が上回る。H4の上表はK2系列であり、厳精度の確認集合minは旧s64/K4である（§2）。H6のs128/256/1024、H4のs256は探索価格のまま。独立確認は隣接全点まで完了したわけではない。

law/biasも不変ではない。[保存law実装][lawcode] ではtail係数からlambda_Rを作り、tau=lambda_R*(T/q)/r、偶数order確率・rotation angle・正規化をrごとに変更する。同じq8でrを増やす系列でも、単に同一eventを長く並べた対照ではない。保存biasはH6 K2でcos正/sin負のまま絶対値が小さくなり、H4 K2でも同じ符号で小さくなる（元CSVのbias_*_exact列）。bias低下とB低下を独立に交換した因果分解は未実施。

既存図：[H6のB/N/C][h6fig]、[H6 G拡大図][h6gfig]、[H4 G][h4fig]。図の再生成はしていない。s512/s64は有限grid・確認集合の保存最小であり、連続域の単峰性・大域最適性の証明ではない。

### H6 Hybrid prefix78/q8/K2（ε=.0005）

r8は旧確認を再利用、r16/32/64は10/27の独立確認。B/N/C/G/SEは [CANDIDATE_RESULTS.csv][h6csv] と [旧selected_cost_breakdown][evbreak] の原文である。

| r (s) | 確認n | B exact | N_cos / N_sin | C_cos = C_sin | G exact | G SE |
|---|---|---|---|---|---|---|
| 8 (64) | 48 | 2961839000617845/1125899906842624 | 755225928 / 1053563646 | 74945/24 | 22593289103905/4 | 6378678743.383964 |
| 16 (128) | 64 | 3664256682156955/2251799813685248 | 288969543 / 403143584 | 233501/64 | 161609107267627/64 | 2700894285.578722 |
| 32 (256) | 64 | 5747355790514877/4503599627370496 | 177727546 / 247950732 | 299249/64 | 63691899506611/32 | 2568645672.459057 |
| 64 (512) | 64 | 1271969024947877/1125899906842624 | 139280852 / 194313258 | 432931/64 | 72211615818205/32 | 2769615401.911481 |

r16→32では保存Nの減少がC増加を上回り、r64ではNがさらに減ってもGが上がる。全5精度でr32が探索nomineeと新集合確認後minの双方になり、追加後も純RTE低Gだった。保存H/Rは約1.129/1.152/1.183/1.489/2.092（[比較CSV][h6compare]・[報告][h6report]）。純RTE側の成功cohortを再取得していない。

同じr系列でもlawは変わる。[sampling_design][sampling] の保存nonzero event massはr16/32/64で `0.001899427875450832` / `0.0004761372297216182` / `0.00011911463855007742`。一方、biasは元CSVのexact値で同符号・近い値に留まり、r拡張で純RTE並みに小さくなってはいない。**固定q8の保持S2・split由来のbiasが残るという説明には整合するが、その由来を個別に実証した対照ではない。** event数、角度・order law、B、compile入力が同時に変わっている。

新探索n128、確認は適格4候補すべてn64で、確認開始前に固定しtop-upなし。p684/q1/r16/K4はBが小さくても.05以外はbias不適格でN/G=null。小Bだけで勝てるという説明への別の反例となる。p684では希少event観測の名目90%設計要求346本が探索上限128を超え、未観測寄与を0としていない。[price_statistics][prices] はpaired共分散を保持し母平均上界はnull、[verification][h6verify] も実ρ=null・population tail未保証のままである。

## 4. 原因候補と既存の対照：どこまで確認済みか

「観測・技術的に確認済み」は固定データ・SOURCEが直接支持する内容、「整合的・因果未分離」は説明と観測の対応まで、「未検証」は必要な対照が未確認という意味で使う。

| 原因候補 | 観測・技術的に確認済み | 説明として整合的だが因果未分離／未検証 |
|---|---|---|
| 1. 決定論的prefixの追加費用 | [Pauli adapter][adapter] は各q stepで保持forward half→r tail event→保持reverse halfを生成する。L0には保持演算がない。H4厳の同じq/r/KでC差も保存されている | prefix変更はtail Hamiltonian、component確率、tau、B、bias、回路・compiler応答も変える。「保持だけがG差を生む」という単一因果説明は未検証。central系列の保持-only価格はD1から転用できない |
| 2. ランダム化tailの違い | 保存law/SOURCEではL0は全非identity項、Hybridは残りのtail。signed Pauli productsと最後のrotationを実装し、constant/relative/global phaseを保持する | どの項の重み・support・可換性・順序がどれだけN/C/G差を作るかは未検証。同一tail/lawを固定した回路診断と、物理的候補の比較を分ける必要がある |
| 3. event数・生成law | §3のB/N/C/Gトレードオフとrごとの確率変化は直接保存されている | H6 s512とHybrid r32は当該gridの低G点。rだけの費用効果、q/r同じ積の同値性、未探索域の最適性は未検証 |
| 4. normalization・bias→shot | 保存shot十分式・§2 headroomが、H6厳の「Bは大きいのにNは小さい」を説明する。H2/H4ではNとCの競合が異なる | Hybrid biasの保持S2/tail truncation/分割相互作用別の帰属、各因子の寄与率は未分離。uと実ρも別物である |
| 5. control・compile | 後述D1/D2で非加法性とcontrol方式差を実測。現代表比較はすべてcentralで、旧genericとの差ではない | 当該H2/H4/H6最良候補の保持/event別central価格、compiler pass別因果寄与は未検証。D2の改善量を最新RTE/Hybrid差へ代入できない |
| 6. 探索・確認範囲 | H2追加確認、H6両familyのevent拡張による保存比較の変化は確認済み。探索と確認のn/選抜を区別できる | 全prefix/q/r/Kの対称な探索、選抜後同時coverage、希少費用tailを含む母平均winnerは未保証。現在の14/15や全5精度の観測を一般的方式優位へ拡張しない |

### 同じq/r/Kの対照は既にある

H4厳のL0対L1（q8/r8/K4、§2）に加え、[H6旧fairness][fairness] は確認各n48のL684対L0/q1/r8/K4（.05）、L78対L0/q8/r8/K2（他4精度）を保存している。後者の最厳精度はHybrid G `5.64832e+12`、matched RTE G `2.58077e+13` で、旧s64範囲ではHybridが低い。H6追加event後のfamily比較へこの旧順位を持ち越さない。

これらはq/r/Kを固定した**prefix変更全体**の対照であり、保持回路部分だけを操作した比較ではない。新r32/64のL0側には保存探索点があっても、10/27の全4Hybrid候補にmatched RTEの独立cohortが揃ったという意味ではない。

### D1/D2の実証範囲を取り違えない

[10/24構造調査][structure] はSOURCE・descriptorからprefix27がZ/ZZのみであること、Pauli/DFの保持・event・controlの違いを同定した。その時点の「成分別未測定」には後続証拠がある。

- **後続D1は9/9測定完了**：[SOURCE基準B][d1]、[interaction CSV][d1csv]。H4/q8/r4/K4、Pauli prefix27対DF prefix1＋one-body、cosine/PR index0。Pauli genericのfull/retained/event/empty RZは9652/6524/3110/0、保存非加法性Δ=+18。DF nativeは23021/13483/5773/0、Δ=+3765。保持/eventは科学的に同等な進化候補ではなく診断ablationで、全cohortや母平均に外挿しない。初回D1のfingerprint FAIL/STOPと原因UNKNOWNは消えていない。
- **D2は同一保存bodyのcontrol対照**：[実行報告][d2]、[比較CSV][d2csv]。同じH4条件でPauli PFのgeneric→central RZは255500→15028、PR index0は9652→1145。control構成＋固定compiler応答の差は実測済み。個々のpass寄与、表現単独の因果効果、最新純RTE対Hybridの差ではない。

DFはこの補助対照だけtol-only1e-8/rank10、Pauliは184項。DF basis単独やD3 reuse-off対照は未分離。genericのisolated価格をcentralの保持費用として差し引かない。§4の「未検証」はこれら既存D1/D2を無視して全費用分解を一から要求する意味ではない。

## 5. 追加検証は研究上必要か

**現在の有限候補・仮想ρ=0・標本平均Gの観測と会計上の説明を報告するだけなら、追加科学計算は必須ではない。** 一意の機構を主要claimにする場合に限り、次の不足が判断材料になる。以下は提案であり未実行・未採択である。

| 未解決事項・優先度 | 不明点／既存証拠で足りない理由 | 資料調査で解決できる範囲 | 必要なら別承認する最小検証の固定・比較軸 |
|---|---|---|---|
| H6のbiasの由来・優先1（機構claimを望む場合） | rを増やしてもHybrid biasが残る。保持S2、tail finite truncation、分割相互作用が交絡 | 保存target/law・SOURCEから演算順、数値精度、既にあるq/r/K対照をさらに追跡できる。成分別biasの新評価は資料調査ではない | 同じ保存H/state/reference・T・prefix・compiler/精度を維持し、q依存とtail近似依存を事前に分けた限定対照。q変更時のs/law/B変化も記録する。「prefixだけの効果」と呼ばない。新signal/bias/N/G評価には科学実行許可が必要 |
| central保持部とevent部のcompiled費用・優先2 | 最新候補ではfull Cしかなく、generic D1には非加法性がある | 既存保存descriptor・histogramの追加読取で構造を説明できる。既存D1/D2の再実行は不要 | 現在の代表候補の同一保存event列・角度・位相・q/r/K・compilerを固定したfull/retained-only/event-only/empty。既存fullを再利用できるか先に照合。新build/compileや差の数値再解析は別許可。ablationを有効な性能候補にしない |
| tailのどの項が効くか・優先3 | prefixとtail係数・support・probability・B・biasが同時に変わる | 保存law/descriptorの読取で相違箇所は特定可能。新しい寄与率や配分は得られない | 保持/尾部構造の診断目的を先に限定する。tail/law固定の回路診断と、物理的に整合したprefix変更比較を混同しない。追加候補探索を既定にしない |
| compiler passの帰属・低優先 | D2は合成効果、既存pass trace未取得。完全配賦がなくても保存G比較は記述可能 | 既存histogramとSOURCEの確認まで | 特定のcompiler機構をclaimする場合だけ、同一入力・設定で対象passを変える対照等を別承認。全compiler監査を要求しない |
| 母平均・実精度・一般化（強いclaim時のみ） | 希少event tail、実ρ、選抜効果、未探索域が残る。5精度は価格標本を共有 | 保存verificationの保証範囲は読取だけで確定できる | claimを強める場合に限り、必要なtail/ρ/確認設計を別途判断。大量sampling、全候補探索、全law証明へ自動移行しない |

H2は追加確認によって既存s4候補の比較が明確になり、大s拡張は改善しなかった。H4は.05だけHybrid低Gで、厳精度ではC節約がN増加を上回る。H6は大sでB/Nが改善し、Hybrid側の大event候補も追加してなお純RTE低Gだった。ただしH6厳ではbias/headroom、粗では主にC差と、同じ分子内でも説明の比重が異なる。これが単一原因・一般的サイズ則の採択を避ける材料となる。[H2][h2report]、[H4/H6][evreport]、[H6 Hybrid][h6report]。

## 6. 最終整理・保全・停止

1. **実証されていること**：保存G差は候補固有のNとCの競合に対応する。event数増加にはB/N減少とC増加のトレードオフがあり、H6厳ではbias/headroomが「大Bなのに小N」を説明する。D1の部品費用・非加法性、D2のcontrol差も既存の実測証拠である。
2. **未分離なこと**：prefix保持、tail、law、finite approximation、compilerの寄与を一意に配分していない。Hybrid biasの由来、最新central系列の保持部価格、母平均・全域最適性・実ρは未実証である。
3. **判断材料**：限定された条件付き観測の報告には新計算不要。機構を研究の主要claimにしたい場合に、まずH6 biasの帰属、次にcentral部品対照の必要性をユーザー/GPTが判断する。完全な成分配賦やcompiler監査は既定工程にしない。

作業は独立branch `codex/pauli-pure-rte-cost-mechanism-inventory-20261101`。新しい報告書と最小の索引リンクだけを変更した。元研究checkout・固定branch、SOURCE/input/raw/result/grant/ledger/marker/STOPは変更せず、既存のFAIL/null/未保証状態を保持する。コードは静的読取のみで、既存の文書専用 `one`・`source_rows`・`check_links` を再利用した。科学runner・law生成・shot/統計関数は呼び出していない。全検査は主agent自身の技術照合である。

公開前に新規文書の内部絶対path・秘密情報、引用文字列と固定Git参照、変更が上記2ファイルだけであることを確認し、公開後にGitHub本文とbranch SHAをlocal commitへ照合する。これは技術文書の検査であり科学statusの更新ではない。

**公開・照合後mandatory STOP。次の担当はユーザー/GPT：追加の原因検証が研究上必要か判断する。Codexの次の実行作業なし。新科学計算、詳細科学レビュー、論文中心claim採択へ自動移行しない。**

[h6report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/REPORT.md
[h6csv]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/CANDIDATE_RESULTS.csv
[h6compare]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/COMPARISON_RESULTS.csv
[sampling]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/sampling_design.json
[prices]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/price_statistics.json
[h6source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/SOURCE.json
[h6verify]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/verification.json
[h6json]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/candidate_results.json
[evreport]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/docs/pauli_rte_event_extension_20261026.md
[evtrends]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/event_cost_trends.csv
[evbreak]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/selected_cost_breakdown.csv
[evcurrent]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/CURRENT_SOURCE.json
[evsource]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/SOURCE_e755ecf581600192fd2745d7066e410cebd933d3.json
[evjson]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/candidate_results.json
[h2report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/docs/h2_pauli_rte_validation_20261026.md
[h2csv]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/candidate_precision_table.csv
[h2source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/SOURCE.json
[synthesis]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/research/pauli_jw_resource_evaluation_summary.md
[latest15]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/research/data/pauli_jw_summary/latest15.csv
[oldcost]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/artifacts/pauli_hybrid_rte_comparison/2026-10-25/cost_breakdown.csv
[structure]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/non_df_h4_cost_reversal_structure_20261024.md
[d1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/non_df_h4_d1_source_execution_20261024.md
[d1csv]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/artifacts/non_df_h4_d1_source_execution/2026-10-24/interaction_table.csv
[d2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/non_df_h4_d2_execution_20261024.md
[d2csv]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/artifacts/non_df_h4_d2_execution/2026-10-24/comparison_table.csv
[fairness]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/h6_rte_fairness_20261025.md
[adapter]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/pauli_signal_adapter.py
[lawcode]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/h4_pauli_pr_search.py
[shotcode]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/h4_conditional_cng.py
[h6fig]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/figures/H6_B_N_C_vs_s.png
[h6gfig]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/figures/H6_G_vs_s_extension_zoom.png
[h4fig]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/figures/H4_G_vs_s.png
