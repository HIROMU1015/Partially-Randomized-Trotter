# DF Truncated PF：誤差相殺・安定性の既存証拠調査

2026-11-08 JST、SERVER 2 / Codex C。固定基点は`da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13`。これは保存資料の調査・照合であり、詳細GPT科学レビューではない。

**採用済みH4/H6 policyの条件付き成功保証は維持される。複数の保存順序で資格が成立する有限点の証拠はあるが、近傍hの連続的な安定性、状態摂動への頑健性、相殺を使わない全RPE費用の優位は未確定である。** H6のm14は同符号の誤差で資格を満たし、全roundが同じ相殺機構という説明には反例となる。[元REPORT][BASE_REPORT]・[機構CSV][M]・[資格][V]

今回実行したのはGit object・文書・CSV/JSONの読取、指定rawのSHA256照合、既存値の引用、静的SOURCE確認、文書作成・公開のみ。科学moduleをimportせず、新しい位相・差分・比率・感度・上界・N・総Gを計算していない。保存policyの再選択・再ランキング、simulation、compile、sampling、科学test runnerも実行していない。旧SOURCE/raw/result/grant/ledger/marker/STOPは変更しない。

## 1. 結論と証拠分類

| 論点・主張 | 分類 | 支持範囲・不足 |
|---|---|---|
| 採択したH4/H6の全18round資格・元Hのenergy推定 | CONDITIONALLY_SUPPORTED | 固定保存H、元H exact ground、理想native、fresh独立測定の条件下。今回の調査で変更しない |
| m17でTruncationとPFのsigned位相が逆符号 | SUPPORTED | 機構CSVの保存値。個別signed値そのものはdirected intervalではない |
| 同じretention/hの複数順序でround資格が成立 | CONDITIONALLY_SUPPORTED | H4 Y4の1023/1032と元順、H6 S2の2013/0213と元順。全順序や未評価点へ拡張しない |
| 近傍hでも小さい合成誤差・低Gが維持される | UNDETERMINED | 隣接dyadic hのscreeningはあるが、固定retention/orderの近傍全点にdirected資格・十分N・全18round総Gが揃わない |
| 最終のほぼ完全な相殺点だけが低コスト性の唯一の根拠 | NOT_SUPPORTED | 順序audit前の保存policyもTruncatedを勝者と記録。ただし当該policyにも相殺がある |
| 相殺を使わなくてもTruncatedがFull/PRより低G | UNDETERMINED | 相殺を抑えた共通条件のpolicy・独立比較が保存されていない |
| 全roundで逆符号の相殺が起き、切り捨ては必ずNを減らす | NOT_SUPPORTED | 前者にはH6 m14の同符号値、後者にはm17のH6 NとH4全trial数という直接の反例がある |
| 近似ground・別固有状態・有限overlap摂動にも頑健 | UNDETERMINED | 保存stateの数値residual包囲はあるが、意図的な状態摂動系列は取得できていない |
| 任意h/order/stateに頑健、screen外候補は一般に不可能 | NOT_SUPPORTED | その一般化を支える証拠がない。不可能性の証明ではない |

SUPPORTEDは保存記録から直接確認できる事項、CONDITIONALLY_SUPPORTEDは明示された科学条件内の結論、UNDETERMINEDは問いの判断不足、NOT_SUPPORTEDは肯定的主張を現状支持できないことを表す。NOT_SUPPORTEDを一律に「科学的に反証済み」と読まない。直接の反例がある場合だけ表に明記した。[M][C][R][V]

## 2. 科学条件・SOURCE・証拠の階層

Linear H4/H6、1.00 Å、STO-3G、RHF積分由来。H4は保存DF12/generation order、N=4 sector70・8 system qubits。H6は保存full XDF18/元frame順、Nα=Nβ=3 sector400・12 system qubits。元Hは保存数値行列のHermitian部分であり、微小なDF再構成差も元Hへの誤差に含む。18round m=0..17、δ0=.02、energy精度1.5936e-4 Ha、方式1回のfailure上限.05、位相予算.4 rad、元H exact-ground oracle・状態準備費除外、Qiskit1.3.0/compiler A、rz/sx/x/cxを固定する。PRは既存Closed P3のweighted estimatorであり、無重みHadamard測定へ置換しない。[BASE_REPORT][DISCOVERY]

| 証拠 | 読み取れるもの | その証拠だけでは言えないもの |
|---|---|---|
| `ordering/TRUNCATION_MECHANISMS.csv` | 最終採択roundのsigned trunc/PF/combined、combined上界、rho/β/N | 非採択h・順序のsigned成分、摂動感度 |
| `ordering/CANDIDATES.csv` | 固定drop集合の名前、数値norm/ground差/overlap/半径、screen・資格round・採択round | 全候補の全18round native資格・最良総G |
| `ordering/ROUNDS.csv`、`ERROR_BUDGET.csv`、`FINAL.json` | 凍結policyの実価格、N/α、directed数値・SOURCE/native会計、alias、decoder | 隣接パラメータへの保証の移送 |
| `ordering/QUALIFIED_COMPLETE.json` | auditの48点の実module価格とround別directed資格 | その各点を組み替えた新policyの全G・branch保証 |
| raw `Hn/scan/`、`ordering/pilot/` | 数値dominant bias/overlap、`viable_rounds`やproxy、保存module価格 | proxyを実N・実総G・directed上界と扱うこと |
| raw `DETERMINISTIC_FINAL.json`、`RESULTS.json`、`TRUNCATION_MECHANISMS.csv` | 順序audit前の別の凍結済み全18round policyと保存結論 | 相殺を完全に除いた反実仮想 |

rawは固定[RAW_INDEX][RAW]のmember bytes/SHA256と照合し、元worktreeの展開済み同一fileを読んだ。本文の`raw:`以降は同indexに収録されたtar member名であり、repository直下に個別公開されているという意味ではない。復元は固定[REPRODUCE][REPRO]、9分割archiveは[ARCHIVE_PARTS][PARTS]を参照。新しいrawやSOURCE manifestは作らない。

科学実行SOURCEは[SOURCE.json][SOURCE]の`stage_loaded_sources`を正本とする。初期scanは`10c2c028c615eb770a463d5b841233c40e587212`、corrected qualificationは`c8aa6291dae026e251637813225b6286fa314ee8`、order pilotは`160fa8343c8cddc92b19cd6c7cfc95d23cf6376b`、order qualificationは`ac3c28de6d148422bc4ab193cb429073f0ea4601`、最終表は`ba4cfc48015d2d09c0f2c6fcd3aa17897ec5a65a`。元commitは同研究のSOURCE_HISTORY.bundleで保全されている。初期rawの`source`がprocess中に観測したHEADである場合を、load時SOURCEと混同しない。

## 3. 相殺の構造と採択m17

保存SOURCEの定義は、元H stateについて`zkeep=<ψ0|exp(-it Hkeep)|ψ0>`とPF信号`zpf`を用い、`φT=Arg(zkeep exp(+iE0t))`、`φPF=Arg(zpf conj(zkeep))`、`φC=Arg(zpf exp(+iE0t))`とする。加法関係はmodulo 2πであり、wrapped endpointの小ささを非共鳴なbiasの小ささと同一視しない。今回は既存CSVを引用し、これらのArgや複素信号を再計算していない。[表生成SOURCE][EXPORT]

| m17の最終採択 | H4 | H6 |
|---|---:|---:|
| retention | coeff_L1_2、204係数除去、identity保持 | prefix13_I0、18中13 DF＋one-body保持、末尾identityも除去 |
| PF / h / 先頭順序 | Y4 / .16 / 1023 | S2 / .08 / 2013 |
| φT：保存signed点値 [rad] | −0.021852026035331144 | +0.27980999192240813 |
| φPF：同 [rad] | +0.021344614090985415 | −0.31381601998768394 |
| φC：同 [rad] | −0.0005074119443457366 | −0.034006028065275785 |
| `total_truncation_PF_phase_upper` [rad] | 0.0005288528667727131 | 0.03458846713437962 |
| 最終 `beta_stat_lower` [rad] | 0.3994689606868961 | 0.36537159080081605 |
| 最終 `rho_lower` | 0.9999765036163492 | 0.9993883877747775 |
| 十分N_X / N_Y | 130 / 128 | 157 / 155 |
| round総RZ / 総CX | 178754649452 / 30980253258 | 585711692747 / 101254819536 |

出典：[M][R][E][Q]。`total_truncation_PF_phase_upper`はcombined signalをdirectedに囲む欄であり、signed成分の差から作った新上界ではない。SOURCE/native/角度/supplementを追加で会計した後の統計余裕が最終βである。この一つの上界欄を、全native誤差の独立上界や個別φT/φPFの区間と読み替えない。[資格SOURCE][DET]

H6は大きい逆符号成分の相殺を含むが、それを理由に固定点のdirected保証を無効とはしない。一方、βは既に統計へ割り当てられた量であり、同じNのまま自由にhの変動へ転用できる「パラメータ許容幅」ではない。

## 4. h・PF formula・orderingの証拠

### 4.1 固定retention/orderで保存された隣接h

次表は**m17の非共鳴screening proxy**の転記。directed bound、signed φC、実trial数、実総Gではない。φTは固定Hkeep/state/tならhで変わらないが、PF側と信号モードはhで変わる。

| 分子・retention・formula・順序 | h | 保存 `phase_nonresonant_proxy` [rad] | m17の状態 |
|---|---:|---:|---|
| H4 coeff_L1_2 / Y4 / 1023 | .08 | 0.02051206986138786 | proxy行あり、当該点のdirected資格・全Gは未取得 |
| 同 | .16 | 0.0005074129879347761 | proxyに加えて採択点のdirected資格あり |
| 同 | .32 | 0.3136210961082743 | proxy行あり、当該点のdirected資格・全Gは未取得 |
| H6 prefix13_I0 / S2 / 2013 | .04 | 0.2013678470994114 | proxy行あり、当該点のdirected資格・全Gは未取得 |
| 同 | .08 | 0.03400606446455154 | proxyに加えて採択点のdirected資格あり |
| 同 | .16 | 未保存 | `rounds`はm15のみ。m17は十分screenの行に残っていない |

raw: `ordering/pilot/a3_full_Y4_1023_{0.08,0.16,0.32}.json`、`ordering/pilot/a5_full_S2_2013_{0.04,0.08,0.16}.json`。[RAW][ORDER]

H4の非常に小さいcombined値はh=.16に特徴的で、隣接dyadic hでは保存proxyが異なる。H6もh=.08と半分/倍でscreening状況が異なる。従って**有限のcoarse点での非孤立なscreen通過は確認できるが、採択h近傍の小摂動で相殺と低コスト性が続くとは判定できない**。h=.16のH6にm17 proxy行がないことを、任意の順序・retention・formulaで不可能という結論に使わない。

保存探索menuはh=[.0025,.005,.01,.02,.04,.08,.16,.32,.64,1.28,2.56]。単一retentionでhだけを変えた全点の最終signed分解・directed資格・最適N/Gのresponse surfaceは保存されていない。選択roundから異なるm、順序、retentionをまたいで「h感度」を作ることもしない。[BASE_REPORT]

### 4.2 同じhの複数順序でのdirected資格

元順はone-body index0から始まるgeneration order。1023等は先頭4因子だけを表し、その後の因子は元順のまま。次の数値は各保存点の資格欄から引用する。

| 分子・retention・h・m17 | 順序 | combined上界 [rad] | 最終β下限 [rad] | 証拠 |
|---|---|---:|---:|---|
| H4 coeff_L1_2 / .16 | 元順 Y4 | 0.008812874472572002 | 0.3911852854496319 | raw `DETERMINISTIC_FINAL.json`の採択点、supplement後 |
| 同 | Y4 1023 | 0.0005288528667727131 | 0.3994689606868961 | 最終採択、Q |
| 同 | Y4 1032 | 0.0005037910979541337 | 0.39949402245571464 | Qの非採択round点 |
| H6 prefix13_I0 / .08 | 元順 S2 | 0.08201202535748046 | 0.3179480325778243 | raw `DETERMINISTIC_FINAL.json`の採択点、supplement後 |
| 同 | S2 2013 | 0.03458846713437962 | 0.36537159080081605 | 最終採択、Q |
| 同 | S2 0213 | 0.0536335096032039 | 0.3463265483321159 | Qの非採択round点 |

全て上表の保存資格ではclosed/aliasがtrue。最終採択policyの18round資格と、非採択点が一つのroundでclosedであることは区別する。1032の上界だけを見て新たな全G順位を付けない。[Q][V][RAW]

これは「特定の順序一つでしか資格がない」という見方を支えない。同時に、全順列・任意変更への頑健性も証明しない。元順の保存signed分解はH4がφPF=+0.013059465734029935、φC=−0.008792560301301171、H6がφPF=−0.3612395782123745、φC=−0.0814295862899664であり、元順でも相殺は残る。raw: `TRUNCATION_MECHANISMS.csv`。

### 4.3 PF formulaを替えた保存診断

以下は元順scanの`dominant_energy_bias` [Ha]と保存`viable_rounds`の抜粋。これは元Hに対するFloquet dominant energy biasであり、PF単独のsigned phaseでもdirected保証でもない。今回は時間との積を新しく計算していない。

| retention | formula / h | 保存dominant energy bias [Ha] | m17がviable_roundsに含まれるか |
|---|---|---:|---|
| H4 coeff_L1_2 | S2 / .08 | +9.389890135593797e-05 | 含まれる |
| 同 | S2 / .16 | +0.0003508186217290803 | 含まれない（m3..15） |
| 同 | Y4 / .08 | +8.023605390511937e-06 | 含まれる |
| 同 | Y4 / .16 | +3.3540953368883963e-06 | 含まれる |
| H6 prefix13_I0 | S2 / .08 | +3.106292245060433e-05 | 含まれる |
| 同 | S2 / .16 | +0.0004448419617454434 | 含まれない（m3..15） |
| 同 | Y4 / .08 | −0.00010722472691397121 | 含まれる |
| 同 | Y4 / .16 | −0.00011447910385120656 | 含まれる |

raw: `H4/scan/coeff_L1_2_{S2,Y4}_{0.08,0.16}.json`、`H6/scan/prefix13_I0_{S2,Y4}_{0.08,0.16}.json`。formulaでbiasの符号や大きさ・screen結果が違うことは読み取れる。screenを通った別formulaが最終native条件・全Gで同等に良いとは判定できない。[RAW][SCAN]

## 5. retention・小係数cutoffの依存

切り捨ては固定DF frameのprefix I0/I1、またはexact係数のabs順・nested累積L1 budget。cutoff名の倍率は指定精度に対する探索budgetであり、除去係数が全て同じ閾値という意味ではない。identity保持のI1と丸ごと除くI0を同一視しない。[C][DISCOVERY][SCAN]

| 分子 / 候補 | 除去係数 | 保存norm(Hdrop) [Ha] | 保存ground差 [Ha] | Hkeep最小信号半径 | Cに保存された資格round |
|---|---:|---:|---:|---:|---|
| H4 / coeff_L1_1 | 186 | 8.505524550134638e-05 | -6.742175465479505e-06 | 0.9999999987104714 | PILOT_ONLY |
| H4 / coeff_L1_2 | 204 | 0.00021804159964791824 | 8.335886139576587e-06 | 0.9999999974033474 | 14,15,16,17 |
| H4 / coeff_L1_4 | 213 | 0.000464152216874845 | -1.9790919035589383e-05 | 0.9999999523784733 | 0,1,2,3,4,5,6,7,8,9,10,11,12,13 |
| H4 / prefix7_I0 | 185 | 0.00026334366436069176 | -3.696617857285034e-05 | 0.9999999974033467 | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 |
| H4 / prefix7_I1 | 180 | 0.0002180415996479139 | 8.335886140020676e-06 | 0.9999999974033467 | 15,16,17 |
| H6 / coeff_L1_8 | 635 | 0.0005248371749680587 | -2.7977582603977424e-05 | 0.9999998958802955 | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 |
| H6 / prefix12_I0 | 474 | 0.0026083682508893265 | -0.0006456696345784607 | 0.9999991486818659 | OUTSIDE_SUFFICIENT_PILOT_SCREEN |
| H6 / prefix13_I0 | 395 | 0.0004632769171766353 | -0.00010673904141356161 | 0.999999992170293 | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17 |
| H6 / prefix14_I0 | 316 | 0.00010247385484885897 | -2.4804427755764635e-05 | 0.9999999956976283 | PILOT_ONLY |
| H6 / prefix15_I0 | 237 | 1.4219629467312374e-06 | -6.547656585809136e-07 | 0.9999999999999075 | 17 |

上表はCの**保存数値観測**であり、norm/ground差/半径を新たに囲んだものではない。資格round欄はCに記録されたround別statusの引用で、全18roundを単一Hkeepで実行した資格ではない。非採択raw qualificationのclosedフラグと、最終policyのsupplement・α/N・branchを含む完成した資格も区別する。

H4のcutoff倍率1→2→4でground差は負・正・負となり、切り捨て量だけからsigned biasを単調に予測する根拠はない。prefix7_I0/I1もscalar保持が異なり、近い数値norm/overlapだけから同一演算子とは認定しない。H6のprefix12はscreen外、prefix14はpilotのみ、prefix15はm17の資格記録があるが最終採択ではない。いずれも全方式に関する不可能性・最小費用の証明ではない。

H4 coeff_L1_2のHkeep dominant overlapは0.9999999983502053、H6 prefix13_I0は0.9999999946360796。これらは元H groundをHkeepの固有モードに展開した観測であり、近似初期状態一般についてのoverlap許容保証ではない。[C] 非採択retentionについては、PFを含むsigned成分が採択点と同じ形式では揃っておらず、同一PF/hでのcutoff感度は追加検証事項となる。

## 6. 全18round：相殺とRPE資格を分ける

以下は最終Mの既存値をそのまま転記する。βは最終の統計位相余裕。rho/N/α/価格はR、詳細diskとaliasはE、18roundのdecoder/十分条件はVとFINALにある。

### H4：最終採択各roundの位相成分（X/Y共通）

| m | retention / formula / h | φT [rad] | φPF [rad] | φC [rad] | combined上界 [rad] | β下限 [rad] |
|---:|---|---:|---:|---:|---:|---:|
| 0 | prefix7_I0 / S2 / 0.02 | 7.392919788229848e-07 | -1.0694001491292763e-07 | 6.323519639101088e-07 | 6.325888910722115e-07 | 0.39999936740480124 |
| 1 | prefix7_I0 / S2 / 0.04 | 1.4785839741902112e-06 | -8.556308473738414e-07 | 6.229531268084608e-07 | 6.232024636088268e-07 | 0.39999937678688186 |
| 2 | prefix7_I0 / S2 / 0.08 | 2.9571680803735145e-06 | -6.848586339802609e-06 | -3.891418259403737e-06 | 3.891692415497719e-06 | 0.39999610828823606 |
| 3 | prefix7_I0 / S2 / 0.16 | 5.914337214304483e-06 | -5.490147895018948e-05 | -4.898714173587632e-05 | 4.898746553063437e-05 | 0.3999510124977333 |
| 4 | prefix7_I0 / S2 / 0.32 | 1.1828682770294527e-05 | -0.00044276036183034314 | -0.000430931679060046 | 0.0004309321021323395 | 0.39956906782635615 |
| 5 | prefix7_I0 / S2 / 0.64 | 2.3657429623048577e-05 | -0.003648097855519873 | -0.0036244404258968447 | 0.0036244410475523995 | 0.39637555881137876 |
| 6 | prefix7_I0 / S2 / 1.28 | 4.731529767816431e-05 | -0.03172513871706961 | -0.03167782341939143 | 0.031677824440866156 | 0.3683221752782167 |
| 7 | prefix7_I0 / S2 / 1.28 | 9.463261653443575e-05 | -0.05967513319350695 | -0.059580500576972416 | 0.059580502604231735 | 0.34041949683401873 |
| 8 | prefix7_I0 / S2 / 1.28 | 0.0001892672899096057 | -0.1186644189627821 | -0.1184751516728725 | 0.11847515570890463 | 0.2815248431684547 |
| 9 | prefix7_I0 / S2 / 0.64 | 0.0003785347810915415 | -0.056882881034731374 | -0.05650434625363985 | 0.056504355947346985 | 0.3434956418018763 |
| 10 | prefix7_I0 / S2 / 0.64 | 0.0007570670497119942 | -0.11373389985712036 | -0.1129768328074084 | 0.11297685217772153 | 0.28702314332120415 |
| 11 | prefix7_I0 / S2 / 0.32 | 0.00151413503514835 | -0.056264388528766265 | -0.05475025349361797 | 0.05475030549684411 | 0.3452496854055628 |
| 12 | prefix7_I0 / S2 / 0.32 | 0.003028269094667126 | -0.11252777466881878 | -0.10949950557415164 | 0.10949960956286725 | 0.2905003722425273 |
| 13 | prefix7_I0 / S2 / 0.16 | 0.006056539149070938 | -0.05611221428327548 | -0.05005567513420454 | 0.05005598937722159 | 0.34994397345585565 |
| 14 | prefix7_I0 / S2 / 0.16 | 0.012113077046561473 | -0.11222491225152446 | -0.10011183520496302 | 0.10011246367428452 | 0.29988746199227295 |
| 15 | coeff_L1_2 / Y4_order0231 / 0.32 | -0.0054630066523008464 | 0.04668913814690071 | 0.04122613149459986 | 0.041229208013014836 | 0.35877034725596946 |
| 16 | coeff_L1_2 / Y4_order0231 / 0.32 | -0.010926011634201122 | 0.09337827508625728 | 0.08245226345205618 | 0.08245841648121015 | 0.3175406940540529 |
| 17 | coeff_L1_2 / Y4_order1023 / 0.16 | -0.021852026035331144 | 0.021344614090985415 | -0.0005074119443457366 | 0.0005288528667727131 | 0.3994689606868961 |

### H6：最終採択各roundの位相成分（X/Y共通）

| m | retention / formula / h | φT [rad] | φPF [rad] | φC [rad] | combined上界 [rad] | β下限 [rad] |
|---:|---|---:|---:|---:|---:|---:|
| 0 | prefix13_I0 / S2 / 0.02 | 2.1346374265501003e-06 | -1.7219680324780424e-07 | 1.9624406232876483e-06 | 1.9761917782818e-06 | 0.39999802349884167 |
| 1 | prefix13_I0 / S2 / 0.04 | 4.2692750704136734e-06 | -1.3777468127174093e-06 | 2.89152825769152e-06 | 2.906767095555282e-06 | 0.39999709262057204 |
| 2 | prefix13_I0 / S2 / 0.08 | 8.538551875087889e-06 | -1.1027483617049729e-05 | -2.488931741978549e-06 | 2.50714594553986e-06 | 0.39999749163581727 |
| 3 | prefix13_I0 / S2 / 0.16 | 1.7077117538641952e-05 | -8.839522816275179e-05 | -7.131811062407632e-05 | 7.134227555957517e-05 | 0.3999286552943937 |
| 4 | prefix13_I0 / S2 / 0.32 | 3.415434258912168e-05 | -0.0007126553004762557 | -0.0006785009578871121 | 0.0006785370243303793 | 0.3993214581219976 |
| 5 | prefix13_I0 / S2 / 0.64 | 6.830946101790785e-05 | -0.005862593430974979 | -0.005794283969957037 | 0.0057943438439925134 | 0.3942056464543397 |
| 6 | prefix13_I0 / S2 / 1.28 | 0.00013662299122782249 | -0.05052273451142835 | -0.05038611152020056 | 0.05038621945508505 | 0.34961376106657244 |
| 7 | prefix13_I0 / S2 / 1.28 | 0.0002732506178168839 | -0.09647322501622738 | -0.09619997439841048 | 0.09620018980285719 | 0.3037997712438647 |
| 8 | prefix13_I0 / S2 / 0.64 | 0.0005465047089966403 | -0.045776705508029224 | -0.04523020079903262 | 0.04523067676278656 | 0.3547692456102045 |
| 9 | prefix13_I0 / S2 / 0.64 | 0.0010930066794923297 | -0.0916821631310944 | -0.09058915645160205 | 0.09059010781072682 | 0.30940973695540236 |
| 10 | prefix13_I0 / S2 / 0.32 | 0.0021860138421381185 | -0.04531197192322963 | -0.0431259580810915 | 0.04312823834819238 | 0.356871451013378 |
| 11 | prefix13_I0 / S2 / 0.32 | 0.004372031045596073 | -0.09062056675280279 | -0.08624853570720672 | 0.08625309581184437 | 0.31374628290853707 |
| 12 | prefix13_I0 / S2 / 0.16 | 0.008744063064497584 | -0.04518554916887492 | -0.03644148610437731 | 0.03645363145273675 | 0.3635451243480821 |
| 13 | prefix13_I0 / S2 / 0.16 | 0.0174881261039864 | -0.09037078264392305 | -0.07288265653993672 | 0.07290694693282447 | 0.3270905646391786 |
| 14 | coeff_L1_8 / Y4 / 0.32 | 0.009167664164515707 | 0.03992237424558882 | 0.049090038410104515 | 0.04920390007164007 | 0.35078015946992186 |
| 15 | prefix13_I0 / S2_order2013 / 0.08 | 0.06995250142495053 | -0.07845399306018003 | -0.008501491635229471 | 0.008647071101930786 | 0.3913429476752619 |
| 16 | prefix13_I0 / S2_order2013 / 0.08 | 0.1399049955122913 | -0.15690801247487698 | -0.017003016962585746 | 0.017294195878157347 | 0.382685838814641 |
| 17 | prefix13_I0 / S2_order2013 / 0.08 | 0.27980999192240813 | -0.31381601998768394 | -0.034006028065275785 | 0.03458846713437962 | 0.36537159080081605 |

H4はm0..14がprefix7_I0/S2、m15..17がcoeff_L1_2/Y4へ切り替わる。H6はm0..13とm15..17がprefix13_I0だが、m14だけcoeff_L1_8/Y4で、φTとφPFはともに正。このroundにも十分なcombined上界・βが保存され、資格は閉じる。[M][V]

逆符号でも初期・中間roundではPF成分が主となる点があり、逆符号というだけで強い相殺や価格改善を認定しない。特にhとroundを同時に変えた列を、同一tのh感度実験と扱わない。m17の最小combined値だけから全RPEの安定性を推定しない。

VはH4/H6 Truncatedのall_18_closed、all_36_sufficient_trials、no_E0_decoder_argumentをtrueとし、mean decoder errorをそれぞれ1.935623061832814e-07 / 1.2972270227784577e-05 Haと保存する。aliasは元Hの初期phaseの包含とround間branch条件を担う別の条件であり、相殺の持続性を測る指標ではない。E/FINALでは各選択roundのalias=true。E0は資格・検証に使い、decoderには渡していない。全9方式の同時failure .05保証や実機成功率としては読まない。[V][E][BASE_REPORT]

## 7. 価格削減・相殺・Nの寄与

### ① 位相の相殺とは別に確認できるmodule価格

同じformula/h、generation order、compiler Aで保存されたmiddle moduleのnative価格を引用する。H6は切り捨て後も保持因子の相対順を維持している。

| 分子・formula/h | Full / Truncated | RZ | CX | serial depth | size |
|---|---|---:|---:|---:|---:|
| H4 Y4/.16 | Full | 68232 | 12267 | 25233 | 101163 |
| 同 | coeff_L1_2 | 42888 | 7542 | 15354 | 63534 |
| H6 S2/.08 | Full | 78096 | 13802 | 22295 | 115802 |
| 同 | prefix13_I0 | 57854 | 10094 | 16039 | 85708 |

raw: `H4/scan/full_Y4_0.16.json`、`H4/scan_repriced/coeff_L1_2_Y4_0.16.json`、`H6/scan/{full,prefix13_I0}_S2_0.08.json`の`groups.middle`。phaseやNを変えてこの価格を作ったのではなく、保存compiled回路の構造的削減である。ただしmiddle単体であり、wrapper・step数・Nを含む全資源の削減率ではない。H4の`retained_blocks=13`にはidentity-only blockも含まれる。`circuit_piece`は非identity係数がないblockのGaussianを省くため、保存block数と実行basis変換数を同一視しない。単価低下をbasis削減と対角削減へ数値分配する新しい集計はしていない。[RAW][SCAN]

### ② 相殺が統計余裕を変える経路

保存SOURCEはcombined phase上界と追加diskからβを得て、rhoと36軸αに対応する十分Nを求める。順序変更では位相に加えてmodule価格・native/source errorも変わり、全policyのαも再配分される。βが大きいことだけを、相殺による総G改善率へ変換できない。[DET][ORDER]

順序audit前の保存全policyはH4が総RZ=349981801810、trial=14743、H6が1465026077434、trial=13883で、raw `RESULTS.json`は両分子のwinnerをTruncatedと記録している。最終のnear-zero H4点やH6 order2013だけが全低コスト性を作った、という主張は支えられない。ただし元順のsigned分解にも相殺があり、これは「相殺なし」の対照ではない。[RAW][BASE_REPORT]

### ③ 全RPE資源では両効果が混在

| 保存最終policy | 総RZ | 総CX | 十分trial総数 |
|---|---:|---:|---:|
| H4 Full | 564793402143 | 101231319369 | 14225 |
| H4 Truncated | 333777223814 | 58303791508 | 14629 |
| H6 Full | 2082026044290 | 365783391210 | 14248 |
| H6 Truncated | 1222907255629 | 210671332696 | 13539 |

[保存比較表][COMPARE]の転記。H4は全trial総数がFullより多くても低RZ。H6 m17はTruncated N_X/N_Y=157/155、Full=141/141であり、このroundのNが減ったとは言えない。m17のstep数もH6 Truncated=32768、Full Y4=16384で、選択結果を「PF step数が減ったから」と一律に説明できない。H4 m17は両方式16384 stepsだがretention/order/Nが違う。[R]

結論として、module単価低下、combined biasがもたらす統計余裕、h/formula/αの選択は全Gで結合している。各寄与の因果的な百分率、相殺を弱めたcounterfactualの十分N/G、切り捨てだけを替えた全18round ablationは保存されておらず、今回は作らない。

## 8. 元H groundへの依存

Eのm17ではH4/H6の`original_state_distance_upper`が4.0518761323405345e-12 / 7.98225422925186e-11、`ground_energy_error_upper`が5.10493303860063e-11 / 4.109739005176174e-09 Ha。これは保存数値state/energyからexact元H groundへ移すためのresidual/gap・normalization会計である。制御回路をfull statevectorで照合したことも、初期状態の物理的摂動sweepとは異なる。[E][DET]

確認したraw indexとSOURCEでは元H stateを固定し、HkeepやPF固有モードとのoverlapを変える候補を診断している。入力stateそのもののθ-sweep、別固有状態、近似state preparation、overlap誤差familyについての同条件結果は取得できなかった。8人工testのGaussian/weighted mean/負時間modular correctnessも、分子state摂動の科学的頑健性の証拠ではない。[TEST]

関連D内部BCH研究はstate-specificなground期待値による係数選択で、REPORT自身が他状態・近似情報への代替を未検証とする。Closed P3・共同最適化も元H exact ground仮定を保つ。これらからTruncated PFの状態摂動頑健性を引き継がない。[DINT][CP][JOINT]

## 9. 次の検証案（提案のみ・全て未実行）

最も情報価値が高いのは**P1：H6の採択prefix13/S2/order2013で、固定tを保ったh近傍のdirected信号・位相余裕を確認すること**。次にP2で相殺依存を弱めた全Gの対照を作る。H4・別retention・状態摂動は同じ問いではないため、対象と承認範囲を分ける。新PF family・P5・新sampling law・H8以上を必要としない。

### P1：局所hと登録順序の感度

1. **目的**：coarseなhalf/double proxyと、採択hの小変動での科学的資格を区別する。
2. **固定条件**：H6 full XDF18、元H exact ground、prefix13_I0、S2、m15..17と元18round目標、compiler A。最初はorder2013を固定。
3. **変更パラメータ**：m17の保存steps=32768を中心に、整数step数の有限近傍（例：n0、n0±1、n0±16）を事前固定してh=T_m/nとする。次に元順/0213を登録対照として別軸で確認。連続hを`round(T/h)`で丸め、時間を黙って変えない。
4. **再利用**：保存INPUT/coefficients/basis certificate、DETの`block_unitary/det_qualification/qualification_supplement`、ORDERのpilot/qualification接続、元policyとQ。SCF/DF fit/元ground生成は不要。
5. **新規科学計算**：新しいhのexact時刻・binary64実装差を囲むadapter、directed signal/powering、signed成分、native角度誤差、必要なmodular compile・独立expm/Schur/X/Y確認。既存dyadic前提の資格を無検証で流用しない。最初は位相・rho/βまでとし、価格比較を行うなら必要点のN/Gと全18roundを別途完成させる。
6. **判定指標**：個別signed値とその適切なdirected包含、combined上界、rho、β、alias/branch、必要N、実module価格。現在のβをh許容幅とみなさない。新閾値・h値は今回計算しない。
7. **重複**：coarse hと3登録順序の既存点は再利用するが、小さい整数step近傍の同条件結果は未取得。旧Track A noncancellation等は異なるscope。
8. **STOP**：事前の有限gridで結果・不適格点・未解決を保存してSTOP。予算非正・branch不成立を資格なしとして残し、精度緩和や新familyへ自動展開しない。

### P2：相殺依存を弱めた低コストpolicyの対照

1. **目的**：①回路構造の削減と②相殺によるβ/Nの寄与を分け、相殺が弱くても価格上の利点が残るか調べる。
2. **固定条件**：H4/H6を別scopeとし、同じ元H/state・精度/failure/18round・compiler・native basis・prep除外を維持。基点のFull/PR policyを参照対照とする。
3. **変更パラメータ**：既存候補・formula/order/hの有限部分集合。相殺の小さい条件の選び方、または個別誤差の保守的加算で判定する規則を実行前に固定する。post-hocの有利な定義にしない。
4. **再利用**：C/M/Q、raw全scan・元Hkeep pilot、DETの`choose/rigorous_shots`と既存整数α配分、`price_piece/wrapper`。保存compiled価格と固定referenceは対応条件が同じ場合だけ再利用。
5. **新規科学計算**：非採択点のindividual/combined directed評価、必要な誤差会計・compile、十分N、凍結policy・独立18round decoder、総G。単に保存signed値を絶対値加算して既存保証を取消す操作にはしない。
6. **判定指標**：個別/combined位相、phase/rho余裕、N/α、RZ/CX/size/depth、相殺依存を弱めたpolicyの資格と価格。古典価格MCのSEとRPE failureは分離。deterministic価格に新MCは不要。
7. **重複**：順序audit前policyは「相殺の程度が異なる対照」に再利用可能だが、相殺なし対照ではない。旧H6 noncancellationのT=.8/DF19/別評価は数値を転用しない。
8. **STOP**：有限集合で対照が成立する/しない/未解決を保存してSTOP。新たな全候補最適化や他Trackの再開をしない。

### P3：retention/cutoffの持続性

1. **目的**：signed biasの符号変化と、drop集合を少し替えたときの相殺・資格・価格の関係を分離する。
2. **固定条件**：各分子の元DF・state・m/t・一つのformula/order/h・compilerを固定。
3. **変更パラメータ**：H4は既存coeff_L1_1/2/4とprefix7_I0/I1、H6は既存prefix12/13/14/15やcoeff_L1_8のうち事前に選ぶ有限集合。任意部分集合探索や新fitをしない。
4. **再利用**：`candidates/kept/pilot_one/det_scan`、保存exact drop集合とHkeep信号、Cの数値norm/ground差。保存計算を繰り返すことと、未評価PF組合せを新規計算することを区別。
5. **新規科学計算**：未保存の同一PF/hでのsigned成分・directed資格・必要な価格。新policyを比較する場合は全18round、N/α、独立評価まで。
6. **判定指標**：drop集合、数値norm/ground差と保証の区別、Hkeepモード/半径、individual/combined位相、rho/β/N/G。単調性を仮定しない。
7. **重複**：Hkeepだけのpilotと一部round資格は既存。全候補をゼロから再評価する必要はない。旧Pauli B0やDF19研究との条件混同を避ける。
8. **STOP**：固定候補集合の完了、または必要な資格未成立を記録してSTOP。screen外から無制限探索へ拡張しない。

### P4：初期状態の摂動

1. **目的**：保存stateの丸め包囲と、物理的な近似ground入力への頑健性を分ける。
2. **固定条件**：元H/target E0、保存DF、採択retention/PF/h/order、δ0と精度/failure、compiler。state prep課金の扱いは元scopeを維持。
3. **変更パラメータ**：sectorを保つ、元groundと直交方向との混合stateを、事前に固定した角度・relative phase・有限方向で定義。低励起方向を使うなら保存Hから必要な状態取得を新計算として含める。
4. **再利用**：INPUT_ARRAYSのH/state、固定controlled回路、source/Schur経路・decoder、同一回路の保存価格。既存`det_qualification`はexact-ground promiseに依存するため、stateを差し替えるだけで完成とはしない。
5. **新規科学計算**：状態family構成とnormalization/overlap、各roundの信号・モード、state依存の誤差資格adapter、branch/alias/N・全G。E0をdecoderへ渡さず、未知stateをoracle groundとして扱わない。
6. **判定指標**：state距離/ground overlap、信号半径とmulti-mode、individual/combined位相、RPE branch、N/G。数値residual allowanceを物理摂動の許容幅に読み替えない。
7. **重複**：D内部/Closed/共同最適化もexact ground仮定であり、今回に移せる同条件state sweepは未取得。
8. **STOP**：有限state集合の判定まで。不成立ならその条件を保持してSTOP。新state準備法や他分子の研究を自動開始しない。

**承認境界**：P1〜P4は新しい数値再解析・条件評価・simulation/compileを含み、全て未実行。過去の3分子比較の一括承認・今回の文書公開承認は、STOP後のこれらの新規実行承認に読み替えない。ユーザーが対象分子、有限grid、SOURCE/input、必要call、全18round評価範囲とSTOPを明示的に承認した後の候補である。

## 10. 他研究・他branchの確認範囲

2026-11-08時点に、SERVER 2のGit refs/worktree metadataとGitHub branch検索（cancellation、stability、df-trunc、各取得済みcursor末尾まで）を確認した。元DF比較と本調査以外で、同じda8b2c3条件のh/state感度検証の公開成果は取得できなかった。非公開チャット、SERVER 1、未公開run、全研究者の実行中jobの不存在は確認していない。他者へのメッセージやprocessへの操作は行わない。

| 関連既存branch・固定資料 | 条件差・重複の判断 |
|---|---|
| [Track A H6 noncancellation Stage B][A_NONCANCEL]（3983fe4f1ede003953735e757d2c098c12b4644f） | tol-only DF19、T=.8、P=0、別精度・analytic shot/profile。元REPORTも厳密bias certificate等を主張しない。今回のfull XDF18/compiler A/18roundの感度結果ではない。方法・保存候補の参考に留める |
| [研究C block witness cancellation][BLOCK]（c298af7ef06192a7d448507abba0ba3b587784e0） | H4 CDF8/LD2、P0、native local errorの三角不等式で失うPauli補正の相殺。Hamiltonian truncationとPF biasの相殺とは別。科学call0・次pilot未承認という記録を保全 |
| [Track D C7-4][D_STABILITY]（64072ba81bb70db1df1b34e94adb08e78926f24b） | 保存88 pathの価格モデルfit・数値安定性。今回のh/retention/ground state摂動の科学検証ではない |
| [Track C D内部BCH][DINT] | D内部biasとD/R混合biasのground相殺。状態特異性の参考。Truncated Hkeepの誤差成分と混同しない |

## 11. 再利用するコードと今は実行しない処理

| 固定SOURCE/関数 | 次の承認済み検証での再利用先 | 追加時の注意 |
|---|---|---|
| `track_c_df_truncation.replay_model`、`candidates/kept` | 保存入力・exact係数・drop集合 | 元入力は読取専用。新SCF/DF fitは通常不要 |
| 同 `pilot_one/det_scan` | Hkeep信号・元順formula/hのscreen | 呼出し自体が新科学計算。proxyは最終資格ではない |
| 同 `det_qualification/qualification_supplement/ground_bounds`、`track_c_source_guarantee` | directed signal・ground/state/source/native/disk | dyadic時刻前提やexact-ground promiseを、新条件へ無検証で移さない |
| 同 `price_piece/circuit_piece`、`track_c_compiler_a.compile_reference` | native first/middle/last、同じRZ優先合成 | saved countsを別angle/構造へ無検証で移さない |
| `track_c_df_order_audit.pilot_one/qualify_one`、`QUALIFIED_COMPLETE` | 登録order比較と完全48点 | pilot/qualificationを再実行せず、同一点は保存結果を再利用 |
| 同 `choose`、`rigorous_shots`、`track_c_closed_p3_joint_opt_alpha.discrete_allocate` | β/rho/αからNとpolicy | 今回はN/再配分/再ランキングを実行しない |
| 同 `independent_one`、order `finish`、RPE decoder | 独立expm/Schur/native X/Yと18round branch | 既存チェックは別AIレビューでも状態摂動testでもない |
| `track_c_df_truncation_report.summarize` | 将来の数値再集計・export | 読取という名前でもArg、shot検査、総G計算を含むため今回は実行しない |
| `tests/test_track_c_df_truncation.py` | Gaussian、controlled scalar、weighted mean、負時間moduleの人工回帰 | 今回再実行なし。state頑健性・性能の科学証拠に格上げしない |

出典：[DET][ORDER][EXPORT][TEST][DISCOVERY]。現在の作業ではCSV/JSONから既存fieldを転記し、raw hashと引用pathを技術的に照合しただけである。新しいアルゴリズム・検証framework・実行guardは追加していない。

## 12. 完了・mandatory STOP

本調査は既存保証を維持したまま、有限orderの支持、h近傍・cutoff/状態感度の不足、回路単価と相殺/Nの寄与を区別した。新しい科学数値・資源順位・性能保証は取得していない。今回の変更はこのREPORTと必要な既存研究入口の追記のみで、旧REPORT・科学SOURCE・raw・STOPは不変。

**調査・資料公開後mandatory STOP。** 次の担当はユーザー：最優先P1、および相殺依存を直接問うP2を含む後続検証の採否と、新しい科学実行範囲を判断する。Codexは追加計算・別分子・新PF family・H8以上・詳細GPT科学レビューへ進まない。

## 固定参照

[BASE_REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/docs/research/track_c_df_truncation_20261108/REPORT.md
[DISCOVERY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/docs/research/track_c_df_truncation_20261108/DISCOVERY.md
[REPRO]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/docs/research/track_c_df_truncation_20261108/REPRODUCE.md
[M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/ordering/TRUNCATION_MECHANISMS.csv
[C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/ordering/CANDIDATES.csv
[R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/ordering/ROUNDS.csv
[E]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/ordering/ERROR_BUDGET.csv
[Q]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/ordering/QUALIFIED_COMPLETE.json
[V]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/ordering/VALIDATION.json
[COMPARE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/ordering/COMPARISON.csv
[RAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/RAW_INDEX.json
[PARTS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/ARCHIVE_PARTS.json
[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/SOURCE.json
[DET]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/src/trottertracks/track_c_df_truncation.py
[SCAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/src/trottertracks/track_c_df_truncation.py#L202
[ORDER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/src/trottertracks/track_c_df_order_audit.py
[EXPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/src/trottertracks/track_c_df_truncation_report.py
[TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/tests/test_track_c_df_truncation.py
[DINT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/docs/research/track_c_d_internal_20261108/REPORT.md
[CP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/docs/research/track_c_closed_p3_20261108/REPORT.md
[JOINT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md
[A_NONCANCEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3983fe4f1ede003953735e757d2c098c12b4644f/docs/research/track_a_h6_noncancellation_stage_b_rerun_20261020.md
[BLOCK]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c298af7ef06192a7d448507abba0ba3b587784e0/docs/research/research_c_block_witness_cancellation_readiness_20261105.md
[D_STABILITY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64072ba81bb70db1df1b34e94adb08e78926f24b/docs/research/track_d_c7_4_model_stability_20261107/REPORT.md
