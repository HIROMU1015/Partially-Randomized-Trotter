# 決定論PF：ステップ幅とwhole-shot配分の共同比較計画

2026-11-15 JST、Codex B。**固定資料の照合と計画のみ。新Q3/DP/二項確率/信号/価格/費用の計算は0件。** 本案の予算・生成規則・最大12 callは次回の採択・実行許可の対象であり、今回実行していない。

## 1. 達成済み基準点と、幅別の入力coverage

固定H4 linear 1.00 Å/STO-3G/tol-only1e−8 DF10、sector(2,2)、保存H/state/snapshot、h_RPE=0.1、M=12、P=0。以下のh表記は幅の単位h_RPEである。探索するPF要求幅Δ_PFと、固定するRPE基準時間h_RPEを区別する。

[Aの完成報告][A]・科学SOURCE fb9a1878b677b7ee116b52e804e5a01cdaa96004・carrier 06a15ee45beb177c018f6260ea4a5ac8cf66cfe6の6 packet、[Bの固定配分Q3][B]、[B2の156軸会計][B2]を再利用する。各公式・現幅の保存配分は両軸共通：

~~~text
N0_X = N0_Y = [59,55,51,47,43,39,35,31,27,23,19,15,11]  (m=0..12)
~~~

6件とも保存componentで採択Q3閾値適合。原API return・CPU/peak RAMの欠測という**部分保存・運用UNKNOWN**は不変で、再実行して埋めない。現在の幅・N0で達成済みの費用基準はFull PF S2 **838,693,580 RZ**、B0I p7 S2 **593,555,352 RZ**（B2の引用）。他4候補も同じN0で適合・費用集計済みである。

幅候補の起点は[旧T2][T2]、その参照先の[Full PF scan][SCAN]・[follow-up][FOLLOW]・[元T2候補表][TP]。旧Aの[保存信号audit][AUDIT]の照合方法・SOURCEを再利用し、後続の完成6 packetを重ねた。新しいmean値の比較・再解析はせず、候補の物理metadataだけを対応付けた。

**[WIDTH_SIGNAL_COVERAGE.json](WIDTH_SIGNAL_COVERAGE.json)** が29保存configurationの索引。各幅について全桁configuration ID、全13roundのphysical ID・T・q_PF・実幅がある固定schedule locator、保存meanのroundとSOURCE、旧meanだけのround、現行mean不足round、packet・Q3・B2の有無を保持する。表のIDはconfiguration先頭8桁であり、roundのphysical IDとは別である。

分類：①全13roundのsource-bound packetあり、②一部roundの現行meanあり（別scheduleからの同物理roundを含み、再packet/binding確認が必要）、③step/schedule/旧誤差評価のみで独立した13round packetなし、④保存Q3・実際の配分あり。旧legacy meanは②の現行meanへ昇格せず索引で分離する。

| 系列 | 現幅：①＋④、N0で適合 | 保存された細かい側 | 保存された粗い側／未取得域 |
|---|---|---|---|
| Full PF S2 | 16h/7：11cf8ca6 | h/2：ee716448、h：f2d58f32、2h：9e96ed75。各② | 8h/3：7754671b、4h：dde62a35、8h：968e5ba8。各②、Q3なし |
| Full PF Yoshida4 | 16h/3：a799697f | h/2：114da23a、h：da864ed2、2h：29848171、4h：99e53903。各② | 32h/5：9e930a47、8h：b0ef551a。各②、Q3なし |
| Full PF M5 | 16h：70f7b317 | h/2：0a7d9bef、h：8f751424、2h：825e12b0、4h：5b7e26fc、8h：24286ac5。各② | 16h超の保存評価は確認範囲になし。32hは旧T2b等の提案のみ③、固定新configurationなし |
| B0I p7 S2 | 16h/7：a6e55777 | h：e6754564は①（total metadata受入済み、Q3なし）。2h：155a3a2eは② | 8h/3：5576418dは②、Q3なし |
| B0I p7 Yoshida4 | 16h/3：defc5e88 | 4h：05d246cbは②。h・2hはT2のearly actual step cellのみ③で、独立要求幅の保存configurationではない | 32h/5：dd8bfc24は②、Q3なし |
| B0I p7 M5 | 16h：424eecbf | 8h：a62d4ab0は②。h・2h・4hはearly actual step cellのみ③ | 16h超は上のFull PFと同様に未取得。32hの独立固定候補なし |

T2のp8/p9やFull-H/PRは今回の6系列へ混入しない。現6幅以外に、確認した同契約・同配分のtotal Q3結果はない。旧T2・H_effの判定やtotal metadata受入だけをQ3結果として数えない。

### 全roundを一つの表示幅で代用しない

現S2 16h/7のm0〜m4は、A記録でq=1,1,2,4,7、実幅=h,2h,2h,2h,16h/7である。Y4/M5もearly例外・signed pieceを含む**元scheduleをそのまま使う**。表示のΔ_PFだけからqを丸め直さず、索引の固定locatorで各roundを指定する。新しい幅は別configuration/physical scheduleで、元total IDや価格receiptを付け替えない。

旧T2ではS2 8h/3、Y4 32h/5が局所ε_trot予算超過だった。**それだけを理由に最終Q3候補から外さない。** 逆に旧局所予算内・小さいH_eff shiftだけでQ3適合とも判断しない。旧数値・旧Minkowski判定・STOPはそのまま保全する。

## 2. 共同探索の単位と比較規約

各方式の候補は「公式＋要求幅＋固定全round schedule＋whole-shot配分」の組とする。目的はT(Δ_PF,N)=Σ N[m,a] C_Δ[m,a]、制約はMSE_Q3(Δ_PF,N)≤τ。Nのround独立最適化や、MSE∝1/N、幅・Nに関するMSE単調性は仮定しない。

共通精度契約は採択Z1＋T1 total decoder、保存DF10 solver energy proxy −2.1663874486347634 Ha、全事象loss=(Ehat−Eproxy)²、十進ε_E=0.00159360010199040 Haのexact二乗τ。shots・X/Y・round間IID二項、EXACT_AFFINE_OF_BINARY64_RAWを維持する。**決定論PFのpacketはdeterministic raw、B=1であり、PRのfinite-K/RTE law・normalizationを後付けしない。** 各幅の時間発展近似・p7切断の影響はraw直接MSEに入り得るため、AのδEやδE²は二重加算しない。

価格は同じ物理候補・round・軸の測定付きfull controlled Hadamard wrapper、P0。現6候補はB2 AXIS_TERMSのCと元SOURCE/receiptを再利用する。96直接＋60条件付き閉包という区分、PF_FIXED_COMPILER_CLOSURE_CONDITIONAL_60_AXES_V1、CI/誤差上界null・2%保証なしを保持する。N変更は1shotの物理回路を変えないが、**新しい固定配分の条件付き会計も次の実行範囲として明示する**。別幅のCへは横展開しない。

比較はFull PF内3系列、B0I p7内3系列それぞれと、指定有限menu全体で行える。同じ幅やshot総数に揃える必要はなく、各系列が別のΔ_PF・Nを選べる。結果は「調査した有限menu内で確認できた最小適合費用」。未評価配分・未取得幅・UNKNOWNが残れば、その範囲も明記し、大域最適・方式全体の勝者とはしない。

## 3. 最初の有限Q3案：現6幅×2配分族、最大12件

### 同じ費用を目指す2配分族（生成は今回実施しない）

現6 packetと価格で開始できる。各方式の達成済みS2費用をB_methodとする。**提案する最初の費用challengeは一段だけ、β=3/4、予算B*=β B_method**。これは全系列に25%低い費用目標を課す有限実験の設計値であり、shotを3/4にする指定、risk予測、安全率、新しい合否閾値ではない。ユーザー採択が必要で、今回はB*や新整数列を算出しない。

同じ整数制約1≤N[m,a]≤N0[m]の下で、各候補iに次の2本を生成する（X/Yの単価が異なれば列も別になる）。

~~~text
A：形保持       w[m,a] = N0[m]
B：価格で振替   w[m,a] = N0[m] / C_i[m,a]

N[m,a](λ) = max(1, min(N0[m], floor(λ w[m,a])))   (λ >= 0)
~~~

いずれも、整数列が変化する有限breakpointの中から、Σ N C_i≤B*を満たす最大の列を選ぶ。λの上限は全成分がN0で飽和する点まで。breakpointはk/w[m,a]（k=1..N0[m]）、演算は整数・Fractionとし、予算に対する探索の単調性だけを使う。**riskの単調性やriskを見たλ調整は使わない。** 全成分1でも予算に収まらなければBUDGET_MENU_EMPTYとして未投入にする。価格欠測・不整合・非正単価なら生成を保留し、0やモデル価格を補わない。

Aは元の配分形状を保つ。Bは同じ費用上限で単価の低いround/axisへ配分を寄せるため、Aに比べて安い側を増やし高い側を減らす選択肢を持つ。N0上限は**初回の有限探索領域**であってCPU/RAM制限ではない。振替がMSEを改善するとは予測せず、whole-vectorごとのQ3で直接調べる。後にN0超の振替が研究判断に必要なら別の有限案にする。

| 系列／現幅・元ID | 初回の配分 | 固定価格SOURCE | 判定したい問い | 新Q3上限 |
|---|---|---|---|---:|
| Full PF S2・16h/7・11cf8ca6 | A_i、B_i | B2 AXIS_TERMS、同configuration/m/axis | S2自身も達成費用より低い予算で適合を保てるか | 2 |
| Full PF Y4・16h/3・a799697f | 同じ2生成規則、Full PF予算 | 同上、Y4固有行 | S2で達成済み838,693,580 RZ以下（本案ではさらにβ倍の上限）で適合できるか | 2 |
| Full PF M5・16h・70f7b317 | 同じ2生成規則、Full PF予算 | 同上、M5固有行 | 高い1shot価格を別shot配分で補い、同じ費用目標を満たせるか | 2 |
| B0I p7 S2・16h/7・a6e55777 | A_i、B_i | B2 AXIS_TERMS、p7 S2固有行 | p7 S2自身にも配分改善の機会を与える | 2 |
| B0I p7 Y4・16h/3・defc5e88 | 同じ2生成規則、p7予算 | 同上、p7 Y4固有行 | p7 S2の593,555,352 RZ以下（本案ではβ倍）で適合できるか | 2 |
| B0I p7 M5・16h・424eecbf | 同じ2生成規則、p7予算 | 同上、p7 M5固有行 | 同じ最終精度・費用目標で公式の選択が変わるか | 2 |

整数制約で予算を使い切れない場合は、実際のTと未使用予算を保存する。同額と偽らず「同じ費用上限内」で比較し、新しい許容差は置かない。

全6系列で同じβ・生成規則・判定手順を使い、**S2にも2本の改善機会を与える**。初回は費用budgetへの適合とQ3適合を別々に記録する。適合すれば元S2基準点を上回る資源削減の候補が得られ、不適合ならこの2本で達成できなかったことだけが判明する。幅全体や公式全体を棄却しない。

各allocationには独立ID・SOURCEを付し、元configuration/physical IDは維持する。実行前に12候補以下の**具体的なX/Y整数列・C参照・条件付き費用**を確定してINPUTS/callerをcommitする。同じ物理候補でA/Bが同一列になった場合、N0と同じ場合、又は§10.5で同等の保存結果が見つかった場合は1回へ統合／再利用し、新規callを減らす。12件を埋めるために別列を追加しない。

### 既存入口・有限範囲・保存経路

- signalはA PACKET_INDEXのconfiguration/packet locatorを直接使用し、既存rpe_total_binding.create→bind_signal→evaluate、total kernel710aa69b・高速backend9e294778を再利用。MC用diagnoseを呼ばない。rpe_shot_allocationのallocation仕様・元configurationに対するprepareを再利用する。実際のT.evaluateは元configurationとkernel_signal_packetをQ.evaluateへ渡して旧閉包basisを使い、返却後にtotal canonicalへ束縛する。新total IDだけを旧prepareへ直接渡さない。各jobはallocation1本のみとし、1 API callに複数配分を詰めて科学件数を曖昧にしない。
- PF元callerのnumericsを再利用し、後続run_pr_df_fullh_total_q3_fixed.pyのworker自身によるAPI_RETURN／RESULT／partials／RESOURCE_FINAL／PERSISTED保存、例外後の監視保護を用いる。旧6件の保存欠測は保持する。新callerのPF route・複数allocationのnamespaceと、実際の入口から保存までを人工fixtureで検査してからSOURCEを固定する。
- ρ=1、maximum_round_outcomes=4,225、maximum_verification_pairs=55,043,173、maximum_frontier_states=8,192（非終端のみ）、maximum_transition_pairs=141,559,036／callを提案。全N≤N0なので、旧count-onlyの包含上限3,600／38,582,272／117,985,808を上から収容する。ただし**新しいcount格子・偶奇でのbinary64閉包PASSは旧配分から流用しない**。具体列でcount-onlyを再照合し、閉包は各call内で確認する。
- zero HOLD、tie argmin、prior/alias/energy shift、principal guard、残余幾何loss、engineering allowanceを変えない。範囲外質量を捨てない。1候補2配分を含む独立jobの並列化・既存batch処理を利用し、同一入力の二重投入を防ぐ。CPU/RAM/workerに恒常的な上限を追加せず、空き資源と既存実測に合わせる。
- **通常のrpe_shot_allocation.compare／controller正式cost/selectは資格Nを要求する**。本件のQ3適合を十分N packetへ偽装して通さない。条件付きTの比較にはB2のprice join・integer検査と既存exact算術を再利用する最小callerを使い、結果をCONDITIONAL_FIXED_ALLOCATION_COSTとして別記する。新controller・N資格producerは作らない。

### 判定・重複・STOP

開始直前は、今回参照した固定Q3、aae9f217の選定記録、関連branchの後続成果・同一job・出力先の差分を§10.5で照合する。物理候補＋全13round X/Y整数列＋decoder＋target＋law＋lossが同じ受入可能結果はSOURCE付きで再利用。旧H_eff、partial decoder、別幅、別Nの結果は代用しない。既取得か不明なら保留する。

各callの有効上界≤τでMODEL_BOUND_MEETS_ADOPTED_THRESHOLD、下界>τでMODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD、下界≤τ<上界でINCONCLUSIVE。必要なSOURCE・閉包・質量保存・残余会計・数値allowance・上下界/moments整合gateが不成立ならUNKNOWNとし、数値判定と保存状態を分ける。lemmaの保守的超過だけで最終不適合とせず、既存kernelの有効上下界を使う。

重大な入力/gate/資源/保存異常時は新規call投入を止め、開始済みjobの証拠を保存。無断retry、ρ/範囲/backend/target/loss変更、結果依存の列変更は禁止。**最大12 callの終了又は重大異常でmandatory STOP**。結果を見てβ変更・新幅・別配分へ自動進行しない。

## 4. 幅を動かす次段階：A/Dに必要な不足

初回12件は現6幅だけなので、**共同探索の配分側から始める初回batch**である。幅も選び直した最適性はまだ判定できない。次は同じ費用予算と配分生成規則を、A/Dの入力が揃った別幅へ適用する。全幅を一括取得しない。

| 別幅／元configuration先頭 | 保存現行mean候補（同物理roundからの共有を含む） | Aへの不足と、その取得で可能になる比較 | Dへの依頼 |
|---|---|---|---|
| S2の近い粗幅8h/3：Full 7754671b、p7 5576418d | m0,1,2,3,12 | **m4〜m11**の現行meanとsource-bound packet。旧局所予算超過の幅で、最終Q3では費用節約が可能かを調べられる | 索引の各元ID・全roundのphysical/T/q/実幅へ同profile価格coverageを照合。既存6幅のCをq比例で補わない |
| Y4の近い粗幅32h/5：Full 9e930a47、p7 dd8bfc24 | m0〜m5,12 | **m6〜m11**の現行meanとpacket。同じ問いを高次PFで確認できる | 同上。元cohort/receiptが再利用可能かをDが確認し、不足だけを別範囲で計画 |
| 細幅の対照：S2 2h（Full 9e96ed75／p7 155a3a2e）、Y4 4h（99e53903／05d246cb）、M5 8h（24286ac5／a62d4ab0） | 各m0,1,2,3,12。m4にはlegacy meanのみ | **m4〜m11**の現行meanが必要。細かい幅が許すshot削減で1shot価格増を相殺できるかの対照。legacy再資格化の可否はA判断で、無条件流用しない | 新幅として価格のcoverage・資格を確認。既存m3/m4 locatorの存在だけで現guard受入としない |
| p7 S2 h：e6754564 | 全13round packet／total metadataあり、Q3なし | signal再取得は不要。SOURCE再照合後のQ3自体は保存入力で可能。ただし初回12件には加えない | 同候補26軸価格が未確認なので、費用budgetによる配分生成には価格側の不足を先に明確化 |
| M5のさらに粗い32h：Full/p7とも新固定IDなし | 旧提案のみ。16hのH_eff窓PASSを移送しない | まず既存generatorによる新schedule・signed step/branch方針と必要signal scopeを別計画で固定。全round取得やQ3を今回の許可に含めない | 固定configuration/physical一覧を得てから同profileの必要価格を計画。ID未確定の価格取得を先行させない |

Aへの依頼は保存step・現在の同物理meanの再利用を優先し、全roundを再生成する依頼ではない。新canonical viewへの物理同一性・元SOURCE・law/normalization/targetの照合を伴う。索引の「不足」は確認した固定資料で現行meanを同定できなかった範囲であり、他serverの不存在証明ではない。

近い粗幅・細幅のどちらを次に取得するかは、初回の費用予算における適合状況と既存価格coverageで絞る。M5もS2/Y4と同じ選択手順を使い、保存幅上端が違うことを未探索域として明記する。1配分の不適合、旧H_eff超過、構造piece数だけで幅を打ち切らない。Dの別幅価格調査とAの信号準備は独立して進められ、現6候補の初回計画を止める依存ではない。

## 5. 今回の確認範囲と保全

固定29 configurationのschedule/保存round metadata、旧A audit・legacy locator、後続6 packet、B Q3とB2・選定記録を照合した。確認したGit履歴では現6候補の別whole-shot Q3や幅・配分の共同実行結果は未同定である。これは科学job投入前の最終照合の代わりではない。元コード・既存人工検査結果を参照し、今回科学module import／数値testは起動しない。

静的検査PASS：29件の全13round schedule locator、7完成packet、索引内のSOURCE/path/selectorが解決すること、元q/physical IDの対応、現行/不足roundの区分、旧legacy区分、引用N0・B2基準金額、文書リンクを確認した。科学値の比較・配分生成・費用算術を行う検査ではない。

追加は本REPORT・[HANDOFF](HANDOFF.md)とmetadata索引のみ。旧T2・PF6の原結果、SOURCE、raw、契約、grant/ledger/marker/KEEP/STOPを保全し、他担当worktree・ロードマップ本体は変更しない。ground/state/target包含・実sampler移送・数値certificateの未資格と、sufficient_N/global_RMSE_bound/G_qualified/winner=nullを維持する。

**計画公開・remote照合でmandatory STOP。** 次回は少なくともβ・2生成規則・具体列確定の保存価格算術・新固定配分の条件付きT会計・最大12 Q3と有限範囲を明示採択・許可する必要がある。信号取得・別幅価格・compileは別範囲であり、この計画の採択だけでは実行しない。

[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1adb7e65a9e7c3adc32491d853f31d124393ead7/docs/research/pf_shortlist_mean_completion_20261115/REPORT.md
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dba096dc39788b8948af9eafe394f03d386c520/docs/research/pf_shortlist_q3_fixed_results_20261115/REPORT.md
[B2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1002f1dfea34de439ca1b704a0192bca473c5387/docs/research/pf6_allround_cost_20261115/REPORT.md
[T2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5fb2c32e39e63267ea1748861269c362a93a1b86/docs/research/truncated_pf_t2_results_20261114/REPORT.md
[TP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c3ae31f353d27a9f720f51fa0b089acd3d57e211/docs/research/truncated_pf_optimization_plan_20261114/PLAN_METADATA.json
[SCAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/714247619afb4714fd663eb15c0470ef32bdc7af/docs/research/h4_full_pf_heff_scan_results_20261114/REPORT.md
[FOLLOW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/714247619afb4714fd663eb15c0470ef32bdc7af/docs/research/h4_full_pf_followup_results_20261114/REPORT.md
[AUDIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c212cee857d639d2400819bbca3afa1dc052a9f4/docs/research/a_signal_candidate_identity_coverage_20261114/REPORT.md
