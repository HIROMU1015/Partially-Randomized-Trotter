# 研究C：保存RC2／PR-tailの条件付きshots・W限定科学実行

2026-11-01 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。

**固定SOURCE・契約を無変更で一度実行し、105/105 logical units COMPLETE、correctness FAIL／retry／resume0。Ideal／assumedの両laneでCONDITIONAL_SHOTS_AVAILABLEとなり、PR-tailの予定十分shots・coherent kernel反復重みWはRC2より小さかった。Evidence laneはEVIDENCE_UNQUALIFIEDのまま。** これは未収束の固定保存候補の仮定付き比較であり、最適RC-DF一般への優位、実機CAや総資源削減ではない。

[全結果JSON](../../artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/run01/result.json)／[156 axis行CSV](../../artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/run01/axes156.csv)／[call journal](../../artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/run01/science_calls.jsonl)／[STOP](../../artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/run01/STOP.json)。以下の表は保存結果の転記で、実行後の追加shots／energy評価・新しい比率計算は行っていない。

## 1. SOURCE・条件・実行範囲

| 固定物 | Identity |
|---|---|
| 科学実行SOURCE | e88fa599af7724f34e11a0f2eab959f92135cf13、無変更で一度実行 |
| 実装commit | b4e466af6b3502b7fe777c4c160bda65d0b9a1ea |
| [105-call契約](research_c_rc2_pr_conditional_shots_contract_unapproved_20261101.json) SHA256 | 10701a6a0f2c92ac2df0bd20b693fe0bb613c754b1969d985e381fdb44cd724e |
| [SOURCE101](../../artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/technical/SOURCE.json) SHA256 | dc3caba2e3cf797f742f3b1ae4a026f353bd2d0a87d24565b529314ddb81a0a4 |
| Stage B科学SOURCE／結果公開 | d6b17b79461a39bd8ba2e6b7389a1f4fad4d74e6／e33ab6e03344c8179ff9b378dc446bf4cfc55e10 |
| 元の共通2-fit科学SOURCE | e2f7dfd44357510d54ab9c6c9c5032e0072ab376 |
| RC2 final J/U SHA256 | d6cb5d56dfc6a83090a6ad1bc1207fdaee62a14cf120542fcb5a7d116644a798 |
| PR-tail final J/U SHA256 | 3fdb0cb77afc770294cbe7aebf47d8a4529e3567ef9e886895495f4490b7ee21 |
| 元h/g/constant snapshot SHA256 | 1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db |

保存直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2／sector36、通常number CDF rank8・一般対称J、元frame順LD2。Partial ordered symmetric S2、δ=h=.1、s1／r128／K0、m0..12、t=.1×2^m。Own-DF ground-state oracle、状態準備費用除外。CA=.00159 Ha、DF reserve=.00019875 Ha、βtotal=π/6、βstat=π/12、candidate failure=.01／axis=.01/26を維持した。

対象は同じ元x0、初期寄与一致α=γ=.1、共通solver・関数評価上限の2保存終点。両DF予算PASS／両optimizer一次収束資格未達のまま。旧γ0／旧RC2／旧PR、旧M4の価格・shotsを移送しない。

[技術準備](research_c_rc2_pr_conditional_shots_preparation_20261101.md)の既存C1 accuracy／energy、Hoeffding、整数range拡張と専用scalar adapterを使った。ユーザーの別途チャット許可に基づいて実行し、歴史的なunapprovedファイル名・metadataは保全した。承認証明・承認原文保存・新grantは作成しない。共通運用指示2026-10-25版に従い、重要GPT科学レビューは行っていない。

## 2. Lane別の結果と整数会計

Nは予定十分測定回数のX＋Y合計。WはΣ_m2^m(N_X＋N_Y)で、s1のkernel occurrence重みである。r128はkernel内部のmicrostep数でありWに追加で掛けていない。実際に量子測定した回数は0。

| Lane | 候補 | N_total | W | Lane比較 |
|---|---|---:|---:|---|
| evidence | RC2／PR | null | null | EVIDENCE_UNQUALIFIED |
| ideal_exact_native | RC2 | 18,200,774,092,602 | 74,550,091,248,614,488 | CONDITIONAL_SHOTS_AVAILABLE |
| ideal_exact_native | PR-tail | 14,504 | 11,822,240 | CONDITIONAL_SHOTS_AVAILABLE |
| assumed_raw_signal_budget | RC2 | 18,282,907,461,654 | 74,886,508,341,850,424 | CONDITIONAL_SHOTS_AVAILABLE |
| assumed_raw_signal_budget | PR-tail | 14,562 | 11,875,484 | CONDITIONAL_SHOTS_AVAILABLE |

Idealはexact native synthesis／ideal finite-RTE IIDを仮定する。Assumedは既存C1と同じ各round epsSyn≤.001A、epsLaw≤.001Aという未検証の感度条件。双方ともASSUMED_SCENARIOであり、未知のactual誤差を実測0や資格付き上界へ昇格していない。

比較callで保存したPR−RC2の整数差は、idealでN=−18200774078098／W=−74550091236792248、assumedでN=−18282907447092／W=−74886508329974940。両仮定laneで予定N・WともPRの方が小さい。総RZ／CX・native価格・shell費用を含む差ではない。

### 13roundのper-axis十分shots

同じ十分数をX/Yの各軸へ割り当て、合計では両軸を一度ずつ計上した。

| m | RC2 ideal／axis | PR ideal／axis | RC2 assumed／axis | PR assumed／axis |
|---:|---:|---:|---:|---:|
| 0 | 514 | 512 | 516 | 514 |
| 1 | 517 | 512 | 520 | 514 |
| 2 | 524 | 512 | 526 | 514 |
| 3 | 536 | 512 | 538 | 514 |
| 4 | 561 | 513 | 564 | 515 |
| 5 | 615 | 514 | 618 | 516 |
| 6 | 739 | 516 | 742 | 518 |
| 7 | 1069 | 520 | 1073 | 522 |
| 8 | 2232 | 527 | 2241 | 529 |
| 9 | 9750 | 544 | 9789 | 546 |
| 10 | 186105 | 578 | 186873 | 580 |
| 11 | 67910911 | 653 | 68199279 | 656 |
| 12 | 9100318932228 | 839 | 9141385327548 | 843 |

全52仮定roundはCONDITIONAL_ROUND_PASS。Evidenceの26roundはUNRESOLVED_SIGNAL_ERRORで、shots／raw-signal誤差／半径はnull。INFEASIBLEな仮定roundやTECHNICAL_FAILUREはなかった。

### 高mの保存group値

| 候補／lane | N(m0..9) | N(m10..12) | N(m12) | W(m12) |
|---|---:|---:|---:|---:|
| RC2 ideal | 34,114 | 18,200,774,058,488 | 18,200,637,864,456 | 74,549,812,692,811,776 |
| PR ideal | 10,364 | 4,140 | 1,678 | 6,873,088 |
| RC2 assumed | 34,254 | 18,282,907,427,400 | 18,282,770,655,096 | 74,886,228,603,273,216 |
| PR assumed | 10,404 | 4,158 | 1,686 | 6,905,856 |

m12はm10..12に含まれるdiagnostic subsetで、groupを重複加算しない。RC2の大きな負担は長round、特にm12に集中している。PF／補正RTE誤差を0にした比較ではなく、保存した両誤差を残したままの結果である。

## 3. Raw-signalとconditional energyの資格

既存のd=A(εPF＋εRTE)＋epsSyn＋epsLaw、ρ=A−d、βdet=asin(d/A)、εcoord=ρsin(βstat)/√2を使った。DF誤差[Ha]はraw-signal誤差へ加えず、既存energy評価に一度だけ入れた。

| m12の数値diagnostic | RC2 ideal | PR ideal | RC2 assumed | PR assumed |
|---|---:|---:|---:|---:|
| Raw-signal誤差d | 9.317767265139566e-7 | .08395865220319684 | 9.486261287738976e-7 | .0856873402388465 |
| Radius lower ρ | 7.4929244034565766e-6 | .7803853656216324 | 7.476075001196635e-6 | .7786566775859828 |
| βdet [rad] | .11082730453400348 | .09728909322261163 | .11283987713339307 | .099298794308805 |

RC2の絶対raw-signal誤差dが小さくても、Aとρも小さいため十分shotsは大きい。Raw誤差だけを小さい順に並べて資源を判断しない。

| Energy十分条件（ideal／assumedで同じ） | RC2 | PR-tail |
|---|---:|---:|
| Status | CONDITIONAL_CA_PASS | CONDITIONAL_CA_PASS |
| 登録式のenergy error値 [Ha] | .0012783212039525452 | .0012783245400760227 |
| Statistical failure upper | .01 | .01 |

このenergy値は、保存DF bound＋登録βtotal/(δ2^12)という既存の十分条件評価。実際にenergyを測定した誤差ではない。β配分を結果後に最適化していない。Failure上界は各candidate/lane内の13round×2軸についてで、複数laneや両候補のjoint confidenceは新設していない。

Evidence laneは、全law synthesis／implemented-law誤差が未知のため、既存energy判定がUNQUALIFIED_ROUNDS／same_accuracy=falseとなる。Actual synthesis／law誤差、actual_CA_achieved、guaranteed shots、native μ、shell S、総資源は全laneでnullのまま。Ideal／assumedのCONDITIONAL_CA_PASSを実機CAや厳密CA保証と扱わない。

## 4. 数値精度・範囲

実データの全52仮定roundは既存binary64 Hoeffding helperで算定され、range_extension=false。Extreme-A／有理数ceil fallbackを実データで使用したとは主張しない。準備時の人工テストがその拡張域を検証している。

Per-axis shotsはPython整数、N・WはPython整数で集計し、canonical decimal文字列を保存した。Wはbinary64で1単位を表現できない大きさでも、floatやint64へ変換していない。表の桁区切りは表示用で、JSON／CSVの正確な文字列を正本とする。JS Number／Excel等への自動数値変換による精度喪失に注意が必要である。

保存156 axis行はJSONの同じper-axis countと一致確認した。Clip、任意のshot上限、Wへのr重複乗算、量子測定／samplingは0。Binary64 operator誤差・log／sin定数に対する数値pilotであり、整数の保存精度を超越関数・全law誤差の厳密enclosureへ昇格しない。

## 5. Calls・資源・保全

| 科学処理 | Units | 結果 |
|---|---:|---|
| Bundle／identity／target-horizon | 3 | COMPLETE |
| 両arm scalar attachment | 2 | COMPLETE |
| Qualification | 6 | COMPLETE |
| Round accuracy | 78 | COMPLETE |
| Conditional energy | 6 | COMPLETE |
| Shots／W集計 | 6 | COMPLETE |
| Lane比較 | 3 | COMPLETE |
| 最終result | 1 | COMPLETE |
| 合計 | **105** | **105 COMPLETE、FAIL0、inflight0** |

[実行preflight](../../artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/execution_preflight.json)でSOURCE101／入力14／契約・lock・package版・fresh outputを確認し、[人工39件](../../artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/execution_synthetic_tests.txt)を再確認してPASSした。準備時125件を今回全再実行したとは扱わない。

新fit、optimizer、J/Uやh/gの数値decode、Hamiltonian／ground／PF／RTE／A再計算、native build／compile、価格校正、sampling／実量子測定、総資源評価は0。Retry／resume0、実行SOURCE・科学条件・許容差の事後変更もない。

計算は契約どおり小さいscalarと整数の逐次処理。Preflight時affinity32 CPU、available RAM56,134,582,272 bytes、load約1.14。CPU/RAM/worker/wall/shotの人工上限や共有環境変更はない。Runner内部wall1.2189184483140707秒、process CPU9.57451053秒、peak RSS346728 KiB。CPU/RSSはprocess計測でimport区間も含み、wallと測定区間が異なる。速度向上率やGPU実行を主張しない。

[Postflight](../../artifacts/research_c_rc2_pr_conditional_shots/2026-11-01/execution_postflight.json)でSOURCE101／入力14、旧Stage B SOURCE74／raw69／CSV、旧fit SOURCE44／raw37、旧JSON修復SOURCE47／raw7、旧partial／failed STOP、今回raw92件を照合した。新しい科学計算を追加せず、journal、decimal型、CSV／JSONの一致を確認した。

## 6. Q1〜Q10と次の研究判断

| 問い | 今回得られた情報／限界 |
|---|---|
| Q1：候補別shots | 両仮定laneでPRが小さい。Ideal NはRC2 18200774092602／PR14504 |
| Q2：長roundの影響 | RC2はm12でper-axis 9100318932228。保存group値にN／Wの集中が現れた |
| Q3：PF悪化を入れても残るか | 保存PF／RTE誤差を維持した上で、両仮定laneのPR予定shotsが小さい |
| Q4：Wでも改善するか | Ideal WはRC2 74550091248614488／PR11822240。AssumedでもPRが小さい |
| Q5：Ideal対assumed | この固定感度条件ではshots／Wの方向は変わらなかった |
| Q6：Evidence未資格 | Synthesis／implemented-lawが未知で、実証的なshots・CA資格なし |
| Q7：価格取得の情報価値 | 候補固有μ/Sを取得すれば、今回のW差と合わせた条件付き費用比較を検討できる。実施GOはユーザー／研究Cが判断 |
| Q8：未収束の制限 | 固定予算終点の観測に限定。十分最適化されたRC-DF全般への優位とはいえない |
| Q9：最小native scope | 次の技術準備案は両armの同じh-step controlled partial-S2 kernelと、外側head／X/Y shell。保存referenceを再利用し、公平なcompiler・融合・正しさ範囲を設計する。標本数・call範囲は未採択 |
| Q10：研究方向 | 下流の条件付きshots/W差は確認できた。実装誤差の資格と候補固有価格が不足しており、総資源や最終GOは未判断 |

**次の担当：ユーザー／研究Cが、両候補のnative価格・誤差資格を取得する最小技術準備へ進むか判断する。Codexは具体的依頼後に準備し、新しい科学計算は固定範囲の別明示許可後だけ実施する。**

結果・SOURCE・raw・journal・STOPをcommit/pushし、remote SHA/tree・公開本文・原本との整合性を確認する。**公開照合後mandatory STOP。** 今回からnative、合成/law追加計算、総資源、追加fit、重要科学レビューへ自動進行しない。
