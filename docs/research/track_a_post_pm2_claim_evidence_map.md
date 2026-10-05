# Track A 原稿の主張と証拠の対応

2026-10-05。PM-2後の方針レビューを受け、追加科学計算をせず、現在の証拠で閉じる事例研究の主張を整理する。
本書は原稿化の判断・根拠表であり、新しい検証結果、実行authorization、査読誌での新規性確定ではない。
図と章の設計は[原稿設計](track_a_post_pm2_manuscript_design.md)を参照する。

## 採用する範囲

主RQは「固定DF表現・二次PF・canonical finite-RTE・所定のshot規則のもとで、
有限時間coherent signalを同じ要求精度で推定するとき、登録した残差処理構成の資源競争力を
総bias、normalization、測定負担、full-wrapper費用からどう説明できるか」とする。

寄与は、既知の残差処理を条件の揃った実装資源台帳で比較した定量的事例である。
partial randomization、DFへの適用、discardとの比較、cost×shotsという原理そのものを新規成果にしない。
新しいalgorithm、最適sampling法、未知系に使える安価なselectorを提案しない。

| 共通条件 | 原稿で必ず示す内容 |
|---|---|
| Hamiltonian・状態 | 保存されたH4 linear、STO-3G、DF rank12、8 system qubits、保存参照状態とfull-H target。新たな状態・Hamiltonianを生成しない |
| development | 隣接距離1.00 Å、M1の210構成＋PM-1の8構成＝218。参照値を用いたbenchmarkで、truth-free運用policyではない |
| transfer | 1.30 Å、developmentから結果前に固定した元M2の5構成のみ。PM-2では使用済みデータのPOSTHOC感度 |
| 時間・実装 | T=0.8、二次DF-prefix PF。q=1/2/4/8、delta=T/q=0.8/0.4/0.2/0.1、登録されたr/Kだけ |
| split | DF rank12は表現のrank、candidate名のrankはprefix長L_D。B0は残差discard、B1は全fragment決定論、B2は残差finite-RTE補完、B3は二体prefix0でone-bodyを残すrandom-dominant |
| shot規則 | 実部・虚部へε/√2ずつ配分、α_real=α_imag=0.025、補正後axis biasとnormalizationを使う十分shot式 |
| 費用 | 状態準備を含まない測定付きHadamard full wrapper。Qiskit1.3.0、opt1、rz/sx/x/cx、seed17、topology指定なし |
| primary / secondary | primaryは解析shot数×軸別期待compiled RZ。secondaryは残る5 compiled metrics、共通仮想P感度 |
| 精度domain | 保存された301対数点＋正確な0.05の302点、ε=0.005〜0.1。点の追加、連続境界の再探索はしない |

## 根拠の正本と固定identity

以下の相対pathは、このworktreeのrepository rootに対するもの。最新結果を収録した固定commitは
5a1adffad780f0ec4272f5e8bb94713f9ff0f2bcである。
文書の過去の「未commit」「未実行」は当時の履歴として扱い、現在のblobの所在と区別する。

| 根拠 | 文書・数値の所在 | 証拠階層とidentity |
|---|---|---|
| E1 development compile | [M1-B1照合](../pr2_matched_accuracy_m1_b1_result_validation.md)、[result JSON](../../artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/pr2_matched_accuracy_m1_b1_compile_map_result_v2.json) | 元のlocal execution。result commit 8e0814e70c14ecf526444fac8a2142799610dc96、science source 33f436bb3a7d5b9cefa23604bb22c8d1fb17cd62 |
| E2 近接discard | [PM-1照合](../pr2_pm1_discard_result_validation.md)、[result JSON](../../artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04/result.json) | 8 signal・16 deterministic wrappersのlocal execution。source fd7552edc0334ccf57ecf501a128c85c8d22822a、入力を収録したcommit 194cc604b90c56a0e7e949b91b064a4bcfc846da |
| E3 固定transfer | [M2照合](../pr2_matched_accuracy_m2_transfer_result_validation.md)、[result JSON](../../artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json) | 結果前固定5構成のlocal held-out execution。source 2978e2fea672b7a1ff20cac74269ec9a610159dc、launch HEAD 40a11d02a67954175686b2e532b83c7953ec8316 |
| E4 帰属・same-R | [PM-0報告](pr2_post_m2_evidence_attribution.md)、[summary](../../artifacts/resource_applicability/pr2_post_m2_evidence_attribution/2026-10-04/summary.json)、[same-R CSV](../../artifacts/resource_applicability/pr2_post_m2_evidence_attribution/2026-10-04/same_R_candidate_comparison.csv) | POSTHOC保存値・source監査。元入力commit b6e65c6123475add5e620ec1064f361378bead95。今回は再実行しない |
| E5 精度・資源感度 | [PM-2照合](../pr2_pm2_precision_resource_result_validation.md)、[summary](../../artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/summary.json)、[precision ledger](../../artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/precision_ledger.csv)、[eligibility](../../artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/eligibility_boundaries.csv) | POSTHOC保存値解析。source 324435d77b6642dbd44e8d1f178420daf62e77ed、結果commit 5a1adffad780f0ec4272f5e8bb94713f9ff0f2bc |
| E6 準備費用・費用項 | [P envelope](../../artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/P_envelope.csv)、[代表decomposition](../../artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/representative_decomposition.csv)、[claim audit](../../artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/claim_audit.json) | E5と同じ保存標本・同じ結果commitからの感度。独立試験ではない |

PM-2 summary SHA-256：
7b026d4cc657cf43ad23fd7d6e10d5aa31cd58af555649b7a8e858ea12155845。
runner manifest SHA-256：
546cdfaf8c77f349f6f55b346e93749ce5956c9a605281d169887257377c843f。
今回、両fileが上記結果commitのblobと同一であることだけを再照合した。
全ledger検証の根拠は既存[validation audit](../../artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/validation_audit_v1.json)であり、今回独立に全数値を再検証したとはしない。

利用者提供reviewはworkspace rootの
pr2_post_pm2_research_redesign_20261005.md
（SHA-256 815e84d32fa44c5b536cdd8f6c725b3799f06698831ad4559a1142458b4e116a）。
研究判断の入力であり、このworktreeのcommit済み科学artifactと混同しない。移動・編集していない。

## 採用する主張と、同時に書く限定

表中の数値には上記共通scopeを適用する。C1〜C3を本論の中心、C4〜C6を解釈・補足に使う。

| 主張 | 根拠・比較集合 | 言えること | 言わないこと | 対応 |
|---|---|---|---|---|
| C1 近接discard追加後もB2の低いprimary点推定が残る | E1・E2・E5、development218、ε=0.05、P=0 | B0 L_D5 q1はG_RZ=229,718,060、B2 L_D3 q1 r4 K2は130,774,896.65625。後者の点推定は約43.07%低い | 全prefix・高次PF・強い合成法への優位、統計的に認証された最適構成 | 図1 |
| C2 要求精度でB2内の望ましい設定が変わる | E5、development218、固定302点、P=0 | primary点最小は厳しい側からL_D3のq4 r4 K4、q2 r4 K4、q1 r4 K2へ変化。全表示点の最小methodはB2 | deterministicからpartialへのmethod逆転、q/r/Kの厳密winner、302回の独立成功 | 図2 |
| C3 元精度の固定transferと別精度の適格性は別 | E3の元ε=0.05とE5のM2元5構成 | 元TRANSFER_SUPPORTEDは保持。事後精度曲線では低ε側30点がB3、残り272点がB2のprimary点最小 | 1.30 Åでのmethod最適性、新しいheld-out試験、未登録q2/q4・B0 L_D5の推定 | 図3 |
| C4 固定qからの変化とselector損失を限定する | E4、M1同一domainのq8部分集合／全登録集合、旧16-cell selector | q8でもprimary最小はB2。旧selectorのprimary RZ regretは0だが6指標Paretoの一件を落とした | S2→M1差のqだけへの因果帰属、selectorがprimary最良点を落としたとの説明 | 本文・補足表 |
| C5 biasを減らすqが測定込み資源を必ず減らすわけではない | E4、同じL_D・K・T・R=qrの56 group | normalization・random action期待値を揃えた組でも、bias／shotの減少と1-shot費用の増加が競合する | compiled RZをdet/random/basis成分へ分解済み、q効果の一般法則 | 図4 |
| C6 状態準備感度は候補集合に依存する | E4・E6、development218とM1/M2共通5構成を分離 | 共通仮想Pの下でlower envelopeが変わる。共通5構成では両geometryとも大PでB1が入る | 実準備回路評価、異なる候補集合の差をgeometry効果だけとする説明 | 補足図 |

C2の点最小区間は、q4がε=0.005〜0.0054158189504、
q2が0.00547017101788〜0.0235054643684、
q1が0.0237413604716〜0.1。これは隣接する表示点間の切替で、連続精度の正確な根ではない（E5）。

C3ではB3の最後の表示点が0.00667938131366、B2の最初が0.00674641423837。
B2 r4の現行受理境界0.00525637654710と、費用順位切替は一致しない（E5）。

## 適格性・費用・不確かさの読み方

正式評価式は[PM-2契約](pr2_pm2_precision_resource_contract_v1.md)を保持する。
aを実部／虚部として、e=ε/√2、s_x,a=e−b_x,a、
N_x,a=ceil(2 B_x² log(2/α_a)/s_x,a²)、G_x=Σ_a N_x,a(C̄_x,a+P)である。
strictに全s_x,a>0を要求し、不適格行のN/GはnullまたはMISSINGであって0ではない。

- ε_min=√2 max_a b_x,aは「現行の対称軸配分・Hoeffding十分shot規則の適格境界」と呼ぶ。
  M2 B2 r4の総複素bias約0.003725912は0.005未満でも、現規則ではε=0.005に不適格。
  別配分の実現性・費用を検証したわけではなく、原理的精度限界とも呼ばない（E3・E5）。
- 解析shot数は実行した量子shot数ではない。biasは保存参照signalに依存する。
  α_axis=0.025はcandidate/grid全体の同時winner保証ではない。
- random平均費用は元32 paired trajectoryに依存する。Re/Im covarianceを保持した点±2SEは
  engineering intervalでありformal CIではない。family内の設定順位とmethod間の比較を分ける。
- 同じcost標本とbias/Bを302精度点で再利用している。独立試験数・成功率へ換算しない。
- B0ではdiscard＋PF総biasだけが保存され、pure discard/PFは欠測。
  q依存の非単調性は観測だが、誤差相殺の機構を実証したとは書かない（E2・E4）。
- Pは共通の仮想RZ-equivalent準備費用/shot。物理時間、T count、Clifford+T総資源、
  chemical-accuracy energy/QPE/RPE全体の資源へ換算しない。
- 既存照合では巨大shot boundの浮動小数点演算順によるceil差をchecker側で訂正した。
  binary64の保存ledgerを任意精度の厳密最少shot数とは呼ばない（E5）。

説明用にはh_x,a=s_x,a/eを使い、ceilを除くG*をB²、h^−2、軸別Cの積・和として整理できる。
これは評価式の代数的整理で、新しい理論・未知系へのcertificateではない。
正式図の費用には保存整数shotを使い、この連続式でledgerを置換しない。

## 一次文献とのclaim単位の照合

2026-10-05に一次資料を確認した。最も近いW1は本文Sec. VII.B、Fig.10、
Appendix D.2とFigs.19–20の本文・captionまで確認した。他の6件は一次公開ページのAbstractを再確認した範囲であり、
全定理・全実装・全比較条件を精査済みとはしない。versionを固定し、既存prior-art gateを結果後に再採点しない。

| 文献 | 既知として扱う内容 | 今回に残す差分と非claim |
|---|---|---|
| W1 Güntherら、Phase estimation with partially randomized time evolution、v2 | partial randomization、single-ancilla QPEの資源接続、DF fragmentの部分保持、truncation比較を既に扱う。Appendix D.2の切断比較はground-state energyとTrotter誤差を分ける。[本文](https://arxiv.org/pdf/2503.05647v2) | 固定有限signal、保存総bias、有限cutoff、整数shot、実測wrapper、登録domain／固定transferの定量的台帳に限定。「DFに適用」「discardを比較」「測定費用を入れた」だけを新規性にしない |
| W2 Hagan–Wiebe、Composite Quantum Simulations、v3 | Trotter–SuzukiとQDriftの分割、誤差・費用の理論比較。[一次資料](https://arxiv.org/abs/2206.06409v3) | hybrid分割原理の新提案ではない。channelの誤差保証をfinite-RTE amplitudeの保証と自動的に同一視しない |
| W3 Casaresら、Theory and practice of Trotter product formulas in quantum chemistry、v1 | SPRINT/GRADEによるrandomization・factorization・化学simulation資源設計。[一次資料](https://arxiv.org/abs/2606.30741v1) | 統合設計一般やfactorization法への優位を主張せず、固定二次DF-prefix実装の比較に限定 |
| W4 Oumarouら、Accelerating quantum computations of chemistry through regularized compressed double factorization、v3 | 圧縮・正則化DFと化学計算資源削減。[一次資料](https://arxiv.org/abs/2212.07957v3) | 固定prefixを新しいHamiltonian圧縮法と呼ばない。RC-DFへ再最適化した比較は未実施 |
| W5 Kanasugiら、v2 | single-ancilla Trotter QPE、partial randomization、量子化学のend-to-end資源評価を既に扱う。[一次資料](https://arxiv.org/abs/2603.22778v2) | 本研究は固定有限signalのcompiled RZ-work。化学での部分ランダム化資源評価自体を新規性にせず、物理資源taskへの換算を主張しない |
| W6 Cugini–Atif–Subasi、Resource-Optimal Importance Sampling for Randomized Quantum Algorithms、v1 | circuit実行費用とestimator varianceを考慮するimportance sampling最適化。[一次資料](https://arxiv.org/abs/2603.13495v1) | canonical samplingを固定して実装構成を比較する。「resource-optimal partial randomization」「新しいcost×shots原理」は使わない |
| W7 Simon–Love、Reduced-Cost Quantum Compilation of Controlled Time-Evolution、v1 | controlled対称Trotter回路で任意角rotationの増加を抑える合成。[一次資料](https://arxiv.org/abs/2511.13855v1) | 強い既知baseline候補。現在のDF wrapperのRZが一律半減するとは推定せず、強化合成一般への頑健性は未検証 |

今回と完全に同じ数値比較を上記で確認したわけではないが、それは不存在証明ではない。
「初めて」「世界初」は使わない。方法論論文としての十分な新規性・投稿先は未確定で、
限定された再現可能なcase studyとしての情報価値を原稿レビューで評価する。

## 原稿段階で残す確認事項と停止境界

未解決なのは、限定claimで読者に何が分かるか、図・captionが比較集合と証拠階層を守るか、
関連研究の差分を過大に書いていないかである。未知の誤差成分を埋めることを完成要件に追加しない。
全本文執筆時に文献の詳細を使うなら、その節・定理の一次資料を追加確認する。

今回行ったのは文書・保存CSV/JSON・一次資料の確認と原稿設計だけ。
新signal、trajectory、circuit build/compile、NPZ/NPY/pickle/runtime/registry、GPUへのアクセスは0。
旧result/status/manifestとTrack Bは変更しない。PM-3、追加96、別条件、strong synthesis、
高次PF、energy/RPE接続を認可しない。図生成・通し原稿は次の原稿化作業であり、今回は未実施。
