# M1-S：Stage S（整数shot・費用成立可能性）の科学条件に関する限定独立レビュー

- **レビュー日**：2026-10-17（JST）
- **レビュー開始承認**：ユーザー「レビューを開始して」。直前の提案に従い、M1-S shot・cost feasibility Decision Packetの後続として、**Stage Sの科学的条件と最小検証の選択**を検討することへの承認。
- **重要度**：Level 2（GPT-6 xhighで十分）。既に採択されたTask A→Task B研究方針の範囲で、実験設計・数理的意味・進行gateを限定して判断する。M1-S全体の新規性や研究本線の再採択を今回の範囲に含めない。
- **対象repository**：[`HIROMU1015/Partially-Randomized-Trotter`](https://github.com/HIROMU1015/Partially-Randomized-Trotter)
- **固定対象commit**：`fb5519bf3f44392dc050feb9db44057dbc5207b1`、branch `track-b-m1s-shot-cost-feasibility-preparation-20261017`。GitHub connectorで公開HEADとの一致を確認した。
- **判定**：`STAGE_S_OPTIMISTIC_INTEGER_HEADROOM_SCREEN_RECOMMENDED_NOT_ADOPTED / FIXED_GAMMA_WITHOUT_EXTERNAL_ACCURACY_NOT_JUSTIFIED / TASK_B_AND_NATIVE_RESOURCE_GO_NOT_ESTABLISHED / NEW_SCIENTIFIC_EXECUTION_NOT_AUTHORIZED / MANDATORY_STOP`。
- **実施境界**：固定文書・既存SOURCE・結果報告・未採択契約の読解、既存の集中不等式および三角関数からの**紙上の解析的検討**のみ。保存信号の新しい数値読出し・再集計・個別γへの代入・整数shot値算出・境界の数値再評価・Pythonによる科学計算・native compile・RPE simulation・科学SOURCE変更・GitHub変更・authorization発行は行っていない。

## 1. 要旨と中心判断

**Stage Sは限定的に進める価値がある。ただし、新しい外生的な目標位相精度が存在しない現在、一つのγを任意に選んで「shot数が減った／減らない」と判断するのは不適切である。**

推奨は、既存の保守的RPE角誤差上限候補 `γ<π/3` の**直下極限（cap-edge）を用いた楽観的なfeasibility screen**をStage Sの第一段階に置くことである。共有label手法と対照の十分shot設計において、(i) 連続shot費用比にどれほどの上限余地があるか、(ii) 整数化したshot数の厳密減少をcap直下の開領域で証明できるか、を分ける。この上限評価は計算した後でも実用精度、物理的RPE、native費用の勝利を示すものではない。

G3の16対照、G6・T=4の8対照を**事後的な探索対象**とする。G6・T=1の8対照は`γ*<π/3`不成立、G9の16対照は絶対visibility改善未解像として保持する。**全48対照を報告する**。24件を未使用データやheld-out集合として扱わない。

Stage Sに科学的なnative metric・compiler・HD合成精度・state preparation価格を今すぐ要求しない。これらはStage Cの条件であり、Stage Sのモデルから勝手に0を代入してはいけない。一方、**Task Bのstate、energy prior、全dyadic round、実装/SPAM位相誤差の欠落**を明示し、Stage SをRPE適用保証と呼ばない。

Stage Sの設計方針をユーザーが採択した場合だけCodexが新SOURCE・未採択実行契約の技術準備へ進む。SOURCE完成後の契約正式採択および新たな科学実行には、それぞれ別のユーザー承認が必要である。

## 2. 固定証拠と実際の取得範囲

### 2.1 本レビューの正本資料

|役割|固定資料／参照|確認範囲|
|---|---|---|
|直近Decision Packet|[`m1s_shot_cost_feasibility_decision_packet_20261017.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fb5519bf3f44392dc050feb9db44057dbc5207b1/docs/tracks/algorithm_codesign/m1s_shot_cost_feasibility_decision_packet_20261017.md)|P1–P4、S1–S3、C1–C3、Stage S/C/Bの提案を本文取得・確認|
|未採択契約候補|[`m1s_shot_cost_verification_contract_candidate_20261017.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fb5519bf3f44392dc050feb9db44057dbc5207b1/docs/tracks/algorithm_codesign/m1s_shot_cost_verification_contract_candidate_20261017.md)|D1–D7、数値engine未実装、closed guard、resource候補を確認|
|機械可読契約候補|[`verification_contract_candidate.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fb5519bf3f44392dc050feb9db44057dbc5207b1/artifacts/track_b_m1s_shot_cost_preparation/2026-10-17/v1/verification_contract_candidate.json)|`adopted=false`、`run_ready=false`、science欄全null、`future_source_identity=null`を確認|
|技術引継ぎ|[`m1s_shot_cost_source_resource_and_handoff_20261017.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fb5519bf3f44392dc050feb9db44057dbc5207b1/docs/tracks/algorithm_codesign/m1s_shot_cost_source_resource_and_handoff_20261017.md)|既存SOURCE再利用可否・新SOURCE未実装・資源未採択を確認|
|未決科学事項|[`pending_scientific_decisions.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fb5519bf3f44392dc050feb9db44057dbc5207b1/artifacts/track_b_m1s_shot_cost_preparation/2026-10-17/v1/pending_scientific_decisions.json)|D1–D7がすべて未採択であることを確認|
|π/3限定gate結果|[`m1s_gamma_pi3_completed_results_and_stop_20261017.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81d6103ec89d485cb7336c3a90bd95dad66992aa/docs/tracks/algorithm_codesign/m1s_gamma_pi3_completed_results_and_stop_20261017.md)|既存分類・適用限界・SOURCE/contract/authorityとSTOPを確認|
|γ依存V1科学レビュー|[`m1s_gamma_v1_completed_results_scientific_review_20261017.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/docs/research/m1s_gamma_v1_completed_results_scientific_review_20261017.md)|保守的shot設計の数式、γ境界、decoderとの接続制限を確認|
|前回のRPE Decision Packet|[`m1s_rpe_applicability_and_total_cost_decision_packet_20261017.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/870af3f65e3e31e19d2b002c556216c7340c06f0/docs/tracks/algorithm_codesign/m1s_rpe_applicability_and_total_cost_decision_packet_20261017.md)|Task A/B、raw直接route、state/implementation、shot・費用条件を前提として参照|
|共通運用指示|[`research_project_instructions_unified_2026-10-16.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fb5519bf3f44392dc050feb9db44057dbc5207b1/docs/research/research_project_instructions_unified_2026-10-16.md)|本文の§§0–12をGitHubから取得・読解|

直近のbranchの公開HEADは上記固定commitと一致した。公開receiptは26 payloadファイルのbytes同一性を記録し、最終receipt commitを含む報告は公開27ファイルとしている。本レビューが全27ファイルを独立再hash・テスト再実行したという意味ではない。

### 2.2 保存された科学証拠・固定したスコープ

- 線形H4、隣接距離1.00 Å、STO-3G、8 spin orbitals、保存DF rank12。G3/G6/G9、`δ={1/8,1/16} Ha^-1`、`R={2,4}`、`T={1,4} Ha^-1`。固定された72非zero方式行、36zero control、48対照。
- 比較対象はshared ±、ordinary、independent ±の**同じ概念的label数**。branch間で同じlabelを共有するshared構成の負枝は同じ順序のsign reversalであり、full-word adjointではない。native回路費用の同一性を主張しない。
- γ依存V1の数学的結果は、G3/G6の32対照でsharedが**小さいγでは不利・大きいγでは有利**という保守的連続marginのtrade-off境界、G9で絶対差未解像というもの。
- π/3限定gateはG3の16件とG6・T=4の8件で`SATISFIED`、G6・T=1の8件で`NOT_SATISFIED`、G9の16件は`NOT_TESTED`。成立は`γ*<π/3`の開区間の交差のみを示す。
- V1の元信号のevidenceは`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`。条件付きIEEE/BLAS/trigonometric allowancesと保存oracleの非認証性が残る。
- π/3限定gateの固定結果はSOURCE S=`2f800f179c7b29589b2f9a7e532408b16b8fe6d2`、契約 C=`bf97fcb2607904f8b96a625c5b64440a78bbbfdc`、authority A=`a1dd87a1f81467bea0ce27cacf9d21e49cff7908`、結果commit `81d6103ec89d485cb7336c3a90bd95dad66992aa`。新しい契約へそれらの消費済markerを転用しない。

**資料検証の限界**：元H4 Hamiltonian/複素信号/γ*を異なる数値経路で独立再生成していない。今回のStage Sの各候補について新しい科学数値を計算・代入していない。Codexの静的検査116 PASSは数学的結果や実機動作の独立認証ではない。

## 3. 結論を左右する意味論：Task AとTask B

**Task A**は保存raw-H理想coherent平均信号の各単一時刻の円周位相を対象とする。保存した方式ごとの総raw-H phase bias `b_{m,raw}^+` と、保存数値のvisibility下界 `A^-_m` から十分shot設計を条件付きに構成する。raw参照からraw目標への差はTask Aの定義上ゼロである。ただし、これは保存oracle由来の**事後的情報**であり、実用の量子shot scheduleで無料利用できる保証ではない。

**Task B**は対象固有エネルギー`E_*`の多round RPE。`arg z_raw(T)` が `E_* T` に一致すること、guiding-stateの対象固有state overlap、gapやenergy prior、alias-free branchの初期化、全dyadic round、状態準備・実装/SPAM/synthesis誤差、全round confidenceが必要である。保存stateの正規化と小さいsolver残差だけでは、これらを認定できない。保存時間T=1,4のみを完全なRPE scheduleと見なさない。

`π/3`は前回検討した保守的nearest-consistent-branch routeの**一round総角誤差の上限候補**であり、実際の外生的要求γでも、一般のRPE全方式に対する必要条件でもない。**γ→π/3^-の評価では、追加のstate/implementation誤差へ使用できる余裕を保証しない。**

## 4. 十分shot設計の独立数式検討

同じcellで方式`m`ごとの条件付き保存信号下界と総raw参照位相bias上界を

\[
 A_m^- = \max\{0,\widehat A_m-r_{A,m}\},\qquad
 b_m^+=\min\{\pi,|\widehat b_m|+r_{b,m}\}
\tag{1}
\]

とする。保存`phase_bias_rad`は`TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H`である。`H_tilde-H_full`モデル差は既に含まれ、別途`η=|T|B`を加算しない。

単一方式行の理想化された±1のX/Y Hadamard結果が、対応する理想複素信号のRe/Imを平均として持ち、shotごとにfresh-IIDとする。`A_m^->0`、`0<b_m^+<γ<π/2`なら、円周位相に残る統計角予算は`q_m=γ-b_m^+`である。Hoeffdingの二側不等式・複素diskの幾何による一軸の十分shot数は

\[
 M_m(\gamma)=A_m^-\sin(\gamma-b_m^+),\quad
 S_m(\gamma)=\frac{K}{M_m(\gamma)^2},\quad
 n_m(\gamma)=\lceil S_m(\gamma)\rceil,\quad
 K=4\ln(2/\alpha_{\mathrm{axis}}).
\tag{2}
\]

`PER_METHOD_ROW`を選ぶ場合は`α_axis=α_row/2`、X/Yの両軸それぞれ`n_m`、合計`N_m=2 n_m`。例えば**旧V1で採択された**`α_row=0.05`なら`K=4 ln(4/0.05)`という**代数的な置換**になる（今回Kの数値評価はしない）。このα値とscopeはStage Sの新契約には**自動継承されない**。

`n_s<n_c`は同じ**十分設計**の厳密な整数shot節約であり、実機での最小必要shotsに関する下界でも最適性の証明でもない。等しい`n`は`INTEGER_TIE`。仮定のない物理IIDや実機の平均信号をこの結果から認定しない。

## 5. 独立の紙上結果：連続shot利得比はγに対して単調増加

この結果は今回のレビューで既知の三角関数を**記号的に整理した推論**であり、固定H4値への新しい代入・fitではない。

G3/G6で記録された方向のtrade-offを、一般的な二方式の仮定

\[
 a=A_s^->c=A_c^->0,\qquad 0\le v=b_c^+<u=b_s^+<\pi/3
\tag{3}
\]

として記述する。添字`s`はshared、`c`は対照である。γ依存V1が示した一意の連続margin交点を`γ_*`とし、`γ_*<π/3`が成立すると仮定する。`γ∈(γ_*,π/3)`では`M_s>M_c>0`。

連続ショット予算の**対照／shared比**を

\[
 R_{\mathrm{cont}}(\gamma)
 :=\frac{S_c(\gamma)}{S_s(\gamma)}
 =\left[\frac{a\sin(\gamma-u)}{c\sin(\gamma-v)}\right]^2
\tag{4}
\]

と定義する。`γ>u>v`かつ`γ<π/3`では両sinの引数は`(0,π/2)`に入る。従って

\[
 \frac{d}{d\gamma}\ln R_{\mathrm{cont}}(\gamma)
 =2\bigl[\cot(\gamma-u)-\cot(\gamma-v)\bigr] >0.
\tag{5}
\]

不等号はcotが`(0,π)`で厳密単調減少することから従う。つまり、**この直接trade-offの仮定下では、連続十分shot比は共有有利領域でγを緩くするほど増加する**。したがって`γ<π/3`での最も楽観的な連続shot比の上限は

\[
 \boxed{R_{\mathrm{cont}}^{\mathrm{cap}}
 =\lim_{\gamma\uparrow\pi/3} R_{\mathrm{cont}}(\gamma)
 =\left[\frac{a\sin(\pi/3-u)}{c\sin(\pi/3-v)}\right]^2.}
\tag{6}
\]

この値は開領域内で一般に**達成されないsupremum**であり、RPEの実用誤差予算を満たした回路の実績ではない。`γ_*<π/3`だけでは`R_cont^cap`が資源に意味のある大きさかは分からない。将来の数値評価によって上限の大きさを把握することが、単なるgate再反復よりも有用である。

**Native費用との限定関係**：一時刻、X/Y同一の一shot費用、固定費差なし、整数丸めなしとする連続proxyでは、sharedの追加一shot費用比`c_s/c_c`が`R_cont(γ)`未満でなければ、このproxyでshot節約が追加費用を埋め合わせない。従って`R_cont^cap`は、**そのモデルで許容できる費用比の最も楽観的な上限**である。実際のnative比・状態準備費・固定費は現在未知であり、この条件に数値代入することはできない。

逆trade-offやdominanceへ(5)を無条件適用しない。Stage Sの有資格24件では保存された向きが`SHARED_LOWER`→`SHARED_HIGHER`である点を、今後のSOURCEで必ず検証する。

## 6. 整数ceil：cap-edgeの存在証拠とその非対称性

`n_m(γ)=ceil S_m(γ)`における`S_m(γ)`は、固定された正のA・bのもとでは`γ`とともに**厳密減少**する。開端点`γ_c=π/3`で、`S_m(γ_c)`が有限であれば、片側極限は

\[
 \boxed{n_m^{\mathrm{cap}-}
 :=\lim_{\gamma\uparrow\gamma_c}\lceil S_m(\gamma)\rceil
 =\lfloor S_m(\gamma_c)\rfloor+1.}
\tag{7}
\]

特に`S_m(γ_c)`がちょうど整数なら、`ceil S_m(γ_c)`をそのまま代入すると片側極限を**1 shot過小評価**する。これは開端点の厳密性に由来し、今回の設計で重要な数学的注意点である。今後の計算は採択した有理数区間enclosureでこの境界を判別し、整数境界を挟むときは`ARITHMETIC_UNRESOLVED`とすべきである。

もし`n_s^{cap-}<n_c^{cap-}`を厳密認証できれば、連続性とceilの片側局所一定性により、**π/3より厳密に小さいγの非空開領域で整数shot削減が存在する**。これはその領域の存在についての十分証拠となる。

しかし逆に`n_s^{cap-}=n_c^{cap-}`であっても、さらに小さいγの一部で整数ceilが異なる可能性は排除できない。`n_s^{cap-}\ge n_c^{cap-}`という端点のみの結果を`NO_INTEGER_SAVING_ANYWHERE`と呼んではならない。端点評価を採用するなら、判定は`CAP_EDGE_STRICT_INTEGER_WITNESS`／`CAP_EDGE_NO_WITNESS_GLOBAL_UNDETERMINED`／`ARITHMETIC_UNRESOLVED`等に限定する。

従って、**cap-edgeでの陽性は存在の十分条件、陰性は全領域における不可能性の証明ではない**。全γ領域の必要十分分類を希望するなら、別途、完全性が証明されたinterval subdivision／整数閾値列挙と有限resource capを採択する必要がある。今回そこまでを既定計画に含めない。

## 7. 何が実際のRPEを制約するか

Stage Sで用いるγは保存raw理想位相のTask A予算である。Task Bの方法別総位相budgetは、少なくとも

\[
 b_{m,\mathrm{total}}^+
 \le b_{m,\mathrm{raw}}^+ + b_{\mathrm{raw}\to E_*T}^+
 + b_{m,\mathrm{state/impl/SPAM}}^+ ,\qquad
 b_{m,\mathrm{total}}^+ + q_{m,\mathrm{physical}}\le\gamma_k<\pi/3
\tag{8}
\]

を伴う。`b_raw→E_*T`は保存stateの目的固有値への相対位相であり、正規化された状態を厳密基底状態と扱わない。実装誤差やstate差は二重計上しない定義を契約で明示する。未知項を0と置かない。さらに全roundの初期alias/energy prior、最終energy誤差条件`γ_K/T_K≤ε_E`が必要である。

**Stage Sのπ/3上限評価は、上記未知の正の誤差が存在する場合の利用可能位相領域を過大評価する楽観的screening**である。したがってStage Sで成立しても、次はphysical budgetやstate/priorの証拠を調達しなければStage C／Task Bの資源GOに近づかない。

もし将来、両方式に共通の追加位相上界`d`を正当化できるなら、形式的には利用可能なraw理想角上限が`γ<π/3-d`へ縮む。しかし**追加誤差が方式依存なら各方式のbiasへ別々に加算する必要がある**ため、同じdを仮定することは一般には許されない。今dの値を選ばない。

## 8. D1–D7：科学条件の推奨と明示的な未採択

|ID|今回の科学的判断（**推奨・未採択**）|Stage Sでの扱い|
|---|---|---|
|D1 位相要求|**任意の固定γや結果依存γ gridを選ばない。** 第一候補は`0<γ<π/3`の楽観的cap-edge headroomと厳密整数ceilの片側判定。これは外生的RPE精度を代替しない|Stage S診断の評価policyをユーザーが正式採択する必要あり|
|D2 confidence|旧V1と整合する`PER_METHOD_ROW`、`α_row=0.05`、fresh-IID/Hoeffding/X/Y等配分を**継続候補として推奨**。全72行同時95%ではない|次期契約で改めて明示採択。比較の両方式に同一policy|
|D3 位相・state|Stage Sは**保存raw-H理想Task A**に限定。`b_raw→*`はTask Aの定義上0、実機/state/HD/SPAM保証は**範囲外・未取得**。Task Bへ拡大しない|欠けたstate情報をTask Aのgate失敗にしてはならない|
|D4 coverage|既存半径の`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`を保持し、`SAVED_CONDITIONAL_RADII_IDEAL_MEAN_ONLY`をStage Sの候補とする。新しい区間算術は旧pilotのplatform条件を認証しない|用途を限定した再採択が必要|
|D5 費用metric|Stage Sではnative metricを選択・数値化しない。`R_cont^cap`は単一shot費用比の記号的な**楽観的必要headroom proxy**のみ|Stage Cまで費用指標の最終採択を延期|
|D6 native実装|M1-Sのcontrolled/shared branch、HD、state準備、identity相対位相・符号の費用/誤差は未取得。既存DF/RTE価格を流用しない|Stage Cの新SOURCE・実施契約で判断|
|D7 比較・stage|Stage S-optimisticのみ。全48件を保持し、24件の事後的eligibility、8件既知不成立、16件G9未解像を分離。no scientific GO、retry/threshold変更禁止|得られたscreen結果に基づく継続／縮小判断はユーザーへ戻す|

`p=128 bits`、CPU1/GPUなし、aggregate memory2GiBなどは既存SOURCEに由来する**技術上の開始候補**であって、新Stage Sの正式値ではない。γ-gridなしのcap-edge screenなら評価callの構造は比較的限定できるが、実装・数値correctness・wall/CPU・call capについては採択された方法の下でCodexが技術設計する。新契約ではCPU／wall time capを過去値から自動転用しない。

## 9. Stage Sの最小検証仕様案

### 9.1 目的

「保存H4のTask A理想信号を前提とした、π/3以内で得られ得る**最大連続shot費用headroom**と、cap直下での**厳密整数shot節約の存在証拠**はどの程度か」を問う。実用精度での最適ショット数や総回路価格は問わない。

### 9.2 入力とregistry

1. 既存のSOURCE/input/result・SHA256をSOURCE versionごとに結び付ける。既存V1やπ/3を再実行しない。
2. G3 16件＋G6 T4 8件の計24件は**探索的なcap screen候補**。G6 T1の8件はpi3-negative carry、G9の16件はabsolute-gain-unresolved carry。全48件と72方式行・zero36件のidentityを報告する。
3. 元signalをoracle真値や物理実測・独立holdoutと呼ばない。保存round-trip binary64・既存半径の条件付きcoverageを維持する。

### 9.3 採択後の出力候補

- 全件の`old_pi3_gate_status`、`old_absolute_contrast_status`、`new_stage_s_status`を**別列**に保存。
- 有資格24件について、保守的`A_s^-,A_c^-,u,v`の入力由来・適用domain、区間としての`R_cont^cap`、`n_s^{cap-},n_c^{cap-}`の区間または確定整数、cap-edgeのstrict削減証拠または未解像を保存。
- 結果表示は正確な有理数区間を正本とし、decimalは表示用のみ。cap-open端点のfloor+1を採用。厳密整数条件不成立を全γの不成立へ読み替えない。
- G3/G6の全対象を報告し、最良cellを選んでも**development上の探索例**と明記。G9の絶対未解像を形式的shot比で救済しない。
- 新たな数値結果への一致で精度・閾値・探索回数・対象cellを調整せず、条件未解像はそのまま保存。`GO`は常に未発行。

### 9.4 成功／不十分の判定

|Stage Sの観測候補|支持できる判断|支持できない判断|
|---|---|---|
|capの連続headroomが小さい|今回の理想化された費用比で使える余裕が小さいとの記述|同一条件外の手法全般のNO-GO、native実費のNO-GO|
|cap-edge strict整数shot減少の区間証拠|π/3直下の**ある非空領域**に十分shot設計の整数削減が存在|実際のγ要求で削減、Task B/RPE成功、総資源GO|
|cap-edgeでtie／no witness|当該片側極限では削減証拠を得られない|より小さなγの全領域でも削減できないという結論|
|算術・coverage未解像|事前指定したpolicyでは判断を出せない|性能差なし、物理的negative|
|state/prior/native費用情報欠落|Task BやStage CのGOを保留|Task A理想shot設計の完全な不成立|

Stage Cへ進むための自動GOは発行しない。特に`R_cont^cap>1`だけでは、実装に必要なshot単価の上昇や状態準備費用を補えない。次の段階ではRPE位相誤差budgetの成立可能性と実費の粗い下限・上限が別途必要である。

## 10. 他のStage S設計との比較

**案A：外生的な単一γに対する整数shot評価。** 外部から研究目的上の要求位相精度が明示された場合は直接的であり、将来最も意味がある。しかし現状にその数値はなく、観測済み境界から「勝つγ」を選ぶのは事後最適化となる。今は第一候補にしない。

**案B：γの有限grid／profileを事前登録する。** 全48件の固定profile比較としては可能だが、何radが意味あるか不明ならgridを選ぶ根拠が弱い。計算してから都合のよい点のみ報告する危険もある。定量感度分析として将来選択可能だが、今回の最小検証より優先しない。

**案C：開区間内の厳密整数削減の有無を完全分類する。** 認証済みinterval subdivisionや整数閾値列挙で原理的に検討できるが、有限完了性とresourceの証明、整数境界近傍の精度問題が追加で必要になる。cap-edgeに陽性があれば既に存在証拠が得られるので、最初から完全解探索を要求しない。

**案D：Task Bへ直接進みRPE budgetとnative compileを全て試す。** 保存stateのspectral保証、全round、physical wrapper平均、実装費用が欠けており不推奨。特に古い単位半径/effective eigenstateの会計をM1-Sへ無条件流用しない。

**推奨は、案A/B/Cを既定にせず、短いcap-edgeの楽観的headroomと整数片側証拠を取得する限定Stage Sである。** その結果が実用精度の必要性を代替するとは認めない。

## 11. 費用モデル・baselineに関する今後の選択

同じTask/γ/confidenceを満たすscheduleについて、metric`g`ごとに

\[
G_m^{(g)}=F_m^{(g)}+\sum_{k,a}N_{m,k,a}\left(C_{m,k,a}^{(g)}+P_{m,k,a}^{(g)}\right).
\tag{9}
\]

- `C`：shared/ordinary/independentの全control・directional Pauli tail・HD近似・ancilla・readoutの一shot native費用。SOURCEが存在するだけで価格が分かるわけではない。
- `P`：state preparationの同品質・同用途での各shot費用。同じ準備回路でもshot数が違えば相殺されない。
- `F`：較正、bias情報の入手、classical preparation等の固定費。単位変換できない費用は別列。

Stage Sの連続費用headroomは(9)の代用ではない。量子T、CX、RZ、depth、peak workspaceを恣意的な単一scalarに足さない。強いbaselineは同じlawのordinary/independentに加え、将来のresource claimでは既知の部分乱択RTEやdeterministic endpointと公平に比較する必要がある。

Stage Cの最初の重要課題は、**RPEで使える位相精度の現実的上限と、各方式の追加費用の大まかな区間**である。Stage Sが楽観的であることを忘れてfull native benchmarkに直行しない。

## 12. 新規性、既存理論と研究成果の位置付け

Hoeffdingでの十分shot数、trigonometricな交点、ceilと連続予算の差は、それ自体では先行研究にない統計手法・新しいRPE保証ではない。Stage Sは既存手法の実用資源性を**スクリーニングするための内部意思決定用解析**である。

M1-Sに独立した研究貢献があるとすれば、gate labelを二branchで共有するcoherent平均の構造、T1a/T1bの理論と適用域、有限stepにおけるvisibility増大とphase bias増大、固定H4での位相–visibility trade-offという従来結果の範囲にある。これらの先行研究上のpriority・論文十分性は別途調査中であり、今回評価・認定しない。

資源改善がStage Sの探索的上限評価で支持されなくても、有限step信号と誤差構造の理論・数値ノートとして残す選択は妥当である。ただし、Stage Sの不成立だけでM1-S一般のNO-GOにはしない。

## 13. 次の担当、承認、mandatory STOP

**科学的推奨（未採択）**：Stage Sを**Task A理想保存信号に限定したcap-edgeの楽観的整数shot・連続費用headroom screen**として設計する。`PER_METHOD_ROW`・`α_row=0.05`・fresh-IID・Hoeffding・X/Y等配分および条件付きcoverageは旧V1との比較可能性のため継承を推奨するが、次期契約では明示採択が必要。位相γの数値一点やγ grid、physical state、native cost metric、Stage C実行は採択しない。

1. **ユーザー**：このStage S設計・統計/coverage条件の採否を判断。
2. **Codex（採択後）**：採択されたStage Sだけの新SOURCE・保存interval adapter・open-cap厳密算術・結果schema・guard・有限resource cap・登録外技術検査を準備。固定Sと新契約Cの候補を公開しmandatory STOP。
3. **ユーザー**：SOURCEを結合した正式契約の採否を判断。
4. **ユーザー**：数値評価を行う場合、別途source-boundの一回実行承認。
5. **Codex（実行承認後）**：one-shot実施、既存の全48 registry・成功/未解像・prefix・provenance・失敗を保存、公開同一性確認後mandatory STOP。
6. **GPTまたはユーザー（結果後）**：Stage Cの実費照合やTask Bに必要な証拠への投資価値を、**既存の結果と必要情報に応じて**判断。自動レビューや自動実行は行わない。

現時点の新Stage S正式採択、新科学実行、SOURCEの変更、承認記録の書き込みは行っていない。既存SOURCE・raw/CSV・契約・authorization・grant・ledger・消費済marker、親・他worktreeは変更していない。**mandatory STOP維持。**

## 14. 参考文献・一次資料の確認範囲

- W. Hoeffding, *Probability Inequalities for Sums of Bounded Random Variables*, Journal of the American Statistical Association **58** (1963), DOI [`10.1080/01621459.1963.10500830`](https://doi.org/10.1080/01621459.1963.10500830)。今回の±1二側式は標準の集中不等式の特殊化であり、独立した統計的革新ではない。
- S. Kimmel, G. H. Low, T. J. Yoder, *Robust Calibration of a Universal Single-Qubit Gate-Set via Robust Phase Estimation*, Physical Review A **92**, 062315 (2015), DOI [`10.1103/PhysRevA.92.062315`](https://doi.org/10.1103/PhysRevA.92.062315)、および2021年の[Erratum](https://doi.org/10.1103/PhysRevA.104.069901)。今回π/3は**既存の研究契約による保守的候補**を再利用したもので、全定理の再証明・一般的保証の新たな認定ではない。
- J. Günther *et al.*, *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum **7**, 020332（2026年5月19日公開）、DOI [`10.1103/ynxb-p2xq`](https://doi.org/10.1103/ynxb-p2xq)。公式abstract・書誌および既存研究資料を参照。先行の部分乱択によるphase estimation/resource見積りと新たなM1-Sの比較には、対象task、oracle情報、強いbaseline、native費用scopeを合わせる必要がある。本レビューで同論文の全数学を再導出・priority監査したものではない。

---

**最終判断**：`STAGE_S_CONDITIONAL_OPTIMISTIC_CAP_EDGE_DESIGN_RECOMMENDED_NOT_ADOPTED`。完全なRPE成功・実用資源改善は未確立。詳細科学レビューは完了したが、**実験契約・新計算実行は承認していない**。
