# H4非DF/DF：条件付きC/N/G比較の実行計画

2026-10-24 JST。ユーザーの今回の指示§1により、**固定4方式の数値的に検証した条件付きC/N/G比較を主評価とする方針が正式採択された。** 旧compiled program全体の形式certificateを比較開始の必須条件にしない。厳密な部分証拠、仮定・経験的評価・未解決誤差を区別し、元の契約・結果を保持する。

今回は**計画・静的照合・人工検証のみ**。新しい実H4のN/G・順位・数値集計・感度曲線・sampling/compile・科学grantはすべて0。専用branch `codex/non-df-h4-conditional-cng-plan-20261024`、基点`934f3d09db03ab9efbdad7da32eaaf432b6eb1ac`。成果物はこの計画、[非実行契約](../artifacts/non_df_h4_conditional_cng_plan/2026-10-24/execution_contract_DRAFT.json)、[40軸の条件・入力pointer](../artifacts/non_df_h4_conditional_cng_plan/2026-10-24/axis_conditions.json)を中心とし、科学計算用の汎用基盤や新しいcertificate checkerは作らない。Mandatory STOP。

## 1. 固定対象・到達可能な比較

Linear H4、隣接1.00 Å、STO-3G、8 system qubit、共通の保存stateの同じray、S2/T=.8/q8、PR r4/K4。Pauli-JW184項・PR prefix27、DF-native tol-only1e-8/rank10・PR prefix1＋one-body。Pauli-JW PF/PR、DF-native PF/PRの4方式×ε=.05/.01/.005/.001/.0005×cosine/sine＝40軸。Compilerは元のQiskit1.3.0/rz,sx,x,cx/opt1/seed17、費用は状態準備を除くfull Hadamard wrapperのcompiled RZ数。

4方式は異なる分割・保持規則・回路実装を含む固定構成である。最適化済み方式全体の優劣、純粋なDF近似効果、全QPE/エネルギー精度、T/Toffoli費用へ一般化しない。H2高次PFや別DF cohortを混ぜない。

既存証拠から、次回は**数値誤差と実装残差に条件付けたshot/G関数、PR/PFの比較が成立する残差・費用tailの領域**を作成できる。現在のデータだけで実残差の小ささやPR母平均費用の厳密上界を確定するわけではない。無条件の適格性や最終順位は依然未確定である。

## 2. 既存資料・コード・保全

GitHub固定commitから27資料の全bytesを取得し、Git blob SHAとローカルbytesを照合した。[取得記録](../artifacts/non_df_h4_conditional_cng_plan/2026-10-24/github_verification.json)。仮に探索した`run01/plan.json`は404で、実際の`run01/contract.json`にprecision/numerical/MC設定が存在することを確認した。数値入力の再作成や設定推測はしない。

元H4 SOURCE`0c329b30a81accb35c212be8defa66fee0dd0f12`、C`e80d7510880f8db993b777c627bb8e95b05a070d`、L`eb195bf2dedd0aed9d80ad7f84948b59ea53d854`、C/L技術接続`0880f4af98b125d62599e970db89fc33634d1b61`を維持。[入力550 fileのSHAとrole](../artifacts/non_df_h4_conditional_cng_plan/2026-10-24/inputs.json)には514 trajectoryの既存paired price_joinを含める。大量の旧runtime・schemaを新しく重複封印せず、次回が使う入力を特定した。

| 再利用対象 | 次回の用途と適合性 | 今回の実施 |
|---|---|---|
| fixed_signal_accounting.fixed_signal_shots | 同じaxis配分・Hoeffding式・finite/large-N処理。EMPIRICAL条件付き評価を許す | 人工値のみで呼出し |
| h2_non_df_pilot.precision_accounting | H4からも利用されるu/headroom≤.01 guard、paired weighted価格によるG | 関数ASTをそのまま人工入力へ使用。H2分子処理は呼ばない |
| fixed_signal_evaluation._statistics | 同じ標本平均・SE・nominal Student-t95、整数費用のexact mean | 関数ASTをそのまま人工入力へ使用 |
| h4_audited_evaluation / h4_pilot_science | 元の価格・bias・normalization・precisionの保存規約 | 静的読解。evaluate入口は再利用しない：chemistry/reference/sampling/compileが起動するため |
| rte_event_audit.verify_joinと既存記録 | trajectory ID・event hash・価格・lawの対応 | 保存514本の構造対応を確認。次回も既存join規則を薄いadapterから再利用 |
| C40軸結果・normalization | encoded正規化biasと旧signed biasの距離、finite implemented law | endpoint文字列の場所とcase/axisを照合。大小・加減乗算なし |
| L296局所/11basis/324supportと一様上界 | 局所保証のscope説明・残差を分解できる場合の既存証拠 | CSV・worker結果を参照。再計算もCとの数値結合もしない |
| C/L proof adapter・K調査 | normalized-ray/phase/control/scale/compilerの未解決事項 | 既存状態を接続。全compiler proofや435件診断を前提にしない |

[政策・レビューの受領SHA](../artifacts/non_df_h4_conditional_cng_plan/2026-10-24/policy_receipt.json)。最新版共通指示の§8.4/§10を適用した。手元の2026-10-23版には§10.3の見出しがないが、今回のユーザー指示と提示レビューに「研究成果優先・過剰監査を避ける」規則が明示されているため、その規則に沿って進めた。これを追加の科学実行承認とは解釈しない。

## 3. 測定量・誤差会計を一つに揃える

保存stateを改変せず、n=〈ψ|ψ〉>0、φ=ψ/√nとする。各表現自身のfull-Hに対する正規化referenceを目標信号とし、有限K4の実装確率law・B_savedを使うC encoded meanを対応させる。無限RTEの不偏性やexact tail exponentialへ置換しない。

軸aの旧signed biasをb⁰_a、Cに保存された`axes[i].bias_distance.upper`をu_enc,aとする。Cの距離はsame-ray physical biasと旧b⁰_aを比較しているため、これを**保存された独立項として直接利用**する。C aggregate totalからsemanticを引く必要はない。

未保証の全実装転送を、測定期待値の単位で

`rho_ca >= |m_actual_compiled,ca - m_C_normalized,ca|`, `rho_ca >= 0`

という明示変数にする。Encoded/literal/compiler、phase/control、B尺度対応、同じray・実際のpreparationとの未保証差を必要に応じてこの残差へ含める。固定の実測値・推定値・厳密上界が得られたとは扱わない。既定値は`null`であり、0ではない。

条件付きの会計候補は

`u_a(rho)=u_enc,a+rho_a`、`beta_a=|b⁰_a|+u_a`、`h_a=epsilon_a-beta_a`。

Cのsemantic allowanceやLのoperator上界を自動的に加えない。Operatorとmeasurement、raw stateとnormalized rayを混同しない。旧C semanticのn係数の適用問題は残し、今回その安全性を仮定しない。

Lを使ってrhoを狭める場合は、C collapsed meanとevent familyの辺、両control branch、law、norm増幅、scaleを証明して初めて、`B_saved*delta_operator`等へ接続できる。条件が不足する間も、rhoをパラメータとした比較は可能である。C/Lの厳密結合はCOMBINATION_BLOCKEDのまま保持する。

## 4. N・guard・B scale・Gの計算設計

Axis配分は旧条件のまま`epsilon_a=epsilon_sig/sqrt(2)`、`alpha_a=.025`。一つの方式・精度に対する複素信号failure budgetは.05であり、全20条件の同時95%保証を意味しない。

同じ独立fresh shot・IID uniform wordsの条件付き仮定で、固定B_upが実際の±B_savedを覆いh_a>0なら、

`N_a(rho)=ceil[2*B_up^2*log(2/alpha_a)/h_a(rho)^2]`

を既存`fixed_signal_shots`を基準に評価する。Evidence_kindはEMPIRICAL/conditional diagnosticとして保持し、C包含を参照しても全実装certificateへ格上げしない。`certificate_verified=false`自体は元のN/G nullの直接条件ではない。

**追加guardも固定する：`u_a/h_a <= g`, g=.01。** s_a=epsilon_a−|b⁰_a|と置くと、s_a>0での許容残差は

`rho_a <= g*s_a/(1+g)-u_enc,a`。

この式は解析設計として示したもので、実H4値はまだ代入していない。右辺が負なら非負rhoの許容域が空、0ならrho=0という仮定的境界だけとなる。これを物理方式一般の不可能性と呼ばない。旧proxyのguard判定・旧nullと新評価版の判定を別に保存する。ceil前のN増加率やbreak-even式だけからceil後の順位を確定しない。

### B_savedとshot upper

二値記録は定義により±B_savedだが、これはencoded event norm1や測定平均の一致を保証しない。Nominal normalization、保存B、旧log_B_upperは区別する。

次回の新評価版では、保存Bを変えず、shot用の保守的尺度を

`B_up=max(1,B_saved,exp(saved_log_B_upper))`

とする設計を承認対象に含める。元のB/log upperを書き換えず、新版が使う実効upperを別列にする。新しいlog/exp/ceilのscalar計算で丸めが十分条件を破らないよう、exact saved入力と外向きscalar検査で確認する。既存float evaluatorとの差は別列に記録し、必要なら新しい条件付きNをscalar ceilingへ上向きにする。証明できない・元のsafe integer域を超える場合はsymbolic/nullと理由を返す。これも新解析のため今回は計算しない。

### Paired費用とG

保存trajectory iの2軸価格をC_i,c/C_i,sとして、Nが固定された条件で

`Z_i=N_c*C_i,c+N_s*C_i,s`、`Ghat=mean(Z_i)`。

PRでは256本、PFでは決定論的1本をそのまま使う。費用標本数256は量子shot数Nではない。`_statistics(Z_i)`を再利用し、exact整数平均の恒等式を確認する。PR母平均に対する期待総費用は`G_expected=sum_a N_a*mu_C,a`であり、Ghatはその有限MC推定である。

Paired covarianceを保つSEは`SE(Ghat)^2 = N^T*S_C*N/256`。軸別SEを独立と仮定して足さない。Nominal t95は実現標本分散とIID費用trajectoryの仮定に依存し、未観測tail・厳密母平均upper・実現総費用worst-case・全比較の同時coverageを保証しない。残差やscenarioを選んだ後のintervalも、その固定入力に条件付けた診断と明記する。

## 5. PR費用tail：追加samplingなしで何を調べるか

既存event/price joinを使い、観測order/componentとfull-wrapper価格の対応を再解析する。今回は新しいorder数・平均・共分散・母平均を集計していない。

次回はtrajectory単位で、(1)order4を一つ以上含む、(2)order4なしで過去未観測component集合Uを含む、(3)残り、という排他的な説明用stratumを作る。重複eventの確率・費用を二重に足さない。Uは同じ標本から選ばれた集合であり、その質量はdescriptive診断であって、選択補正のない新しい厳密信頼保証には使わない。

固定implemented IID lawのeffective order massをpi_k、Uのcomponent massをm_U、M=q*rとした場合、対応が確認できれば

`p_order4=1-(1-pi_4)^M`

`p_no4_withU=(1-pi_4)^M-[sum_{k in {0,2}} pi_k*(1-m_U)^(k+1)]^M`

を使える。Order nではproduct n個＋rotation1個という固定法則に依存する。Nominal確率とCに保存されたeffective grid massesを別にし、event側との同一law joinが不成立なら確率を補完せず未解決とする。PCG64 replayや無条件独立性の新証明は要求しない。

母平均はstratumごとの`mu_C=sum_h p_h*mu_h`で表す。未観測mu_hは0にせず未知変数とする。真正のfull-wrapper費用範囲[L_h,U_h]が既存証拠から保証できれば寄与範囲を出す。観測最大値・有限eventテスト・primitive費用の単純加算をcompiler後のU_hとして使わない。範囲がなければ、PR/PF順位を変えるために必要なmu_hの閾値を報告し、母平均順位は条件付き/判定不能として残す。

追加sampling/compileは、残差を考慮しても「このtailの寄与が比較結論を変え得る」という研究判断が出た場合に限り別案にする。Predeclared stratumと必要なwhole-wrapper情報、要求SE/判断幅、最小標本設計を示して別途承認を受ける。単に256本だから増やす、order4未観測だから一律に追加する、という既定方針にしない。

## 6. 次回に作る比較結果と感度

次回は一つの有限packageで以下を作成し、科学的結果が出た時点でSTOPする。

| 単位 | 計画した範囲 | 意味 |
|---|---|---|
| 入力・base error/scale | 40軸を一度照合、2PRのscaleを共通利用 | C40軸・Lを再計算しない |
| 回路単価 | PF2本＋PR各256、計514価格joinを一度集計 | Cの差とpaired uncertainty |
| 条件付きN/G関数 | 4方式×5精度＝20条件 | rhoを明示した関数・許容域 |
| 表示用scenario | 各条件θ=0,1/4,1/2,3/4,1の5点、計100行/200軸評価 | rho_case=θ*min_axis(rho_guard)。空域はnull |
| 残差guard閾値 | 40軸 | 実残差がこの値以下との証明ではない |
| 比較領域 | 各精度の4方式全6pair＝30、うち同表現PR/PF10が主比較 | 2方式の残差・費用tailを独立変数として扱う |
| PRのtail/母平均診断 | 2方式で各1 batch | 確率・mu_h変数・比較の反転閾値 |

θ=0は**仮想的な最小誤差例**で、残差推定値や無条件適格行ではない。方式間で同じθを使っても同じ物理残差ではない。主な結論は独立rho変数を持つ許容域と比較領域から示し、共通θ断面だけで順位を確定しない。

N/Gはceilを含む区分単調関数として扱い、break-evenを必要なprecisionの区間で囲む。名目MC intervalを使う領域は経験的/条件付きと表示する。証明された費用envelopeを使う場合も、その条件の範囲内だけでrobustと呼ぶ。

成果の中心は、(A)C削減、(B)bias/B/rhoによるshot負担、(C)その積Gの利益・不利益、(D)残差/tailでの反転・判定不能、(E)固定4実装から支持できる結論である。数値が出ない条件も理由を含めて結果とする。

比較不能理由は、base guard域が空、実残差未確定、scale/丸め十分条件未解決、母平均費用envelope欠測、比較条件の不一致等を分ける。未知N/Gを0や無条件の有限値へ変換しない。

## 7. 次回の一括承認対象・実行境界

[execution_contract_DRAFT.json](../artifacts/non_df_h4_conditional_cng_plan/2026-10-24/execution_contract_DRAFT.json)は対象・入力SHA・式・残差・統計・20/100/30の出力範囲・禁止call・STOPを固定した**非実行契約**。現在`execution_authorized/grant_valid/dispatch_enabled/N_G_authorized=false`、grantなし、数値runner SOURCE/runtime未設定。計画checkerは存在するが、実解析runnerではない。

次回ユーザーが一括承認する具体的範囲は、**指定550入力の保存値再解析、既存関数を呼ぶ薄いadapterの実装・人工検証・SOURCE/runtime固定、20条件のN/G関数・100表示scenario・30比較領域・PR2方式の費用/tail感度、結果保存とnative終了/STOP**である。SOURCE固定・入力照合は最初の実データ算術より前に行う。新namespace・未使用grantだけを使い、各有限batchを非返却予約し、既存owned witnessで終了・回収を記録する。自動retry/resumeはしない。

この再解析はreference/law/bias/C40軸/Lの再計算ではない。新しいSCF/FCI/DF/state、sampling、RNG replay、compile、435件診断、candidate再選択、正式certificate受理、旧結果書換えを予算0とする。新系列・guard変更が必要なら別判断へ戻す。

**435件診断と旧compiler全証明は既定の必須工程にしない。** 追加数値保証を優先するのは、その未被覆誤差の範囲が許容域や結論を跨ぎ、残差として条件表示するだけでは目的の主張を支えられない場合。どの命題・数値がどの結論を変えるかを示して最小案を作る。

## 8. 今回の検証・保全・公開・引継ぎ

[検証記録](../artifacts/non_df_h4_conditional_cng_plan/2026-10-24/verification.json)：40軸のpointer、514価格join、550 input SHA、元20精度行のnullを構造的に確認。実H4 endpoint・価格を会計関数へ渡さず、新しい平均・SE・順位・感度を計算していない。

既存`fixed_signal_shots`、guard/paired G集計、`_statistics`を人工値だけで再利用し、8件PASS。初回は手作りfixtureのevidence_ref不足、次にFLINT専用環境にSciPyがなく統計関数を実行できなかった記録を保持した。Fixtureを補い、既存の元SciPy環境を変更せず利用して通した。[環境・試行記録](../artifacts/non_df_h4_conditional_cng_plan/2026-10-24/test_runtime.json)。パッケージ追加は0。旧202件は数値production codeを変更していないため今回再実行していない。新規検査は主agent自身によるもので独立科学レビューではない。

元H4/C/L/K/C-LのSOURCE/raw/grant/ledger/marker/STOPは変更しない。旧N/G null・C40軸TARGET_MET・certificate未受理・compiler MISSING_EVIDENCE・元native UNKNOWN・L mtime失敗履歴を維持する。原本への変更がないことをGit差分と今回の入力SHAで確認し、同じ多数の旧監査を重ねて作り直さない。

資源は小さいsavedデータとscalar式中心であり、入力を一度読んで再利用する。必要な並列化は次回workloadに応じて選び、CPU/RAM/workerに人工上限を設けない。共有環境・他ユーザー・既存quota/権限を変更しない。

専用branchへcommitし、新commitのpushとremote SHA照合を試みる。[公開結果](../artifacts/non_df_h4_conditional_cng_plan/2026-10-24/publication_status.json)。**今回はmandatory STOP。実H4数値再解析・新N/G・sampling/compile/grantは0。**

次の担当：**ユーザー**は本契約の有限な保存値再解析packageを一括承認するか判断する。**Codex**はその明示承認後にadapter準備・SOURCE/runtime固定・新grantでの限定再解析を実行し、条件付きの結果・不確かさ・判定不能を報告してSTOPする。追加sampling/compile、未保証残差を閉じる新診断、正式certificate受理は自動で進めない。
