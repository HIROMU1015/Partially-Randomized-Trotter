# M1-S：整数shot・総回路費用の成立条件 — Decision Packet

2026-10-17 JST。基点 `81d6103ec89d485cb7336c3a90bd95dad66992aa`。
`NON_NUMERICAL_DESIGN / CANDIDATE_NOT_ADOPTED / NOT_RUN_READY / MANDATORY_STOP`。

本書は既存のRPE適用条件packetを、完了済みγ依存V1・π/3限定gateを踏まえて具体化する技術資料である。新しい科学レビュー、保存数値の再解析、整数shot予測、native合成を行っていない。次期契約は未採択である。数式は次期検証の候補であり、科学条件や実行契約を今回採択しない。

## 1. 今、判断できることとできないこと

Task A先行、Task Bを最終目標、無補正raw直接route、保存全cellの公平比較という既存方針を継承する。次期案は**整数化後も十分shot設計の削減が残るか**を先に確認し、その後に必要なnative費用の検証範囲を選ぶ構成である。この段階分け自体も次期契約の候補として示す。

π/3限定gateの成立は、連続marginのshared有利域と保守的decoderの角誤差領域が交差するという証拠である。採用するγ、他の位相誤差を引いた残余、切り上げ後の整数shot差、実装費用の余裕を示したものではない。小さいvisibility差を相対比だけで救済しない。

|保存された区分|対照数|本packetでの扱い|
|---|---:|---|
|G3、T=1・4：限定gate成立|16|保存区分を保持。整数shot・費用の成否は未取得|
|G6、T=4：限定gate成立|8|同上|
|G6、T=1：限定gate不成立|8|この保存cell・連続十分設計・保守的routeについての不成立を保持|
|G9：絶対改善未解像|16|`G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED`を保持。形式的相対順位で昇格しない|

Scopeはlinear H4 1.00 Å／STO-3G／8 modes／保存DF rank12、G3/G6/G9、δ={1/8,1/16} Ha^-1、R={2,4}、T={1,4} Ha^-1、保存state・元norm²、非冗長Pauli-tail。同じ概念的独立label数のshared ±／ordinary／independent ±である。24非zero cell・72方式行・48対照、36 zero controlを含む元108行のinventoryを保持する。上表はユーザー指示と[固定完了報告][pi3]の転記であり、raw/CSVの再集計ではない。

passing 24対照だけを新しい独立検証集合と呼ばない。候補のeligibility層として使う場合も事後情報を利用した探索的層別と記録し、全48対照の分母、8不成立、16未解像を報告する。次期の主対象やγを既知の勝敗に合わせて選んだ検証をheld-out再現性確認と呼ばない。

## 2. 証拠と欠けている情報

|項目|利用できる既存証拠|まだ得ていない情報|
|---|---|---|
|Hamiltonian・signal|保存raw DF Hamiltonianと保存stateに対する有限step複素信号、条件付き丸め半径|実機wrapperの平均信号、実装誤差、追加timeの信号|
|モデル差|H_fullと各H_tildeとの差の上界B_j、η_j(T)=abs(T)B_jの会計|旧v5の未保存full-H/HD行列との直接一致。これを確認済みとしない|
|位相bias|保存raw-H参照に対する総principal biasとその数値半径|特定E_* Tとの円周差、全roundで有効な上界|
|状態|保存stateのnorm/basis情報、oracle残差の受理条件1e-8 Ha+B_res|認証されたtarget overlap、gap、target同定、状態準備回路・品質・費用|
|統計|fresh-IID・Hoeffding・X/Y等配分の条件付き十分設計、連続γ境界|整数shot数、物理測定でIID・平均値が成立する証拠|
|decoder|π/3限定gateの保存区間比較結果|energy prior、初期branch、符号、全dyadic round、最終energy精度の契約|
|回路費用|既存DF/RTE wrapper会計のSOURCEと除外項|M1-S固有のcontrolled wrapper、HD、合成誤差、state準備、native cost|

保存誤差coverageは `CONDITIONAL_ENGINEERING_NOT_CERTIFIED`。IEEE／BLAS／fsum／三角関数等の未検証条件を引き継ぎ、統計confidenceと足し合わせてplatform certificateにはしない。保存solver残差は固有状態距離・ground overlapの保証ではない。

## 3. 位相予算：Task AとTask Bを区別する

円周距離を d_S1、単位はrad、時刻はHa^-1、energyはHaとする。保存理想方式信号をz_m^0、raw参照をz_raw、物理測定が与える平均をz_m^p、目的位相をφ_*とする。保存中心値と誤差半径から構成する候補は

\[
 A_{m,0}^-=\max(0,\widehat A_m-r_{A,m}),\qquad
 b_{m,\mathrm{raw}}^+=\min(\pi,|\widehat b_m|+r_{b,m}). \tag{P1}
\]

保存phaseが定義でき、branchとcoverage条件を満たす場合に限る。半径は数値不確かさであり、許可された物理biasやRPE目標精度ではない。旧pilotの2^-24 signal、2^-18 rad phase、visibility floor1/4を新taskのγ・精度へ流用しない。

raw直接routeでは

\[
 b_{m,\mathrm{tot}}^+
 =b_{m,\mathrm{raw}}^+
  +b_{\mathrm{raw}\to *}^+
  +b_{m,\mathrm{impl/SPAM}}^+,\qquad
 q_m=\gamma-b_{m,\mathrm{tot}}^+>0. \tag{P2}
\]

各項を円周距離の三角不等式で結ぶ。Task Aの**保存理想信号による診断**ならφ_*=Arg z_rawと定義し、raw→目的項は定義上0である。実装前の理想比較なので物理誤差を評価したことにもならない。Task Bではφ_*=E_*T（符号規約を固定）とし、raw→目的項を別証拠で与える必要がある。保存stateの混合energy成分による位相非線形性を無視できない。保存stateから実装stateへの変化はimpl/state-preparation項へ入れ、raw→目的の既存state contaminationと二重計上しない。

**保存総biasはH_tilde−H_fullモデル差を既に含む。** P2へη=abs(T)B_jを再加算しない。モデル差ηは複素信号差の上界であってradではない。split分離routeへ変更するなら新契約が必要で、visibility下界L>η、非zero・branch条件の下でarcsin(η/L)等を用いる。現routeで純アルゴリズムbiasを分離測定済みとしない。

たとえば実装による複素信号誤差d_mについて、A_{m,0}^->d_mが証明できれば

\[
 A_{m,p}^-\ge A_{m,0}^- -d_m>0,\qquad
 b_{m,\mathrm{impl}}^+\le\arcsin(d_m/A_{m,0}^-). \tag{P3}
\]

原点を含まない誤差diskに対する円周角の評価である。principal位相の実数差へ読み替える際はbranchを跨がない条件も必要。d_mを未取得のまま0と置かない。readoutのaffine歪み等は、同じmean/diskモデルへ結び付く証拠がなければ別項として未解決とする。

Task Bの保守的decoder候補は、全round kで

\[
 T_k=2^k t_0,\qquad b_{m,k,\mathrm{tot}}^++q_{m,k}\le\gamma_k<\pi/3,
 \qquad \sum_{k,a}\alpha_{k,a}\le\alpha_{\mathrm{total}}. \tag{P4}
\]

alias-free priorと初期branch、nearest-consistent unwrap、全roundの誤差事象が必要である。最終角誤差をenergyへ写す条件はγ_K/T_K≤ε_Eであり、高確率絶対誤差をRMSEと同一視しない。保存T=1,4だけでは中間T=2を含む完全なround列は揃わない。π/3 gate成立だけではP2–P4の不足は埋まらない。

保存oracleを使って評価したbiasを、運用時に無料で既知の情報として補正しない。無補正routeを維持する。state品質の独立証拠、priorや較正の取得費用も未取得情報として扱う。

## 4. 整数shotの十分設計と厳密な削減条件

候補統計policyはfresh-IIDな各Hadamard shot、X/Y等配分、軸別Hoeffdingである。sharedのlabelは一shot内の両branchで共有し、別shot間ではfreshに生成する。±1のX/Y結果の平均が同一のz_mのRe/Imであることが必要。軌道の跨shot再利用・相関はこのpolicyで保証しない。

A_m^->0、0<q_m<π/2とし、M_m=A_m^- sin q_mと置く。各軸平均の座標誤差t_m=M_m/√2以内ならcomplex disk半径M_m以内となり、位相誤差q_m以内が十分に保証される。各軸の失敗確率をα_axisとすると

\[
 S_m={4\log(2/\alpha_{\mathrm{axis}})\over M_m^2},\quad
 n_m=\lceil S_m\rceil,\quad N_m=2n_m. \tag{S1}
\]

2exp(−n t²/2)≤α_axisからの十分shot設計であり、最小shot数の定理ではない。PER_METHOD_ROWならα_axis=α_row/2で分子は4log(4/α_row)。両方式が個別にこのconfidenceを持つことと、全方式・全cell・全roundの同時confidenceは別である。新契約のαとscopeは未選択。旧V1のα=0.05を自動継承しない。

同じγ・confidence・policy・目的位相でsharedと対照cを比較し、他の予算も満たして初めて

\[
 n_s<n_c\iff \lceil S_s\rceil<\lceil S_c\rceil
 \iff \exists n\in\mathbb Z:\ S_s\le n<S_c. \tag{S2}
\]

M_s>M_cは連続設計の優位を与えるが、S2を保証しない。**INTEGER_TIEは削減ではない。** γ*<π/3はS2へ代入するγの指定でも、切り上げの余裕でもない。γは外部の精度要求または事前登録した共通domain/profileで指定する。既知境界を見て勝つγだけを選ぶ運用をしない。共通の数学的domainは0<γ<π/2。保守的RPE routeへ接続するclaimには、さらにγ<π/3とP2/P4の残余予算が必要であり、それより広いdomainの診断をRPE適用証拠にしない。

算術enclosure S_m∈[l_m,u_m]が得られる将来の候補実装では、整数区間[ceil l_m,ceil u_m]を保存する。ceil l_m=ceil u_mの場合だけ単一n_mを確定する。全domain条件を満たし

\[
 \lceil u_s\rceil<\lceil l_c\rceil \tag{S3}
\]

なら十分設計の厳密な整数削減が確定する。反対にceil l_s≥ceil u_cならこのdesignで厳密削減しない。他は`ARITHMETIC_UNRESOLVED`。単一整数が確定すれば同数と対照優位も区別できる。閾値近傍をfloatのepsilonで押し分けない。固定算術精度・有界の丸め規則を採択してから実行し、不確定なら保持してSTOPする。

保存bias/visibility上界を用いた設計同士の比較であり、実際の必要shot数の順序は証明しない。精度下のG9絶対差を、S1の整数差や比率で有意な物理改善に昇格しない。今回S1–S3への数値代入・ceil計算は0件である。

## 5. 総回路費用：shotsだけでは決まらない

費用metric gを先に固定し、Fは同じ単位に換算できる場合だけ含める。候補式は

\[
 G_m^{(g)}=F_m^{(g)}+\sum_{k,a}n_{m,k,a}
       (C_{m,k,a}^{(g)}+P_{m,k,a}^{(g)}). \tag{C1}
\]

Cは一shotの**全controlled測定wrapper**、Pは一shotごとの状態準備。labelで変動する場合は固定lawでの期待費用か、別途採択する確率的上界かを明示する。期待native費用と実現値の保証を混同しない。compiler Monte Carlo sample数は量子shot数ではなく、もう一度掛けない。

|構成要素|必要な会計／公平性|現時点|
|---|---|---|
|controlled ± branch|制御・負角・Pauli parity・相対位相・ancillaを含むM1-S固有回路|SOURCE上の意味論のみ。native価格は未取得|
|HD evolution|同じ精度・duration・呼出し規約で分割ごとに価格を付ける|dense matrix指数を無料native操作としない|
|tail rotations|sharedの追加branch演算、ordinaryのcontrolled演算、independentのlabel数を区別|概念label数の公平性をnative費用同一へ読み替えない|
|X/Y・readout|H、Sdg、basis変更、測定、reset、失敗処理等、選んだmetricの範囲を登録|既存wrapper costのscopeとadapterの整合が必要|
|状態準備P|同品質・同用途、必要ならuncompute／reprepareを含む|保存vectorの存在は無料state preparationの証拠ではない|
|固定費F|較正、prior、bias情報、合成・前処理等|native gate数とCPU時間を根拠なく足さず、別勘定でも報告|

同じPでもshot数が違えば総額から相殺できない。T/CX/RZ数やdepthは原則別metricであり、恣意的に重み付き単一価格へ変えない。depthのshot和は直列workであり並列実時間ではない。workspaceはpeak制約で、C1の加算metricではない。

単一cell、X/Yで同じ一shot費用c_m=C_m+P_m、n_m>0なら、固定費差も含むbreak-evenは

\[
 2(n_c c_c-n_s c_s)>F_s-F_c. \tag{C2}
\]

F_s=F_cの特別な場合だけ c_s/c_c<n_c/n_s と書ける。**整数化を捨てた**連続proxy c_s/c_c<(M_s/M_c)^2はscreening式であり、C2の実測成立ではない。未知のP/Fを0と置いて結論を作らず、感度領域の変数として残す。

有効な費用区間があれば G_s^+<G_c^- は区間を跨いだ厳密優位の十分条件。二つの上界の比較だけで実費の優位は保証できない。科学的に必要な改善幅μを設けるならG_s^+<(1−μ)G_c^-等の規則も別途採択する。今回はμを選ばない。

枝の代数として、同一Pauli Pなら

\[
 \operatorname{diag}(e^{-i\theta P},e^{+i\theta P})=e^{-i\theta Z_a\otimes P},\qquad
 \operatorname{diag}(I,e^{iaP})=e^{iaI\otimes P/2}e^{-iaZ_a\otimes P/2}. \tag{C3}
\]

これらは実装候補の恒等式でありnative価格比ではない。sharedを一律2倍とも無料とも置かない。Pauli列の符号反転・積の順序、HD dressing、identityのancilla相対位相、pilotの+iTHと既存wrapperの符号を保つ。逆符号の同順序積をword adjointへ取り替えない。

## 6. 最小検証候補と必要な承認

|段階候補|最小対象・出力|進む前に必要なこと|
|---|---|---|
|S：保存信号の整数shot診断|全48対照registryを保持し、共通γ条件下のn区間・分類・前提充足を保存。36zeroはinventoryのみ|γ/domain/profile、α/scope、Task A予算、算術policy、capsの採択。SOURCEを新versionで実装・静的検査し正式契約固定。その後別科学実施承認|
|C：native cost feasibility|採択されたcase選択規則と同精度の3方式wrapper、HD、P/Fのcost情報。shotsとのC1/C2照合|shot結果を踏まえた実施範囲、metric・backend・compiler・HD・合成/実装誤差・P/F policy採択。native SOURCE・契約を別固定し別実施承認|
|B：多round RPE|energy prior、target state品質、全round位相予算、統計配分、総額|S/CのPASSだけで自動移行しない。欠けた証拠と最小追加検証を別の判断に戻す|

実行後の都合に合わせたγ／cell／閾値変更、最良例だけの報告、失敗や未解像の自動retryは候補契約でも禁止する。数値的PASSは十分設計の技術判定であり、研究GOを自動発行しない。

技術設計・再利用候補・resource根拠は[技術引継ぎ][handoff]、未決科学条件は[契約候補][candidate]、機械可読候補・証拠は[artifact][art]にある。今回のclosed guardは設計資料を実行可能にする入口ではない。**採択後も新しい実行SOURCEと固定契約の準備、別途実施承認が必要。**

## 7. 今回の作業境界

原本・保存raw・SOURCEのidentityはbytes/SHA256と固定Git objectで確認する。raw/CSVは数値データとしてdecodeしない。検査は新しいmetadata、schema、閉じたguard、リンク、旧成果保全のみ。旧v4のTECHNICAL_STOP/null、v5／独立検証／pilot／V1／π3の結果と消費済markerを保持する。

新規科学計算・数値再解析・shot forecast・native compile・RPE simulation・新authorization／grant／marker・正式採択は全て0件。公開後**mandatory STOP**し、次期条件の採否判断をユーザーへ返す。

[pi3]: m1s_gamma_pi3_completed_results_and_stop_20261017.md
[candidate]: m1s_shot_cost_verification_contract_candidate_20261017.md
[handoff]: m1s_shot_cost_source_resource_and_handoff_20261017.md
[art]: ../../../artifacts/track_b_m1s_shot_cost_preparation/2026-10-17/v1/evidence_index.json
