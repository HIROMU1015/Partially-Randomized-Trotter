# Track A：H4〜H8リソース評価の追加検証設計

2026-10-14 JST。**DESIGN_DRAFT_NOT_AUTHORIZATION**。技術調査・保存metadataの照合・設計を完了し、科学的な採択事項を分離した。新SCF/DF/state/signal/trajectory/build/compile、H8操作、新grant、原稿改訂は0。以下のpilotと数値許容条件は提案であり、今回実行していない。

正式ノート`4675730e7e40d25ba3618334a4a0e28b4bd2f145`、サーバー受領`2ef5ecbdbdc222d73d9106618e20dc27b5466ea2`は開始時にremote HEAD一致・固定GitHub bytes一致・前者が後者の祖先であることを確認した。今回の独立branchは後者を起点とする。
[共通研究ノート](track_a_shared_research_status.md)、[同期方針](track_a_environment_sync_policy.md)、[科学review判断](track_a_scientific_review_decisions.md)、[条件対応表](track_a_resource_comparison_conditions_20261014_v0_4.md)、[受領確認](track_a_cross_environment_server_acceptance_20261014.md)を基礎とする。
共通指示`research_project_instructions_gpt_codex_review_approval_2026-10-12.md`原本50,268 bytes／SHA-256 `709524f856a3117c85042309c390a3697d9b2c8c8d79af6b6c290e9d20814bcf`を全文受領・適用。Codexの技術調査を先行し、科学的採択・レビュー開始・計算実行の承認を分ける。GPT詳細科学レビューは今回開始していない。

監査入口：[SOURCE/result索引](../../artifacts/resource_applicability/track_a_resource_evaluation_expansion_design/2026-10-14/source_reference_index_v1.json)、[検証matrix・予算](../../artifacts/resource_applicability/track_a_resource_evaluation_expansion_design/2026-10-14/validation_matrix_and_budget_v1.json)、[保存精度診断](../../artifacts/resource_applicability/track_a_resource_evaluation_expansion_design/2026-10-14/saved_precision_census_v1.json)、[過去資源実測](../../artifacts/resource_applicability/track_a_resource_evaluation_expansion_design/2026-10-14/historical_resource_calibration_v1.json)。原科学SOURCE/resultを編集せず、Git objectのコードを静的に読んだ。

## 1. 研究目的と今回の設計範囲

RQ-R：固定DF Hamiltonianと指定stateの有限時間complex signalに対し、bias・有限RTE normalization・解析的shot数・実compiled full-wrapper費用を結び、PRの低費用領域と限界を比較する。主指標はP=0のshot-weighted RZ。εはsignal精度でありHaではない。
将来候補はlinear hydrogen chains/STO-3G/1.00 Å、H4〜H8、ε=.05/.01/.005/.001、T=.4/.8/1.6。H5/H7、より厳しいε、native B0+I等は以下で具体化するが未採択。H9/H10以上は保留。
R09はH4を限定benchmarkとして進める判断を既に採用し、全H4への追加独立32・全support保証・全面高次PFを必須工程にしていない。本設計はその判断を取り消さない。新しいサイズ/T/比較契約に必要な不足と、現H4原稿の完成必須条件を混同しない。

## 2. 既存のH4/H6/H8結果と条件差

|条件|H4 geometry＋B0+I|H4 native anchor|H6 native|H8 native|
|---|---|---|---|---|
|geometry／modes／sector|.70/.80/.90/1.10/1.40/1.60 Å、8 modes、α2β2/36。1Åなし|1Å、8、α2β2/36|1Å、12、α3β3/400|1Å、16、α4β4/4900|
|H/state|各凍結DF Hamiltonianと保存DF state。元参照はexp(-iET)|保存1Å DF/stateと独立action参照|保存DF/state、独立occupation/expm/eigh参照|保存DF/state、CSR action・split-time・残差参照。feasibilityとcore別|
|DF policy／rank／order|legacy固定12。生成時の重み順・signed λ・固有vector phaseを凍結|legacy12を保持|tol-only1e-8、actual19、cutoff0、returned signed generation order、weighted Hermitian projection|同政策、actual28、別input/projection identity|
|prefix|0/3/4/5/6/9/12。B0+Iは3/4/5/6/9|PR3/full12の16固定panel。B0なし|B0/B2 5/10/15、B1 full19|B0/B2 7/14/21、B1 full28の登録設計|
|PF／random tail|S2、finite RTE、q1/2/4/8、r1/2/4/8、K2/4|PR S2、full S2/global Yoshida S4、q1/2/4/8、r4（q8はr2）、K2/4|B0/B2 S2、full S2/S4。q1/2/4/8、r1/2/4、K2/4、R=qr|同92候補の設計。feasibilityは少数固定caseのみ|
|T／δ|.8／.8/q|同左|同左|同左|
|control／Gaussian|ordinary、H4 structured JW Givens＋境界basis再利用|symmetric_directional、native JW Bogoliubov/Givens、raw concatenation|同native。ordinaryは2 anchorの限定感度|同native。joint-branch/hash高速化は別closure|
|compiler|Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、approximation1、topologyなし。compiler fbe36b72…|native固定profileを参照|同label、compiler1239304c…、profile4e32bb9b…|同native hash、実行SOURCE/profileは独立binding|
|allowance／shot|明示u差引なし。Bとaxis biasからceil会計|経験的u/B_upper、1倍/10倍感度|同政策、u/headroom≤.01を費用取得guard|同政策。certificateではない|
|selection／確認|元1308候補、PR各32、same32で選抜/推定。I120構成240軸|事前固定16候補、PR8×32、det8×1。全domain探索ではない|探索92/592 group、PR各8。union8を別seed各32、探索とpoolしない|core92→探索8→union最大8×独立32の登録設計。最終result未取得|
|費用／対照|元74784 logical wrappers＋I240。RZ/CX/各depth/size。S4なし|532 wrappers、full S2/S4、B0/B0+Iなし|探索1188・確認512 wrappers、全6 raw metrics。B0/full S2/S4あり、B0+Iなし|feasibility8 case/20 wrappersあり。最終mapと対照費用はPENDING|

Hamiltonian/state/sector/DF/compiler/environment/seed/phase/controlをidentityとして分ける。compiler labelが同じでも回路系列・入力・丸め受理・環境hashが異なれば同じ結果にしない。H6 science SOURCEと各execution SOURCEを区別し、旧失敗と後続成功も混ぜない。
既存stateはSCF determinantそのものではなく、保存DF modelのsector solver stateである。残差チェックはglobal ground-state certificateではない。state-preparationの実装資源は主wrapperに含まれず、P=0である。
H8は開始時の公開refsに正式core最終resultを確認できなかったため**PENDING**。過去556/592 snapshotからlive状態を推測せず、H8のprocess/worktree/checkpoint/ledger/observerへ今回新たな操作を行っていない。

## 3. 現在の検証範囲の不足

|不足|既存証拠の意味|最小の対処|
|---|---|---|
|統制1Å size比較|geometry H4は1Åを含まずcontrol/DF/loweringも異なる。native H4もDF12と限定panel|既存nativeを別caseとして並べるか、厳密共通DF契約を新設するかを選ぶ。旧geometryを変換したことにはしない|
|native B0+I|H4 ordinaryでの240軸差0はnative H6/H8の証拠ではない|H6の3固定構成/新I6軸pilot。必要ならmatched B0も6軸を別登録。結果に応じて12構成24軸まで別承認|
|H4 native B0の不足|native16panelはPR vs full S2/S4で、最良deterministic family全体ではない|旧DF/state保持ならquarter-prefix B0/B0+I各4q、計16軸を追加候補にする|
|H5/H7|奇数電子sector APIはあるが、旧geometry既定値・input ports・高速primitiveが対象と不一致|明示charge/spin/SCF/state契約と汎用portを別SOURCEで実装・検証|
|ε拡張|保存値で再会計可能。ただし選抜済み32と全候補8の確認範囲は異なる|headroom/整数shot/metric/P解析を先に行う。新精度で選ぶ候補を旧確認済みと呼ばない|
|T依存性|T=.8でのsignal/費用を他Tへ外挿できない|H6保存入力・固定qの小panelで新Tだけ評価|
|補助指標|全6 raw metricsと公開集計の範囲は異なる|per-axis meanから可能なGと、per-trajectory covarianceが必要なSEを区別|
|不確かさ|same-sample selection、未観測tail、経験的u、旧compiler bit一致未検証|限定benchmarkの留保を保持。検証の目的が保証へ拡張される場合だけ別設計|

比較条件を揃える選択肢は二つある。**A：nativeの共通表示panelを作り、H4のlegacy DF/state政策差を別caseとして明記する**。既存H6/H8とH4 anchorを多く再利用できる。**B：tol-only/weighted projection/state policy/prefix ruleを共通化した新H4入力を追加する**。厳密なpolicy比較に向くが、新DF Hamiltonian/state/signal/費用が必要で、旧H4を置換しない。
技術的推奨はAの不足対照を小pilotで確認し、厳密size claimが本当に必要と採択された場合だけBへ進むこと。最終選択はGPT判断待ち。同じtolは同じrankや元化学Hamiltonianに対する同じoperator誤差を保証しない。
共通prefix候補`ceil(L/4), ceil(L/2), ceil(3L/4)`はH6=5/10/15、H8=7/14/21、legacy H4=3/6/9に対応する。小rankで重複するprefixを別候補として水増しせず、0<p<Lとorderingを検査する。新しい候補契約として未採択であり、旧panelのp3/q/rを変更しない。

## 4. H5/H7の開殻対応調査

### 再利用できるAPIと明示的な障害

1. `PhysicalSector.spin_sector`は任意のNa/Nbを受け、HF初期vector・matrix-free DF action・JW occupation oracleも固定Na=Nbを要求しない。ただしこれは開殻のend-to-end検証済みという意味ではない。
2. **既存`chemistry_hamiltonian.call_geometry`の奇数鎖はcharge=+1、multiplicity=3**。中性H5/H7 doublet候補に流用すると対象が変わる。neutral charge0/multiplicity2なら提案sectorはH5 α3β2/100、H7 α4β3/1225。旧既定のH5+ tripletはα3β1/50、H7+はα4β2/735で別modelである。
3. H6/H8のinput/loader/solverは12/16 modes、3/3または4/4、400/4900を固定。H6 dense参照/solverはdimension≤400、H8 bounded solverは**dimension==4900**なので、H7にはそのまま使えない。
4. **`h8_sparse_reference_v1.SpinPrimitive`はNa=Nb、Gα=Gβ、単一のoccupation basis/eigensystemを両spinに使う**。H5/H7のneutral候補でもcountは異なる。組合せ数が同じ10×10／35×35でも、alpha/betaのoccupation基底とspectraは別に構築しなければならない。
一般化ではalpha/beta別の基底・固有値・固有vectorとfermionic permutation/signを使う。one-body phaseは`exp[-it(e_alpha+e_beta)]`、DF平方primitiveは`exp[-it lambda(e_alpha+e_beta)^2]`で、cross termを落とさない。同じshapeだけからbeta spectrumをalphaへ流用しない。

5. `OccupationCSR`の上限n≤16/dimension≤4900は両neutral候補を含み、cross-spin exact0とsector escape検査を維持して再利用候補になる。primitiveをsectorへ射影してから平方することは、sectorが各primitiveで不変と証明した場合に限る。
6. native controlled DF/finite-RTE/JW loweringは準備blockとnを取るので再利用可能性が高い。basis回転の途中を物理sectorへ射影せず、full system＋ancillaの枝を検査する。各basis gateのspin保存と、完成primitiveのsector保存は別である。しかしodd-countのcorrectness・global/ancilla phase・spin ordering・negative S4 timesは未検証。元H8 sourceを変更して試さず、新module/namespaceにする。

### SCF・orbital・stateの選択肢（未採択）

|候補|既存実装との接続|問題・必要な変更|
|---|---|---|
|ROHF＋共通spatial orbitals|既存providerはmol.spin非0でROHFを選ぶ。現integral transformの単一2D MO行列に接続しやすい|neutral/multiplicityを明示。Na/Nb primitive一般化、SCF収束・orbitals/ordering固定、DF/state/S²検査は別途必要|
|UHFのspin別orbitals|独立alpha/beta軌道を許す候補|現`compute_integrals`は単一mo_coeffを前提とし、そのままUHFへ差替え不可。αα/αβ/ββ integral transform、spin-block primitive、physical S²のorbital overlap等を別実装・検証。既存H8 weighted projectionのGα=Gβ gateも無条件には通せない|
|共通のorthonormal orbital basisと別SCF reference|spin-independent積分規約を保てる可能性|既存MO/DF/circuit cost identityと異なる。orbital policy自体を科学契約として選ぶ必要がある|

ROHFを先に実装候補とする技術的理由は変更量が小さいこと。ROHFかUHFの科学的採択はしていない。既存providerの実file bytesを読んだ証拠を索引へ保存し、一般API説明は[PySCF SCF docs](https://pyscf.org/pyscf_api_docs/pyscf.scf.html)、[OpenFermion-PySCF provider](https://github.com/quantumlib/OpenFermion-PySCF/blob/main/openfermionpyscf/_run_pyscf.py)を参照した。現在web版と既存installの完全同一性を前提にしない。
SCF referenceはorbital取得/solver初期値の候補であり、観測stateをHF determinantへ変更する指示ではない。fixed Na/NbはMsを指定するだけでtotal Sのcertificateではない。DF stateのS²期待値・variance等を検査し、doubletやglobal ground-stateを主張する範囲は別に決める。[PySCF spin diagnostics](https://pyscf.org/_modules/pyscf/fci/spin_op.html)
full integralの完全なorbital回転と、DF threshold/projection後のHamiltonian・prefix・costの一致は別問題。orbitals/DF/state政策を変えたら新input identityにし、旧保存stateを無検証に移さない。

### H5最小pilot案とH7の入口

- **実装だけの前段**：新`open_shell_contract/input_port/snapshot_loader/spin_primitives`相当を独立SOURCEに用意。Na≠Nb、α/β basisのfermion permutation/sign、N/Sz保存、identity・zero/nonfinite、S4負時間、ancilla branchを人工fixtureで検査する計画。今回はsource作成・fixture実行もしていない。
- **入力段**：選ばれたcharge/multiplicity/SCFで1入力・1DF・1state solve。SCF policy候補conv_tol1e-9/max_cycle50、solver matvec予約上限10000、保存phase canonicalization・state norm1e-12・residual gateは既存native政策を起点に具体化する。rankはUNKNOWN、tol-only1e-8/cutoff0/returned order、weighted変化予算とcross-spin構造を検査し、rank削減/fragment削除/SCF retryで救済しない。
- **参照/回路段**：H5ならsector100で独立occupationとmatrix-free actionを全column照合する余地がある。HDF reference action vs split/eigen residual、native/Horner vs independent forward、norm/leakage、保存state枝とX/Y期待値、S²政策、実測metricsを検査。原則engineering gateは既存abs1e-9＋relative1e-10、reference agreement1e-10、u empirical/safety10を継承候補とし、厳密certificateにしない。
- **固定panel**：T=.8、q1、p=ceil(L/4)、B0/B0+I/full S2/full S4/B2(r2,K2)/B2(r2,K4)の6構成。det4×1＋PR2×8＝20 group、**40 compile／16 trajectory／32 occurrence**を最悪取得数とする。両軸で同じtrajectory、quantum shotsのsamplingなし。B0+Iは保存B0 ensemble signalから位相で導出できる場合がある。control validation候補上限32、primitive/reference call数は実rankから事前登録する。
- **取得順**：まず各構成1 groupの12 compileでcorrectnessとwall/RSSを取得。適格PRにだけ残7 trajectoryを取得する同一one-shot契約なら合計最大40。全ε不適格な候補を費用0で扱わず、不適格/未取得を別記する。q2/r増量などの自動fallbackはしない。
- **GO/STOP**：input/DF/sector/state/reference/control/費用bindingのtechnical PASSがH7準備の必要条件。PR勝利や全4精度の適格性をcorrectnessのGO条件にしない。S²政策未成立、非zero cross-spin、失敗/欠測/予算超過ならSTOP。H7の科学的実施と起動は別承認で、自動開始しない。
- **H7**：一般化port・Na/Nb primitive・native lowering・会計を再利用候補とし、新SCF/DF/state/reference/rank/input bindingが必要。sector1225に適したbounded CSR/solver adapterを新規に用意し、H8==4900 gateの単純削除で済ませない。同じ6構成/最大40 compileを独立pilot案にする。

neutral proposalのD=binom(n,Na)binom(n,Nb)は100/1225。binary64 complex vectorはH5 sector1600 B/full16 KiB、H7 sector19600 B/full256 KiB、dense sector matrixだけなら約0.153/22.90 MiB。これは**必要RAM見積りの全体ではない**。native build/serialization/transpileのmemoryが支配し得る。

## 5. 要求精度の拡張計画

保存scalarのheadroom診断を実施した。新signal、Hamiltonian/NPZ decode、N/G/winnerの新mapは作成していない。H4 u=0は元会計の規約であって真の誤差0認定ではない。

|保存domain|ε=.001で残るB0/B1/B2|ε=.0005で残るB0/B1/B2|解釈|
|---|---|---|---|
|H4 native fixed16|なし/5/4|なし/4/2|PRはq8 r2 K2/K4を含む固定panel内のみ|
|H6 native92|2/5/36|1/4/18|q≤8で再会計可能な候補が残る。全18を独立32確認済みとはしない|

geometry6距離の元4方式＋I120の保存biasについても診断JSONへ区別して保存した。geometryは1Å nativeの代用ではない。PM-1追加B0はB0 familyへ分類した。B0+Iの過去scalar診断に残るUNMEASURED注記は歴史で、現在のcompiled費用は98cdb35cの別証拠を使う。
native guardは`h_a=ε/sqrt(2)-|bias_a|-u_a>0`かつ`u_a/h_a≤.01`。今回の診断ではmax-axis-uを保守的に使った。保存量からの最小ε目安はH6 B2約1.9974e-4、H4 anchor B2約1.4910e-4。一方S4 q8はより小さいbiasを持つ。これは登録q≤8・固定input/Tの適格性の目安であり、低費用winnerや保証された連続境界ではない。
**推奨**：.0005の再会計を候補にし、同じ候補のlog_B_upper・u・整数ceil・overflow/headroom guardを保持して検査する。その結果から新探索の必要性を判断する。.0001等では現B2が不適格になり得るので、PF biasとfinite-RTE残差を分け、q16/高次PF/r/K拡張のうち必要な軸だけ別scopeにする。rを増やすだけでPF biasが消えるとはしない。
shotは`ceil(2 B_upper² log80/h_a²)`。εを半分にしてもbias/uが非zeroなら単純4倍ではなく、同じ候補でおおむね`(h_old/h_new)²`に従う。境界では発散・整数precision問題を検査し、Nをbinary64へ安易に変換しない。
新ε追加は新しい独立証拠ではない。旧H6 confirmationは4εから選んだunion8であり、5番目のεの上位2を足すとunion最大10等になり得る。旧grantを流用し、既存union8へ恣意的に切り詰めたり、探索8を確認32へpoolしたりしない。
energy/QPE/RPEへ進むには複数時間・位相unwrap/aliasing・state overlap/基底状態の意味・全query誤差/成功確率配分・state prep・実backend/rotation synthesis等が別に必要。εをHaへ換算して達成済みと表示せず、今回その全面実装へ拡張しない。

## 6. T依存性の検証計画

同じH/stateでもT変更によりreference phase、PF bias、有限RTEの`τ=λ_R T/(qr)`、normalization `logB=R log b(τ,K)`、headroom・shot・candidate最良が変わる。S2/S4の漸近orderだけで実測bias/Gを予測しない。

|再利用項目|T変更時の扱い|
|---|---|
|H/DF/state・basis/spectrum|input identity不変なら再利用可能。新SCF/DF/state不要|
|reference/candidate signal、u|新Tで再評価・独立照合が必要。保存Eの位相だけで全reference/action誤差を検証済みにしない|
|PR trajectory/events|finite-lawが変わるため旧eventsをそのままsample扱いにしない。新T/seed namespaceで登録。CRNならlawの正しいcouplingを別設計|
|uncompiled skeleton|q等固定なら構造再利用の余地はあるが角度・位相・fingerprintは新しい|
|compiled counts/depth|角度依存の最適化があるため無条件再利用不可。exact numeric fingerprint/profile一致の証拠がなければ新compile|
|最良q/r/K/prefix|Tごとに変わり得る。小panelでは全family optimumを主張しない|

最小案D3は**H6保存入力・q2固定**。B0/B0+I p15、full S2/full S4、B2 p5/r4/K2の5構成を新T=.4/1.6で評価する。新signal evaluation8＋位相導出2、det8 group×2軸＝16 compile、PR2 T×8 trajectory×2軸＝32 compile、合計**48 compile／16 trajectory／128 occurrence**。T=.8ではPR確認済み候補を参照し、未取得native B0+I費用はD1/D6で別に取得・照合する。
fixed-qではδ=.2/.4/.8。fixed-δ=.2ならq2/4/8となり、精度/回路長/occurrenceの変化を含む別の問いになる。両factorを最初から全系・全候補へ展開せず、D3結果でfixed-δや第二系の追加価値を判断する。negative Yoshida時間・phase・control coverageを省略しない。

## 7. 比較対照の整備

- **B0**：prefix fragmentだけ残すdiscard。既存`_prepare_discard`はtruncated Hを作ってからidentityを抽出するため、discarded tailのidentityを保持するB0+Iではない。H6/H8のB0 oracleも元constant＋retained fragmentsのtruncated Hを参照している。
- **B0+I**：full-Fock Pauli identityを保持する追加対照候補。signed λと実際のGから`c_tail=Σ_discard λ[(Tr G)²+Tr(G²)]/4`を凍結する。sector平均trace、fit、絶対λへの置換は別対照であり採用しない。B0 signalの`exp(-iTc_tail)`回転、empirical scalar/phase誤差伝播、native ancilla phaseを結合し、retained primitive/global phaseを二重計上しない。
- **native実装候補**：専用method descriptor/registry/loaderにB0+Iを追加し、r/K/Rなし・B=1・旧B0不変を保持。truncated preparation＋追加scalar phaseで構築し、旧B0へ結果を偽装しない。B0Iがcanonical cell registryに未登録であることも修正対象。
- **B1/full S2/S4**：full PFの登録q集合で比較。global Yoshida S4は各S2を3stageで合成する既存定義・負時間を保持。B1とfull S2/S4を重複実験として数えない。
- **B2**：partial S2＋finite RTE、deterministic prefix・r・K・λ_R・normalizationを保存。H6/H8は72候補の登録集合であり、全PR手法一般ではない。
- **B3**：H4 geometryには保存signal/cost、H6は84 normalization診断のみ。未取得native B3を科学的に排除したとはしない。主張に全PR family比較が不可欠と判断された場合だけ新scopeへ追加する。

native B0+IはまずH6 p5/q1、p10/q2、p15/q8の3構成6軸。旧B0のnumeric/profile/来歴がmatched比較に足りるかを先に検査し、旧host/compiler bit一致を仮定しない。新B0を実測し直す必要がある場合は追加最大6軸を事前登録し、pilot最大12 compileとする。I6だけのgrant下で自動追加しない。既存B0 scalar/metricとmatched比較をし、I↔PRのfamily再選択と混同しない。H4の240軸差0を保証として使わず、RZ/CX/depth/sizeの軸別deltaを実測する案である。scalar metadataが公開receiptにない場合は、許可された保存inputからの抽出・hash固定が先に必要。
最良deterministicの分母は、取得済み同scopeのB0/B0+I/full S2/full S4最小。未取得familyや別input/profileを分母へ混ぜない。旧H6のB0/full PF比較はそのscopeで保全し、新しい強化対照採用後のclaimを別にする。

## 8. リソース指標の拡張

`G_m=Σ_a N_a mean(C_m,a)`をRZ/CX/RZ_depth/CX_depth/total_depth/sizeで計算する。RZを主、他指標は補助を維持。RZ順位の候補へ別metricを表示する比較と、metricごとに再選択する比較を別表にする。
1-shot depthは各wrapperの層数。shot-weighted depthは**depth-shotというserial会計量**であり、実際のwall time・shot並列時のcritical path・hardware周期・T-depthではない。mean/max/per-axis値とweighted sumを同じ列名にしない。
H6 explorationには1188・confirmationには512のraw6 metricsが公開CSVにあり、RZ_depth/CX_depthのweighted mapは保存整数から追加できる。現在のconfirmation weighted mapは4指標であり、raw depthの存在を既集計の証拠としない。
H4 geometry公開projectionはpaired RZ列と6指標のaxis meansがある。各Gのmeanは再会計可能だが、RZ以外のpaired SEはper-trajectory covarianceなしでは復元できない。meanを32回複製して架空のSEを作らない。rawを別に受領しない場合はUNAVAILABLE/nullを保持する。
共通の仮想per-shot state-preparation費用なら`G_j(P)=G_j(0)+N_total,j P`。固定pair交点と、全取得済み候補の下側envelope再選択は別に出す。P_RZ/P_CX/P_depthの単位を分け、未測定の実state-preparation費用やcomposition最適化を含む総費用と主張しない。P=0 primaryやshot規則を変更しない。

## 9. 既存データで可能な検証

優先D0：保存scalarだけで、(a)native H4/H6の共通表示panel、(b)全6指標の平均G、(c)paired SEの取得可否、(d)P fixed-pair/envelope、(e).0005等のheadroom/整数shot、(f)B0/B1/S4/B2の適格/欠測/選抜済み範囲を整理する。H8は正式result取得後に同じprojectionを作る。
これは新SCF/DF/state/signal/trajectory/compileを要しない。科学arrayをdecodeしないstd-library解析を優先する。前の研究で決まったshot/normalizationを読み、保存値との完全一致を小さいscalar caseで検証してから新gridを生成する設計とする。
H4旧M1/PMの1Å ordinary結果も独立の履歴として参照可能だが、nativeへの費用変換ではない。geometryごとの1812点占有率をサイズ比較や独立成功率へ流用しない。旧exportの再実行・NOT_SAVED eventの復元はD0に含めない。

## 10. 新しい科学計算が必要な検証

|変更・検証|SCF/DF/state|新signal/action|新trajectory／compile|
|---|---|---|---|
|同input/T/candidateのε/P/metric再会計|不要|不要|不要。既存integer費用と標本scopeを再利用|
|native B0+I追加|原input維持。scalar抽出は必要な場合あり|B0の保存ensembleから位相導出可能。独立control/phase検査とu結合が必要|deterministic新compile。H4差0を仮定しない|
|同inputのq/r/K/PF order拡張|不要|新candidate signal・normalization・検証時間が必要|新lawなら新trajectory、新numeric wrapper compile|
|T変更|不要|新reference/candidate/uを評価|新T-specific eventsとcompile。旧Tの費用再利用は未保証|
|strict native H4 DF政策統一|同1Å raw積分が検証済みならSCF不要の場合あり。新DF＋stateは必要|新HDF/stateへ再評価|新namespaceの費用・trajectory|
|H5/H7新target|新SCF/integral/DF/state。ROHF/UHF等の採択後|新sector/reference/nativecorrectness|最小pilotの新費用・trajectory|

H4 geometryの凍結入力はAO/MO積分等を保存するが6距離に1Åがない。別の1Å入力系列のraw積分の受領可否は未確定なので、無条件に「SCF再生成が必要」とも「raw再利用可能」とも表示しない。DF政策/投影でHが変われば旧stateも自動再利用しない。
既存T=.8の1Å H6/H8に新full S2/S4計算を課す必要はない。固定SOURCE/compiler/task/control/入力identityが合えば保存費用を参照する。旧partialや別sourceの未受理cacheを新結果へ混ぜない。

## 11. 優先順位と計算規模

全て**未承認案**。compileはX/Yの1 full-wrapperにつき1actual invocation、trajectoryは両軸に共有する1event列、occurrenceは1trajectory中R=qr回。ε/P点数をcompile数へ掛けない。

|順位・ID|研究question／成果|必要な実装・既存証拠|新取得の最悪規模|GO/STOP・残る限界|
|---|---|---|---|---|
|0 D0|保存データで既に答えられる不足を埋める|fixed scalar/CSV、std-library会計とprovenance|科学call0／compile0|同identity矛盾でSTOP。独立標本は増えない|
|1 D1|native強化対照の影響を小さく調べる|新I descriptor/phase binding＋H6既存B0|新I6＋必要時matched B0最大6、計最大12compile、trajectory0|phase/control/metrics検査後GPT。全面差0は未保証|
|2 D2|odd-electron実装が成立するか|明示target・input/state・Na/Nb primitive一般化|H5 SCF1/DF1/state1、6構成、最大40compile/16trajectory/32occurrence|入力/数値/資源に失敗ならSTOP。1panelで最良性認定しない|
|3 D3|Tによるsignal/shot/回路競合を切り分ける|H6固定入力＋T-aware contract/参照/namespace|新T2×5構成、48compile/16trajectory/128occurrence|固定qの結果から必要な拡張だけ。全面T mapではない|
|4 D4|H5一般化がH7でも成立するか|H5 port＋1225対応solver/CSRと新input|H7 SCF1/DF1/state1、最大40compile/16trajectory/32occurrence|H5成功だけでH7のGO/起動を認可しない|
|条件付き D5|strict DF policyのサイズ比較が必要か|新H4 input/stateの小panel、legacyを保全|SCF0又は1/DF1/state1、最大40compile/16trajectory|GPTが必要性を選ぶ。Hamiltonian変更を明示|
|結果後 D6|nativeの決定論family不足を限定補完|D1の実測とverified numeric reuse|H6 I全24軸（pilot6照合再利用なら追加18、保証なければ追加24）。H8 I24軸は元core終了後。旧H4 anchor B0/I16軸|別grant。旧PRを再取得せず、結果後に採択/claimを判断|

技術的な推奨順であり、優先順位の科学的採択は未実施。H5が不要と判断されればD2/D4を省き、D0/D1/D3だけに縮小できる。strict size claimを選ぶ場合はD5を前に置く。既存H4/H6の論文化のためだけに全て実行する計画ではない。
比較用の全面案を静的に数えた。新1系でB0 12＋I12＋full S2/S4 8＋B2 72＝**104候補**（base signal92＋I位相導出12）。全適格ならdet32group＋PR72×8＝608探索group、探索1216wrapper。確認union≤8×32＝256group/512wrapper、ordinary感度4を加え**1732 compile／832 trajectory**。occurrence上限は探索5040＋最大8cell確認7168＝12208。旧結果をそのまま新targetへreuseする案ではない。1系/T=.8/4εの全mapは参考上限で、default作業・実行認可ではない。H5/H7/新H4や複数Tへ最初から掛け算しない。

### 過去実測からの資源目安

H4 native anchor532 wrappers/8workerはsupervisor1068.455秒、aggregate peak RSS約6.46 GiB。H6確認の新97group/194 wrappers/8workerは972.793秒、aggregate peak約4.97 GiB。どちらも全研究の古典時間ではない。
H6探索1188 wrapperには旧host/複数SOURCEの保全値が混在し、現host専用benchmarkではない。その保存「wrapper生成/fingerprint/compile section」wallはmin3.81／median21.23／max1037.17秒、worker peak RSSは約0.548〜7.929 GiB。confirmation CSVにはper-wrapper wall/RSS列がなく、探索値を確認実測と呼ばない。
過去medianを単純に6/40/48回へ掛けたserial算術scenarioは約127/849/1019秒。**H5/H7/Tの予測ではなく**input/DF/state/reference/build全体・制御検証・共有負荷・重いtailを含まない。新caseのwall/peak RSSはUNKNOWN。最初の12 compile等でphase別時間/RSS/AS/instructions/outputを採り、拡張費用を再見積りする。
起動時資源の候補はworker1・thread1・Qiskit num_processes1、既存nativeに合わせworker AS32 GiB/RSS9 GiB。CPU番号未提案/allowed_cpus=[]、実行許可false、memory予約ではない。H5 sectorが小さいことを理由に総RAMを小さいvector量で保証しない。必要なら初期pilot後にworker数を増やす別grantを検討し、H8や他jobを操作しない。
新pilotの運用予算はfuture planにstage別wall/output/journal/block/inode/quotaを固定する。候補は最大40/48 wrapper stageでoutput512 MiB、worker log32 MiB、journal64 MiB、10,000 inode、4hのfail-closed STOP上限。これは**所要時間/必要容量の実測値ではなく未承認の停止予算案**であり、input/snapshot・一時二重保存・terminal reserveは別に積算する。未知のH7/H5 solver時間を4h以内と保証しない。
process限定cache/TMPDIR等を全てhomeの専用output下へ置く。shared設定/venv変更、install/GPU/tasksetは今回0。future grantではfresh CPU/memory/pressure/OOM/FS/inode/quota・one-shot・source/profile/input binding・own PID/start identity・wait/reap/FD閉鎖を検査する。現nativeの監視/own-run STOP/cleanup条件を勝手に緩めない。
carry/failed/unknown/reservedは元lineageごとに保全する。これらの新取得数は新proposalの増分で、歴史的call返却やbudget resetではない。H4/H6/H8の異なる会計を一つの新総上限へ足し替えず、future scope採択後に適切なledgerへ正確に結合する。

## 12. 未決定の科学的事項

GPTへ返す論点は次の6点。証拠は本書と固定索引に揃えたが、今回そのレビューを開始していない。

1. H5/H7をneutral doublet候補として追加するか、旧charged tripletを別caseにするか。対象・S²・fixed-sector stateとglobal ground-state主張を区別する。
2. ROHF/UHF/common orbital policyのどれを使うか。変更量だけで科学的公正性を確定しない。
3. native限定panel＋政策差明記で足りるか、strict共通DF/state契約の新H4が必要か。比較candidate/標本/数値政策の統一範囲を決める。
4. native B0+Iのfull-Fock identity規約・採用scope。小pilot後のclaim修正と全面24軸取得の必要性。
5. ε=.0005を正式domainへ追加する価値と、選抜/独立確認のscope。old union8を変更する場合の契約。
6. Tをfixed-qで診断するかfixed-δで比較するか。小panelから何を主張するか。

希少tail保証・全面高次PF・実state-preparation・energy/QPE/RPEは別研究へ無制限に拡張しない。R09の採択を再reviewする必要はなく、新scopeの論点だけを集約する。

## 13. 推奨する次の実作業

**Codex：D0の保存scalar/6指標/P/精度再会計をまとめ、native B0+Iとodd-sector汎用portの実装仕様・人工fixture・source closureの準備を一つの作業として進める案**。今回認可したのは調査/設計のみなので、自動実装・fixture実行へは移らない。これらの技術作業に、source変更ごとのGPT詳細reviewを要求する案ではない。
new SCF/state/S²/baseline/size/Tの比較契約は上記科学判断後に固定する。実測D1/D2/D3等はSOURCE/profile/input/task数/carry/CPU/resource/STOPを結合した新one-shot grantと明示launch承認が必要。mandatory STOP後のH7/全map拡張・retryは別承認。
本設計branchを科学実行branchにしない。実装は固定native SOURCE c31ef424…から別worktree/branchを作り、本設計の文書参照だけを持ち込む。稼働H8のsourceやcheckoutを編集しない。実装のtechnical PASSを科学的採択へ昇格しない。
H8担当から正式result/receipt/sourceを受領した時点で、H8部分だけを固定追補する。最終map、scalar処理、選抜/独立標本、u、費用、cleanupを確認し、科学採否はGPTに返す。H8が未取得でもD0/D1/H5/Tの設計・承認判断を過去snapshotから補完せず進められる。

## 14. 固定SOURCE・結果・参照資料

|単位|固定SOURCE|result／公開記録|
|---|---|---|
|正式ノート／受領|科学SOURCEを置換しない|4675730e7e40d25ba3618334a4a0e28b4bd2f145／2ef5ecbdbdc222d73d9106618e20dc27b5466ea2|
|H4 geometry|d0726e946f89f8c30a08d383fb8ff48b363629ec、data5aa6c3685afb9c3afcfbdc36f80244da0a7fe525|7c4483378047b57f514c47dad98905e3f2a9a756|
|H4 B0+I|924b779c2bc6f6e844d6283c355f91c73aed6445|98cdb35cd08469714f20d6cc3b982d9c80908f46|
|H4 native anchor|217c971dd537b1cc7fbc13742bc0f6c263ff7f45|0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41|
|H6 exploration|science5a903aea2492c2d211c1e15e7737df0d50694c12、execution78a39051d8b6d41c33d592ecfe172f8824adfaeb|df99d6ee28942df8d12aac3c4db7cd202df55d77|
|H6 confirmation|87b6b9f46b7e8ba9abe132ee21478c114b6f46f7|321cd8c3066194fe98741cbebbcdba1515d1deaa|
|H6 conditional s2|843265b624eac8fbb9acea47ef21cab1207e1880|e0e6473d7047158a7f0f788d026a7a89a01898b8|
|H8 feasibility|9bfea8bde44cbb2ea9187476ddd496066a8853b3|a55da5c2b012912c485184845aa73ca053ac00b9|
|H8 current lineage|execution c31ef424175c02379eafebe8fc6c7548c1d9aabe、seed cb8a56c6b2a772827275f9200b02638030f33f72|起動38d386c8732295e6fab5d6fcbd47fbe586dc207d。最終result PENDING|

file単位のGit blob/bytes/SHAとコード位置はSOURCE/result索引へ記録した。全raw再hash・scientific array読込・provider実行・旧benchmark再実行は0。environment/compilerの同labelをoutput bit equivalenceと扱わない。

## 15. Mandatory STOP

科学的採択、GPT詳細review開始、新grant発行、pilot launchは未実施。`approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`、`absolute_launch_command=null`、`automatic_retry=false`、`next_stage_authorized=false`。
設計文書・検証matrix・軽量証拠・索引追補だけを明示stage/commitし、non-force pushを試みる。認証失敗時は設定を変えず未公開状態と手動commandを報告する。公開確認または認証fallback報告後**MANDATORY_STOP**。
