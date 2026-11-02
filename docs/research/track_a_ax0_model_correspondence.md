# Track A AX-0：原論文・実装・予測モデルの対応

2026-10-09。設計・静的監査のみ。AX-1の照合・fitは未実施、未認可。[研究契約](track_a_ax0_research_contract.md)を上位文書とする。

## 1. 一次資料の版と識別子

原論文はGünther et al., [Phase estimation with partially randomized time evolution, arXiv:2503.05647v2](https://arxiv.org/pdf/2503.05647v2)、2026-07-10改訂版に固定する。[刊行版PRX Quantum 7, 020332](https://link.aps.org/pdf/10.1103/ynxb-p2xq)も確認した。以下の式番号はv2の番号である。著者の[公開archive](https://zenodo.org/records/15387187)は所在を確認したが、download・実行・結果再現は行っていない。

repository同梱 `Phase estimation with partially randomized time evolution.pdf` は表紙がarXiv v1、2025-03-10付である。既存source/研究文書の`paper_d6`/「論文Eq. (D6)」という名称は、その名称のまま証拠目録に残す。しかし、確認したPR論文v1/v2・刊行版のAppendix Dから、その名称の一次式への対応を確定できなかった。同梱の別原稿 `Evaluation of gate numbers for ground state energy calculations using higher-order.pdf`（Abe et al.）にはAppendix F Eq. (F6)として同形式の推定式がある。**U01：旧D6名称との版・出典・符号規約の対応を追跡するまで、PR原論文で確認済みの式とは表示しない。** 既存結果・statusを訂正しない。この留保はsaved wrapper費用の監査を妨げないが、そのPF係数をPR原論文由来の保証として使うことを妨げる。

モデル名M0/M1/M2は既存の実験段階M1/M2と異なる。以下では **AX-M0/AX-M1/AX-M2** と表記し、利用者指定の三層モデルに対応させる。

## 2. 条件の対応表

原論文側の短い記述はv2の該当節・式を要約したもの。最終列は本研究の比較判断であり、原論文の主張ではない。

| 項目 | 原論文v2：一次箇所 | 現行Track A：source/保存契約 | 比較方法・不一致 |
|---|---|---|---|
| task/状態 | Sec. II、App. E冒頭：ground-state energy RMSE、exact eigenstateを仮定 | `_target_signal`：保存状態のRayleigh residualを確認し、T=0.8のcomplex signalを参照。state preparation費用なし | energy RMSEとcomplex-signal誤差は別task。原論文総QPE費用と現行総signal費用の数値比はN/A |
| 水素鎖 | App. C.2：STO-6G、1.4 Bohr | H4、STO-3G、1.00/1.30 Å、DF rank12 | basis・距離・Hamiltonianが異なる。原論文の水素鎖係数を同じinstanceとして比較しない |
| DF表現 | Sec. VII.B、App. C.1、E.1.b：外側rankと内側rank、relative basis変換 | `DFHamiltonian`：constant + dΓ(one_body) + Σλ_l[dΓ(g_l)]² | 外側Lと内側ρ_lを区別。H6/H8でrank12を固定しない |
| basis/weight | App. C.3：orbital/weight最適化、symmetry shift | 保存されたDF fragment、固定prefix、identity extraction、source規則に基づくordering | BLISS等を実施したとの主張なし。exact RTE λとfragment ranking proxy λを区別 |
| mapping/order | App. D：symmetry-preserving Bravyi–Kitaev、lexicographic term orderによる係数評価 | DF circuitのfermionic/Jordan–Wigner表現、Qiskit/OpenFermion state orderingの明示変換 | 別mapping/順序のPF係数を同じ実装の係数として移さない |
| deterministic PF | App. A.3 Eq. (A35)、E.1：主にsecond order | `df_partial_s2`、M1 dense signal path、forward-half/tail/reverse-half | 同じstage構成に対応する部分のみ比較。高次PFは別baseline契約 |
| random event | App. A.2：even Taylor orderのLCU、Pauli productとrotation | symbolic DF event → basis変換、I/Z/ZZ component作用・rotation | 論文のPauli rotation数とsourceのcomponent application数は異なる |
| cutoff | App. A.2–3：RTE展開とそのsignal関係 | canonical finite K=2/4、平均作用はdegree K+1の多項式 | finite cutoff biasを残す。無限展開でbiasなしという結論をfinite実装にそのまま移さない |
| normalization | App. A.2 Eq. (A26)–(A29)、A.3 Eq. (A38) | exact finite normalization b_K(τ)、全体B=b_K(τ)^(qr)、raw/correctedを保存 | exact finite値と解析上界を別columnで比較。上界のslackは予測不正確と同義でない |
| PF誤差 | Sec. III、VI.B、App. D、E.1 Eq. (E1) | signal評価ではcorrected/pf-exact-tail/targetとの差を直接保存。旧C_useは限定δ窓の経験envelope | C_gsδ^pはenergy量。C_useをaxis biasや厳密上界として使わない |
| RTE誤差 | App. A.2–3、App. B：normalization後のsignalと測定統計 | finite polynomialによるtruncation bias、B²によるshot増加 | finite bias、outer PF、統計を分離。absolute biasの加算をexact decompositionと呼ばない |
| basis transitions | App. E.1.b Eq. (E10)–(E11)：隣接relative rotation、内側切捨て | local cost proxyとrepeated full-circuit builder、boundary optimization | n_detだけでは費用を再現できない。保存basis transition/eventが不足する場合はN/A |
| controlled evolution | App. A.4、E.1/E.3：PF対称性を使ったcontrol | 現行full Hadamard wrapper、controlled diagonal evolution、各axis別transpile | 一律係数2や1/2を適用しない。対称controlの改善を強いbaselineとして別検証 |
| synthesis/単位 | App. E.1 Eq. (E6)–(E9)、E.2–3：arbitrary rotation、two-qubit、T/Toffoli、Hamming weight等 | Qiskit 1.3.0、basis `rz,sx,x,cx`、opt1、seed17、coupling mapなし | RZ count≠non-Clifford rotation≠T≠Toffoli。FT変換は誤差・ancilla・synthesis契約がなければN/A |
| wrapper/preparation | App. E冒頭の状態アクセス仮定、single-ancilla測定 | full wrapperのancilla/測定を含む。state preparation・backend quantum shotsは含まない | preparation感度は別項。総量子化学計算費用や実機runtimeとは呼ばない |
| rounds/shots | App. B、E.1 Eq. (E1)–(E5)、E.3 Eq. (E20)–(E23)：round schedule、B²、energy予算 | 一つのTのreal/imag軸、各failure α=.025、Hoeffding shot | round m、q、r、総Rの記号は翻訳が必要。round数や失敗確率の同一視は禁止 |

原論文の既知のdiscard比較（App. D.2）はground-state energy truncationである。現在のB0は同じfull-target signalに対するdiscard+PF誤差を含むので、別の誤差量で追試したという限定表現にする。

## 3. 会計式と分類

### AX-M0：原論文の前提を維持するモデル

原論文App. E.3 Eq. (E21)の会計構造を固定する：

\[
G_{\rm paper}=\sum_m2N_m\{G_{\rm det}N_{\rm stage}L_D2^{m-1}
+G_{\rm rand}\kappa\lambda_R^2\delta^2 2^{2m}\}.
\]

round範囲、rounding、N_m、δ、κ、error allocationはEq. (E1)–(E4)、(E20)–(E23)と著者archiveの仕様を確認してから凍結する。これは同式の会計構造であって、本書でQPE総費用を再現したという意味ではない。G_det/G_randを現行compiled wrapperで置換するとAX-M0ではなく、条件付き別モデルになる。

| 構成要素 | 分類 | 評価上の扱い |
|---|---|---|
| 指定回路とround scheduleの会計式 | 会計式 | units、state access、gate realization、roundingを維持して初めて再現可能 |
| normalization/誤差の解析不等式 | 仮定付き上界 | 仮定とslackを報告。保守的なfalse rejectionと実際の不適格を区別 |
| 小分子のC_gs/weight fitを大規模系へ移す部分 | 経験的予測・heuristic | 厳密上界ではない。現在のsignal taskへの転用は別の仮定を要する |

AX-1のM0は文献入力の来歴・会計項の対応の監査まででよい。保存値で同一taskへ投影できるnormalization等を独立に検査し、full QPE reproductionやFT優劣をAX-1の完成条件にしない。

### AX-M1：現行実装に対応する未校正モデル

task、T、identity、finite distribution、wrapper scope、cost unitを現行Track Aに揃える。出力を次の三つに分ける。

1. **action accounting**：n_det、n_fixed、E[n_rand]=qrΣ_n p_n(n+1)。sourceの保存`n_rand`は`ceil(E[n_rand]−1e−15)`、policy labelは`ceil_expected_applications_v1`。`W_action=N_total(n_det+n_rand+n_fixed)`は比較用action indexであり、RZ予測値ではない。未丸めE[n_rand]を主特徴にする。
2. **native cost prediction**：deterministic diagonal、basis変換、random components、control、scalar phase、wrapperごとの静的gate会計を同じ単位へ積み上げる。source-defined係数のみの未校正モデルをAX-M1とする。実際のrelative basis sparsity、event順序、boundary cancellationが不明なら不明のまま残し、generic構造仮定を明示する。最適化後compiler出力への近似なので厳密なcompiled-cost上界とは呼ばない。
3. **shot prediction**：exact finite Bと登録したPF/truncation **axis** bias予測からmarginとN_aを算出する。biasが参照値ならoracle-assisted。PF energy係数しかなければoperational Nは未定義とする。

既存`df_deterministic_step_rz_cost`はdiagonal会計とlocal compiled U_opsを組み合わせる。これをpure analytical modelと呼ばない。`rpe_hadamard_compiled_cost_proxy`、connected-cluster/order-stratified/boundary proxy等は既存実装として再利用可能だが、各校正データ・compiler・scopeを明示する。direct test cellをcompileして得た量を「安価な予測入力」に含めない。

### AX-M2：H4で校正する経験予測

AX-M1の未校正出力を残したうえで、少数parameterのcost補正をH4だけで開発する。初期fit候補はaxisごとに

\[
\widehat C_a=\theta_{0,a}+\theta_{D,a}n_{\rm det}
+\theta_{R,a}E[n_{\rm rand}]+\theta_{F,a}n_{\rm fixed}+\theta_{q,a}q,
\quad\theta\ge0.
\]

AX-1開始前に、非負least squares、主targetをone-shot RZ mean、feature/unit、fit weight、tie/欠測処理を固定する。featureの完全従属・rank不足があれば、n_fixed等の従属項をsource定義に基づいて事前規則で除き、parameterを減らす。係数を各物理費用の識別結果とは解釈しない。cost sampleのSE=0を無限weightにしない。candidate均等weightを初期規則とし、K・q・prefix単位のgroup診断で複雑化の必要性を判断する。新特徴の選択履歴を残す。

H4の未校正モデルとの照合と、H4でfitした後のin-sample精度は別に出す。leave-one-q-group等は内部診断に留める。既に観察されたH4 1.30 Åも新研究の独立held-outではない。H6初回ではH4凍結モデルを評価し、H6で改良した後はH8のみがその改良モデルの独立検査になる。

## 4. 予測時点の情報契約

| 情報 | AX-M0 | AX-M1 operational | AX-M2 operational | oracle-assisted |
|---|---|---|---|---|
| 原論文のHamiltonian・round/synthesis契約 | 必須。保存入力の来歴を確認 | 同一task部分だけ翻訳 | 同左 | 同左 |
| test instanceの登録Hamiltonian記述、L、prefix、q/R/K、exact λ_R、finite distribution | original条件の範囲 | 許可。ただし前処理費用を別記 | 許可 | 許可 |
| test stateのexact energy、PF signal、corrected bias、参照shot | energy oracle仮定を明記 | 禁止 | 禁止 | 許可したfieldと段階を明記 |
| test cellのactual compile cost・trajectory eventの再生成 | original compile scope以外は不一致 | 禁止 | 禁止 | cost oracleと明示した分解診断のみ |
| H4保存compiled/bias | original再現値の代用不可 | 未校正係数へのfitは禁止 | 登録development部分のみ許可 | 許可 |
| H6参照結果 | original同条件の再現以外N/A | 事前予測には禁止 | 初回評価後の改良は許可、以後H6=development | 評価段階を表示 |
| H8参照結果 | 同上 | 予測凍結前は禁止 | 予測凍結前は禁止 | 確認後の分析はexploratory |

exact λ_R、basis/fragment解析、低コストcommutator、state/energy surrogateにも古典前処理費用がある。input acquisition class、CPU/RSS、対象サイズに必要な計算を申告する。exact state/energyを使用したshot predictorは、cost部分がcompile-freeでもtruth-freeではない。

予測recordにはmodel version/source commit、calibration入力hash、予測時点、candidate fingerprint、unit/scope、input field来歴、oracle flags、bias/shot/normalization/cost予測、欠測理由を保持する。H8参照結果の前にrecordを凍結する。

## 5. costとshotの要因分離

同じdirect候補集合で、以下を別々に比較する。Nは精度とaxisを固定したshot、Cはone-shot wrapper費用、G=Σ_aN_aC_aである。

| 評価 | 使用情報 | 検査するもの |
|---|---|---|
| N_ref × C_pred | 参照bias/shotを許可 | cost modelだけ。既存M2に近い条件付き評価 |
| N_pred × C_ref | 参照compile費用を許可 | shot/eligibility modelだけ |
| N_pred × C_pred | 参照値を予測前に使わない | operational総費用・構成選択。両予測が定義できたcellのみ |
| N_ref × C_ref | direct参照 | 指定taskの参照会計。量子実験を実施した意味ではない |

**既存M2**の`predicted_work_by_metric`はdevelopment one-shot compile costにheld-out参照axis_shotsを掛ける。したがって固定構成の条件付き費用移送である。AX-1でそれをtruth-free予測として再分類しない。

AX-1のshot照合は保存bias/BとPM-2 shot式の対応確認が中心であり、独立shotモデルの正確さを実証するものではない。operational bias predictorが欠けた場合はU09を残し、cost-only oracle-assisted studyで閉じることを許容する。

## 6. 公平な報告と未解決事項

上界にはbound/reference ratio、適用仮定、false rejectionを出す。経験予測にはsigned relative error、過小評価、regret、不適格選択を出す。会計式にはscope差・未計上項・rounding差を出す。上界が参照費用より高いだけで「理論が誤り」とは言わない。

source、保存field、欠測は[証拠目録](track_a_ax0_evidence_inventory.md)、metric/selectorは[実験仕様](track_a_ax0_benchmark_protocol.md)に従う。U01はPF係数を原論文式として使う前、U02はbasis/event依存のAX-M1を評価する前、U09はoperational shot・eligibilityを主張する前に解消が必要。それまでは該当出力をN/Aとして報告する。missingを推定値で埋めて完全な対応表にしない。
