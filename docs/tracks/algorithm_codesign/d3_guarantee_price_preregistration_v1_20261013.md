# D3：誤差保証と合成価格の頑健性比較・結果前計画v1

2026-10-13 JST。**D3_PLAN_FIXED_AWAITING_TECHNICAL_SOURCE_PREPARATION**。
今回の許可は数式確認、静的調査、key洗い出し、比較設計・実行契約草案と公開まで。
登録資源再会計・新合成は0。execution runner/source/authorizationは未作成。このcommitを実行sourceと呼ばない。
mandatory STOP。

## 1. 採択済み目的と入力

[GPT統合レビュー原本](../../research/track_b_d1_d2_integrated_scientific_review_20261013.md)をbyte一致で保存。
原本34,473 bytes、SHA256：
adaff342a1e263873e4dd2bd69bf3eaea51da0f2a639e55bb91ca5dac59e4949。
remote取得commitは b51924226f89273f70ab2f76254c1a8939f91321。

目的は、同じ物理時間発展・accuracy/confidenceで、表現とrの選択が共通Taylor保証と実際の有限精度合成価格へどう依存するかの識別。
P5が必ず勝つこと、新アルゴリズム採択、最終総資源優位を成功条件にしない。

固定development入力は3-system-qubit explicit involution provider：

- \(p=(1/5,3/10,1/2)\)、\(\tau=5/7\)、\(\sigma=+1\)、状態000。
- \(Q_0=Z_0\)、\(V_1=R_{XX01}(\pi/4)\)、\(V_2=R_{XX12}(\pi/4)R_{ZZ01}(\pi/4)\)、\(Q_i=V_i^\dagger Z_iV_i\)。
- \(R=\sum_i p_iQ_i\)、目標 \(e^{-i\tau R}\)、有限平均 \(P_5(-i\tau R/r)^r\)、\(r=1,2\)。
- ordinary、partial-return tail、closed P3 tail、local full-return、closed P5 full、matched CTSの六方式・12構成。
- 同じ明示I1 access。geometry、DF rank/orbital basis、split \(L_D\)、決定論PFのdelta windowは本taskに該当しない。
- 単位回転 \(R_P(\theta)=e^{-i\theta P/2}\)。

主比較はpartial/P5に**双方とも同じr=1/2の選択**を認める比較。固定r対照も別保存する。
全六方式へ必要36 unique keysを同じ新精度で取得する。新r/provider/precision grid/seed探索/SP/DF/分子/Track A/N3は追加しない。

## 2. 二因子比較

| ledger | Taylor保証 | 合成価格 | 用途 |
|---|---|---|---|
| G0_P0 | 旧 \(3|x|^6/720\) | 保存D1 \(10^{-6}\) 価格 | r1 D1＋r2 D2のanchor |
| G1_P0 | Hermitian \( |x|^6/720\) | 同じ保存価格 | 保証だけの変更 |
| G0_P1 | 旧保証 | 一条件 \(10^{-7}\) の新sequence価格 | 価格だけの変更 |
| G1_P1 | 改善保証 | 同じ新sequence価格 | 交差比較 |

native誤差**会計**は四ledgerとも旧 \(10^{-6}\)。新sequenceの余剰精度をNへ返さない。
同じG内ではN、moment、range、acceptance、lawが一致する。
将来G0_P0のexact reproductionを要求するが、旧結果・分類を上書きしない。本準備ではその再会計も未実施。

費用 \(G_{g,p}(h)=A_{g,p}+B_g h\) について
\[
D_{\rm guarantee}=G_{1,0}-G_{0,0},\quad
D_{\rm price}^{(g)}=G_{g,1}-G_{g,0},\quad
D_{\rm interaction}=(G_{1,1}-G_{1,0})-(G_{0,1}-G_{0,0})
\]
を保存。切片・slope、整数Nのceil効果、固定rの差、r選択後のenvelopeを分ける。比だけで効果を加算しない。

## 3. 共通誤差・confidence

[独立数学確認](d3_mathematical_confirmation_20261013.md)で積分剰余と反復を導出。
実数時間、Hermitian、\(\|R\|\le1\) が必要。単なるHermitian性ではnorm因子を落とせない。
新保証をrounded mean/native演算子自身のTaylor保証と取り違えない。

\[
e=8\rho+16(1+\rho)10^{-6},\quad
b_g=2[(1+\delta_g+e)^r-1],\quad s_g=1/200-b_g.
\]

旧root256 bit、proposal160 bit、\(\rho=\eta=10^{-12}\)、interval100 dpsを維持。
Taylor productとimplementation productを分離保存。finite-bitは保存dyadic qとexact rational weight \(a/q\) を使用し、
probability roundingを平均biasへ二重加算しない。s<=0はINCONCLUSIVE、精度・r変更による救済なし。

production step boundsと旧log上界を用いて
\[
N=\left\lceil \log_{\rm upper}(2/\alpha_{\rm axis})
\left(\frac{2M_{2,\rm step}^{r}}{s_g^2}+\frac{4W_{\rm step}^{r}}{3s_g}\right)\right\rceil.
\]

旧log_upper/accepted_cap/resource-tailを維持。exact signal、observed reference moment、
matrix residual、full exact normalizerをbudget oracleにしない。
axis accuracy \(1/200\)、complex上界 \(\sqrt2/200<1/100\)、
\(\alpha_{\rm axis}=49/24000\)、row resource failure \(1/12000\)：
\[
24(49/24000)+12(1/12000)=1/20.
\]

各12-row ledgerの同じconfidence条件を比較する。四ledgerは代替resource forecastであり、
**96 axesの一つの同時量子実験についてfamilywise0.05を保証したとは呼ばない**。
実際の量子shotは0。将来の同時実験には別contractが必要。

## 4. 一つの新合成条件

提案・結果前固定はprimitive strict operator precision \(10^{-7}\)。
一段だけ厳しくして旧 \(10^{-6}\) 会計への適合を維持し、価格変化を分離する。epsilon/seed gridを作らない。

- pygridsynth 2.0.0、固定runtime/tool identity、seed0、100 dps、旧全optionsを維持。
- backend requestは既存adapterどおり \(\epsilon/4=1/40000000\)。
- 36 positive rational atan ratiosに各一回。全六方式/rのunionは36件、未使用old cache keyは0。
- 同じraw-radian angle、scale1、up_to_phase=false。W位相も保持し、projective guardを使わない。
- 新sequenceとactual adjointを正負primitiveに使用。符号別の独立合成をしない。
- 全sequence bytes/hash、T/T†/1Q/W、strict error、inventory/job/worker/source/contract/tool/marker bindingを保存。
- P0は旧cache、P1は新cache。missing/failed keyを旧looser sequence、安いold/new選択、別seed、再合成で補完しない。
- 36件の完全coverage前にP1の全方式順位を主張しない。

[key inventory](../../../artifacts/track_b_d3_preparation/2026-10-13/v1/key_inventory_plan_v1.json)に全使用row、old/new key spec・hashを固定。
旧key hashのJSON whitespaceまで維持。全new priceはnull、合成実行0。
新generatorではなく保存使用表から列挙した。

## 5. 再会計とD2相殺の引継ぎ

r1はD1 saved step IR、r2はD2の同じsymbolic inverse passを固定する。
将来は保存1890 eventを一度読み、literal費用とR multiplicityから新T/1Q価格へ戻す。
新provider matrix、signal、native回路、compilerは不要。CXは価格条件だけでは変わらない。

\[
c_{\rm step,T}=Z\left(d_{\rm literal,T}+\sum_j\mu_jt_j\right).
\]

1Qもnew sequence価格を使う。Tは1Qに含まれるので両者を足したscalarを作らない。
[静的literal証明](../../../artifacts/track_b_d3_preparation/2026-10-13/v1/static_boundary_price_certificate_v1.json)で、
保存r2六方式の先頭word集合とinverse-reversed末尾word集合に、Rを含む共通wordがないことを確認。
従ってどの二step連結もRを含む相殺は不可能。削減は固定literal gateだけであり、新sequence価格に依存しない。

元D2のper-trial savings \(S_{i,k}\) を使い
\[
C_{1,k}=c_{\rm step,k},\qquad C_{2,k}=2Zc_{\rm step,k}-S_{i,k},\qquad
G_{i,k}(h_k)=2N_iC_{i,k}+2N_iZ_i^{r_i}h_k.
\]

219,303 pair・相殺深さ・削減価格を再評価していない。depth×costのjoint trace欠測も残る。
new Nが変わる場合にold総費用差をそのまま差し引かず、per-trial savingsを戻す。
any-zeroはinit前判定、零trialも試行数へ含める。
hard/tailはnew N、同じproduction acceptance boundとaccepted_cap、new-price worst saved single-step eventを用い、
保守的 \(r C_{\rm worst}\) を維持する。旧hard/tail totalsをnew価格へ流用しない。
sequence展開後の追加相殺、commutation、角度融合、whole-circuit compilerは実施しない。
古典online費用は未取得で、ゼロにしない。

## 6. 事前予測・保存物・解釈

保証のみの変更では同rの共通biasが減り、固定moment/rangeならNは増えない。
Taylor寄与は \(r(\tau/r)^6\) なので小rに保証改善が強く作用する。一方、実装誤差はrとともに累積する。
これだけで新しい最良r・表現を決定しない。

価格側は既存root/tail/literal key依存性から
\[
\Delta G_T=a_0+\sum_j a_jt_j+\Delta B h_T
\]
を事前モデルとする。未取得の価格幅・整数tieを合法な合成結果と呼ばない。
全36件を取得し、rootだけの移送で全方式に勝ったとしない。

将来の保存物：

1. 48 ledger-rowのexact rational bias/N、moment/range/acceptance、T/CX/1Q切片・slope、workspace、hard/tail。
2. fixed-r partial/P5と同r選択、全方式h=0順位・全非負h最小包絡線、quantum-coordinate Pareto。
3. 保証だけ・価格だけ・交差項、残余幅/ceil、key/root価格差。
4. formal hとphysically attainable h vectorの区別。少なくとも \(h_{1Q}\ge h_T\) を明記し、未知prep/readout価格を取得済みとしない。
5. 全36 keys/jobs、cache、source/contract/tool binding、marker、inner terminal・outer status、partial/error/log/resource evidence。

Nは十分試行数であり最低必要shot数ではない。新precisionは同providerのdevelopment移送で、
独立Hamiltonian validationではない。維持・反転・tieを同じ規則で保存し、P5勝利や後付けmaterialityを成功条件にしない。
重要な科学的解釈・一般化・論文化・追加providerは取得後のGPT判断。

## 7. 資源上限と未実装事項

準備時CPU32 logical、MemAvailable約51.6 GiBをread-only確認。
旧17新規key workerのCPU合計7.855002秒、max0.476555秒、RSS165,108 KiB。
旧全runの631.759秒にはmatrix等も含まれ、合成時間やonline sampler費用に読み替えない。
新precisionの実行時間は未測定。GPUは不要でqueryしていない。

caps：workers2、numeric threads各1、CPU total1800秒/per-key20秒、
parent/worker RSS各512 MiB、aggregate1536 MiB、per-process address1536 MiB。
output16 MiB、log64 KiB/key、cache4 MiB/36件、sequence20,000 chars、
review files512 KiB以下、shot forecast \(10^8\)/axis、48 rows、saved metadata1890 events、new synthesis36 calls。
wall capはtotal/phase/per-keyともnull。polling timeoutと科学runのwall上限を混同しない。
guardはworkerと外側observerの両方でCPU/RSS/log/output/callsを監視する。

独立keysを2 workersで並行化、BLAS/OpenMP/MKL/NUMEXPRを1へ固定。
同じratioの取得・係数を共有し、8×8 matrix検算・全pair列挙・IR複製を除く。speedup倍率は主張しない。

**今回の計画範囲では未実装：**

- D3専用ledger runner/worker/outer observer、fixed input/tool/source/contract/manifestのlaunch gate。
- 旧log/accepted_capの精度を継承するbudget関数、価格adapter、比較・streaming evidence。
- off-domain semantic/guard/error-accounting/inventory tamper/cap/partial failure tests。
- execution sourceのfull SHA、source manifest、old protected ledger、source-bound final technical verification。

D1/D2 runnerは別契約で凍結されている。旧markerを解除して流用起動しない。
旧独立review・authorization要件は旧runに維持。D3は新sourceの技術確認と明示実行承認が必要。
採択済み契約に新科学問題がなければ、source準備のたびのGPT詳細レビューを追加条件にしない。

## 8. provenanceと停止

最終focused testsは15件PASS。人工scalar・literal word・key identityのみで、登録fixtureを読まない。初回は14PASS/1FAILで、逆順の組合せがR相殺を許す人工fixtureの誤った期待を修正し、失敗記録を保持した。source/protected362 hashes、旧result165 files、親dirty50 file identitiesの不変を確認。immutable CI・外部再現とは呼ばない。

基点4cf2827c14fec96613a60bb1427287b30766d5daから専用branch
track-b-d3-guarantee-price-preparation-20261013を作成。
親worktreeのdirty変更をcopy/stage/reset/clean/stashせず、review原本だけ明示受領。
旧D1/D2 code/contracts/auth/results/consumed markers/STOPはbyte不変。
[design contract](../../../artifacts/track_b_d3_preparation/2026-10-13/v1/comparison_design_contract_v1.json)は
science_execution_authorized=false、source_commit=null。新authorization/markerは未作成。

将来は新D3 source Sのauthorization-only**直接子A**、remote A一致、clean worktree、
fresh namespace、exclusive markerを要求する。runs1/retries0。
tool/guard/cap/coverage/output failureならtechnical inconclusive、全partial記録を保存してmandatory STOP。
precision/seed変更、second-best、retryによる救済なし。完了後も次stageを自動認可しない。

数学・静的確認に、GPT採否を必要とするblocking scientific issueは見つからなかった。
次工程は上記のCodex技術source準備。その完了と実行条件照合後に別途明示one-shot指示へ進める。
計画を固定しただけで研究方針全体の再レビューを要求しない。

**mandatory STOP。登録D3再会計・新合成・旧run再開・他研究への移行は実施していない。**
