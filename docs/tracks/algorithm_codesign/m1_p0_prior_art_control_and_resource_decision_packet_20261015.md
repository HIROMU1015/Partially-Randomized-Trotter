# M1 P0：採択前の先行研究・±controlled実装・Track A接続 Decision Packet

2026-10-15 JST。状態：**`P0_EVIDENCE_COLLECTION_COMPLETE_M1_ADOPTION_PENDING`**。

ユーザーが依頼したP0（既存資料の読取りと整理）のみを実施した。**M1の正式採択、G0の新しい理論検証、V1の科学計算は開始していない。** Parent allocation D＋Eと既存のSTOPを保持する。[最新GPTレビュー原本][REVIEW]は37,750 bytes、SHA256 `b65ee709e408a8c3eabcccb644efc1d2d7f597a1c61dfbe9b02ebc72eec61230`でbyte保存し、先に公開した。

## 1. P0で確認できたことと、採択前に残る問い

| 項目 | 取得して確認できた事実 | まだ認定していないこと |
|---|---|---|
| qDRIFTの位相推定 | PR原論文は既にqDRIFTの正確なHadamard平均、arctanによるenergy補正、guiding overlap付きRPE解析を与える | これをM1固有の新しい結果とはしない |
| 内側乱択＋決定論block | Hagan–WiebeとComposite QDriftは既知の組合せ。富士通2026はqDRIFTの費用係数を部分乱択RPEの式に含める | 組合せを新アルゴリズムと認定しない。費用係数の存在だけでM1の位相保証が証明済みとも言わない |
| 一次位相不変性 | Gu et al.にはHermitian Kraus noiseの位相一次不変性がある。Liにはrandom-unitary QPEのgap/誤差保証がある | それらの定理がM1へ直接適用できるかの新しい証明は行っていない |
| ±制御 | PR Appendix A.4は乱択部分のUとU†へ**独立乱数**でRTEを適用する。決定論部分の半減と乱択部分の費用を区別する | 同じ乱択列を共有したqDRIFTを同じ平均と扱えない。M1の最終protocolは未固定 |
| 既存source | 旧RPEはordinary `diag(I,U)`。H6の`directional`方針もrandom tailはordinary制御で受け取る | source名だけから紙面の全±protocolやM1実装が完成しているとは言わない |
| Track Aとの接続 | 長roundへの費用集中を記録した旧H4 proxyと、有限時間H6 signalの資源記録がある | 今回確認した固定資料では、M1と整合する長roundのrandom部の総費用割合は未取得。新規集計していない |

以上は文献・sourceの対応結果である。M1と同じ仮定・同じ出力の**部分乱択qDRIFTのmulti-round coherent/RPE保証**を、調べた本文の定理として特定するところまでは至らなかった。この限定的な未特定は「その定理が存在しない」「M1が新規」という意味ではない。既存定理の直接系になる可能性、protocolの相違、既存FR研究との重複をGPTの採択判断へ返す。

## 2. M1の対象を他の既知問題から切り分ける

M1は、内側のqDRIFT平均を決定論blockと交互に置いたとき、最終的なground-energy/phase推定のbiasと信号を、通常のnorm誤差より有用な条件で保証できるか、という**未採択の候補**である。単にqDRIFTを使うこと、正scalar減衰を除くこと、単純固有値の一次位相が変わらないことだけでは差分にならない。[REVIEW][REVIEW] §§8,10。

| 照合する出力量 | 既存の代表資料 | M1へ移す前に必要な区別 |
|---|---|---|
| channel \(E[U\rho U^\dagger]\) のdiamond誤差 | Campbell、Hagan–Wiebe、Composite QDrift | ancillaを含めた実際のcontrolled問題への適用条件。system channelとcoherent平均を同じにしない |
| coherent first moment \(E[U]\) とHadamard信号 | PR Lemma A.1、A.3 | inner hの関数だけの場合と、Dと非可換にinterleaveした場合 |
| noisy channelの固有位相 | Gu et al. Thm1 | channel eigenoperatorと平均振幅行列、noiseの配置、摂動・縮退条件 |
| QPE成功確率 | Li Thm4.1 | register-QPE、random unitaryの扱い、gap/残差とsingle-ancilla RPEの条件 |
| RPEのRMS energy誤差 | PR ThmB.2 | overlap・branch・round別位相写像・damping。高確率ε保証と同一ではない |
| 有限時間complex-signal精度 | Track A H6 | energy精度でも長時間RPEの成功保証でもない |

「既知のphase-specific保証を超えるか」は、同じtarget、初期情報、gap/alias、overlap、失敗規則、controlled protocolを固定して初めて判断できる。P0ではそれらを新しい契約として採択していない。

## 3. 一次文献の該当箇所と対応

公式PDF9件を取得し、関連節を読んだ。**全論文の全定理を独立に検証したという意味ではない。** 版・著者・公開日・PDF bytes/SHA256・確認箇所は[文献locator CSV](../../../artifacts/track_b_m1_p0_preflight/2026-10-15/v1/primary_literature_locator.csv)。以下のページはPDFの1始まりページ番号であり、論文内の式番号と併記する。

### 3.1 PR原論文：完全qDRIFTの解析は既にある

[Günther et al., arXiv:2503.05647v2](https://arxiv.org/pdf/2503.05647v2)のSection IV、Eqs(22)–(26)、Lemma A.1／Eqs(A14)–(A17)（PDF p20）は、involution/Pauliを確率pで選ぶ等角qDRIFTをHadamard平均で解析する。本文の式(24)と(A14)は

\[
\sum_l p_l e^{-i\arctan(\tau)P_l}=\frac{I-i\tau H}{\sqrt{1+\tau^2}},\qquad
E Z=(1+\tau^2)^{-r/2}\langle\psi|(I-i\tau H)^r|\psi\rangle
\]

という既存の恒等式である。ここは論文のλ=1規約で、M1のh=H_R/λ_Rへ対応する部分。**P0で導出した新しい定理ではない。**

Appendix BのThmB.2（pp27–29）は完全qDRIFT/RTEのguiding-state RPEを扱い、qDRIFTのarctan位相からenergyを戻す写像、roundに共通の変換固有値を使う選択、damping/overlap条件を記す。従って「qDRIFTのenergy位相をdiamond norm以外で扱うことが初めて」という主張は支持されない。

一方、部分乱択のSection V、Lemma A.3／Eq(A41)（p23）、Appendix E.3／Eqs(E20)–(E23)（p42）の正確な平均信号は**RTE**を使う。内側qDRIFTの減衰が非可換DとinterleaveするM1の信号を、これらのRTE式へ単純置換できるという定理は確認箇所で特定していない。

同論文Appendix B.1のdepth/damping最適化とAppendix E.2にはqDRIFT対RTEのfactor-twoに関する記述が既にある。これをM1の新しい全PR改善率として扱わない。HWP・catalyst・共通角の費用はAppendix E.2、deterministic/random費用の分離はEq(E21)を参照する。

### 3.2 Hagan–WiebeとComposite QDrift：構成とchannel保証

[Hagan–Wiebe, *Composite Quantum Simulations*, arXiv:2206.06409v3](https://arxiv.org/pdf/2206.06409v3)は、Trotter側AとqDRIFT側Bへの分割を扱う。Definition3／Eq(27)、Thm4（p8）、Thm5（p9）、higher-order Thm2.1（pp3,13）が直接の入口。保証対象はchannelのdiamond距離、資源はoperator-exponential queryである。

[Pocrnic et al., arXiv:2306.16572v1](https://arxiv.org/pdf/2306.16572v1)はSecII D–E／Eqs(11)–(16)（pp6–7）で同じ系統を定義し、imaginary-timeとlocal compositeへ拡張する。構成、分割自由度、channel-norm保証は既知として残す。今回確認した節で、M1と同じnon-normalなcoherent平均からRPE信号を保証する定理は特定していない。

ただし「channel定理はM1へ一切利用できない」とも言わない。controlled問題へのlift、対象generator、保証の強さを対応させる作業は、必要ならG0の科学契約で扱う。P0ではその新しい適用証明をしていない。

### 3.3 富士通2026：qDRIFTを使う費用モデルも既知

[Kanasugi et al., arXiv:2603.22778v2](https://arxiv.org/pdf/2603.22778v2)はSecII B.2／Eqs(10)–(13)（pp4–5）に完全qDRIFT信号を、II B.3–4／Eqs(14)–(18)（pp5–6）に部分乱択とRPE費用モデルを記す。Appendix B／Eqs(B3)–(B5)（p21）はdeterministic/random費用を分け、**γ=1をqDRIFT、γ=2をRTE/TE-PAI**としている。STARではC_gate=1というlogical rotation会計であり、S4の各角度Clifford+T価格モデルと同じではない。

これらはM1の新規性判断で必ず照合する近接部分である。ただし、費用式が採用されていることと、M1レビューで未解決とされた±乱数・interleaved coherent平均・non-normal spectral weights・gap/overlapの保証が同じ仮定で証明されていることは分ける。今回読んだSection II／Appendix A,Bに、後者を一体で与える別定理は特定していない。未特定を論文の誤りの認定には使わない。

### 3.4 Günther博士論文：既知解析と資源修正の履歴

大学公式[PDF](https://www.math.ku.dk/bibliotek/arkivet/phd-theses/2025/phd25jg.pdf)、タイトルページJune2025を取得した。大学portalの学位授与/公開日は2025-10-02。旧flat URLは404だったが、公式defenseページのリンクから`/2025/`配下へ解決できた。**博士論文本文は取得済み**である。

Chapter2 pp47–119、特に§2.4/2.5、§2.9（qDRIFT pp77–79、partial RTE pp82–84、±制御p85）、§2.10 pp116–119を確認した。Ch2は主に初期PR preprintの再録であり、PR v2と別の独立再現や同じ版ではない。p85はUとU†へ独立乱数でRTEを適用することを明記する。

§2.10はHWP ancilla、depth reduction、qDRIFT prefactorの修正履歴を記す。Table2.2（p119）にはFeMocoのPauli/HWP aggregate費用の\(G_{det}/G_{tot}\)が保存されている。これはReiher/Liの外部benchmarkに対する全roundのモデル値で、Track AのDF rank・split・geometry/basis・δ窓と対応を確定した値ではない。本Packetでは割合の数値を移し替えず、**本プロジェクトの長round random割合として使わない**。補数や新しいforecastも計算していない。

### 3.5 一次位相不変性・random-unitary QPEの近接保証

[Gu et al., arXiv:2208.04100v2](https://arxiv.org/pdf/2208.04100v2) Thm1（p2）と同梱SI SecI–IIは、Hermitian Kraus noiseを持つunitary channelの位相の一次安定性を扱う。対象はchannelの固有operatorsで、M1の平均振幅行列そのものではない。**一次Hermitian摂動の位相不変性だけは独立新規性にならない**という既存レビューの注意を具体的な先行定理へ対応付けた。SIのEqs(S7)–(S8)（p8）はchannel固有値のisolation distanceによる摂動収束条件、Eqs(S11)–(S16)（p9）は左右固有operatorsと信号展開も扱う。従ってgapやnon-normal spectral weightsを扱う一般的枠組みまで新規としない。ここでのchannel/induced-trace-norm条件をM1の平均振幅・実際のRPEへ対応付ける証明は未実施で、M1が直接系かどうかは未確定。

[Li, arXiv:2111.10430v3](https://arxiv.org/pdf/2111.10430v3)はThms2.2/3.1/4.1（pp8/11/16）、Section4（pp12–16）でresidual/gapとrandom-unitary QPEの成功条件を扱う。register-QPEの条件をsingle-ancilla RPEのM1信号へそのまま転記してはいない。適用可能性の最終確認は残る。

[Campbell, arXiv:1811.08017v2](https://arxiv.org/pdf/1811.08017v2) Appendix D／Eq(D1)（p8）にはcontrolled Hamiltonianを用いる実装がある。qDRIFTやcontrolled-QPEへの使用自体は既知。[Resource-Optimal IS, arXiv:2603.13495v1](https://arxiv.org/pdf/2603.13495v1) SecII Thm1は固定ensembleのcostと統計量の交換を扱い、M1の物理的な平均演算子変更によるphase biasを認証する定理ではない。

### 3.6 検索の到達範囲

指定4系統と、直接関連するqDRIFT・位相摂動・QPE・ISを照合した。対象は上記9PDFの関連節、既存FRの先行研究表、指定レビューの引用である。arXiv上でqDRIFT＋partial＋phase/bias、non-unitary phase estimation等も限定検索したが、全citation networkの網羅調査ではない。検索ヒットが無いことやkeywordが無いことを不在証明に使わない。

## 4. ±controlled protocolの定義照合

### 4.1 既存paperにあるもの

PR Appendix A.4／Eqs(A43)–(A46)（p24）は\(C_\pm U=|0\rangle\langle0|\otimes U^\dagger+|1\rangle\langle1|\otimes U\)を定義する。対称PFのhalvingと、一般の乱択productがその対称性を持たないことを区別し、乱択blockはUとU†を分けて**独立にRTEで実装**する。半減するのはdeterministic部分であり、randomized部分まで同じfactorを掛けない。

一般Hadamardの照合対象は、二branchの演算子をB0,B1としたときの\(\langle\psi|B_0^\dagger B_1|\psi\rangle\)である。以下は既存定義とレビュー§8.2の対応整理であり、新しいM1保証の証明ではない。

| 候補protocol／既存実装 | 乱数の意味 | first momentを照合する対象 | P0の状態 |
|---|---|---|---|
| ordinary `diag(I,Uω)` | shot内に一つの列ω | \(\langle\psi\vert E[U_\omega]\vert\psi\rangle\) | 旧RPE wrapperの明示契約 |
| paperのpartial RTE ±実装 | U側とU†側で独立draw | paper Eq(A46)とRTEの正確なLCU平均 | 原文確認済み。M1のqDRIFTへ代入した証明はしない |
| M1独立branch案 | B0とB1の列が独立 | \(E[B_0^\dagger B_1]\)。単一blockの負側が正側lawのliteral adjointである等の条件を満たす場合の因数分解を要する | interleaving、順序、物理時間の定義が未採択。いきなり\(M^2\)を全回路の式にしない |
| M1 shared full-block案 | 全blockのUωとそのliteral adjointを共有 | レビューの指摘する\(E[U_\omega^2]\)型。一般に\((E U_\omega)^2\)とは異なる | 共有の仕方を指定せずRPE保証へ進めない |
| gateごとに同じlabelを共有して符号だけ反転 | 同じorderのgate列に各\(C_\pm\)を適用 | 実際の二branchのordered product。full-blockのliteral adjointかは別確認 | paperの対称PF恒等式を任意乱択列へ使わない |

最後の二つも同じ「shared」と呼べてしまう。負branchを作る際のinverse、gateの逆順、sign、scalar phaseを明示し、まず対象の\(E[B_0^\dagger B_1]\)を確定する必要がある。これはsource実装を変える指示ではない。

### 4.2 今のsourceが実装している範囲

| 固定source | 読み取れる契約 | M1への境界 |
|---|---|---|
| [RPE wrapper][RPE-WRAPPER-SRC] `build/_build_validated`、[説明][RPE-WRAPPER] | H–already-controlled evolution–(Y軸S†)–H。`signal_definition=Z_m=<psi\|U_m\|psi>`、追加controlもresamplingもしない。\(e^{-iHt}\)ならIm Z=−sin Et | ±branchの二列生成をするsourceではない |
| [RPE benchmark][RPE-PROXY-SRC]、[proxy test][RPE-PROXY-TEST] | `ordinary_controlled_diag_I_U_m`をbinding。テストには`paper_cpm_halving`拒否例がある | ordinary用のcompiled proxyを±へ無条件に移送できない。今回はテストを実行せず静的に確認 |
| [Track A H6 native builder][H6-NATIVE] `build_partial_native`、`partial_native_from_step_requests` | supplied tailは`diag(I,U_event_sequence)`。deterministic halvesをordinaryまたはUNCONTROLLED/DIRECTIONALで実装。explicit RTE requestをreplayし、新sampling/並べ替えをしない | `symmetric_directional`という名称だけでM1のshared/independent ±lawと同一にしない |
| [RTE source][RTE-SRC]・[規約][RTE-CONV] | finite paired Taylor、正規化、符号付きevent、controlled relative phase | qDRIFTの新runnerやM1のRPE保証の完成証拠ではない |

shot間のfresh IID、±branch間の独立性、X/Y間の依存、同じcost標本の両axis再利用は別の概念。既存wrapperの`fresh_iid_trajectory_per_hadamard_shot_verified=False`はbuilder単体で量子shot freshnessを検証しないことを表す。既存のclassical MC列を再利用した事実から、量子測定のIID保証を推定しない。

## 5. Track A/RPEへの接続で既存資料が言えること

### 5.1 既存の誤差・confidence会計

[RPE resource accounting][RPE-ACCOUNT]（version3）の現在のspecializationは\(\rho_{\star,lb}=1\)。観測半径はattenuationとfinite-RTE residualを戻し、statistical phase budgetから十分Hoeffding shot予算を作る。PF係数は外部入力で、経験値なら`empirical_screening`。厳密係数、alias-free、適切な状態、phase budget、fresh IID、round/axis union bound等が揃った場合だけconditionalな`certified`が得られる。

full-H ground stateを与えることだけではeffective PFのeigenstate/単位半径を保証せず、\(C_{use}\)は経験的delta窓のenvelopeである。**これらの既存APIはM1のnon-normal平均のoverlapやphase biasを既に証明しているものではない。** PR ThmB.2のRMS ε、既存APIの高確率ε条件、H6のcomplex-signal εを区別する。

### 5.2 長round費用集中とrandom部割合は別

[旧H4 G08/M08][RPE-LATE]はH4 linear1.0 Å/STO-3G/8qubit/DF12/\(L_D=3\)、δ=.02/r32/K2、Qiskit1.3.0 rz/sx/x/cx opt1/seed17の文脈で、最後3roundの費用割合90.94%を記録する。全deterministic \(L_D=12\)候補では83.03%。scheduleの最後3roundはq=32768/65536/131072、直接wrapper holdoutはq≤32までである。

**これは全round間の費用配分であって、random circuit部が費用の90.94%を占めるという測定ではない。** 大qではvalidated proxy/移送感度の範囲も残る。P0では既存費用を再集計せず、random/deterministic fractionを新しく計算していない。

[H6独立確認][H6-RESULT]は、linear H6/1.00 Å/STO-3G/12 modes、tol-only1e-8 DF rank19、cutoff0、prefix5、q1/2/4/8、T=.8（δ=.8/.4/.2/.1）、r4または2、K2/K4、Qiskit1.3.0 opt1/seed17、state-preparation除外の有限時間complex信号の記録である。32trajectoryのnative平均と十分shot予算はあるが、未観測Taylor次数・経験的allowanceが残る。これは長時間RPEのrandom割合、energy保証、母平均winnerの認証ではない。[H6科学レビュー][H6-REVIEW]。

legacyの[費用plot helper][LEGACY-COST-PLOT]にはdet/random fractionを表示する実装能力がある。しかしplot用の旧解析model、paper E21、博士論文Table2.2を、現在のcompiled・同state/precision・全roundの実測分解へ読み替えない。P0対象の固定資料では**M1と同じprotocolに使える長round random割合は`NOT_ESTABLISHED`**、state-preparationとclassical online費用も一致した契約の値は`NOT_MEASURED`。プロジェクト全履歴に一切存在しないという断定ではない。

## 6. 既存FR研究との内部的な重複

[FR正本][FR-CURRENT]は有限paired-Taylorの平均振幅を正scalarと方向依存成分へ分け、phase/radius境界、情報層、反例を`TECHNICAL_NOTE`として保持している。scalar分離、Hermitian/anti-Hermitian分解、単なる高次phase-lagは既知部分として整理済み。数値は`MECHANISM_ONLY_NO_PRACTICAL_GO`のまま、追加計算を止めている。[既存先行研究表][FR-ART]もGu、Li、PRとの対応を記録している。

M1はqDRIFT内側とRPE全体を対象候補としているためFRと自動的に同一ではないが、「phaseと減衰を分ける」「norm誤差よりphase誤差が小さい」という表現だけで新しいRQにしない。FRの未解決事項をM1で解決済みともみなさない。今回はFRの証明・数値監査を再実行せず、既存closureを維持する。

## 7. GPTへ返す採択前の判断事項

以下は採択条件の**判断材料**であり、Codexが新しい定理・protocol・実行契約を採択したものではない。

| 論点 | P0の証拠 | 採択前に判断が必要なこと |
|---|---|---|
| 主結果の差分 | 完全qDRIFTの信号/RPE解析、Composite構成、γ係数、一次位相不変性は先行にある | M1が新アルゴリズムか、既知手法の新保証か。具体的な定理statementが既存結果の直接系か |
| protocol | PRは独立RTE ±branches、旧RPEはordinary、H6 directionalはtailをordinaryで受け取る | ordinary／独立±／sharedのどれを対象にするか。物理時間と積順序も含める |
| 対象誤差 | eigenvalue phase、survival signal、RPE energy、channel normは別 | どの誤差・RMS/高確率保証が主claimか |
| 成立条件 | 最新レビューはgap/alias/overlap/non-normal weightsを未解決とする | 既知Gu/Li/PRの仮定との対応、必要な条件が実taskで意味を持つか |
| 費用の接続 | paper E21/富士通B3–B5に分離モデル、Track Aに限定compiled記録 | rotation/Toffoli/T/CX、HWP/STAR、準備・古典費用のどのtask会計を使うか。未取得割合を0にしない |
| G0の範囲 | P0は証明を開始していない | 採択するなら担当、定理scope、強い既知対照、計算との境界を別途固定する |

今回の資料で「M1と同等の完全保証は無い」「M1なら2倍改善」「総T費用が半分」「必ず研究継続／終了」を認定しない。既知のfactor-twoの記述と、新しい部分乱択の信号保証と、実task総費用の三つを分けて判断する。

## 8. 取得性・残存不足・停止

- 指定のPR/Hagan–Wiebe/富士通2026/博士論文を含む9一次PDFと、固定GitHub21ファイルを取得した。GitHubはfixed blobとのbytes/SHA256一致を確認。PDFは取得版のhash・ページ数・確認locusを記録した。
- 博士論文の旧URLは404、公式リンクから新URLで取得済み。取得失敗と文献不存在を混同していない。
- GPTレビュー原本はbyte保存済み。レビューが参照したClaudeの`レビュー_inner.md`は今回の提供ファイルとしては未取得で、GPTレビュー内の記録だけから原本を復元したり全文確認済みと主張したりしていない。M1のP0範囲はユーザー指示とGPTレビューから確定できる。
- PDF/抽出全文は一時cacheで読取り、論文全文をリポジトリへ複製していない。再取得可能な公式URLとhash、関連箇所のlocatorを残す。全citation網羅と既存定理のM1への新しい適用証明は未実施。
- 科学source、contract、旧results、authorization、marker、STOPは変更しない。新runner/source実装、新証明、科学計算、N/資源forecast、compile/synthesis、operator guard、matrix、trajectory、sampling、LPは0。

**P0はここで完了。次は、ユーザーが明示的に開始を承認したGPT科学判断でM1採択の可否と必要ならG0の範囲を決める。CodexはG0/V1へ自動進行せず、mandatory STOP。**

## 固定GitHub参照

[取得索引](../../../artifacts/track_b_m1_p0_preflight/2026-10-15/v1/evidence_index.json)にcommit・bytes・SHA256を記録した。

[FR-ART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/finite_rte_phase_amplitude_prior_art.md
[FR-CURRENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/fr_research_claim_and_manuscript.md
[H6-NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/src/trottertracks/resource_applicability/ax2a_native_df.py
[H6-RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/docs/research/track_a_h6_independent_confirmation_result_v1.md
[H6-REVIEW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/docs/research/track_a_h6_independent_confirmation_scientific_review_20261013.md
[LEGACY-COST-PLOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/src/trotterlib/df_cost_plotting.py
[REVIEW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ef2c62b4b6fea5dcef2674a29db821cf91bab6/docs/research/track_b_next_direction_gpt_scientific_review_20261015.md
[RPE-ACCOUNT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/rpe_resource_accounting.md
[RPE-LATE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research_direction_late_round_proxy.md
[RPE-PROXY-SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/src/trotterlib/rpe_hadamard_compiled_cost_benchmark.py
[RPE-PROXY-TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/tests/test_rpe_hadamard_compiled_cost_proxy.py
[RPE-WRAPPER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/rpe_hadamard_interrogation.md
[RPE-WRAPPER-SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/src/trotterlib/rpe_hadamard_interrogation.py
[RTE-CONV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/rte_conventions.md
[RTE-SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/src/trotterlib/rte.py
