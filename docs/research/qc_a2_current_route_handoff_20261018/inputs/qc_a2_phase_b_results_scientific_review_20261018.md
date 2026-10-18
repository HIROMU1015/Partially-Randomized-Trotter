# A2固定DF pilot Phase B：結果の科学的解釈と継続可否レビュー

- 作成日：2026-10-18 JST
- 文書区分：ユーザーの「レビューを開始して　丁寧に考えて」に基づく重要科学レビュー
- 対象：④のA2有限step相対形式保証、固定DF実入力pilot Phase Bの結果・証拠取得費用・今後の投資判断
- 結果固定commit：`817d27b5a9af076b6205c3d54a7b60f646c6ad6f`
- 最終公開commit：`c1b80c0fabc1706d69585d993cbecfa12e4d09a6`
- Branch：`qc-a2-phase-b-execution-20261018`
- 科学SOURCE S：`8d483f9bfa5983bb52fef55887e2f9f73a790511`
- 正式科学契約C：`a7f8307123f6ba5bbb8d8b84e124400ac04a60ce`
- 状態：レビュー完了。以下の研究方針は推奨であり、正式採択・GitHub状態変更・新規科学実行承認ではない。

## 0. 結論

**現行の「係数三角capでPSD化し、raw K0行和とCauchy型有限剰余からRF0/RF1/EW2の認証step優位を得る」実用化経路については、今回のpilotで区切り、追加分子・精度走査・SPEC-08へ進まないことを推奨する。**

これはA2一般、低エネルギーPF理論一般、または量子化学の保証研究一般の不可能性の判断ではない。RF0/RF1/EW2の数学的成立は撤回せず、理論・認証実装・実入力での負の比較結果を保存する。A2全体の正式状態はユーザーの別判断までは`NOT_ADOPTED`を維持する。

推奨を前回の限定継続から変更する理由は、単に一例で負けたことだけではない。

1. 実装上の未解像・中断ではなく、同じ固定入力・条件・証拠を両側へ与えた認証比較が完了し、対照側の十分stepが明確に大きかった。[E1–E3]
2. 同じ認証値を固定したままのd・α変更では順位が逆転しない十分条件を、既存の二つの確率上界から紙上で確認できる（§5）。これは新しい数値再解析ではない。
3. 現RF適用域の端は、今回対照が受理したstepより小さい。同じΛと適用域のままtrialやK_SOSだけを改善しても、現対照のstepに届かない（§6）。
4. 残る改善には、単なる算術精度の増加ではなく、構造を保持する有限保証、より有用なPSD証拠、または別の費用目的を要する。それらを実現する新しい構成は今回得ていない。

一方、**約343秒をすべてA2固有の取得費として扱ってはならない。** 保存計測では構造付きCbar2対照の生成・検証が時間を占め、相対側のK_SOS計算は軽い。したがって、量子側の十分stepでは不利だが、古典証拠費まで含む全指標で一様に劣ると結論する証拠はない（§8）。この留保を、当初のstep優位という目的を事後的に変更して成功とする理由にも使わない。

## 1. 読取範囲、証拠層、実施していない処理

### 1.1 実際に確認したもの

GitHub connectorを用い、固定commitの結果報告[E1]、保存summaryの該当範囲[E2]、raw resultの取得可能な内容[E3]、実行承認記録[E4]、正式契約の説明[E5]を確認した。Branch HEADは最終公開commitと一致した。[E6]

数学・比較の正本としてEW2監査の有限step比較[T1]を参照し、固定SOURCEのstep predicate・bracketing[S1]、U/K_SOS生成[S2]、grouped交換子と二経路min[S3]を静的に読んだ。前回の重要科学レビュー[P0]は、この会話に実在するMarkdown全文を読み、GitHub保存copyの冒頭・結論との対応も確認した。

SOURCE全31ファイル、全proof、全checker、全Pauli witnessを今回独立再検証したわけではない。raw JSONと長いツール出力には表示上限があり、数値と条件は保存summary・結果報告の対応部分で補った。約67 MBの提案証拠全体を読み直したり、checkerを再実行したりはしていない。公開全ファイルのbyte/SHA一致は実行側の報告を根拠とし、本レビューで全公開bytesをもう一度独立照合したという主張はしない。

### 1.2 本文中の区分

- **保存結果**：既存checkerの値、enclosure、判定、計測時間等の引用。
- **既存の理論・仕様**：RF0/RF1/EW2、centered Cbar2対照、SOURCEの計算・受理規則。
- **今回の紙上推論**：これらの式の比・適用域から得られる限定的な比較結論、研究資源配分の判断。
- **未確認**：真のPF bias、元分子へのモデル誤差、改善後のcertificate値、別入力の結果、実QPE/native費、独立priority等。

新規Hamiltonian生成、係数decode、固有値・RDM、保存科学値の再集計・再ランキング、root再評価、Xiの新数値評価、counterfactualなcertificate生成、symbolic checker、simulation、compile、SOURCE実行・変更は行っていない。保存表示値と既存式の粗い大小関係による紙上の推論を、新しく実行した数値実験や追加certificateと呼ばない。

①PRリソース評価、H8 Phase C、②PR内部改善、③M1-S、高次PF側の科学実行や状態変更は本レビューの対象外である。今回の結論をそれらへ転用しない。

## 2. 対象と固定結果

### 2.1 問題設定

対象は名目上H4／1.00 Å／STO-3G／保存DF rank12のsnapshotである。保証する数学的対象は、保存binary64係数を正確な数としたfull DF Hamiltonianであって、元の電子積分からの厳密なモデル変換を認証したものではない。[E1,E5]

- total N=4、保存順8 modes、全12 DF blocks。
- 一体項→block0〜11の固定positive palindromic S2。
- U用trialは保存mode 0,1,2,3のoccupation determinant。HF・groundという認定はない。
- d=1/1000 Ha、α_spec=1/10。
- sqrt_bits=96、root_width=2^-40 Ha^-1、max_bisections=256。
- PSD capは`ENTRYWISE_RE_IM_ABS_SUM_V1`。
- 条件付きexact ground access。ground stateを今回生成・準備・測定したわけではない。

α_specは保証式に与えた出力失敗率の予算であり、今回測定した成功頻度、比較差のp値、実験の信頼区間ではない。dはspectral energy-windowの幅で、読出し・合成・準備誤差込みの最終エネルギー精度ではない。

### 2.2 保存値

以下は既に公開された表示用丸め値の転記である。判定の正本はexact Ratと有理数enclosureであり、表示値を新しいcertificateにしない。[E1–E3]

| 量 | 保存表示値 | 単位 |
|---|---:|---|
| L | -13.6185006472 | Ha |
| U | -2.098545937 | Ha |
| W | 11.5199547102 | Ha |
| Λ | 71.2833149076 | Ha |
| K_SOS_upper | 551.865627899 | Ha² |
| Cbar2 | 0.444112532037 | Ha³ |

| 保存route | 十分step境界の表示enclosure / Ha^-1 | 二分回数 |
|---|---|---:|
| RF0 | [1.29926081615e-05, 1.29926089781e-05] | 32 |
| RF1 | [0.000170959270173, 0.000170959270990] | 32 |
| EW2 | [0.000170959270173, 0.000170959270990] | 32 |
| CENTERED_CBAR2 | [0.0168583787630, 0.0168583787639] | 35 |

全routeが`ENCLOSED`、比較は`COUNTER_LARGER_SUFFICIENT_STEP`。対照のstepは最良の相対routeより約99倍大きいという既出の概数で表せる。ただし99倍は量子gate、wall time、真のPF誤差の倍率ではない。

SOURCE上、保存名RF0・RF1はいずれもEW2の確率式を使い、ηにRF0またはRF1を選ぶ。保存名EW2はη=min(η0,η1)を使う。従ってRF1とEW2が同じ境界なのは二つの独立実験の再現ではなく、この固定入力でmin routeがRF1側を選んだ結果である。[S1]

### 2.3 実行状態

初回science pipeline1回、修正0、retry0、二つの独立checkerが受理。人工test134件もPASS。人工testの数は独立した分子実験数ではない。使用済みclaimは保全済みである。[E1,E4]

認証verdictは`VERIFIED_EXACT_FOR_FIXED_DF_UNDER_STATED_CHECKER_MODEL`。これは明示したSOURCEとcheckerモデルの下での受理であり、形式証明、完全独立runtime、元分子の厳密保証、immutable CIを意味しない。

## 3. この負の結果の正確な意味

今回比較したのは、同じ固定S2に対する**二種類の十分誤差保証が許すstep**である。異なるPFを実際にQPEで動かして真の誤差を比較した実験ではない。

したがって支持されるのは、

> この固定DF入力と採択条件では、実装したRF0/RF1/EW2証拠より、宣言したcentered・構造付きCbar2証拠の方が大きいstepを認証した。

という限定的だが明確な結論である。

次は支持されない。

- 相対形式の定理が誤っている。
- 実際のPF biasが対照の99倍である。
- A2のあらゆる改良や全分子で優位性が不可能である。
- QPE、native gate、状態準備込みの総費用で99倍劣る。
- 元の電子HamiltonianのE0を0.001 Ha以内で今回推定できた。

逆に「同じPFなので保証の比較は研究として無意味」という解釈も適切ではない。保証に基づいて実行stepを選ぶ方式なら、利用可能な証拠の強さは実行する反復数を変え得る。ただし、その量子費用への変換は別の共通実装・読出し条件を要する。[T1,P0]

## 4. 上界の構造と不利になった要因

### 4.1 既存の二つの確率上界

κ=K_SOS_upper、C=Cbar2、β=Λ/2とする。clip前の上界を区別して

\[
\widetilde B_R(h)=\frac{\eta(h)^2W(W+d)}{(1-\eta(h))d^2},
\qquad
\widetilde B_C(h)=\frac{h^4C^2}{(1-\beta h)^2d^2}
\]

と書く。実際の確率上界はそれぞれmin(1,上式)。相対側の適用条件はh>0、hΛ<1/4、0≤η<1、対照側はh>0、βh<1である。[T1,S1]

\[
\eta_0=\frac{32(h\Lambda)^2}{1-16(h\Lambda)^2},
\qquad
\eta_1=\kappa h^2+\rho_4(h),
\qquad
\rho_4(h)=\frac{512(h\Lambda)^4}{1-16(h\Lambda)^2}.
\]

恒等式η0=32Λ²h²+ρ4より、

\[
\eta(h)=\min(\eta_0,\eta_1)=a h^2+\rho_4(h),
\qquad a=\min(32\Lambda^2,\kappa).
\]

これはleading近似ではない。[T1]

### 4.2 Wは物理的な励起gapではない

W=U−Lは選んだPSD分割の下界とtrial変分上界の差である。e=E0−Lの認証上界だが、E1−E0ではない。真のE0は今回求めていないので、Wの粗さを「下界の不備何%」「trialの不備何%」と定量分解することはできない。[E1,E5]

PSD化に用いた係数三角capは安全だが、相殺や占有制約を十分に活用したtightなsector boundとは限らない。大きいW・Λを入力とする現在の相対式は保守的になる。この方向性は式から説明できるが、tight capへ置き換えた後にどこまで改善するかは未計算である。

### 4.3 相対側と対照側で残している構造が異なる

K_SOSはraw BCH係数の絶対値、fragment cap、平方根上界と行和を使う。実際のgrouped交換子のPauli係数相殺をこの段階に取り込んでいない。[S2]

対照側はtailを先に集約し、各grouped double commutator全体を正確に展開したPauli L1と、centered cap上界の小さい方を採用する。[S3] この構造は同じ入力から取得しており、不当に対照だけに未知のground情報を与えたものではない。

この違いは不公平さではなく、**相対側の安い粗い証拠と、対照側の高価だが構造を残す証拠の違い**である。個々のgroupでどの候補がどれだけ勝ったかという追加集計は今回していない。Cが小さい理由の寄与率を推測して報告しない。

## 5. 今回の紙上診断：d・αだけの変更では救済されない

**この節は、保存された認証値と既存の上界式から今回整理した紙上の比較である。Codexが既に認証した追加定理・追加数値実験・新しいRF保証とは扱わない。**

### 5.1 一般の比較関係

Λ,W,C,a>0、d>0、0<α<1、h>0、hΛ<1/4、η<1を仮定する。両clip前上界の比は正確に

\[
\frac{\widetilde B_R(h)}{\widetilde B_C(h)}
=
\left(\frac{\eta(h)}{h^2}\right)^2
\frac{W(W+d)}{C^2}
\frac{(1-\beta h)^2}{1-\eta(h)}.
\tag{D1}
\]

η/h²≥a、W(W+d)≥W²、βh=hΛ/2<1/8、1/(1−η)≥1より、

\[
\frac{\widetilde B_R(h)}{\widetilde B_C(h)}
\ge
\left(\frac{7aW}{8C}\right)^2.
\tag{D2}
\]

厳密性の不要な端点議論を避け、十分条件を

\[
\boxed{7aW>8C}
\tag{D3}
\]

とする。この条件下では、相対側がB_R≤α<1で受理する任意のhにおいて、対照はB_C<B_R≤αで受理する。両上界が1でclipされる領域では厳密な大小を主張せず、受理集合の包含を結論する。相対域は対照域に含まれているため、domainの欠落もない。

非退化な最大十分stepは相対側の適用域の内側で確率条件を飽和する。そこで対照には厳密な余裕があり、連続性とより広い対照domainによりh_C>h_Rとなる。これは適用した十分上界の順序であって、真の失敗確率の順序ではない。

### 5.2 今回の保存値への限定適用

保存表示値は、余裕を持った粗い関係としてΛ>70 Ha、κ>500 Ha²、W>11 Ha、0<C<1 Ha³を満たす。従ってa=min(32Λ²,κ)>500 Ha²となり、(D3)は十分な余裕で満たされる。[E1–E3]

新しいXiの数値、別d・αでの根、別精度gridを評価する必要はない。この粗い大小関係と(D1)–(D3)から、

> **今回の同じ認証値・同じ保証式を固定したままなら、dとα_specだけを変える探索には、この対照との十分step順位を逆転させる仕組みがない。**

と分かる。

制限も明確である。trialを変えてWが変わる場合、cap・分割・順序・κ・Cが変わる場合、別の保証式や費用目的の場合にはこの固定値の結論を転用できない。a=0、C=0、W=0等の退化caseも本比較の対象外である。全A2の普遍的NO-GOではない。

(D1)–(D3)は今回の研究継続判断を補強する。ただし、元の認証済み負の結果そのものはこの追加整理に依存せず、仮に将来この紙上説明を修正しても、既存のh_C>h_Rの保存判定を勝手に変更しない。

## 6. もう一つの制約：適用域による上限

現相対保証は、確率条件とは別に

\[
h<\frac{1}{4\Lambda}
\tag{D4}
\]

を必要とする。[S1]

今回の保存された対照側受理点h_C,loとΛについては、保存表示値からも余裕を持って

\[
h_{C,\mathrm{lo}}>\frac{1}{4\Lambda}
\tag{D5}
\]

が読める。例えばΛ>70 Ha、h_C,lo>10^-2 Ha^-1という粗い関係だけで十分である。これは保存値と既存適用域の紙上照合であり、新しいrootやenclosureを生成したものではない。

従って現在のd・αと比較先を固定すると、

- U/trialだけを改善してWを小さくする。
- κだけを小さくする。
- root_widthやmax_bisectionsだけを増強する。
- 同じ適用域のまま上界の係数定数だけを調整する。

といった変更では、非自明な現RF方式のhを(D4)の外へ出せず、今回の対照stepには届かない。W=0になりE0が既知となる場合は、比較による量子優位とは別の自明終了である。

**ただし、適用域の端が現在のh_Rを直接止めた、という説明は誤りである。** 現在のh_Rはその端より小さく、Wとηを含む確率十分条件が先に制限している。(D4)–(D5)は、単一要因だけの改良で到達できる範囲の上限に関する診断である。

Λを小さくする認証、構造付き有限剰余、より広い正当なlog/energy-window保証は、この制約を変え得る。しかし、それらは新しい証拠・比較の設計を要する。既存式の分母やdomain判定を外すだけでは正当化されない。

## 7. 算術精度の問題と保証の鋭さの問題

今回のrouteはすべてENCLOSEDであり、反復上限256に対して32または35回で終了した。[E1] 原因を「反復数不足」「rootが未解像」「保存上限で止まった」とする根拠はない。

sqrt_bits=96はdyadic sqrt witnessの上側丸め解像度であり、全演算が96-bit floating pointという意味ではない。主要演算はexact rationalで、同じ粗いcapからの正確な計算と、tightな物理上界を得ることは別問題である。[E5,S2]

今回の結果だけでsqrt丸めの寄与を厳密に分解したとは言えない。しかし(D4)–(D5)により、Λとdomainを変えずsqrt由来のκを改善するだけでは現対照のstepへ届かない。従って算術精度増加を次の救済策として試す理由は弱い。

## 8. 費用の評価：対照取得費をA2へ丸ごと課金しない

### 8.1 保存された段別計測

以下は[E1,E3]の記録の引用であり、standalone pipelineを別実行した結果ではない。

| 処理 | 生成wall秒 | checker wall秒 |
|---|---:|---:|
| Uの証拠 | 0.002511394 | 1.704099742 |
| K_SOSの証拠 | 0.029035078 | 0.021150763 |
| 構造付きCbar2 | 218.457223173 | 116.447777066 |
| step処理（各routeを含む実装段階） | 0.534835510 | 0.006132496 |

全runは約343秒。記録されたprocess peak RSSは721304 KiB。保存ledger合計は67523176 bytes。RSSは累積peakであり、各stage独立の増分メモリではない。最終log/result I/Oの計測外部分があり、全表の親ラベルを子の小計として二重計上しない。

### 8.2 許される解釈

現在の実装では、強い構造付き対照を取得・検証する工程が高価で、K_SOS計算自体は軽い。よって「相対証拠を得るのに343秒かかり、stepも不利だから全費用で一様に劣る」とは言えない。

今回得た関係は、概念的には

> 安いが弱い相対証拠と、高価だが強い構造付き交換子証拠

のtrade-offである。ただし相対側だけ／対照側だけを独立実行した費用は未測定で、共通入力・PSD/U・serialization・cache・検証を除いた単独総費用を保存表から作ったことにはしない。

量子資源を同じ回路単価・同じ読出し条件へ接続できる場合、base-S2適用数はhの増加で減り得る。しかしquery、T/CX/depth、古典秒を換算規則なしで足せない。state preparation、合成・読出し・償却回数も未評価である。総費用でのPareto支配・break-evenは未確立のままとする。

### 8.3 この留保を研究目的の事後変更に使わない

「古典証拠が安い」ことを主目的に変えるなら、それは別の比較契約である。同じ低い取得予算で使えるcheapな既知交換子上界も対照へ認め、比較する資源座標・反復・償却を先に定めなければならない。

現在の全結果には高価なCbar2証拠が既に含まれている。それを比較から取り除いたり、その値を一方だけ無料で再利用したりして相対側の勝利へ書き換えない。今回その別目標の研究を開始・採択することは推奨しない。

## 9. 先行研究・新規性・他系列との関係

### 9.1 今回再確認した一次資料

ChildsらのarXiv v3 §5.1 Proposition 16、Eqs.(150)–(152)は、二重交換子を使う二次PFの有限実時間上界を与える。該当PDF本文と印刷p.39の画像を確認した。[P1] この上界とcentered inverse-log、spectral momentを接続する既存project比較[T1]を基準とする。

Mizuta–Kuwaharaの固定v1 Theorem S1は、PSD局所項等の条件の下で任意の射影を扱う。rank-oneへの特殊化が先導交換子の全vector評価に接続する点は前回から変わらない。同論文はPSD shift後のエネルギーを用いる制約も明示する。[P2]

Hejazi–Zini–Arrazolaの固定v1は低エネルギー射影、漏洩、局所性・fermionic条件を区別し、PSD化のconstant shiftが低エネルギーの利点を失わせ得る点を述べる。[P3]

これらは今回の結果を一般化する根拠ではない。ここでは比較相手が既知の正当な保証であることと、shift・低エネルギー性の限界が既に研究対象であることを確認した。最新全文献の網羅調査、出版版との全数式同一性、RF全statementのpriority判定は行っていない。

### 9.2 実用性の負の結果と定理の新規性は別

今回のH4結果はRFの独立新規性を証明せず、否定もしない。保証が弱いことから定理の誤りや完全な既知還元を導くことはできない。

一方、低エネルギー・gap-free・全vector・BCHという特徴や、既知のexact算術/checkerを接続したことだけで、新しい実用アルゴリズムの成立を認定することもできない。独立priorityが未確定であるという理由だけで調査・実装を無限に継続しない。

### 9.3 一つの対照に負けた時点で言えること

利用可能な保証kの受理集合をA_kとすると、既知保証全体のenvelopeはその和集合である。今回、利用可能な一つの対照が現RFより大きいstepを許したので、この固定条件で「現RFが利用可能な最良証拠より大きいstepを与える」という主張は成立しない。

この負の判定のために、全ての既知保証を追加実装する必要はない。他の保証を加えれば既知側の受理集合は狭まらないからである。ただし、既知側の全体が最適であるという主張や、古典取得費込みの最良性はしていない。

### 9.4 高次PF・PR・M1-Sとの非混同

前回レビュー[P0]が確認した高次PF側のtruth-free幅・情報取得費・資源判断という課題との重なりは保持する。今回そのrepository全体を新たに監査していない。

今回の保存DF・正stage S2・条件付きgroundという結果を、負stageの高次PF、別のPF選択、PRの乱択則、M1-Sの平均複素信号、H8資源評価へ転用しない。今回の研究方針推奨は現A2実用化経路に限る。

## 10. 前回判断からの変更

前回[P0]は、具体的なRF/EW2命題と強い対照があり、実入力の有用性が未確認であったため、限定継続を推奨した。その後、技術実装と認証付きpilotが完了した。これは当時の「証拠を取得できるか」「step優位が残るか」という問いへ情報を与えている。

現在は、単に未計算だから続ける段階ではない。現証拠法の十分stepは明確に不利であり、d/αだけの変更では救済されない条件と、単一要因の改良を制限する適用域も分かった。従って、**現行実用化経路の継続推奨を取り下げ、ここで区切る推奨へ変更する。**

この変更は、RF0/RF1/EW2の成立を撤回するものでも、今回の実装努力が無価値だったとするものでもない。実入力での認証生成・独立検査、強い基準との比較、費用構造と限界の把握が成果として残る。

## 11. 継続・方向転換・終了の選択

| 選択 | 今回の推奨 | 理由 |
|---|---|---|
| 同じ証拠法で別分子・geometry・d/αを追加 | 推奨しない | 差を改善する新機構がなく、固定値でd/αのみの探索は§5の制約を受ける |
| root/sqrt精度や反復数だけを増やす | 推奨しない | 未解像ではなく保証の保守性。適用域の壁も残る |
| trial/Uだけを改善 | 現比較を救済する既定作業にしない | Wには効き得るが、同Λのまま現対照stepに届かない。W=0の自明caseは別 |
| tight cap・構造付き有限剰余へ改良 | 数学的可能性は残すが、自動開始しない | 全証拠・比較先にも改善情報を渡す必要。具体構成・有利な条件がまだない |
| cheap certificateを主目的に変更 | 今回は採択しない | 正当な別研究だが、費用目的とcheap対照を先に固定すべきで、負の結果の事後的成功化にはしない |
| 純理論としてpriority調査を継続 | 自動の次作業にしない | 数値的有用性とは独立。明示した理論目的と未解決点が必要 |
| 現行の実用化経路を区切り、成果を保存 | **推奨** | 得られた証拠に見合う結論。追加計算や新SOURCEなしで研究上の着地点が明確 |

将来再開する根拠となり得るのは、同じ合法な情報条件でΛ・相対係数・有限剰余の問題を変える具体的な構成、あるいは有用な入力classと証拠費用を説明できる結果である。「精度を変える」「例を増やす」「まだ新規性を否定されていない」だけでは十分な再開理由にならない。

それらの可能性を調べる科学設計・新計算を今回追加発注しない。任意の小例が常に無意味とも断定しないが、目的を持たない例探しは既定の次作業にしない。

## 12. 残す成果と、次の実務

残す成果は次の三層である。

1. RF0/RF1/EW/EW2と固定powersへの条件付き接続に関する理論・監査。独立priorityは未確定のまま。
2. exact intake、PSD/scalar、U/W、K_SOS、grouped Cbar2、step enclosure、独立checker、one-shot保全の実装・人工検査。
3. 固定H4 DFで初回認証比較が完了し、宣言した強い対照が優位だった負の結果と、実測証拠取得費用。

論文化・原稿作成は今回要求されていない。将来Codexが分かる研究記録として、現入力・現方式・限界・再開条件を一つの終了ハンドオフへリンクするので十分である。repository全体を再整理しない。

次の担当は**ユーザー**で、現行実用化経路を区切る推奨を採択するか判断する。採択後のCodex作業は、固定結果と本レビューを参照した最小限の方針記録・索引更新・保全であり、追加scienceは含まない。実行済みclaimやSOURCE・契約・結果を編集しない。

この結論に対する同じ詳細レビューを直ちに繰り返す必要はない。新しい証拠・構成・重要な見落としが示された場合だけ、その差分を科学判断する。

## 13. 主張台帳

| 主張 | 判定 |
|---|---|
| 固定DF入力で認証pipelineが初回完了した | 保存結果により支持 |
| 同じ条件でcentered構造対照の十分stepが大きい | 保存された独立checkerのenclosure判定により支持 |
| 境界未解像や計算中断が今回の負の結果の原因 | 不支持 |
| 現認証値を固定したd/α変更で順位逆転を狙える | 既存式の紙上比較§5では不支持 |
| 同Λ・同適用域のままtrial/Uだけで現対照stepへ到達できる | 非自明な現routeについて§6により不支持 |
| tight capや新しい構造付き保証でも不可能 | 未証明・主張しない |
| A2全体、低エネルギー保証一般が無価値 | 不支持 |
| RF定理の独立新規性が今回確立または否定された | どちらも未確定 |
| 約343秒はA2固有の証拠取得費である | 不適切な費用帰属 |
| 相対証拠は安いので総資源で優位 | 未確立 |
| 対照がnative・state準備込みで99倍有利 | 未確立 |
| 現行実用化経路へ追加投資せず成果保存する | 今回の科学的推奨 |

## 14. 状態・承認・不確かさ

今回の結論の確信度は、保存結果の限定解釈と現在の比較式の制約について高い。将来のtightな保証や別classの可能性については未確定であり、成功・失敗を予測しない。現行経路をここで区切るのは研究資源配分の判断で、全分野の不可能性の証明ではない。

A2は`NOT_ADOPTED`、A1は`CLOSED`、K1-Fは未承認・未実行、SPEC-08未承認のままとする。今回の推奨を正式な`CLOSED`記録へ自動反映していない。新規science実行・SOURCE変更・GitHub書込み・authorization/grant/marker発行は0件。消費済みone-shot claimは保全対象である。

**レビュー完了。mandatory STOP。研究方針の採択と、将来の新規科学計算の実行承認は別である。**

## 15. 固定資料と参考文献

### 実行・結果

- [E1] [Phase B結果・対象・費用・STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c1b80c0fabc1706d69585d993cbecfa12e4d09a6/docs/research/qc_a2_phase_b_execution_20261018/results_and_stop.md)。結果payload commitは817d27b5…、最終receipt commitはc1b80c0f…。本文と段別時間を確認。
- [E2] [保存結果summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c1b80c0fabc1706d69585d993cbecfa12e4d09a6/artifacts/qc_a2_phase_b_execution/2026-10-18/saved_results_summary.json)。メタデータ・値・enclosure等の該当部分を確認。全probeの再検算なし。
- [E3] [raw result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c1b80c0fabc1706d69585d993cbecfa12e4d09a6/artifacts/qc_a2_phase_b_execution/2026-10-18/attempt_1/ledger/result.json)。直接取得したが長文表示上限あり。科学値・費用・context・記録一覧を確認、全JSONの独立再認証なし。報告raw SHA256は`1c66113d8e18626a7e682879682fdde221ff1495e16e8d4271ffaa5d0b69f281`。
- [E4] [実行承認記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c1b80c0fabc1706d69585d993cbecfa12e4d09a6/docs/research/qc_a2_phase_b_execution_20261018/execution_authorization_record.md)。実行許可と、技術中断時の限定対応、旧claim保護の記録。今回実際には修正・retryなし。
- [E5] [正式契約Cの意味論](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a7f8307123f6ba5bbb8d8b84e124400ac04a60ce/docs/research/qc_a2_phase_b_contract_adoption_20261018/formal_contract.md)。採択時点で実行未承認という履歴を、後続実行と混同しない。
- [E6] [実行branch](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/qc-a2-phase-b-execution-20261018)。本レビュー時のGitHub APIでHEAD=c1b80c0fabc1706d69585d993cbecfa12e4d09a6を確認。

### 科学仕様・SOURCE

- [T1] [EW2・有限step・先行研究比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/db1819587df36ee8833b37227ca0083faa0408ec/docs/research/qc_relative_form_ew2_step_audit_20261017/finite_step_and_prior_art_comparison.md)。主に§§1–4を確認。相対式、centered対照、受理集合、Xiの漸近域を再利用。
- [S1] [producer_steps.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8d483f9bfa5983bb52fef55887e2f9f73a790511/src/trottercert/a2_spec04_07/producer_steps.py)。predicate、bracket、route labels、退化case、comparisonを静的読取。実行なし。
- [S2] [producer_algebra.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8d483f9bfa5983bb52fef55887e2f9f73a790511/src/trottercert/a2_pilot/producer_algebra.py)。U・K_SOS部分を確認。全SOURCEの形式的正当性を認証したものではない。
- [S3] 同producer_algebra.pyのgrouped commutator、構造L1／center capのminと1/24・1/12和。今回各groupの実値再集計なし。
- [P0] [前回重要科学レビュー保存copy](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/db1819587df36ee8833b37227ca0083faa0408ec/docs/research/qc_relative_form_ew2_step_audit_20261017/inputs/qc_relative_form_post_audit_novelty_value_review_20261017.md)。ローカル提供Markdown全文を読取、GitHubの冒頭・結論を確認。過去結論を変更する理由は§10。

### 外部一次資料（今回の限定確認）

- [P1] A. M. Childs, Y. Su, M. C. Tran, N. Wiebe, S. Zhu, [A Theory of Trotter Error, arXiv:1912.08854v3](https://arxiv.org/pdf/1912.08854v3)。§5.1 Eqs.(150)–(152)、Proposition 16。versioned PDFテキスト取得、最初の画像取得失敗後に同arXiv非version URLの82頁PDF・印刷p.39を画像確認。出版版全差分の監査なし。
- [P2] K. Mizuta, T. Kuwahara, [Trotterization is substantially efficient for low-energy states, arXiv:2504.20746v1](https://arxiv.org/html/2504.20746v1)。setup、Supplement Theorem S1、PSD shiftに関するdiscussionをHTMLで確認。全証明の再構成・出版版全照合なし。
- [P3] K. Hejazi, M. S. Zini, J. M. Arrazola, [Better bounds for low-energy product formulas, arXiv:2402.10362v1](https://arxiv.org/html/2402.10362v1)。低エネルギー射影・fermionic条件、PSD shiftに関する補足discussionを確認。表・図の数値を今回の根拠に使用していない。

外部文献から今回新しい分子向け定数・有利な入力を輸入していない。一般的な低エネルギー保証の存在を、現在のRF方式の成功へ読み替えない。
