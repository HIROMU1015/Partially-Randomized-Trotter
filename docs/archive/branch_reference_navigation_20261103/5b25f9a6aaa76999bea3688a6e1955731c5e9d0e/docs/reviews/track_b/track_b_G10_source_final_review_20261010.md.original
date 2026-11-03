# Track B G10：固定sourceの実行前最終レビュー

- 作成日：2026-10-10 JST
- レビュー開始承認：利用者の「レビューを開始して」
- 対象repository：`HIROMU1015/Partially-Randomized-Trotter`
- 対象branch：`track-b-g10-degree-comparison-preparation-20261010`
- 対象source S：`05c5ef23fce775a822ab5686f5da2f0d77675864`
- machine contract：`artifacts/track_b_g10_degree_preparation/2026-10-10/contract_v1.json`
- contract SHA256（source manifest記載）：`71ae2310a08f9d111faab76f27e6de277189ab132370d8996a0e2420c9510818`
- レビュー区分：G9 v2レビュー§13に沿って準備されたG10の数学・実装・比較・実行境界の最終評価。G10の登録結果のレビューではない。
- 判定：**指定sourceと固定scopeについて、別途明示的なone-shot実行承認へ進めてよい。実行前の必須source修正は、確認した範囲では見つからなかった。**
- 実行許可の状態：本レビュー時点のauthorizationはpending。レビュー完了は、authorization作成・更新、runner呼出し、marker消費を意味しない。

## 1. 結論と、その意味

G10は、全return集約familyの一般生成器が、安価な低次数特殊化で代替できない追加価値を持つかを、同じ入力構造の中で調べるための比較である。今回のsourceは、その目的に沿っている。

特に、m=7に「closed P5 + degrees 6/7 ordinary pair」が含まれ、m=5は保存native資料の共通policy再会計だけになっている。次数・p・x・providerを同時に変える比較ではなく、同じp/x/providerで各有限P_mの内部を比較する構成である。[R1–R4]

最終判定は次のとおり。

| 項目 | 判定 |
|---|---|
| G9 v2レビュー§13とのscope整合 | 整合 |
| 同一次数で同じfinite first operator momentを比較 | source上整合。登録P3/P7の実測結果は未取得 |
| closed P5+6/7 tailの構成 | 係数保存・proposal置換・相対位相を確認 |
| confidence・bias・m5再利用 | 固定policyとして整合 |
| native実装・合成API境界 | inherited実装と新runnerの接続を確認。旧Fraction境界問題への対処あり |
| 任意proposal下界 | 指定辞書・価格・係数・policyに限る数学的下界として妥当 |
| one-shot・source/authorization分離 | 設計とsourceを確認。局所synthetic検算でも拒否条件を確認 |
| 登録実行を妨げる必須修正 | 確認した範囲ではなし |
| G10の勝敗、新規性、論文十分性 | 本レビューでは未判定 |

これは、完走・全keyの合成成功・七次での改善を保証する判定ではない。また、任意の改変sourceに対する承認でもない。判定対象は上記Sと固定contractである。

## 2. 資料の取得と確認範囲

### 2.1 正本と固定ref

レビュー中にGitHub接続でbranchのremote HEADがSと一致することを確認した。[R24]

G10の数学契約、machine contract、新規5 module、future runner、focused test source、launch/preparation verifier、保存policy監査、先行研究claim表、runtime/preparationの記録を取得して確認した。継承部分ではP5Closed、Rz合成・保存値検査API、共通BudgetGuardも確認した。[R1–R18]

前回G9 v2レビューのローカル添付は32,825 bytesで、SHA256は次のとおりであった。

`fb02c75e3976a7ef8f9ba3753173272bbd6a82108296e04135cc005c7e2fae80`

これはG10 contract/source manifestが指す採用レビューのhashと一致する。今回のscope判断では特に§13-A–Dを使用した。[R2,R17,R23]

### 2.2 今回実施したこと

1. 上記資料の読解とsource間のcall/dataflow照合。
2. prefix+tailの確率補正、Bernstein式、共通準備費用下界の数学的検討。
3. 本文に示す、登録入力を使わない有理算術・組合せ上限の自己検算。
4. 取得した`g10_launch.py`のGit blob SHAを照合したローカルコピーから、二つのlaunch関数だけをAST抽出し、fake Git・一時的なsynthetic filesystemで動作確認。
5. 詳細レビューと検算資料の作成。

### 2.3 今回実施していないこと

登録P3/P7の係数・angle inventory・予算・性能評価は実行していない。G10 runner、実backend、native synthesis、providerの行列評価、量子測定、LP、DF・分子計算も実行していない。

repositoryの41件のfocused testsをこの環境で再実行してはいない。113 critical pathsと1,241 protected pathsの全再hashも行っていない。これらについては、保存された準備照合の範囲・実装と記録を確認したのであり、今回のGPTが同じ全検査を再現したとは表現しない。[R14–R17]

G9の11.3 MBの全raw resultを、この回答で再取得して全eventを再認証する作業も行っていない。保存監査の内容と、その一般式・実装・source bindingを確認することに絞った。

コードや資料はGitHubの固定refを正本とした。実行環境側のworktreeが現在cleanであることは、この環境から直接観察した事実ではない。実行時にlaunch gateが再確認する。

## 3. 研究目的と比較設計の評価

### 3.1 今回の問い

G9でclosed P5の有効点が得られていても、一般Green-generatorがより高い次数で必要になるとは限らない。低次数の効果を残して高次数だけordinaryにする対照を置くことが、G10の中心である。

固定条件は次のとおり。[R1,R2]

- `p=(1/5,3/10,1/2)`、`x=5/7`。
- G9と同じ3-system-qubit synthetic provider。
- `Q0=Z0`、`V1=R_XX01(pi/4)`、`V2=R_XX12(pi/4) R_ZZ01(pi/4)`、`Qi=Vi† Zi Vi`。
- `R_P(theta)=exp(-i theta P/2)`、演算子積は右から作用。
- exact Clifford+T provider、同direct lowering、同strict Rz epsilon。
- 各mでtargetは`P_m(-ixR)`のfull first operator moment。

| 次数 | 登録direct方式 | 処理 |
|---|---|---|
| 3 | ordinary、partial P3、closed P3、general full、literal CTS | 5新規row |
| 5 | ordinary、partial+tail、closed P3+tail、general full、closed P5 full、literal CTS | 6保存rowの再予算化 |
| 7 | ordinary、partial P3+tail、closed P3+tail、general full、closed P5+tail、literal CTS | 6新規row |

17 rows／34 axesである。G9のhelper5診断rowを新primaryへ混入させていない。

### 3.2 答えられること

同じp/x/providerにおける、各有限次数内の構成法の資源交換を比較できる。特にP7では、P5までの閉形式を使い切った後に、全returnを追加で扱う価値があるかを調べられる。

### 3.3 答えられないこと

異なるmではtarget polynomial自体が異なる。従ってm3とm7のT値を、そのまま同じexponential精度を達成する費用として比較できない。Taylor remainder、step分割、deterministic側、QPE全体の誤差・費用は未接続である。

また、同じp/x/providerは既知development構造である。七次が今回新しく評価されるとしても、一般分子、全provider、独立外部再現への結論ではない。

**これらの限界は実行設計を無効にしない。G10が答える問いを、有限P_mの構成比較に限定するための境界である。**

## 4. ClosedP5Tailの数学と実装

### 4.1 finite targetを変えていない

`ClosedP5Tail`は`P5Closed`をprefixとして保持し、ordinary generatorの`degree>=6`群だけを追加する。m=7では追加群はdegree6の一群であり、6次と7次をpairする。[R3]

\[
P_7(-ixR)=P_5(-ixR)+\frac{(-ixR)^6}{6!}+\frac{(-ixR)^7}{7!}.
\]

`(-i)^6=-1`、`(-i)^7=i`なので、tailには6次parentの負の相対位相が必要である。既存ordinary eventからdegree6位相を受け取り、prefixの位相を再定義しない設計を確認した。

新たなHamiltonian、別のexp近似、別meanへ置き換える手順は含まれていない。

### 4.2 proposalだけを正しく置き換える

prefix内の旧group確率を`q_old_group`、統合後のgroup確率を`q_new_group`とすると、実装は

\[
q_e^{new}=q_e^{old}\frac{q_{new\ group}}{q_{old\ group}},\qquad
W_e^{new}=\frac{\widetilde\alpha_e}{q_e^{new}}
\]

としている。

従って

\[
q_e^{new}W_e^{new}=\widetilde\alpha_e
\]

であり、prefixの有限rational係数を再fitせずに保存できる。旧global group確率を二重に掛ける実装ではない。[R3]

P5群とtail群を一つのdyadic group lawへ統合するため、prefixを選ぶ確率が変わっても、それに応じて補正weightを変更する必要がある。本sourceはこの操作を実施している。

### 4.3 sampleとreferenceの役割

productionでは、群を引いた後に必要なword・childを条件付き生成する。`ClosedP5Tail.sample`はprefixの`reference_events`を呼ばない。

reference側は、prefix内のroot/two/four群とtailの生wordを列挙して、operator meanとnative期待費用を確認する。この列挙はproductionの入力として使われない。[R3,R4,R5]

この構造は非列挙生成の比較として整合する。ただし、G10のnative key集合は参照層から事前に作る設計であり、G8で試したon-demand native取得を長期運用で検証する実験ではない。

## 5. general fullとliteral CTSの確認

### 5.1 general full

既存の`FullReturnGenerator`を指定mへ接続し、`B_new`の参照和で予算を縮めず、ordinary envelopeと`U`を使用する。runnerは非CTS generatorの予算をevent traversalより先に作る。[R5,R6]

raw nonreduced trialはzeroとなり、reduced先の別parentへ付け替えないという旧意味論を保持している。今回、そのproduction結果を別のproposalへ置き換える探索も追加していない。

### 5.2 CTS

`g10_reference.exact_target`は、固定providerのQ(sqrt(2)) Pauli記述から、指定mの有限polynomialを積み上げる。`cts_events`はidentity real correctionを残し、odd成分を一つのcommon-angle群へpairする。[R4]

係数intervalの符号、rational midpoint、rotation normalizerの誤差を別に扱う。ゼロTのreal eventsを除去したり、正の仮想Tへ置き換えたりしていない。

m5との一致、P3/P7のfirst operator mean、Q(sqrt(2))と別matrix表現の整合性に関するoff-domain testsがある。[R9]

これをliteral CTS全般や任意LCUの最適性比較へ拡張しない。Pauli grouping、identity吸収、precision配分、別confidence方式は、現在の固定辞書の外である。[R20]

## 6. confidence・finite-bit・誤差会計

### 6.1 共通失敗確率

machine contractは、34 axesに`alpha_axis=49/34000`、17 resource rowsに`beta_row=1/17000`を割り当てる。[R2]

\[
34\frac{49}{34000}=0.049,\qquad
17\frac1{17000}=0.001,\qquad
0.049+0.001=0.05.
\]

自己検算でもこの和をexact Fractionで確認した。

resource tailは`t=10`。exp(10)>17000を有理Taylor部分和で確認できるため、exp(-10)<1/17000となる。canonical rowでresource予算が余っても事後再配分しない。

### 6.2 biasと残余精度

共通に`rho=eta=10^-12`、`epsilon_Rz=10^-6`、`epsilon_axis=1/200`を保持する。

\[
b=2\{8\rho+8(1+\rho)2\epsilon_{Rz}\},\qquad
s=\epsilon_{axis}-b>0.
\]

exact Clifford+T providerの下で、各eventの近似部分は最大二つのRz primitiveであり、word・basis・phaseはexact modelとして扱う。長いwordであることを理由に、Rz近似がm回繰り返される会計にはなっていない。

このprovider条件は、現実のhardware雑音や一般分子providerでdelta=0を達成したという主張ではない。

### 6.3 momentとrange

\[
\kappa=\frac{(1+\rho)^2}{(1-\eta)^{m+2}}.
\]

canonical側は`m2<=kappa B^2`、local full側は`m2<=kappa B U`、rangeは`(1+rho)B/(1-eta)^(m+2)`を使用する。[R6]

prefix+tailで旧group確率を新group確率へ置き換える操作は、proposal補正により係数を保存する。旧と新のgroup丸めを独立に二重加算する必要はない。最大m-1個のraw label、group、acceptance、childという因子数に対しm+2は安全側である。

\[
N=\left\lceil\ell_+\left(\frac{2m_{2,+}}{s^2}+\frac{4W_+}{3s}\right)\right\rceil.
\]

sourceはlogとrootの上端を十分予算に使い、saved policy下界では下端を使う。必要な向きを取り違えていない。

### 6.4 matrix参照は診断

float matrixの`1e-10`は位相・演算子順序・meanの診断であり、confidence上界の入力ではない。実際に小さかったsignal、matrix残差、取得errorを使ってNを事後縮小しない。[R1,R5,R6]

### 6.5 accepted・hard・T上限

zero-fillでは全試行数M=2Nとaccepted回数を分ける。accepted tail capはC、絶対上限はMである。各rowで最大event Tを掛けてT上限を作る場合も、それは期待Tとは別である。

このconfidence/resource設計は、backendの全角度での成功確率や技術的完了確率を保証するものではない。technical abortを統計失敗0.05へ黙って混ぜない。

## 7. m=5保存anchorの再予算化

`rebudget_anchor`はG9 rowをdeepcopyし、保存m2/range/acceptance/費用を用いて、新しい共通alphaとresource配分のNを再計算する。[R6]

保存per-trial費用Cに対する2-axisの期待費用は`2N C`、準備係数は`2N q`である。operator・native IR・合成列そのものは変更しない。

旧`original_G9_budget`を別に保存するので、新しい34-axis会計と旧22-axis登録結果を区別できる。旧G9 status・marker・resultの上書きは含まれない。

`finish_plan`のnorm確認用引数には保存range上端が使われる。これはBそのものを再同定する操作ではなく、正で8未満というsanity checkである。Nを決めるmoment/rangeは、別に渡された保存値が保持される。この点を新しいnormalizer取得と表現しない。

m5に関するnew matrix/guard/synthesisを呼ばない経路がsourceにあり、その境界をmockで確認するfocused testもある。[R6,R9]

## 8. 任意proposalと共通準備費用の下界

### 8.1 sourceにある一般式

固定digital係数`a_e>0`、固定native T価格`T_e>=0`、共通準備価格h>=0を考える。Cauchy–Schwarzから、同じ十分shot policyの2-axis費用は

\[
G(q;h)\ge\frac{4\ell}{s^2}\left(\sum_e a_e\sqrt{T_e+h}\right)^2.
\]

proposalがzero-fillの非zero event上でsubprobabilityになっても、この積の不等式自体は成立する。

さらに

\[
\sqrt{(T_i+h)(T_j+h)}\ge\sqrt{T_iT_j}+h
\]

である。両辺を平方して差を取ると

\[
h(T_i+T_j-2\sqrt{T_iT_j})\ge0
\]

なので、h>=0、T_i,T_j>=0で正しい。従って

\[
G(q;h)\ge A_{lo}+K_{lo}h,
\]

\[
A_{lo}=\frac{4\ell_{lo}}{s^2}\left(\sum_ea_e\sqrt{T_e}_{lo}\right)^2,
\quad
K_{lo}=\frac{4\ell_{lo}}{s^2}\left(\sum_ea_e\right)^2.
\]

`g10_saved.affine_policy_lower`はこの式を使っている。T=0のeventをそのまま残すことも妥当である。[R7,R19]

### 8.2 既存G9への保存監査

G10準備資料は、945 direct bindingsを保存IRから再計数し、固定CTSだけでなくordinary/partial/P3にも、保存closed P5との全共通h>=0分離を確認したと報告している。[R19]

| G9固定辞書 | T切片下界の表示値 | 保存closed P5との全h>=0分離（準備報告） |
|---|---:|---|
| ordinary | 347,918,490.395 | 成立 |
| partial_return_tail | 258,117,355.764 | 成立 |
| closed_P3_tail | 266,899,436.516 | 成立 |
| full_return | 255,172,895.627 | この下界では未分離 |
| closed_P5_full | 255,172,895.627 | この下界では未分離 |
| matched_CTS | 305,097,828.828 | 成立 |

これは準備資料由来の保存結果であり、今回のGPTが945件を再計数した新しい結果ではない。切片だけで全hを判定できないため、元監査はK下界との符号も使う。

### 8.3 この比較を選ぶ妥当性

前回レビュー§13は、cost-aware有限proposalを追加する場合の公平条件を定めたが、それを必須の探索作業にはしていない。G10は有限proposal探索をせず、全固定辞書へ同じ任意proposal解析下界を適用する。これは採用済みscopeに反していない。[R23]

一つの実行可能なupperが競合のlowerを下回れば、その固定policy・辞書について分離を述べられる。反対にupper>=lowerだった場合は、未分離であり、改善不可能とは言えない。

固定precision、係数、価格、compiler、confidence方式の外の最適性は出ない。実行結果の全winner、真の物理shot下界、新規性の証明にも置き換えない。

## 9. source・authorization・実行境界

### 9.1 準備sourceをそのまま実行しない

現authorizationは`PENDING_SEPARATE_G10_AUTHORIZATION`、`science_execution_authorized=false`、source/contract/instructionはnullである。[R18]

`verify_launch`は、これが承認済みでなければGit・protected data・result directoryの処理に入る前に拒否する。runnerのscience importsとmarker消費より前にgateがある。[R5,R8]

### 9.2 必要な実行時条件

- 新しいG10固有statusと明示one-shot instruction。
- 完全な40桁source SHAとcontract hash。
- runsはint型の1、retriesはint型の0、mandatory STOPはTrue。
- 実行HEAD Aの唯一のparentがS。
- A≠S、worktree clean、許された差分のみ。
- authorization JSON自身がAへcommitされていること。
- execution branchのremote HEADがAに一致。
- critical source hash、protected history、runtimeが一致。
- result directoryに既存証拠・markerがないこと。

receiptだけを変更したchildは外側の`verify_launch`で拒否される。内部`validate_binding`だけのPASSを、完全launch承認と同一視しない。

### 9.3 旧G9の権限と結果を流用しない

source-boundのG10 authorizationが必要で、旧G9 statusでは拒否される。markerは新しい独立result directoryにexclusive作成する。sourceとauthorizationを分ける手順は整合する。

旧v1/v2の失敗・成功履歴は保持し、今回source準備を旧runのretryとして扱わない。今回のレビューも、marker作成や承認JSONの変更をしていない。

### 9.4 このゲートが証明しないこと

ゲートは指定source・契約・記録の整合を確認するもので、利用者の意思そのものを自然言語で認証する機構ではない。明示実行指示は別途利用者から受け取り、正確に保存する必要がある。

また、保存テストPASSフラグの真正性はsource・成果物の来歴と合わせて評価する。フラグだけから数学・科学的採択を決めない。

## 10. API、native取得、資源上限

### 10.1 旧Fraction→mpf問題への対処

新規synthesisの呼出しでは、runnerは`c['primitive_error']`の文字列`"1/1000000"`を渡している。保存列のidentity比較だけはFractionへ変換して使う。この二つを区別している。[R5,R11]

テストには、実際のsynthesize wrapperを通過してbackend入口をstubへ差し替え、mpfへ渡されたepsilonを確認する項目がある。import成功だけのテストではない。[R10]

mpmathの公式資料もfraction文字列の変換を認めている。ただし、公式資料の確認だけを固定package tree上の成功と同一視せず、固定runtimeのstubテスト記録を併用している。[W1,W2]

### 10.2 再利用と新規取得

旧19 primitive keyは、angle・epsilon・strict phase・sequence/count/hash/errorのidentityを照合して再利用する。旧wrapper全体の費用を新しいm3/m7へ転送しない。

新規keyは固定sourceの式から、合成を開始する前に集合を確定する。別precision、別seed、別backend、追加angleを結果後に探索しない。負角は全sequenceのactual adjointを利用する。[R1,R2,R5,R11]

### 10.3 静的な上限の再確認

全wordの係数・登録angleを評価せず、組合せ上限だけを検算した。

| m | rows | full parent上界 | event上界の合計 |
|---|---:|---:|---:|
| 3 | 5 | 7 | 233 |
| 5 | 6 | 31 | 1,073 |
| 7 | 6 | 127 | 10,013 |
| 合計 | 17 | — | **11,319** |

11,319<12,000 bindings。新規angle上界15+147=162、旧19を加えて181<cache192 entriesである。これらはoverlapを無視した安全側の見積りで、実取得key数を予測したものではない。[R2,R3]

非CTS production diagnosticは4+5=9 arms、64 trialsずつ、計576 trials。短い固定bitstreamのinterface確認であり、量子trajectoryや性能の統計的推定ではない。

### 10.4 時間・memory・出力

固定上限はwall1,200秒、CPU900秒、RSS512MiB、AS1,536MiB、per-key wall30秒/CPU20秒、新synthesis162、出力128MiBである。これは本レビューからの所要時間予告ではなく、contractにある停止上限である。[R2]

guardは周期的なwall/CPU/RSS確認とPOSIX上限を使う。sourceにresult/markerの再利用を行うループはない。[R12,R13]

**静的event/key数の上限は、メモリ・serialization・合成の完走保証ではない。** 大きいFractionの集計とJSON化のmemory/bit費用は残る。Pythonの整数文字列変換には設定による長さ制限もあり、出力byte capだけが唯一の制約ではない。[W3]

登録範囲でそれらが確実に失敗すると示す証拠はなく、今回登録P3/P7を開いて予測することもしなかった。この残余リスクだけを理由にsourceを変更したり、上限を増やしたりすることは要求しない。

## 11. 失敗時の扱いと残余リスク

sourceは、通常の実行例外に対し`G10_TECHNICAL_INCONCLUSIVE`、原因文字列、取得済み情報を保存し、prefixを科学的結論に使わない設計である。全17rowと保護条件が成立した場合だけcomplete statusになる。[R5]

ただし、次の限界は残る。

1. marker後のimport失敗、OS強制終了、極端なMemoryError、disk failureまで含めて、完全なresult/STOP JSONが必ず書かれる保証ではない。
2. exception handlerも結果のJSON化を行うため、serializationそのものの障害では、完全な失敗報告を書けない可能性がある。
3. 出力上限確認はpayloadを構成した後なので、最大memory量を128MiBに制限する仕組みではない。

これは確認したsourceの失敗回復上の限界として記録する。登録scopeで必ず発生するblocking defectを確認したものではなく、今回のsource差替えを必須とは判定しない。

発生時は消費済みmarkerを保持して停止する。result/STOPの欠落を隠したり、markerを消して再実行したりせず、残存証拠と原因を報告する。科学結果が取得できなかったことと、方法が不利であることを区別する。

## 12. 保存テスト・今回の局所検算・非claim

### 12.1 repositoryの準備証拠

保存記録では41 focused testsがPASSしている。内容には以下が含まれる。[R9,R10,R14]

- P5 prefix維持、6/7 tail、proposal正規化と係数補正。
- P7 full mean、CTS P3/P7 mean、exact Q(sqrt(2))と独立matrix。
- degree6 phase、zero-T CTS、productionへのreference混入拒否。
- m2/range・failure union・saved-only m5再会計。
- pending/旧権限/merge/dirty/source差分/remote不一致/消費済みmarkerの拒否。
- synthesis API入口の文字列epsilonをstubで確認。

41件の多くはoff-domainの二label fixture`p=(2/9,7/9),x=1/3`、synthetic cost、mock gateを使う。これは未実行の登録P3/P7の成功を事前に意味しない。

保存read-only verifierは1,378 checks、113 critical paths、1,241 protected pathsの照合を報告し、registered science/real backend=0としている。[R15]

### 12.2 GPTの局所検算

添付`review_selfcheck.py`は、49の明示assertion/checkを行った。内容は次のものに限定した。

- `g10_launch.py`の3,708 bytesをGit blob SHA `42f99b314f098d3029bacf613ac33692c44de048`と照合。
- sourceの二つの関数をAST抽出し、synthetic approval/ref/contract、fake Git、一時directoryで拒否条件を検査。
- 17rows/34axes、11,319event、162new/181total key、576interface trial、failure配分のexact算術。
- 抽象prefix/tailのproposal置換で係数と確率総和を保存すること。
- degree6のscalar位相、準備費用pair不等式のsynthetic有理値。

**49件はrepositoryの41testsの追加実行・全独立再現ではない。** 数字を合計して90件のproduction test合格と表現しない。

実際のGitコマンド、ネットワーク、repository mutation、実authorization、実markerは作成していない。synthetic temporary directoryは終了時に削除される。

### 12.3 新規性と文献

今回は実行前source reviewである。G10 claim/prior-art mapを読み、既知componentsと差候補を区別していることを確認したが、すべての一次文献を再精読してpriorityを確定する別レビューはしていない。[R20]

外部確認はAPIの数値入力とserializationの公式資料に限定した。数学・sourceにない実装成功や性能を一般知識で補完していない。

## 13. 実行後の科学的解釈を事前に固定する

### 13.1 肯定結果

m7のgeneral fullがclosed P5+tailを上回れば、この固定input/provider/policyで、P5までの集約を超える追加価値がある証拠になる。新しいm9、別p、別provider、分子の優位を自動的に認めない。

実行可能費用が競合の固定辞書下界を下回れば、その辞書内の任意proposalに対する分離を述べられる。下界が未分離なら、canonical勝敗と任意proposal最適性を分ける。

### 13.2 差がない・低次数対照が有利

一般生成器を性能の主役にする根拠を縮小する材料になる。一方、一般係数query・finite-bit・有限平均保存の数学自体を反証したことにはならない。

構成ノートへまとめる、低次数fast pathを中心にする、別の具体的な課題へ移る、という選択肢を結果後のGPTレビューで比較する。勝つ条件を求めて自動的にm9や新seedを追加しない。

### 13.3 技術的失敗

prefixの部分勝敗を採点しない。取得失敗・guard失敗・source mismatch・出力失敗は、それぞれ研究仮説の反証と区別する。旧markerを解除して再実行しない。

### 13.4 十分性と限界

このG10は「次に判断を一段進めるための限定比較」として妥当である。七次の一例に勝つことを論文の普遍的必要条件にはしないが、結果が新規性・方法の役割をどう変えるかは評価する。

新規性・投稿十分性・実分子native優位は、G10のcomplete statusから自動採択しない。多数のtest/監査や小さいerror診断を、科学的貢献の代用品にしない。

## 14. 次の担当と実行承認の扱い

**本レビューは完了した。指定Sに対する必須source修正は見つからなかった。次は利用者の明示的なG10 one-shot実行承認待ちである。**

承認後の実行担当はCodex。手順は既定contractの範囲で次のとおり。

1. Sを親とする別authorization-only child Aを作る。変更は新G10 authorization JSONと任意receiptのみ。
2. instructionを実際の利用者の明示承認へbindし、source Sとcontract hashを記録する。
3. Aをpushし、execution branchのremote SHA、clean状態、critical/protected/runtimeのgateを通す。
4. 新G10 markerにより固定bundleを一回だけ実行する。runs=1、retries=0。
5. 全結果でSTOPし、必要なresult・source・監査を固定remoteから取得可能にして返す。

この段落自体を利用者の実行指示として代用しない。現在のpending authorizationは変更していない。

科学的意味論を変えない技術作業を、各testごとにGPTへ戻す必要はない。一方、実行前にsourceまたはcontractを実質的に変更した場合、現Sに対するレビューを別SHAの保証として流用できない。差分と影響を示し、必要な範囲で再確認する。

G10結果後の重要な研究レビューは別の節目であり、資料可用性の確認と利用者の開始承認を経て行う。

## 15. 証拠一覧

以下のR参照は、特記がない限り同じ固定source SのGitHub URLである。URLは証拠locatorであり、この資料で内容をすべて再実行したことを意味しない。

- [R1] [数学・実行契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/docs/tracks/algorithm_codesign/g10_degree_comparison_contract_20261010.md)
- [R2] [machine contract](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/artifacts/track_b_g10_degree_preparation/2026-10-10/contract_v1.json)
- [R3] [ClosedP5Tailと静的上限](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/src/trottertracks/algorithm_codesign/g10_generator.py)
- [R4] [有限CTS・reference](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/src/trottertracks/algorithm_codesign/g10_reference.py)
- [R5] [future runner](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/scripts/tracks/algorithm_codesign/g10_degree_matched_native.py)
- [R6] [予算・m5再会計・native会計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/src/trottertracks/algorithm_codesign/g10_comparison.py)
- [R7] [保存値の任意proposal下界](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/src/trottertracks/algorithm_codesign/g10_saved.py)
- [R8] [source/authorization launch gate](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/src/trottertracks/algorithm_codesign/g10_launch.py)
- [R9] [focused test source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/tests/tracks/algorithm_codesign/test_g10_degree_preparation.py)
- [R10] [focused test source：API/launch testsも同ファイル](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/tests/tracks/algorithm_codesign/test_g10_degree_preparation.py)
- [R11] [継承numeric API](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/src/trottertracks/algorithm_codesign/rte_reallocation/numeric.py)
- [R12] [per-key guard](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/src/trottertracks/algorithm_codesign/rte_reallocation/launch.py)
- [R13] [shared runtime/guard](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/src/trottertracks/algorithm_codesign/synthesis_placement/wrapper_launch.py)
- [R14] [focused test保存記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/artifacts/track_b_g10_degree_preparation/2026-10-10/focused_tests_v1.json)
- [R15] [read-only準備照合記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/artifacts/track_b_g10_degree_preparation/2026-10-10/preparation_verification_v1.json)
- [R16] [固定runtime preflight](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/artifacts/track_b_g10_degree_preparation/2026-10-10/runtime_preflight_v1.json)
- [R17] [source manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/artifacts/track_b_g10_degree_preparation/2026-10-10/source_manifest_v1.json)
- [R18] [pending authorization](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/artifacts/track_b_g10_degree_preparation/2026-10-10/authorization.json)
- [R19] [保存policy・claim監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/docs/tracks/algorithm_codesign/g10_saved_policy_and_claim_audit_20261010.md)
- [R20] [先行研究・claim map](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/docs/tracks/algorithm_codesign/g10_prior_art_and_claim_map_20261010.md)
- [R21] [継承P5Closed](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/src/trottertracks/algorithm_codesign/g9_p5.py)
- [R22] [read-only準備verifier](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/scripts/tracks/algorithm_codesign/verify_g10_source_preparation.py)
- [R23] [採用G9 v2レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/docs/research/track_b_G9_v2_scientific_review_20261010.md)
- [R24] [レビュー中に確認したbranch ref](https://api.github.com/repos/HIROMU1015/Partially-Randomized-Trotter/git/ref/heads/track-b-g10-degree-comparison-preparation-20261010)：HEAD=S。
- [W1] [mpmath 1.3.0 Utility functions](https://mpmath.org/doc/1.3.0/general.html)：fraction文字列の変換。
- [W2] [mpmath 1.3.0 Basic usage](https://mpmath.org/doc/1.3.0/basics.html)：高精度入力の扱い。
- [W3] [Python 3.10 json公式資料](https://docs.python.org/3.10/library/json.html)：JSON、整数文字列変換の制限。参照時のpatch版は固定runtimeそのものではない。

## 16. 同梱のレビュー検算資料

`g10_source_review_support_20261010/`には、以下を保存した。

- `review_selfcheck.py`：登録科学入力を扱わないレビュー専用検算。
- `review_selfcheck_result.json`：49 checksの記録。
- `reviewed_g10_launch.py.txt`：Git blob identityを照合したsourceのデータコピー。実research packageとしてimportしない。

自己検算を再実行する場合は、このディレクトリ内で`python review_selfcheck.py`を実行する。実repository・Git・backendへアクセスするプログラムではない。固定Codex runtimeの再現実行を代行するものでもない。

**最終判定：Sの固定G10 bundleは、別途明示one-shot承認を受けて実行へ進めてよい。今回のレビューによる新しい科学実行・実authorization・実marker作成は0。**
