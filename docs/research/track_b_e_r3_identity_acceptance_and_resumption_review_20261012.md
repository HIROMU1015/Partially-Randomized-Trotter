# Track B：E-r3診断結果を踏まえた受理基準・既存証拠の扱い・比較再開の科学的レビュー

- 作成日：2026-10-12 JST
- 開始承認：利用者の「レビューを開始して」
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 対象の公開commit：`ef0b2419df2d25187ad14e865ae0a38d17a04628`
- 対象branch：`track-b-e-r3-identity-only-v2-one-shot-execution-20261012`
- 今回の診断source S2：`b9519c29b5f1528ad5c8a6e398b2fa12062db8c6`
- 今回の診断authorization A2：`5e8baf475dd086e85e4b2611cc20b1f77161e7d5`
- レビューの範囲：保存されたE-r3診断の解釈、fingerprintを用いた科学的受理基準、旧資料の再利用、限定A/B比較へ戻るための方針。
- 対象外：新しい診断・行列計算・コンパイル・T合成、whole方式の優位性や独立新規性の確定、Track B全体の再設計、他のPR研究の横断調査。
- 運用：共通指示2026-10-12版を適用。レビューの実施と計算実行の認可を分離する。

## 0. 最終判断

**E-r3の原因調査をさらに反復するのではなく、過去のrecursive fingerprintとの完全一致を必須とする受理基準を、新しい比較契約で改訂する。そのうえで、既定のE-r3／E-r6の局所A/B費用比較と限定T合成へ戻る準備を進める。**

今回採る方針は、単にguardを削除することではない。次の四つを分ける。

1. 保存入力・source・runtime・科学条件の同一性。
2. 指定された理想演算子に対する、位相を保持した数値的な正当性。
3. 実際に得たnative回路・角度・合成sequence・費用の一貫した結合。
4. 生のrecursive payloadとそのhashによる表現・来歴の記録。

旧expectedとの一致は、4の歴史的な照合値として保持し、1〜3の代わりに用いない。原payloadとhashの内容は変更しない。全operation名の一括削除、新expectedへの置換、name counterの調整は採用しない。

この判断は**今後の比較に用いる科学的な受理方針の改訂**であり、旧失敗の取消でも、新実行の承認でもない。旧比較の`TECHNICAL_INCONCLUSIVE`、旧marker、旧認可、今回の診断resultはそのまま保持する。

次の担当はCodex。契約への差分反映、比較側の受理gateの実装、登録外の独立検証、provenanceと実行保護、source固定までを一つの技術準備作業にまとめる。その後の登録E/T実行には、別途のユーザー明示承認を必要とする。

本レビューの判断を表す説明用ラベル：

`ADOPT_LAYERED_ACCEPTANCE_AND_PREPARE_E_T_REEVALUATION / EXECUTION_NOT_AUTHORIZED`

これは本書の評価であり、リポジトリ内のstatusを変更したものではない。

## 1. 根拠と読取り範囲

### 1.1 正本

結果・引き継ぎ報告、`identity_result.json`、行列段階比較CSV、current／legacy原結果の関連部分、実際の診断sourceを取得し、相互に対応させた。[R1–R6]

旧失敗の欠測記録、旧A/B実行報告、比較契約v2、実行計画、CD保存値抽出、共通fingerprint sourceも参照した。[R7–R12]

取得時の対象branch HEADは公開commit `ef0b2419...` と一致していた。今回はこの固定snapshotを評価し、その後の別branchや未報告の結果を混入させていない。[R13]

### 1.2 自分で実施していないこと

Codexの保存値監査421項目をGPTが全項目再実行したわけではない。current／legacyの全raw bytesと全payloadを別プログラムで再監査したわけでもない。今回の評価は、保存された一次の比較結果、原観測の関連箇所、比較処理のsource、既存契約の読解に基づく。[R1–R6]

登録Q/T/Uの再生成、payload再取得、native transpile、T合成、量子shot、分子計算、既存runnerの起動は行っていない。新しい実験数・テストPASS数を作っていない。

確認範囲の説明と、後述の科学的判断を区別する。新たな数値検証をしなかったことは、保存された証拠から受理方針を判断しなかったという意味ではない。

### 1.3 外部資料

Qiskit 1.3の公式`QuantumCircuit`文書を補助的に確認した。回路名が`to_instruction()`／`to_gate()`後の名前になり、global phaseがcontrolled利用に影響し得るというAPI上の説明を参照した。[E1]

本件では既にリポジトリに観測された差分があるため、外部資料を使って欠測した過去の原因を埋めてはいない。包括的な先行研究調査や最新アルゴリズムの新規性調査は、今回の範囲外である。

## 2. 診断で確定したことと、確定していないこと

### 2.1 完了したのはidentity-only診断

保存結果のstatusは`E_R3_IDENTITY_DIAGNOSTIC_COMPLETE`。current→legacyの順に各一回、同じ保存frameと固定runtimeを用いて、uncontrolledのE-r3を再構成した。診断sourceとauthorizationは、以前のA/B比較のsource・authorizationとは別である。[R1,R2]

旧A/B比較の最終statusは依然として`TECHNICAL_INCONCLUSIVE`である。今回の診断完了を、A/B比較の完了と読み替えない。[R8]

### 2.2 数値行列の一致

次の10段階でdtype・shape・C-orderのraw bytesが一致した。[R2,R3]

| 段階 | current／legacyの比較 |
|---|---|
| frame_0、frame_1、frame_2 | 一致 |
| Q_0、Q_1、Q_2 | 一致 |
| T | 一致 |
| U | 一致 |
| input_target | 一致 |
| actual_gate_parameter | 一致 |

`first_different_stage=null`である。

U、input target、実gate parameterの共通SHA256は、

`744d3c221bd0064dded5032aec24cd13d8f0f718ceb87d63acbda410b8fa0e77`

である。[R1,R3]

これは各配列の値のbytesと表現型の比較であり、メモリの所有権・writeable flag・全過去runtime・全BLAS条件の一致を主張しない。currentとlegacyが一般の全入力で常にbit一致するという定理でもない。

### 2.3 回路表現の差

二つのrecursive payloadの差分は、次の4 fieldだけだった。[R1,R2]

| definition内instruction index | current | legacy |
|---:|---|---|
| 0 | `circuit-230` | `circuit-311` |
| 8 | `circuit-233` | `circuit-314` |
| 18 | `circuit-236` | `circuit-317` |
| 28 | `circuit-239` | `circuit-320` |

差分pathはいずれも`.../operation/definition/instructions/<index>/operation/name`。

current hash：
`406640bb455fae569df5f41c0079ba2d90fbc34993b1765ceca8b49714cdb1d4`

legacy hash：
`73965b732709af19321489dd1a4dbdfb899ad39743b54ba5c60e94da2b0bfb7e`

旧expected：
`171f321e2f3bcc51b177e7b62735b5fbf9d80e36a5f56639b7f03809efac41b5`

両actualは非nullであり、旧expectedとは一致しない。今回のfingerprint例外は0だった。[R1,R2,R4,R5]

### 2.4 definitionを評価する仕事も実行されている

二経路合計でpayload取得2回、`UnitaryGate._define` entry 2回、QSD Python entry 10回を記録している。native transpileは0である。[R2]

「native transpileをしていない」ことと、「内部の回路分解や数値処理をしていない」ことは別である。今回の診断は前者に限定されており、内部仕事は実行監査へ計上されている。

### 2.5 旧失敗の直接原因は欠測が残る

旧実行ではactual hash、例外、入力unitary bytes、失敗native IRが未保存である。旧expectedを生成した完全payloadも利用可能な証拠に含まれない。[R7]

従って、次は区別する。

- **確定**：今回の二経路の行列bytesは一致し、その二つのpayloadの差は4個の自動名に局在する。
- **未確定**：旧実行の不一致も同じ4 fieldだけで生じたか。
- **未確定**：旧実行で非null hashが異なったのか、例外によってNoneとなったのか。
- **未確定**：当時の数値経路・layout・runtimeに別の差があったか。

新たにlegacy helperを実行しても、それは現在のruntimeによるlegacy-source reconstructionであり、欠測した歴史的な実行そのものではない。[R1,R6]

## 3. 科学的解釈：このhashを何の検査に使えるか

### 3.1 hash関数の故障ではなく、検査対象の問題

共通fingerprint sourceは、operationの型、名前、parameters、qubit数、control、global phase、wire順序、recursive definitionなどをsorted JSONへ直列化してhash化する。[R12]

概念的に、回路CのpayloadをP(C)、そのhashをh(P(C))、実装演算子をU(C)と書く。

今回の証拠は、同じ入力行列を持ち、捕捉されたpayloadの差が自動生成名だけである二つのCについて、hashが異なるという具体例である。[R2,R3]

従って、少なくとも今回の用途において、

`同じ入力演算子・同じ数値定義 ⇒ 同じrecursive hash`

は成立しない。異なる文字列をhashすれば異なる値が得られること自体は、hash関数の誤動作ではない。

**名前まで含む表現の厳密一致を、入力演算子が正しいかどうかの必要条件として使ったことが不適切だった。**

### 3.2 旧設計への評価変更

旧比較契約v2では、Aの四つのgeneric compileについて過去のE fingerprint一致を必須とした。[R9]

前回まで、その条件を維持したまま診断を整備していた。診断前は、値の違いなのか、表現だけの違いなのか、None分岐なのかを観測できなかったためである。

今回は登録E-r3の証拠が得られたため、**今後の科学的な受理基準として旧recursive hash一致を必須とする方針を変更する。** 過去のレビューでこの条件を実行前提としていた判断も、現在の証拠に照らして修正する。

この変更は「失敗したので条件を緩くして通す」こととは異なる。目的に適合しない判定を、入力・数学的正当性・実測実装の結合を確認する判定へ置き換える。位相、数値精度、source、入力、資源上限についての条件は緩めない。

### 3.3 名前の一括削除は採用しない

今回の4つの自動名の差が値の差を意味しないことは支持される。しかし、任意のoperation名は無意味であり、全回路の全名前を除けば安全なsemantic hashになる、とは示されていない。

native basisのgate名やcontrolの意味まで消してはならない。shared library全体のhash policyを新規設計することも、今回の限定比較に必須ではない。

raw payload・raw hashはそのまま保存する。必要なのは、新しい比較の受理gateが過去のraw hash一致に依存しないことであり、hashから名前を削って過去expectedを再現することではない。

### 3.4 旧原因の完全確定を再開条件にしない

今回の証拠は、将来の受理基準を変更する理由として十分である。過去の原因の一意な確定とは別の命題である。

追加診断が無価値だという普遍的判断はしない。ただし、旧payloadを取得できないままcurrent／legacyを何度も作り直すより、今後の検証が正しいtargetを扱う仕組みを定める方が、本研究の資源比較に直接役立つ。

**本件の原因探索を、通常の次工程としてはここで区切る。** 新たな矛盾する証拠や数値保証の破綻が出た場合のみ、その問題へ戻る。

## 4. 今回採用する新しい受理構造

以下は保存resultの主張ではなく、本レビューで採る将来の比較設計である。

### 4.1 層A：入力・科学条件・実行来歴を固定する

共通入力は、既に保存されたE-inputsのr3／r6 frameとする。既存fileと各frameのbytes・dtype・shape・順序を参照し、再seed・別frame選択をしない。元の保存frameから契約の理想frameを定義する。

r3／r6、rank2×3、signs(+,-,+)、nested commutator、u=1/128、control条件、basis、compiler version／seed／optimization、誤差会計を維持する。[R9]

旧recipeの再実行を増やす代わりに、保存入力の読込みを新しい契約へ明記する。入力取得の実装経路が変わることと、別の科学的入力を選ぶことは区別する。

source・runtime・contract・manifest・authorization・marker・fresh result namespaceの保護も維持する。

### 4.2 層B：同じ理想targetへの距離を、位相を保持して確認する

Aのevent targetは、共通の理想Tから定義する

`V_A = exp(i u T)`

である。現在のraw行列をそのまま「理想target」と定義して自己一致をPASSにしてはならない。

既存のassembly distanceと独立spectral referenceの条件付き誤差モデルを用い、概念的には

`||V_native - V_target|| <= reference error + guarded native/reference difference`

を固定許容域内で確認する。

T合成も含むfull local accuracyは、既存のnumeric allowance 2e-7とsynthesis allowance 8e-7、合計1e-6の条件を維持する。[R9,R10]

Bの一eventはPauli回転であり、weighted平均がAのeventと全く同じになるわけではない。BにはPauli proxy距離、確率丸め、weight丸め、線形化誤差を別に課金する。A/Bを同じ有限step平均だと仮定する受理検査にはしない。[R9]

### 4.3 controlled位相を捨てない

system上でV'=exp(i phi)Vであっても、controlled化すると

`C(V') = |0><0| ⊗ I + |1><1| ⊗ exp(i phi)V`

となり、一般にC(V)と共通global phaseを除いて同じとは言えない。従って、phaseを最適化して小さくした残差だけでは、今回のcoherent signal用の正当性確認には足りない。

control-zero block、active block、wire順序、global／relative phaseの扱い、strict RZ／W phaseの既存条件を維持する。Qiskit公式文書にも、回路global phaseのcontrolled利用への影響が記載されている。[E1]

### 4.4 層C：実際に使うnative実装と費用を結合する

compile後のnative IRを保存し、順序付きgate、qubit、角度のbinary64値または固定semantic表現、その変換誤差、global phase、basis、settingsを結び付ける。

費用はこの保存された実装から測定し、T合成対象inventoryも同じIRへ結び付ける。inventory作成時のdigest、各key、取得sequence、出現multiplicity、最終費用・誤差coverageの関係を保持する。

**同じ演算子を実装できることと、以前の回路費用を流用できることは別である。** 同じunitaryに異なる分解を使えば費用は変わり得るため、native単価は新しい実装から取得する。

最小回路を証明することまでは要求しない。固定した実装方針で得た費用であることを明示し、best-of-many compiler探索は行わない。

### 4.5 層D：raw recursive payloadは来歴として保持する

今回のhash policyを改造せず、raw payload、actual hash、historical expected、一致／不一致を保存する。

historical mismatchは、それだけを理由に新比較を不受理にしない。一方、sourceや保存入力の改変、別target、位相不整合、数値保証失敗、native IRとT inventoryの不一致は依然としてSTOP理由である。

payload作成例外や必要資料の保存失敗を、単なるhistorical mismatchと混同して成功扱いしない。必要な観測・来歴を取得できない問題は、technical failureとしてCodexが切り分ける。

## 5. 旧資料をどう再利用するか

| 資料 | 扱い | 今回認めない扱い |
|---|---|---|
| 完了済みCDの6 task | 元source・contract・exact保存値の来歴付きで、条件付き誤差／十分shot予算／記号費用の資料として保持 | 新Eの単価を掛けて実測PR総費用にする、旧全体runを成功へ書換え |
| 保存E-inputsのr3／r6 frame | 新比較の固定入力として再利用 | 新seed・有利なframeの選び直し |
| 旧Phase Eの費用表 | 旧環境・旧sourceで得た歴史的な参考値 | 新しいnative実装の実測単価としてコピー |
| 失敗したE-r3のprefix／E-r6の未完了記録 | 失敗履歴として保持 | 完了した新費用表の不足行を補完する |
| 今回のE-r3行列／payload | 受理設計の根拠と技術regressionの参考 | r6・controlled・全runtimeの同一性証明として拡張 |
| 過去expected hash | 歴史的照合値として保持 | 新actualへの上書き、差分を隠した受理hashの作成 |

CDは旧全体runでは`INCOMPLETE_RUN_DIAGNOSTIC_ONLY`という位置付けで保存されている。このstatusを無視せず、後の新報告で「旧runから保持した完了substage」と明示する。[R8,R11]

本レビューで許容するのは、この条件付き資料の再利用であり、CD6行の全数学保証を改めて独立再認証したという主張ではない。新sourceがCD数式・確率・誤差予算まで変える必要がある場合、その再利用判断は見直す。

旧Eの費用を新実装へ流用するには本来、targetだけでなく実際のIR・角度・phase・compiler・合成計画の対応が必要になる。本件ではそこまでの歴史的対応を確認できないため、追加の復元調査より新計測を選ぶ。

## 6. 推奨する再開範囲

### 6.1 E/Tへ戻り、CDやidentity-only診断は繰り返さない

新しい比較契約と実行sourceの準備対象を、元のE-r3／E-r6についての局所native比較と限定T合成に絞る。

- A：r3、r6 × uncontrolled、controlledの四つの固定generic compile。
- B：同じlocal T情報を用いた固定Pauli IRと、既定の代表compile。
- native compileの上限は合計22回。
- Tは最大64 logical keysの既定選択規則、B必要keys→phase→A位置keyの順序、no refillを維持。
- 実際のliteral angleは新しいIRから生成し、合成前に固定する。旧角度が同じだったことを仮定しない。
- CDは保存値を読んで来歴を結合するだけとし、再計算しない。
- 診断用のcurrent→legacy比較も再実行しない。

r6を含めるのは新規の科学的対象拡張ではなく、元の限定比較で未完了だった固定対象を評価するためである。r3診断の結論をr6へ証明として適用するのではなく、r6でも同じ新受理規則を実際に検証する。

### 6.2 既定の資源上限と意味論を維持する

新E/T計画は、既存の上限を拡大しない。[R10]

| 対象 | 既存上限・規則 |
|---|---|
| E native | wall 600秒、aggregate CPU 450秒、worker RSS 1024 MiB、aggregate RSS 2560 MiB |
| T synthesis | wall 1200秒、aggregate CPU 900秒、worker RSS 512 MiB、aggregate RSS 1536 MiB |
| T per key | wall 30秒、CPU 20秒 |
| 新実行全体の上限案 | 旧比較のwall 2400秒、CPU 1800秒、aggregate RSS 3072 MiBを超えない |
| 出力・log | 64 MiB／4 MiB、terminal予約と512 KiB以下の取得可能なレビュー資料 |
| 呼出し | native最大22、T最大64、retry=0 |

E-r3 identity-only診断に使ったCPU120秒等の契約を、E/T比較へ流用しない。逆に、今回の方針を理由に旧比較の上限を緩めない。

ここで記載したのは次の実行計画の科学的な範囲であり、実行認可ではない。新しい実行は、既に消費した計算とは別の計算資源を使うため、別途の承認と独立した実行監査を必要とする。旧runのretry=0を書換えない。

### 6.3 native入力の生成順序を後から合わせない

同じ固定frameとsourceから一度生成した実装を評価する。historical hashが偶然一致するまで回路を作り直したり、counterを操作したり、複数compileから安いものを選んだりしない。

過去とのraw hash差は記録し、妥当性は層A〜Cで判断する。これはruntime・精度・compilerの変更許可ではない。

## 7. 公平性：A/Bで何を共通化し、何を共通化しないか

AとBに同じframe、理想T、有限bit・proxy方針、control条件、compiler基盤、評価精度、資源座標を与える。Aのwhole回転とBのPauli pairingという方式差は保持する。

Bのweighted平均はidealでは`I+i u T`であり、Aの`exp(iuT)`とは有限uで異なる。したがって、Bの軽い一回路だけをAの一回路と比べ、同じ精度の方法として即座に勝敗を決めない。[R9]

考えるべき負担は、少なくとも次である。

- A：whole回転の実装費用、positive correctorのbias。
- B：Pauli回転の実装費用、proxy／linearizationのbias、測定weightとmoment・range。
- 共通：元PR／RTE、状態準備、全体の精度・成功条件。

ただし現計画のEはaccepted-correction条件付きの局所screeningであり、元PRの全費用を実測していない。全体への外挿をしないという条件は変更しない。

## 8. この比較から結論できる範囲

### 8.1 局所実装の結果

新E結果が揃えば、固定r3／r6、固定control、固定precisionでのnative資源と、取得範囲内のT費用を比較できる。

Aは固定したgeneric decompositionの一実装である。今回Aが高価なら、その実装の結果であり、構造を利用する全てのwhole方式の下界ではない。[R9,R10]

### 8.2 部分T下界

Aの取得済みkeyについて、出現回数m_kとstrict-valid sequenceのT数t_kから

`L_A = Σ_obtained m_k t_k`

という固定加法実装のcompletion下界を作る方針は維持する。Bにcompleteかつcertifiedな上界U_Bがあれば、`L_A > U_B`をその局所実装の分離として報告できる。

未取得Aの全T数は不明のままでよい。ただし、この下界は最短合成・別分解・cross-gate cancellation・一般whole方式の下界ではない。sourceにある既存のscope限定を維持する。[R9]

coverageが不足し、比較に必要な境界が得られなければ`T_INCONCLUSIVE`である。計算が足りないことを非優位の証拠にしない。結果を見て64枠を埋め直したり、上限を自動拡張したりしない。

### 8.3 CDの記号的費用

旧CDの資源式は、座標xに対し

`G_i,x = 2 N_i [h_x + C_base,x + 2 n Σ_e p_exec,i(e) C_i,e,x]`

である。[R9,R11]

同じCD入力の費用をC0とc_iでまとめれば、純粋な代数として

`G_A < G_B ⇔ (N_A-N_B) C0 < N_B c_B - N_A c_A`

となる。ただし、未知のCD単価を別fixtureのE単価で埋めてはならない。現段階では、これを損益分岐条件として扱う。

Eでの費用差をCDへ数値移植して、分子PR総T、QPE成功率、ground-state energy精度を達成したと報告することはできない。

## 9. 結果を見た後の契約変更であることを隠さない

受理基準の改訂は、元の比較停止と登録診断の結果を見た後に行う。従って、新比較を「最初のv2契約を無修正で完遂した実験」とは書かない。

変更の理由、旧条件、診断証拠、新条件、維持する数値・物理条件を明記したamendmentを、新実行前に固定する。

許容する変更は、主に以下である。

1. 歴史的recursive hash完全一致を科学的受理の必須条件から外し、多層の確認へ置換。
2. 保存済みE-inputsを固定入力として読込むことを明示。
3. 未完了E/Tを新実行で取得し、CDは既存の完了substageとして保持。
4. 新result・authorization・manifest・sourceを明示し、旧結果から区別。
5. 今回解決した科学方針の範囲における最終source技術検証をCodexへ委任する運用を新契約に明示。

fixture、seed由来、rank、sign、u、A/B手法、confidence、bias配分、gate精度、compiler policy、T-key選択規則、上限、retryは結果に応じて変更しない。

旧契約のsource-review要件を黙って無視しない。今回のレビューで受理方針と委任範囲を決めたことを新契約へ記録し、追加の詳細GPT sourceレビューを機械的に必須とする運用は採らない。

## 10. 主な代替案と採否

| 案 | 評価 |
|---|---|
| historical expectedへ一致するまで診断を反復 | 採用しない。欠測した旧payloadと当時のcontextが残り、将来の科学的受理を閉じるのに必要とは限らない |
| expectedを新actualに書換える | 採用しない。旧結果の意味を変え、次のプロセスでも同じ問題を起こし得る |
| 全operation名を除いたhashを主受理基準にする | 採用しない。今回の自動名差の証拠を、任意のgate名が無意味だという一般化へ拡張する必要がない |
| operatorの点残差だけで通す | 採用しない。conditional error guard、phase、input/source/provenance、費用結合が欠ける |
| whole方式を終了する | 現証拠では採用しない。得たのはidentity上の知見でありA/B資源結果ではない |
| layered acceptanceに改訂しE/Tを新規測定 | 採用。科学的目的へ戻りつつ、入力・数値妥当性・実測費用の追跡を維持できる |

## 11. 次の担当と完了条件

### 11.1 Codexへまとめて任せる作業

新しい比較契約差分、Track B比較側の受理gate、入力・結果・T inventoryの結合、診断記録、登録外control、source／manifest／authorization準備、remote取得確認を一つの作業にまとめる。

共通libraryのfingerprint policyを研究全体へ一括変更する必要はない。新比較での採用判定だけを今回の設計へ接続し、旧raw payload・旧helper・旧結果の参照可能性を保持する。

具体的なfile構成、追加test数、小規模mock設計はCodexに任せる。重要なのは、名前だけの差を過去hashの科学的不一致と誤判定しない一方、入力改変・符号・位相・target違い・費用とsequenceの取り違えは見逃さないことである。

準備中に登録E/Tの値を先取りしない。実験開始前にsourceとamendmentを固定し、one-shotの独立認可を待つ。

### 11.2 GPTへ毎回戻す必要がないもの

sourceの小修正、保存不具合の修正、off-domain tests、metadata照合、manifest作成、通常の技術preflightは、今回採った方針内ならCodexで完結させる。

新しいcommitやtests PASSを理由に、同じ数学・受理方針をGPTが再レビューする必要はない。

### 11.3 GPTへ戻す必要があるもの

- 既存の数値保証でtargetを検証できず、誤差許容値や保証の意味を変える必要がある。
- 位相、入力、A/Bの平均、proxy、control条件を実質的に変更する。
- baseline、compiler選択、T precision、主要資源指標、実験対象を変える必要がある。
- 新しいA/B結果を基に、研究の採択・継続・縮小・終了や論文claimを決める。

sourceが実行可能という技術判断と、別途のユーザー実行承認は分離する。レビュー承認を計算承認として扱わない。

## 12. 研究としての着地点

今回の診断は、PRアルゴリズムの新規性や資源優位を直接増やしたわけではない。比較を妨げていた受理基準の役割を明確にし、正しい入力と実装の対応を保って再開する根拠を得た技術的成果である。

従って、主目的を「過去のhashを一致させる研究」へ移さない。既存の問い、すなわちwhole-commutator回転とlocal Pauli pairingのbias・weight・実装費用の交換を評価する作業へ戻る。

新E/Tで差が得られても、まず固定局所実装の知見である。全PRの独立アルゴリズムとして有望かは、同入力の全費用、強い対照、適用条件と新規性を含む後続の科学的判断で扱う。

一方、generic whole実装が不利、あるいはcoverage不足という結果だけで、全whole方式やTrack B全体を終了しない。どこまで識別できたかに合わせて結論を限定する。追加研究は結果後に情報価値を評価し、自動実行しない。

## 13. 本レビューの最終整理

- **レビュー：完了。** E-r3の観測と科学的受理方針を判断した。
- **診断：通常の追加反復は不要。** 旧原因の一意確定は未完了のまま明記する。
- **受理方針：新契約でlayered acceptanceを採用。** historical raw hash一致は来歴照合へ移す。
- **旧結果：変更しない。** incomplete status、marker、認可、欠測を保持する。
- **旧費用：新測定の代わりにコピーしない。** CDは条件付き保存substageとして保持する。
- **次：Codexで新E/T比較の準備をまとめる。** routine sourceレビューの反復は不要。
- **実行：未承認。** 新sourceの直接子認可と、ユーザーの別途明示承認が必要。
- **研究の採択・全PR優位・新規性：未確定。** 今回のtechnical evidenceから先取りしない。

## 参考資料・固定参照

以下の[R1]〜[R13]は、特記しない限り対象公開commit `ef0b2419df2d25187ad14e865ae0a38d17a04628`上の資料を参照する。GitHubのsourceを正本とする。

- [R1] [E-r3 identity v2結果・引き継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ef0b2419df2d25187ad14e865ae0a38d17a04628/docs/tracks/algorithm_codesign/e_r3_identity_v2_results_and_gpt_handoff_20261012.md)。全文取得。
- [R2] [identity_result.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ef0b2419df2d25187ad14e865ae0a38d17a04628/artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v2/identity_result.json)。全文取得。観測、段階比較、payload差分、call counts。
- [R3] [matrix_stage_identity_v2.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ef0b2419df2d25187ad14e865ae0a38d17a04628/artifacts/track_b_e_r3_identity_diagnostic_execution_audit/2026-10-12/v2/matrix_stage_identity_v2.csv)。全文取得。
- [R4] [current.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ef0b2419df2d25187ad14e865ae0a38d17a04628/artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v2/current.json)。今回は先頭130行のprovenance、hash、frame・Qの原記録を取得。全payloadの独立byte再照合とはしない。
- [R5] [legacy.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ef0b2419df2d25187ad14e865ae0a38d17a04628/artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v2/legacy.json)。今回は先頭27行を取得。全結果比較には[R1–R3]と[R6]を用いた。
- [R6] [実行source S2のevaluation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9519c29b5f1528ad5c8a6e398b2fa12062db8c6/src/trottertracks/algorithm_codesign/e_r3_identity_diagnostic/evaluation.py)。全文取得。current／legacyの分離、observer、bytes比較、差分類。
- [R7] [旧失敗の保存metadataと欠測](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ef0b2419df2d25187ad14e865ae0a38d17a04628/artifacts/track_b_df_correction_fingerprint_cause_audit/2026-10-12/saved_input_and_failure_metadata_v1.json)。全文取得。
- [R8] [旧A/B比較の実行結果・technical STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ef0b2419df2d25187ad14e865ae0a38d17a04628/docs/tracks/algorithm_codesign/df_correction_comparison_results_and_gpt_handoff_20261012.md)。全文取得。
- [R9] [比較契約v2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ef0b2419df2d25187ad14e865ae0a38d17a04628/docs/tracks/algorithm_codesign/df_correction_comparison_preregistration_v2_20261011.md)。全文取得。旧fingerprint条件と、変更しない科学的意味論。
- [R10] [比較実行計画v2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ef0b2419df2d25187ad14e865ae0a38d17a04628/docs/tracks/algorithm_codesign/df_correction_comparison_execution_plan_v2_20261011.md)。全文取得。資源上限、strict phase、T coverage、CD/E区別。
- [R11] [CD保存field summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ef0b2419df2d25187ad14e865ae0a38d17a04628/artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/CD_saved_summary_v1.json)。今回は先頭65行取得。原値を再算せず、保存substageの位置付けを確認。
- [R12] [共通fingerprint source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ef0b2419df2d25187ad14e865ae0a38d17a04628/src/trotterlib/rte_compiled_cost.py)。今回は277–430行を取得。名前・recursive definition・phase・wireを含むpayload規則。
- [R13] [対象branch](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/track-b-e-r3-identity-only-v2-one-shot-execution-20261012)。レビュー時にGitHub branch endpointからHEADが対象公開commitと一致すると確認。
- [E1] [IBM Quantum：Qiskit 1.3 QuantumCircuit](https://docs.quantum.ibm.com/api/qiskit/1.3/qiskit.circuit.QuantumCircuit)。外部補助資料。`name`からinstruction/gate名への対応、global phaseのcontrolled利用への注意。

### 参照上の留意点

Codexの全421項目の独立再実行、全31取得receiptのGPTによる再照合、新runtimeでのQiskit再現試験は行っていない。全raw payloadのfield比較は保存された結果と監査に依拠し、その比較sourceを読んでいる。

この範囲で、名前を含む歴史的hashを今後の科学受理の必須条件とする設計を見直す判断は可能である。一方、旧実行の直接原因、任意の入力での数値bit同一性、任意のcompilerの費用、新規アルゴリズムの成立を確定する根拠にはしない。
