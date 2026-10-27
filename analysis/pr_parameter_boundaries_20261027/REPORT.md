# PR資源評価：探索パラメータ境界の調査

2026-10-27 JST。**既存SOURCE・保存結果の調査、離散パラメータ境界の照合、追加検証設計まで完了。新規科学計算・sampling・build/compile・signal/normalization/shot/費用統計の再計算・再ランキングは0。ここでmandatory STOP。**

最も価値の高い次の対象は、H4ではm5・T=2のR_outer上限側、H6では同じm5・T=2のR_outer下限側である。H8はまず既存q16結果を再利用できる範囲を確認し、旧希少RTE研究を再開しない。H2はqの下限ではなく、固定していたR_outer=32の両側を調べる余地がある。Pauli-JWの純RTEでは大event数の追加結果が既にあり、同じ延長を繰り返す優先度は低い。

## 調査の範囲と読み方

DF-nativeの主比較はlinear H2/H4/H6/H8、隣接1.00 Å、STO-3G、保存tol-only DF threshold 1e-8、rank 3/10/19/28。H6/H8の保存weighted-Hermitian completion、元state/reference/scalar/fragment順を維持する。T=.8（共通PF・H8）、T=.8/1.2/1.6/2（H4/H6長時間）。精度は主にε=.05/.01/.005/.001/.0005で、元H8 mapだけは最後の.0005を含まない。δ=T/q、状態準備P=0、cosine/+Im sineのfull Hadamard wrapper、Qiskit1.3.0・rz/sx/x/cx・opt1・seed17の保存会計を参照する。DFのprefixはone-body correctionに加えて保持するfragment数である。

Pauli-JWは別区分。linear H2/H4/H6・1.00 Å・STO-3G・T=.8・S2、非identity項数14/184/918、abs係数順prefix、central-rotation control、仮想compiled残差ρ=0の保存会計である。DFのprefix・R_outerとPauliのL_D・rを同一視しない。旧generic-controlや旧DF rank12等の固定pilotは、主比較へ統合していない。固定Tのsignal資源評価であり、energy精度・全QPE/RPE最終総費用の評価ではない。

**BOUNDARY_INVENTORY.csvの1行は独立実験ではない。** 保存最良行×パラメータ×境界の定義ごとの記録で、同一物理候補の複数ε・複数cohort・履歴snapshotは分けている。`physical_candidate_key`、candidate IDと科学条件で再利用を追跡する。旧H4/H6と追加7候補は同じ11,632候補の価格coverage差であり、二つの独立探索として加算しない。

|boundary_basis|意味|
|---|---|
|REGISTERED|そのsystem/T/ε/family/**実際の公式**に登録されたパラメータ値の境界|
|ELIGIBLE|保存signal条件が適格だった候補の境界|
|PRICED_REGISTERED|その物理候補のcompatible価格が保存されている領域の境界。別εの不適格性と価格の有無は別|
|PRICED_ELIGIBLE|適格かつ価格取得済みで、今回の保存最良値の比較に使用できる領域の境界|

`LOWER/UPPER`は下限/上限、`FIXED`は値が一つの固定設計またはsingletonである。固定Full PFのprefix=rank、純RTEのprefix=0、物理的にq≥1という制約を、拡張すべき探索端とは扱わない。ELIGIBLE/PRICEDだけの境界にあることは、登録領域外の拡張が必要という意味ではない。条件付き登録後に未実行のPauli K4構成は登録領域に含め、適格性は未判定として扱った。

選抜範囲は`formula_scope`に別記する。`ALL`は保存済み全公式比較、`2nd/4th/4th(m5_best)`は公式内比較。同じALL最良がその公式内でも最良であるという論理的な制限だけを追加7候補へ適用した。他公式の更新後最良を新たに計算してはいない。

## 系ごとの優先対象

|系|研究価値が最も高い境界／不足|最小の次候補案|今回の判断|
|---|---|---|---|
|H2 DF|R_outer=32、K4、prefix1を固定したPR設計|S2/p1/q1/K4とYoshida4/p1/q2/K4でR16/64をR32と対照|P2。全公式PFは保存5精度全てでPRより低費用。q1→q0は不可|
|H4 DF|T=2、m5/p3/q1/R64/K2が追加後のPR探索点最良|既存R16/K2の未取得価格を補う対照と、新R128/K2|P0 coverage補完＋P1境界拡張。回路単価だけの外挿では結論不可|
|H6 DF|T=2、m5/p7/q1/R16/K6のR下限・K上限|まずR11/K6、K8は後順位|P1。m5は11 pieceなのでR8は実装上不可|
|H8 DF|元mapのq8/r4/K4上限とprefix7下限|先に保存q16/p7の再利用。新規ならp6/q8/r4/K4またはp7/q8/r8/K4を別scopeで|P0再利用確認→P2別研究案。旧希少RTEは正式終了のまま|
|Pauli-JW|Hybridのq/r/K境界、純RTEのevent数とq分解の区別|純RTEの既存拡張を再利用。H4 Hybrid q16は任意の低順位対照|大event数への一律拡張は推奨しない。H8の保存Pauli結果は指定資料にない|

具体的な案は[BOUNDARY_EXTENSION_CANDIDATES.csv](BOUNDARY_EXTENSION_CANDIDATES.csv)の12行に整理した。12は実行候補数の上限ではなく、再利用のみ・coverage補完・拡張・保留を含む設計項目数である。H2の4案のように1行が複数物理候補を表す場合も明記した。

## H4・T=2・m5_best：R64の先を調べる価値

追加7候補の保存探索集合では、`45f9fc0a53c1a37c7345`（prefix3/q1/R64/K2）がε=.005/.001/.0005のPR点最良を更新した。Rは元m5の登録値16/32/64の上限、q1は物理下限、K2は登録下限である。独立確認を探索へ混ぜない。以下はε=.0005の保存値の転記。

|同じprefix3/q1の構成|保存N_cos / N_sin|保存log_B|保存G|標本SE・取得状況|
|---|---|---|---|---|
|R16/K2 `f85eb7b56ba5131d1eb1`|145036571 / 114372825|0.22022101070674038|未取得|signal適格、compiled価格なし|
|R32/K2 `d9cb4e923530349e940a`|113379811 / 93874975|0.10424677484445503|7498501185789.556|55880659384.189064、元探索|
|R64/K2 `45f9fc0a53c1a37c7345`|101917842 / 84750408|0.05158697006970798|7427614516704.546|44085966117.88573、探索n33|
|同じR64/K2、独立確認|同じ保存N|同じ保存値|7411817334226.875|8660467438.08822、n800|

R32→64では保存shot数とlog_Bが減り、回路単価は増えた。観測Gの小幅低下は、さらにRを増やす価値を調べる理由になるが、R32価格のSEと希少eventの未保証が残るため単調改善や母平均の順位は確定できない。旧PR最良R32/K6との確認点差も、双方SEを含む既存診断では判断不能だった。[追加7候補報告][seven]

R128は16→32→64の次の倍増で、既存allocationは整数R≥11を受理し、R64というhard capを持たない。新規値のsignal・normalization・bias/headroom・shotが未取得なので、**新R128の費用改善やbreak-evenを今回は計算しない**。R16/K2の価格取得は新しいsignalを必要とせず、同じKのR16/32/64という有用な対照を少数の新規物理候補で得られる。R32/R64の完成価格を再取得しない。

R64/K4・K6も登録・適格だが価格未取得である。これはK下限にあるというだけでK0や全K拡張を優先する根拠にはならない。初回はRの効果を同じK2で調べ、別Kは必要性が判明した場合の対照に留める。[元候補会計][long-account]

## H6・T=2・m5_best：登録下限16と実装下限11

`efbb07c848b303a6abab`（prefix7/q1/R16/K6）はε=.01/.005/.001/.0005の保存PR全公式点最良。m5のregistered R16/32/64の下限、K2/4/6の上限にある。ε=.0005の保存Gは`311674817242750/9`、SE=58544949454.47112、元探索n9。全PFの保存最良Gは103354217529528で、PR有利な既存セルの改善を調べる位置づけである。

`h46_m5_plan.weights`は11 signed piece（負のpieceを含む）を返す。元`allocation`は`total < len(coefficients)`を拒否し、各pieceへ最初に1 step、残りを絶対時間比例の整数allocationに配る。したがって、同じscheduleなら**R≥11が必要、R8は不可**。qを変えてこのouter内制約を回避できない。R11は各piece1の最小点で、R12〜15も整数として可能だが初回に全点を広げる理由はない。R11案はbinary64 signed係数・piece順・scalar・controlを維持する。[allocation SOURCE][allocation]

R16のK2/K4/K6は既に価格取得済み。ε=.0005ではK4/K6の保存Nがどちらも81429322 / 111370753、log_Bも0.01778372647230693 / 0.017783726531665912と近い。保存Gは34704570477994.445 / 34630535249194.445で、有限価格標本の差をK6の母平均優位と解釈しない。R32/K4はG=35816308699749.0で、保存点ではR16より悪化している。R32/K2・K6とR64群には価格未取得が残る。[元候補会計][long-account]

K8は一般のfinite Taylor法則が許す次の偶数cutoff。signal側多項式は0〜K+1、builderはeventのapplication sequenceを処理する構造で、固定上限6ではない。ただし、**一般実装の受理可能性と、このH6/m5/K8の正確性・費用検証は別**。K8のlaw/mean/signal、normalization/shot、sample、native/compiled代表検査が新たに必要。K4→6で保存shotの改善が見えないため、K8はR11より低いP3とする。[Taylor SOURCE][rte-source]

## H8：元mapの境界と後続q16は分ける

元resource-mapはB0 prefix S2、Full PF S2/Yoshida4、PR partial S2、prefix7/14/21、q1/2/4/8、r1/2/4、K2/4、計92 signal候補。実適格候補のcompiled価格に欠測はなかった。ε=.001の独立n32後の点最良`B2_p7_q8_2nd_r4_K4`は、q・r・Kの上限、prefixの下限を同時に持つ。保存G=15360006659421.375、paired SE=45479861536.20314。これはFull PF比較の保存結果で、後続の強化B0+I比較と同じ母集団ではない。[resource-map報告][h8-map]

後続には`B2_p7_q16_2nd_r4_K4`の通常signed・ε=.0005、独立n48の完成価格がある。G=91082200598761.833…、paired SE=144211830223.72、固定B0+I p15/q4/Yoshida4のG=94193016652860。別の候補`B2_p8_q2_2nd_r4_K4`もε=.05で独立n48を取得済みである。よって「H8のq16を初めて測る」「prefixは7しかない」は誤り。[後続固定2候補][h8-confirm]

ただし、引用したq16確認CSVの主評価はε=.0005で、元mapのε=.001の全候補最良を更新した保存行ではない。`current_PR_shots_cost_saved_literals.csv`の120行は旧p7/q2/4/8・K2/4の価格転用で、q16を含まなかった。**q16のε=.001の未発見会計をその場で再計算しない**。まず元mean・law・reference・u・scalar/controlの互換性と保存endpointを照合し、保存価格の再利用を優先する。欠けるのがshot/Gの会計だけなら、新しいsampling/compileを直ちに必要としない。

後続確認の説明にはordinary ancilla、元mapにはsymmetric-directionalという記載がある。同じcompilerの版・basisだけでnative wrapperの価格互換性まで成立したとみなさない。今回のq16発見は既存実験の存在を確定するもので、両scopeのcompiled価格を無条件で移植する受入監査ではない。互換性が未確認なら旧scopeの結果として保持し、重複ではない別条件の取得が必要かを次の設計で区別する。

旧mapの境界を越える別研究が必要なら、同じ保存snapshotでprefix6（q8/r4/K4）、またはr8（prefix7/q8/K4）の1軸変更を優先する。q8/r8とq16/r4は同じq*rでもdeterministic分割とfinite lawが異なる。K6は一般法則上の候補だがK4未観測層の問題を解消する手段ではなく、初回最小範囲から外す。

H8希少RTE研究は**正式終了済み**。S0＋S2_SINGLEの層別有限標本評価と残余層の条件付き感度という採用範囲を保持し、S2_MULTI/S4_PRESENTの追加samplingを提案の前提にしない。旧STOP・SOURCE・grant等は不変。ここで示す境界案は、その終了方針を解除するものでも、既存層別結果の母平均保証を強化するものでもない。[正式終了方針][h8-close]

## H4/H6のS2：公式内の端を全公式の改善と混同しない

保存S2最良のqが**登録上限**にある条件は以下。価格取得済み領域だけの端はCSVに別記した。

|系/T|S2のq上限|PR最良が上限となるε|PF集合最良が上限となるε|
|---|---:|---|---|
|H4/1.2|16|.001, .0005|.0005|
|H4/1.6|32|.0005|.0005|
|H4/2|32|.001, .0005|.001, .0005|
|H6/.8|16|.0005|該当なし|
|H6/1.2|16|.001, .0005|.001, .0005|
|H6/1.6|32|.001, .0005|.0005|
|H6/2|32|.001, .0005|.001, .0005|

T=1.6/2のq32は元研究で既に追加評価済みで、q16が現在も上限とは扱わない。m5のq上限はその拡張の対象外であり、S2のq32を他公式へコピーしていない。[既存拡張SOURCE][s2-extension]

これらS2の上限候補とは別に、全公式の保存最良は高次公式の小さいqを選んでいる。例としてH4/T2/ε=.0005のS2 PR Gは`68897767663795/3`、元の全公式PRは`22381832539567/3`、追加後はR64の`81703759683750/11`。H6/T2/ε=.0005ではS2 PR `419808667909184/3`に対しm5 `311674817242750/9`。これらの数値は保存行の転記で比率を新たに計算していない。

したがってS2のq32/64への一律延長は、S2研究として意味があっても全方式最良Gの改善を直ちに狙う最小案ではない。例示するS2/q32対照は、未登録のH4/T1.2・prefix1/R4/K4の1候補だけをP3とする。H6/T.8のq32も未登録だが、全公式最良を重視する初回案では優先しない。

## H2：qの物理下限と固定outer32

共通PFはq1/2/4/8、rank3、Full/B0/B0+I、3公式を比較した。対応する保存PRはprefix1・K4・R_outer32・q1/2/4/8の12候補。R32は「探索して32が下限・上限で最適だった」結果ではなく、各outerで全piece合計32に固定した設計である。[outer32結果][common-pr]

保存全公式PF最良はε=.05/.01/.005/.001/.0005の全てで保存PR最良より低費用。厳しいε=.0005ではPF S2/Full/q4のG=370101635730、PR Yoshida4/p1/q2/R32/K4のG=463148646634.41174（SEは保存行に保持）。q1に下げられないセルもあるが、q0は正の整数stepとして不正で、改善候補にはしない。[共通PF最終結果][common]

H2で最も有用な探索設計の不足はR固定である。低qのS2と厳精度のYoshida4についてR16/64をR32と対照にする4候補案を示す。S2の1 piece、Yoshida4の3 pieceという最低step条件を満たす。同じK4・prefix1に固定してRの効果を調べ、K・prefix・全公式を一度に拡張しない。元H2固有compiler phase guardを含めて再利用可能性を確認し、同名compiler hashだけでH4/H6の取得kernelと同一とはしない。

## Pauli-JWの別区分

最新の保存family最良は[latest15.csv][pauli15]をそのまま引用した。Hybrid・純RTE・Full PF・B0/B0Iは別familyで、B0/B0Iがconstantを保持して同等となるPauli区分をDFのB0/B0+I差へ転用しない。S2だけが比較されているため、他PF公式の最良はない。

- **H2**：Hybrid p1/q1/r4/K4はq下限・prefix下限・K上限。純RTEは緩い2精度でq1/r4/K2、厳しい3精度でq2/r2/K4が保存最良。prefix0は既に測った純RTE境界で、Hybrid p1の下限から未探索の負prefixへ延長する話ではない。追加確認で既存s4の価格coverageが改善した結果であり、大s64/128は既に高費用だった。
- **H4**：緩いε=.05はHybrid p36/q2/r8/K4、他4精度のHybridはp1/q8/r8/K4。後者はq・r・K上限、prefix下限にある。純RTEの保存最良はq8/r8でK2またはK4、s64。**純RTEのr上限は後続で16/32へ拡張済み**なのでr8を現在の登録上限と誤記しない。s128/256はB/Nが減ってもC増加が勝って悪化。Hybrid p1/q16/r8/K4は別lawの任意P3対照で、この純RTE結果から数学的に除外したわけではない。
- **H6**：厳精度のHybrid p78/q8/r8/K2はq・r上限、K下限。ε=.05のHybridはp684/q1/r8/K4。純RTEはq8/r64/K2、s512が全5精度の保存点最良で、**s1024（r128）まで既に探索され悪化した**。したがってq8が端でもevent数の未探索上限に最良があるとはいえない。q/rの分解変更が丸め・seed・compileまで同一になるとも仮定しない。
- **決定論PF/B0**：H4の厳しい2精度はFull p184/q8・B0 p180/q8、H6はFull p918/q8・B0 p864/q8がq上限。これは決定論family内の課題で、最新Hybrid/純RTEに対する全方式勝者を新たに計算していない。
- **H8 Pauli-JW**：指定総合資料と関連結果索引に保存資源評価を見つけていない。未取得・不適格・費用0という意味ではない。

純RTEのK4条件付き拡張には、事前triggerが成立せず**登録のみで実行しなかった候補**がある。登録済みでも適格性・価格は未取得であり、適格候補の価格欠測と同じ扱いにしていない。元研究の未確定数値guard、仮想ρ=0、有限標本・事後選択・母平均tail未保証は残る。[Pauli総合報告][pauli]

## 最小の追加検証設計と停止境界

最初に価値が高いのは、H4/R16の**既存signalに対する価格不足補完**、H4/R128とH6/R11の**新規signal候補**、H8/q16の**再利用確認**である。これらを一括の全格子探索へ広げない。

新しいR・K・q・prefixでは、signalだけでなくnormalization、bias/headroom、保存規則に沿うN_cos/N_sinも変わり得る。別途実行承認後の順序案は、(1)保存input/referenceとコード互換性の確認、(2)対象を固定した新signal・normalization・shot評価、(3)適格候補の価格標本計画を事前固定、(4)元builder/代表correctness付きの両軸取得、(5)探索と独立確認を分けた費用比較、(6)終了監査とSTOP。価格単価の比例外挿や元選抜proxyで候補を完全除外しない。

今回の禁止範囲に数値再解析が含まれるため、新しい標本数・差の区間・break-even価格は計算していない。CSVの`trajectory_n=NOT_FIXED`はこの境界を示し、元のn9や7候補研究の大nを無条件に流用しない。少なくとも新候補側と旧PR参照側のSE、目標とする費用差、rare event・未観測費用を含む限界を、sampling前の別設計で固定する必要がある。保存Nをそのまま使える既存R16価格補完と、新しいNが必要なR128/R11を区別した。

すべての案で元Hamiltonian/state/referenceの保存再利用を優先する。元入力が復元できなければ、別モデルで代替・無断生成せずpreflightで不足を報告する。新科学範囲は未承認、既存H8終了方針やB3・別チャット研究・整理作業は変更しない。

## coverage・判断不能箇所・保存物

主なデータ範囲は、H4/H6長時間11,632物理候補×5εの58,160行、共通PF588＋固定outer32 PR36の624候補、元H8 map92候補×4ε。これらのendpointは独立実験数ではない。境界表は保存選抜行569件を参照し、境界4,665行、後続H8固定対照40行、保存method別最小行がない495行、H8 Pauli資料なし1行の計5,201行。表の行数は実測候補数ではない。

**網羅性の限界を欠測行として残した。** 長時間の`best_configurations`はPF集合とPRの全公式/公式別最良を保存しているが、Full/B0/B0+Iを別々に全セル再最小化した最良表ではない。共通PFも同様。元endpointの各methodの登録・適格・価格領域は抽出できるが、その別最小Gを今回新計算することはしなかった。該当495行は`NO_SAVED_SEPARATE_METHOD_MINIMUM`で、最良候補が境界にないという意味ではない。

追加7候補後も全公式PR最良表は保存されているが、全公式以外の全family/formulaの更新最小表はない。元320行のsnapshotと追加後ALL40行を混ぜず、追加ALL勝者の自公式に関する論理的帰結以外は未確定として残す。H8後続p8/q16の2つの固定確認も、全登録域の新しい最小表ではない。元mapと後続研究の探索域を無条件にunionした新順位は作成していない。

主表は元研究の通常signed・主allowance・P=0に対応する。非相殺・allowance倍率10・P感度等の保存診断を、新しい比較規則として混ぜていない。これらはアルゴリズムのq/R/K/prefixとは別の評価政策であり、本表からの全条件最適性の主張はしない。

成果物：

- [BOUNDARY_INVENTORY.csv](BOUNDARY_INVENTORY.csv)：候補条件、3種類の領域と価格・適格性の交差、G/SE/n、近傍保存行、未取得ID、技術条件、結果commit・SOURCE参照・保存row。
- [BOUNDARY_EXTENSION_CANDIDATES.csv](BOUNDARY_EXTENSION_CANDIDATES.csv)：優先度、具体案と最小範囲、signal/normalization/shot/sampling/compileの新規必要性。
- [inventory.py](inventory.py)：既存JSON/CSVを読むmetadata-only処理。候補索引と保存winner selectorを再利用し、G/N/B/SE・paired統計の既存関数を実行しない。
- [verification.json](verification.json)：読取元blobのSHA256、境界の離散値照合、call=0・再ランキング=0、未確定区分。

旧科学SOURCE・raw・grant・ledger・marker・STOP、共有環境、他worktreeを変更していない。専用branch `codex/pr-parameter-boundaries-20261027`、base `bb97a93d65992fdb53e642dde1584dd7d5b811d7`。今回の結果はローカル保存資料の調査で、独立外部再現・最終科学保証ではない。

**mandatory STOP：追加科学計算、数値再解析、sampling、build/compileは開始しない。**

[common]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f75a7c25d584b69d90de454731a7f343a421e059/docs/research/track_a_h246_pf_common_final_20261022.md
[common-pr]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f75a7c25d584b69d90de454731a7f343a421e059/docs/research/track_a_h246_saved_cost_results_20261021.md
[long-account]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/candidate_precision_accounting.csv
[seven]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb97a93d65992fdb53e642dde1584dd7d5b811d7/analysis/h46_seven_cost_readiness_20261027/REPORT.md
[allocation]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/scripts/h46_new2_core.py
[rte-source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/src/trotterlib/rte.py
[s2-extension]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/scripts/h46_long_time_extension.py
[h8-map]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/docs/research/track_a_h8_resource_map_result_v1.md
[h8-confirm]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/track_a_h8_new_pr_confirmation_results_20261019.md
[h8-close]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md
[pauli]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/research/pauli_jw_resource_evaluation_summary.md
[pauli15]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/research/data/pauli_jw_summary/latest15.csv
