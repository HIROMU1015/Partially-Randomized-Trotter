# Codex C向け：残差用構造特徴量と期待値の定義

2026-11-13 JST。[評価方法](REPORT.md)／[最小科学案](MINIMAL_SCIENCE_PLAN.md)。
**定義の引渡しだけであり、Cへの数値実行指示ではない。** 既存E[S]・K0/K1・SOURCEを変更しない。
以下の新feature期待値は今回未計算。Dのwindow補正を実測する前に、全featureを一括実装する必要もない。

## 1. identity・law・回路範囲

固定対象は元H4 DF10/S2/L_D1/ΔPF=.1/K2/h=.1/M12/P0、canonical
`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`、preparation
`b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7`。
元H/state/snapshot、fragment/basis/qubit順、symmetric-directional、scalar一回、full Hadamard X/Yを保持する。
loweringは`CONTROLLED_ZZ_CX_CRZ_CX_V1`、compilerはQiskit1.3/rz,sx,x,cx/opt1/seed17。

featureは①whole full-wrapper S、②tail application列のdescriptor、③boundary rewardを区別する。
②をfull compiled priceへ直接ラベル変更しない。component/basis辞書とlaw source、R、q、軸を必ず対応付ける。
既存の`9d95d3ea:run01/COVERAGE.json`にある全26 E[S]は元identity・数値区間・kernel_source付きでcopyできる。
それ以外のfeatureを既存K1が既に保存したとは扱わない。

以下の数式は宣言したideal finite-K lawに対するもの。π_k=P(order=k)、p_cはcomponentの
正規化確率、b(c)はbasis key `(basis_id,basis_hash)`、p_b=Σ_(c:b(c)=b)p_cとする。
数学的component選択は各applicationでIID、order kに条件付けても同じp_cであることが条件。
実PCG64固定seed列のIID・同lawは別問題で、CDF/finite-bit/law差をこれらの式で0としない。

## 2. 最小優先セット：閉じた式か既存状態で扱える量

まずF0とF1〜F4だけを候補にする。F5以降は検証上必要になった場合に追加する。
係数を全てfitするという意味ではなく、window補正のdomain/不一致を説明する記述量である。

| ID | event列からの定義・順序/境界 | 期待値に必要な情報・Cでの取得方法 |
|---|---|---|
| F0 `S_FULL` | 元AccountingのH→q回のhead/tail/reverse→scalar→X/Y suffix→measure→finish一回 | 全26 E[S]を再利用。途中finishやresetを追加しない |
| F1 `ROT[k,support,sign]` | 各event最後のrotationを1回計数。supportは元順序付きtuple、Z/ZZを分け、k=0/2・符号を保持 | qR π_k Σp_c（該当component集合）。p_c・π_kがあれば新state不要。P1のevent_metricsは既存path計数の再利用元 |
| F2 `PRODUCT[support,sign]` / `ORDER2` | products→rotationのapplication順で各productとorder2 eventを計数。P1の既存計数を流用 | E[ORDER2]=qRπ_2、E[PRODUCT_c]=2qRπ_2 p_c。negative productも同様。未観測でも期待寄与0としない |
| F3 `TAIL_BASIS_PAIR[b,b']` | **同じoccurrence内**の隣接nonidentity applicationのbasis pair。event境界を跨ぐがouterのheadを跨がない。同basisも数え、switchはb≠b'に限定 | 現H4は非identity324 component。E[count_(b,b')]=q{R(1+2π_2)−1}p_b p_b'。p_bとπ_2で計算。これはcompiler相殺数ではない |
| F4 `TAIL_ENTRY[b]`, `TAIL_EXIT[b]` | 各occurrenceの最初/最後のnonidentity applicationのbasis。独立のsynthetic head IDへ置換せず、元head operationsとの接口を別recordにする | 現条件で各期待count=q p_b。最後はrotation。role/order付き入口・出口には下記grammarかModel.switch/closingを使う |
| F5 `S_WIRE` / `LITERAL_ZFOLD` | wire別のnonzero pending-Z flush。native展開RZとfold後RZの差は、prefix/terminalを含めた全wordで定義 | 既存Wire.step_word/applicationの状態とlocal integer reward、prefix/terminal/suffixを再利用できる候補。新reward/parity・数値enclosureが必要。単に9個の独立SEへしない |
| F6 `LITERAL_BASIS_CANCEL` | 元Accounting.basisのoperations比較で遅延inverse pairが相殺された回数。basis key一致と同一視しない | 既存source_tail_functions、Model.switch/prefix/closingのSOURCE由来local traceでrewardを作る。元データ型・tuple/list・matrix metadataを保存してparityを確認する |
| F7 `PHASE_SEEN` | 一つのoccurrenceでorder2による非零phase加算があったindicator。outerごとに数え、scalar/最後のsuffixと混ぜない | ideal lawではq(1−π_0^R)。既存control flagの射影で扱える。**compilerに渡るphase角度・位相parityの代替ではない** |

F1/F2はroleごとにcomponent単位で定義できるが、出力は必要な集約だけにする。
全81 basis pair係数や全324 component係数をfitする提案ではない。
global countは1 channelで評価し、同じbasis/order countを9 wireぶん加算しない。
wire別RZ rewardは各wireへの帰属で足せるので、平均の加算にwire独立性は不要である。

### F3をrole別にする場合

order0 grammarは`rotation`、order2は`product, product, rotation`。1 occurrenceのR eventでは

| 隣接role | 期待pair数（basis指定前） |
|---|---|
| product→product | Rπ_2 |
| product→rotation | Rπ_2 |
| rotation→product（event境界） | (R−1)π_2 |
| rotation→rotation（event境界） | (R−1)π_0 |

同basis pair確率p_b p_b'を掛け、q occurrenceで足せる。これはorder/component独立を仮定する式。
rotation角に依存するrewardではorder kを落とさない。
identity applicationが存在する別scopeではactive basisが閉じるため、上式をそのまま使わず
identity/reset transitionを含むsource automatonへ戻る。現在の非identity前提を黙って拡張しない。

## 3. 境界の取り扱い

Sのscopeを保つsource wordは、1st-prefix、steady-prefix、tailのapplication、close＋reverse head、
outer接続、最後のscalar/X/Y/measure/terminalである。

- **head→tail**：Model.switch[-1,b]の元wordを使用。sourceのentry pending-Zとheadの遅延basisを保持する。
- **tail内部**：同basisの再利用はeventを跨ぐ。productsの適用順を元のまま保つ。
- **tail→head**：Model.closing[b,flag]を使い、相対ancilla phaseを閉じる処理も含める。
- **outer→outer**：compilerのresetとは呼ばない。Sだけは既存の固定出口証明に従い、first/steadyを分けられる。
- **終端**：suffixはtarget Tのscalar一回とX/Yを含む。閉じたSのterminal rewardを最後に一度加える。
  tailの最終pending-Zを各event・各outerで独断でflushしない。

phaseは実装でπの順次binary64加算を含む。数学的なmod 2πや奇偶だけへ置き換えると、
compilerの小角toleranceに関わる可能性がある。Sが保持するbool flagの十分性を
compiler側にも証明されたと解釈しない。

## 4. C既存kernelとの接続

SOURCE `9d95d3ea:h4_structural_kernel.py` のWire状態は
`s=(active_basis, pending_Z_hex, phase_flag)`。component j/role/orderと
Model.switch/app/closing、step_wordから遷移とrewardを作る。
この状態で決まる非負feature φ_f(s,j)は、**既存Pを保ったまま別reward**として集計できる候補である。
必要なのは新rewardのSOURCE parityであり、E[S]の再実装ではない。

新しいφが状態で決まらないなら、無理に現在のstateに押し込まない。
特にSU(2) run行列、未決のCX frontier、複数wireの同時状態、長いwindow全体のcompiler差は
現在のS stateだけでは一般に決まらない。これらは優先featureから外す。

有限state rewardには既存MeanKernel.from_events/then/repeat/expectedを再利用できる。
半群は(P,g)·(Q,h)=(PQ,g+Ph)。q/R反復は二分累乗を使い、qR列をmaterializeしない。
同じtransitionを共有する複数channelでも、それぞれのrewardと元boundaryを保持する。
first/steady/terminalを含め、既存の数値契約を適用できる部分と、新rewardに必要な部分を区別する。

**符号付き補正の注意**：CのBand.rounded等は非負の確率/rewardに対する下端clampを含む。
負のC−S rewardをそのまま渡すと不正になる。非負の正/負channelに分けて区間を差し引くか、
符号対応の別の数値契約を用意する。既存full_wrapper_meanの負値拒否も残差meanへ流用しない。
feature期待値×符号付き係数の合成では、係数符号に応じて区間端を選ぶ。

平均に必要な周辺rewardと、分散に必要なjoint momentsは別である。
wireやfeatureの共分散は不明ならnullのまま。X/Yも同じpathの応答として保持する。

## 5. Dへの納品単位（将来評価する場合）

各(m,axis,feature_id)について、定義version、元identity/law_binding/physical candidate、
全wrapperかtail descriptorかのscope、lowering/compiler profile、元component/basis辞書source、
meanと数値区間、reward SOURCE/parity、boundary条件を付ける。
既存26 E[S]のquantity・出典・区間はそのまま保持する。
新featureが未評価ならmean=null、state拡大が必要ならその理由を返す。

法則の確率、componentの順序付きsupport/符号/basis key、headの元operations表現、R/T/q、
order0/2の角度、prefix/closing/suffixは必要な入力情報である。新しい分子inputや再分解を求めない。
F1〜F4は解析式で十分ならkernelを増やさずに評価できる。F5〜F7も必要な集約だけに限定する。

**今回はC実行なし。** Dはfeature期待値だけからE[C]を認定しない。
window local係数・compiler influence tail・law差・rare/tail・cohort不確かさを別に保持し、
最終的な加算は同じfull-wrapper/lawのE[S]+E[D]について既存adapterを使用する。

次：**Codex C**は明示されたfeatureだけを別SOURCEで実装・評価する。現時点の追加作業は不要。
**Codex D**は最小科学案の許可待ちでSTOP。**ユーザー**が必要精度・scopeを判断する。
