# O2-S2：P5 developmentの価格・law・生成アクセスpreflight

2026-11-06 JST。静的調査の基点は [O2成果 2b71117ae0f6727cfc9b4968bb6fd96be7265a60][O2]。新科学計算、保存値の科学的再集計、sampling、回路生成・compile・合成、新sampler実装、重要科学レビューは行っていない。

**総合準備状況：`PARTIALLY_READY`。** 既存P5には、finite-bitの係数・proposal・weightと、各event内のordered wordを含むnative IR/価格を結ぶ保存bindingがある。小さい明示table上で既知ISを比較する設計は可能である。一方、価格依存lawの非列挙生成、複数blockの非加法価格を利用するnormalizer、元Hの厳密基底状態による全RPEへの接続は未確定である。科学的GO・新規性・研究採択の判定ではない。

## 1. 対象入力と直前草案との対応

| 項目 | 既存条件・今回の扱い |
|---|---|
| 入力識別 | `p=(1/5,3/10,1/2), x=5/7, m=5`。今回新たに選んだ数値ではない |
| target | `P5(-ix R), R=Σ p_i Q_i` のfull first operator moment。system channel、厳密exponential、分子energy推定とは区別 |
| provider | 3-system-qubit synthetic。`Q0=Z0; V1=R_XX01(π/4); V2=R_XX12(π/4)R_ZZ01(π/4); Qi=Vi†ZiVi`。右から作用。明示Pauli情報を利用できる既知I1 context |
| 分子条件 | 分子、geometry、AO basis、DF rank、split L_D、PFのδ窓は非該当。D1/Parentのstate `000` も元物理Hの厳密groundとみなさない |
| G9 science SOURCE | `0ef2b92738750a3c0d187743dd4b7c9b927db802`。結果carrier `78cb65be373b0808641b5a5e44d9b69913bf84e9`。contract、結果、SOURCEの役割を分ける。[G9結果][G9]、[契約][G9CONTRACT] |
| G10 science SOURCE | `b9ed01455351628c9073748f5ba5751aa794b789`。結果carrier `fcd3ea6217bc00b667180cec149a70102d75f07e`。m5の6行はG9 binding/sequenceの再利用＋confidence再会計で、新しい独立価格標本ではない。[G10][G10] |
| 閲覧用保存資料 | 基点2b71117…のG10 primary-data access packageを使用。分割event binding、exact resource table、cache metadata、元結果identityの対応がある。[アクセスREADME][ACCESS]、[manifest][MANIFEST] |
| finite bit/native | 既存H=160、root K=256、η=ρ=10⁻¹²、strict RZ ε=10⁻⁶。pygridsynth 2.0.0/seed0、`up_to_phase=false`、既存interval dps100。今回は再設定・再評価しない |
| compiler | literal direct V/P controlled lowering、CZ→H CX H、隣接逆対の相殺。汎用compilerの最良回路・routing・Qiskit optimization-levelの価格ではない |
| 歴史的confidence | G9はaxis誤差1/200、22軸にα=49/22000、11資源行へ別失敗配分。G10は同axis誤差、34軸にα=49/34000、17資源行。両方familywise1/20。今回の新実行値に採用したわけではなく、元Nを混用しない |

直前のLOCAL `NEXT_STEP.md` を実際に読んだ。そこでは対象が未選択で、元H/state・law/価格・normalizer・confidenceの一組を要求していた。今回はユーザー指示がP5を暫定の**技術入力**に指定したため、その範囲を調べる。元Hの厳密groundという最終科学前提をP5/000へ置換したのではない。この適用範囲の限定を明示し、別入力への切替は行わない。

草案の「RPE平均価格だけではpath価格が得られない」「B²のBernstein単純移植はshot削減にならない」「旧C1/Parentの停止を解除しない」は維持する。補足点として、D2/D3には既にliteralなstep境界相殺kernelと完了結果がある。草案のD1加法会計だけを見て、全ての境界処理を未実装へ戻さない。[D系列の後続整理][D123]

## 2. 保存identityから価格へ辿る経路

### 2.1 既存データの所在

G10 access packageの元結果commitはfcd3ea…、SOURCEはb9ed014…である。m5は次の保存行へ対応する。件数は既存README/indexの引用で、新しい集計ではない。

| 保存row | 方式 | 保存event数 | 取得方法 |
|---:|---|---:|---|
| 0 | ordinary | 273 | `events/row_00_m5_ordinary_index.json` と2 part |
| 1 | partial_return_tail | 264 | 同名indexとpart |
| 2 | closed_P3_tail | 258 | 同名indexとpart |
| 3 | full_return | 63 | [index][FULLINDEX] とpart |
| 4 | closed_P5_full | 63 | [index][P5INDEX]、[全63 bindingのpart][P5PART] |
| 5 | matched_CTS | 24 | 同名indexとpart。Pauli固有fieldsを保持 |

各indexの `source_pointer=/rows/i/events`、`event_start/event_end_exclusive` を用い、元event jはpartの `/bindings/(j-start)` にある。bindingは `event`、`native_ir`、`cost`、保存phase/reference診断を含む。全行の非event fieldsは [resource_table_exact.json][TABLE] の `/rows/i/saved_fields` で取得できる。

このcheckoutではindexに書かれた個別 `rows/*.json` が揃わないため、存在を確認したexact tableの `saved_fields` を用いた。G9 v2の大きいrawもcheckoutの存在だけで判断せず、固定78cb65…のGit objectとして存在することを確認した。欠けた閲覧用pathを科学情報の欠落や再計算の理由にしない。raw全展開・再構成はしていない。

public GitHubからP5 indexと全63 bindingのpartを取得し、blob IDが基点のGit objectと一致することを確認した。残りの指定m5 index/part、table/cacheはローカル固定保存資料の構造を読んだ。全240公開ファイルの再downloadや旧保存監査の再実行は行っていない。

### 2.2 path identityの粒度

identityは少なくとも、固定結果commit/原result identity、row index、binding index、implementationと、eventの `raw_word/word/child/ratio/phase_i_power` を結ぶ。CTSでは `pauli/rotation_sign` を使う。`word` はreduced word、`raw_word` は元proposalの語なので、同じreduced wordだけをkeyにすると異なる係数・proposal atomを潰し得る。角度keyだけでもword・basis・phaseが違うevent価格を区別できない。

保存例として、row4/binding0は `word=[]、child=0、ratio=2189372/2921709、phase=0` で、保存native価格はT=136/CX=2/1Q=340。これは**一つのeventの価格identityの例**で、改善率・平均・方式間優位を新たに評価したものではない。`helper:1` というevent内fieldを実装選択と解釈せず、rowの `implementation=direct_primary` と実際の `native_ir` を参照する。[P5PART][P5PART]、[native SOURCE][NATIVE]

### 2.3 「whole-path」で既にある範囲

一回のP5 trialでも、ωをgroup選択・raw ordered word・child・phase・角度・必要な受理判定まで含むものとすれば、複数Qとchild回転からなる一つの回路pathになる。保存native IRは、そのevent内の順序とliteral逆対相殺後の価格を持つ。この範囲では単なる単一Q価格より細かい情報がある。

ただし、それを複数P5 block、deterministic PF、RPE interrogation全体のpath価格へ改名しない。以下の三層を分ける。

| 層 | 既存資産 | まだ得られていないもの |
|---|---|---|
| 既存1 P5 event内のword | 完全binding内のIR/価格。event内basis・CQ・child・相対位相を含む | 選んだ別optimizer/精度/metricに対する新価格。depthの保証価格はこのT/CX/1Q表から得られない |
| 複数event境界 | [boundary.py][BOUNDARY] の `inverse/simplify/cancellation_depth/native_price/aggregate_pairs`。D2/D3の登録済みr=2境界会計は完了済み | kernelは全pair IR/個別価格表を保存せず集約値を返す。新しいIS用pair価格・normalizerの具体化は別の保存値再解析であり今回未実施 |
| 分子full interrogation | 主RPEの期待価格・cost scope、既存builder、条件付き成功接続の入口 | 元H/state、D halves・全wrapper、全roundのlawと非加法価格、energy精度・branch/alias保証 |

D1/D2の `tau=5/7,r=2` はmicrostep x=5/14の別条件を含む。既存x=5/7のeventを二つ反復したtargetと無説明で同一視しない。path長・r・時刻を今回は新たに固定しない。

## 3. 元law、有限bit、weight、normalizer

記号を分ける。P₀(e)は元SOURCEが実装する**digital proposal**、ã_eは保存されたrational coefficient、w₀(e)=ã_e/P₀(e)。理想のirrational係数a_eや、その正規化lawとは区別する。位相・符号はevent unitary/controlled branchに含める。

| 方式・接口 | 元lawとsupport | normalizer/weightの取得と限界 |
|---|---|---|
| ordinary等 `CanonicalGenerator.event/sample` | group law×label/条件付きlaw。raw wordがreduceされてもraw identityを残す | order/groupのroot intervalとmidpoint係数。`B`はroot和のinterval。実qはdyadic drawの積、weightはã/q |
| Closed P5 `P5Closed.event/sample` | root/two/four group、completion DP、child law。P5専用の非列挙生成が既存 | `a0/child0/level2/completion` と群root和からB enclosureを得る。samplingは `reference_events` を呼ばない。setupの群構築O(L²)と、現linear-scan selectorの費用を区別 |
| General Full `FullReturnGenerator.sample/packet/event` | ordinary order/label envelope、raw-reduced判定、dyadic acceptance、child。rejectは量子実行前のzero | `ReturnKernel.digital_parent/rational_event_weight` がlocal係数/受理率/実proposalからweightを返す。productionの `B` はordinary envelope、`U`は上界。真の全Full係数normを無料で使わない |
| CTS | 明示I1のPauli collectionから得るfinite atomとdigital law | 通常のword/child欄を補作しない。P5 full meanとの対応は元certificateへ帰属 |
| 既知IS `digital_proposal/reweight_events` | 明示的に渡した全係数tableと価格tableからqを作る | cost×M2用の `ã/√C`、または既存regularized規則。root enclosureとdyadic roundingを使うが、table取得費は外部。非列挙の価格依存samplerではない |

root256、drawごとのH160という元精度を、path全体が160bit一回で生成される保証にしない。`dyadic_distribution` は正supportの最大剰余丸めと相対誤差条件、`dyadic_bernoulli` は下向き閾値を用いる。正確な実lawは各drawの確率の積であり、理想lawへの近さだけでweightを評価しない。[GEN][GEN]、[P5SOURCE][P5SOURCE]、[KERNEL][KERNEL]

SOURCEの `DeterministicBits` による保存traceは有限interface確認である。独立一様bitsを仮定した確率式の正しさと、hash-counter traceの独立性・sampling精度の科学的実証は別。今回はbit生成もRNG replayもしていない。

### 3.1 likelihood補正とmeanの単位

元のweighted estimatorから変える場合は

\[
X_b(e)=\frac{P_0(e)}{q(e)}w_0(e)Y_b(e)
       =\frac{\tilde a_e}{q(e)}Y_b(e),\qquad Y_b\in\{-1,1\}.
\]

単にP₀/qだけを掛けて元weightを落とすと、同じ補正finite meanを推定しない。全nonzero coefficientをsupportに含め、既存と同じevent回路を用いることが条件である。q自体は今回生成していない。[PARENT][PARENT]

Fullのpositive-event proposalはsubprobabilityで、zeroを含む全trial lawに戻す必要がある。保存 `reference_acceptance`、`reference_m2`、`reference_range` と、productionの上界/normalizerは別。保存 `coefficient_norm_rational` をどのsamplerも無料で取得できる普遍normalizerとしない。

明示table ISでzeroを直接生成しない構成へ変えるなら、ã/q、全trial費用、normalizer取得費を定義する。既存accepted lawからzeroを除いて同じweightのまま条件付き平均を取る操作ではない。

## 4. path価格とfull回路境界

| 確認項目 | 取得できた情報 | 制約・不足 |
|---|---|---|
| event価格 | bindingのT/CX/1Q/strict errorとIR、sequence cache | accepted event価格であって無条件full interrogation価格ではない |
| controlled実装 | directはbasis Vを無制御にして中央演算を制御、scalar phaseをancilla上へ保持。negative Rはactual adjoint | helper診断の2 workspaceとdirectの1 workspaceを混合しない |
| 基底・相殺 | CQ、childのbasis、CZ lowering、隣接inverseをSOURCEで追える | Pauli特有の追加簡約、任意commutation、routing、全回路optimizerは含まない |
| strict合成 | [metadata][CACHE] の `/synthesis_cache` にangle key、ε、sequence、hash、価格、strict error/pass。signed inverseも同sequenceのadjoint | G10全cacheは46key、m5はG9の19key再利用という既存記録。別条件のkeyへ転用しない。今回はguardや合成を再実行しない |
| cache取得 | G8 `AcquisitionCache/NativePipeline.step` がlocal event要求後のmiss取得・reuse・zero時無要求を実装 | 生成した後に価格が分かることは、価格依存lawから先に引けることを意味しない。cold/warm、miss、abort、setup、online費をbaselineにも同条件で与える |
| D部分 | P5には分子のdeterministic PF splitがない | D費を「保存値0」として分子へ移送しない |
| ancilla・測定wrapper | G9/G10は共通prep/readoutのaffine係数とouter1Qを別に保持 | 既存hは準備/読出しをまとめたparameter。元H state準備だけを除外したfull-wrapper価格へ、そのままh=0を代入しない。ancilla回転・X/Y readoutは費用境界に残る |
| depth/実時間 | 保存native countと限定stage時刻 | native1QはTを含む。T+1Qを足さず、depthや並列wall timeをcountから推定しない |

zeroを事前に判定できればそのtrialの量子event呼出しは不要でも、古典生成費は残る。実行済みprefixがあればその量子費用も残す。confidenceの標本数Kを全planned trial、実量子interrogationをaccepted数として分ける。期待価格は全trialの無条件価格でNと組にするか、受理率とaccepted条件付き価格を明示する。受理率の二重計上やaccepted-onlyの統計分母への置換をしない。[G8PIPE][G8PIPE]、[RTEACCOUNT][RTEACCOUNT]

## 5. 統計保証とRPEへ必要な情報

### 5.1 今ある情報から何が定義できるか

固定有限tableと、supportを保つ実qが与えられれば、既存 `event_statistics` が使う量は

\[
M_{2,b}=\sum_e\tilde a_e^2/q(e),\qquad
L_b=\max_e|\tilde a_e/q(e)|,
\qquad\operatorname{Var}(X_b)=M_{2,b}-\mu_b^2.
\]

これは独立trial、実±1測定、zeroを含めた既存の会計の再掲である。今回は和・max・varianceを計算していない。未知のμを無料の入力にせず、既存の保守的なvariance上界M2を使う方法がある。

既存G9/G10/ParentのBernstein十分式は、bias控除後の幅s_b>0に対して

\[
N_b\ge\left\lceil\log(2/\alpha_b)
\left(2M_{2,b}/s_b^2+4L_b/(3s_b)\right)\right\rceil.
\]

`g9_comparison.plan`、`g10_comparison.finish_plan`、Parent `ledger` は有限target/bias/軸数に固有のpolicyも含む。関数全体を別taskに無変更で呼べるとはせず、共通化できる式と固定契約の依存を分ける。[G9COMP][G9COMP]、[G10COMP][G10COMP]、[DESIGN][DESIGN]

### 5.2 B²の単純移植を改善機構にしない

同じ[-B,B]範囲、同じ補正target、同じs/αで、B²をvariance上界とする既存Bernstein式は、Hoeffdingの `2B² log(2/α)/s²` に正のrange項を加える。ceilを含め、単純置換だけで十分shot数は減らない。直前草案の静的確認を維持する。

ISで残る問いはshot数だけの減少ではなく、M2/rangeとpath価格の交換である。より鋭いvarianceによるN改善を別途主張するなら、正当な信号情報・outcome variance上界または適用可能な既知empirical保証、その情報取得費が必要になる。保存matrix診断、pathごとの期待値、古典価格標本のvariance、人工PASSを実Hadamard outcomeのvarianceへ置換しない。

### 5.3 bias、半径、全round

| 層 | 既存P5で確認できる情報 | 元HのPR＋RPEで追加するもの |
|---|---|---|
| finite mean | digital coefficient・phase・native平均誤差の保存certificate。再重み付けのみなら同じ実装mean/biasを保つ条件を特定可能 | 元Hとの差、PF/有限Taylor/表現/合成の系統誤差を共通energyへ戻す |
| 単軸confidence | q、ã、M2/range、bias allowance、α、残余sという必要入力を特定 | round/axisごとの実用可能情報と失敗配分 |
| signal radius | G9のNは未知signalで削減せず、半径を入力にしない固定有限mean task | 正当な基準半径、attenuation、位相への座標誤差変換。元H groundだけでunit radiusにしない |
| branch/energy | P5単独にはRPE round/alias/energy目標がない | 共通E₀、初期alias区間、decoder条件、全round union、最長時間、energy変換。既存条件付き18round接続を再利用候補とし、新lawへの保証移送は別確認 |

主 `rpe_resource_accounting.py` の `_hoeffding_shots/evaluate_rpe_round_candidate` はこの接続の既存入口だが、weighted/zero-fillを無修正で入れない。[RPE][RPE]

## 6. 既知ISと公平に比較できる範囲

`p/√c` は既知のcost×second-moment最適化で、有限confidenceのrange/ceilや生成費を含む最適lawの新発見ではない。[原著 §II Thm.1][IS]。G8 `oracle_IS_diagnostic` は列挙したG7係数と**RZ合成だけの価格**を用いた既済診断で、native provider込み・非列挙production・full-wrapper比較ではない。Parent/S4には別のnative table/IS比較も既にある。単にnative価格を入れ直すことだけを未検証の新方法としない。[G8IS][G8IS]、[Parent終了][CLOSURE]

| 対照 | 同じ機会を与える条件 |
|---|---|
| 元lawと同辞書の既知IS | 同じã/実装回路/bias、同じpath価格・精度、共通confidenceと情報取得費。有限tableの比較はそのtableに限定 |
| ordinary/partial/Closed P3＋tail/Closed P5/Full | 同じ有限P5 target。Closed P3＋tailを低精度P3対照にしない。表現ごとの係数差・digital biasを保持 |
| CTS | このsynthetic I1では明示Pauli情報を全方式へ開放。CTSのCX利点とzero-T eventを保持 |
| native最適化 | direct control、actual adjoint、phase、basis・境界の同じ合法な相殺。候補だけを強くcompileしない |
| 明示tableとlocal sampler | 明示tableには列挙/価格取得/setupを課金し、local側に全support/全価格を無料注入しない。warm cacheだけの都合のよい比較にしない |

保存CTSの一部eventはT=0である。既存 `digital_proposal(mode='cost_aware')` は正価格を要し、その場合のanalytic ISは不適用となる。zero-Tを削除したり架空の正価格floorを実価格として加えたりしない。既存regularized法を使うなら規則・parameter・実価格との違いを設計時に明示し、全対照に同じ機会を与える。主指標はまだ選んでいないので、今回qを決めない。

## 7. 技術的準備状況

| 評価対象 | 分類 | 理由 |
|---|---|---|
| 既存1 P5 trialの有限tableを使う既知IS/access診断の設計 | `READY_FOR_SCIENCE_DESIGN` | 係数・実proposal・weight・IR・価格・誤差・normalizer取得接口を固定保存資料から特定できる。次の目的・価格境界・主指標・confidenceは未採択で、実行可能な契約完成を意味しない |
| 非加法whole-path価格を使う非列挙S2全体 | `PARTIALLY_READY` | event内価格とboundary kernelはあるが、対象pathの範囲、joint law/価格依存normalizer、実sampler、生成費・共通wrapperが未接続 |
| 元Hの厳密groundによる分子PR＋RPE総費用をP5だけで直接判定 | `UNSUITABLE_INPUT` | P5はsynthetic有限operator。物理H/state、DF/PF、全round/energy条件を持たない。この用途へ証拠を拡張できない |
| 新候補の資源利益・独立した研究寄与 | `UNRESOLVED` | 今回新しいN/価格再会計・normalizer評価・性能比較を行っていない |

総合は `PARTIALLY_READY`。P5を捨てて別入力へ移る判断ではなく、限定した技術fixtureとしての適合と、主目的への不足を分けた判定である。

## 8. 次の最小科学検証案（実行しない）

P5には係数/価格取得の接続を限定確認する用途が残る。ただし、既知table ISの再計算だけならG8/Parentと重なるため、次の判断を変える問いを先に選ぶ。新samplerの正式設計や未決定数値の固定は行わない。

| 項目 | 設計案・未決定事項 |
|---|---|
| 問い | 一P5 trial内のordered word価格を使うtable access診断か、複数event境界を含むS2アクセスかをユーザー／GPTが区別する。後者のpath長/r/時刻は未決定 |
| 入力/SOURCE | p/x/mとsynthetic providerは本入力を候補として維持。ã/P₀/IR/cacheは固定G9/G10保存版を使用候補にする。元G9とG10のNを混ぜない |
| 比較 | 元law、同辞書の既知IS、適合する低次数Closed/CTS。新しい候補機構が未指定なら、既知法の適用確認を越える研究比較は開始しない |
| 価格取得 | 一trialなら既存bindingの価格を読み、必要なwrapper費を分離する。複数eventなら既存boundary kernel/保存sequenceを再利用候補にするが、新しいpath価格評価は別科学許可。単純event和を非加法価格と呼ばない |
| q/normalizer/weight | explicit-tableの場合は既存 `digital_proposal/reweight_events` と実qのã/qを利用候補にする。全support取得・root和・dyadic selectorのsetup/online費を明記。非列挙にする場合は価格依存normalizerと生成手順が不足していることを先に解決する |
| 統計 | 両側・両軸で共通のbias控除、実qのM2/rangeによる既存有限confidenceを適用する。B²式の単純置換をshot改善としない。新しい精度/α/主指標は未決定 |
| 会計 | 全planned trialのNと無条件量子価格、accepted回数、prep/readout、Dの有無を区別する。cold/setup/online/memoryは別座標で記録。実測shotsを行うかも未決定 |
| 最小範囲 | 必要な入力/価格/rangeが揃う一つの固定taskだけ。今回すでにあるSOURCE/test/pipelineを再利用し、旧runの再実行・旧grantの再使用・precision rescueをしない |
| STOP | 同情報の既知table lawへの還元だけ、range/weight/取得費で利益消失、normalizer/support取得が非現実的、同target/bias/価格境界を閉じられない場合はその案の拡張を止めて報告。技術PASSでも科学GOへ進まない |
| 分子への別段階 | 元H/state/sector、DF input、D halves/PF/δ、全interrogation price、round別law、半径/branch/energy保証、古典取得費が必要。P5の結果から外挿せず、別対象・別実行許可とする |

既存Parent同development探索の終了、C1凍結、G/D系列のone-shot/STOPは維持する。P5を使う新しい検証がそれらの停止範囲を変更するなら、その対象と変更をユーザーが明示する必要がある。今回は変更しない。

## 9. 再利用資産と今回の未実施

| 実読した既存資産 | 将来再利用できる部分／今回呼び出していない部分 |
|---|---|
| `return_aggregation.py` | `root_interval/dyadic_distribution/dyadic_bernoulli/dyadic_index/ReturnKernel.digital_parent/rational_event_weight`。生成・数値評価は未実行 |
| `g7_generator.py/g9_p5.py` | canonical/local Full/Closedのlawとpacket接口。`sample/reference_events/audit_p5` を起動していない |
| `g9_native.py/g9_comparison.py/g10_comparison.py` | 保存IRの意味、価格規則、元confidence、m5再利用鎖。native構築・matrix・再会計関数は未実行 |
| `g8_pipeline.py/g8_reference_audit.py` | miss/reuse/zero境界、既知列挙IS診断の条件。sampler/backend/oracle診断は未実行 |
| `parent_allocation.py/parent_experiment/design.py` | explicit table proposal、ã/q、実M2/range、bias控除とBernstein。新law/新N生成は未実行 |
| `rte_development/boundary.py/accounting.py` | literal境界相殺・既存全trial会計。D1の加法条件とD2/D3後続を区別。pair評価/集計は未実行 |
| test SOURCE | `test_g9_p5_and_native.py` のsame-mean、phase、reference非使用、zero-T対照。`test_parent_allocation.py:ProposalControls` の実q/zero価格/係数保存。`test_g8_on_demand_and_bounds.py` のzero時無取得・production/reference分離。静的読解のみで、test PASSを新発行していない |

今回のPython処理は標準ライブラリによる保存JSONのschema/field/pointer読取とGit object/文書の確認だけである。科学SOURCEのimport、係数・確率・moment・N・価格・比較の再計算、全supportの新規生成は行っていない。既存の保存結果・負結果・STOPを変更しない。2文書の公開と取得照合後にmandatory STOPする。

[O2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/docs/research/o2_sampling_return_technical_survey_20261106
[G9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[G9CONTRACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/contract_v2.json
[G10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md
[ACCESS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/artifacts/track_b_g10_v3_review_access/2026-10-10/README.md
[MANIFEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/artifacts/track_b_g10_v3_review_access/2026-10-10/manifest.json
[TABLE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/artifacts/track_b_g10_v3_review_access/2026-10-10/resource_table_exact.json
[FULLINDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/artifacts/track_b_g10_v3_review_access/2026-10-10/events/row_03_m5_full_return_index.json
[P5INDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/artifacts/track_b_g10_v3_review_access/2026-10-10/events/row_04_m5_closed_P5_full_index.json
[P5PART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/artifacts/track_b_g10_v3_review_access/2026-10-10/events/row_04_m5_closed_P5_full_part_000.json
[CACHE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/artifacts/track_b_g10_v3_review_access/2026-10-10/result_metadata.json
[GEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g7_generator.py
[P5SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
[NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_native.py
[G9COMP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_comparison.py
[G10COMP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g10_comparison.py
[KERNEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/return_aggregation.py
[G8PIPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/src/trottertracks/algorithm_codesign/g8_pipeline.py
[G8IS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/src/trottertracks/algorithm_codesign/g8_reference_audit.py
[PARENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ccc1832f63cd323b23fc05623f7e0b58cf34f37/src/trottertracks/algorithm_codesign/parent_allocation.py
[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ccc1832f63cd323b23fc05623f7e0b58cf34f37/src/trottertracks/algorithm_codesign/parent_experiment/design.py
[CLOSURE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md
[BOUNDARY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/src/trottertracks/algorithm_codesign/rte_development/boundary.py
[RTEACCOUNT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/src/trottertracks/algorithm_codesign/rte_development/accounting.py
[D123]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md
[RPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2b71117ae0f6727cfc9b4968bb6fd96be7265a60/src/trotterlib/rpe_resource_accounting.py
[IS]: https://arxiv.org/html/2603.13495v1#S2
