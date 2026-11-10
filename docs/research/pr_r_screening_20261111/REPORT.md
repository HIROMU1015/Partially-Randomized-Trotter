# RPEラウンド別R候補の事前スクリーニング（2026-11-11 JST）

**固定δの下でR∝qPFを中心とする有限候補を生成し、normalizationだけで詳細N計算の優先度を付ける入口を追加した。** Bの候補生成・law/context・coverageをそのまま再利用し、総費用評価器や候補台帳は再実装していない。既存A branch/worktree `codex/pr-r-dependent-n-model-20261111` を継続利用し、B/C worktree・環境・プロセスは変更していない。

今回の証拠は人工/mock12テストと構造metadataだけである。**実H4/H6のlog B、N、保存数値の科学的再解析、R最適化、C取得/予測、実Gは0。** 実系のnormalizationだけの新R評価も今回は開始していない。公開・remote照合後mandatory STOP。

## 再利用SOURCEと接続

- A `5ca2dce3234f3e4a971f4a8c852da676f050e315`：`pr_r_dependent_n.saved_task/with_round_R`、固定H/state/configuration、canonical ID、R-aware allowance、phase/radius/Hoeffding・整数N検査への接続を保持。
- B `7bca8f9642d1c42ca3fede4d31b2cdd8dcef46ae`：`rpe_r_selection.build_menu/request_context/coverage`を再利用。A branchにBのexact費用会計の依存変更を持ち込まず、固定Git blobから**変更していないmetadata関数本体と必要定数だけ**を読んで使う。費用 evaluator は公開せず、呼ばない。SOURCE hashを記録する。
- 既存 `h46_m5_plan` / frozen allocation、`finite_rte_distribution`を再利用。H4 S2/δ=.1/prefix1/K2、H6 m5_best/δ=1.6/prefix7/K2の条件を変更しない。H6 m0..3は保存条件を残し、R変更はm≥4。
- 新規SOURCE `e9d0bb4d266b365da73185b2449e135355d44aec`：[module](../../../src/trottertracks/pr_r_screening.py)、[metadata CLI](../../../scripts/plan_pr_r_screening.py)、[tests](../../../tests/check_pr_r_screening.py)。

h=.1/13round、精度0.00159360010199040 Ha、各方式αtotal=.05/各軸.05/26、βtotal/stat=.4/.2、prior/decoder、保存DF10/DF19・completion・sector・H/state/snapshot、compiler/control/full measurement wrapper/P=0は既存taskからコピーする。PFも保存B0/Full19構成を保持する。

## 1. Rとnormalizationの関係

J個のsigned pieceをw_j、q=qPF、δ=T/q、保存tail normをlambda_R（R_outerとは別の量）、pieceのevent配分をr_jとする。Σr_j=R_outer、全event数はqPF R_outer。

τ_j=lambda_R δ w_j/r_j として、K2の既存finite式は

- b₂(τ)=sqrt(1+τ²)+(τ²/2)sqrt(1+(τ/3)²)、
- log B=q Σ_j r_j log b₂(τ_j)、A=exp(−log B)、
- 小|τ|の主項 L₀=q lambda_R² δ² Σ_j w_j²/r_j。

H4はJ=1、r=RなのでL₀=lambda_R² Tδ/R。固定δではTに比例してRを増やすと主項を一定に保つ目安になる。H6は11 pieceへABS_TIME_PER_OUTERで整数配分し、各r_j≥1（R≥11）。signed timeの符号はfinite meanに残し、normalizerは偶関数を使う。Rのpiece別重複適用やq_RPE=2^mとの混同はしない。

小event時間幅の主項は**候補の構造的な中心を作る目安**に限る。実スコアは毎候補で既存finite式と整数allocationを使う。固定の「小τ閾値」は導入せず、max|τ_j|とfinite/主項を併記し、主項で採否を決めない。大τでもfinite lawのnormalization式を評価できることと、時間発展・位相精度がよいことは別である。整数allocationの段差と非可換積があるため、Rに対するB/N/Cの単調性や最適性を仮定しない。

N側の既存保守geometryはrho_lower=A−raw_error、beta_det=asin(raw_error/A)、epsilon_axis=rho_lower sin(beta_stat)/sqrt(2)を通じてHoeffding十分Nへ接続する。誤差を無視した同geometryのnormalization由来のshot係数はexp(2 log B)なので、その**対数2 log B**を減衰診断として保存する。これはactual N、最小shot数、厳密な資源下界ではない。R変更はcorrected/raw signal・bias自体を変えるため、normalizationだけでは位相/radius/Nを資格できない。

## 2. 各ラウンドの有限候補を作る方法

`metadata_plan`は、保存された最初の対象roundをanchorとして

R_seed(m)=R_anchor × qPF(m)/qPF(anchor)

を作る。H4はanchor m0/R4/qPF1、H6はm4/R16/qPF1。callerが明示する有限の正有理倍率fに対し、`ceil(f R_seed)`を整数最小条件へ合わせ、保存baseline Rと任意の明示reference Rを加えて重複を除く。ID/allocation/event数はBの`build_menu`が作る。全ラウンド同じR集合ではなく、qPFを使い分ける。

今回の**未採択metadata例**は倍率 `[1/2,1,2]` と、後述の小さいN pilot用reference Rをm12へ追加したもの。倍率や最大生成値をRの科学的上限としない。`R_upper_bound=null, range_complete=false`を出し、最大候補にも上側探索が開いていることを残す。倍率・reference値を変えれば別の有限候補を生成できる。R4/8/16/32やR16/32/64だけで十分な探索範囲とは扱わない。

| system/round | 構造metadataの候補R（評価済みB/Nではない） |
|---|---|
| H4 m0 | 2,4,8 |
| H4 m4 | 4,32,64,128 |
| H4 m8 | 4,512,1024,2048 |
| H4 m12 | 4,8,8192,16384,32768 |
| H6 m4 | 11,16,32 |
| H6 m8 | 16,128,256,512 |
| H6 m12 | 16,32,2048,4096,8192 |

[H4 metadata](H4_METADATA_PROPOSAL.json)はPR51候補、[H6 metadata](H6_METADATA_PROPOSAL.json)は対象m4..12のPR35候補。これら86候補を全てfinite meanで計算する計画ではない。大Rではevent数qRも大きくなるため、まず安価なscalar normalizationとC側のdomain/実行可能性を照合し、詳細計算の優先度を決める。現段階は構造metadata生成であり、全実系log B/N/Cはnull、優先度も未評価である。

## 3. スクリーニングと科学的判定の区別

`screen_normalization`はSOURCE/binding/prefix付きの保存lambda_Rを将来入力する科学的数値入口。今回は人工lambdaだけでテストした。実系で呼ぶには別途許可が必要である。

- `shot_inflation_budget=None`：閾値なしの記録モード。2 log B等を記録し、減衰予算に対する分類・候補の後回しは行わない。
- callerが根拠/SOURCE付きの感度・資源予算F>1を明示した場合だけ、log Bとlog(F)/2を比べる。既定Fや既定R上限は設けていない。Fは科学的PASS条件ではない。
- 予算を超える非baselineは`DEFER_EXTREME_ATTENUATION_ONLY`とできるが、全menuから消さず、`detailed_N_required=True`を残す。baselineは保全し、raw/radius/位相やCを知らずにN不成立・費用劣位とは表示しない。
- 最大Rの先も未探索である。候補が全部強減衰でも、自動R増加・retryや「最適R」を返さない。新しい有限bandの判断へ渡す。

人工検査では、小τのb₂における1 ULP差がq r_j回で増幅することを確認した。既存finite normalizerを変更せず、隣接binary64値からのlog B感度を記録する。これはrigorous error boundではない。予算境界がこの感度内なら`NUMERIC_BOUNDARY_RETAIN`として候補を残す。bが1へ丸まる場合、τ underflow/overflow等は`NUMERICAL_NORMALIZATION_UNRESOLVED`とし、B=1で成功したとは扱わない。位相/radius/Nの科学statusは全て`NOT_EVALUATED`のまま。

**Bは最終比較の対象として元の全`menu`を保持する。** `priority_menu`は詳細計算のqueueで、global/全menu最適を宣言するための範囲縮小ではない。欠測候補を残したまま、Bのcoverageで不足/資格不足を表す。normalizationが大きくてもCが十分安ければ費用で勝つ可能性を排除しない。

## 4. A/B/Cとの接続と検証

[HANDOFF](HANDOFF.md)に実APIを示した。各候補にはSystem/m/T/qRPE/qPF/δ、formula/prefix/K/R、signed weights/allocation/event_count、normalization方法と前提・未評価値、詳細N要否、未検証事項、canonical ID、Bの完全`identity/law_binding`を保存する。

`A_request`は1つのB候補をAの1-round R taskへ変換し、identity完全一致を検査する。Aは同roundの複数Rを別の承認済みtaskで評価する。recordへのlaw bindingは実際のSOURCE/law証拠が一致した後だけ返す。旧Rのbias/N/Cを新IDへ付け替えない。Cへは同じrequestを渡し、R/qPF/full candidate/測定wrapper/compiler/法則domainを揃える。価格予測・費用和・R選択はB/C既存実装の担当である。

[TEST_LOG](TEST_LOG.txt)：**12 test method PASS**。metadata時にnumpy/scipy/qiskitロード・tail参照なし、q依存seedと有限候補/上限なし、H6早期round保全/11配分/負time、固定B関数再利用とA identity/law一致、人工K2 finite式の独立評価と丸め感度、予算なし/明示予算/境界保留/精度不足、scopeやlaw/recipe/q不一致の拒否、旧SOURCE保全を確認した。別AIによる独立科学レビューは行っていない。

## 5. 最小科学実行案（未承認・未実行）

[EXECUTION_PLAN](EXECUTION_PLAN.json)は段階を分けた有限案であり、次段階を自動起動しない。

1. **normalization-only pilot**：H4 m0/6/12とH6 m4/8/12の計24候補を、保存tail normのSOURCE/binding照合後、閾値なしで記録する。行列作用・signal/N/C/Gは0。保存Nや位相の再解析も0。丸め感度とevent数をCの未資格domainと並べて検討する。ここでRを採択しない。
2. **最小N/C接続pilot案**：m12のH4 R4→8、H6 R16→32を新PR各1候補とし、新finite mean2/内部対照2。同一`R_AWARE_EMPIRICAL_ACTION_COUNT_V1`で新R scalar2と保存baseline scalar2、XY計8軸。保存2経路referenceが再利用資格を満たせば新reference0、不足ならSTOPして別途最大4 actionの範囲を判断する。旧N原本/PF/H/stateは保全する。これは候補例であり、採択でも全range探索でもない。
3. Cは同R/qPF/candidate/law/domainの価格資格・長q妥当性・共有不確かさを別途設計する。固定Rの校正を別Rへ流用せず、長qのfull compileを自動要求しない。価格実行範囲は本案に含めない。BはN/Cが揃った承認範囲だけでN×Cを比較し、N最小だけでRを採択しない。
4. 全roundへは、normalization-onlyの残りmetadata候補→優先度を固定した有限N/C候補→coverage/不確かさ/境界拡張の判断という順で拡張する。Rの科学的上限・予算F・全roundの最適Rは未決定である。

次の担当：ユーザーが上記の科学範囲と必要な予算/価格資格を判断し、許可後にAがcheap screen/N証拠、Cが価格証拠を供給、Bが既存費用pipelineへ接続する。既存SOURCE/input/raw/結果/grant/ledger/marker/KEEP/過去STOPを保全し、公開照合後mandatory STOP。
