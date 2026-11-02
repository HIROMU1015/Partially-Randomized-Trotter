# Parent allocation S4：採択KKTの訂正・影響確認・補完実行準備

2026-10-15 JST。**READY_FOR_SEPARATE_ONE_SHOT_AUTHORIZATION**。今回の承認はsource修正・登録外検証・静的監査・実行準備のみ。
登録入力の訂正配分／角度／価格／N／資源表は未生成。新authorization-only childも未作成。mandatory STOP。

## 正本と訂正理由

[GPT科学レビュー原本](../../research/track_b_parent_allocation_v3_scientific_review_20261015.md)を変更せず保存した。
原本39,180 bytes、SHA256 `450bdc2e34662f21e8416fbd2d0c1a65590eb4a840627da40c5141af01fe0584`。
原本固定commit `fc9a627a60f53704597a59bd377512270687bf95`をpushし、GitHub再取得でbyte一致を確認した。
基点は[Evidence Packet](parent_allocation_v3_comparison_evidence_packet_20261015.md)の`17bfcaac5a53d648ee94de0067de24d719ed3911`。
旧science S3 `53a36adf9bdd84690e9cdf3e1f97f38a178441dc`と結果公開`cd055232f9612982a60953341b9cbf6168dd0bd8`は保持する。

採択したraw完全event T単価をCとする目的は

\[
S(z)=\sum_i\sqrt{C_i}\sqrt{z_i^2+b_i^2},\quad z_i\ge0,\quad \sum_i z_i=a.
\]

旧sourceはraw Cを受けながら分母`√(C_i²−λ²)`と探索端点`min C`を使った。
これは`Σ C_i √(z_i²+b_i²)`の別proxyで、価格の平方根を取る採択目的と一致しない。
旧保存名kkt／halfwayは実際に評価した別proxy配分の記録として残し、改名・再分類・上書きしない。

## 数値構成の訂正と根拠

[design.kkt_candidate](../../../src/trottertracks/algorithm_codesign/parent_experiment/design.py)はraw価格を維持し、
`t=λ²`で`0≤t<min C`を固定回数で二分する。

\[
z_i(t)=b_i\sqrt{t/(C_i-t)},\quad \sum_i z_i(t)=a,
\qquad \lambda\in(0,\sqrt{\min C}).
\]

正のa,b,Cで目的の二階微分は`√C_i b_i²/(z_i²+b_i²)^(3/2)>0`。
各z_i(t)の和は0からpoleで発散するまで単調増加するため、制約を満たす内部解は一意。
raw Cの単位とsqrt(C)重みを混同せず、等価なt式から区間を構成する。

既存の有理数`root_interval`を使い、固定root bitsの外向き区間からzの下限・上限を得る。
和の上限<aなら下端、下限>aなら上端を更新。曖昧なoverlapでは方向を選ばず前の認証bracketを保持する。
下限=上限=aというexact certificateがある場合だけ、そのtを一点bracketとして採用する。
pole以下を認証できない、lambdaの外向き上限がpoleへ達する、固定精度が不足する場合はtechnical failure。精度救済・retryなし。

ideal z区間、lambda²区間、lambda区間を別々に保存する。
interval midpointを正規化し、既定share bitsのlargest-remainderとindex順tieで有理化する。
非負性と`Σz=a`をexactに保ち、ideal区間の両端からL∞距離上界を保存する。
これは有理surrogate候補で、native価格の連続最適解を実装したとは呼ばない。
halfway、端点、共通角、重複除去、MATHEMATICALLY_NOT_APPLICABLE、非補充policyは保持する。
iterations128／root bits256／share bits64を変更していない。

## 独立した登録外検証

[新21 tests](../../../tests/tracks/algorithm_codesign/test_parent_experiment_kkt_s4.py)と従来93件の対象回帰を実行し、**114 PASS、FAIL/ERROR/SKIP0**。
従来45 experiment testsのうち1件は、旧目的に対応していた入力C=(1,3/4)を採択目的のC=(1,9/16)へ訂正した。
新しいexpectedをsolverの式から生成せず、独立なPythagorean解析例を使う。

- a=7、b=(4,3)、raw C=(1,9/16)、z*=(3,4)、hypotenuse=(5,5)、lambda*=3/5。
  `√C_i z_i/hypotenuse_i=3/5`をexactに確認。旧raw Cの停留条件は一致しない。
- 別の2-child例（5,12,13）／（8,15,17）と、3-child例（3,4,5）／（5,12,13）／（8,15,17）。
  ideal interval、lambda/t interval、丸め距離、exact simplex、独立Decimal勾配を確認。
- raw価格の一様9倍による配分の不変、等単価、一child、exact t hit、ゼロ／不適用、float拒否、固定精度不足、tie、有理化・determinism。
- 人工KKT出力を変更してもcommon／verticesが変わらないこと、重複を補充しないこと。
- 登録外p=(1/7,2/7,4/7)、x=1/3の人工4構成ワークフロー。
  source／matrix／synthesisはmockで、50人工旧rowのN／nativeを再評価せず20新lawへ接続する。
  候補・C*・inventory freezeが取得より先で、旧guard call0、新root単価評価12、root診断24を確認。
  semantic mismatch、row/map改ざん、partial acquisitionのfinal map未保存、scope／terminal不整合を確認。

[実行harnessと記録](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v4/focused_tests_execution.json)は固定Python3.10.12で約2.31秒wall／2.19秒CPU／peak69,716 KiB。
local登録外証拠で、CI／外部再現ではない。新実合成・science worker・登録operator再検査・登録候補／forecast／matrixは0。

## 既存結果への影響と独立性

[影響監査](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v4/impact_scope_audit.json)は20影響rowと50再利用rowをpath／bytes／SHAで列挙する。

| 旧config | 対象 | 保存law数 |
|---|---|---:|
| config_04 | ordinary kkt | 5 |
| config_05 | ordinary halfway | 5 |
| config_10 | closed P5 kkt | 5 |
| config_11 | closed P5 halfway | 5 |

common・verticesはKKT解を使わず直接生成する。
`candidates`、global `proposals`、`ledger`、`comparisons`、integer I/O helperのASTはS3と同一。
RootAllocationGenerator、native bridge、旧pipeline、worker、buffered writerと3入口scriptの8 sourceファイルはS3とbyte同一。
旧pipelineのC*は旧五common baselineからKKTより先に生成され、新候補集合に依存しない。
各lawは構成ごとに作られ、alphaはlaw数で割らず、cacheのkey／sequence価格もKKTラベルに依存しない。
S4は旧C*と50 control ledgerをコピーし、どちらも再計算しない。

この独立性は保存された各control値の再利用を支持するもので、訂正後内部候補を含む最良方式が不変という認定ではない。
旧20 lawを採択KKTの成功／失敗として解釈することは保留。
旧918監査と450資料整形をやり直さず、terminal、classification、旧Evidence Packetを保持する。

## 補完実行source・契約

[contract v4](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v4/execution_contract_v4.json)、
[supplement pipeline](../../../src/trottertracks/algorithm_codesign/parent_experiment/supplement.py)、
[zero-compute preflight](../../../scripts/tracks/algorithm_codesign/preflight_parent_allocation_experiment.py)を固定する。
public one-shot入口は既存`run_parent_allocation_one_shot.py --source S4`、ledgerは補完pipelineだけを呼ぶ。
S4のfull SHAはsource固定後の公開確認receiptと最終報告に記録する。公開確認HEADをscience S4と混同しない。

将来承認された実行の順序は以下。

1. S4のauthorization-only直接子A4・remote・clean・manifest・未使用v4 namespaceを検査し、exclusive新markerを消費する。
2. 固定v3結果・review・contract・cache・manifestをbytes／SHAでbindし、影響20を除いた50 rowの正本を取り込む。
3. ordinary/P5の保存common event、全event T単価、旧proxyとの一致だけを入力に訂正候補を生成。
   共通角・頂点・参照方式の新resource rowは作らない。C*は元freezeのexact値を再利用。
4. config_04／05／10／11の存在する訂正KKT／halfwayだけを選び、候補とinventoryを価格取得前freeze。
   重複／数学的不適用のslotは記録し、代替候補を作らない。全slot重複も正常完遂と区別し、性能negativeにしない。
5. 必要な旧cacheは同じtool／precision／sequence／strict certificateをmetadataでbindして再利用する。
   **旧operator guard再実行0、非root native再計算0、対照N再計算0**。
   新positive root keyだけ各一回取得。signed inverseはactual sequence adjoint、Wを保持する。
6. 新root ideal／realized controlled診断、訂正4構成の最大20 lawに同じ5 proposal・bias・十分N・native会計を適用。
7. 50旧control値＋最大20訂正値を新しい補完比較へ結ぶ。旧20はこの新比較からのみ除き、原結果は不変。
   inherited row内の相対pathは元result directoryと併記し、v4 outputと誤認させない。
8. complete evidence／内外receipt／terminal／STOPを保存。失敗prefixは比較に使わず、retry0・mandatory STOP。

対象は既知3-qubit distinct-basis synthetic、p=(1/5,3/10,1/2)、x=5/7、状態000、sigma+1、finite P5、r1。
分子geometry／basis／DF rank／L_Dは対象外。
K256/H160、rho/eta1e-12、primitive1e-6／strict phase100dps、axis精度1/200、alpha_axis1/40、complex failure1/20はv3と同一。
同じ古典情報だけでlawを選ぶ一complex taskで、alphaのlaw数配分はしない。
primaryは十分N×期待Tのh=0切片。T/CX/native1Qを重複合算せず、online費用はNOT_MEASURED。
新精度1e-7、旧D1/D2/D3再実行、r/provider探索、DF／分子／GPU、全回路compile、trajectory／LP、実量子shotsは対象外。

## 固定する実行上限と性能

| 項目 | S4上限／構成 |
|---|---:|
| 新構成／新law | 4／20 |
| 新positive root synthesis keys | 12、各一回 |
| 保存cache metadata keys | 36、旧operator guard call0 |
| 全必要primitive key上限 | 48 |
| 新root events／ideal+realized診断 | 12／24 |
| event bindings | 1,092（4×273の静的上限） |
| workers／numeric threads | 2／1 |
| per-key CPU／総CPU | 120秒／3,600秒 |
| worker／ledger／aggregate RSS | 512／2,048／3,072 MiB |
| address space | 4,096 MiB |
| output／per-key log | 64 MiB／64 KiB |
| review chunk／JSON write block | 512 KiB／16 KiB |
| cache／sequence | 4 MiB／32,768 characters |
| integer I/O digits | 65,536 |
| shot cap/axis | 1,000,000,000 |
| wall cap／per-key wall cap | なし／なし |
| runs／retries | 1／0 |

hostは32 CPU。固定2 workersとthread1でoversubscriptionを避ける。
v3 timingと静的sourceから、50 control、非root費用、旧primitive証明、C*を再利用し、16 KiB writerを保持する。
新登録benchmarkや速度改善率の測定はしていない。GPUは不要・未使用。
periodic CPU/RSS guard、hard CPU、outer process tree観測、partial・log・output保存、exclusive所有権は継承した。

## 固定・来歴・次の境界

新manifestは修正source、tests、contract、原レビューとreuse入力をhash固定する。
旧v1/v2/v3のsource準備・結果・authorization・marker・STOP、旧Evidence Packetを保護し、変更したS3 codeは元Git blobも照合する。
pending authorizationはscience_execution_authorized=false、source_commit=null。
preflightはbytes／metadata／runtimeのみで、登録ctor、KKT、N、matrix、synthesisを呼ばない。

修正は採択目的へ戻すもので、科学条件・baseline・precision・alphaを変更する追加判断は不要だった。
**次は固定S4の補完計算についてのユーザー明示承認待ち**。今回A4は作成しない。
登録sourceで初めて分かる訂正配分・角度・価格・N・順位はすべて未取得。利益の方向を予測しない。
補完結果取得後の研究継続・新規性・解釈は、開始承認後のGPT科学判断へ戻す。

**mandatory STOP。** 今回のsource準備から科学実行・旧run再開・次stage・GPTレビューを自動開始しない。
