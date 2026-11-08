# N5-B3後：必要最小限の検証計画

2026-11-08 JST。[REPORT](REPORT.md)、[候補比較](POLICY_OPTIONS.md)。**すべて提案・未実行。以下の科学callは今回の静的許可に含まれない。** 旧STOPを解除せず、新しい独立領域で必要な段階だけを明示承認後に行う。承認証明書・grant管理基盤は作らない。

## 1. 共通固定条件

科学入力baseは **2886ac07c1ec706a1180d107e0111745e74aec1f**。本指示のN5-B3は、同commit内の`pr_reference_n5b2_numeric`という保存pathを指す。旧pathや報告名を変更しない。

H4 linear1Å／STO-3G／DF12、元H厳密ground供給・準備費除外、Partial S2 LD3 h=.02 finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit1.3.0/compiler A、B′4-CX、Reference F/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、D融合・basis共有・X/Yを保つ。

最初の数値検証はm17、q131072／r32／88 bits、N_B626／N_R630、alpha1/720、failure.05、phase cap.4、energy1.5936e-4 Haを背景条件とする。N・alpha・PF・lawを最適化しない。

## 2. 最初の推奨実行：V0-A/Bを一括した保存入力診断

### 固定入力

| 入力 | Path／用途 |
|---|---|
| 4中央positiveのdelta・G・旧Q+G | `artifacts/pr_reference_n5b2_numeric/2026-11-08/RESULTS.json`。再計算せずexact_fractionを読む |
| Index／符号の既存検算 | 同`CHECKS.json`／`INPUTS.json`／`STOP.json`。8 index評価を繰り返さない |
| M17物理role上界・joint予算 | `artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json#/condition` |
| Interval関数 | `track_c_compiler_a_18round.I`、`account_pr_reference_n4c.endpoints`をASTで再利用。旧mainを呼ばない |
| Runtime | 既存mpmath1.3.0、192-bit interval。Fractionは分子／分母を別々にIへ渡して割り、先にfloat化しない |

4 positiveは次だけ。

- B′：`0x1.7e4088e024c5fp-12`、`0x1.fdab627b8d37ap-14`。
- Reference：`0x1.7e4088e024c5fp-11`、`0x1.fdab627b8d37ap-13`。

Negativeは既存のcomplete-adjoint／符号対称性から同じ囲いを持つ。新しい負角word・index・行列評価を追加しない。

### V0-A：同じwordに残る誤差の囲い

[未実行wrapper](../../../scripts/diagnose_pr_reference_post_n5b3_bounds.py)を実行準備済み。今回行ったのはsyntaxとSOURCE参照の静的確認だけで、runtime・数値結果は未検証である。

1. 保存4 deltaに対し、共有sqrt(2)区間と**4回のsin区間評価**でD_F=2sqrt(2)sin(delta/4)を囲う。
2. 保存Gを使い、L_F=max(0,D_F_lower-G)、U_F=D_F_upper+Gを作る。一般2×2 norm関係によりL_2=L_F/sqrt(2)_upper、U_2=U_Fも保存する。SU(2)の追加仮定なし。
3. Frobenius／spectralを区別し、4角×X/Y×2 normの**16局所割当比較**。Interval端点を正確な有理数で判定し、下界>cap、上界<cap、未決を区別する。

必要な前提は、保存Gが同じselected basic productとgrid角へのstrict上界であること、exact W、固定written order、unitary basicであること。[POLICY_OPTIONS](POLICY_OPTIONS.md)の数理設計式を条件付きで数値評価するもので、正式な新保証の採択やRPE判定ではない。

### V0-B：中央だけのrole予算probe

各armについて、保存rotation物理上界L_rotと、保存2中央Q+Gのmaxを使い、**2つの中央slice上界** `E_central_up=L_rot*max(Q+G)` を求める。保存joint_synthesis_error_retaining_saved_N_inclusive_upperと比較し、未評価の他roleへ残る余地を記号的帰属付きで保存する。

他role、domain、emitter／shellの資格を0へ補完しない。余地が正でも全RPE PASSではなく、負ならこの粗い中央上界が予算を使い切るという結果だけである。実誤差の下界・安全なsupport相関を利用した別配分の不可能性ではない。Raw geometryやshot式は再実行しない。

### Call境界・担当・STOP

担当は**Codex**。必要なのはV0-A/Bの保存値再解析への明示承認。Wrapperと固定入力・式・判定・STOPは準備済みで、追加の科学条件選択は不要である。

新規合成、strict guard、word生成、provider／emitter／consumer、Operator、PF build/compile、sampling、RPE再計算、T価格集計は0。新しいtarget、precision候補、domain方式も加えない。実行入口は次で、**この文書の公開自体を実行承認としない**。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-c-h4-parallel-pilots-20261106/.venv/bin/python scripts/diagnose_pr_reference_post_n5b3_bounds.py
```

固定4角の囲い・16比較・2 probeと端点検算の完了、又はSOURCE／入力／runtime不一致でmandatory STOP。Intervalが閾値を跨いだら未決とし、自動的なprecision引上げ・新guard・b探索へ進まない。

## 3. V0後の分岐

| 証拠 | 判断が変わる点 | 次の最小候補 |
|---|---|---|
| 同wordの下界が局所capを超える | 単にguard包絡を鋭くするだけでその局所割当へ入れる案は除外できる | Joint／role配分を検討。又はP2–P5の別word構成。RPE失敗とはしない |
| 上下がcapを跨ぐ | 現wordの実誤差は未決 | 必要なら同じ4 wordの直接strict guardだけを別途承認 |
| 中央probeに余地あり | 一様cap超過だけで中央再合成を要求する根拠が弱まる | 残りroleの保証を合わせた配分試験。未評価誤差を0にしない |
| 中央probeが粗い予算を超える | 現在のsafe-max証明では不足 | Supportとkey多重度を鋭くするか、quantization／word改善かを選ぶ。実誤差no-goにしない |

**V0から後続を自動実行しない。** 複数候補が残れば併記し、GPT／ユーザーが保証・policyの科学的採否を判断する。

## 4. 後続の準備済み選択規則

以下は条件付き設計で、現時点の採択値ではない。V0とは別の数値・実装・取得範囲として承認する。

### V1：Precision／domainの有限な構成可能性表

対象は最初は同じ中央4角と保存43 guard。Binary64 theta=p/2^eのeをexactに読み、bの候補を36から`max(36,max e)`までの整数とする。これは有限入力から一意に決まる診断域で、全physical系gridではない。既存b36行は再利用し、未計算行だけを評価する。

各bで最近接index、delta、Q、選択指数Jを求め、保存basicで覆える部分のG_knownとmissing指数を分ける。Missing guard/Tを0と置かない。保存capを**診断用基準**とする表では `R=cap-Q-G_known` を保存する。R<=0はこの既存word選択＋非負missing誤差のadditive証明では足りないという意味、R>0はmissing guardへの必要予算があるという条件付き結果である。bの正式採択・新基本word生成・guard・価格は行わない。

DomainはD1（既存j6反復）とD2（指数域拡張）の記号式を、B′固定scalar literal／保存event r32 superset／Reference保存包絡L17へ対応させる有限パラメータ診断を候補とする。新しいvacuum到達集合やsamplingは生成しない。J_b・反復上界・保証に必要なguardと価格を列挙し、未取得価格を未確定とする。D3周期表現は相対角差の新保証が必要なので別の独立候補に残す。

新numerical wrapperは既存round_indexの整数部分をparameter化する最小差分が必要。現固定providerは不変更。Finite表・欠損一覧・条件式の完了でSTOPし、見つかったb／j／precisionで合成を自動開始しない。

### V2：保存recordと離散Tによる配分

V0/V1で固定された有限key・word構成・誤差割当候補だけを対象に、C0299、43 basic、85 certificate、必要な旧strict inventoryから再利用catalogを作る。旧missing一覧は再探索せず引用し、provenance不足recordは未資格のまま残す。

同じcatalog・候補自由度をB′／Referenceへ与え、actual guard・出現多重度・canonical期待費用・実離散Tを使ってstrong baselineを作る。旧C2のMILP／rational feasibility処理は再利用候補だが、登録2精度に特化したoffer生成はmultiple-choice型へ変更が必要。価格集計・最適化は新科学再解析であり承認対象。Solver下界・実行可能価格・gapを区別する。

Complete catalogがない場合は未資格としてSTOPする。精度を上げれば必ずTが単調に増える／減ると仮定せず、推定価格で補わない。Scopeを閉じる前に取得が必要なら、keyとguard要求を次のV3へ返す。

### V3：必要部分だけのguard／合成／runtime検証

同じwordのdirect guardで済むなら再合成しない。追加合成が必要なら、選択したdomain／word policyが要求する**有限key**と、採用済みrole budgetから各要求epsilonを先に固定する。10^e<=epsilonとなる整数eはexact power比較で決め、科学的に不要なprecision ladderを自動追加しない。

中央だけなら既知4 positiveが最大の初期取得候補である。新dyadic basicならV1で確定したmissing／不適格指数だけ。両armへ同じ取得機会を与え、negativeはpositive完成wordのactual adjoint。既存acquire／strict_guardを再利用し、失敗・guard不足・有限取得完了でSTOP。別epsilonや別keyを探して改善を得ようとしない。

新word、signed recoding、D1/D2/D3のいずれも、literal・W・順序・actual adjointとsame consumer pathを必要最小限に検証する。N5-B1の同一fixtureは引用し、変わった経路だけを追加する。後置control、SOURCE vacuum生成、PF schedule、fusionを無断で変えない。

## 5. RPEと全18round総Tへ進む条件

完成policy、全必要role／域、SOURCE→native C、G又はstrict emitter経路、最終測定sinkが揃った後に、まずm17四条件だけのjoint error→raw disk→radius／phase→保存N十分性を評価する。I／endpoints、CT-0 geometry、N4-Cの両軸接続を再利用し、旧mainの価格・compile処理を起動しない。

その後、同じ固定18round scheduleの全行へ適用資格を確認し、各行の保存N・bits・PF控除を保持して総Tを比較する。M17だけの改善から全roundを外挿しない。Alpha／shot最適化を追加しない。B′とReferenceのstrong baselineがともに実行可能で、欠損価格がないことが必要。

継続条件は、同じ成功保証の下でReferenceの**総T**を減らせる証拠、又は判定に不可欠な未解決事項が有限scopeへ縮んだこと。局所精度向上のみ、N固定の不成立、未取得費用の0埋め、LP下界をbaseline扱いする結果では優越を認定しない。費用が悪化した場合も負結果を保存してSTOPする。

**現在は静的調査のmandatory STOP。次担当はGPT／ユーザー：最初のV0-A/B数値診断を承認するか、候補を保留するか判断する。後続policy採択・guard・合成・RPE/T評価はそれぞれ固定scopeの明示承認まで開始しない。**
