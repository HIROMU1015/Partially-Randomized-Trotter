# Cの採否案：決定論PFの未観測60軸

2026-11-15 JST。**Cは「60軸すべて条件付き採用を推奨」する。** 保存QPY監査と固定実装の局所性を根拠として、固定shot配分による6候補の全13 round比較に、元の凍結式 `C(q)=a+bq` から導く価格を使用する案である。**直接価格96/156と、未観測・導出対象60/156を区別する。今回、60価格の計算・採択・価格record登録はしていない。**

適用するのは、固定H4/tol-only DF10・Full10又はB0I p7、S2・標準Yoshida4・M5の固定実幅、m7/m8/m10/m11/m12、X/Yだけ。元preparation・signed piece・scalar・制御/wrapperとQiskit 1.3.0の固定compiler経路を保つこと、及び後述の閉包論証の前提を明示して比較することが条件となる。

**ユーザーに採択してほしい判断：この限定domainについて、実装上の閉包を前提とする導出価格を、固定shot配分の条件付き全round比較へ使用すること。** 無条件の価格保証や未観測点の2%保証を採択する案ではない。採択後の60価格算出は別指示とし、元の凍結JSONの資格flagは書き換えない。

この条件付き利用には、m12等の追加direct compileを必須条件として提案しない。実装前提を認めない保証が必要な場合の追加確認は§4で分けて示す。

[HANDOFF](HANDOFF.md)／[対象60軸と12モデルの機械可読対応](RECOMMENDATION.json)／[metadata照合](CHECKS.json)。後二者には未観測価格を含めず、`price=null`、`user_adopted=false`を保持した。

## 1. 観測点と使用対象

| 公式・固定実幅（Full/p7共通） | 保存compiled QPYの監査q | 未観測m7/m8 | 未観測m10/m11/m12 | 推奨範囲 |
|---|---|---|---|---|
| S2・16h/7（actual幅0.2285714285714286） | 7,14,28,224 | 56,112 | 448,896,1792 | Full/p7各X/Y、計20軸 |
| 標準Yoshida4・16h/3（0.5333333333333333） | 3,6,12,96 | 24,48 | 192,384,768 | 同20軸 |
| M5・16h（1.6） | 2,4,32 | 8,16 | 64,128,256 | 同20軸 |

h=0.1、m0〜m12の13 round、P=0。Linear H4/1.00 Å/STO-3G・保存H/state/snapshot・tol-only DF10。Fullはone-body＋fragment0..9、p7はone-body＋fragment0..6とB0I tail identity補正。元fragment順・basis/qubit順、負時間を含むsigned weightsを保つ。scalarはFullのc_H又はp7のc_H＋tail identityを総Tで一回だけ掛ける。symmetric-directional制御、X/Y基底変換・測定付きfull controlled Hadamard wrapperのcompiled全RZ数が対象である。

compilerはQiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、coupling mapなし、元のlayout/target/plugin条件。新lowering・hardware target・routing、別幅・別公式・別prefixへは移送しない。決定論PFなので母law平均のsampling誤差はないが、compiler閉包を使う際のsystematicは別問題である。

M5/m4のq=1は保存価格を使った較正点であり、今回のQPY監査点ではない。m7/m8も未観測であり、監査qの間にあることを独立の採用根拠にしない。m7/m8とm10〜m12には、同じ入力反復・局所処理・境界の論証を適用する。

## 2. どの証拠が何を支えるか

| 証拠 | 直接支える結論 | 支えない結論 |
|---|---|---|
| 元m4/m5の12個別fit | 各groupの凍結a,bと較正入力 | 未観測qのaffine性 |
| m6/m9の独立holdout（各12点で完全一致） | 凍結した式がその有限点を予測した事実 | m7/m8/m10〜m12の2%保証・全qの証明 |
| Dの540 schedule check | 同じ候補内のm4〜m12のsigned piece引数がbinary64で一致 | compiled回路が単純連結になること |
| A1 | 公開receipt・fingerprint・compiler・T・token countとの束縛 | 未観測回路の取得 |
| A2 | 44監査点の全wire token列が `H_w+B_w×(q−1)+E_w`、parameter hexまで一致 | 任意qの同一分解・pass途中の不変性 |
| A3 | 観測した周期本体のRZ数が元b、端＋最後のrunがa+bと一致 | 別幅/公式からのpiece単価移送 |
| A4 | 監査点の終端形・scalar傾き・一定θ₂、保存入力wrapperとの整合 | 未観測終端runの直接観測 |
| A5と追加記録 | 監査後本体の各wireにCX、全system wireにCX–RZ–CXが残る | 全最適化反復・全qで障壁が消えないことの単独証明 |
| 未観測60軸のMARGIN_OK | Dが延長した終端角で、事前1e−6余裕を60/60満たす | 未観測full回路をcompileしたという事実・全compiler誤差の区間 |

上表の観測結果は[D最終監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/606881b042ed69f9a1e7f00311417f2b65855d7b/docs/research/d_pf_qpy_closure_audit_20261115/REPORT.md)、[観測前規則](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3338c7ef87a1da1f02690627555bbb221fbdea4/docs/research/d_pf_qpy_closure_audit_20261115/AUDIT_RULES.md)、[元実装調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd435d07e3f0490165ff49889be9c71f573bd5db/docs/research/d_pf_unobserved_round_closure_20261115/REPORT.md)を再利用した。CはQPY/private cacheを開かず、監査を再実行していない。

### 直接確認済みの範囲

compiled QPY44件＋m9 wrapper QPY12件＝56件をDが読み、12 groupすべてA1〜A5 PASS、共通c=1と記録した。周期RZ数はFull S2/Y4/M5で7791/23535/85653、p7で5513/16539/60643。各値は凍結bに一致する。切片は元12モデルで307、端と最後のrunはa+bに一致した。これは保存結果の引用であり、新しいRZ計数ではない。

最後のancilla runは監査点でrz–sx–rz。θ₂のhexはgroup内で一定、θ₁と−scalar·Tの関係は保存残差最大約1.07e−14、事前基準1e−9以内。未観測60軸では、Dがm9の保存角とscalar/Tから延長したθ₁の余裕を検査し、最小は約0.0224754 rad（Full M5 X m12）と報告した。1e−6はDの事前基準、1e−12は参照Qiskitの省略閾値であり、Cが新しい閾値を追加したものではない。

この余裕により、**同じ終端run/ZSX枝が維持される前提の下で**最後のRZ数が変わる懸念は小さい。D HANDOFFの「RZ省略は起きない」は、この終端延長の前提を含む判断として受け取り、実未観測回路の確認とは記載しない。

## 3. 未観測qへ延長する根拠をCとして評価する

### 入力の反復と端効果

固定実幅では各stepのsigned piece列と角度が同一で、qが増えても内側の境界contextが増えるだけとなる。元composeは各pieceをscalar=0で構築し、最後に`p(−scalar·T)`を一回付ける。one-body/basisは固定であり、総Tに依存して変わる入力は最後のancilla runである。これが成立するため、公式内部やPF step境界の相殺を無視した単純gate和ではなく、**実compiled周期本体B**のRZ数bを使用する根拠がある。

Fullとp7、S2/Y4/M5では内部piece価格が異なる。Fullでは角度により単純なpiece数比例からずれる保存事実もある。今回の推奨は12個別モデルをそれぞれ維持するため、この公式間/幅間の移送を必要としない。

### 局所passと障壁を結ぶ論証

元`append_directional_diagonal`はone-bodyの各wireへCX(anc,k)–RZ(k)–CX(anc,k)を配置する。1q runは2q gateで分断され、固定opt1のInverseCancellationは同じqargsの隣接逆元を除去する。translation後の1q再合成はCXを新設しない。固定の汎用basis指定・calibration誤差指定なし・basis内の経路では、1q再合成の採用はgate数の減少による。Dの調査と同じ[opt1 pass manager](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/preset_passmanagers/builtin_plugins.py#L535)、[Rust 1q最適化](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/euler_one_qubit_decomposer.rs#L1076)、[Rust inverse cancellation](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/inverse_cancellation.rs#L42)をCも静的に確認した。

**接続の要点は、各pieceの非identityな分離部分が全反復を通じて保たれること。** これが成り立てば局所書換えが隣接区間を越えて伝播せず、同じ区間に同じ書換えを施した結果がqによらず反復される。外側のH/Eと終端runだけを別に扱うことで、監査で見えた `H+B×(q−1)+E` を延長できる。

ただし、「CXを新設しない」「最終回路にCX–RZ–CXがある」だけでは、この不変条件は証明できない。外側の逆元除去が新たな隣接を作る可能性も考慮する必要がある。A5のgatingは各wireにCXがあること、方向制御motifは追加記録であり、pass各反復のseparator証明を検査したgateではない。Cはこの区別を維持する。**Dの入力構造＋局所passに基づく障壁持続の論証を、固定実装の条件付きモデル前提として受け入れることを推奨する。全q・全中間DAGの証明済み不変量へ昇格しない。**

opt1のDoWhileはdepth/sizeの固定点で停止する。区間が独立で、translation後の更新が厳密にsizeを減らす経路なら、一つでも局所変化がある間は全体sizeも変わり、qの違いだけで未収束区間を残す理由はない。これは区間独立性を条件とする推論であり、ループ回数が実測でqに依存しないという主張ではない。サイズに応じた別pass・上限・例外が意味を変えないことは残る前提である。

### Cが条件付き採用を推奨する理由

1. 係数を後から修正せず、12 groupすべてで、保存回路の同じbit-level周期が元のbと端費用を説明した。単なる観測2点の直線よりも、qに対する機構の根拠がある。
2. 未観測qも同じsigned piece列・境界を持ち、compiler調査は遠隔の相殺・routing・2q再合成を含まない経路を示している。終端の唯一のT依存効果も、全対象で十分な保存余裕を持つ。
3. 12 groupに不一致や異なる失敗条件がなく、m7/m8だけを「補間」として採用する、又は一部公式だけを採用する追加根拠はない。したがって同じ明示前提で60軸を扱うのが一貫している。
4. 残る不確定性は主として実装論証・binary/source対応で、追加fitや統計誤差ではない。固定shot配分の**条件付き研究比較**にこの前提を明示することは合理的である。一方、これを価格の無条件保証へ置き換えるのは不適切である。

## 4. 残る仮定と、必要に応じた最小確認

| 前提/未確認事項 | 既存証拠が支える範囲 | より強い主張が必要な場合の最小確認 |
|---|---|---|
| Installed Rust binaryと参照tagの意味上の対応 | Dはinstalled Python sourceの一致を確認し、Rustはtag1.3.0を調査。44実出力はその論証と整合 | まず使用wheel/extensionの既存配布metadataとbuild/tag来歴を照合。必要なら当該binaryに結び付いた実装範囲の確認。今回の採否案にbit再buildを必須追加しない |
| 全反復の障壁持続 | 入力の分離構造、局所pass、監査点の最終motif/周期が支持。中間passの全状態証明ではない | 固定gate setで外側相殺も含むcut不変条件をSOURCEで閉じる限定検討。対象の中間traceが必要なら別許可。新しい大規模監査は今回実施しない |
| q/規模で別経路に切り替わらない | 調査したpassにqを直接見る分岐は見当たらず、m9までの実出力が整合 | 参照経路のsize-dependent shortcut/iteration-limitの適用範囲を限定して確認。未調査の全libraryを検証済みとはしない |
| 終端の数値枝 | 監査点のA4、固定scalar/T、未観測60点の保存MARGIN_OK | 条件を変えた場合だけ元診断の適用を再確認。今回、margin再計算や閾値緩和は不要 |
| 未観測full回路 | 取得していない | 直接観測を別途要求するなら、必要group/軸/roundだけ有限取得。1 groupのm12は他公式・routeや全中間点を検証しない |

これらは条件付き採用の前提としてユーザーが選ぶ事項であり、現在の60軸を一部保留する差分理由はない。前提を認めない強い保証を求める場合は、共通に関係する範囲の利用を保留する。p7 S2 m12 X/Yの2取得は、その2軸での規模移送の反例探索には有用だが、Full・Y4・M5の不明点まで解消するものではない。Bの部分和/順位逆転条件が得られた場合、取得の優先順位付けにだけ利用できる。今回Bを待たずに採否案を出した。

### 終端直接取得を義務とする既存規定の確認

対象の[旧NEXT_VALIDATION_PLAN](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bcb67b2b2bb78776db1dd8d0625a3facb37cd298/docs/research/d_pf_allround_price_coverage_plan_20261115/NEXT_VALIDATION_PLAN.md)は、m12を「条件付き」とし、終端直接点の取得以外にcompiler閉包調査と採択domain/不確かさによる根拠も認めている。m6のCOMPARISON_POLICY（`91365dcf…:scripts/pf_price_stages.py`）とCのM9_POLICYはそれぞれm6/m9限定。Dの閉包計画§6.6・監査HANDOFFも、直接取得をCが条件にする場合の選択肢としている。

この固定資料群には「今回の全13 round比較には必ずm12を直接compileする」という採択済み条件を認めなかった。古い別rank/PR/RTEの最大長holdout規則をこの決定論PF族へ移植しない。反対に、元凍結JSONの`qualified_for_candidate=false`/`long_q_extrapolation_qualified=false`と旧STOPはそのまま保全する。Cの推奨を自動的な契約変更とは扱わない。

元C `c628e9e8…:docs/research/paper_g_price_routes_20261114/PRICE_VALIDATION_PLAN.md` §1・§6も、m6だけではm7〜m12を保証せず`UNQUALIFIED_MODEL`に残すと明記する。この規定を保全したまま、新たに得た閉包証拠に基づく**条件付き利用policyの採否**をユーザーへ提案する。旧規定を遡って「全域適格だった」と読み替えない。

## 5. 採択後に価格表・Bへ渡す条件

- 96軸は元の直接観測record・SOURCE・receiptを維持し、60軸は導出価格と表示する。直接価格の件数を156へ増やさない。m0〜m3の別actual幅を凍結式で置き換えない。
- 後続の各導出行にはconfiguration/physical ID、m、axis、q/actual幅、元12モデルの固定locator、閉包監査のgroup/対象行、今回の採択policyと仮定、compiler/native/control/scalar/wrapper SOURCEを付ける。今回の[RECOMMENDATION.json](RECOMMENDATION.json)が60行の対応を用意している。X/Yは別行を維持する。
- 既存分類`MODEL_PREDICTION`を使う案とし、`evidence_kind=FIXED_COMPILER_CLOSURE_DERIVATION_CONDITIONAL`等の出所を付ける。これは新しいprovider/schemaの実装ではない。凍結モデルのdomain.kindが元直接入力由来で`DIRECT_COMPILED_DETERMINISTIC`であっても、それを導出行の分類へコピーしない。
- C既存`paper_nominal_cost._tier`は、資格falseのmodelを`UNQUALIFIED_MODEL`とする。**今回の推奨だけで`QUALIFIED_MODEL`にしない。** ユーザーが限定採用した後も、元JSONとは別policyで条件付き使用範囲を明示し、B既存会計の許可されたmodel/scenario経路で受け入れる。現guardが要求する明示policyやsourceを省略しない。
- 固定shot配分を別途正しくbindingしたうえで、96直接＋60導出による全13 roundの**条件付きコスト比較**を許す案である。Nの十分性、RPE成功、母law、他方式の価格資格や最適性は認定しない。同じ仮定を採用した6候補内で比較できるが、無条件のwinner・厳密価格差の保証にはしない。
- 価格誤差の区間/上界は未確定。m6/m9の2%を60軸へ流用せず、残る共通実装systematicを0・独立誤差・95% CIと置かない。前提不成立なら対象group/roundをUNKNOWNへ戻し、再fit・clamp・他候補価格での補完をしない。
- 後でモデル前提に関する反例又はcompiler/input差が出た場合は、影響domainの比較を保留し、既存直接価格と旧結果を保全する。自動の再compileや再fitは行わない。

## 6. 調査・保全・公開範囲

指定C branchをcleanな`259a0b2d7357d740502d6837db2cf3c2a8b75c5e`から継続。関連C履歴/固定結果のpathを確認し、同じ60軸の採否案は確認範囲に見当たらなかった。Dの公開固定SOURCE・JSON/CSV・監査code・native/sourceと参照Qiskit sourceを静的に読んだ。D branch/private QPYへ変更・再読込なし。未列挙の別host/archiveを網羅した重複不存在の主張ではない。

Cで新しく行ったのは、元alignment・元モデル・Dの60行と540-check保存結果をmetadataで対応させること。60軸・6 configuration・12 model_groupが一意に結合でき、保存A1〜A5/terminal labelと元係数locatorの整合を確認した。[CHECKS.json](CHECKS.json)。新しい科学的価格計算・監査test・QPY監査は行わず、Dの事前規則と18人工testの記録を再利用した。今回のmetadata照合を独立したQPY/科学レビューとは呼ばない。

ロードマップ差分：**直接価格96/156は不変。未観測60/156について、固定compiler閉包を前提にした全軸条件付き採用をCが推奨。ユーザー採択・60価格算出・固定shot配分の全round会計は未実施。** ロードマップ本体は統合担当へ委ねる。

**mandatory STOP：採否案のcommit/push・remote照合で終了。価格60値の算出、QPY再監査、build/compile/transpile、再fit、N/G/winnerへ進まない。**
