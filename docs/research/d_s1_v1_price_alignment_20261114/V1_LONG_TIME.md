# V1：長時間compiled価格に必要な根拠と価格精度

研究対象は状態準備費を除く `G_j=Σ_i N_{j,i} μ_{j,i}`、`i=(round,axis)`、`μ=E[C]`。候補ごとのtask/riskが揃うことが先で、固定LD1の価格だけを高精度化する目的へ戻さない。今回μ、model係数、G、保存価格の数値変換を計算していない。

## 再利用できるもの・そのままでは足りないもの

| 方法 | 既存資産とコストの性質 | 長時間への必要条件 |
|---|---|---|
| direct full-wrapper | D P1/P2・Full PF receipt。通常の取得はq×piece×Rのnative/event列とcompile処理を要する | 完全なphysical/source/profile/axis一致。PR/Full-Hは一pathや有限cohortをlaw平均にしない。巨大会計を全候補へ実行する前提は置かない |
| 構造期待値E[S] | C9d95の固定LD1全26値・既存有限状態kernel。展開全event列を作らない部分を再利用できる | 同prep/law/observable/binary64/Accountingへの受入bridge。LD2/Full-H/別PF幅や別lawへ流用不可。SはCでも厳密界でもない |
| E[S]＋残差 | 同一trajectory/axisのfull-wrapper D=C−S、Cの単一残差結合 | idealと実samplerのlaw差、rare、有限cohort、numericとlong-q/R systematicを保持。元Sに含まれる相殺を残差以外の補正で二重計上しない |
| 加法モデル | C1914の供給係数/feature。PF固定＋step、PR head＋event＋境界、Full-H event＋境界等 | 制御/measurement/scalarを含むfull Cの仮説。係数・domainを固定し独立検証、使用域まで有効な誤差根拠が必要。event比例の合い方だけで正式モデルにしない |
| q-affine | C d7c09のFull PF同formula/actual幅のa+bq検証案 | signed weights/順序・scalar cap・wrapper・compiler/native固定。2点で識別できても全qのcompiler相殺を証明しない。early actual幅は別domain |
| outer-window | D952ceのconnected恒等式・短window案 | compilerの有限影響/decay/tail bound又は使用domainの適格参照。δ3が小さいだけでは未測定tailを0にできない。旧24 compile案は今回採択しない |
| 価格区間 | 根拠ある同時区間なら点平均の高精度化を省ける場合がある | model/外挿誤差を含む有効な上下端が必要。S・event数・旧lowering価格を根拠なく端点にしない |

target ideal law P、実sampler Qの場合、

`E_P[C] = E_P[S] + E_Q[D] + (E_P[D]−E_Q[D])`。

最後のlaw補正が未知のまま、生成SOURCE一致やnormal seedという理由で0としない。rareが有限cohortで見えなかったことも寄与0を意味しない。valid sampling CIにrareが含まれる設計なら同じ誤差を重ねて足さず、未解決成分を明示する。

## 使用域までの根拠

A shortlistの長qはm12でS2/.2 q2048、Y4/.4 q1024、M5/.8 q512。現在保存価格はC d7c09の5roundだけ。追加m5の成功からm6〜m12へ資格を移せない。PR LD1/LD2もm7以上の新ZZ平均価格は未資格、Full-Hの固定R448もq4096までの価格を取得していない。

Aの後続d48d873はS2/Y4のexact h相対幅とM5/16h等を未実行計画として提示した。これは元3候補の価格取得を全件開始する理由ではなく、B/Cが最終familyを絞るための追加情報である。requested幅の表示floatではなく、exact分数からのq/actual幅・signed time hexを受け取る。Aが指摘した32h/5と表示.64のceil差も保持し、旧幅のq-affineを新幅へ移さない。追加幅が採択された場合は、その同physical receiptがあれば再利用し、必要な最小cellだけ別scopeにする。

固定compiler SOURCEから有限記憶の閉包又は残差tailの有効な包絡を導けるなら、直接compileしない領域を評価できる可能性がある。しかし現在その証拠はない。PF内部の繰返し構造、局所Sのreset、同じqRのsignal縮約はQiskit full-wrapperの価格閉包の証明ではない。外挿でa+bqが正確、C/S比が一定、Dがevent数比例と仮定するなら、その主張は**未資格modelのscenario評価**として区別する。

独立の実行可能な代表域で検証しても、それはその域の誤差観測。最大使用域への未証明距離を有限のb_iとして扱う根拠がなければ、長域はUNKNOWNで残す。m7→m8→m9という順序も、m12の巨大直接compileも、義務にはしない。数理的な囲い/比較差に十分な根拠付き区間/必要域の参照のいずれも得られない場合は、正式Gを保留する。

## 必要精度の決め方（数値閾値は未採択）

Bに必要な入力は、同じenergy taskの候補family、適格なN又は許容N集合、price target law、識別したい比較差又は許容regret、価格側の同時confidence/主張水準である。Aの局所H_eff screenはN資格ではなく、Bの固定2 DP診断は全methodのshot最適化ではない。現時点で一律2%/95%やn4を設定しない。

固定非負Nについて、**有効な同時誤差上界** `|μhat_i−μ_i|≤b_i` があるときだけ、

`|Ghat−G| ≤ Σ_i N_i b_i`。

候補j,kの差には両側を加える。同じ価格変数を本当に共有する項だけは係数差 `|N_ji−N_ki|` で扱える。異なるprefix/law/scalar/loweringの同event数を共有価格変数としない。N自体が集合なら、許容された結合N集合上のsupremumを使い、都合のよい点Nで誤差を小さくしない。

同時に有効な区間 `[L_i,U_i]` なら `[Σ N_i L_i,Σ N_i U_i]` で比較する。必要に応じて片側の未取得を`[0,∞]`とするが、その0は価格推定値ではない。候補の有効な総下界が実行可能incumbentの総上界を超える場合だけ除外できる。正の重みのUNKNOWNが残るincumbentでは総上界は有限にならない。正式comparisonの実装はC/Bの既存受入仕様と合わせる。

追加検証は「比較係数×有効な区間幅」が結論を左右するセル/共有model componentを優先し、その不確かさを減らす実測費用も考慮する。必要precisionが既に満たされるセルを再compileしない。未知の系統誤差を標本SEで置き換えず、未資格領域へ適応的に追加し続けない。共有fitによるround/XY/formulaの誤差は独立と仮定せず、有限検証family/停止則を固定する。

## DがCへ渡す代表セルの位置付け

- **Full PF**：C d7c09の既存4新round/8 wrapper案と一致させる。最初の候補はA優先S2/.2 m5 q16。保存m3/m4校正の受入・fit・freezeはCの別科学scope。新値を先に見れば独立holdoutの扱いを失い得る。
- **PR**：C3ae2の条件付き新6 pairedセル（LD1 m3/m4、LD2 m0/m3/m6/m4）は既存案を参照し、n_i/精度未定のまま。今の目的だけで全取得を採択しない。旧P2を再compile/holdout化しない。
- **Full-H**：新ZZ m0/q1/R448/K2 normal1 pathの技術接続案。R固定q増大のmean model検証ではない。baselineをBが比較に残すか、必要平均precisionが何かを確認してから後続n/domainを設計する。
- **Truncated**：CのB0I p7 signalだけでは価格証拠がない。Bの同taskのrisk/採用状況を待ち、既存nativeのretained-H/独自scalarを保った最小cellを別提案する。Full PF価格で埋めない。

この段階の短い取得成功は正式G/winnerを埋めない。model採用、保存数値のfit/受入再解析、law/rareの数値評価、使用域へのE[C]/Gはすべて今回の実行範囲外である。
