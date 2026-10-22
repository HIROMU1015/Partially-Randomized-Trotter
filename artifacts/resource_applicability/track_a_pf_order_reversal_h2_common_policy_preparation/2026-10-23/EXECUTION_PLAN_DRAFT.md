# H2固定4候補：比較Cの最小検証計画（未採択・実行不可）

本書は実行前の設計である。有効grant、実行config、科学ledger、marker、予約は作成していない。使用済みgrant・旧STOPは変更しない。private資料は後日提供するとのユーザー回答に従い、取得まで待って科学実行を始めることも、代わりに新計算で補うこともない。

## 1. 固定対象と将来の出力

H2直鎖1.00 Å、STO-3G、tol-only1e-8の保存DF3、4qubit、α/β各1電子・sector4、保存Hamiltonian/state/snapshot、T=.8、ε_sig=.0005、P=0、symmetric_directional control、cos/sinのfull Hadamard wrapper、compiled RZを固定する。

| 対象 | 候補ID | prefix | q | δ | 固定する方式固有構造 |
|---|---|---:|---:|---|---|
| A S2 | B1_p3_q4_2nd | 3 | 4 | 1/5 | 全DF因子、既存Full PF |
| A Y4 | B1_p3_q2_4th | 3 | 2 | 2/5 | 全DF因子、正/負/正の三段S2 |
| B S2 | H2O32_c7ec590a32a859bc | 1 | 4 | 1/5 | K4、ABS_TIME_PER_OUTER32、保存signed scheduleと探索cohort |
| B Y4 | H2O32_8a2ecdd0294ffb74 | 1 | 2 | 2/5 | 同上、三段の全piece合計32 short steps/outer |

4候補のみ。新しいq探索、PF係数、分割、RTE法則、normalization定義、T/ε/δ窓は追加しない。RPE全round/energy推定や状態準備費用モデルを導入しない。

将来の出力schemaは各候補についてreference/bias/uと根拠、logB/margin、軸headroom/適格性/整数状態、N_cos/N_sin、C_cos/C_sinと来歴、G、PRの標本/cohortと既存不確実性、全gateの合否を含む。比較はA内1組・B内1組のS2/Y4、両者の方向関係Cの順に行う。新規ランキングの対象はこの固定4点だけ。

## 2. Gate 0：先に取得する既存資料（科学callなし）

追加資料の正確なサーバー/rootをユーザーが提供後、[missing_evidence.json](missing_evidence.json) の順でread-only照合する。別人のjobや広域projects/tmpを探索しない。

1. 新PRのPIPELINE/stage config、plan、sealed files、runtime/provider/import metadata。plan SHAは公開EXPLORATION `4dda5e68815f8359b3890dc7a9e301956a631d6cea500e8a90703224db719373`、end_audit SHAは `38ad4d594a3be9289bdf1697d9a37a6d677384899a3320666c75761a304f7723` に結合する。Python/package versionだけで代用しない。
2. `cfg.signal_path` の元signals.jsonとreferenceの保存receipt/npzの所在・hash。4候補の正確なreference/bias基準と診断入力を結合する。NPZを復元・科学演算する段階は別承認とし、まずmetadataを読む。
3. 対象PR taskは `H2E1__<candidate_id>__<sample>`。S2保存n33、Y4保存n34。計67taskのresultと、各taskの2軸guard receipt・cost・storage/fingerprint/compiled_SHAを既存end auditのaccepted result hashへ結ぶ。これは既存記録の照合範囲であり、新規trajectory/compileの件数ではない。pilotは混ぜない。
4. 旧2候補のpreparation・native controls・2軸measurement fingerprint・compiler/runtimeを原保存manifestへ結ぶ。`h2_panels_science.compile_one` の取得SOURCEはnative controlとcompile identity/countを検査するが、compiled-vector gateを含まない。追加の既存receipt/保存compiled回路・vectorがあるかを、提示された正確なlocatorから確認する。

原本SHA不一致、acceptance不在、stage/cohortの混入、旧失敗runの部分単価の混入、未知provider/phase補正、入力hash/軸/policyの不一致はSTOP。metadata不足を科学runの自動起動条件にしない。

## 3. Gate 1：ユーザーの政策採択と入力充足

案1（推奨）、案2、案3のいずれをどの強さの条件付き主張として採用するかをユーザーが決める。推奨案1は [ALLOWANCE_OPTIONS.md](ALLOWANCE_OPTIONS.md) の誤差成分・単位・観測範囲を共通schemaにし、A由来の経験的生成式を候補とする。

採択前に、reference基準、combined candidate/reference uncertainty、path signal/vector差、raw/corrected closureの単位、logB差、work count、中間norm観測範囲、floor/係数、欠測処理を固定する。Aのsaved allowanceをBへコピーしない。Bにない項目を0にしない。共通referenceの意味が一致しない場合、保存scalar/meanを共通基準へ結ぶ限定再評価の要否も承認対象とする。

biasは通常signed totalの軸絶対値。α_axis=.025、ε/√2、factor1、両軸h>0かつu/h≤.01、凍結binary64 shot演算順を維持する。NONCANCELLATIONは今回の最小実行に含めず、必要なら別の採択・範囲固定が要る。厳密reference/allowance証明を求める場合、この経験的最小案で代替しない。

## 4. 保存値で済む場合の最小処理（将来の別承認）

| 処理 | 対象・入力 | 新規処理の最小範囲 | 科学実行の要否 |
|---|---|---|---|
| u/margin生成 | 固定4候補、採択済み共通規則、原保存診断 | 4候補の数値評価。Aも同じschemaで再評価し、元保存値は保持 | **必要**。準備中は実行しない |
| 軸適格性とN | 4候補×ε=.0005×通常signed | `u_aware_shots` のpaired評価4件、8軸の出力 | **必要**。既存Nは新政策の結果として再利用不可 |
| compiled C | A2候補各2軸実測、B2候補paired sufficient statistics | provenanceを受入後、保存C/paired sampleをそのまま使用 | 追加build/compileは不要 |
| Gと方向 | 新Nと保存軸別C、Bはn33/n34の同じcohort | 4件のN_cos*C_cos+N_sin*C_sin、2組の固定点比較、Cの方向判定 | **必要**。新費用評価・順位判定として別承認 |

この分岐の新sampling/signal/reference生成/build/compile/simulationは0。uや必要Nの変更は、同じ回路・law・wrapper・compiler・cost単位が維持される限りCを変えない。必要shot数を実量子shotとして実行する提案でもない。

Bでは保存axis平均をshot重みで結び、cos/sinのpaired covarianceやexact sufficient statisticsを保持する。必要なら既存統計式で新Nに対応するGのSEを再重み付けする範囲も承認時に明記する。新しい標本平均・母分散推定、独立確認cohortのpooling、標本数増加は行わない。現保存では両軸のCが一致していても、将来runnerは軸を独立欄として扱う。

「4 paired shot calls」は必要処理の設計上の単位であり、現在の予約/上限消費ではない。全15条件や全候補集合の会計を再実行する必要はない。

## 5. 入力が足りない場合だけ検討する科学call

| 不足 | 先に試すこと | 保存資料で埋まらない場合の限定候補 | 実行前に別途固定するもの |
|---|---|---|---|
| Bの共通adaptive診断 | 元signals/forward記録/reference receipt/実action記録の取得 | **BのS2 q4/Y4 q2だけ**で、必要なcorrected/raw経路・独立経路のsignal/vector差、全stage norm、work count、logB/closureを記録する診断 | 同じ保存H/state、既存K4とsigned schedule、二経路のSOURCE、単位・比較箇所、必要call内訳、1回限りのgrant。MC samplingは不要 |
| common referenceの数値根拠 | 旧FP reference.errorとnew reference経路の原記録・hashを結合 | どうしても必要なら共通reference診断をH2 1件、固定4meanの共通基準への差を評価 | saved input、reference経路/精度、数値許容差、同じreferenceを全4点へ適用するルール。新SCF/DF/状態生成は不要 |
| old compiled correctness | 既存compiled circuit/vector/receiptを取得 | 保存回路があれば対象2候補・2軸の不足したaction検証のみ。SOURCE差を確認するための新compileは不要 | どの保存byteを何の基準で検証するか。simulationは別科学承認 |
| 必須compiled証拠が存在せず、要求する共通受入基準を満たせない | 受入強度を限定するかユーザー判断へ戻す | 同じ旧構造2候補×2軸のbuild/compile/controlを検討。既存Bが受入済みならB67taskの再compileは提案しない | 新SOURCE/runtime・phase policy・4軸のexact scopeとnative/compiled gate、新費用系列であること、fresh grant。現在は必要性未確定 |

上表は選択的な分岐であり、全てをまとめて起動する計画ではない。元SOURCEが保存していないwork/vector等のうち、選択した政策に本当に必要なものだけを設計する。Aの保存入力で十分な診断や既存Cを重複取得しない。新compileが必要になれば旧Cの置換は新系列として扱い、旧値との一致を前提にしない。

**未取得metadataがある現段階では追加診断/compileの正確なcall数を確定しない。** future SOURCEで必要なprimitive/reference/action呼出しを列挙できてから上限を固定し、ユーザーがその追加範囲を承認する。現在はCPU/worker割当や科学call枠を予約していない。

## 6. 数値条件・検証・停止の草案

数値gateは結果を見て緩めない。reuseだけならmetadata hash/JSON identity/整数shot/有理単価はexactに照合し、欠測・非有限値をPASS扱いしない。shot arithmeticは既存凍結kernelの演算順とLOG_DOMAIN_ONLY分岐を保つ。独立算術crosscheckを採用する場合は既存Eと同じDecimal丸め各演算の整数完全一致・有理演算一致を条件にする。libm共有であることを明示する。

新しい診断が必要なら、取得済みH2 SOURCEの基準を出発点に、reference/path/normalization/native/occupation/Hadamard比較 `<1e-9`、compiled action比較 `<1e-7`、phase alignment/state fittingなしを具体的callへ結ぶ。旧basis bridgeの `1e-12` は別用途として保持する。これらはcorrectness gateであり、uの被覆や厳密誤差保証を意味しない。診断値が適応的uへ入る定義と、その診断を実行してよいかを先に固定する。未知項目の許容差はnullで止め、既存契約を黙って改訂しない。

技術gateがPASSでもh≤0ならINELIGIBLE、u/h>.01や不足ならUNDETERMINED、整数安全域外ならLOG_DOMAIN_ONLY・G nullを保持する。この評価結果を失敗扱いしてu変更やretryを行わない。四候補のうち判定不能のものがあればCはUNDETERMINED。費用が同値ならtieとして残す。数値/入力/receiptの不一致は即STOPし、既存科学runのretryや許容差変更を行わない。

## 7. 将来の実行契約と明示承認

次の担当はユーザー：政策案と数値・欠測・reference条件を採択し、固定4点のu/N/G/方向評価を別途承認する。診断/新compileが必要と判明した場合は、その具体的範囲も追加承認する。単なるlocator提供は科学実行承認ではない。

その後Codexが新SOURCE/input SHA/runtime/許容差/出力域/one-shot scopeを固定する。今回の未commit文書を実行SOURCEとみなさない。fresh実行用namespaceと新grantを使い、過去runやgrant/STOPを再使用しない。実行入口で各callをnonrefundableに記録し、native exit/child回収とcompletion auditを保存する。fail時STOP・自動retryなし、policy変更なし、他campaignへの進行なし。call枠を消費してから仕様を決めることはしない。

資源は実行直前に空きCPU/RAM/保存域と他jobへの非干渉を観測して必要に応じて決める。恣意的なCPU/RAM/worker上限や共有環境の変更・他job停止を追加しない。原固定契約を変更する場合は今回の案から自動継承せずユーザー判断へ戻す。

## 8. 判定できることの境界

| 主張 | この最小案での扱い |
|---|---|
| 元保存4候補の元政策下の順位 | 前段監査から引用できる。再計算しない |
| 共通政策下の固定4候補の方向 | 全gateと採択・別承認実行を通過すれば、その政策・保存cohort条件付きで判断可能 |
| 各方式の全候補集合でのbest | 対象外。u変更でbest qが変わり得るが再探索しない |
| PR化が逆転を生む因果機構 | 対象外。q/分割/段数/normalization/compiler/標本が同時に異なる |
| PR母平均の優位・一般化 | 対象外。有限標本・選択効果・未観測eventを保持。独立確認Pをpoolしない |

科学的Cの正式採択はユーザー判断。技術PASSや保存値方向だけをGOにしない。終了後mandatory STOP、詳細科学レビュー・次campaignの自動開始なし。
