# Track B G10 S2：RSS限定修正source・実行前最終レビュー

- 作成日：2026年10月10日（日本時間）
- 対象repository：`HIROMU1015/Partially-Randomized-Trotter`
- 対象branch：`track-b-g10-rss-repair-source-preparation-20261010`
- 固定source S2：`a139b91f119d109430ae3154a045d0fdcf722233`
- レビュー実施者：GPT
- レビュー状態：**完了**
- 判定：**確認範囲で実行前の必須source修正なし。S2を固定し、別途の明示one-shot実行承認へ進める。**
- 実行authorization：**本レビューでは与えていない。A2作成・登録用marker消費・本番実行は未実施。**

## 1. 結論と、その範囲

S2の変更は、主因候補となった全resultの再帰複製・一括JSON化を避け、不要な参照寿命を短くし、正常出力の監視と失敗時の記録を改善するものだった。旧科学計算の処理列、固定入力、比較手法、誤差・confidence・sampling・native費用会計を置き換える変更は、確認したsource・契約・保存監査に見つからなかった。[S1–S8]

したがって、**更なる原因特定や別のメモリ最適化を先行することを、次の実行の必須条件にはしない。** 確認した範囲では、S2を差し替えるべき具体的な停止要因はない。

ただし、これは「本番が512 MiB以内で必ず完了する」「任意の障害でSTOPが必ず保存される」「全return集約法が科学的に優れている」という判定ではない。旧G10の技術的失敗、S2の実行準備、今後の科学的採否は別の状態である。

重要な運用上の境界は次の三点。

1. 同じ論理payloadのJSON同値性と、新run全体のbyte同一性を混同しない。新しいcommit・時刻・資源計測・保護対象数は変わる。
2. `COMPLETED.v2`・成功STOP・result identity等のファイル条件は、外側processの正常終了・出力status・read-only監査を代替しない。
3. 本レビューは実行可否の事前判断であり、利用者の新しいone-shot実行指示を代替しない。

## 2. 背景とレビュー対象

### 2.1 引き継いだ状態

旧science source Sは `05c5ef23fce775a822ab5686f5da2f0d77675864`、旧authorization Aは `f5cd0755424d1b11e2249cc115518d84fb8bb8d3`、旧technical result Rは `e429c99d77b3222c5cca62750b2d111f87e4cb50`。原因調査commitは `21f66137ab205fb0d71ee6ee001844f4c2d725e3`。

旧runは `G10_TECHNICAL_INCONCLUSIVE`、RSS guard記録539.546875 MiB、固定上限512 MiBで停止した。保存17行を科学的勝敗の判定に利用しない契約は継続する。S2が将来成功しても、旧runを後から成功へ変更しない。[S1]

原因調査は、再帰的な全container複製と一括JSON化を最有力の増幅要因としたが、元runの正確な例外行を確定していない。S2の評価でも、この原因の確信度を上げて断定してはいない。

### 2.2 今回の問い

今回確認するのは、**既に採択されたG10の比較目的を保って、RSS限定修正sourceを次の正式な一回実行へ進められるか**である。

G10は同じ `p=(1/5,3/10,1/2)`、`x=5/7`、3-system-qubit synthetic providerで、各次数内の同じ有限演算子 `P_m(-ixR)` を比較する。m=3/7は新規11行、m=5はG9保存6行の再予算化。計17行・34 axesである。[S5]

今回、G9までの研究方針、新規性、全return集約法の優位性を改めて確定するレビューは行っていない。m間を同じexponential精度のランキングにすることもない。

## 3. 使用資料と確認の深さ

GitHub connectorで固定S2のI/O module、新旧runner、contract、launch、共通/per-key guard、静的監査source、validation summary、typed payload監査、tests、source manifestの関連部分、予算・row会計sourceを読んだ。branch APIのHEADはS2と一致していた。[S1–S16]

旧runnerは旧SとS2の双方でGit blob SHA `2e08b324bfcec8026871040b7b9b8fe96872ca59` が同じであることを確認した。I/O moduleはconnectorの全文からローカル写しを作り、Git blob SHAとSHA256を照合した。

| ローカル写し | 照合結果 |
|---|---|
| `g10_io.py` bytes | 14,864 |
| Git blob SHA | `847253d9ba92763251778580bb5fa38b163e9ca8` |
| SHA256 | `ed98d1bbc22a4c58fe8d479f1651373df1af756d4f5d82c73cdf5bff218552bc` |

**確認の限界：** 1,300 protected pathsをGPTが全件取り直して再hashしたわけではない。全科学module・全event certificate・本番runtimeを新規に再認証していない。AST一致は保存監査と検査source・新旧runnerの照合による確認であり、repositoryのAST監査scriptをGPTが実行した結果とは区別する。

元66.84 MB resultを本レビューで読み込んで採点・再比較していない。今回の判断には科学的なrow値の再解析は不要で、保存I/O検査のidentityと修正sourceを使った。

資料が取得不能だから判断を保留している状態ではない。S2のレビューに必要な主要資料は取得できている。Codexへの再push依頼は不要。

## 4. 科学的処理の保持

### 4.1 変更の中心

旧runnerのscientific bodyは新runnerの `collect(c, result, guard, io)` に移されている。主要な順序は次のとおりで保持される。[S3, S4]

- 設定とG9 anchor hashの確認。
- generator構築と、reference event表の列挙に先立つ予算作成。
- 固定bitstreamによるinterface診断。
- 各次数のCTS構成、reference event取得。
- 必要key集合の固定、G9保存keyの検証・再利用、新keyの固定精度合成。
- m5保存rowの再予算化、新しいnative row計算。
- 全17行の集合とevent上限の検査、固定辞書policy下界の追加。
- protected historyの確認。

`affine_policy_lower`、`plan`、`row`、`rebudget_anchor`等の呼出しを、簡便な推定や別方式へ置き換える変更ではない。保存AST監査も、局所`del`・`io.snapshot`・`pending.clear`を除いたbodyの一致を報告している。[S6, S7]

### 4.2 AST一致だけでは確認し切れない箇所

ASTの正規化だけでは、同じ関数名のbindingが違う場合や、削除の時点が不適切な場合を十分評価できない。そのため、以下を個別に照合した。

**最終protected checker。** 新runnerは `protected_check_streaming` を `protected_check` として使う。これは明示されたbinding変更であり、単なるAST一致で隠してよい変更ではない。固定ledgerのfull file hash／append-only prefix hashという規則を、32 KiBずつ読み出す方式へ変更している。旧共通helperは保持される。準備testsのfull/prefix/tamper比較とsourceを確認した。[S2, S7, S9]

**m5 anchor。** `rebudget_anchor`のrow全体と元budgetに対するdeepcopyは維持されている。変更は利用後の元G9 graphの保持終了である。旧budgetやnative descriptionを消して軽くしたわけではない。[S3, S16]

**eventのalias。** `row()`はbinding内の`event`、`native_ir`、cost等を結果側に保持する。したがって完成後に`pending`のlistをclearしても、必要なevent参照は完成rowに残る。生成途中でpendingを消す設計ではない。[S3, S16]

**関数frame。** 正常時にはcollectがreturnしてから出力へ進むため、計算用localの寿命を出力まで延長しない。異常時はtracebackがframeを保持し得るので、failure receipt用の短い位置情報を取り出した後に例外のtraceback/context/causeを切り離す。ただし異常時の完全なメモリ解放を保証するものではない。[S2, S3]

### 4.3 不変なのは処理意味論であり、計測時間ではない

新しいsnapshot、不要参照の解放、I/O照合には時間がかかる。`record_stage`の時刻差の計算式が同じでも、実測値が同じになるとは限らない。また新runnerでは科学依存のimportもguard内に入る。この点は上限を緩める方向ではないが、旧runと完全に同じwall/CPU計測範囲だとは表現しない。

科学的な数式・固定入力・比較の意味を維持することと、実行監査の時間値を同一にすることは別である。

## 5. JSON出力の同値性とメモリ改善

### 5.1 対応する型の範囲

新encoderの対象は、string-key dict、list/tuple、Fraction、JSON scalarである。Fractionを`str(Fraction)`にし、tupleをJSON arrayへ出力し、`indent=2`、`ensure_ascii=False`、`allow_nan=False`、末尾改行を維持する。[S2, S5]

この範囲の同じ論理payload vについての目標は、

`UTF8(json.dumps(serial_old(v), indent=2, ensure_ascii=False, allow_nan=False) + '\n')`

と、逐次encoderが返すbyte列の連結が一致することである。Fractionの桁数やnative IRを減らしてbytesを小さくする変更ではない。

**一般的な全Python objectに対する同値変換ではない。** 非文字列keyは旧serialでは`str(key)`へ変換できるが、新方式は拒否する。cycle・非finite・未対応typeも拒否する。実payloadのdegree index、cache key、cost座標、CTS certificate keyをstringへ明示変換しているというscope監査とsourceを根拠に、固定G10に限定して評価する。[S15]

### 5.2 除去した増幅経路

新方式は全resultの再帰container copy、一括chunk list/join、全JSONの追加UTF-8 copyを必要としない。既存treeの型検査と逐次`iterencode`を使い、書込byte数・SHA256を逐次集計する。1 writeは最大32,768 bytesである。[S2]

ただし**全アルゴリズムのメモリが32 KiBに抑えられるわけではない**。元result treeは保持する。encoderが一つの大きいescaped string tokenを構成する可能性もあり、任意の文字列長に対してtoken allocationまで一定サイズになる保証はない。この限界はsourceと報告書にも明示されている。[S1, S2]

### 5.3 保存されたI/O検証

| 検査対象 | 従来peak RSS | 新方式peak RSS | 同値性 |
|---|---:|---:|---|
| 保存G10 technical JSON | 510.4765625 MiB | 255.50 MiB | 66,842,493 bytes・SHA256一致 |
| 人工50,000行 Fraction/tuple/Unicode | 153.50 MiB | 42.734375 MiB | 8,371,227 bytes・SHA256一致 |

これらは保存validationの値であり、GPTが同じruntime・同じ大きいpayloadを今回再実行した値ではない。保存JSONのwallは約1.917秒から2.798秒へ増えており、validation・write・disk identity確認の追加負担もある。[S8]

**測定versionに関する限定も重要。** Validation summaryは、memory profile取得後にcompletion/failure safeguards、`verify_completed`、未呼出しprotected helperを変更したと記録する。一方で計測対象の`validate_tree`・`iter_json_bytes`・`write_result`本体は不変、最終47 testsは最新moduleを対象としたとされる。

したがって255.50 MiBは「S2の全正常・失敗経路を最終commitで丸ごと測った最大RSS」ではない。計測されたI/O経路の技術証拠として採用する。計測経路不変という記録を、GPTが計測時の全worktreeから再証明したわけでもない。[S8]

### 5.4 原因行の未確定は、今回の修正を無効にしない

元runの唯一の原因を証明しなければ、確認された増幅経路を除去してはならない、とはしない。保存I/Oでの増幅と同値出力の低メモリ化には具体的根拠がある。

しかし、改善量を元runの539.546875 MiBから単純減算して新runのpeakを予測することはできない。live Fraction/alias/native heap、import、allocator、計算中の保持量が異なるためである。**本番512 MiB内完了は未確認のまま。**

## 6. 正常出力の確定手順

### 6.1 同じcap値でも会計範囲の変更を明示する

RSS512 MiB、AS1536 MiB、wall1200秒、CPU900秒、per-key30/20秒、output134,217,728 bytes等の数値は維持されている。[S5, S6]

ただしoutput capは、payloadだけでなくmarker・telemetry・terminal receiptsを含むaggregateへ厳格化される。terminal用32 KiBを予約し、receipt個別16 KiB、telemetry32 KiBを上限とする。したがって「caps不変」は数値の不変であり、受理されるpayloadの最大サイズまで全く同一ではない。上限ぎりぎりの出力には予約分の余裕が必要になる。[S2, S5]

この変更は明示された修正scopeに沿い、失敗記録を無制限に外出しすることを避ける方向なので、本レビューでは受け入れる。上限を後から増やす判断はしていない。

### 6.2 正常経路

source上は以下の順となる。[S2, S3]

1. `.partial`をexclusive作成。
2. 型検査・encode・writeとguard確認。
3. flush/fsync/closeとguard確認。
4. disk上のbytes/hashを32 KiB読みで再検証。
5. exclusive hardlinkでfinal名へ昇格し、自分で作成したtemporary名のみ削除。
6. 成功STOPを監視下で保存。
7. `COMPLETED.v2`を最後に作り、close後にもguard確認。

final名が存在すること、result内のstatusがcompleteであること、全17行が見えることのいずれか単独では成功と判定しない。

### 6.3 file verifierは必要条件

`verify_completed()`はtoken存在、failure receipt不在、成功STOPの条件、result bytes/hash一致を確認する。しかしその関数は、外側processの終了状況、全source/protected identity、row内容の科学監査まで検証しない。[S2]

本レビューでは、報告書に既にある「外側processの正常完了receiptとread-only監査も確認する」という扱いを維持する。外側のexit code 0だけでも不十分である。runnerはhandled technical failureでも正常に終了し得るため、出力されたtechnical/complete statusも合わせる。[S1, S3]

概念上は次を区別する。

`file_commit_conditions`

と

`file_commit_conditions AND external_process_success AND source/result/post_run_audit`。

科学的利用に進む際には後者を確認する。失敗prefixの利用禁止を、file verifierだけで全面的に保証したことにはしない。

## 7. 異常時の記録と残る限界

### 7.1 改善された点

旧実装は例外後に巨大resultを再び一括serializeしていた。S2はresult参照をclearし、短いreason・最大4 frame・stage/phase・RSS・provenance・記録済みprefix identityなどの小型receiptに置き換える。旧巨大fallbackは再実行しない。[S2–S4]

既に超過した`ru_maxrss`はstickyなので、失敗確定後のreceiptだけ周期alarmを止める一方、guard exitまでは既存のAS/CPU制限を残す。これは制限を解除して科学計算を続ける窓ではない。[S2, S13, S14]

### 7.2 boundedの意味を限定する

boundedであるのは主に出力量と、実行する記録処理の範囲である。OS kill、ディスク障害、guard entry/exit障害、receipt作成失敗まで完全に回復する仕組みではない。周期alarmを止めたemergency windowについて、通常経路と同じRSS/wallの周期監視が続くとは表現しない。

特にblockしたI/Oや外部強制終了を含む任意の障害で、有限時間内にSTOPを書けるという保証はない。本レビューはそうした保証を追加要求しているわけではなく、既存のbest-effortという証拠範囲を維持する。

### 7.3 partial identityとsnapshotの時点

失敗receiptのprefix hash/bytesは追跡済みの値であり、writeとbookkeepingの間にsignalが入ればdisk全体の確定identityと一致しない可能性がある。sourceは`identity_reverified_after_failure=false`を記録する。後日のread-only監査が必要であり、失敗後の巨大再hashを回復処理に入れてはいない。[S2, S5]

資源値も時点を区別する。

| 記録 | 主な計測時点 |
|---|---|
| result内`resource` | 科学body後、payload出力前 |
| STOP内`resource_after_result_io`等 | payload出力後、terminal処理完了前 |
| failure receiptのresource | failure handlerでのsnapshot |
| 外側process監査 | runner全体の終了状態を別途確認するための記録 |

`result.resource`をそのまま全I/O・終了処理込みの最終peakと呼ぶべきではない。これはsourceで確認したfield設定順に基づく解釈である。[S2, S3]

### 7.4 今回の追加検査で具体化した境界

人工payloadを成功保存した後にlate failureを模擬し、completion token削除そのものをOSErrorで失敗させた。failure receipt作成前にhandlerが止まるため、file verifierの必要条件だけが残る構成を再現できた。

この人工検査は「通常の成功が誤っている」「元G10でこの障害が起きた」という証拠ではない。また元runtimeのSIGALRM raceを実行したものでもない。**file-only verificationが完全な実行認証ではないという、報告書の限界を具体化したもの**である。

その場合は外側のtechnical status／異常終了／監査未完了を根拠に科学利用を拒否する。外側processがtechnical終了なのにfile verifierだけで採用する運用は本レビューの承認範囲外。この境界は新しい研究条件ではなく、提出報告書に既に含まれている。[S1]

## 8. GPTが今回実施した追加自己検算

### 8.1 実施方法

Git blobとSHA256を照合した`g10_io.py`をASTで読み、唯一の相対`BudgetGuard` importだけをdummy baseに差し替えた。検査対象のencoder・OutputSession・file verifierの関数bodyは変更していない。

実行環境はPython **3.13.5**。固定研究runtimeのPython3.10.12ではない。guardはfakeであり、実際のSIGALRM・RLIMIT・RSS上限試験を行っていない。人工payloadとtemporary directoryだけを使用した。

### 8.2 結果

158項目が想定どおりとなった。内訳は次のとおり。

| 分類 | 件数 | 内容 |
|---|---:|---|
| Encoder | 111 | 固定型、共有非循環tree、seed固定の人工tree、型拒否、bytes/hash・chunk上限・非変更 |
| Explicit guard停止点 | 35 | 人工成功経路の全35 check位置を一つずつ停止させ、failure receipt・成功拒否を確認 |
| その他 | 12 | exclusive衝突、write/short/flush/fsync/close/readback故障、token先行無効化、output cap、file-only判定の限界再現 |

最後の境界検査は「正常に成功するtest」ではなく、file-only verifierの不足が想定どおり再現することを確認した項目である。

この158項目をrepositoryの47 testsに足して「205件の研究テストが通った」とは扱わない。repository47 testsの再実行でも独立研究再現でもなく、GPTの補助的なsynthetic I/O自己検算である。

### 8.3 非実施事項

本番G10 runner、`collect`、native synthesis、行列・量子回路評価、sampling、LP、DF、分子計算、旧17行の科学的比較は実行していない。対象repositoryの書換え、A2作成、登録用marker作成もしていない。

追加コード・結果は `review_selfchecks.py` と `review_selfchecks.json` に保存した。I/O sourceの検証済み写しとREADMEを含めて別ZIPにした。

## 9. Launch・provenance・承認の分離

`g10_launch`は、40桁source、`APPROVED_FOR_ONE_G10_RUN`、science認可true、runs=1、retries=0、mandatory STOP、明示指示、contract hashを要求する。またsourceそのものではなく、唯一のparentがS2であるcleanなauthorization-only childを要求する。[S10]

変更可能pathは新authorization JSONと任意の新receiptに限定し、authorization JSON自体のcommitを必須とする。remote実行branch=HEAD、manifest hashes、protected hashes、fresh result directoryも確認する。

現authorizationは `PENDING_SEPARATE_G10_V2_AUTHORIZATION` でscience認可false、source/hash/instructionはnullである。S2のsource準備や本レビューを実行authorizationとして流用しない。[S11]

**レビュー文書の保存方法にも注意する。** S2を直接書き換えたり、A2へ許可外のレビューMarkdown・index変更を混ぜたりすると、review対象とdirect-child gateを壊す。必要ならレビュー資料は別reference branch/commitに保存し、A2では許可されたreceiptから参照する。実行sourceの親はあくまで固定S2とする。

本レビューでは1,300 pathの全再hashまでは行っていない。保存auditの違反0とsource保護の仕組みを照合した。実行時のmachine gateと実行後のread-only監査はそのまま必要である。[S6, S10, S12]

## 10. 実行可否の判定表

| 論点 | 判定 | 限界 |
|---|---|---|
| 旧科学計算・比較条件の維持 | 実行前の根拠は十分 | live全payloadの動的再認証ではない |
| JSON同値性 | 固定schemaで支持 | 任意nonstring key/objectへ一般化しない |
| 主要なserialization増幅経路の除去 | sourceと保存I/O証拠で支持 | 本番512 MiB内完了は未確認 |
| 参照寿命短縮 | 必要情報の保持と整合 | RSS解放量の保証ではない |
| 正常I/Oの監視 | source上で確認 | OS-level resident hard capではない |
| bounded失敗記録 | 改善を支持 | 全障害での保存・終了保証はない |
| completionの判定 | file条件＋外側process監査で運用 | file verifierだけを十分条件にしない |
| caps・出力会計 | 数値固定、aggregate化を受入れ | payloadの受理域は完全同一ではない |
| one-shot/provenance | 既存gateと整合 | 新明示指示・A2は未作成 |
| 実行前の必須source修正 | **確認範囲ではなし** | 実行可と完走保証を区別 |

## 11. 代替案を今追加しない理由

**RSS上限拡大：採用しない。** 確認された増幅経路へ直接対処できており、監視・failure fallbackを残したままcapだけ変える理由はない。

**row spoolingへの全面移行：今は要求しない。** 同時保持量をさらに下げる可能性はあるが、row集合・後段下界・保存手順の変更範囲が大きくなる。S2が不十分だという新しい技術証拠なしにscopeを広げない。

**元例外行の完全特定を先行：要求しない。** 未確定は明示するが、実証されたserialization増幅と同値な改善経路に対処する価値は既にある。原因特定を名目に旧one-shotを再生しない。

**追加の科学pilotやm9へ進む：採用しない。** 問いは未完了G10の固定次数比較であり、結果を見る前に別入力探索へ移る理由はない。

## 12. 次の担当と手順

**直近の次の段階は、利用者による別途のG10 v2 one-shot実行承認である。** S2について本レビューと同じ論点をもう一度最初からレビューする必要はない。

明示承認が得られた後、Codexへ次を一括して任せる。

1. 固定S2を唯一のparentとするauthorization-only child A2を作成・pushする。新contract SHAと新しい明示指示を固定し、許可されたpathだけを変更する。
2. remote/clean/manifest/runtime/protected/fresh directoryの既定gateを確認する。
3. `g10_degree_matched_native_v2.py`を固定条件で一回実行する。旧失敗runの17行やnew27 keyを、無認可の救済入力として注入しない。認可済みG9 anchor再利用は従来どおり。
4. 結果にかかわらずSTOP。result/token/STOP/failure receiptと外側process記録を照合し、保存identity・provenanceを監査する。
5. 必要なresult/source/監査をrepositoryから取得できる形で報告し、科学的判断をGPTへ戻す。

この段階で新しい合成precision、seed、p/x/provider、比較手法、capを変更する権限は与えない。実行前gateが通らない場合も、source/hash/markerを場当たり的に変更して押し通さない。

## 13. 実行後の判断境界

**有効なcomplete結果が得られた場合。** G10の科学的結果について、資料可用性を確認し、別途開始承認を受けて研究レビューする。焦点はm7でclosed P5＋ordinary tailを超える追加価値が残るか、固定辞書の任意proposal下界と実行可能な費用をどう解釈するか、一般法と低次数特殊化の役割である。

**再度technical inconclusiveの場合。** 旧結果・新marker・partial・小型receiptを保持し、勝敗を判定しない。stage telemetryから原因を切り分けるが、cap拡大・retry・m9を自動実行しない。研究条件を変えない通常の技術問題か、新たな実行契約の判断が必要かをその時点で分類する。

**同じsource・同じscopeの通常技術手順。** 個々のテストやGit操作ごとに研究レビューを増やさない。ただしsource自体を変更する場合は、今回の固定S2への判定を変更後sourceへ自動移転しない。

## 14. 科学的主張への影響

今回のRSS修正とI/O検算は、G10の科学的勝敗を支持しない。一般Green-generatorの価値、新規性、論文化の十分性、実分子・DF・PR/QPE全体の資源優位について新しい肯定結果を出していない。

一方、I/Oで技術停止したことを全return集約法の科学的反証にもしていない。研究目的は従来のまま残り、その問いに答えるために固定G10の次の正式実行へ進めるという判断である。

## 15. 最終結論

**S2 `a139b91f119d109430ae3154a045d0fdcf722233` の実行前レビューは完了した。確認した範囲では必須source修正なし。別途の明示one-shot実行承認へ進めてよい。**

この判断は、メモリ改善の具体的証拠、科学的処理の保持、正常・異常I/Oの境界、旧証拠とauthorizationの保護に基づく。47 testsの件数だけに基づくものではない。

**未承認のまま本番実行しない。512 MiB完走保証と表現しない。file-only verifierを実行認証全体とみなさない。旧technical prefixを科学結果へ昇格させない。**

---

## 16. 固定資料・再現用ファイル

以下のリンクは全て固定S2に紐づく。本文の[S番号]に対応する。外部文献の網羅調査や新規性調査は今回の実行前修正レビューの対象にしていない。

- [S1] [S2引継ぎ報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/docs/tracks/algorithm_codesign/g10_rss_repair_source_and_gpt_review_20261010.md)
- [S2] [修正I/O source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/src/trottertracks/algorithm_codesign/g10_io.py)
- [S3] [修正runner](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/scripts/tracks/algorithm_codesign/g10_degree_matched_native_v2.py)
- [S4] [旧runner（S2内の保持版）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/scripts/tracks/algorithm_codesign/g10_degree_matched_native.py)
- [S5] [Contract v2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/contract_v2.json)
- [S6] [Source preparation audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/source_preparation_audit_v2.json)
- [S7] [静的同値性検査source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/scripts/tracks/algorithm_codesign/verify_g10_rss_repair_preparation.py)
- [S8] [Validation summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/validation_summary_v2.json)
- [S9] [Focused tests source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/tests/tracks/algorithm_codesign/test_g10_streaming_io.py)
- [S10] [Launch gate](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/src/trottertracks/algorithm_codesign/g10_launch.py)
- [S11] [Pending authorization](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/authorization.json)
- [S12] [Critical source manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/source_manifest_v2.json)
- [S13] [共通resource guard](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/src/trottertracks/algorithm_codesign/synthesis_placement/wrapper_launch.py)
- [S14] [Per-key guard](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/src/trottertracks/algorithm_codesign/rte_reallocation/launch.py)
- [S15] [Typed payload scope audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/typed_payload_scope_audit_v2.json)
- [S16] [G10予算・row会計source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a139b91f119d109430ae3154a045d0fdcf722233/src/trottertracks/algorithm_codesign/g10_comparison.py)

旧Sのrunner identity確認：
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/scripts/tracks/algorithm_codesign/g10_degree_matched_native.py

同梱したGPT自己検算ファイル：
- `review_selfchecks.py`：人工I/O・fake guardによる検査コード。
- `review_selfchecks.json`：158項目の結果と証拠区分。
- `sources/g10_io.py`：Git blob/SHA256を確認した固定source写し。
- `review_decision.json`：レビュー状態と実行authorization未付与の明示。
- `bundle_manifest.json`：本bundle内のファイルidentity。

再現コードを実行してもG10 runnerを呼ばない。元研究runtimeを再構成せず、現在のPythonでstdlibのみの人工I/Oを検査する。内容は研究結果のreplayではない。
