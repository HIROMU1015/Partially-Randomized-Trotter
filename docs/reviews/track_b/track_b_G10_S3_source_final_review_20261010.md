# Track B G10 S3：JSONキー互換性修正後の実行前独立レビュー

**作成日：2026-10-10（JST）**  
**レビュー対象：`b9ed01455351628c9073748f5ba5751aa794b789`**  
**Repository：`HIROMU1015/Partially-Randomized-Trotter`**  
**Branch：`track-b-g10-v3-key-compatibility-source-preparation-20261010`**  
**開始承認：このチャットでの利用者の「レビューを開始して」**

## 0. 最終結論と権限

**固定S3は、別途の明示的なG10 v3 one-shot実行承認へ進めてよい。今回確認した範囲で、実行前の必須source修正はない。**

本書のレビュー判定は `G10_S3_SOURCE_REVIEW_PASS_EXECUTION_AUTHORIZATION_PENDING` と記録する。これは本レビュー用の分類であり、既存repositoryのstatusを変更するものではない。本番の実行許可、A3の作成許可、登録用markerの消費許可ではない。現時点のauthorizationはpendingのまま維持する。

受け入れるのは、S2で生じた整数辞書キーの拒否を、旧serializerとの対象領域内の互換性を保ちながら修正したsourceである。以下を主張する判定ではない。

- 本番G10が512 MiB以内で必ず完了すること。
- 全return集約法の科学的優位、新規性、主要手法としての採択。
- 任意のPython objectに対する完全互換性。
- 76件のrepository tests、1,352保護path、180 critical pathをすべて独立再実行・再hashしたこと。
- 本番のlive typed payload、数値計算、量子回路、実SIGALRMの独立再現。

次は利用者の別途の明示実行承認である。その後に限り、Codexが固定S3を唯一の親とするauthorization-only child A3、既定の起動gate、新しいv3出力領域での一回実行、外側process監査、保存監査、mandatory STOPをまとめて担当する。

## 1. 今回のレビュー対象と証拠の区分

### 1.1 対象を限定する理由

現在のTrack Bは、PR内部のRTE・全return集約などの乱択表現を研究している。G10は、固定したsynthetic providerと係数分布について、各Taylor次数内で同じ有限演算子を表現する登録手法を比較する段階である。今回のS3は研究課題や比較条件の変更ではなく、結果出力の互換性修正である。[S04, S06]

v1はRSS上限超過、v2は非文字列の辞書キー拒否でtechnical inconclusiveとなった。失敗runを成功へ再分類しない。旧v1の保存17行、v2のdiagnostic counterを、新しい科学的な比較結果へ昇格させない。

前回S2レビューでは、保存JSONや一部の文字列キー生成箇所を確認する一方、実行中の`provider_calls`が整数labelをキーに持つ点を見落とした。これは前回レビューの不備である。本レビューでは、その生成元からresult保持、serializerまでの対応を重点的に確認する。

Track AのDF・control・RZ資源評価は参考情報に留める。今回のS3へTrack Aの回路、数値政策、比較条件、資源指標を取り込んでいない。

### 1.2 証拠を三段階に分ける

**repository保存事実**：固定commitから取得したsource、契約、tests、監査JSON、計測summaryに記録されている内容。

**今回独立に確認した内容**：sourceの読解、特定sourceのbytes/hash確認、I/O新旧AST差分、runnerのmetadata以外のbytes一致、人工payloadによるserializer／file protocolの追加検査。

**レビュー上の判断・導出**：対象領域内の互換性の論証、メモリ使用の解釈、残余リスクの評価、必須修正の要否、次の担当判断。

保存されたPASSを独立再現済みと読み替えない。以下の数値表で特記しないメモリ値はrepository側のI/O-only計測であり、このレビュー環境の計測値ではない。[S03]

### 1.3 今回実施していないこと

本番runner、`collect`、event generator、sampling、matrix/circuit評価、native synthesis、LP、DF、分子計算、旧結果の科学的再採点、Gitへの書込みは実施していない。人工I/O試験の一時directoryにしか出力しない。そこで用いた人工markerは本番の登録用markerではなく、repositoryのv3領域を作成・消費していない。

## 2. 固定identityと取得状況

| 役割 | 固定identity |
|---|---|
| 旧science S | `05c5ef23fce775a822ab5686f5da2f0d77675864` |
| 旧v1 technical result | `e429c99d77b3222c5cca62750b2d111f87e4cb50` |
| S2 | `a139b91f119d109430ae3154a045d0fdcf722233` |
| 消費済みA2 | `1a2cd261ebe0cf097e71026a9150756dc8c9acc3` |
| 消費済みv2 result R2／S3の親 | `f9d2665283c707e5a025a2c92c1a051153eaf2e1` |
| 今回レビューしたS3 | `b9ed01455351628c9073748f5ba5751aa794b789` |
| S3契約SHA256（source manifest記録） | `f50e9b99a25f859556631c47916dc346295b3d80dd22139ae9cd7af0da0bc7b4` |

branch APIでremote HEADがS3と一致し、親がR2であることを確認した。local worktree cleanはCodexの報告であり、利用者の作業環境をこのレビューで直接検査したものではない。

I/O、runner、契約、pending authorization、tests、型監査、source manifestの関係箇所、validation summary、final preparation auditをGitHub接続から取得した。重要資料の取得不能によるレビュー中断は不要だった。

独立検算用のローカルsourceは、取得したテキストとS2の添付source snapshotを用いて準備し、実行前にSHA256とGit blob SHA1を照合した。照合前の転記差分は検算に使用していない。以下の4ファイルは実際にhash確認したローカルcopyである。

| source | bytes | SHA256 |
|---|---:|---|
| S3 `g10_io.py` | 16,921 | `b29c884840fa9377b071fb0fbda6b96520d06c237ab07e74c13ca97e0d0aa338` |
| S2 `g10_io.py` | 14,864 | `ed98d1bbc22a4c58fe8d479f1651373df1af756d4f5d82c73cdf5bff218552bc` |
| S3 runner | 11,890 | `6af1bf2c0121272aa90c18c49eb05acee76b8f42d9fd99130524eb52fd1fa94a` |
| S2 runner | 11,883 | `e9eae3577ad0939729ee39599005dbb587dfdfbbc70ec34a58223d40041e0b81` |

S3 I/OのGit blob SHA1は`0552f139baaa6016463a1702040f66a657faa2eb`、S3 runnerは`01a1f79c35f423401ef7394e2887e1b3ef71984a`であり、GitHub取得値と一致する。[S01, S02, S09]

## 3. 科学処理はどこまで不変か

### 3.1 I/Oの独立AST差分

S2とS3のtop-level関数・classをASTで比較した。差分は次の3関数に限られる。

- `validate_tree`
- `_compatible_tokens`（追加）
- `iter_json_bytes`

`FractionEncoder`、`IOBudgetGuard`、`OutputSession`、`verify_completed`、`protected_check_streaming`等の関数/class本体は同一である。この比較は保存された`source_diff`の転記だけでなく、hash確認済みsourceから今回独立に計算した。[C01]

### 3.2 runnerの独立bytes比較

S3 runnerで以下の4文字列だけをS2表記へ戻すと、ファイル全体のbytesが固定S2 runnerと一致した。

1. module docstringのv3/v2表示。
2. preparation directory。
3. `contract_v3.json`／`contract_v2.json`。
4. marker-kindのV3/V2ラベル。

その結果、`collect`のASTも完全一致した。runnerはimportもexecuteもしていない。[C01]

これにより、S2→S3の修正が、generator呼出し・順序、reference traversal、固定key集合の作成、G9 cache再利用、新規合成手順、m5 rebudget、row生成、固定辞書下界、行数・binding上限・保護対象検査を変更していないことを確認できる。

S1→S2までの科学AST監査は既存証拠を引き継ぐ。今回S1から全科学証明を独立に作り直したわけではない。

### 3.3 科学条件と資源上限

契約では、`p=(1/5,3/10,1/2)`、`x=5/7`、3-system-qubit provider、`m=3,5,7`、17 rows／34 axes、新規11 rows／保存m5の6 rows、既定arm、precision・seed・synthesis・confidence・proposal-lower政策を維持している。[S04]

比較対象は各m内の

\[
P_m\!\left(-ix\sum_i p_iQ_i\right)
\]

のfull first operator momentである。異なるmで同じexponential accuracyを達成する費用比較、分子DFの取得優位、PR/QPE全体の資源評価ではない。

RSS512 MiB、AS1536 MiB、wall1200秒、CPU900秒、per-key wall30／CPU20秒、output134,217,728 bytes、new synthesis key162、binding12,000、runs1／retries0を維持する。数値capsが同じであることと、任意の障害時にもすべてをhard limitとして完全に強制できることは別である。

## 4. 前回の停止経路と今回の対処

### 4.1 live eventの整数keyは実際に保持される

`g7_generator._event`は`Counter(reduced)`に`calls[child] += 2`を加え、`provider_calls=dict(calls)`としてeventへ保存する。labelは整数である。[S07]

新規m3/m7のeventはreference event列からrowへ渡され、`row()`のevent bindingに保持される。runnerは完成rowをそのままresultへ格納する。S2の文字列key限定validatorとの不一致は、保存JSONだけでは見えなかった。旧`g10_saved.serial`が保存前にkeyをすべて`str(k)`に変換していたためである。[S02, S08]

S3は生成元のlabelや呼出し回数を書き換えず、出力時のkey正規化によってこの不一致を解消する。文字列keyに変更するために科学eventをin-placeで編集する修正ではない。[S01]

### 4.2 他の出力領域

提出型監査は21領域と68辞書AST nodeを記録している。これは静的inventoryであり、本番payload全域を実行traceした証拠ではない。[S06]

今回、特に以下の経路をsourceで確認した。

| 領域 | sourceで確認した型境界 |
|---|---|
| root／provenance／counter | runnerの文字列field名、int／bool／str、runtime辞書 |
| native event | 外側fieldは文字列、wordはtuple、係数・proposal・weight・ratioはFraction、provider labelは整数key |
| native IR／cost | gate列はlist／tuple、wire indexは値として整数、ratioは文字列、cost辞書は文字列keyとint／Fraction |
| synthesis cache | ratio由来の文字列key。保存sequence、hash、error上界は文字列、countsはint、guardはbool |
| CTS certificate | 演算中のtuple-key代数辞書は一時値。保存時は`axis+':'+str(phase)`へ変換し、`A.json()`は2個の文字列のlistを返す |
| m5 anchor | 保存JSONのdecode済み文字列keyを保持し、既存deepcopy／rebudgetを維持 |

数値libraryのmoduleや行列そのものをJSONへ新しく保存する修正はない。新規に加えたserializerはint以外のkeyにも一般的に`str(key)`を使うため、`provider_calls`一か所だけに名前依存の例外を設ける設計ではない。[S01, S02, S06–S12, S15]

## 5. キー衝突とJSON同値性の論証

### 5.1 同値性の対象領域

対象は、通常のdict／list／tuple、Fraction、有限のJSON scalarからなる循環のない値graphである。全keyの`str(key)`は安定し、副作用なく文字列を返すことを仮定する。入力はencoding中に変更しない。recursionや資源制限に抵触せず、処理が正常に完了する範囲について出力bytesを議論する。

固定研究schemaのkeyは整数labelまたは文字列であり、この条件に対応している。副作用を持つ任意のPython objectまで互換性を拡張したとは主張しない。

### 5.2 旧変換と局所変換

旧serializerは、辞書Dについて

```python
{str(k): serial(v) for k, v in D.items()}
```

を作る。S3は各active辞書に対して

```python
normalized = {}
for key, child in D.items():
    normalized[str(key)] = child
```

というshallow mapを作り、その後に残ったchildを再帰的にencodeする。[S01, S08]

同じ文字列へ変換されるkeyが複数ある場合、辞書の位置は最初の挿入位置、保持される値は最後の代入値となる。Pythonの辞書仕様は、既存keyの更新が順序を変えないことを定めている。[E01]

例：

```python
{1: 'first', 'middle': 0, '1': 'last'}
```

では、両方式ともkey順序は`['1', 'middle']`、`'1'`の値は`'last'`になる。単純にkey-valueを逐次emitして同名JSON memberを二つ残す方法とは異なる。

### 5.3 構造帰納による説明

scalarは同じFraction変換とJSON scalar encoderを用いる。list／tupleでは順番を保ったarrayを出力する。辞書では上記の同一key列・同一最終childを選び、各childの出力が一致するため全体が一致する。インデント2、ensure_ascii=False、allow_nan=False、separator、末尾改行を揃えることで、意味的JSON一致だけでなくbytes一致を得る。

共有された循環のないsubtreeは両方式とも各出現箇所でJSONとして展開する。S3が参照identityを保持することは、JSONに共有参照記法を追加するという意味ではない。

これは対象領域内の実装に対する論証であり、全Python object、全深さ、資源不足時を含む総合的な形式検証ではない。

### 5.4 完全互換ではない領域

S3は衝突で消える値も含めて全original valueを検査する。このため、例えば

```python
{1: float('nan'), '1': 'replacement'}
```

は、旧方式ではNaNが上書きされて有効JSONになり得るが、S3では拒否される。unsupported valueを上書きで隠す場合も同様である。今回の追加人工検査でもこの差を確認した。[C02]

これはS3の説明資料で明示された、有限値の入力境界を維持するための制約であり、隠れた完全互換性の証拠ではない。科学的出力でNaNやunsupported valueを隠して成功扱いする必要はないため、本件を必須修正としない。

非finiteな**key**は`str(key)`によって文字列になり得る。一方、非finiteな**value**は拒否される。この二つを同じ規則として説明しない。

## 6. メモリ使用の評価

### 6.1 構造上取り除いたもの

S3は、全resultの再帰的な新container graph、全JSONのchunk list、全JSON文字列へのjoinを必要としない。dictごとのshallow mapが保持するのはchild参照であり、childの再帰copyではない。list／tupleは参照順に走査する。[S01]

補助メモリは概ね、再帰深さd、同時にactiveな辞書の幅とkey文字列、最大単一token長、bounded output bufferに依存する。概念的には

\[
M_{aux}=O\!\left(d^2+\sum_{D\in active}\{width(D)+keybytes(D)\}+token_{max}+buffer\right)
\]

と保守的に整理できる。深い入れ子では各active containerのindent文字列も保持するため、depth項は最悪の場合d²を含めた。これはPython allocatorを含む厳密byte上界ではない。単一の巨大辞書や巨大文字列、極端な入れ子を常に定数メモリで処理できる実装でもない。

1 writeのUTF-8量は32,768 bytes以内に分割する。だがJSON encoderが一つのescaped string tokenを先に作る可能性は残るため、write chunk上限だけからprocess全体のメモリ上界を導けない。

### 6.2 保存されたI/O-only計測

| 対象 | 旧方式peak RSS | S3 peak RSS | 出力bytes | bytes／SHA256 |
|---|---:|---:|---:|---|
| 保存v1 technical JSON | 509.71484375 MiB | 255.5 MiB | 66,842,493 | 一致 |
| 人工typed 50,000 rows | 331.0 MiB | 79.75 MiB | 16,410,125 | 一致 |

保存JSONのhashは`b62695c19964a5a121b965c39142427bfa8048efad9bce05f3221494bb14dfe1`、人工typed payloadは`e1d3ce71482e0f846c9004859880557dbdc6e158b0625157eb98791c19ce9aa4`である。[S03]

保存JSONではguard wallが旧約2.035秒、S3約6.178秒、人工typedでは約0.808秒から約2.774秒となっている。メモリ使用の低下と引換えに処理時間の増加がある。これを隠して「すべての古典性能が改善した」とは言わない。

S2とS3の人工typed payloadは同じサイズ・構成ではないため、S2のtyped peakとS3のtyped peakを直接比べて改悪・改善を主張しない。

### 6.3 本番237.34 MiBとの関係

v2の237.34 MiBは、正式payloadのencode前に停止したrunの値である。これにI/O-onlyの増分を単純加算したり、旧runから削減量を単純減算したりしてS3本番peakを認定しない。数値library、allocator retention、Fraction、共有参照、実行時cache、出力全終端までのheapが異なるためである。

S3の本番512 MiB完了は未検証であり、このレビューで保証しない。一方、具体的に確認された型不一致に修正を対応させ、低メモリ出力の根拠を維持しているため、追加の大規模spoolingやcap引上げを実行前必須にはしない。

### 6.4 診断器の初回失敗を隠さない

validation summaryには、最初のsaved-stream診断でbytes/hash以外の`path` fieldを含むidentity辞書比較が失敗したこと、その比較だけを修正して追加の診断を行ったこと、初回記録を保持したことが記載されている。記録上、診断は5 invocations、本番run／retryは0である。[S03]

本レビューではこの申告を含むsummaryと関連source境界を確認したが、5 processの全raw bytesを独立再hashしたとはしない。「全診断が初回から無条件に成功した」とも表現しない。

## 7. Testsと今回の独立検算

### 7.1 repository側の保存証拠

76 focused testsの内訳は、適用可能なS2 tests46件と新規30件である。旧「非文字列keyを一律拒否」testは今回の仕様変更で除外されたことが明示され、元testファイル自体は保持されている。17人工caseのbytes／SHA256比較は別の記録であり、76と単純合算して独立な科学検証数としない。[S03, S05]

testsは整数provider label、混在key、両順序の衝突、insertion order、Fraction、tuple／list、共有subtree、Unicode、無効値、key変換例外、output session、pending launch等を対象にしている。本番event生成、数値計算、synthesisを含むtestsではない。

### 7.2 今回実施した追加検算

本レビュー環境のPythonは**3.13.5**であり、固定研究環境の3.10.12ではない。hash確認済み`g10_io.py`の唯一のrelative `BudgetGuard` importだけをASTから除き、dummy baseとfake explicit guardを与えた。検算対象の関数・class本体は変更していない。

実施結果は15 test groupsすべてpass。内容は次のとおり。[C02]

| 独立検算の種類 | 確認内容 |
|---|---|
| bytes比較 | 213人工cases。うち120は衝突keyとmiddle fieldの全挿入順序、64は決定的に生成した人工nested typed graph |
| 具体的な失敗型への回帰 | `rows[].events[].event.provider_calls`に整数keyを含む手書きfixtureが成功し、元keyの型・順序が不変 |
| scalar・container | Fraction、Unicode、tuple、scalar subclass、共有subtree、空container等 |
| domain boundary | NaN等のvalue、unsupported value、循環参照、上書きで隠れる無効値の拒否 |
| file protocol | 人工payloadのsuccess、disk identity、bytes会計、partialとfinalの分離 |
| guard injection | 小型成功経路の明示check39か所を一つずつ停止させ、completion不成立・technical receiptを確認 |
| その他 | key変換例外、既存file衝突、output cap、token unlink障害時のfile-only verifierの限界 |

213 casesは「213個の分子系」でも「213回の科学的実験」でもない。39か所は明示checkのfailpointであり、すべてのPython bytecode間、SIGALRM時点、OS障害を網羅したものではない。76 repository testsの独立rerun、実RSS／AS制限の再現とも区別する。

## 8. 終了判定と失敗時保存の評価

### 8.1 維持されたprotocol

S3はS2の以下の順序を維持する。[S01]

```text
exclusive .partial
→ 型検査・逐次encode/write
→ flush/fsync/close
→ disk上のbytes/hash照合
→ exclusive hardlinkでfinal名へ昇格
→ success STOP
→ COMPLETED.v2
→ 最終guard確認
```

`COMPLETED.v2`という名前は、新しいv3 directoryの中で既存file protocolを再利用している。旧v2 markerや旧outputを再利用するという意味ではない。

### 8.2 scientific completionの必要条件

file側ではtoken、正しいsuccess STOP、result identity、failure receipt不在を要求する。それに加えて、外側processの正常終了・COMPLETE status、失敗表示の有無、source/resultの保存監査を確認する。token、exit0、最後のstdout一行のいずれか単独では成功にしない。[S01, S04]

`verify_completed()`はfileの必要条件を確認する関数であり、外側processの全履歴や科学的妥当性を保証する関数ではない。この区別はS2レビューから維持する。

### 8.3 今回も確認したfile-only判定の限界

追加の人工検査で、成功保存後にlate failureを注入し、completion tokenのunlink自体をI/O errorにした。この場合、failure receipt作成に到達せず、file-only verifierが必要条件成立を返すことを確認した。[C02]

これは実際の研究runで発生したと主張するものではない。外側processの終了status・失敗記録を併せて確認するという、既に開示されていた境界の人工再確認である。失敗記録の保存失敗を示す`failure_receipt_failed`等の出力を無視して成功認定してはならない。

固定S3で科学的判定をfile verifierだけに委ねない運用が明示されているため、今回この既知限界を理由にsource差替えを必須とはしない。

### 8.4 bounded failure receiptはbest effort

失敗後は巨大resultの再serializationを行わず、小型receiptへ移る。既に超過したpeakが再び例外を出し続けないよう、小型failure windowで周期alarmを止める。科学処理は継続せず、AS／CPU設定の解除はguard exitで行う。[S01]

OS kill、OOM、disk full、unlink／receipt I/O障害でも必ず記録が完成する保証はない。小型bytes上限は、任意のI/O停止に対するwall-time保証と同じではない。SIGALRM由来例外は任意の実行点で発生し得るため、writeとhash bookkeepingの間のずれ等もあり得る。[E03]

そのため、failure receiptに記録されたpartial identityは追跡中のprefix情報であり、失敗後の全file再検証ではない。後続のread-only監査で保存状態を確認する。

## 9. 保護対象・manifest・認可

S3のsource manifestはcurrent source、tests、contract、ledger、validationへのhashを含む。実行認可用authorizationは将来のA3で変更できるよう意図的に除外されている。[S09]

旧1,300保護pathと拡張1,352保護pathのPASSはrepositoryの保存監査として確認した。既存の`g10_io.py`だけは承認されたmutable pathであり、現在のworktree上のS3版と、固定S2／R2 commitに残る旧版を区別する。「旧sourceを保持」とは、このmutable pathも現在のtreeで旧bytesのままであるという意味ではない。[S10]

現authorizationは`PENDING_SEPARATE_G10_V3_AUTHORIZATION`、`science_execution_authorized=false`、`source_commit=null`である。このnullは準備段階の正常状態であって、レビュー対象commitが不明という意味ではない。[S13]

実行gateでは、固定sourceの指定、唯一parentがそのsourceであるA3、authorization／任意receipt以外を変更しないこと、認可JSONがA3差分に含まれること、clean tree、remote HEAD一致、manifest hash、protected history、新しい空のresult directoryを確認する。[S14]

将来、レビュー記録をrepositoryに保存するcommitを作る場合、そのcommitは参照記録として扱い、A3の親を黙ってレビュー文書commitへ置換しない。今回レビューした固定S3そのものがA3の唯一の親である。

## 10. 残る不確かさと、今は追加しない作業

| 残る事項 | 今回の扱い |
|---|---|
| 全live typed payloadの動的確認 | 未実施。source型境界と人工検算から修正を支持するが、本番成功を既成事実にしない |
| 本番512 MiB完了 | 未確認。現上限を維持した一回実行で観測する |
| native heap／allocator retention | I/O-only検算から厳密推定しない |
| 極端な単一dict・文字列・depth | 一般的な定数メモリ保証なし。実装が述べる範囲を超えて主張しない |
| 任意Python object互換性 | 目標にしない。科学schemaにない副作用object等へscopeを広げない |
| 76 tests／全保護pathの独立再現 | 今回は実施していない。source読解・選択的hash確認・別の人工検算と分離 |
| G10の科学的winner | 未判断。失敗記録やserializerのPASSから決めない |
| Track Aとの統合 | 今回不要。RZ／T、分子DF／synthetic providerを混同しない |

現段階では、追加の科学pilot、全science suiteの無断再実行、row spooling、cap拡大、provider・seed・次数・precision変更を必須条件に追加しない。今回の限定不具合に対応する修正・証拠があり、登録されたG10の問いは未回答のため、同じscopeで次の一回の実行承認へ進む情報価値があると判断する。

これは成功するまで無制限に再実行してよいという判断ではない。次も失敗すれば、その結果を保持してSTOPする。科学的な条件や比較の意味を変える必要が生じる場合は、別の研究判断へ戻す。

## 11. 次の作業と最終停止境界

**今回のレビューは完了。本番実行認可はまだ与えない。**

別途の明示指示が得られた後の担当はCodexである。作業単位は、固定S3に対するA3作成、既定gateの確認、一回実行、外側processと保存artifactの監査、結果公開、mandatory STOPまでとする。通常のgate確認ごとにGPTレビューを挟まない。

有効なG10結果が取得された後は、単に「testsがPASSした」「serializationが完了した」ではなく、全return集約とclosed低次数特殊化の差、次数内比較、費用・normalization・測定回数の関係、CTSや固定辞書下界を含む主張範囲を独立に評価する。それは今回のsourceレビューとは別の重要レビューであり、結果取得可能性の確認と利用者の開始承認を経て実施する。

| 最終判定項目 | 判定 |
|---|---|
| S3対象領域内のkey互換性修正 | 支持する |
| S2→S3科学処理の不変性 | runner bytes／AST比較と契約から支持する |
| 低メモリ出力の継続 | source構造・保存I/O検算から支持する |
| 追加の必須source修正 | 確認範囲ではなし |
| 本番の完走保証 | しない |
| 次の手続 | 利用者の別途明示one-shot承認 |
| A3／本番marker／本番run | 今回未作成・未実行 |
| 研究方針・方法の科学的採択 | 変更しない |

## 12. 参照資料

以下は固定commitに対応する一次source・保存記録である。外部Python文書は言語仕様の補助確認にのみ用い、固定研究runtimeや実験結果の代替にしない。

- **[S01]** [S3 I/O source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g10_io.py)
- **[S02]** [S3 runner](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/scripts/tracks/algorithm_codesign/g10_degree_matched_native_v3.py)
- **[S03]** [Validation summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3/validation_summary_v3.json)
- **[S04]** [Contract v3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3/contract_v3.json)
- **[S05]** [Compatibility tests](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/tests/tracks/algorithm_codesign/test_g10_key_compatibility_v3.py)
- **[S06]** [Payload type/key audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/docs/tracks/algorithm_codesign/g10_v3_payload_type_and_key_audit_20261010.md)
- **[S07]** [Integer-key producer](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g7_generator.py)
- **[S08]** [Legacy serializer](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g10_saved.py)
- **[S09]** [Source manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3/source_manifest_v3.json)
- **[S10]** [Final preparation audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3/final_source_preparation_audit_v3.json)
- **[S11]** [CTS/reference representation](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g10_reference.py)
- **[S12]** [Native representation/cost](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g9_native.py)
- **[S13]** [Pending authorization](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3/authorization.json)
- **[S14]** [Launch gates](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g10_launch.py)
- **[S15]** [Synthesis record types](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/rte_reallocation/numeric.py)
- **[S16]** [S3 preparation/handoff report](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/docs/tracks/algorithm_codesign/g10_v3_key_compatibility_source_and_gpt_review_20261010.md)

- **[E01]** [Python 3.10 documentation: dict insertion order](https://docs.python.org/3.10/library/stdtypes.html#mapping-types-dict)。3.10系列の言語仕様参照であり、固定3.10.12実行環境の再現ではない。
- **[E02]** [Python 3.10 documentation: JSON encoding](https://docs.python.org/3.10/library/json.html)。3.10系列の言語仕様参照であり、固定3.10.12実行環境の再現ではない。
- **[E03]** [Python 3.10 documentation: signal exceptions](https://docs.python.org/3.10/library/signal.html#note-on-signal-handlers-and-exceptions)。3.10系列の言語仕様参照であり、固定3.10.12実行環境の再現ではない。

- **[C01]** 同梱 `review_static_checks.py`／`review_static_checks.json`：今回のsource bytes/hash・AST差分検査。
- **[C02]** 同梱 `review_selfchecks.py`／`review_selfchecks.json`：今回の15 groups、213 bytes比較cases、39明示guard failpoints。

## 13. 再現用資料の範囲

同梱ZIPには本書、今回の追加検算scriptと結果、hash確認したS2/S3のI/O・runner source snapshot、判定JSON、source URL索引、bundle manifestを含める。`review_static_checks.py`はrunnerを読み取るだけで、import／executeしない。`review_selfchecks.py`は人工I/Oだけを行い、元の研究repositoryや登録出力領域を変更しない。

本資料を保存することと、G10 v3を実行することは別である。本書を利用して自動的にA3や本番markerを作らない。
