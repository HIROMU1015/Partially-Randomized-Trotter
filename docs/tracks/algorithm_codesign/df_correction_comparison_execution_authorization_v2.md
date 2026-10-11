# DF correction comparison one-shot authorization receipt

2026-10-12 JST。

固定science source S'：`41349b73157c7781e0b534cf0fe61e06bf439954`。
公開receipt-only HEADは実行に使用しない。S'の直接子authorization-only Aから開始する。

利用者の最新指示（原文）：

> 作業を進めて

直前に確認した実行scope（CD6条件、E最大22 compile、T最大64keysの限定one-shot、
retry0、commit/push/取得確認後mandatory STOP）への指示として受領した。
GPTの最終レビューPASSと利用者の今回の実行指示を別々に記録する。

science contract v2とintegrity source v3を保持。fixture/seed/angle/precision/threshold/
worker/thread/capを変更しない。partial failureを含め一回だけ、再実行・追加科学は行わない。
CD記号資源と別E局所資源を区別し、PR総資源の勝者を自動判定しない。

結果・process status・provenance・分割ファイル完全性を監査しGitHubへ保存してSTOPする。
その後の科学的レビュー開始・研究判断は利用者/GPTへ戻す。

## GPT最終レビュー原本（以下のUTF-8 bytesをauthorization JSONのoffset/lengthで同定）

# Track B：DF補正A/B比較 v3限定差分・最終実行前レビュー

- 作成日：2026-10-12 JST
- 開始承認：利用者の「レビューを開始して」
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 前回のsource S：`b4aba2b698cdc81710dbdbb5769a1686f716ffe2`
- 今回の固定source S'：`41349b73157c7781e0b534cf0fe61e06bf439954`
- 公開HEAD：`a1ec59e6b2d1c25f10daabfc752755bda317f718`
- Branch：`track-b-df-correction-comparison-v3-integrity-preparation-20261011`
- 対象：前回指摘した大きい結果の取得性、作成時T inventoryのSHA256結合、および変更による既存実行制御への影響。
- 対象外：A/Bの数学・新規性の再採択、登録CD/Eの比較実行、native compile、T合成、分子計算、研究全体の再レビュー。

## 0. 最終判断

**今回の限定差分レビューは通過と判断する。前回の2つの実行前阻害要因は、固定source S'で解消されている。追加のsource修正を実行前の必須条件としては求めない。**

本書の判断ラベルを `LIMITED_DIFF_REVIEW_PASS / READY_FOR_SEPARATE_ONE_SHOT_AUTHORIZATION` とする。これはGPTの評価を表すものであり、repositoryのauthorizationやSTOPを変更したものではない。

次は同じsourceの再レビューではなく、利用者による既定one-shot実行の明示承認である。承認後にCodexがS'の直接子authorization Aを作り、既定範囲を実行して結果を保存しSTOPする。今回のレビュー開始承認を、実行の承認へ読み替えない。

この判断は「本番が必ず完走する」「全Tが取得できる」「whole方式が有利」「新規アルゴリズムとして採択」という意味ではない。登録入力での数値余裕・資源上限・T coverageは、実行によって初めて確認する事項として残る。

## 1. 確認したsourceと変更範囲

GitHubの固定refからstorage、inventory_integrity、coordinator、worker、guard、launch、追加tests、focused-test receipt、source-integrity記録、source manifestの関連部分を取得して読んだ。[R1–R10]

GitHubのcommit比較S→S'はahead 2、behind 0であり、科学契約・数学／数値moduleの変更を含まない。既存sourceの変更はcoordinator、worker、guard、launchと関連テスト・索引に限定され、新しいstorageとinventory-integrity moduleが加わっている。保存された不変性監査も、科学契約と数値moduleのbyte一致、旧artifact・marker・STOP・authorization保持を記録している。[R9,R11]

公開branchのHEADは指定されたa1ec59…であり、その親はS'=41349…である。実行sourceはこの公開確認HEADではなくS'である。[R12]

契約v2、固定fixture、seed、初期状態、t・delta・gamma・s、P5、誤差・confidence予算、compiler/synthesizer、最大22 compile・64 T keys、資源上限を、本レビューで変更していない。既存レビューが支持した数学・誤差会計は今回の再検討対象ではない。

## 2. R1：大きい結果の分割保存・完全復元

### 2.1 前回の問題と実装された経路

前回は、全6軌道Pauli係数recordを単一JSONに保存すると、512 KiB以下のreview用取得経路が閉じていなかった。

新しい`storage.py`は、元の`json.dump(default=str, indent=2)`相当のserializationと末尾newlineを維持する。512 KiB以下は元のlogical pathのraw JSONを保持する。大きいJSONは最大256 KiBのUTF-8断片と`.review.json`索引に保存する。[R1]

索引には、logical path、元bytes・SHA256、encoding、serializer、断片の順序・path・bytes・SHA256、主要fieldの型・件数が入る。書き手は断片をディスクから読み戻し、byte数・各digest・全体digestを照合した後に索引を公開する。索引自体にも512 KiBの上限がある。

`load_json`は全断片の検証を完了してからJSON valueを返す。`iter_verified_bytes`は検証済み断片を順にyieldするが、全体digestの最終確認は末尾で行われるため、全結果の認証にはiteratorを最後まで消費する必要がある。現在の`load_json`と`artifact_identity`はその条件を満たす。[R1]

破損したchunk、欠損、順序不整合、全体digest不一致、logical artifactのchunk directoryからのpath escapeは拒否する。数値の省略・閾値削除・丸めにより取得性を確保したものではない。

### 2.2 writerだけでなくreaderも修正されている

workerの通常結果・繰返しprefix、coordinatorのE入力・E結果・T結果・最終会計、guardの結果存在判定・per-key receipt読込み、workerのterminal STOP判定に、新しい共通保存／読込みAPIが接続されている。[R3–R5]

大きい結果では元のlogical `.json`ファイルが存在しない場合があるため、`artifact_exists`と`load_json`への切替えは必要である。単にwriterを分割化してreaderを従来の`Path.read_text`のまま残した変更ではない。

### 2.3 prefix更新と中断

大きい世代では、完成した索引を公開する処理が読込み側の切替え地点になる。

- 新世代のchunk書込みが途中で失敗した場合、旧raw／旧索引の完全な世代を保持する。
- small→largeで新索引公開後に旧raw削除が中断しても、readerは索引を優先する。
- large→smallで新raw設置後、旧索引を取り下げる前に中断した場合、旧索引の完全な世代を読む。
- large→largeで新索引公開後に旧chunk cleanupが失敗しても、新索引の完全な世代を読む。
- 壊れた新索引／chunkを、残存する古いrawで覆い隠すfallbackはしない。

この区別はsource読解に加え、今回の例外注入mockで確認した。実際の電源断・OS crash・ディスク故障を発生させた試験ではない。fsync等を用いた電源断耐久性までは、ここで認証しない。

一時ファイルや旧新世代が一時的に共存すれば、そのbytesも従来のoutput capに入る。保存処理を追加したことで本番出力が必ず上限内に収まるとは主張しない。

**判断：前回R1は解消。今回の比較の実行前に、別の保存形式や追加の保存基盤の開発を要求する必要はない。**

## 3. R2：inventory作成時SHA256の結合

### 3.1 二つのdigestを固定する

`freeze_inventory`はinventoryを排他的に保存し、その作成bytesのSHA256を取得する。さらに、inventoryのbytes・digest・順序付きkey・件数、source/authorization/contractの来歴、marker digestをfreeze receiptへ排他的に保存する。[R2]

親が保持するbindingは、inventory SHA256とfreeze receipt SHA256の組である。read-only permissionも設定するが、ownerによる改変が物理的に不可能だとする保証には使っていない。整合性の基準は元のbindingである。

### 3.2 bindingの伝播

確認した経路は次のとおり。[R2–R4]

| 地点 | 確認内容 |
|---|---|
| coordinator・job生成 | 元bindingを全T jobへ渡す |
| worker開始 | inventory/receipt digest、marker、key membershipを確認し、job-start receiptにbindingを保存 |
| 合成前 | source/authorization/contractを含めてinventory来歴を確認 |
| 合成後・結果保存前 | 元digestとmarkerを再確認し、T結果へ同じbindingを保存 |
| 親の各batch前・集計前 | メモリに保持する元bindingに照合 |
| 各T結果の読込み | 元binding・期待angleとの一致を検査 |
| T集計 | bindingと各結果のbytes・SHA256を保存 |
| terminal保存前 | 元inventoryを再検査し、不一致ならTECHNICAL_INCONCLUSIVEへ戻す |

現在のinventoryへ合わせてjobのdigestだけを更新しても、元receiptのdigest／内容との整合を満たさない。inventoryとreceiptを両方書き換えても、親が保持する元bindingでは拒否される。

これは同一ownerが親のメモリや実行コードを含む全環境を支配する状況への暗号学的な防御を主張するものではない。また、未合成sequenceの正しさをdigestだけで証明するものでもない。対象は、宣言されたone-shot処理中の「最初に固定した対象を使い続ける」という来歴管理である。

**判断：前回R2は解消。membershipだけで元inventoryを代用していた不足は埋まっている。**

## 4. 既存実行制御への影響

`launch.py`は新source manifest v3を参照するようになったが、S'の直接子authorization、許可path、source/contract hash、remote一致・clean状態、既存markerによる再実行拒否という基本gateは保持されている。[R6]

coordinator自身のnumeric runtime確認がmarker作成前に追加されている。今回の読解では、実行認可前に登録fixtureへ到達する新しい経路は確認していない。[R3]

guardの変更は、chunk形式の結果を存在／読込み判定で扱うためのAPI切替えであり、CPU/RSS/wall/output capの値を緩めていない。50 msごとのRSS監視、上限検出時のprocess-group停止、queue取消、終了済みchildのCPU会計などの既存範囲を維持する。[R5]

RSSの瞬間的overshootゼロ、親の全metadata処理に対する独立watchdog、電源断時の完全な永続性は今回の保証対象ではない。これらは前回から区別されていた制約であり、今回の修正を理由に無条件保証へ昇格させない。

## 5. 今回の確認方法と証拠の区別

### 5.1 Codexの保存記録

固定source S'には88件PASSのfocused-test receiptがある。内訳は既存62件と追加integrity26件。登録CD/E・T合成は0と記録されている。追加test sourceは分割・復元・Unicode・破損・prefix中断・inventory変更・terminal STOP・親runtime拒否等を扱う。[R7,R8]

259件PASSの不変性／metadata記録も取得した。[R9]

これらはCodexのsource-bound記録であり、GPTが88件全てを同じruntimeで再実行した結果ではない。

### 5.2 GPTによる限定mock確認

今回、GitHub connectorから取得した`storage.py`と`inventory_integrity.py`をローカルへ複写し、Git blob SHA1で取得内容との完全一致を照合してから読み込んだ。

| file | bytes | Git blob |
|---|---:|---|
| storage.py | 7,128 | cc4968eed910954f08655ceedaec08fb4d2a63c8 |
| inventory_integrity.py | 2,795 | 7cb4efa89a2bb6060675d16eaed60494fb2fd9ee |

inventoryの唯一のguard依存`exclusive_json`は、取得したguard sourceから関数を抜粋したreview用依存である。Supervisorやscientific workerを実行したものではない。

Python 3.13.5のreview環境で、**独立に書いたmetadata/mock 30件が全てPASS**した。固定production runtimeの再認証とは区別する。

- 保存・復元21件：小さいraw、512 KiB境界の前後、4,097件の架空record、長い文字列とUnicode、破損・欠損・順序・全体digest・byte数・path escape・symlink、古いrawへの誤fallback、prefixの各切替えと中断、上書き拒否。
- inventory9件：元bindingの有効性と再freeze拒否、whitespace・順序・angle・key削除・同path置換・receipt変更、元bindingに対する差替え拒否、marker/provenance/result keyの不一致。

主なサイズ境界確認：524,287 bytesと524,288 bytesはraw、524,289 bytesはchunk形式となった。4,097架空recordの1,184,097 bytesは5断片から完全復元した。別のUnicode入力1,190,028 bytesは6断片から完全復元した。これらは今回用意したmockであり、Codexのremote mockや未実行のE出力と同じデータではない。

詳細は同梱`review_checks.py`、`review_checks.json`、`retrieved_source_identity.json`を参照する。この30件を88件に加算して科学的検証数を増やさない。

### 5.3 今回行っていないこと

登録frameの生成、CD6task、Eのnative compile、T合成、分子／GPU／LP／量子shot、actual runnerの起動、repository変更、実行authorization・本物のone-shot marker作成は行っていない。

全40公開資料の独立byte再取得やCodexのremote分割mockの再結合は行っていない。レビューに必要なsource等を個別取得し、二つの新moduleをbyte照合して、別のmockで検証した。

コードを読んだ範囲とテストした範囲を区別する。coordinator→実際のsynthesis worker→T集計の全end-to-end実行は未実施で、接続はsource読解に基づく。

## 6. 残る事項と、その扱い

今回の2点とは別に、従来から残るものは以下である。

| 事項 | 今回の扱い |
|---|---|
| 本番inputのnumeric margin・fingerprint・cap内完走 | 事前に保証しない。固定条件で失敗したらtechnical STOP |
| 全A T coverage | 未取得ならT_INCONCLUSIVE。実装の非優位の証拠へ変換しない |
| 部分T下界 | 固定された既得sequenceの加法completionに限定する |
| CDとE | 異なるdomainを保持し、E単価×CD頻度を禁止する |
| PR全体の実測総T・一般whole方式の優劣 | この計画だけでは確定しない |
| 新規性・有用性の採択 | 今回の限定差分レビューの対象外 |

これらの未確定性は、実験前に全て計算して消すべき不足資料とは扱わない。既定の停止・分類・報告規則の下で、一回の実行によって確認する対象である。

## 7. 次の担当と実行承認

**source修正の準備はここで区切る。利用者が明示的に既定one-shotを承認した後は、Codexが実行担当となる。**

実行時の基準はS'=41349b73157c7781e0b534cf0fe61e06bf439954とする。公開receipt-only HEADから実行branchを作らない。S'から専用実行branchを作り、既定のauthorization JSONと許容される実行承認receiptだけを変更した直接子Aを固定する。remote A、clean状態、source/runtime/contract identityを既存gateで確認して実行する。[R6]

今回のレビュー文書をsource S'へ追加してから、さらに再freeze・再レビューする工程は要求しない。承認後のreceiptに本レビューを参照させる等、既存のauthorization-only path制約を守る。

同じR1/R2をもう一度レビューする必要はない。新しい科学条件やsource変更、意味論に関わる新問題がなければ、次は実際の比較データ取得である。実行結果をcommit・push・取得確認してSTOPした後、必要資料の可用性を確認し、結果の科学的レビューはその開始承認を得て行う。

## 8. 参照source

全sourceは特記しない限り固定S'に対するURLである。本文のsource由来の記述と、今回のmock確認・レビュー判断を区別する。

- [R1] [storage.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41349b73157c7781e0b534cf0fe61e06bf439954/src/trottertracks/algorithm_codesign/df_correction_comparison/storage.py)
- [R2] [inventory_integrity.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41349b73157c7781e0b534cf0fe61e06bf439954/src/trottertracks/algorithm_codesign/df_correction_comparison/inventory_integrity.py)
- [R3] [coordinator](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41349b73157c7781e0b534cf0fe61e06bf439954/scripts/tracks/algorithm_codesign/run_df_correction_comparison_v2.py)
- [R4] [worker.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41349b73157c7781e0b534cf0fe61e06bf439954/src/trottertracks/algorithm_codesign/df_correction_comparison/worker.py)
- [R5] [guard.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41349b73157c7781e0b534cf0fe61e06bf439954/src/trottertracks/algorithm_codesign/df_correction_comparison/guard.py)
- [R6] [launch.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41349b73157c7781e0b534cf0fe61e06bf439954/src/trottertracks/algorithm_codesign/df_correction_comparison/launch.py)
- [R7] [追加integrity tests](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41349b73157c7781e0b534cf0fe61e06bf439954/tests/tracks/algorithm_codesign/test_df_correction_integrity_v3.py)
- [R8] [focused tests receipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41349b73157c7781e0b534cf0fe61e06bf439954/artifacts/track_b_df_correction_comparison_integrity_v3/2026-10-11/focused_tests_final_v3.json)
- [R9] [不変性・metadata監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41349b73157c7781e0b534cf0fe61e06bf439954/artifacts/track_b_df_correction_comparison_integrity_v3/2026-10-11/source_integrity_verification_v3.json)
- [R10] [source manifest v3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41349b73157c7781e0b534cf0fe61e06bf439954/artifacts/track_b_df_correction_comparison_integrity_v3/2026-10-11/source_manifest_v3.json)
- [R11] [S→S' commit比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/compare/b4aba2b698cdc81710dbdbb5769a1686f716ffe2...41349b73157c7781e0b534cf0fe61e06bf439954)
- [R12] [公開HEADのmetadata・親source](https://api.github.com/repos/HIROMU1015/Partially-Randomized-Trotter/branches/track-b-df-correction-comparison-v3-integrity-preparation-20261011)（branchは可変。本文の固定SHAを正本とする。）


<!-- END BYTE-PRESERVED GPT REVIEW -->
