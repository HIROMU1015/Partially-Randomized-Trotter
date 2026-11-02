# 固定4候補の最小科学実行契約案

状態：**DRAFT_NOT_EXECUTABLE**。機械可読版は[execution_contract_DRAFT.json](execution_contract_DRAFT.json)。科学実行の承認・正式政策・live grantはない。以下は将来の計画であり、今回起動しない。

## Gate 0：保存証拠

固定SOURCEは旧監査366b2fd2…、準備/受領cc7404ed…、元export e3531956…。新kernelは4ec26921…とprovider c31ef424…、旧providerは217c971d…。今回の技術codeは契約案のimplementation_source_commitとfile hashesに固定し、将来のscience runner SOURCEとは区別する。

150受領file、provider278 file、PR67 task/134軸、旧2候補原本control/resultを静的照合済み。個別guard receiptの今回のexport hashを、元accepted taskが当時封印したhashと誤認しない。保存compiled resultを再実行したPASSではない。

senderのhistorical runtime seal2012 path中、sender観測2011一致・liblzma1差が残る。Python binary sealも旧新で異なる。共通runtimeは実際に使用するcompiler/providerとloaded closureを新たに封印する。旧runtime差の影響を無根拠に許容せず、新旧環境全体の一致も要求しない。

## Gate 1：採択に必要な判断

ユーザーがnative提案（adjacent QR＋signed-piece compose、compiler guard両方式適用）、旧A由来の共通適応規則、共通reference案を判断する。4候補・物理条件・数値閾値は[設計](NATIVE_DESIGN.md)と[schema](ALLOWANCE_SCHEMA.md)にある。

correctnessはnative occupation/wrapper strict <1e-9、compiled saved-state vector strict <1e-7を提案する。phase alignment/state fitを禁止し、失敗時STOP。これらはshot allowanceではない。実際のSOURCE/runtime/input、correctness gateの適用範囲と新系列の識別を採択時に固定する。

## Gate 2：欠測の最小回収順

| 優先 | 対象 | 保存資料で分かったこと | 次の必要範囲 |
|---|---|---|---|
| 1 | reference対応 | 旧combined診断は保存、新PR referenceは別来歴 | 旧reference再利用の判断。新評価を自動要求しない |
| 2 | PR独立診断 | forward signal/log_B差あり | 2固定候補のpath-vector差、両path実work、全stage norm、corrected-signal closureだけをinstrument |
| 3 | Gaussian/common prep | 固定kernel・異なるprefix/hashを特定 | 同一rotation/blockからcommon prep/actionへ受入witness |
| 4 | Full correctness | control/result原本一致、該当workerにQPY/vectorなし | 新common Full2候補のnative/compiled受入。旧結果を無条件代用しない |
| 5 | PR保存C | 67 paired taskの来歴対応 | 同一kernel/prep/wrapper/runtime/event lawのwitnessで再利用判断 |

診断は必要な2候補・欠測項目に限定し、q/gridを再探索しない。診断用の新signal/vector評価は科学callであり別承認が必要。新sampling/compileはこの診断だけのためには要求しない。受領されていない全stage/vector証跡を保存値から捏造しない。

既存PR Cの再利用は、候補/H/state/native/prep/schedule/wrapper/compiler/runtime/event law/cohortのidentityとaccepted correctnessが揃う場合に限る。歴史policy名を新policy名へ書き換えてcheckを通さず、静的または新科学witnessを別receiptで記録する。witnessが不足する場合は影響する軸の追加検証だけを再提案し、自動で134軸を再compileしない。

新Full common nativeは旧Fullとprepが異なる。新compiled Cが必要となる可能性が高いが、保存された等価な受入回路が見つかれば再利用を先に評価する。2候補×cos/sinという対象範囲を示すことと、simulation/build/compileの正確な内部call数を宣言することは別である。現段階では各countをnullとし、実行承認用のbudgetとしない。

## Gate 3：共通評価

全実診断のschema受入、正式採択済み規則、identityが揃ってから、固定4候補の新u/margin、eligibility、必要shot N、paired RZ単価、shot込みG、S2/Y4順位を計算する。保存値を用いたこの算術も今回未承認。quantum shotそのものの実行を要求する計画ではない。empirical allowance/単価の不確実性とbias保証を分け、比較Cと原因を自動認定しない。

## Gate 4：one-shot・保存・STOP

将来の実行前に次を満たす。今回これらのliveファイルは発行しない。

1. 使用するscience SOURCE/input/loader/native/compiler/runtime、各kernelの正確なcall recipe・種類別budget、出力namespaceを封印する。
2. ユーザーの政策採択と科学実行承認を別receiptへ固定し、未使用run専用one-shot grantを結合する。
3. 最初の科学callより前にconsumed markerをatomicに作成する。各call前に自分のledgerへ予約し、例外・中断後も再利用しない。
4. fail・非有限・数値不一致・欠測・使用入力の破損ではSTOPし、科学retryを行わない。子processを回収し、partial結果をPASSにしない。
5. 実際に使うSOURCE/input/compiler/runtime/result/grant/ledger/markerだけを変更拒否対象にする。無関係なworktree更新や一時的PSI上昇だけで正常workerをkillしない。
6. 結果、runtime/入力/operation identity、診断trace、gate証拠、count ledgerを新系列へ保存する。旧SOURCE/results/grants/STOPを変更しない。
7. 契約範囲の技術検証後mandatory STOP。次campaign・詳細科学レビューを自動開始しない。

## 実行前に残る技術作業と承認

Codex：採択された仕様に合わせ、遅延recipeから実backendへ接続する別SOURCEを作る。scientific import/dispatch前のgrant・call予約、実runtime/input seal、exact budgetを実装・人工検査する。現在のmock adapterに `science_authorized=true` を渡して実行する経路はない。

ユーザー：native/allowance/referenceの政策、診断とC再利用の条件を判断する。その後、具体的SOURCEとexact budgetが揃ったone-shot範囲を明示承認する。現時点のnull countを含む契約案は即実行用の承認対象ではない。

今回のcampaignはこの設計・人工検証・公開で終了し、ここでmandatory STOPする。
