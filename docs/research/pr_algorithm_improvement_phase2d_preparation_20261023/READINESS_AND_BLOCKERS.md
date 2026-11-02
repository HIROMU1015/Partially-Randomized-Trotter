# 準備状態と承認前blocker

総合状態：**`PREPARATION_INCOMPLETE`**。input binding状態：**`INPUT_SPECIFICATION_PENDING`**。これは科学的否定結果でも性能GOでもない。

| status | 意味 | 現在 |
|---|---|---|
| READY_FOR_F1_APPROVAL | SOURCE/input/件数/env/閾値がfreezeされ、承認可否を判断可能 | いいえ |
| PREPARATION_INCOMPLETE | 契約案はあるが新prototypeとbindingが未完了 | **該当** |
| INPUT_SPECIFICATION_PENDING | 候補数値仕様の採用、生成物とhash、Lを未binding | **該当** |
| CONTRACT_INCONSISTENT | 契約内に未解消の矛盾 | 管理verifierの範囲では未検出。科学整合を証明したとはしない |
| SOURCE_OR_REFERENCE_UNAVAILABLE | 必須SOURCE／reference不足で実行できない | 将来SOURCE closureは未成立。最新版private review原本も未取得だが、指示の代替方針でP0は進行可能 |

## 実行承認前の必要事項

| blocker | 解消する内容 | 担当・必要承認 |
|---|---|---|
| B1 新SOURCE未実装 | mixed router、sequence bridge、wrapper adapter、独立reference、literal/interval provider、runner、tests定義 | ユーザーがI承認を判断、Codexが承認後だけ実装 |
| B2 source版差 | 2978/f29 policyと起点版のport／接続、support helperを含むclosure、全arm公平性 | Codex、I範囲。SOURCE commit/hashを提出 |
| B3 input binding | 静的候補の採用、2input case、4arm maps、Lとcoverage、exact serialization/hash | ユーザーがBの限定科学生成を明示承認、Codexが承認後生成しSTOP |
| B4 env/precision | Python patch・transitive wheels・BLAS、90/128reference、outward interval/ceil資格 | Codex、I/Bの境界を契約に明示。未承認import／検査をしない |
| B5 numeric contract | 候補閾値、truncation_tolerance API値、finite-bit／bias予算の割当と表示範囲 | Codexが静的案を更新、ユーザーが採択 |
| B6 price provider | actual literal recipe由来RZ数／support/full／phase reserveの値とhash | Codex、承認されたbinding範囲。compiled予測とはしない |
| B7 call/output exactness | L、E、L²、Q+2、unique IR misses、output bytes、registered case IDを正確整数へ解決 | Codex、入力binding後。F1/F2それぞれ承認前 |
| B8 execution authorization | final SOURCE/input/contract台帳、fresh output、one-shot/STOPを明示承認 | ユーザー。今回なし |

private review原本の未取得は記録したが、ユーザー指示に採択済み方針がありP0のblockerにはしない。原本の無断公開や旧Phase2Bを最新版に読み替えることはしない。

## F2に残す条件

F1各gate PASS、意味のある構成差、原本保全・STOPを確認してからユーザー/GPTがF2可否を判断する。F2 q/Kの再binding、全trajectory exact counts、full-wrapper費用、statistical qualification、physical biasの表示資格、classical費用を別承認する。L²が過大でもMCへ変更しない。mixedが選ばれない結果も保持する。

P0の成果は「何を固定すれば承認判断できるかを具体化した契約案」。新SOURCEやinputが無い事実を架空hash・推測件数で隠してREADYにしない。次の担当はユーザーであり、I/B/F1の範囲を別途明示承認するかを決める。Codexは今回の公開・管理照合後に停止する。
