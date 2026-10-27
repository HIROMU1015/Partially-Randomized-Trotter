# SERVER_A 保全・SOURCE・Git参照監査

2026-10-27 JST。科学計算・数値再解析ではなく、Git/ref・file metadata・bytes保全の技術監査。既存原本を変更せず、コピー書込みは本人専用非公開領域のみ。

| 確認対象 | 結果 |
| --- | --- |
| 現在snapshot作業コピー | 42（研究40、管理2） |
| HEADを保持する現存ref | 全42件で確認 |
| 旧SOURCE宣言 | 22宣言を再利用、13固有commit全てobject存在・ref保持 |
| 旧36件からのHEAD/status/ignored差分 | 各rowで保持。候補ではtracked clean、重要な新変化なし、0041旧未追跡5件はcopy後に更新。新commitの最新5件を実bytes/Git blob・保持refで追加確認 |
| 非破壊の追加copy | 237 files / 130,363,450 bytes |
| source安定size/mtime/inode・SHA一致 | 受入れた全237copyで確認 |
| 元原本へのhardlink・移動・上書き | 0 |
| 保全copyの障害域 | 同diskの別directory。独立障害域backupではない |
| 科学payloadの数値解釈・runner実行 | 0 |

## 追加copy内訳

| ID | files | bytes | 判断への影響 |
| --- | ---: | ---: | --- |
| SERVER_A-WT-0002 | 220 | 4714189 | 専用venvの固定参照があるためKEEPは継続 |
| SERVER_A-WT-0006 | 1 | 3279 | 固有ignored入力/ログの保全gateを閉じ、owner判断へ |
| SERVER_A-WT-0007 | 1 | 3278 | 固有ignored入力/ログの保全gateを閉じ、owner判断へ |
| SERVER_A-WT-0008 | 1 | 3279 | 固有ignored入力/ログの保全gateを閉じ、owner判断へ |
| SERVER_A-WT-0013 | 4 | 54496 | 固定実行providerのKEEPは継続 |
| SERVER_A-WT-0019 | 5 | 118964065 | 固有ignored入力/ログの保全gateを閉じ、owner判断へ |
| SERVER_A-WT-0041 | 5 | 6620864 | 古いsnapshot5件を保持。更新後の最新5件はGit blobと実bytesを個別照合し、保持refから復旧可能。用途終了は別判断 |

中身の数値計算・pickle/array読込み・科学module importは行わず、元bytesをストリームコピー・SHA照合した。ログ・checkpoint・入力・sourceの具体的な非公開file名や配置は公開しない。元のSOURCE/grant/ledger/marker/STOPのbytes・条件は変更していない。

## 保持refと復旧

各IDの完全HEADとretaining refは詳細原本にある。detached0034は保持0030/handoff refを用いる。未公開commitもlocal refと共通objectsを残せばcheckoutの解除とGit履歴喪失を分離できる。SHA文字列の記録だけで保持済みとは扱っていない。branch削除・gc・shared Git移設は未実施かつ計画外。

13固有科学SOURCE commitは全て既存refに包含されることをread-onlyで確認した。SOURCE commitとそのmanifestを含むcheckout HEAD、同じHEADの複数作業場所、private raw/custodyは別に扱う。同HEADだけでraw保全済みとはしていない。

0010の外部custody、0034のbundle/原本、H2/H4/H6やrun02〜05の既存raw/control/STOP、共有venv/alternate、未公開SOURCEは既存索引・役割を再利用して保護した。原本全件再探索・hash、全Git履歴再計算、archive再展開、復旧実験や科学再実行はしていない。

## 固定保持とscope限界

0026・0029・0031の固定実行pathを再利用しKEEPを継続。WP11のvenvを新H46計画が指定する新しい具体的参照も保持理由として追加した。保存copyだけでruntime可搬性や固定path契約の解除を認定しない。main/common rootと独立clone rootは削除候補に含めない。

検査時に本人可視processが候補pathを利用していなかったことは、未来・別Codex・他host・別namespaceの不使用証明ではない。既知の他環境不足は0015へ具体化し、候補群には研究単位のowner利用終了判断を残す。LOCALの利用終了/削除承認を転用しない。

詳細監査には実path、元branch、HEAD、retaining refs、追加copy SHA/元相対階層/復旧先、観測metadataを記録し、Publicコピーには内部path・host/IP・個人名・private配置・機密本文・process識別子を含めない。今回の公開はこの新しい5資料のみ。
