# 段階Iの状態と残る条件

**SOURCE_IMPLEMENTATION_PREPARED_UNEXECUTED**：型、serializer、係数代数、RTE接続、router、sequence policy、wrapper、独立参照、literal価格、interval score／selector、F1 gate/runner、test定義を新規SOURCEとして準備し、静的検査と公開保全を行う段階である。runtimeの正しさは未確認。

今回の静的調査で、既存SOURCEの変更やDF型偽装を要する未解消integration blockerは認めなかった。必要な参照helperを固定版から確認し、限定portへ分離した。これは動的integrationの成功を意味しない。

| 残る条件 | 担当・次の承認 |
|---|---|
| fixture／catalog materialize、全armのLとcoverage、実input SHA | ユーザーがB範囲を明示承認、Codexが承認後生成しSTOP |
| runtime API、native Gaussian convention、relative phase、support completion、Y符号 | BまたはF1のどの登録callで確認するかを契約で固定。今回は未実行 |
| environment closure、mpmath interval内部API／sqrt/log／ceil資格 | 専用envと必要callをBで承認。未資格はfallback、ceil不明はSTOP |
| literal price provider、recipe生成のcall範囲 | Bのbuild有無を明示判断。未取得価格を0としない |
| 最終SOURCE closure／input／env／case ID／output容量／truncation API値 | B後の最終契約とreceiptでfreeze、ユーザーがF1を明示承認 |
| 数値testの登録・重複防止 | 最終callに対応付ける。現在のtest一括実行は未承認 |
| compile workloadの実資源・並列化構成 | 最終F1/F2 run前に準備し、変更SOURCEをfreeze。今回はworker実行なし |
| F2全trajectory実費用・統計／bias資格 | F1完了STOP後にユーザー/GPT確認、別明示承認 |

`SOURCE_INTEGRATION_BLOCKED` や `SOURCE_PARTIAL_UNEXECUTED` は、今後具体的な不足が判明すればその範囲で採用する。現状をREADY_FOR_F1_APPROVAL、F1_VALIDATED、SCIENTIFIC_GO、PERFORMANCE_GOとは表示しない。候補SOURCEが準備されたことと、input/environment/科学callが実行承認可能にbindingされたことを区別する。

独立新規性は未確立、mixed最良は必要条件ではない。旧STOPやgrantを再利用せず、公開確認後mandatory STOP。次の担当はユーザーである。
