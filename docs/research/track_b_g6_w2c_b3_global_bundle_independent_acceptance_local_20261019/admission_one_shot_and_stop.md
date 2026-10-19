# Admission・one-shot・STOP

FAKE_ONLY.sqlite3をimmutable/ro URIで開きquick_check、scope/claim/binding/spent、保存historyとreport snapshotを照合し、読取前後SHA不変を確認。過去atomic操作の全履歴をsnapshot単独で証明しない。公開rawのSTOP/drain/予約境界と別fresh probeを分離する。

[新probe](guard_and_fake_store_probe.json)では32同時競合winner1/loser31、spent再使用拒否、commit-ACK例外後もSPENT、STOPを人工確認。実authority、real approval、電源断耐久性の証明ではない。

S失敗で後続P1を開始せず、P1失敗でS receipt/rawを保持。STOP以後新ENTERを拒否し、既permit in-flightだけをdrain。UNKNOWNと未ACK creditを返却せず、未確認CPUを0にしない。親arrival順が証拠で、全workerの物理例外時刻の順序ではない。

旧47/P07/P08/P11、旧18のsnapshot不足、旧V2 108の不完全66等は変更・再分類0。旧one-shot/STOP/結果も保全する。
