# H4 PR複数幅探索 R07：R06からの復旧継続

利用者の「前回探索が止まったところから続きを行って」に従い、修正版44fadf8b8と限定追加修正82dd08d6cを使う新runを固定した。旧R06のSTOP・raw・失敗stageは変更しない。状態は[STATUS](STATUS.json)、継続方法は[HANDOFF](HANDOFF.md)。本書は開始時点の報告で、探索後の幅別最安の最終集計ではない。

## 引継ぎとSOURCE

[RESUME_INPUT_CHECK](RESUME_INPUT_CHECK.json)のとおり、R06の239点・36 family、READY入力5構成、既存価格registry、DP cache約1.1 GBを引き継いだ。旧完了Q3は元のExecutorのidentity/route/contract/価格条件で受け入れ、同等jobの二重起動を避ける。API前に止まったPauli8件・DF5件は新SOURCEの復旧対象とした。旧markerの削除や無条件の科学retryは行わない。

H4 DF10、元H/state/snapshot、h、target、decoder、law、loss、exact τを保持。Pauliの要求幅はJSON数値0.1/0.2、DFは0.1/0.05、L_D=1〜9、元mapperの異なるκ scheduleと自動shot探索を継続する。0.2を文字列1/5に置換しない。実幅T/qは候補configurationに残す。[SOURCE](SOURCE.json)と[ROUND](ROUND.json)が条件と実行設定の正本。

## 再開前に修正した適用漏れ

- 保存catalog-onlyのκ候補に対する価格quoteが、候補自身のconfigurationを送らず、registryの基準構成を使う経路を修正した。全quoteを候補IDに照合する。
- 26軸が保存表で省略されている場合は、その候補自身の採択済みMODEL quoteが全保存総額をexactに再現するときだけ価格入力を復元する。競合する既存軸価格は置換しない。新しいモデルfitや保証の採択ではない。
- DFの固定entryをpacketごとにそのまま使って保存recordを検査し、今回の候補のphysical keyだけをmergeする。無関係なkeyの競合を全候補へ広げない。対象keyで異なるrecordがあれば、元の完全一致STOPを維持する。元record・scalar・provider数値処理は変更しない。
- 新runのprivate event列とDP checkpointはローカル保全し、公開対象から除く。R06の約118 MB/rawを含む履歴は保全したまま参照する。ローカルのみの固定SOURCEは、外部取得可能と扱わない。

人工25件PASS。前回の接続修正の72件/元worker metadata検査も再利用した。新たな実信号・sampling・compile・Q3の開始前に[PRELAUNCH_ACCEPTANCE](PRELAUNCH_ACCEPTANCE.json)を通した。

|方式|今回の開始時確認済み最安、条件付きRZ|出所|
|---|---:|---|
|PR-Pauli|410,199,237|固定C catalogのLD5/要求幅0.1、候補自身の26軸quoteと全保存総額一致|
|PR-DF|117575550907/126 ≈ 933,139,292.91|R06の有効な完了Q3、LD7/要求幅0.1、元jobをidentity/価格条件で受入|

前回の表示上のPauli最安437,758,513は、価格不足で除外された保存最安を含まない範囲だった。今回はその欠落を修正した。開始後の改善、規則採択点、未採択の有効な先読み結果は別々に保存する。

## 並列実行・停止

観測時は64 CPU、利用可能メモリ約488 GiB、関連科学workerなし。queue20 slot×DP3 workerと既存の選択済みnative準備/compileを使う。初期のQ3と欠測準備は重ね、準備後も同じqueue/累積予算へ戻す。prefix checkpointはm5/m11、cache HIT/新規Q3/完全一致再利用を区別する。実H4の速度倍率は未測定。

SOURCE固定後はtmuxで通常CLIを1回起動する。wrapperのflockは本番入口で重複起動exit75を確認済み。科学的・技術的UNKNOWNと旧失敗markerを保全し、自己retry loopは持たない。終了後の幅別最安・SOURCE・実時間・資源・価格出所の集計は同runの結果から回収する。計算中のチャット終了は探索の停止を意味しない。
