# PR-Pauli LD4・LD6：Aのdescriptor入口から通常intake・total/Q3入力へ

**技術接続はPASS、実信号は両構成とも全13 roundが未取得。新規実signal・compile・Q3各0件。** 実装SOURCE `33e324b43690caeaa353ecdb9e3d42f3139198b6`。新5＋回帰21＝26技術test PASS。B2固定報告`d7a976a62`のLD4/6保留を、SOURCEの版と不足科学データに分けて解消する。

## 先に比較した公開版

B2の取込み版`bd42dd1e9`と公開済み汎用取込み`ff9d25d62`は、`rpe_pauli_signal_intake.py`・`rpe_c_interfaces.py`がbyte同一だった。したがってff9への差替えだけでは、新A descriptor入口のpacketは通らない。どちらもAの組立てSOURCE `93a5ef3b9`を対象とし、後続のB `830cf9bd8`でも対応は93a/4e5の2版までだった。

不足はprefixではなく**A入口の実装版**。A `6e4b6bd6a491815c2a3cb8f809cbfc0240b60fdb`の`--tail-descriptor`経路を、既存のSOURCE検査へ追加した。module/runnerに加えて`pauli_split_descriptor.py`も固定版と照合する。新しい分割ごとのallowlistは作っていない。A `ebab42097`はこの3fileが6e4とbyte同一のため同じ実装版として受入可能で、出力packetや各roundの数値SOURCEは元のlocatorを保持する。

B2 `common_metadata`にも `assembly_source != A.ENTRY_SOURCE`（93aだけ）の判定が残る。[最小patch](B2_COMMON_METADATA_VERSION.patch)で、通常accept_savedが検査した固定版集合`A.ENTRY_SOURCES`への所属確認へ変更する。patchなしの拒否と、patchありの正常接続を実際のB2関数で再現した。B2の作業checkout・branchは変更していない。[SOURCE_COMPARISON.json](SOURCE_COMPARISON.json)に比較を保存。

## 実データで確認した範囲

| prefix | 元configuration | total configuration（metadata生成のみ） | A新入口の計画 |
|---|---|---|---|
| LD4 | 8bf6eb88c735daf2499c0ecffd07cf66e86d66630c0cbba4db252ee704918012 | 5a659a71154f9bea21711c8a931ede86c74e33090c42b4fcc1ad413ebdaaa19f | m0..m12 REQUEST、reuse/unsupportedなし |
| LD6 | 6b6e0f174b3f2b1bbf1a95e563921e6be4ff87f4b41e9d68c0e44558a1c66142 | 5b507c2ff0cdbcde7d78b68413bc8278f4fedc03e15a047ac6a7da48722f10a3 | 同上 |

B2 d7a976a62の固定configurationとdescriptor envelope、P0辞書（LD4 c35495565／LD6 f3f47f7f9）、SOURCE/hash/bit順/λ/scalar/H/state/lawを読取照合した。canonical validateとtotal createは元configurationのままPASS。

Aの独立checkoutを6e4に固定し、実CLIの`--plan --config ... --tail-descriptor ...`をLD4/6で実行した。**planはmetadataだけ**で、新しい正規化子・平均・参照値を計算していない。結果は[A_PLAN_LD4](A_PLAN_LD4/PLAN.json)、[A_PLAN_LD6](A_PLAN_LD6/PLAN.json)。B2 signal branchの762286b05とd7のroute registry、関連履歴にも、この2構成の完備signal packetは見つからなかった。既存packetの再取得はせず、未取得を[READINESS.json](READINESS.json)に保持した。

## 人工fixtureで確認した範囲

保存configurationと辞書をそのまま使い、raw X/Yだけ人工値と明示した13carrierを一時Git内に作成。Aの固定`packet()`、P1の固定`job()`を再利用し、通常の`accept_saved`→C.signal_input/provider_binding_check→T.create/bind_signal→B2の実`common_case`・版判定修正後`common_metadata`→既存K.prepareを通した。K.prepareはkernel側の元configと、total view側の両方で確認した。

fixtureのX/Y shot列は2/3各13要素で、非対称軸・全round保持を調べる技術入力だけ。科学的な配分採択ではない。元packet、raw/normalization、target、元/total ID、law、SOURCE、UNQUALIFIEDとnull N/MSE/Gを保全。binomial/Q3/MC/cost/selectはテストで呼出し禁止にした。fixtureのcommitは独立temporary Git内だけで、実signal registry/catalogに登録せず終了時に削除した。

変更したdescriptor依存file・未知の組立てSOURCE・別configuration・carrier・law・欠測roundは拒否。旧93a/4e5の保存LD1/2/3/5も回帰PASS。[TESTS.txt](TESTS.txt)と`tests/check_pauli_ld4_ld6_intake.py`が正本。

## 残る科学データと境界

- **両構成のm0〜m12の実mean（各roundのraw X/Y）が必要**。今回の人工完備packetを実signalとして使わない。A新planはSUPPORTEDだが、これは計算済み/数値gate PASSを意味しない。
- Cのmetadata物理bridgeは人工入力でPASSしたが、26軸の価格較正・価格資格は今回未評価。B2/D2の別担当範囲を保つ。
- 実signal取得後、通常intakeで原carrierを照合しtotal viewとQ3入力を組み立てられる。実Q3・shot探索・MSE/N/Gは今回未実行で、別の有限実行許可が必要。
- 厳密ΔPF=1/5のA `ACTUAL_STEP_OR_T_NOT_BINARY64_EXACT`とC価格UNKNOWNは従来どおり別。今回のLD4/6固定schedule（ΔPF=h、κ=2）はその未対応に該当しなかった。

API・B2側の固定版設定・Aへのdescriptor渡し方は[HANDOFF.md](HANDOFF.md)。ロードマップ本体は更新せず、入力不足を「Bのprefix拒否」ではなく「更新版intakeの利用＋実signal全round待ち」として統合担当へ渡す。この接続作業はmandatory STOP。別途承認済みの決定論PF探索には変更・停止・再投入を行っていない。
