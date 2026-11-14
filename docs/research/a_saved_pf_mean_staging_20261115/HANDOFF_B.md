# A → B：段階packetと既存intakeの互換性

[REPORT](REPORT.md) §2／[B_INTERFACE_CHECK.json](B_INTERFACE_CHECK.json)。**Bへの新しい作業依頼はない。**

## 1. 確認した結果（B `9c0e12d87`を無変更のまま実行）

- 6候補のconfiguration IDは、Bのprovidersで再生成した値と一致した。
- 未完成の段階packetは、`T.bind_signal`が拒否する。
  - 理由は、single-source provenanceのcheckと、provider bindingの欠如である。
  - 全roundの揃いを求めるBの条件は、そのまま有効である。
- 人工の完成fixtureは受理された。取得済みroundは変化しなかった。
  - これにより、取得済みS2 roundのidentity・law・物理context・provider bindingが、Bの既存checkを通る形であることを確認した。
- Bの回帰test 13件はPASSした。

## 2. 全13roundが揃った後の形（Bの変更は不要）

**BのsaveされたpacketのadapterはSOURCEを1つしか許さない。** packet内のすべてのmeanの`source`が、packet全体の`source`と一致していなければならない。そのため、A（またはC）は次のようにして1つのpacketを作る。

1. 保存S2 meanを、`C.execute(..., saved=records)`のsaved recordとして渡す。
   - recordの`source`は、今回のstagingと同じく、その実行のSOURCEにする。
   - 元fileの情報は`provider_binding.input_source`に残す。
2. 新しいmean（S3、承認後）は、同じ実行で評価する。
3. その結果を、Bの既存intakeに渡す：`T.create`（固定SOURCEのconfiguration）→`T.bind_signal`（固定SOURCEのpacket）。

**round単位で別のSOURCEを持たせる場合**
- Bのbaseline adapterに`implementation_sources`の経路を追加する必要がある（最小の変更）。
- ただし上の手順で足りるので、今は依頼しない。

## 3. 渡さないもの

- 未完成の段階packetは、risk・Nの入力にしない。
- S2Lのmeanは昇格させない。
- S3を0や推定値で埋めない。
