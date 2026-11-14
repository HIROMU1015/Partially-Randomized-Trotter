# 2026-11-15：5方式の物理候補と保存信号の対応、Full PFの全13round原本調査（照合のみ）

## 作業の範囲

- Aが、固定blobの照合とmetadataの処理だけを行った。
- 新しい信号・mean・固有系の算出、MSE・N・Gの評価、compileは0。

## 全13roundの信号がそろっている候補

- **受入可能（Bのintake PASS）**：PR-Pauli LD1/LD2、Full-H κ2、Truncated B0I p7・S2・0.1。
- **13 rawは保存済みだが条件付き**
  - PR-DFは、Bの版付きbridgeを待っている。
  - R448は比較候補ではない。

## Full PF

- 確認した資料の範囲では、正式な全13round packetが見つからない。これは「未実施」という意味ではない。S2・0.1は、Cの39round計画のうち未承認の部分である。
- shortlist（S2・16h/7、Y4・16h/3、M5・16h）の保存状況：
  - 現行SOURCEのdirect mean：m0〜m4・m12に4〜6 round。
  - Track Aの旧mean：m3。
  - 保存meanなし：m5〜m11（S2は加えてm2）。step spectrumは保存済みで、meanを計算する必要がある。
- Truncated B0I p7のshortlistも同じ構成である。
- Truncatedは、信号がそろっている候補（S2・0.1）と、価格を取得している候補（shortlist）が別である。

## 精度側と価格側のID

- 価格cellがあるroundでは、configuration ID・physical ID・identity/lawがすべて一致した。不一致は0件。

## 最小の追加計算（承認待ち）

- Full PF・B0Iの各shortlistについて、新しいround meanが22件。m3の旧meanを使わない場合は24件。

## 参照

[REPORT](../a_signal_candidate_identity_coverage_20261114/REPORT.md)。

mandatory STOP。
