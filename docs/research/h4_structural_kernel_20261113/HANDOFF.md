# B/C HANDOFF：H4 K0入力参照STOP・技術修正版

**実K0は未開始、status UNKNOWN。K1は未実行、4値ともnull。** [REPORT](REPORT.md)／[run01 K0](run01/K0.json)／[STOP](run01/STOP.json)を正本とする。今回の結果を価格recordや物理kernelのFAIL/PASSとして扱わない。

## SOURCE

- C基点：`a637ed1b813dc3952339cb05f36beb79168ace23`。
- 初回SOURCE：`0f7d65d5f20de26c449b33c8da9d110188245808`。
- 技術修正版：`19b95d8b3c88063d2dbae4e5bb47455b52b00903`。科学再実行なし。
- branch：`codex/r-variable-compiled-price-design-20261111`を継続。Bのbranch/worktreeや共通APIを変更していない。

初回はB最新commit内にAのconfiguration fileがあると仮定し、Git読取りで停止した。正しいB公開SOURCE chainは次の通り。

```text
f0c4bb559d2a64dabe88e2eba31ff1dadad6fca8
  docs/research/abc_d_v2_integration_20261114/PUBLISHED_INTAKE.json
  source_files ->
030f3f548a35d98a88e62e92a35f132e0243a293
  docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json
  sha256 22a1bd1af8704006568c545eb8bcbce5bd476202ad59aaeaf79f50850815f943
```

修正後のmetadata testでこのchainと同canonical ID、m0/m6の全identity/law/physical candidateを確認した。候補IDの付替えはしていない。

## 科学scopeと未実施事項

固定H4 DF10/S2/prefix1/δ.1/K2/P0、同preparation hash、共通exactκ/M12。m0はq1/R2、m6はq64/R128。X/Y full-wrapper構造RZだけを対象とする。元C1のfull to_dict表現・binary64演算順・scalar/basis/phase/境界を維持する実装である。

初回は保存descriptor復元前に停止したため、今回の実preparation load、kernel構成、物理parity、実確率伝播、実E[S]はいずれも0。compiled価格・sampling・fit・N/Gは0。既存閉包論証の反証ではない。前段の有限状態十分条件と、未実施のK0実装検証を区別する。

新8技術method（人工7/固定metadata1）と旧10件PASS。人工fixtureでK0/K1が動くことは確認したが、実H4 kernelの資格にはしない。`KERNELS.npz`/`K1.json`は未生成。値ありのB price recordを供給しない。

## 修正実装の取込みと再開

新module `h4_structural_kernel.py`、runner、test、数値契約・実行計画が成果物。既存Accounting/MeanKernel/Streaming/providerは変更していない。Bは科学runnerを取り込む必要はなく、[COVERAGE](COVERAGE.json)の欠測とSOURCE/STOPだけを参照すればよい。

[RUN02_PLAN](RUN02_PLAN.json)に技術修正版SOURCE・新出力先・同じK0→PASS時だけK1四値という未実行範囲を固定した。UNKNOWN時STOPを維持するため、修正後も科学再実行はしていない。再開にはユーザーの明示指示が必要。新grant/証明書は不要で、run01 SOURCE/UNKNOWN/STOP/logは不変保持する。

将来のK1 quantityは`FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT`。実行できた場合でもE[C]ではなく、`FULL_WRAPPER_MEAN_COMPILED_PRICE`へ渡さない。実samplerとideal lawの距離、compiler残差、共有共分散・total CI・全Gは未資格のまま。

次：**ユーザー**が修正版の再実行を指示、**Codex C**が同scopeを実施してK0 PASS時のみK1へ進む、**Codex B**は価格欠測を保持する。今回の公開・remote照合後mandatory STOP。
