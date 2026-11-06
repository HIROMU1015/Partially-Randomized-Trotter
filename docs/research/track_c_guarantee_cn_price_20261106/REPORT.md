# Track C：保証検証AとC-N価格比較B

2026-11-06 JST、SERVER 2 `GPUA100-gleap02`。
基準commit：`3258a89f225d9530fba38e9d2343bc0165dd6d7d`。
branch：`codex/track-c-guarantee-cn-price-20261106`。

**Aは固定保存行列モデルで全18roundの十分条件が閉じた。元Hと保存行列表現の厳密包含まで含む無条件保証は未資格。Bは予定39セル・2,307標本を完了し、S2の全標本でfullと強baselineの価格が一致した。長roundの価格移送と全RPEの勝者はUNDETERMINED。**

| stream | 実施内容・結果 | 限界 |
|---|---|---|
| [A：保証検証](A_REPORT.md) | SOURCE basis/diagonal/finite-RTE mean、数値disk、保存groundの誤差、aliasを構造・区間/前進誤差上界で接続。全36条件で正半径・正統計予算 | 固定保存HamiltonianのHermitian行列表現・理想native gate・独立bitsのモデル。元H表現の新しい包含証明ではない |
| [B：C-N価格](B_REPORT.md) | S2 LD3/Full Y4 LD12、q1/2較正・独立q8 holdout、参照A。S2の2,304 paired標本は全4指標で差0 | q8最大点誤差6.09%、予測相対95%半幅最大7.26%。終端q131072の価格保証なし |

共通物理条件はLinear H4、隣接1.00 Å、STO-3G、DF12、LD3/12、元Hの厳密ground供給仮定、状態準備費除外。
h=.02、18round、delta0=.02、epsilon_E=1.5936e-4 Ha、alpha_total=.05、beta_total=.4 radを維持した。
compiler Aと旧SOURCEは変更せず、SCF・DF fit・ground solve・H4入力照合・旧44ケース・旧q1 suiteの再実行は0件。

A/Bは独立した出力領域・計算プロセスで並行実行した。Aの資格がBを停止させたり、Bの価格がAの保証を正当化したりする扱いはしていない。
CPU/RAMの恒常的上限を新設せず、BはOS affinityに基づく64 worker、Aの独立generator検証は14 task/workerを使用した。
共有設定・パッケージ・他ユーザーのプロセスは変更していない。

## shot数の扱い

Bのprimaryは指示どおり、保存された条件付きshot数S2 **8,598**、Full Y4 **6,912**で固定した。
Aの新しい保守的な数値証拠ではS2 **8,598**、Full Y4 **6,914**が十分数となる。
Full Y4はm17の各軸が192から193へ1ずつ増えた。Bのprimary値を無断で置き換えていない。
[接続結果](../../../artifacts/track_c_guarantee_cn_price/2026-11-06/JOINT_STATUS.json)に、Aのshotで再重み付けした副次的な価格点モデルも分記した。これは新価格取得ではなく、長qの未資格を解消するものでもない。

## 保全と成果物

[成果物入口](../../../artifacts/track_c_guarantee_cn_price/2026-11-06/README.md)からA/Bの数値、raw bundle、SOURCE/input、環境、STOPを辿る。
Aの初回実装では正規化等の微小な積の項を上界式へ残す補正が必要となった。初回SOURCE・結果・STOPを保全し、別`verified/`領域で同じ条件を補正評価した。最終採用はverified側である。
Bの価格取得は1 cohortだけ。保存rawを用いる後処理で絶対MC標準誤差と直接域の小計を補足した。追加標本・追加価格cellはない。
いずれも開発中のlocal実行であり、immutable CI・外部再現としては報告しない。

旧run01、旧SOURCE/raw/grant/ledger/marker/STOPは不変更。
C-S/C-P追加実験、T合成、q>8の新価格compile、全価格gridへの拡張は0件。
**承認範囲を完了しmandatory STOP。science workerはすべて終了済み。**

次の担当：**ユーザー**が元H表現の包含検証の要否と、C-Nをこの負結果で停止するか、精度を保つ最適化等へ問いを限定して継続するかを判断する。
**Codex**は次の対象について明示実行許可を受けた場合のみ実施する。追加計算・標本増加・価格gridの自動再開はしない。
