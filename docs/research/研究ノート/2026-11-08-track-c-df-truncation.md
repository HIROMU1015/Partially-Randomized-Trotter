# 2026-11-08 研究ノート：Track C DFの3方式比較

## 記録情報

| 項目 | 内容 |
|---|---|
| 日付・時刻 | 2026-11-08 15:13 JST 科学検証完了 |
| 基準commit | c5a3c102199fe97f4d5c21837c059ae2684f5ef2 |
| 科学SOURCE履歴tip | a4e32484254a090eab6d221776a50a1cfd9f0f17、bundleで保全 |
| branch | codex/track-c-df-truncated-pf-20261108 |
| 記録者 | Codex primary agent |
| 対象scope | 保存DF H2/H4/H6、元H ground、Full/Truncated/Closed P3 |
| 状態 | 確認済み：固定保存H・oracle ground・理想native・独立drawsの条件付き |

## この時点の目的

同じ精度・成功条件の18roundで3方式の総RZを比較する。PR優越を前提にしない。

## 採用した方針と理由

既存の保存積分・DF、PF family、G7、compiler A、interval/shot/decoderを再利用。
切り捨ては元H stateの信号を直接評価し、元H E0をdecoderへ渡さない。
候補選択と新しい価格holdoutを分離し、旧H4の強baselineを保全。

## 実装・解析内容

新adapter4 module、runner、人工tests。DF prefix I0/I1、小係数L1 budget、同じh menu、分子別PR split/BCH/QR、有限の作用順audit。
SOURCE・raw・技術修正・失敗記録はREPORTに対応付けた。旧Track・固定契約・STOPは変更しない。

## 検証結果

| 状態 | 確認内容 | 根拠 | 結果 |
|---|---|---|---|
| 確認済み | 全9方式の18round資格・N・decoder | FINAL_CHECK、ordering/FINAL・VALIDATION | 精度・失敗予算を共通化 |
| 確認済み | 入力replay・元copy | INPUT_REPLAY_CHECK、SOURCE | 数値/IRが完全一致、private原本保持 |
| 確認済み | 独立実装経路144check・人工8tests | raw/check log | source/native X/Yとmean branch照合 |
| 確認済み | 新価格holdout3712本 | ordering/HOLDOUT・FINAL | trainingとのseed交差0、cell共分散を保持 |

## 得られた知見

有限menuの総RZはH2 PR約5.00955e9、H4 Truncated約3.33777e11、H6 Truncated約1.22291e12が最小。H2の非Fullは極小係数除去のみで、fragment切り捨てはscreen外。H4/H6ではtruncationとPFのsigned biasの相殺を含む。

## 問題・制約・保留事項

oracle ground、固定保存行列、理想gate、独立drawsという前提。有限3分子から一般scalingやglobal optimumを主張しない。技術修正と48点viewの出力名重複の修復は元証拠・policy・holdoutを保全して確認した。

## 次に行うこと

ユーザー：成果の採否・後続研究の要否を判断。Codexの科学計算・再解析はmandatory STOP済み。公開梱包・commit・byte照合のみ実施。

## 関連資料

- [REPORT](../track_c_df_truncation_20261108/REPORT.md)
- [DISCOVERY](../track_c_df_truncation_20261108/DISCOVERY.md)
- [SOURCE](../../../artifacts/track_c_df_truncation/2026-11-08/SOURCE.json)
- [FINAL_CHECK](../../../artifacts/track_c_df_truncation/2026-11-08/FINAL_CHECK.json)
