# 研究ノート

このディレクトリには、研究実装を進めた時点の方針、判断、検証結果および
未解決事項を日付順に記録する。後から「なぜこの実装になったか」「その時点で
何が確認済みだったか」を、commitと検証コマンドまで含めて追跡できるようにする。

## 資料としての位置付け

研究ノートは時点ごとの作業記録であり、現行仕様の正本ではない。

- 現在の研究方針と評価条件：`docs/research/`の主資料
- API、回路scopeおよび数学的規約：各実装文書
- 再現可能性と保証status：`VALIDATION_STATUS.md`と
  `artifacts/validation_manifest.json`
- 実際の数値結果：fingerprintと生成条件を持つmachine-readable artifact

過去のノートと現行仕様が異なる場合は現行仕様を優先し、変更理由を新しい日付の
ノートに追記する。過去の記録を現在の理解に合わせて黙って書き換えない。

## 記録規則

1. ファイル名は日本時間の日付に対応する `YYYY-MM-DD.md` とする。
2. 同じ日に複数回更新する場合は、ファイル内に `HH:MM JST` の節を追加する。
3. 実装を記録するときは、基準commit、対象scope、採用方針と採用しなかった範囲を
   明記する。
4. 検証結果は実行コマンド、pass/fail/skip/warning数、既知の環境制約を記録する。
5. 結果には `確認済み`、`部分確認`、`未確認`、`blocked` のいずれかを付ける。
6. 科学的な結論は、対応するartifactとfingerprintがない限り、実装能力の確認と
   区別する。
7. 失敗や方針変更も削除せず、後続ノートから訂正内容を参照する。

新しい記録は[テンプレート](テンプレート.md)を複製して作成する。

## 時系列索引

| 日付 | 主題 | 基準commit | 到達点 | 次の主要課題 |
|---|---|---|---|---|
| [2026-08-26](2026-08-26.md) | H5 connected-cluster系サイズ検証と回路cost modelの区切り | `e07a5e6` + dirty worktree | H5、rank 9、$L_D=4$、$K=2$、$L=4,6,8$でpaired K1--K3最大1.665%。独立calibration/holdoutは最大3.776%、予測半幅1.459%で5%/2%基準を通過 | cost providerをRPE shot・誤差/失敗確率配分へ接続。新compiler・$L>8$・不通過条件だけ追加holdout |
| [2026-08-24](2026-08-24.md) | 階層compiled-cost model、$K=2$次数条件付き再検証、connected-cluster運用推定と軽量化 | `e07a5e6` | 固定DF snapshotの$L=4,6,8$ holdoutでK1--K3運用推定は全metric最大2.936%。点誤差5%内だが95%診断5.724%の留保。calibration/prediction/transfer分離と厳密key cacheを実装 | 別$L_D$・short-step・compiler/coupling条件への移送と角度不変性検証 |
| [2026-08-25](2026-08-25.md) | 複数order-2、独立K4、controlled $q=8$の追加・follow-up batch | dirty worktree | paired複数order-2は最大1.679%。$L_D=6$のK1--K4 paired $L=8$は4.008%。controlled $q=8$は0.0529%。全job完走・validator通過 | 系サイズ方向の独立holdoutで運用規則を確認 |
| [2026-08-23](2026-08-23.md) | ランダム回路加法モデルとRTE境界補正の高統計検証 | `e07a5e6` | 1000標本・独立2 seedでcount/sizeのpair-only残差を確認。same/different二分類が別seed pair holdoutを最大0.849%で予測 | count/sizeの$\mu_3$または$L=8$、$L_D,K$、controlled・compiler条件のholdout検証 |
| [2026-08-19](2026-08-19.md) | 論文Eq. (D6)によるPF摂動係数の再検証 | `8418192` | H4全$L_D$とH2--H5の支配位相比較を通過し、H6のD6係数をstate-actionで算出 | GPU経路をH8/H10で確認し、H12の候補$L_D$ごとにD6係数を決定 |
| [2026-08-18](2026-08-18.md) | finite-RTEとPF・摂動・QPE分枝誤差の検証 | `8fdc6b3` | H4全$L_D$の単一位相条件、H2--H5のdense比較、H6のstate-action係数までlocal確認 | GPU経路をH8/H10で確認し、H12の候補$L_D$ごとに$C$を決定 |
