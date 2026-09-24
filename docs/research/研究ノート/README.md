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
| [2026-09-25](2026-09-25.md) | M06-F fresh-32・coherent opt2再最適化 | `7753eb7` + dirty worktree | 51/51完了、両gate通過。点推定は`L_D=3`が4.858%低いが全区間重複 | 局所compiler精密化を止め、外部instance pilotを次のdiscriminatorとして検討 |
| [2026-09-24](2026-09-24.md) | M06-F all-r coherent opt2初期計算・解析 | `26aa95c` + dirty worktree | 36/36 cell完了。12 group中7通過、5 groupはRZ相対SE 2%基準でfresh-32待ち | 15 taskのfresh 32 trajectory拡張後にcoherent再最適化 |
| [2026-09-23](2026-09-23.md) | M06/L08、N07/P03、WP11限定判断統合 | `efa90e0` + dirty worktree | T4/T7を主軸、T1を範囲変更、T3を保留。次段はall-r coherent opt2再最適化 | opt2未測定$r=1,2,4,8,16$のCPU transpile・再最適化 |
| [2026-09-22](2026-09-22.md) | WP03、Gate S1、WP06-a/b、WP05-a/b/R、WP01-D/C07、G08/M08 | `efa90e0` + dirty worktree | M08の$q=16,32$直接holdoutはRZ最大3.286%で通過し、実測幅によるlocal再集計区間も分離。ただし25%移送区間は重なる | 主張範囲の見直しまたは外部条件での移送検証 |
| [2026-09-21](2026-09-21.md) | 研究方向screeningの実行gateとWP00/WP02/WP01-S/WP04 | `efa90e0` + dirty worktree | $L_D=0$をscreen out。WP04で公平な配分改善後の決定論endpoint差は4.96%へ縮み、5%・25%区間とも重なり未決定。主要因は$\beta$、次いで$\alpha$再配分 | WP03でPF係数選択感度を評価 |
| [2026-09-20](2026-09-20.md) | 4段RPE分枝復元、目標round診断、$\delta$/round別scheduleと中央RTE cost検証 | `2bf3116` + dirty worktree | H4の固定長round設定を棄却し、3個の$\delta$に行列検査を通るscheduleを構成。局所角度・$L=8,16,32$検証後の中央RTE proxyは0.02を全6指標で最小とした | $\delta=0.02$と0.01の制御付きpartial-$S_2$反復・Hadamard 1 shot costを検証 |
| [2026-09-18](2026-09-18.md) | 暫定配分を使った限定4段集計と新配分の短段失敗率 | `2bf3116` + dirty worktree | H4固定条件の$q=1,2,4,8$で1,572 shot、RZ数$3.2673961\times10^7$、8軸$\alpha$和0.05。$q=1,2,4$の厳密二項座標失敗率$2.2246\times10^{-4}$ | $q=8$物理信号、branch復元、必要全round・最終コストを別途検証 |
| [2026-09-01](2026-09-01.md) | RPE短段の信号・shot・cost接続、失敗率、$q=8$代理モデル、配分感度 | `2bf3116`に至る前のdirty worktree | 配分感度から$\beta=(0.02,0.02,0.36)$と重み付き$\alpha$を固定条件の暫定入力に選択 | 限定4段集計（2026-09-18に実施） |
| [2026-08-26](2026-08-26.md) | H5 connected-cluster系サイズ検証と回路cost modelの区切り | `e07a5e6` + dirty worktree | H5、rank 9、$L_D=4$、$K=2$、$L=4,6,8$でpaired K1--K3最大1.665%。独立calibration/holdoutは最大3.776%、予測半幅1.459%で5%/2%基準を通過 | cost providerをRPE shot・誤差/失敗確率配分へ接続。新compiler・$L>8$・不通過条件だけ追加holdout |
| [2026-08-24](2026-08-24.md) | 階層compiled-cost model、$K=2$次数条件付き再検証、connected-cluster運用推定と軽量化 | `e07a5e6` | 固定DF snapshotの$L=4,6,8$ holdoutでK1--K3運用推定は全metric最大2.936%。点誤差5%内だが95%診断5.724%の留保。calibration/prediction/transfer分離と厳密key cacheを実装 | 別$L_D$・short-step・compiler/coupling条件への移送と角度不変性検証 |
| [2026-08-25](2026-08-25.md) | 複数order-2、独立K4、controlled $q=8$の追加・follow-up batch | dirty worktree | paired複数order-2は最大1.679%。$L_D=6$のK1--K4 paired $L=8$は4.008%。controlled $q=8$は0.0529%。全job完走・validator通過 | 系サイズ方向の独立holdoutで運用規則を確認 |
| [2026-08-23](2026-08-23.md) | ランダム回路加法モデルとRTE境界補正の高統計検証 | `e07a5e6` | 1000標本・独立2 seedでcount/sizeのpair-only残差を確認。same/different二分類が別seed pair holdoutを最大0.849%で予測 | count/sizeの$\mu_3$または$L=8$、$L_D,K$、controlled・compiler条件のholdout検証 |
| [2026-08-19](2026-08-19.md) | 論文Eq. (D6)によるPF摂動係数の再検証 | `8418192` | H4全$L_D$とH2--H5の支配位相比較を通過し、H6のD6係数をstate-actionで算出 | GPU経路をH8/H10で確認し、H12の候補$L_D$ごとにD6係数を決定 |
| [2026-08-18](2026-08-18.md) | finite-RTEとPF・摂動・QPE分枝誤差の検証 | `8fdc6b3` | H4全$L_D$の単一位相条件、H2--H5のdense比較、H6のstate-action係数までlocal確認 | GPU経路をH8/H10で確認し、H12の候補$L_D$ごとに$C$を決定 |
