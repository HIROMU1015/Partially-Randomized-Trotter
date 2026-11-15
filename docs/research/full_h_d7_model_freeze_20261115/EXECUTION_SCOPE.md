# Full-H κ2 D7：受入・軸別fit・観測前4予測の固定scope

2026-11-15 JST。今回の利用者指示により本範囲は実行承認済み。新しい承認証明・grantは作らない。

- 正本：D1 e628c0d6のHANDOFF_C.json。先行4 cellは7d0c3906、G36223は完成f98cb963。元18軸を一度だけ数え、旧partial・再開時4軸を追加しない。producer SOURCEと最終validator 8cce5406を区別する。
- 非configuration較正5 cell：G566(1,566,n2)、G4528(1,4528,n2)、G36223(1,36223,n2)、Q8R566(8,566,n2)、P2R566(2,566,n1)。X/Y別PRICE_CALIBRATION_RECORD_V1を10件使用。元receipt・profile/law/実装/seed/測定と、index0のみの7数値gateを保持する。
- 元保存meanを目的変数にする。Cはcohort平均を再生成せず、既存公開照合と個別price/compile bytes/hash/実装証拠の対応を受入に再利用する。元gzip rawは専用decoderの固定locatorを保持し、generic readerへ渡さない。
- 特徴量はA7a0c9468のFULL_H_GROUP_V1、順にone,event_count,outer_boundary_count = 1,qR,q−1。D7 planはa43050c7の設計と、D1固定FROZEN_JOBSの採択済みplanへ対応付ける。
- 各軸5行・3係数のexact有理数最小二乗。既存A fit_identified→固定C c628e9e8 price_model_validation.fitを使う。各cellの重みは1、path数による追加重み付け・X/Y poolなし。rank3、係数・較正残差・元入力locatorを保存する。
- P2R566は保存G566 path0→path1の連結。設計の1較正行として使い、独立な第3親標本に数えない。共分散・係数CIは評価しない。
- 予測は固定M5_ROUND(q32,R283,s9056,n2)・M6_ROUND(q64,R566,s36224,n1)のX/Y、4件だけ。観測価格は開かない。後続compileは6件であり今回は0件。
- 各モデルはPRICE_LINEAR_FEATURE_MODEL_V1、status CALIBRATED_UNVALIDATED、qualified=false。元domain/design_sourceとcalibration.source内のcohort/step_seedsを保持する。許容差・モデルPASS/FAILを追加しない。
- fit前と公開前に関連SOURCE/ref/path/job/既知namespace metadataを確認する。完全性は確認範囲に限定し、未解決の重複やholdout取得の疑いがあれば先に解決する。
- 固定SOURCEとINPUTSをcommit後、fresh run01に2fit・4predictを一括実行。入力不一致の軸は保留、科学的条件変更・自動retry/refitは行わない。
- 原D1のfrozen_model_gate・seed_table・physical_row・validate等の無変更metadata関数を固定SOURCEから抽出し、科学launcherをimportせず読取を確認する。公開JSONの固定commit/pathでも同じgateを呼ぶ。
- 保存文書cache・既存exact helperを再利用する。5×3の小さい系でprocess並列の起動は不要。段階開始/終了と数値段階wallを記録し、科学worker/共有環境を変更しない。

**終了点：モデル2件と予測4件を固定・公開しD1へ引渡し。holdout取得/開封/判定、他round価格、母law E[C]・CI・N/G/winnerには進まずmandatory STOP。**
