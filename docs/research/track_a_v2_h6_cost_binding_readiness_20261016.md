# Track A V2：H6理論モデルと保存compiled費用の接続準備

2026-10-16 JST。**既存完全H6 archiveへのadapter接続と人工検証を実装した。実H6/H8の新理論費用・primitive数値評価・native build・compileは0。**
専用branch `track-a-v2-h6-cost-binding-readiness-20261016`。
既存監査46c06e56から、作業中に固定commit化された追加metadata受領
`b0c7a2855482175e438ffebe6c821ce36b0b2860` をfast-forwardで再利用した。
別worktreeを変更・commitしたものではない。V1完成をこの作業の条件にしない。
[証跡・計画入口](../../artifacts/resource_applicability/track_a_v2_h6_readiness/2026-10-16/README.md)。

## metadataの所在と分類

対象は既存linear H6/1Å/STO-3G/保存DF19/12 modes/sector400/T=.8。
H8・V1の新候補は今回のV2数値scopeから除く。新SCF/DF/state/preparationは不要。

|不足項目|既存所在・今回の状態|取得分類と将来の処理|
|---|---|---|
|PF Gaussian全block|private H6 frozen archive 5,670,237 bytes、SHA ad52f9f6…、7 roots|保存値から取得可能。既存description readerを再利用しhash/order/parameter/局所行列hashの記録値を照合|
|PF diagonal係数|同archiveのone-body eigenvalues、fragment eta/lambda|保存値から取得可能。係数→時間依存RZ/RZZ/global phaseは**追加数値処理**であり今回未実行|
|完全RTE component catalog|faithful:p5/p10/p15、1,092/702/312 components|保存値から取得可能。未観測componentも保持。event観測頻度から確率を作らない|
|完全Taylor次数law|固定producer SOURCEのfinite-RTE規約、K2/K4とtau|追加数値処理。保存binary64 lawと元評価順序・normalizationへ結合し、未観測次数を0にしない|
|全candidate/accounting/wrapper identity|既存b0c7a285のH6 index132 scope entries/104固有candidate、保存1320会計条件への参照|保存値・軽量形式変換。元SOURCE、探索8/確認32、両軸、seed/event/compiler/receiptを維持|
|primitive/phase/lawと全候補の完全envelope|一部未結合。既存indexは全envelope認証完了とはしていない|追加数値receiptと静的joinが必要。未解決dataset seed SOURCE・basis variantをexecution SOURCEで代用しない|
|compiled RZ/CX値|固定既存CSV/会計receipt|保存値から取得可能。新compileなし。law期待値と有限標本平均を区別|
|compiled回転角の分類|保存transpiled gate-listがある範囲だけ可能|保存countsしかなければ未取得。新compileを自動追加しない。モデル角分類をcompiled分類と表示しない|
|history/support-dependent plan等|今回registered-full-native経路には不要|現在のモデルでは一般対応不可。発見時STOPし別scope|

archiveのdiscard:p5/p10/p15、faithful:p5/p10/p15/p19を静的検査した。
今回は各rootを新adapterへ渡し、block/component記述のdigestを保存した。
Gate/Operatorを復元せず、時刻を係数へ掛けず、確率の再正規化・ゲート数・期待費用は評価していない。
局所行列hashの記録一致は行列意味論の新certificateではない。
archive旧profile `4e32bb9b…` とB0I現profile `622ce4a8…` は別に保存する。

## 接続仕様と実装

[接続adapter](../../src/trotterlib/native_df_h6_cost_adapter.py)、[段階別runner](../../scripts/track_a_v2_runner.py)、
[人工テスト](../../tests/test_native_df_h6_cost_adapter.py)。
既存 `native_df_archive_metadata.py`、`native_df_metadata_binding.py`、`native_df_gate_model.py` を再利用し、元producer/counterのbytesは変えない。
primitiveは固定provider `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7` の `df_trotter.ops` を呼ぶ設計で、多項式を別実装しない。
SOURCE/registry hashとモデルのoperation digestは別の認証領域として保持する。

モデルAは各basis sandwichを閉じ、モデルBは宣言済みの同basis隣接相殺だけを許す。
**literal nativeはA-prefix/B-tailの混合**。A又はB全体を実nativeの構成と同一視しない。
RTEは各occurrenceで閉じ、prefix half/外側step/occurrenceを跨いでflattenしない。
scalar・identity成分は一度だけ相対位相へ含め、normalization Bはgateとして足さずshot会計へ接続する。

lawは元producerの全component重みと全偶数次数を保持する。
モデルがFractionを必要とするため、保存binary64重みの正確な有理数化と再正規化を明示政策として要求する。
これはnative samplerの累積binary64境界までbit同一だという証明ではない。元lawの意味・tau・R=qr・normalizationへのbindingがなければ理論評価を拒否する。
確率0/underflow・missing orderを発見した場合、再sampling・勝手な近似lawへ切り替えない。
一般identity位相は0〜1 P/closed occurrenceの既存幅を保持する。

envelope validatorは既存の型を含むidentity照合を再利用し、各receipt実体のSHAも照合する。
expectedをcandidate自身からコピーするだけで認証としない。生成元SOURCEと独立に固定したexpected envelopeが必要。
未測定B0/B0IはMISSING。両軸のSOURCE/seed/event/profileが異なるpairは拒否する。
全104候補の数値envelopeが完成したとの主張はしない。

## 承認後の段階と必要計算量

最初の限定stageは**保存係数からのtimed primitive receipt生成**。
7 rootsの既存係数とGaussian記述を使い、元producerで指定時間だけ評価する。新native full wrapper・Gaussian生成・compileは不要。
未承認private configのmetadata_tasksは既存104候補から固定し、root/source/primitive scheduleを新生成前に照合する。
最大2,048個のroot/block/time primitive recordで停止する。完了後mandatory STOP、理論stageへ自動移行しない。

次の理論stageは、完全receipt bindingを静的に仕上げた後、既存104候補×A/B×両軸、最大416行。
PR72候補×A/Bのoccurrence期待値評価を最大144回、1DP最大1,000,000 transitionsとする。
native同等の混合政策はadapterに明示し、A/Bと別の構成として検査できる。混合政策の全候補追加評価はこの416行に勝手に追加しない。
DP cap超過時はSTOP、上限増量・近似fallbackなし。
既存shot数とε/factorを再利用し、理論RZ/CX、compiled RZ/CX、PR/PF比、event期待費用・希少次数寄与を接続する。
保存済みT panelのdeterministic費用receiptを静的に照合する計画も残す。T=.4/1.6のB0費用をT=.8 B0Iで無条件代用しない。

対応式は互換なSOURCE/profile/compiler/axis/control/分解stratum内の `compiled=a+b*theory`。
共通較正後の各行残差 `(compiled-fit)/compiled` をPR/PF別に保持し、平均残差差の絶対値が.05以上なら停止して報告する。
方式別fitは補助とする。零費用・不足行・未結合行を残差0として補完しない。
non-Clifford分類はlowered RZ角のπ/2整数倍との距離で行い、角許容差を明示する。T-count/Toffoli数には換算しない。

各stageの未承認STOP ceilingはprocess1/thread1/CPU1枠、AS8GiB/RSS4GiB、CPU4時間/wall6時間、output256MiB、保存枠512MiB。
計算時間・peakの実測予測ではない。static inspectionと人工試験は別の軽量上限で実施した。
実native buildが追加で必要と判明した場合は生成対象・意味・資源を再定義し、その範囲の明示承認を得る。今回のgrant案にnative buildは含めない。

## 実行前gateと未解決事項

metadata stage：archive/input/source/producer/current Pythonのbytes、期待7root hash、104候補へのroot割当、CPU/RAM/PSI・外部監視・保存枠・旧job非干渉をfresh確認。
theory stage：primitive/phase/law/shot/allowance/seed SOURCE/basis variantのreceipt、両軸pair、compiled来歴の完全bindingも必要。
profile/compiler差、未観測次数、identity phase幅、経験的uの意味を保持する。
科学的未解決は厳密費用母平均保証やFT総費用であり、今回の人工PASSだけでは閉じない。

新grantなし、`approved=false`、`runtime_authorization=false`、CPU[]、command=null。旧grant/ledger/実行済markerを再利用・変更しない。
初回primitive生成も新規数値処理なので、今回実行しない。
主担当の人工検査であり、独立した別AI reviewerの検査ではない。H5停止・変更、H8拡張、PR-S4/q16、samplingを行わない。
旧manifest checkerは起点の拡張キーにも対応していないため、全体CI成功とはしない。今回のentry・旧entry不変・リンク・bytesを限定検査する。
commit/pushとremote検査後、**MANDATORY STOP**。限定計算の開始には固定範囲・予算への別途利用者承認が必要。
