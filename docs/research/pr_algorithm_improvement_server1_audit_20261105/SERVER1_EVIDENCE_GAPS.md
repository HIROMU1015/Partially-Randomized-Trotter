# SERVER 1：証拠の未確認・非公開・範囲外

2026-11-05 JST。これは所在と検証階層の整理であり、既存の科学判定・研究優先順位を変更しない。調査範囲と固定基点は[REPORT](REPORT.md)。

## SERVER 1で原本まで確認できなかった資料

| ID | 対象 | 確認できたもの／残る限界 |
|---|---|---|
| S1I-GAP-01 | Phase 2D B/F1/F2/PS48の非公開原run | 公開報告・要約・input hash・相対path hash台帳・旧STOPは固定Gitで辿れる。private QPY/行列/attempt journal/grant/markerの原本をSERVER 1の所在へ結合していない。原報告の保全記載を現在の現存確認へ格上げしない |
| S1I-GAP-02 | Track B G1–G10、Parent、M1-S、C1のprivate実行資産 | 公開SOURCE・契約・保存結果と後続報告は確認できる。実行時のprovider cache、全生成log、private streamとSOURCE closureの現在の完全性は未確認。別環境での欠落・不存在も推定しない |
| S1I-GAP-03 | familyカードのrun別SOURCE/input完全join | 47カードの条件/statusと固定参照を保持した。複数runを含むfamilyについて、source-labelled参考commitを一つの実行SOURCEへ集約しない。CSVの`NOT_BOUND_IN_THIS_AUDIT`と個別参照を残した。SOURCE自体の不存在とは異なる |
| S1I-GAP-04 | 全条件の一意化 | カードに記載されないgeometry、basis、rank policy、state、delta窓、compiler、評価selectorを推測補完していない。数値を研究間比較に使う際は原報告の条件表まで戻る必要がある |
| S1I-GAP-05 | LOCAL/SERVER 2との独立性 | 同じ公開commit/path/blobの引用は同一保存結果の再利用候補。今回、他環境のfilesystem・private seed/streamを調査していない。候補集合や数値の一致だけから独立cohort・独立再現を認定しない |
| S1I-GAP-06 | 古い索引以後の未知保存先 | 既存ファイル名索引と既知rootに限定した。未知root・未登録symlink先・unreachable object/reflog・nested archiveを探索していない。NOT_LOCATED_IN_CHECKED_SCOPEを普遍的不存在へ変換しない |
| S1I-GAP-07 | 全SOURCE/全archiveの復元 | 固定object、選択したSOURCE AST、bundle header、既存保全receiptを確認した。巨大bundleのpayload・全raw/hash・実行可能環境の復元検査はしていない。header/refの保持と完全な実行再現性は別 |

## 存在を確認した非公開資料と公開済み原版

| 保存ID | 現在の確認 | 公開・利用範囲 |
|---|---|---|
| S1I-WP11-01 | 既知RPE/cost worktreeのcompute log二つと既存backup verification metadataが現存し、本人所有 | baseline補助資料。本文・checkpoint数値・rawは公開しない。既存backup検査の歴史的PASSを今回の全復元成功とはしない |
| S1I-INDEX-01..04 | 以前の名前索引、repository/custody台帳、収録判断、delta照合と内部逆引きが現存 | 私有保存先の対応に使用。内部絶対path・host・private fingerprintを含む元台帳は転載しない |
| S1I-BUNDLE-01 | 本人所有の既存bundleとheaderが現存 | 固定履歴の保持候補。展開・全hash・実行再現は未検査。独立媒体backupではない |
| S1I-DOC-01..03 | 原checkoutにあるG6/H4後レビューとTrack B案内二つが現存 | 固定main内の保存原版とblob一致を確認済み。現在は「GitHubにない資料」として数えない。科学的採択・独立runでもない |

今回、**追加の未公開「改善研究science原本」についてSERVER 1現存を新規確定した項目はない**。上表の非公開存在確認はmetadata/保全/baselineに限定する。原報告にprivate raw保全の記載がある資料は、上の未確認区分に残した。

## 所在不足と区別する、記録上の未検証・未実行

| 系列 | 保存された境界 |
|---|---|
| Phase 2D-I | testは定義済み・当該段階では未実行。後続F1 runnerのcorrectness完了を、旧20 testの実行へ読み替えない |
| F1修正版 | 固定合成fixtureの登録correctness完了。interval未資格・fallback。認証score/最適選択/十分Nは取得していない |
| F2統計資格gate | `F2_QUALIFICATION_BLOCKED_BEFORE_EXECUTION_MANDATORY_STOP`。後続限定RESOURCE契約は別の主張範囲 |
| F2修正版・PS48 | 数値correctnessと限定actual費用を保持。γ・η/α・interval資格、認証N/N加重同精度総費用は未成立。追加fixture・分子・長時間・noiseへ移送しない |
| C1 H4 | 分子へのlogical pilot記録はあり、古い「syntheticだけ」を更新する。一方tail-onlyで、deterministic halves・strict native・共通十分N・全coherent estimatorは未取得 |
| Gaussian選択 | 同じPS48へ適合したcase ID表は独立holdoutではない。一般化可能なcompile前選択規則は未確立。方向の一時終了は一般no-goではない |
| 非prefix / sampling | 非prefix固有の増分機構が未確立という対象資料の状態と、実験全体の未実施を分ける。samplingの人工テスト/既知法対応は新しいH4科学runを意味しない |

## 今回の範囲外

他repositoryの独立研究・他ユーザー・未知root、LOCAL/SERVER 2へのSSH・転送調査、raw全面読出し・archive全展開・QPY decode、全filesystem走査を行っていない。旧DF screening/UWCの回収問題は前回索引の履歴として保持し、この追加調査で科学値を復活させていない。

保存数値の新しい平均・比・順位・fit・費用・shotsを計算せず、新しいsampler/シミュレーション/compile/研究runnerも実行していない。科学的coverageの横断確定、一般的新規性・優位性のレビュー、次の実験設計は範囲外である。

**SERVER 1の公開照合後mandatory STOP。** 未確認事項を埋める目的の新実験・他環境接続へ進まない。
