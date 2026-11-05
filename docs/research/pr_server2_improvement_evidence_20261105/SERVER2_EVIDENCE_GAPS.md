# SERVER 2：取得・同一性・証拠の未解決

2026-11-05。これは保存証拠の状態表で、科学的coverageのGO/FAILや追加実験の要求ではない。元status・STOPを維持する。[REPORT](REPORT.md)／[52行CSV](SERVER2_IMPROVEMENT_EVIDENCE.csv)。

| 論点 | 今回確認した存在・取得状態 | 未公開・同一性・残るUNKNOWN |
|---|---|---|
| 選定既存127参照 | 固定Git blobを読取、旧SOURCE_AND_PROVENANCE_INDEXのblob一致 | 全研究/全branch/全rawの網羅ではない。全127のlive GitHub再取得は未実施 |
| 明示されたSOURCE commit | CSVで明示したfull commitは既知Git storeに存在 | SOURCE列UNKNOWNは単一run SOURCEが未結合/複数段階/設計のみの場合を含み、不存在とは限らない。containing snapshotを実行SOURCEにしない |
| Phase 1 8,761観測・360参照 | 元REPORT、integrationのSOURCE_COMPONENTS/REFERENCE_INDEX等を再利用 | 元CSV/DETAILS全行の再生成・新しい全量custody走査はしない。今回の52行を独立実験52件として足さない |
| F1旧停止と修正版 | 旧summary594bab9e…、修正版b0dad8cc…とも取得。B物理/arm hashは保持 | 私有QPY/NPY/journal/raw全量未取得。公開summary/raw相対hash台帳は原本の代替ではない |
| F2旧停止と修正版 | 06ce7e5b…とdc7378ab…のsummary、旧/新SOURCE、compiled CSV.gz参照がある | 圧縮case表をdecode/再集計せず、全caseのhash/意味論を今回再検証していない。raw/private grant/markerは公開しない |
| F2統計資格停止 | qualification.jsonに `F2_QUALIFICATION_BLOCKED_BEFORE_EXECUTION_MANDATORY_STOP` | 後続per-shot資源計測はこの資格を満たさない。gamma、interval、eta/alphaの統計採択、認証N/総費用は未成立 |
| PS48 | 実行報告・SOURCE770e53ad…・固定baseline・保存target再利用を確認 | 等case平均と全law重み期待値は違う。全F2拡張、独立held-out、compile前選択規則は未実施/未確立。一時終了を保持 |
| G6独立人工受入/512/pilot | それぞれの固定記録に別のSOURCEとscopeがある | 別session技術受入を物理host独立性としない。512報告のartificial_only継承矛盾はtrial記録との関係を保持。今回raw全archiveを復元していない |
| C1 R1入力 | 公開exact core/input identityと新旧partition hash、stateの異なるhash定義がある | 元PR-2係数との差を保持。数値model bindingはrigorous operator boundでない。旧P5 workerの直接原因/処理済trialはUNKNOWN |
| C1論理pilotのnative移送 | 3方式logical探索完了、selected-event correctness・SOURCE接続表がある | 全label/event/pathのnative認証、deterministic halves、state/readout、十分N、同機会CPU対照は未取得。現在一時凍結 |
| M1-S | 保存DF入力とfinite signal、gamma/pi3/Stage S、部分独立照合の索引がある | 旧full-H/HD行列は未保存と原報告に記載。全scope独立correctnessはUNDETERMINED。後続は同信号の事後解析を含む |
| 非prefix分割 | RQ-02とPR-2 B4 gate/未実行記載 | prefix実績を非prefixの実測優位へ転用不可。別研究に同名語があっても同一入力/新算法と認定しない |
| UWC GAP-001 | 旧prose-onlyに対し後続探索が指定JSON3件・sweep資料発見を報告 | `FOUND_UNVERIFIED`。独立期待raw hash/元run SOURCE/command/当時環境/H-state-grouping join未確定。private fingerprint/rawは転載しない。旧環境aliasをSERVER 2と新認定しない |
| 旧DF GAP-008 | 旧invalidated/stale履歴と後続探索のscopeを確認 | 修正後canonical集約/split/index・元anchor rawは `NOT_FOUND_IN_SEARCH_SCOPE`。関連refinement/logを代用しない。不存在の全環境一般化はしない |
| Gaussian積・相関/IS構成 | 具体追加生成差未定義/既知法対応の保存監査がある | 「性能敗北の実験」や新生成器の完成ではない。今回新prior-art探索・証明を実施していない |
| 研究B | toyのD内部/outer/finite-RTE/native/conditional shot結果と強PF後続を確認 | u_num/gamma、rare費用、Full PFを含むshot加重比較、一般分子transferは未確立。一時保留 |
| 研究C | 固定H4候補hash、PR内部conditional N/Gと強決定論対照の記録、11月2日小step後続 | 全母平均価格・厳密CA・分子間汎化・独立方法差は未確立。小step自己目的の受理はPF/native/shot改善の結果ではない |
| 既存H8/B3の公開/raw差 | 今回起点の別SERVER 2差分報告へ接続 | B3主要表一致と圧縮raw全体一致、旧H8限定公開表とprivate詳細監査を分離。今回はprivate rawを追加公開しない |
| 他環境との重複 | study ID・条件・SOURCE・input/result hash・run/cohort・blobを提供 | LOCAL/SERVER 1へのSSH/直接調査なし。host alias、物理実行元、全過去seed独立性、別cohort同一性はSERVER 2単独で確定しない |

取得不能の表現は、(a)固定Gitで読めた、(b)非公開原本の存在が既存記録にあるが今回は未取得、(c)元探索範囲でNOT_FOUND、(d)SOURCE/run同一性未結合、(e)検証未実施、を分けた。権限拒否の発生していない未調査資料を `ACCESS_DENIED` と表示しない。UNKNOWNをnull費用0や失敗へ置換しない。

公開できるのは今回のmetadata整理4ファイルで、原raw・承認原文・資格情報・private fingerprint・内部host/IP/絶対pathは含めない。旧原本を移送・展開・整理していない。調査完了後はmandatory STOPし、未解決を埋めるための科学計算を自動開始しない。
