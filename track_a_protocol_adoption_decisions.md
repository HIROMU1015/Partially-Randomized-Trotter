# Track A共通プロトコル：採択判断表 v1 正式版

2026-10-19。**ADOPTED_V1_DOCUMENT_ONLY**。ユーザーの今回の明示判断を記録する。元DRAFTは `9c54d89f6154525e3c7c814a4da5ecdc37453b8a`。基本方針6件、条件付き5件、追加PF保留1件。科学計算・数値再解析・公開push・既存契約への自動適用の承認はいずれもない。

**暫定追補CT01（2026-10-19）**：ユーザーの「一旦 －７ にして」により、compile前後のraw状態ベクトル2-norm差だけを、finiteかつ1e-9未満から**finiteかつ1e-7未満**へ変更する。全Track A系列・公式・PR/PF・両軸の共通文書基準。既存PASS・事前検査の再実行は不要、旧STOPは保持する。他の閾値・compiler・shot allowance・AD03は維持し、今回は実行コード変更・保存値再判定・新科学計算を行わない。以下のAD08はCT01の範囲だけ補足されたものとして読む。機械可読`adoption_record.json`は初回v1採択の履歴原本として不変に保つ。

|ID|採択状態|対象|決定内容|条件・保留境界|
|---|---|---|---|---|
|AD01|ADOPTED_BASIC_POLICY|共通DF政策|tol-only 1e-8・rank自動決定を採択。H4 legacy DF12は独立した既存成果として保全。|rank数は統一しない。legacyをtol-onlyへ読み替えず、全生成規則同一性は未認定。|
|AD02|ADOPTED_BASIC_POLICY|入力manifestと来歴|SCF・軌道・積分・DF・状態・runtimeをmanifestで照合する基本方針を採択。|未照合をPASSへ変更しない。solver差等の個別同等性や再生成は今回認定・承認しない。|
|AD03|DEFERRED_ADDITIONAL_PF|追加PF公式|既存S2/S4の比較条件を維持。m5_best等の追加公式を主比較へ採用する判断は保留。|m5 pilotは再開しない。保留は非ZERO残差だけによる棄却や既存S4の無効化を意味しない。|
|AD04|ADOPTED_BASIC_POLICY|共通比較・会計条件|線形1 Å/STO-3G/T=.8/P=0/5ε/通常signed・倍率1/full wrapper/directional/固定compilerと、既存ε/√2・α_axis=.025の基本会計を採択。|SOURCE固有のallowance・数値診断・guardを維持。shot αと標本推論のαを分離し、厳密bias保証は付けない。|
|AD05|CONDITIONALLY_ADOPTED|実測・model・欠測区分|既存DIRECT_MEASURED/VALIDATED_MODEL_ESTIMATE/MISSING区分と適用scopeを明示する条件で採択。|全候補や全P対照の新規直接compileを一律必須化しない。model条件付き最良を実測最良へ改称しない。|
|AD06|ADOPTED_BASIC_POLICY|探索規則・境界|系別の根拠ある疎な事前登録domain、5ε評価、有限標本・選択効果・境界の明示を採択。|同候補数・同直積格子を強制せず、既存強化集合を弱めない。新候補数や境界拡張は個別契約。|
|AD07|ADOPTED_BASIC_POLICY|独立確認P|H8型の事前固定・新cohort・新seed・標本非混合を共通手順とし、原則両端ε=.05/.0005を対象とする。|n=48は強制しない。分散・差・希少event・目的から個別設計。n・power・CI/多重比較は別固定、母平均保証とは区別。|
|AD08|CONDITIONALLY_ADOPTED|技術・model閾値|CT01によりcompile前後の状態差だけ共通の暫定1e-7未満とする。他のnative tolerance・closure・guard・model閾値は既存SOURCE/契約の条件を維持。|過去2%等は不変。既存PASS・事前検査は再利用。旧STOPは改変せず、新基準での照合は別記録。封印済みコード・契約への自動適用なし。|
|AD09|CONDITIONALLY_ADOPTED|機構分解と説明欠測|exact両軸費用分解と説明側statusを分け、説明欠測を明示する条件で採択。|説明未検証の点は費用coreまでに限定。A82回収を説明完了としない。新cohortの説明も自動接続しない。|
|AD10|CONDITIONALLY_ADOPTED|legacy・掲載例外|legacy/model/profile未確認/限定集合/探索点の例外ID・scope・caption表示を条件として採択。|個別主図・主表の最終採否や全系COMMON_CORE_READYを自動認定しない。条件付き掲載と完全共通条件を区別。|
|AD11|CONDITIONALLY_ADOPTED|完了・主張範囲|証拠範囲を限定したCore/条件付きbenchmarkで一区切りにできる方針を採択。|Extended/Guaranteeへ自動進行せず、PR不利・非再現も有効結果。母平均・tail・厳密biasの保証は未取得。|
|AD12|ADOPTED_BASIC_POLICY|移行・担当|正式文書と個別実行承認を分離し、既存契約へは自動適用しない方針を採択。|Codex A/Bを含む既存SOURCE/結果/grant/ledger/marker/STOPは不変。将来移行は担当と対象差分を別承認。|

採択は基準文書の正式化であり、各系の科学的受入・主表の最終freeze・新たな実験条件変更の承認ではない。未達/未照合のstatusは採択後も維持する。旧DRAFTの提案は、そのまま全細目が採択されたとは扱わず、上表の採択scopeと条件だけをv1へ反映した。

機械可読原本：[adoption_record.json](artifacts/resource_applicability/track_a_protocol_v1/2026-10-19/adoption_record.json)。[正式プロトコル](docs/research/track_a_pr_resource_evaluation_protocol_v1.md)、[充足表](track_a_protocol_compliance_matrix.csv)、[差異・例外](track_a_protocol_gap_and_exception_report.md)を併読する。

次の担当：**ユーザー**。公開の要否、H6の個別数値設計、A82説明検証等を必要とするかを別判断する。Codexは今回のlocal commit後mandatory STOPし、未承認stageを起動しない。
