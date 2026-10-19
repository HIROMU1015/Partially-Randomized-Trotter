# Track A共通プロトコル：採択判断表 v1 DRAFT

2026-10-19。**全項目PROPOSED_AWAITING_USER_ADOPTION。正式採択0、実行承認0、既存契約への適用0。** 提案欄はCodexの整理であり、ユーザーの回答を記録した欄ではない。

|ID|ユーザーが判断する科学条件/最低基準|提案・選択肢|未採択のまま保留する内容|
|---|---|---|---|
|AD01|主比較のDF政策|tol-only1e-8・自動rankを共通目標にする。H4 legacy12は明示例外として残す案、tol-only10の主比較完成後に共通panelを作る案|H4 legacy D/Pを無効化/置換しない。rank10/19/28の全入力規則同一性は未認定|
|AD02|入力生成の共通manifestと例外|SCF/MO/積分/DF correction・signed順・cutoff/projection/sector/state・runtimeを照合する案。異なるsolver等の条件付き同等性を許す範囲も事前判断|名前・tolだけの一致ではPASSにしない。自動再SCF/DF/state生成なし|
|AD03|S4公式・係数版・m5_best|既存S2/S4を最低対照とし公式ID/係数原本を固定。m5を追加主方式にするか別補足にするか選ぶ。数値残差のみの機械的棄却はしない|m5 pilotは保留。新PFを旧Yoshida modelや旧費用と同一扱いにしない|
|AD04|共通endpointとshot会計|T=.8/P0/5ε/通常signed/factor1/full wrapper/directional/固定compilerと、既存ε/√2・α_axis=.025の会計案を採択するか|allowance値の根拠・数値診断・guardはSOURCEへ結合。厳密bias保証は付かない|
|AD05|PF対照の証拠水準|固定検証modelを明示して主表に許す案、独立Pの選択PFは直接測定を原則とする案。未知候補がある場合の固定subset例外も判断|全候補直接compileを自動要求しない。modelをDIRECTに変えない|
|AD06|D/Aの探索範囲と境界|分子ごとの疎な登録域・根拠・予算を許容し、同数同格子を要求しない案。端なら登録内限定か別承認拡張かを選ぶ|candidate数/prefix/q/r/K/追加q16等の新範囲は未固定。既存A/Bを変更しない|
|AD07|Pの最低対象と標本設計|H8型の事前固定/新cohort/無poolingを採用し、原則ε=.05/.0005。候補数・n・差精度・近似CI/多重比較の有無は対象ごと事前設計|n48/n32/13・45・45やpower90%を全系の正式基準へ流用しない。稀少上界保証を含めない|
|AD08|技術/モデル受入閾値|SOURCE固有のnative tolerance、closure、丸めguardを明示し、共通値が必要なら実装・allowanceとの整合で選ぶ。過去のmodel2%は採択候補として保持|1e-9/1e-8、説明1e-12、2%を無承認で一律変更しない。旧STOPを解除しない|
|AD09|費用分解と説明欠測の掲載|掲載cohort/対照版でexact両軸分解を要求。normalization/headroom説明の完全性を主機構図では要求し、欠測点は会計のみ/別枠とする案|H4新D/Pへの自動再解析なし。A82_OLD_PR.PRは逆算補完しない|
|AD10|主図・主表の例外|COMMON/CONDITIONAL/EXPLORATORYの表示案と、legacy/model/profile未確認/限定subset/中間ε未確認をどこまで主表へ許すか|正式な論文採否や全系Core完了を今回認定しない|
|AD11|研究の完了範囲|Coreの条件付きbenchmarkで一区切りにするか、どのExtended/Guarantee主張を追加で求めるかを選ぶ|Core後の自動拡張なし。PR不利/非再現を失敗扱いして追加計算しない|
|AD12|採択後の移行・担当|新契約だけへ将来適用する案。Codex A/Bの進行条件とのdiffを別途静的照合し、必要な科学変更は担当・対象・承認を固定する|既存grant/ledger/marker/SOURCEや承認済み固定契約を遡及変更しない|

推奨する判断順はAD01–04（対象/条件）→AD05–09（証拠/最低手順）→AD10–12（掲載/一区切り/移行）。全項目を一度に決める必要はなく、未採択は未採択として残せる。

採択は文書の正式版作成を認める判断であり、sampling/compile/数値再解析等の実行承認とは別である。正式版を作成しても既存Codex A/Bや保留m5契約へ自動適用しない。

次の担当：**ユーザー**。各IDの採択・変更・保留を判断する。Codexは判断を受けて必要な文書改訂/差分整理を行うが、今回はlocal commit後mandatory STOP。
