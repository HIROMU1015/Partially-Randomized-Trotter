# H2高次PF：outer32 pilotから本探索取得まで

2026-10-21。利用者の「私の承認は飛ばしてよいので現在H4とH6がやっている検証地点と同じところまで進めてほしい」を、以下4段階の実行承認として結合する。中間の再承認は不要とし、各段階の独立したone-shot契約・grant・ledger・marker・STOPを保持する。異常STOP後の自動retry/top-upや条件変更は認めない。

## 対象と到達点

H2直鎖1 Å/STO-3G/tol-only1e-8・rank自動決定済み3、保存Hamiltonian/state、T=.8/P=0、prefix1、S2/Yoshida4/m5_best、q1/2/4/8、K4、ABS_TIME_PER_OUTER=32、計12群。H4/H6と同じ固定PF係数・signed schedule・RTE法則と実装を使う。旧H2のpieceごとのr条件とは別系列で、旧n8・pilot・独立確認を混ぜない。

1. **H2_SIGNAL_OUTER32**：12候補を新signal評価し、5精度×2軸の経験的適格性を保存する。保存Hamiltonian/stateを復元し、H2のfull-H reference vectorを1件作成する。保存FPのreference複素値と照合する。SCF・DF・状態生成は0。
2. **H2_DESIGN_PILOT**：各33 trajectory、計396本・792 compile軸を独立cohortで取得し、技術終了監査を行う。旧H4/H6と同じ正規IID参照の分散相対SE25%の目安に基づく設計で、分布仮定やtail保証ではない。
3. **H2_PILOT_DESIGN**：終了監査PASSのpilotのみを評価する。paired cosine/sineの共分散、単価分散、SE/MCSE、希少層・未観測null、108名目n案を保存する。H2自身の観測値から相対MCSE目標2%・分散倍率4・群別配分・下限33を採用する。H4/H6の本数・分散は移送しない。
4. **H2_PR_EXPLORATION**：確定したH2群別n_Eで、新seed/cohortの本探索単価を取得し、技術終了監査を行う。この段階の統計評価・shot会計・ランキングへ進まずmandatory STOPする。

観測された軸別分散が0の場合、旧統計moduleの名目nはnullのまま保持する。新探索33本は有限標本比較の事前固定下限であり、MCSE2%が設計できたとは表示しない。該当候補を`zero_variance_MCSE_design_unresolved`に記録し、未知の母分散・希少層費用の上界はnullのまま保持する。通常分散が非zeroの候補は既存の軸別名目nの大きい方に下限33を適用する。

pilotの条件付き希少層の費用・分散は、0観測ならnull、1観測なら分散nullを維持する。強制・棄却・自動top-upなし。本探索後の統計、正式shot政策、総費用再会計、順位、独立確認P、AD03採用、詳細科学レビュー、最終freezeは対象外。

## 再利用と実装の区別

既存H2のtol-only snapshot・保存状態・F/FP/D/A資料の入力identityを再利用する。旧m5 run03のSTOPはそのまま保持し、原本や標本を新cohortへ取り込まない。既存H2の統合比較成果とも科学契約を分ける。決定論的対照の旧費用を新系列へ無条件に混合しない。

H4/H6の固定native演算、占有数oracle、paired Hadamard wrapper、canonical cutoff、hash cache、archive resolver、終了監査、pilot統計関数を再利用する。cohort文字列と契約binding以外のnative numerical bytecode・定数・演算順序は変えない。新signalはspectral-tail経路と独立したascending matrix polynomial/expm経路を1e-9未満で照合する。

Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric-directional、scalar once、cosine/sine pairedを維持する。H2で既に使用した`H2_QISKIT130_SYMBOLIC_ZERO_SCALAR_GUARD_V1`をbyte同一で再利用する。このprocess内の位相帳簿修正は、元translatorと全gate/角度/wire列が同一であることを毎pass確認する。入力状態へのfit・事後位相合わせ・gate削除は行わない。guard receiptを軸ごとに保存し、H4/H6の元compiler経路と同一のSOURCEだったとは扱わない。

compiled vectorのCT01は1e-7未満、他のnative/oracle/Hadamard/logB検査は1e-9未満。通常signed bias・保存経験的allowanceの評価規則を再利用するが、正式shot政策や厳密bias保証へ変更しない。全科学入力・SOURCE・runtime/providerとstage planをhashで結合し、結果・終了証拠を保存する。

## 並列化・資源・保全

fork/read-only context共有、SHA確認付きJSON cache、native fingerprint cache、長いq×pieceからのdispatch、外側processと内側thread重複回避を再利用する。現在空いているCPU・実効RAM・scratch・inodeからworker数を決める。historical worker数は上限にしない。

H2の起動計画には既存H4の保存/RSS proxyを保守的に使用する。proxyは出力サイズやRAMの上界ではない。H4/H6の既存338708447996 byte容量計画は見積り直さず、両stageがexit/reapするまではその固定予約をH2の起動確認で保全する。H4/H6のjob・SOURCE・grant・ledger・marker・STOPや他worktreeのHEAD/statusを変更しない。

新規QPY/NPZのlevel1 byte可逆保存・元SHA/archive SHA・durable manifest・原byte復元検査を継続する。archive検証後の解放は新契約自身の一時rawだけで、旧原本・worktree・cacheの整理や削除はしない。共有設定・venv・他ユーザーのprocessに変更を加えない。

各段階の成功後にnative exit/reapを確認する。取得stageは全登録・native gate・call台帳・結果SHA・archive復元・compiler guard・全worker回収を監査し、PASSの場合のみ次の承認済みH2 stageへ進む。欠測・数値FAIL・hash不一致・資源不足・異常exitは当該H2契約をSTOPする。H4/H6へsignalを送らない。host PSIや無関係な資料更新だけで中断しない。

## 技術検証と次の担当

新14件の人工検査と既存100件の回帰検査がPASSした。人工H2-layout snapshotの復元・独立signal、3公式×2cohortのnative/compiled両軸とarchive、SOURCE演算不変、seed非混合、無効grant/CLI拒否、二重予約拒否を検査した。実H2の取得成功を意味しない。作者自身の技術検証であり独立レビューではない。

担当Codex：起動前gateを通し、承認済み4段階を技術監査付きで実行する。終了・異常時は記録を保存してmandatory STOP。担当ユーザー：終了結果を見て、本探索後の限定統計評価を別途承認するか判断する。
