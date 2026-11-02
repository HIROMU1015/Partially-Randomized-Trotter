# R2-CS C0.6：一方向判定条件の最終限定調査

2026-10-24。研究ID `R2-CS-20261024`。基点C0.5 `25cbd26d27057bd3d5af9c18d1c4b7be13d9cedf`。C0/C0.5全体の再調査はせず、G6・finite-RTEのlaw情報から、全signal/reference/native費用を取得せずに使える具体的な一方向条件が追加できるかだけを調べた。

**提案はC：R2-CSを独立した構成選択法の研究として終了することを推奨する。正式採択はGPT・ユーザーへ返す。** 条件式そのものは書けるが、既存G6の保証・一般の上下界比較・既知IS・cacheを超える新しい選択機構は特定できなかった。これは有用な条件が一つもないという主張でも、将来の方法を否定する不可能性定理でもない。

## 最後に絞った3候補

|候補|成立する条件と使える帰結|新しいR2-CSの選択条件に至らない理由|
|---|---|---|
|P-A 安全除外|適格な対照cの認証費用上界H_cと、候補dの同一目的下界L_dがあり、L_d>H_cならdを除外。G6固定辞書のcost–moment下界、W2b/W2cの限定logical非劣も既存|対照の適格性、provider価格、取得費用が未解決。一般の区間比較/既知ISの範囲。限定logical順位は全native総費用順位でない|
|P-B 一方向適格性|全pathの独立bias上界b_plus<epsilon、正当なmoment/range/confidenceから十分予算を出せる。finite polynomialへのG6保証は既存|指数/分子PRにはPF・積増幅・input・native誤差等の接続が要る。上界不通過から不適格とは言えない。安価で有用な新bias certificateはない|
|P-C law同値class|S2の固定split/lawで同じR=qrならlaw/normalization算術を共有可能。高次はpieceごとのsigned time/整数配分を保持|省けるのは重複setupで、signal/bias/native評価の省略保証ではない。既存cache・共通部分式の再利用|

適用対象、前提、入力とTier、根拠、反例、実用性、不足責務は [candidate_propositions.md](candidate_propositions.md)。具体的な数値代入による候補費用評価は行っていない。

## 今回明確になった差分

C0.5では「局所情報から一方向条件へ接続できるか」を残した。今回、G6_NOTEだけでなく、その固定参照先W2B/W2Cの条件式とSOURCEを確認した。**十分Nの比較、限定構造費用の非劣、区間による安全比較、古典追加費用のbreak-evenは既に整理済み**である。これを新しい第2研究の成果として再計上する余地はない。

残る非自明な部分は、全candidateのsignal・provider価格を知らずに、適格な対照上界と有用な候補下界、または十分強い全path bias certificateを、直接評価より小さい取得費用で得る機構である。本調査ではそれを具体化できなかった。Oracleのheadroom・真のnormalizer・真の信号半径・全event価格を安価な特徴量へ移す案は採らない。[情報取得表](information_access_matrix.csv)

## 保持する既存成果と反証

- G6のGreen局所query、有限係数、positive parent/child、P5少数群、finite-bit補正・全trial会計は既存成果として保持。局所queryは無料でも全取得でもない。
- Uは真のB_fullではなく、その上界式。P_Uのm7反例は既存の普遍的上界比較を否定するもので、真のB_full・総G・G6の価値を一律に否定しない。U.loもB_fullの下限として使わない。
- 十分予算policyの費用下界は物理的最低shot数の下界ではない。moment/range、accepted回数、hard capを区別し、zero/rejectの古典費用を残す。
- 同じR=qr、normalization、lawが一致しても、deterministic挿入・signed piece・整数配分・boundary fusion・signal・N・Gは一致しない。

対応する既存証明・反例・未解決責務は [proof_and_counterexample_map.md](proof_and_counterexample_map.md)。先行研究はC0.5の索引を継承し、cost-aware IS・SPRINT・MF-BAIの関係節だけ一次本文で再確認した。[差分整理](prior_art_delta.md)

## 判断と作業境界

C0.5のBからCを提案する理由は、今回が最後の限定調査であり、具体的な新機構が出ず、残った条件が既存資料に含まれていたためである。有用なG6構造を失敗と判定し直したためではない。同種の追加調査、P1、grid拡張、proxy探索、追加分子を継続理由として推奨しない。新しい検証計画・科学SOURCE・契約・grantは作らない。[判断資料](decision_packet.md)

**16 endpoint P1はHOLDのまま。** R3のSTOP、C1 pause、G6・PR-2・第1研究のSOURCE/結果/grant/ledger/marker/STOPは変更しない。終了推奨の対象はR2-CSの独立構成選択法という研究scopeだけである。正式な終了採択や旧状態変更はしていない。

新科学計算・実データ数値再解析・費用比/regret/ranking/fit・sampling・simulation・compile・科学module import・詳細科学レビュー・grantはすべて0。必要最小限の人工fixtureとmetadata/AST/SHA照合のみ。[技術検査](technical_checks.json)、[固定SOURCE](source_catalog.json)、[範囲・状態](scope_and_state.json)、[mandatory STOP](STOP.json)

固定参照33件・情報取得18項目・候補命題3件を整理し、技術検査29件はPASS。技術検査の再実行はworktree起点で `python3 -B docs/research/r2_one_sided_c06_20261024/technical/check_c06.py`。PASSは著者自身の技術確認であり、数学の機械証明・独立科学レビュー・新規性の確定ではない。正式共通規約の指定SHA原本は未取得、手元版と今回の明示指示を適用した。版差はscope記録に残す。

**次担当はGPT・ユーザー：本資料からR2-CS終了推奨の採否を判断する。Codexは公開・remote照合後mandatory STOP。自動実行・同種調査の反復・他研究の再開は行わない。**
