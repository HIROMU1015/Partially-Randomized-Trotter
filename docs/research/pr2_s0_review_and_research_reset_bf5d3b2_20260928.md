# PR-2：S0停止の独立レビューと、研究目的を保った再開方針

日付：2026-09-28  
対象：`HIROMU1015/Partially-Randomized-Trotter`  
branch：`all-r-coherent-opt2-reoptimization`  
レビュー固定commit：`bf5d3b2405b2bec4f88dd0589b75dd737b13e052`  
authorization：`e9bffb85f9ed57712bb83150172a6a4662cecf7f`  
S0/S1 source：`c644925b50587072784846df09bf02c39e8453e1`

## 0. 判断

**推奨する研究上の方針は `AMEND_AND_RESTART_FROM_NEW_S0`。ただし、旧S0をPASSへ変更する意味ではない。**

- 旧実験系列の `STOP_INPUT_REPRODUCTION_MISMATCH` と `S1_authorized=false` は維持する。
- PR-2の研究仮説は、このSTOPでは検証されていない。研究そのものを否定する根拠にはならない。
- 旧入力を回復できるかの確認を一巡行う。回復できなければ、その制約を明記して新しい固定入力の実験系列へ移る。
- 旧pilotと新系列の数値を、同じ入力に対する連続した実験結果として混ぜない。
- 新系列の入力を固定した後は、毎stageで分子から作り直さず、同じsnapshotを読み込む。
- 新しい研究課題へ乗り換えたり、canonicalization自体を主論文にしたりする必要はない。
- 次の検証を一件へ制限しない。ただし、再現性回復・研究対象の確定・S1接続という必要事項を一つの有限な作業パッケージにする。

この文書はレビューと改訂案である。実験authorizationを記載したリポジトリ原本を変更しておらず、今回こちらではsimulation、分子生成、回路compile、trajectory sampling、testを実行していない。

## 1. 旧S0停止の評価

### 1.1 停止は正しい

amendment v3 §3–4は、S0が `S0_PASS_S1_AUTHORIZED` 以外ならS1を実行しないと定める。sourceの `run_s0` も、development hash不一致時にはprefix identity、estimator、wrapperを評価せずSTOPを返す。[R1,R4]

保存結果ではexpected hashが `d8b4aaf2…a3dc3`、observed hashが `de7a5492…12424` であり、S1 authorizationはfalseである。[R2,R3]

確認したv3文書はauthorization commitとレビューcommitで同じGit blob SHAだった。確認したS0/S1 moduleもsource commitとレビューcommitで同じGit blob SHAだった。これは該当ファイルの版の照合であり、実行環境や全artifactの独立再現を意味しない。

### 1.2 STOPの意味を拡張しない

この結果は「事前に要求したbyte-level input reproductionが成立しなかった」という結果である。PRの資源優位性、residual補完の必要性、corrected signalの精度についての科学的negativeではない。

旧S0ではcandidate signal、compile、trajectory sampling、quantum shots、S1の評価に到達していない。[R2,R3]

一方、test packetにはtoy上の回路・演算子検証も含まれる。『研究candidateについて0件』と『testでも一切回路を扱っていない』は区別する。XMLの123 tests、failures=0、errors=0を確認したが、こちらで再実行はしていない。[R6,R7]

### 1.3 これまでの助言の補正

同じファイルを使い続けるためのbyte hashは重要である。しかし、毎回SCF/DFから再生成して同じbyte hashに戻れることを、研究全体の継続要件にする必要はない。

適切な運用は、**生成時に完全な入力を保存し、その後の比較で同じ保存物を使い続けること**である。旧pilotで入力が保存されていなかった問題を、将来の全実験でも旧hashへの復帰を要求し続けることで解決しようとしない。

## 2. 現状の診断は何に十分か

| 問い | 現在の証拠による判断 |
|---|---|
| 旧S0を止めるべきだったか | 十分。登録されたhash gateが不成立 |
| 新snapshotが旧snapshotと同じか | 不十分。scalarの近さは同一性の証明にならない |
| 原因がDF factorの符号だけか | 不十分。旧配列との照合がない |
| 原因が縮退部分空間の回転だけか | 不十分。旧配列・スペクトル・変換の対応がない |
| PR-2の仮説が否定されたか | 否定されていない。候補性能を測っていない |
| 旧S1へそのまま進めるか | 進めない |
| 新しい固定入力で同じ研究RQを検証できるか | 可能。新しい来歴・事前契約・入力検査が必要 |

環境gateで一致したのは記録されたPython/packageのversionである。BLAS/LAPACK、CPU実行経路、thread、ビルド設定、solver初期値まで同一と確認した記録ではない。NumPyにもこれらを別途表示する機能がある。[R3,R4,W1] これは原因候補の列挙であり、今回のmismatch原因と断定するものではない。

E0差約1.38e-14 Ha、lambda_R差約1e-15は大きい数値異常を疑う必要性を下げる材料ではあるが、H、固有状態、DF分割、回路が等しいことの証明にはならない。[R2]

例えば理論上、ZとXは同じ固有値・operator normを持つが、同じ入力|0>に対する信号は異なる。energyやnormの一致だけでは、比較対象の演算子を同定できない。

## 3. read-onlyで確認する範囲と、確認の打切り点

### 3.1 一巡だけ調べるもの

1. 旧pilotのarray snapshot、raw integrals、MO係数、MolecularDataファイル、既存cache、当時のログが実際に残っているか。
2. 旧pilot artifactが持つ情報：full H hash、tail hash、成分確率、cost record、参照energy等。完全なfactor配列・one-body・状態まで復元できるか。
3. hash関数の構成：array bytesだけでなくmetadata、dtype、byte order、index、weight rule等を含むか。フィールド別の旧digestがあるか。
4. authorization、実行source、pilot sourceの対応。Python/package version以外の実行情報が残っているか。
5. 現snapshotのファイル完全性と、保存・読込み時の型／metadataの整合。

read-onlyとは原本を変更しないことを意味する。hash計算・JSON/XML解析はよいが、既存配列から新たなoperatorやsignalを計算する行為は『数値処理なし』とは呼ばず、次の段階へ分ける。

### 3.2 回復不能の場合

旧runnerは、計算中にはdense HやDF配列を作るが、戻り値のpilot recordには入力全配列を保存していない。[R8,R9] ただし別cache等が残っている可能性は、このコードだけでは否定できない。

完全な旧入力を見つけられず、保存情報もそれを一意に規定しない場合、厳密な差分やgaugeの原因はその証拠から同定できない。hashは復元可能な配列保存の代わりではない。

この時点で『原因未同定・旧入力同一性未証明』と記録し、探索を止める。旧hashに偶然戻るまで再生成したり、丸め桁を変え続けたりしない。

## 4. canonical hashは再開の必須条件ではない

### 4.1 三つの同一性を分離する

| 層 | 記録するもの | 目的 |
|---|---|---|
| 保存物の同一性 | raw file hash、array/dtype/order hash | 同じ保存入力を再利用し、改変を検出 |
| 物理モデルの対応 | 固定basis/sectorのHamiltonian、変換があるならstate・observableの対応、明示的な残差 | 物理的な比較対象が同じか／どこまで近いか |
| アルゴリズムの同一性 | ordered DF factors、D/R partition、sampling分布、identity phase、PF順序、compiler仕様 | 近似誤差と実装costを同じ設計として比較できるか |

raw hashをsemantic hashで置き換えない。新seriesではraw snapshotがあるので、gauge不変hashが未完成でも再現可能な実験を行える。

### 4.2 DFのgaugeを全て無害と扱わない

DFは概念的に

\[
H=H_1+\sum_\ell\lambda_\ell L_\ell^2
\]

という形を取る。[W2]

- 実数の符号反転 \(L_\ell\to-L_\ell\) は \(L_\ell^2\) を変えない。一般の複素位相は同じではない。
- 完全に等しい係数を持つgroup内の実直交変換は、group全体の和を保存する。非可換なLでも直交行列の恒等式で示せる。
- しかし、そのgroupの途中でprefixを切ると、個別のD/R分割は保存されない場合がある。
- groupを丸ごと残してH_Dを保存しても、individual fragmentとそのPF順序・sampling表現・basis回路の費用まで保存されるとは限らない。
- near-degenerateな係数を同じと見なす操作は厳密なgauge変換ではなく、新たな近似である。
- 軌道basisの変換でHがunitarily equivalentでも、入力状態・observable・回路費用も対応させなければ同じbenchmarkではない。

従って、**full-Hのcanonical hashが一致しても、部分ランダム化のresource datasetを統合してよいとは限らない。**

### 4.3 導入する場合の最低条件

許す変換group、保持する物理量、保持しないalgorithm情報、縮退／near-degenerateの扱い、zero/pivot規約、scalar phase、serializationを先に定義する。

テストは、許す符号変換でcanonical表示が一致する正例に加え、異なる係数・prefix境界・PF順序を誤って同じと判定しない負例を含める。raw hashとcanonical metadataの両方を残す。

旧pilotとの比較に使うには、**旧入力と新入力の両方**へ同じ規則を適用できなければならない。片方のhashしかない場合にはできない。

数値許容誤差つきの『近さ』はhashの等値関係と同じではない。丸めによる一致だけを同一性認証に使わない。

## 5. 研究方針の修正：主題をhash再現へ移さない

### 5.1 維持するRQ

> 固定したDF Hamiltonianに対し、弱いfragmentを捨てる、決定論的に保持する、ランダムに補完する、という選択を同じcoherent-signal精度で比べると、どの条件で中間的な部分ランダム化が有利になり、その利益をどの費用項が打ち消すか。

partialであることの中心は、rankや分割によって

\[
\text{毎shotのdeterministic work}
\quad\leftrightarrow\quad
\text{tailの誤差・normalization・sample work}
\]

が競合することである。

現行のgeneration-prefix圧縮はprefix PRであり、新しい圧縮algorithmとは呼ばない。PRのresource estimate、composite法、compressed DF、SPRINT/GRADEは既存である。[W3–W6] 狙うのは、特定のmatched taskで費用項を戻したときの使い分けを説明する実証・資源評価である。

### 5.2 完成時に必要な答え

- 中間partial候補がdeterministic／discard／random-dominant endpointに対して、精度条件下でどの位置にあるか。
- 中央tailの軽さをcontrol・PF repetition・normalization補正・shotに戻すと、どの因子が差を作るか。
- 限定された入力と実装で成立する結論と、他条件へ移送していない部分が明確か。

単一H4の順位表や『新しいhashを作った』だけで独立論文の完成としない。新しい説明・公平比較・既知研究との具体的差分が残るかを、結果に基づいて判断する。

### 5.3 今変えるもの／変えないもの

**変える**：入力保存、再生成gateの運用、実験seriesの来歴、STOPと科学的結論の区別。

**変えない**：部分ランダム化を中心にすること、rank 6 anchorとrank 3/9 control、比較target・時間・精度の既定範囲、旧結果の保存、held-outの保護。

snapshot再基準化は、その場でPR-3/4/5/6や停止済み理論系列へ移る理由ではない。

## 6. 再開には別研究そのものが必要か

**新しい実験系列は必要になり得るが、別の研究テーマに変える必要はない。**

旧pilotは来歴付きの探索的証拠として残す。旧pilotのcost値を新seriesのbaselineへ転記しない。

完全な旧snapshotが回復すれば、原本を固定して入力再検査する経路がある。回復できなければ、保存済みS0 development snapshotを新しい研究入力候補にできる。ただし、これは旧snapshotと同一だから採用するのではない。

採用理由は『失敗したS0で最初に保存された、将来のcandidate性能を見て選別していない入力』である。この理由と、energy・lambda_R等の一部情報は既に見たことを開示する。新入力はblindではない。

新seriesでは以下を必要とする。

1. 入力の物理・構造検査、load後のhash検査。
2. 新manifestとsourceを固定し、旧STOPをsupersedeしたと偽らない。
3. 新入力で必要なbaseline、分割、誤差参照を再構成する。
4. 旧pilotの『GO』を新入力の通過証拠に使わない。
5. 同じPR-2研究の継続だが、データ系列は分けると明記する。

全PR-3 pilotや、旧qDRIFTの全成分compileを機械的に繰り返す必要はない。新しいS1接続に必要なものだけを再評価する。

## 7. 検証パッケージ：一件に限定しないが、目的を固定する

| 作業 | 内容 | 新たな数値処理 | 研究判断への役割 |
|---|---|---|---|
| V0 | 旧配列・cache・ログ・hash構成の一巡監査 | 文書・file digest処理のみ。再生成なし | 旧入力回復経路か新series経路かを選ぶ |
| V1 | 現snapshotのload、shape、Hermiticity、sector、state残差、固定fileの再利用検査 | 既存development配列の決定論的処理あり | 新seriesの正しい比較targetを確定 |
| V2 | rank3/6/9のH_D+H_R再構成、ordered prefix G/W、sampling規約 | 決定論的な小系処理あり。cost/勝者評価なし | 捨てる／補うの比較が何の差を測るかを確定 |
| V3 | hash・counter・stop guard等の修正と限定unit test | toy/test処理あり。held-out解析なし | 同じ記録上の問題を再発させない |
| V4 | 改訂済みS1のsignal/control/compile smoke test | 明示的承認後のみ。既定scopeの軽量処理 | 研究比較へ進むsemantic接続を確認 |

V1–V4は新しいamendmentを固定してから実施する。今回の旧STOP artifactを読み替えて直接走らせない。

### V0の終了条件

監査した具体的な保存先と見つかったデータを書き、回復できる／回復できない／アクセス不足を区別して一巡で終了する。無限の再生成によるhash一致探索は禁止。

### V1の最小内容

入力のconstant、one-body、lambda、g、basis/sector、stateを完全保存し、同じsnapshotを2回loadして同じ入力digestであることを確認する。分子buildを繰り返す必要はない。

stateのnormalizationとHに対するresidualを確認する。コードが指定するN_alpha=N_beta=2はS_z=0の条件であり、それだけでS^2=0を証明したとは言えない。『singlet sector』という記述を用いるなら別途確認するか、実際のsector規約へ直す。[R3,R4]

受理閾値は結果前に固定し、近いenergyから旧入力とのoperator距離を推定しない。

### V2の最小内容

rankは既定の3/6/9、G/W双方を同じ新snapshotから作る。one-body・constant・identity抽出を含めH_D+H_Rを再構成する。集合、順序、tail分布を別に比較する。同値なら重複候補をまとめ、不一致なら原因を記録する。

これはresource winner判定ではない。新S1に意味があるかを確認する構造検査である。

### V3で先に直す記録上の問題

`run_s0`のPASS側は `_actual_wrapper_probe` でseed付きRTE trajectoryを構築する一方、payloadは `trajectory_samples_drawn=0` に固定されている。validatorも0を要求する。[R4]

今回のmismatchはその前に止まったため、今回の0件記録とは矛盾しない。再開後には、入力freeze、toy test、wrapper probe、candidate samplingのカウンタを分け、実際に呼んだ処理を正しく記録する必要がある。

また `_actual_wrapper_probe` は主としてbuild/mapping metadataを返し、H4 q1/q8の演算子一致を数値検査する処理ではない。toy unit testでのcontrolled equalityと混同しない。新入力のS1で何を検査するかを明記する。[R4,R6]

### V4の終了条件

旧S1の範囲を上限とし、normalization-corrected mean、control relative phase、Re/Im、短いwrapper pathを接続する。32/128 expected-cost MC、resource winner、held-out signal/cost/rankingは実行しない。

S1が通ったら自動的にS2を始めず、結果packetを返す。ただしS1はcorrectness段階なので、そこで資源優位が出ないことを理由に主研究を否定しない。次の判断は『本来のresource questionを評価する準備が整ったか』である。

## 8. 研究判断を変える条件と、変えない条件

| 観測 | 方針 |
|---|---|
| 旧入力が回復できない | 旧同一系列は閉じる。PR-2は新固定入力で継続可能 |
| 新入力のhashは安定し、構造検査が通る | 来歴問題を解決した。次のS1接続へ |
| G/Wが一致する | 新手法claimをしない。既知PRのresource比較として継続可 |
| 小さなgauge変更で分割が変わる | gauge不変として統合しない。representation/orderを設計要因として保存 |
| 新入力の再構成・state・controlが不整合 | 修正可能な実装問題か、taskの再定義が要るかを切り分ける |
| 後の公平な資源評価で中間partialが負ける | そのとき初めて科学的negativeと、その原因を判断 |
| 同じ文献・同じ条件の既知結果を再現するだけ | 独立した研究主張を縮小する。hash修復を新規性の代用にしない |

## 9. 今回残せるnegative knowledge

残せるのは、**保存配列なしのrecipe＋package version＋hashでは、旧inputのbyte再現と相違原因の同定を保証できなかった**という実装・provenance上の知見である。これは再現性運用の改善に使える。

残せないのは、部分ランダム化の非有効性、DF factor gaugeが実際の原因だったとの断定、資源順位の不安定性、普遍的なcanonicalization不可能性である。今回それらは測っていない。

新しいcanonical hashの研究を始めるより、完全なsnapshotとlayer別記録を整え、部分ランダム化の本来の問いへ戻る方を優先する。

## 10. 直近の作業と停止位置

**旧結果のread-only回復監査と、新series manifest／実装修正案を並行して準備する。**

回復不能なら新固定入力による継続を明示し、旧STOPを維持したamendmentを固定する。その後、V1–V3を一つのパッケージとして行い、通過条件を満たした場合だけ軽量S1へ接続する。S1後に一度まとめて研究レビューする。

途中の細かいunit testごとに研究方針を立て直したり、全てのbranchを独立した新gate文書へしたりしない。成果物は原則として、改訂契約一つ、入力／実装検証artifact一つ、S1結果packet一つへまとめる。

今回提案するのはこのパッケージまでであり、S2/S3、expected-cost MC、追加rank/geometry/precision、held-out開封を承認するものではない。

---

## 参照と確認範囲

以下のRは特記しない限りreview commit固定。

- R1: [amendment v3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf5d3b2405b2bec4f88dd0589b75dd737b13e052/docs/research/pr2_s0_s1_execution_amendment_v3.md)、§3–4。authorization commitの同ファイルともblob照合。
- R2: [S0停止報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf5d3b2405b2bec4f88dd0589b75dd737b13e052/docs/research/pr2_s0_reproduction_stop_c644925.md)。
- R3: [S0 artifact](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf5d3b2405b2bec4f88dd0589b75dd737b13e052/artifacts/pr2_s0_s1_validation/2026-09-28/pr2_s0_validation_v1.json)、environment/provenance/development metadata。held-outは記録済みmetadataのみ参照し、binaryやsignalを評価していない。
- R4: [S0/S1 module](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf5d3b2405b2bec4f88dd0589b75dd737b13e052/src/trotterlib/pr2_s0_s1_validation.py)、`write_snapshot`、`load_snapshot`、`prefix_identity_record`、`_actual_wrapper_probe`、`run_s0`、`validate_s0_payload`。停止分岐についてsource commitともblob照合。
- R5: [runner](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf5d3b2405b2bec4f88dd0589b75dd737b13e052/scripts/run_pr2_s0_s1_validation.py)。
- R6: [tests](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf5d3b2405b2bec4f88dd0589b75dd737b13e052/tests/test_pr2_s0_s1_validation.py)。
- R7: [test XML](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf5d3b2405b2bec4f88dd0589b75dd737b13e052/artifacts/pr2_s0_s1_validation/2026-09-28/pr2_s0_s1_tests_c644925.xml)、suite headerと該当test。
- R8: [pilot source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf5d3b2405b2bec4f88dd0589b75dd737b13e052/src/trotterlib/pr2_pr3_minimal_pilot.py)、`run_pr2_pilot`の入力生成・返却record。
- R9: [pilot artifact](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf5d3b2405b2bec4f88dd0589b75dd737b13e052/artifacts/pr2_pr3_minimal_pilot/2026-09-27/pr2_pr3_minimal_pilot_v1.json)、input/referenceと代表成分record。
- W1: [NumPy show_runtime公式資料](https://numpy.org/doc/1.24/reference/generated/numpy.show_runtime.html)。BLAS/LAPACK、CPU feature、thread等の記録対象を確認。
- W2: [OpenFermion low_rank_two_body_decomposition公式資料](https://quantumai.google/reference/python/openfermion/circuits/low_rank_two_body_decomposition)。DFの二乗one-body分解、correction、rankの定義。今回の実装versionの動作を最新版の文書だけで再現認定しない。
- W3: [Günther et al., Phase estimation with partially randomized time evolution](https://arxiv.org/abs/2503.05647)。部分ランダム化・single-ancilla資源評価が既存であることの確認。
- W4: [Hagan and Wiebe, Composite Quantum Simulations](https://quantum-journal.org/papers/q-2023-11-14-1181/)。既存法合成とpartitionの研究範囲。
- W5: [Oumarou et al., Regularized Compressed Double Factorization](https://quantum-journal.org/papers/q-2024-06-13-1371/)。compressed DF自体の先行研究。
- W6: [Casares et al., Theory and practice of Trotter product formulas for quantum chemistry](https://arxiv.org/abs/2606.30741)。SPRINT/GRADEの統合設計のscope。

今回の公開文献照合は研究の位置付けに必要なscope確認であり、全ての定理を再監査したものではない。PR-2の新規性・採録可能性を確定するものでもない。
