# EW2・認証step比較：GPT／ユーザー向けハンドオフ

2026-10-17 JST。④の未採択A2に関する限定技術検証を完了した。論文原稿、新しいGPT重要科学レビュー、研究採択、科学実行契約ではない。

**今回の主技術分類：`VALID_SYMBOLIC_ADVANTAGE_CONDITION`。** 範囲は、**同じ入力でcenterした実時間交換子対照に比べ、相対形式の十分証拠が大きいstepを許す記号的条件**まで。現実の入力で条件が満たされること、適用可能な既知保証全体への優位、総資源改善を確認したという意味ではない。

## 1. 異なる判断軸を上書きしない

| 軸 | 結論 |
|---|---|
| EW2の数学 | 明示条件内でVALID。逆の存在・積順序・spectral moment・両tail・Wの単調置換・η*を独立検算 |
| 元レビューの交換子対照 | C2の係数・PF順序はChilds v3原文と一致。逆log式・3/2・二次momentの式は成立 |
| 公平な有限step比較 | 元対照はまだ強化できる。同じcapから既知scalarをcenterし、`b_C=h²Cbar2/(1−hΛ/2)` を使った条件(C2)を導出。元H・PF・絶対位相は保持 |
| Xi | 小step・小η・正leading等の条件で整合。d/W→0ではstep比のleadingがXi^{-1/2}。有限stepやGOの判定ではない |
| 既知性 | 交換子対照の接続、S1のrank-oneによる先導交換子の全vector評価は既知定理の直接系。EW2単独のpriorityと相対log全statementの完全還元は未確定 |
| 比較上の実例・費用 | **NO_COMPARATIVE_ADVANTAGE_ESTABLISHED**。実入力の条件成立、best-known envelope、認証取得・native・state費用を評価していない |
| 正式研究状態 | **変更なし**。A1 CLOSED、A2 NOT_ADOPTED、K1-F未承認・未実行 |

VALIDはCodexの紙上検算の判定であり、形式証明器の証明、数値PASS、正式な新定理・新規算法の認定ではない。主分類を `REDUCES_TO_KNOWN_RESULT` として全statementの既知性まで断定せず、部分的な直接還元を個別に記録した。

## 2. 成果と、比較上の重要な留保

- [数学的監査](mathematical_audit.md)：EW2の逆はproof-only。e=0はPSD共通kernelで別処理。ground縮退・混合stateを許す。元EWとの全域の優劣はない。
- [有限step比較・文献表](finite_step_and_prior_art_comparison.md)：`h_R=max(h0,h1)` と、centerした対照のh_Cを定義し、

\[
h_R>h_C\quad\Longleftrightarrow\quad
\overline C_2>\sqrt{\alpha_{\rm spec}}\,d\,
\frac{1-(\Lambda/2)h_R}{h_R^2}
\]

この同値条件は、共通readout/実装条件も満たすときに実行step差へ接続できる。値を計算・forecastしたものではない。
- 同じlog級数の評価でも、実時間unitarityと既知scalar centeringを使えば、元レビューのexp型分母や3/2より強くできる。対照を1/4領域へ固定したり、Cbar2を意図的に粗くしたりして相対側を勝たせない。
- さらにQPE回路安定性、state residual、適用可能な低energy保証、他のtightなinverse-log情報を除外してはいけない。上式はそのenvelope全体への優位条件ではない。
- 可換limitでは対照C2=0でS2が厳密。RF0/粗いRF1が正の上界を返すことから、現相対保証の一様支配はない。新しい有利なPF分割生成規則を今回得たわけでもない。

主要な新しい注意点は、EW2の撤回ではなく、**比較対象をさらに強くしても条件が残るかを評価する必要**である。今回はその強化した一つの対照に対する正確な条件まで閉じた。最良既知法を超えたという主張へは進めない。

## 3. 原本・一次資料の確認状況

指定GPTレビュー全文を取得し、[byte保存copy](inputs/qc_relative_form_post_audit_novelty_value_review_20261017.md)を作成した。SHA256は `12df94860dde8cafe977baf12ae9ceacb35175959c0a7ef8d41fcb24b0f51b4c`。今回の[指示原文](inputs/user_instruction_ew2_step_audit_20261017.txt)も改行を含め保全する。

前回監査 `051df466…`、K0 `58d86727…`、K0.5 `01bdc56a…`、過去GPT `dbe51f18…`、正式状態 `67576526…` を固定commitで参照した。RF0/RF1は依存部分を再確認し、全面的に再開発していない。

Childs `1912.08854v3` PDFを新規取得し、§5.1 Eqs.(146)–(152)・Proposition 16と印刷p.39画像を確認。Mizuta–Kuwahara `2504.20746v1` とHejaziら `2402.10362v1` は前回取得したPDFのbytes/SHA256を再確認し、今回必要な箇所を再読した。特にS1のrank-one特殊化は肯定し、locality・PSD局所項・sector制限を明示した。

高次PF repositoryの指定 `b4a7a820…` evidence mapを固定raw URLで取得し、U1/U3/U7と直接の重複だけを照合した。現物のbytes/SHA256を記録したが、同repositoryを変更・pushせず、全体の再監査・保存値の再解析も行っていない。

必須レビュー・指定一次資料は取得できた。**未確認**なのは、出版版・後続版との全式同一性、全既知結果からの完全還元、EW2のpriority、利用可能な全baseline envelopeの実入力比較、certificate・state・native費用である。補助的な過去論文やClaude原本の全文を新しく全取得したとはしない。[固定証拠索引](evidence_index.json)で版・読んだ範囲・取得限界を区別する。

## 4. 次に科学的判断は必要か

EW2やC2の式ごとにGPT詳細レビューを繰り返す必要はない。今回の通常の技術照合は完了した。新規性・主要claim・④の限定継続を正式に判断する場合には、**有用な入力classと証拠取得費がまだない条件付き成果を研究課題として採用するか**が科学的な判断対象になる。その判断を求めるかはユーザーが決め、必要なら別途レビュー開始を明示承認する。今回はGPT/Claudeを新しく起動していない。

追加研究を依頼する場合の最小候補は、比較表§6の「可換性・fragment構造を残す有限相対剰余と、その安価な認証情報」である。ただし、現在は問題の所在の記録だけで、別定理の探索・新SOURCE・新しい検証契約の準備をしていない。実分子・native費・benchmarkへは移行しない。計算を望む場合には、研究採択とは別の実行承認が必要である。

## 5. 保全・公開・STOP

基点は前回監査commit `051df4661eeffb6c6c4c978f70f43eeeaadfddd6`。専用branch `qc-relative-form-ew2-step-audit-20261017`、独立作業領域 `/tmp/prt-qc-relative-form-ew2-step-audit-20261017`。新規ディレクトリだけに追記し、既存資料・原本・SOURCE・raw・科学契約・authorization・grant・ledger・markerを変更しない。親worktreeのHEAD、local heads、registry、index/exclude、tracked filesと選択原本を開始時snapshotと照合する。

正式状態は[固定decision](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/67576526c4e96ae7235fa645fdf744278d049f8d/docs/research/qc_exact_encoding_direction_adoption_20261017/decision.json)のまま。Exact Encoding現構成はREDUCES_TO_KNOWN_METHODとして区切り済み、④の次の具体課題は未採択。①②③、他のCLOSED/STOP/PAUSED/HOLDも変更しない。研究概要・PROJECT_MAP・global validation inventoryを採択済みへ書き換えない。

共通運用指示2026-10-17 §8.4を適用。科学計算を起動しておらず、恣意的なCPU/RAM/worker上限や共有環境・他jobの変更も導入していない。文書読取・PDF text/image表示、Git/JSON/link/hash確認は文書管理であり、科学的な計算・proof-checkerの実行ではない。

本文commitを公開照合後、[publication receipt](publication_receipt.json)を子commitへ保存する。最終HEADと全公開ファイルのbytes/SHA256は最後に再照合し、その最終SHAをチャット完了報告に示す。receipt自身のSHAを自己参照で埋め込まない。

新規科学数値計算、保存値再解析、Hamiltonian/固有値/RDM生成、symbolic checker、simulation/native compile/T合成、科学SOURCE、科学実行契約、authorization、grant、marker、新しい重要科学レビュー開始、正式研究採択は**各0件**。

**次の担当：ユーザー。** 条件付き成果を保留するか、上記の科学的判断を別途依頼するかを決める。Codexは公開・保全確認で今回の作業を終了する。**Mandatory STOP。**
