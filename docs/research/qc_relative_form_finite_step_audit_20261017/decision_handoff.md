# GPT・ユーザー向け判断用ハンドオフ

2026-10-17 JST。④「PRに限定しないアルゴリズム改善」の、有限step相対形式保証に関する**限定技術監査の完了記録**。論文草稿・新しいGPT科学レビュー・研究採択記録・実行契約ではない。

**最終技術分類：`VALID_WITH_UNRESOLVED_NOVELTY`。** VALIDはCodexの紙上検算の範囲を指す。正式な新定理の受理・独立priority・有用性の認定とは区別する。

## 1. 結論と根拠

| 対象 | 検算結果 | 追跡箇所 |
|---|---|---|
| RF0 | 仮定内で成立。二端telescopingの全語を数え、非可換な内部積を保持。複素log、偶対称性、Cauchyから定数32を再導出 | [数学的監査 §1–2](mathematical_audit.md) |
| RF1/K0 | 成立。K0 §4の分散・行和にはground固有方程式が不要。全vectorのD相対形式と、RF0からの四次剰余512を確認 | [数学的監査 §3](mathematical_audit.md) |
| EW | η<1とexact ground支持で成立。上側Markovと下側支持を別に使う。ground縮退を許す | [数学的監査 §4](mathematical_audit.md) |
| QPE/step | 固定unitary powers、既知支持とunwrap、一様読出し保証・別実装予算の下で成立。scalar Lを一回復元 | [数学的監査 §5–6](mathematical_audit.md) |
| 補助式(29) | 非自明eventには `0<|h|d≤π` が必要。空event、h=0を別扱いする補足。中心証明への依存なし | [数学的監査 §7](mathematical_audit.md) |
| 新規性・費用 | 全statementの直接包含は未確定。強対照への実用優位なし。文献の低energy/gap-free保証や先導係数を新発見としない | [文献・実用条件表](prior_art_and_practical_conditions.md) |

中心命題を破綻させる重大な欠陥は見つからなかった。元レビューのbyte原本は変更せず、補助式の適用域をこの監査に明記した。旧K0のscalar剰余・gap付き固有枝保証を上書きしていない。

## 2. 確定していない科学的判断

1. **新規性の位置**：指定3論文とK0/K0.5は一次本文で照合したが、全ての既知定理からの直接還元を排除したわけではない。PSD・log・Cauchy・Markov・QPEの道具は既知である。新しい定理・生成規則と呼ぶ判断は残る。
2. **研究として追う価値**：有限h領域、32/512、W/Λ、認証されたK_SOS、classical certificate費用、controlled実装を払っても強対照より有利かは未取得である。実数値を評価していない。
3. **次の課題と承認**：A2を採択するか、さらに限定した調査を依頼するか、ここで保留するかはユーザーの判断である。本監査は候補・条件・thresholdを正式採択しない。

数学的検算の同じ作業をもう一度行うためだけのGPT詳細レビューは不要である。**新規性・主要claim・④の優先方針を決める段階へ進むなら、GPTによる科学的判断の価値がある。** 理由は証明の単純な再確認ではなく、既知保証への包含と、費用証拠未取得のまま研究課題として採用する妥当性である。そのレビューを開始する場合はユーザーの別途明示承認が必要であり、今回は開始していない。共通指示の区分ではこの研究方針・新規性判断はLevel 1（GPT-6 Pro推奨）に相当する。モデルを自動変更・起動したものではない。

将来数値・実装を検討する場合にも、その科学scope・SOURCE・比較条件を別途定め、ユーザーの実行承認を要する。この文書から実験契約準備・入力生成・benchmarkへ移行しない。

## 3. 取得資料と不足の区別

取得済み：指定GPTレビュー全文（exact bytes保存）、今回指示原文、2026-10-17共通運用指示、固定K0/K0.5/過去GPT/正式状態、指定arXiv v1 PDF 3件の関連定理・証明・補遺。固定commit・ファイルhash・読了範囲は[証拠索引](evidence_index.json)。原文の確認と全既引用文献の証明再監査は同じではない。

不足・未確認：網羅的な文献priority、各arXivと出版版・後続版の完全同一性、原レビューの補助abstract群の全文、過去資料が参照する全Claude原本の再取得・全文照合、化学でのcertificate費・native費・state準備費。これらを取得済み・実測済みと扱わない。今回の中心命題の紙上成立性はこれらの不足に依存していない。

## 4. 正式状態・保全・公開手順

基点は [正式状態commit `67576526c4e96ae7235fa645fdf744278d049f8d`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/67576526c4e96ae7235fa645fdf744278d049f8d/docs/research/qc_exact_encoding_direction_adoption_20261017/decision.json)。Exact Encodingの限定構成はREDUCES_TO_KNOWN_METHODとして区切り済み、④は研究課題選定段階で次の具体課題は未採択。

- A1の現PF-native分割生成探索：**CLOSEDのまま**。
- A2：**NOT_ADOPTEDのまま**。
- K1-F：**未承認・未実行のまま**。
- 他のCLOSED/STOP/PAUSED/HOLD、①②③、M1-Sの既存結果とmarker：変更しない。

独立作業領域 `/tmp/prt-qc-relative-form-finite-step-audit-20261017`、専用branch `qc-relative-form-finite-step-audit-20261017`。共有Git objectを参照する独立checkoutを用い、親worktreeのindex/ref/registryを更新しない。既存ファイルは一切変更せず、この新規ディレクトリだけを公開する。研究状態・数値validationの変更がないため、PROJECT_MAP・研究概要・validation manifestへ採択済みと見える更新を加えない。

公開記録は [publication_receipt.json](publication_receipt.json)。payload固定commitを先にpushしてraw bytes/SHA256とremote SHAを照合し、その結果をreceiptの子commitへ保存する。receipt自身のhashを同じreceiptへ再帰的に埋め込まない。最終HEADでの全追加ファイルの公開再照合を行い、その最終SHAは完了報告に示す。

ファイル/schema/link/hash検査と保全検査は文書管理である。数式・SOURCE・数値correctnessが自動testでPASSしたとは報告しない。PDF読取りとhash以外に科学的計算は行っていない。共通指示2026-10-17 §8.4を適用し、人工的なCPU/RAM/worker上限、共有環境設定や他jobの操作を導入していない。

新規科学数値計算、保存数値再解析、symbolic checker、Hamiltonian生成、simulation/native compile、科学SOURCE、科学実行契約、authorization、grant、marker、新しいGPT詳細レビュー開始、研究正式採択は**各0件**。

## 5. 次の担当

**ユーザー**：この技術監査を踏まえ、追加の科学的判断を求めるか、未採択のまま保留するかを決める。追加判断を求める場合だけ、対象を限定したGPTレビューの開始を別途指示する。**Codex**：今回の公開・保全確認で終了し、次の探索・実装・計算を開始しない。

**Mandatory STOP。** この監査はA2採択も、停止解除も、科学計算の承認も与えない。
