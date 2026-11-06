# Claudeレビュー追補：技術調査の報告

2026-11-06 JST / SERVER Codex。独立clone・専用branchで実施したSOURCE読解、一次文献調査、保存metadataの照合。新しい独立科学レビューやMC1–MC5・案Xの正式採択ではない。

## Q1. 二乗則はSOURCEと整合するか

**`CONSISTENT_WITH_ASSUMPTIONS`。** paired-Taylor β_K、signed stage、各stage中央のRTE occurrence、整数short-step割当、Bの積による合成は `SOURCE_CONSISTENT`。SOURCEが実際に評価するのは

\[
\ln B=q\sum_i r_i\ln\beta_K(\lambda_R w_i\delta/r_i).
\]

K≥2の小τ展開に、連続配分r_i∝|w_i|を加えたときに、Claudeのλ_R²T²(Σ|w_i|)²/n_evへなる。実装は各stage最低1 step＋残余の最大剰余配分であり、全stage共通τは厳密には成り立たない。n_ev=qR_outerはRTEEvent数で、量子shot・event内部component draw・compiled gate数とは別。K0のleading係数は1/2。詳細な固定SHA、関数、行番号は[SOURCE_FORMULA_TRACE](SOURCE_FORMULA_TRACE.md)。

対象はlinear H4/H6、1.0 Å、STO-3G、保存tol-only1e-8 DF10/19、元state/reference/fragment順、S2・3段Yoshida4・保存11段m5、T=.8/1.2/1.6/2、δ=T/q、K2/4/6。B2の保存prefix集合とB3のL_D=0を区別する。H6 weighted-Hermitian completionを保持する。この技術整合を全H・全δ・Pauli高次・H12へ一般化しない。

結果carrier `155529e2…` のnormalization/signal/native kernelは役割別の実行SOURCEとbyte一致。一方、pipeline/reportはprovenanceの初期実行hashと異なり、そのhashはcontinuation SOURCE側に一致する。最終reportには保存価格の再利用・固定δ価格・副指標・binding処理等の追加がある。従ってcarrier＝全科学実行SOURCEという扱いは不可。これは今回のnormalization式との矛盾ではなく、最終会計を再現する際の版固定上の留意点である。

## Q2. B2/B3の機構説明へどこまで適用できるか

**normalization機構は双方に対応するが、総費用Gへの因果説明は `PARTIALLY_SUPPORTED`。** B3はone-bodyを保持し二体prefixを0とした経路で、同じstage/配分/normalization実装を使う。全Hamiltonian純RTEとは異なる。

長時間B2で条件によって高次が低費用となり、B3で高次nomineeがS2より高費用となることは、固定REPORTにある保存傾向として保持する。A²だけでGの比や勝敗を説明したとは認定しない。λ_R、PF bias、finite-RTE bias、headroom、N、C、q、R/K、prefix、候補選抜が同時に変わる。Claudeの1点log_B比も今回再計算していない。近似残差や整数配分寄与の大きさを評価するには、承認されたB2再解析が必要。

## Q3. どこまで先行研究に存在するか

**Güntherらv2 App.A.3 (A36)–(A40), Lemma A.3が `DIRECT_PRIOR_ART`。** signed stage、絶対時間、stageへの比例配分、Bの指数和、絶対時間の二乗と整数丸め項を明示する。Claudeが未読としたD/Eに加えてA.3まで確認したことで、中心となる理論構造の直接重複が分かった。[一次論文v2](https://arxiv.org/pdf/2503.05647v2)

Hagan–Wiebe §5は高次composite＋qDRIFTのchannel誤差/query上界、Pocrnic §II.5/§Vはinner/outer次数・分割・時間の数値競合、Kanasugi §II/App.Bは分割・randomized資源・shot/RPE会計の先行である。同じ「高次」「randomized」でもlawと資源の意味を区別した。[PRIOR_ART_MATRIX](PRIOR_ART_MATRIX.md)に固定版、式・定理、分類、差分と未確認範囲を記載した。

## Q4. MC2の新規性評価に残るもの

二乗構造自体を「初めての原理」とする根拠にはできない。差分候補は、保存m5を含むfinite K・最低step付き整数配分、DFのB2/B3・強い決定論対照・full-wrapper compiler価格・独立確認cohortを接続した具体的な実証である。それが論文の中心新規性になるか、既知理論の条件付き資源検証として位置付けるかは、ユーザーの科学判断へ戻す。

本調査は新規性がないという全面否定や、案Xの採択／棄却を行っていない。有限Kと整数配分の差だけで独立新理論と認定することもしない。2025–2026年の関連検索を行ったが、網羅性・未読文献における非存在は主張しない。

## Q5. RZ以外の指標は取得できるか

**多くは保存されており、同一cohortへの結合を条件にcompileなしのB1が可能。全比較でcoverage完了とは言わない。** [METRIC_AVAILABILITY.csv](METRIC_AVAILABILITY.csv)は数値比較表ではなく、存在と適用可能性の9行台帳である。

| 比較 | 確認した保存範囲 | 残る点 |
|---|---|---|
| DF H4/H6長時間 | per_circuit_metrics、saved_secondary_metrics、探索／確認JSONにRZ/CX count・各filtered depth・total depth・circuit_size | mean表とcohortの対応、非RZのpaired SEに必要なraw trial |
| DF B3 | secondary_native_metricsと確認JSONに同じ六指標、id/cohort/axis | 別保存baselineのcohortとraw uncertaintyを保持 |
| Pauli H2/H4/H6 | H2/H6公開archive内のresult JSONに六指標・QPY metadata、H4表にRZ/CX/depthのtrajectory値 | 最新純RTE／Hybridの全extension・旧baseline coverageの結合は未完了 |
| H8 Stage C | 各確認trialの六指標。固定PFもStage Bの同一IDに六指標 | 表示CSVはRZのみだが原JSONに副指標あり。新mean/G/SEは未算出 |
| H4固定実装比較 | centralのpaired表と代表metric JSONにCX/depth等・gate histogram | quoted summaryはRZのみ。generic Pauli／DF側の非RZ raw coverageは未確認 |
| P3 | Stage3/run04に六指標・histogram・QPY hash。Stage4でRZ/CX/depthの費用会計済み | rz_depth/cx_depth/sizeのGは未会計。既存三指標の引用は再解析不要 |

RZはtranspile後 `post_counts.get('rz',0)` で、角度によるClifford除外をしない（[provider rte_compiled_cost.py L830–893](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c31ef424175c02379eafebe8fc6c7548c1d9aabe/src/trotterlib/rte_compiled_cost.py#L830)）。Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17を固定したfull Hadamard wrapper価格で、SX/X/CXは別operation。DFはsymmetric-directional、Pauli主系列はcentral制御。RZを非Clifford回転数、T count、物理runtimeと解釈しない。depthはQiskitの回路／gate-filtered depth、sizeは同ライブラリのsizeで、unitary gateのみの総数とは区別する。

SXはP3 output_histogramとH4 central代表histogramに存在するが、主要全cohortで独立統計が保存されているとは認定しない。非Clifford分類は保存されていない。PauliのQPYはarchive内の存在とmetadataを確認しただけでdecodeしていない。DFは代表QPY保存というSOURCE記載を、全trajectoryの回路保存と読み替えない。Track B native T/CXは別資源体系であり、Track AのRZやCXへ無条件に換算しない。

探索・独立確認・確認後min・再利用価格を区別した。DIRECT、モデル、外挿の区分を保持し、screening proxyをcompiled価格にしない。今回の主対象で確認した保存副指標と、RPEの長qモデル／外挿価格は別である。

## Q6. B1/B2は判断を変え得るか

**推奨する。ただし目的を絞り、別承認後に実行する。** B2-aはexact finite-K式、整数配分leading式、連続A²式を順に分ける。B1はRZで固定したnomineeのCX感度から始める。これにより、MC2を定量的に整合する説明として使える範囲と、主要比較の指標依存性が変わり得る。

費用G全体のfit、最適B²や最良candidateのmetric別再選択は追加の範囲。Claudeの「約10%一致」を新しい固定科学閾値へ採用しない。保存精度、近似次数、元allowance、比較目的から受入条件を定める。詳細は[MINIMAL_NEXT_VALIDATION](MINIMAL_NEXT_VALIDATION.md)。

## Q7. 今回できた作業と次の境界

今回完了するのは指定資料の取得、SOURCE・hash/provenance照合、一次文献の該当箇所確認、保存schema/候補/cohortの確認、成果物作成、技術検査、commit・通常push・remote本文/blob照合まで。

次の科学的保存値再解析（B1/B2）、新規normalization/N/C/G/SE/比/fit/順位、QPYからの新gate計数、simulation・sampling・compile、案X／MC2の正式判断、論文執筆は別承認。今回それらは実行していない。

## 技術確認・保全・限界

- 元の指示はLOCAL担当からSERVER担当へ読み替えた。同サーバーの専用temporary directoryにGitHubから新cloneを作り、新branchを使用。既存repositoryのdirty/untrackedを保全し、元checkoutには書いていない。
- `PROJECT_MAP.md`、研究概要、共通運用指示2026-10-25の§8.1/8.4/10、cloneのAGENTSを確認。AI sub-agentは使わなかった。共有環境・他ユーザーprocess・job・設定を変更せず、空きCPU/RAM/diskを確認。低い固定worker上限は設けず、軽量なI/Oと静的読解だけを実施。
- Claudeレビューはサーバーで参照可能な同日・同題の `(1)` なしMarkdownの指定節を読んだ。指示にある `(1).md` 別名とのbyte同一性は未確認。元ファイルを変更・公開していない。hashと直接照合範囲をVALIDATIONに記録。
- 統合資料・O2台帳・11図表の固定版を参照し、`CURRENT_CLAIM_SCOPE_EVIDENCE_SUFFICIENT`、正式研究status、既存manifestを更新していない。先行研究調査は全履歴の科学的再レビューではない。
- 保存JSON/CSVのfield・pointer、Git blob/hash、SOURCE ASTの定義・呼出構造を技術確認。既存catalog validatorのpointer/CSV helperを再利用し、科学runnerや既存の会計・再解析validatorは実行しない。人工の科学数値テストも不要と判断した。
- 公開対象は本directoryの新規文書・metadata・静的validatorだけ。review原本、raw input、seed値、event列、QPY、内部host/path、認証情報を含めない。SOURCE_INDEXの参照先は既に公開された固定Git blob。

技術結果は[VALIDATION.json](VALIDATION.json)。CIや外部再現の証拠ではない。全private rawのcustody・依存閉包・全cohort coverage、非RZ費用順位、漸近式の全candidate精度を確認済みとはしない。

## 公開と引渡し

専用branch `codex/o2-claude-review-followup-20261106` へcommit/push後、remote SHAと公開全成果物のblobおよび本文を照合する。公開commitと照合結果は最終チャットへ記載する。照合後は **mandatory STOP**。B1/B2、案X、MC2新規性の最終判断、論文作成の次担当はユーザーである。
