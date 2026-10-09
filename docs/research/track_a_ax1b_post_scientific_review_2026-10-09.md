# Track A：AX-1b後の独立研究レビューと研究方針決定

**副題：DF-native Partially Randomized Trotterの実コンパイル資源効率、競争力の成立条件、費用予測の適用限界**

| 項目 | 内容 |
|---|---|
| レビュー日 | 2026年10月9日（JST） |
| 文書の位置づけ | GPTによる科学的判断・詳細研究レビュー。実装契約や数値実行のauthorizationではない |
| 対象研究 | Track A（PR資源評価の拡張研究）。同じプロジェクト内の他研究へ無断適用しない |
| 対象repository | [`HIROMU1015/Partially-Randomized-Trotter`](https://github.com/HIROMU1015/Partially-Randomized-Trotter) |
| 主な証拠commit | AX-1b結果 [`b2e1bf65e21893b6c617223b42313623d3186f12`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/b2e1bf65e21893b6c617223b42313623d3186f12) |
| 実行時source | `fc297cd9ab840018c4f35b2764d0b8be07c57285` |
| 原科学実験の追加 | なし。既存保存値と研究文書のレビュー、文献照合、数学的考察のみ |
| **研究レビュー判定** | **研究方向は決定。Track A継続・AX-2A設計へGO。AX-2Bの新規科学計算、AX-3/H6本検証、AX-4/H8は本書では認可しない** |
| 次の主担当 | **Codex：AX-2Aの設計・実装・synthetic/既存H4 correctness準備。研究的に重要な新結果を得たらGPTへ戻る** |

> **証拠の読み方**：本文では **[観測]** を既存の保存出力・sourceに基づく事実、**[解析]** をその数学的・科学的解釈、**[提案]** を今後の方針、**[未検証]** を証拠不足の仮説として区別する。パーセンテージは、特記のない限り2026-10-09付AX-1b保存結果の表示値であり、独立な追試ではない。参照したGitHub・一次文献は末尾に列挙する。

---

## エグゼクティブ・サマリー（最終判断）

1. **Track Aは継続する価値がある。** ただし、H6/H8へサイズを拡張すること、PRの勝ちを再確認すること、回帰モデルの相対誤差をさらに下げることだけでは独立論文の中心貢献として弱い。
2. **主研究課題はRQ-Rとする。** 同一finite-time coherent-signal taskで、PR、discard、決定論PFの**直接的な測定込み実コンパイル資源**を、要求精度・実装・DF表現・系サイズの条件をそろえて比較し、PRの競争力が成立する範囲と理由を明らかにする。
3. **RQ-P1は補助課題として残す。** 参照費用を見ない時点で何を知ればone-shot費用を予測できるか、少数パラメータモデルの有効範囲・失敗範囲を調べる。H4での追加のfit探索は行わない。
4. **RQ-P2（truth-free operationalなshot・総資源・候補選択）は未達成として分離する。** その完成をTrack Aの必須条件にはしない。AX-1bのconditional-oracle regretを、未知分子での選択性能と表示しない。
5. **強いbaseline、数値headroom、DF policy、sample uncertaintyが優先論点である。** 高次PF、対称controlled synthesis、複素誤差分解、誤差予算の感度を最小限の検証で扱う。先行研究ですでに扱われている資源評価との差分を特定する。
6. **H4→H6→H8の段階設計は維持する。** H6は最初の新サイズでの開発・サイズ移送検査。H8はH6までで凍結したモデル・説明仮説・比較条件の独立評価に使う。ただしH6/H8の3点だけで漸近スケーリングを主張しない。
7. **次はCodex。** AX-2Aで研究契約の技術化、既存source接続、synthetic tests、H4比較・H6技術pilotの準備をまとめて任せる。大きな科学計算を許可する前の重要な判断はGPTが担う。

### この研究で最終的に答えたい問い

> 同じDF Hamiltonian、初期状態、時間、complex-signal誤差、測定・制御回路費用、探索規則の下で、部分ランダム化はどの条件で競争力を持つか。その結論を変える要因は、回路費用・finite-RTE normalization・bias margin・実装規則のうち何か。また、参照結果を知る前に取得可能な情報だけで、その競争力をどこまで判断できるか。

この問いは**新PRアルゴリズムの提案**ではない。アルゴリズムの評価・適用範囲・予測可能性を対象とする研究である。

---

# 第I部　現在の証拠とAX-1b結果の意味

## 1. 到達点と来歴

### 1.1 Track Aの実験履歴

| 段階 | 実施内容 | 現在証拠と扱い |
|---|---|---|
| M1-A/B1 | H4 1.00 Å、DF rank12、T=0.8、登録210構成のcomplex signal・full-wrapper compiled cost | 開発結果。random costは保存された32 trajectory標本に基づく |
| M2 | H4 1.30 Åの事前固定5構成でtransfer | 以前のheld-out geometry実験だが、今回のAX-1bでは**既観測診断**。探索全体の最適性ではない |
| PM-0 | 既存結果の帰属・同一R等の比較 | 保存値の事後解析。独立標本ではない |
| PM-1 | B0 discardの近接prefix 4/5 × q=1,2,4,8の8構成を追加 | H4 development。新しいPR優位を保証しない |
| PM-2 | 保存bias・費用を使い精度gridを再会計 | 追加の独立signal計算ではない |
| AX-0 | RQ、文献との差分、DF policy、予算、候補設計の研究契約 | 事前方針と未解決事項の定義 |
| AX-1a | 保存データの利用、fit、group診断、regret、情報区分を事前固定 | 結果を見る前の解析契約 |
| **AX-1b** | 保存データを1回解析し、モデルfit、cross-fitted診断、条件付きregret、normalizationを照合 | **正常完了。研究方針の判断材料** |

AX-1bのterminalは`AX1B_COMPLETE_WITH_DECLARED_NA`。出力は12登録ファイル、45入力のbyte hash/schema照合、197件のfinite distribution会計、PM-2の67,346行照合を報告。launch 1回、retry/resume 0、実行約8.74秒、peak RSS約296 MiB、12出力合計約9.85 MiB。これらは**古典解析の運用記録**であり、量子回路の使用資源ではない。保存検証は`READ_ONLY_VALIDATION_PASS`。170 synthetic testsは保存auditとして確認され、今回の独立再実行・CIではない。参考：[AX-1b結果レビュー][R1]、[terminal][R2]。

### 1.2 評価taskと「費用」の定義

ターゲットは保存されたDF Hamiltonian \(H_{\mathrm{DF}}\)、正規化状態\(|\psi\rangle\)、有限時間\(T\)に対する

\[
z(T)=\langle\psi|e^{-iH_{\mathrm{DF}}T}|\psi\rangle.
\]

実部・虚部のHadamard測定を含む**full measured wrapper**を対象とする。保存compilerはQiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなし。初期状態準備費用と実機での量子shot実行は含まれない。主要目的変数は、各軸の**one-shot RZ count**および十分shot数を掛けた**RZ-work**。RZ countは任意角回転やT/Toffoli数、RZ layer depth、物理runtimeと同義ではない。

方式は以下を比較する。

- **B0**：one-body/scalarとdeterministic prefixを残し、tailを捨てるdiscard方式。biasにはdiscardと残存PFの寄与が混在する。
- **B1**：全DF blockを決定論的に扱う二次PF。
- **B2**：deterministic prefixとcanonical finite-RTE tailの部分ランダム化。
- **B3**：prefix 0のrandom-dominant方式。ただし**one-body/scalarまで全てランダム化する方式ではない**。

M1では、主にprefix \(L_D=0,3,6,9,12\)、\(q=1,2,4,8\)、random cutoff \(K=2,4\)等、登録した\((q,r,K)\)の候補を評価。H4 1.00 Åと1.30 Åの違い、基底・geometry・DF政策の違いは常に区別する。詳細：[AX-0 benchmark契約][C3]。

## 2. AX-1bのモデルと情報制約

### 2.1 モデル

**Action index（未校正）**：

\[
A_{\mathrm{exact}}=n_{\mathrm{det}}+\mathbb E[n_{\mathrm{rand}}]+n_{\mathrm{fixed}},
\qquad
A_{\mathrm{ceil}}=n_{\mathrm{det}}+\lceil\mathbb E[n_{\mathrm{rand}}]-10^{-15}\rceil+n_{\mathrm{fixed}}.
\]

単位は**action count**。compiled RZ数と同じ単位の誤差を出さない。

**SINGLE_COEFF（校正モデル）**：

\[
\widehat C_a=\beta_a A_{\mathrm{exact}},\quad\beta_a\ge0,
\]

軸\(a\)ごとの非負係数をM1 trainingでfitし、interceptなし。

**FEW_PARAM（少数パラメータモデル）**：

\[
\widehat C_a=\theta_{0,a}+\theta_{D,a}n_{\mathrm{det}}+\theta_{R,a}\mathbb E[n_{\mathrm{rand}}]
+\theta_{F,a}n_{\mathrm{fixed}}+\theta_{q,a}q,\qquad\theta\ge0.
\]

M1 210候補だけで、train-only scaling・列従属検査・非負最小二乗（NNLS）を用いる。PM-1とM2は学習に含めない。coefficientsは**条件付き統計モデルの係数**であり、各回路操作の独立・因果的な実費用と解釈しない。

**STRUCT_ACCOUNT**：basis transition・ordered event・control・boundary cancellationの情報がallowlistでは不十分なため、N/A。

**Finite normalization accounting**：source定義に従う\(b_K(\tau)\)と\(B=b_K(\tau)^{qr}\)の会計。これはRZ予測モデルやbias predictorとは別である。

### 2.2 情報時点の区別

| 区分 | 利用情報 | 主張可能範囲 |
|---|---|---|
| 未校正index | action/q等、登録した計算可能な特徴 | 順位の参考指標 |
| 校正費用モデル | H4学習標本のone-shot compiled cost＋targetのI1特徴 | one-shot費用予測。未使用サイズは未検証 |
| **conditional-oracle** | 上記＋候補ごとの**参照bias・参照shot・参照eligibility** | その条件を知ったうえでの費用予測・選択損失 |
| operational | 参照biasやtest costを知らずにshot・eligibilityを予測 | **未実装・未検証（N/A）** |

AX-1bの`regret`はconditional-oracleであり、未知分子の真の総資源を実用的に予測した結果ではない。情報取得の古典計算コストも新しいサイズではUNKNOWNである。[AX-1a評価契約][C5]、[shot availability][R3]。

## 3. 費用予測の主要結果

### 3.1 Primary / group診断

| 評価条件 | SINGLE MARE | FEW MARE | 重要な注意 |
|---|---:|---:|---|
| M1 in-sample | 72.1751% | **8.6875%** | 開発集合に対する再評価 |
| q-fold均等平均 | 72.7447% | **8.9043%** | 事前固定complexity gateの主判定 |
| Pooled q-OOF | 72.7665% | **8.9096%** | 異なる学習集合でfitしたモデルの集計 |
| PM-1固定8構成 | 43.1147% | **3.4205%** | 既観測prefix診断で独立選定ではない |
| M2固定5構成 | 58.6569% | **5.7600%** | 既観測1.30 Å geometryの限定診断 |
| B3を学習から完全除外 | 249.5685% | **48.3696%** | method外挿。構造の種類を変えたときの弱点 |

**[観測]** complexity gateは`PASS`で、採用は`PRED_BASE_FEW_PARAM`。4つのq-fold全てでSINGLEより低誤差、coverageはどちらも100%。pooled OOFで「10%超の過小評価」はSINGLE 53.3333%、FEW 4.7619%。ただしFEWのpooled最大相対誤差は86.1507%であり、全候補に対する一様な誤差保証はない。[AX-1bレビュー][R1]。

FEWの選択後も、B3除外でMARE 48.37%、最大135.90%を示したことは重大な制約である。B3とprefix 0のholdout集合は重なるため、これは独立した2種類の検証ではない。`method`と`prefix`の交絡があり、原因を`basis transition`と決めつけない。

**[解析]** action 1回の意味が不均質なため、SINGLEの比較的な不調は自然である。既存sourceではdeterministic fragment、one-body/fixed action、RTE component applicationは、同一のnative RZ量に対応していない。FEWによる改善は「操作種別を区別すべき」という主張を支持するが、各係数が物理的費用を一意に識別することまでは示さない。[action source][S4]。

### 3.2 凍結係数（H4 full210 fit）

| 係数 | 保存値 |
|---|---:|
| SINGLE：\(\beta\) | 343.72166538309943 |
| FEW：intercept | 0 |
| FEW：\(\theta_R\)（expected random applications） | 77.86907618903217 |
| FEW：\(\theta_D\)（deterministic） | 793.2211607640618 |
| FEW：\(\theta_F\)（fixed） | 213.1315997001303 |
| FEW：\(\theta_q\) | 121.25057418130754 |

**[観測]** sourceで保存されているfit係数。一般のH6/H8へのパラメータ移送を保証しない。特にorbital数、各fragmentの内部rank、actual basis transformの費用が明示的にない。H4からH8へ持ち出して外れること自体は、構造情報欠如による当然のミスマッチである可能性がある。後の評価では、**旧FEWは凍結した対照モデル**として残す。[model_fits][R4]。

### 3.3 Action indexの順位診断

保存SpearmanはM1で約0.840、PM-1で1.000（8構成）、M2で約0.821（5構成）。順位診断としては参考になるが、同一グループ内での単調関係が強くても、RZの絶対値や重要な最良候補の順位が再現されるとは限らない。PM-1/M2の標本数も少ない。`A_ceil`をRZ予測と同義に扱わない。

## 4. Conditional-oracle資源・選択損失

### 4.1 固定評価の定義

各方式\(x\)について、参照shotを使用し、

\[
\widehat G_{\mathrm{conditional}}(x)=\sum_{a\in\{\cos,\sin\}}N^{\mathrm{ref}}_{x,a}\widehat C_{x,a}
\]

を候補選択用に評価する。参照bestは同じeligible集合\(\mathcal X\)内の\(G_{\mathrm{ref}}\)最小値。Regretは

\[
\mathrm{regret}(x_{\mathrm{chosen}})=
\frac{G_{\mathrm{ref}}(x_{\mathrm{chosen}})}{\min_{x\in\mathcal X,\,\mathrm{eligible}}G_{\mathrm{ref}}(x)}-1.
\]

正確な対象は固定候補集合内の参照shot×保存費用であり、探索空間全体の最適性ではない。

### 4.2 H4 development218の結果

| \(\epsilon_{\mathrm{sig}}\) | 参照eligible数／218 | SINGLE regret | FEW regret |
|---|---:|---:|---:|
| 0.05 | 214 | 7.4952% | **1.4753%** |
| 0.01 | 163 | 2.9633% | **0.5631%** |
| 0.005 | 151 | 3.5642% | **0.8122%** |
| 0.001 | 101 | 0.6208% | **0.0000%** |

**[観測]** この固定開発集合ではFEWが4 anchorすべてで低regret。PM-1追加8構成は0.05では8件eligible、厳しい3 anchorでは0件eligibleで、参照最小値を下げず、選択された候補にもならなかった。210候補pooled cross-fitted診断で同じregret値が得られたが、**異なるfoldモデルを組み合わせた内部診断**であり、full210 fit→218候補への診断と同一の検証ではない。[AX-1bレビュー][R1]。

### 4.3 M2固定5構成での選択

| \(\epsilon_{\mathrm{sig}}\) | eligible候補数／5 | SINGLE regret | FEW regret |
|---|---:|---:|---:|
| 0.05 | 5 | 0% | 0.3577% |
| 0.01 | 5 | 0% | 0.3510% |
| 0.005 | 1 | 0% | 0% |
| 0.001 | 1 | 0% | 0% |

**[解析]** FEWはM2のMAREを減らしても、固定5候補内のregretは常にSINGLE以下とはならない。\(\epsilon=.005,.001\)の0%は候補が1件のみ適格なので、選択モデルの良さの証拠にならない。また、\(~0.35\%\)差が統計的に確定した優劣とも言えない（random costの標本数32、cross-candidate covariance未評価）。

### 4.4 費用精度と選択性能は異なる

単一軸・共通の正比例係数\(\beta\)を用いる場合、

\[
\widehat C_x=\beta A_x\quad\Rightarrow\quad
\arg\min_x N_x^{\mathrm{ref}}\widehat C_x=
\arg\min_x N_x^{\mathrm{ref}}A_x.
\]

従って、\(\beta\)がRZを大きく過小/過大評価しても、**同じ比較集合・同じ参照shotでの候補順位は変わらない**。現実の二軸でも両軸の係数が同じ場合は同様。ただし軸ごとに係数が異なる場合やfoldごとに別係数のOOF集計には無条件に当てはまらない。

FEWは平均誤差を小さくしても、最良候補の僅差に対して相対順位を変える可能性がある。研究では「絶対資源予算予測」と「候補選択」のどちらを評価しているかを分離する。H4内の過去データでのregret改善から、未知H6/H8でのoperationalな最適構成選択は主張できない。

## 5. 結果の統計・方法上の弱点

### 5.1 Random cost sampleとwinner's curse

random候補197件は各32 trajectory、deterministic候補26件は各1 trajectoryを用いている。paired cosine/sineの共分散は保存されている。ただし**32個は回路費用標本数であって、量子実験のshot数ではない**。rare-orderの高費用イベントは未観測の可能性があり、点推定\(\pm2\mathrm{SE}\)は**engineering interval**に留まる。同時信頼区間・全候補でのformal winner保証ではない。[AX-1bレビュー][R1]。

最小費用候補を多数から選び、その同じ費用標本で利益を見積もると、偶然安く見えた候補が選ばれる選択バイアスがあり得る。今後は主要な競争候補と境界候補について、追加独立cost samplingまたは既知のfinite-order確率を使った層化診断の情報価値を検討する。ただし結果後に都合のよい候補だけを救済する選び方を避け、登録方針を明示する。

### 5.2 Eligibleと欠測

今回の実行は、参照eligibility未確定0、予測欠測0、common-support除外0。144のstandalone診断のうち138がvalid、6がeligible集合空だった。**有効coverage100%は、未知サイズでも100%の保証ではない**。候補がaccuracy限界の近くにあるときは、数値誤差によりtrue/false/undeterminedが変わり得る。[shot availability][R3]。

### 5.3 同一保存値の再利用

PM-2の多精度gridは同一signal/costの再会計であり、各精度点は独立な追加実験ではない。M2の5構成は以前の移送検証で既観測であり、今回のAX-1bが未知geometryでの盲検予測を新たに達成したわけではない。

---

# 第II部　先行研究、新規性、研究方向の選択

## 6. 先行研究の位置づけ（2026-10-09時点）

### 6.1 原論文との関係

**Günther et al., _Phase estimation with partially randomized time evolution_**, arXiv:2503.05647v2（2026-07-10改訂）、PRX Quantum 7, 020332 (2026)。部分決定論・部分ランダム化、single-ancilla QPE、量子化学benchmark、具体的な資源会計、水素鎖のサイズ依存を扱う。[P1]

**研究上の禁止claim**：「PRの資源評価を初めて実施した」「原論文は抽象boundしか計算していない」「二次PF＋PRの水素鎖資源削減を初めて示した」。いずれも妥当でない。

ただし原論文の評価は、主としてエネルギー/QPE総資源、特定のbasis・ordering・合成規則を使う。一方Track Aは、STO-3G、別geometry、DF-native finite-RTE、有限時間complex signal、実際のfull-wrapper compiler後RZ等を使う。**同じtask、単位、targetへ翻訳できない項目は直接数値比を出さない**。元論文側の計算で本研究と同じ条件の「結果が未検証」かどうかを見極めることが、差分の評価になる。[AX-0 model対応][C2]

### 6.2 他の強い先行研究

| 資料 | 確認された内容 | 本研究への含意 |
|---|---|---|
| **Miller et al., _phase2_ (2026)** [P2] | PRも扱うPauli-rotation state-vector simulator。40-qubit量子状態での時間発展を実行。Trotter誤差係数の信頼できる抽出は最大32 qubitsまで | 「より大きなサイズでPRを数値計算した」のみでは新規性が弱い。**実行可能サイズと誤差を信頼できるサイズは別** |
| **Casares et al., SPRINT/GRADE (2026)** [P3] | randomization、対称性、factorization、誤差見積もり、具体的gate/Toffoli資源を結合 | 回路費用・誤差・factorizationを組み合わせること自体は新規ではない。比較条件・説明的結論が必要 |
| **Kanasugi et al. (2026)** [P4] | 部分ランダム化を含むsingle-ancilla QPE、unitary weight concentration、具体的なearly-FTQC実行資源 | fault-tolerant end-to-end資源は先行研究にある。現在のnative RZとは同じ指標でない |
| **Paganelli et al. (2026)** [P5] | ランダム化の利益がHamiltonian構造と許容誤差で変わることを、別のsparse-QSVT/ensemble taskで検討 | 「ランダム化が有利な領域を調べる」という一般的問いは既にある。ただし同じDF-native finite-RTE signal taskでの境界ではない |
| **Cugini et al., ROIS (2026)** [P6] | 回路実行コストと推定量分散を同時に考慮する重要度sampling | `cost×shots`を併用するだけでは新規性にならない。sampling rule最適化は別課題 |
| **Simon–Love (2025)** [P7] | 対称なTrotter formulaのcontrolled synthesisで任意角回転の追加を抑える構成 | naiveなcontrolled synthesisを強いdeterministic baselineと扱わない。一律なRZ半減係数を既存値に適用しない |
| **Abe et al. (2026)** [P8] | 水素鎖の決定論高次PFのエネルギー誤差とゲート数/RZ depthを比較し、新四次formulaも提案 | 高次決定論baselineを検討する根拠。ただし**energy誤差での順位をfinite-time signalの順位と同一視しない** |

**新規性の最終評価**：Track Aと全く同じ問題設定で、同じDF-native/finite-RTE/wrapper/情報時点での適用境界が既に確定されているとは確認していない。一方、類似の構成要素・資源最適化は多数存在し、**個々の方法・数値指標に独自性があるとは主張しない**。新規性は「公平なtaskと情報条件での新しい知見・成立境界・その原因」に依存する。

## 7. 複数研究方向の比較

| 候補 | 得られる可能性のある知見 | 最大のリスク | 推奨 |
|---|---|---|---|
| **A：H4の費用回帰を追加改善** | in-sample/OOFのMARE低下 | 経験的fitの改善で終わる、過適合、説明力なし | **主題としてSTOP**。AX-1bで一旦閉じる |
| **B：H6/H8へ単純なサイズ拡張** | 直接資源のsize差 | 原論文と重複、DF policy/対照が不公平 | そのままは行わない |
| **C：構造的予測モデルを中心に新設計** | circuit costの説明可能性、未知サイズ移送 | 実際のbasis/event入力が未整備、既存proxyとの差分が不明 | **補助的exploratory**。情報価値が高い場合に限定 |
| **D：強い対照と精度をそろえた直接資源評価** | PRの資源的利益/非利益の成立条件・原因 | strong baseline、数値検証、statisticsの費用 | **主研究として採用** |
| **E：full QPE/化学的energy accuracyへ直結** | 初期の動機に直接関係 | 時刻列、aliasing、state overlap、合成・量子測定を別途評価する大きな拡張 | 別gateで後続判断 |
| **F：新規PR内部アルゴリズム** | 方法論上の新規性 | 今回の評価研究と目的が異なる | 別研究として分離 |

**決定**：方向Dを中心に、方向Cのうち、実際の選択や比較の信頼性へ寄与する部分だけを統合する。優位境界に十分な科学的意味がない、または別taskの原論文との違いを説明できないと判明した場合は、対象を狭めるか終了する。**「強い論文になる」と現時点で確定したわけではない**。

## 8. 主RQと反証可能な仮説

### RQ-R（主軸）

> 同一DF targetとcoherent-signal taskにおいて、PR・discard・決定論PFの実装依存の測定込み費用は、精度・system size・回路合成でどのように変わり、どの要因によって最良方式が変わるか。

- **H-R1 [未検証]**：B2の利益の一部は、同じtarget accuracyでのone-shot costの差とshot amplificationのトレードオフとして説明できる。
- **H-R2 [未検証]**：改良された対称controlled synthesisや四次PFを加えると、既存二次PFだけの比較で得たB2の優位の大きさ、場合によっては勝敗が変化する。
- **H-R3 [未検証]**：精度・時間・DF policyを変えると、normalizationとbias marginが資源境界を変える場合がある。
- **H-R4 [未検証]**：H4で確認された比較関係はH6/H8へ無条件に保存されない。ただし異なる結果が出ても、サイズと表現・合成条件の違いを分離すれば説明可能である。

PRが勝つこと、四次PFが勝つこと、サイズで逆転することを成功条件にしない。**どちらの場合でも識別可能な理由が得られるか**を重視する。

### RQ-P1（補助）

> 実際のfull-wrapper参照費用を知らずに取得可能な特徴から、one-shot費用・費用順位をどこまで予測できるか。H4内の成功と、未知method・未知sizeへの失敗をどう区別するか。

- **H-P1 [未検証]**：H4の固定FEWは、q方向のgroup診断より未知サイズで性能が劣化し得る。
- **H-P2 [未検証]**：basis transition、fragment rank、control/boundary情報を使うことで、同一cost oracleの情報漏洩なしに外挿誤差の一部を説明できる。
- **H-P3 [未検証]**：平均費用誤差の改善と候補選択regretの改善は一致しない場合がある。

これらの仮説を検証するために追加featureの数だけを増やすことは目的にしない。必要な入力を生成・局所compileする**古典情報取得費用**も記録する。

### RQ-P2（非必須・将来課題）

参照biasを用いないshot/eligibility予測ができない限り、fully operationalな総資源regretはN/Aのままとする。RQ-Rの直接評価を完了するために、この困難なpredictorの完成を必須としない。将来の独立課題として扱う。

---

# 第III部　数学的・方法論的検討

## 9. 資源を決める三要因：回路費用、normalization、bias margin

固定の複素信号誤差配分では、各axis\(a\)の保存参照biasを\(b_{x,a}\)、numerical uncertaintyを\(u_{x,a}\ge0\)とし、

\[
h_{x,a}=\frac{\epsilon_{\mathrm{sig}}}{\sqrt2}-b_{x,a}-u_{x,a}.
\]

適格な場合の十分shot数は

\[
N_{x,a}=\left\lceil\frac{2B_x^2\log(2/\alpha_a)}{h_{x,a}^2}\right\rceil,
\qquad \alpha_{\cos}=\alpha_{\sin}=0.025,
\qquad G_x=\sum_a N_{x,a}\,\bar C_{x,a}.
\]

**[注意]** AX-1b/PM-2保存再現では当時の\(u=0\)を維持した。後続検証で数値headroomを導入しても、旧結果を書き換えない。上式のshotは採用したHoeffding型会計の**十分数**であり、全推定器の最小shot数ではない。[C3][C5]。

単一軸・ceilを無視した説明式なら、PRとDの資源比は、

\[
\frac{G_{\mathrm{PR}}}{G_{\mathrm D}}
\simeq
\frac{C_{\mathrm{PR}}}{C_{\mathrm D}}
\left(\frac{B_{\mathrm{PR}}}{B_{\mathrm D}}\right)^2
\left(\frac{h_{\mathrm D}}{h_{\mathrm{PR}}}\right)^2.
\]

**[解析]** PRが不利になったとき、それが(1) one-shot回路費用、(2) normalizationの統計負担、(3) bias marginの不足のいずれに由来するか区別して説明できる。これは会計式の分解であり、新しい理論保証ではない。二軸総和にこの単軸積の式を厳密に適用しない。各軸の寄与とshot重みを保存する。

**注意点：AX-0の「shot感度が小さなcost-model誤差を増幅する」という表現は、数学的に修正すべき。** 参照shotを固定したsingle-axis比較では、\(\delta G/G=\delta C/C\)であり、shot数がcostの相対誤差そのものを増幅するわけではない。実際に強い感度を持つのは、**bias/normalizationの誤差がshot推定に入る場合**、および候補順位が変わる場合である。今後はこれらを別々に分析する。

## 10. Bias marginの数値感度

単一軸・ceilを無視し、\(B,\epsilon,u\)が一定なら、

\[
\log N\simeq \mathrm{const}+2\log B-2\log h,
\qquad
\frac{\partial\log N}{\partial b}=\frac{2}{h}.
\]

したがって、\(h\)が小さい候補では、\(b\)の僅かな誤差がshot数へ大きく影響する。例えば\(|\delta b|\ll h\)でも、相対影響の一次近似は\(2|\delta b|/h\)。数値誤差が\(\epsilon\)の1%以下という条件のみでは、境界付近の信頼性を保証しない。

**[提案]** AX-2以降では、同じ数値\(u_a\)を全候補へ機械的に付加するのではなく、参照計算の実誤差評価と\(h_a\)に基づく判定不能域を定義する。\(h\le0\)、非finite、overflow、数値不確定を「0 shot」「不適格」と無条件にまとめない。

## 11. 複素信号誤差の方向とaxis配分

\(\Delta z=z_{\mathrm{approx}}-z\)について、現在の対称配分によるeligibility境界は

\[
\epsilon_{\mathrm{min,axis}}=\sqrt2\max(|\Re\Delta z|,|\Im\Delta z|).
\]

一方、\(d=|\Delta z|\)には

\[
d\le\epsilon_{\mathrm{min,axis}}\le\sqrt2\,d
\]

が成り立つ。**[解析]** 複素誤差の大きさが同じでも、実部・虚部方向への分配により、現在の十分条件の厳しさが変わる。これは方式の信号誤差ではなく、採用した誤差・測定配分が結果へ与える影響として理解すべきである。

**[提案：一回限りの感度診断]** 対称配分をprimaryとして保ちつつ、\(\epsilon_\Re^2+\epsilon_\Im^2\le\epsilon^2\)および総failure budgetを満たす別の軸配分規則を結果前に固定し、同じ候補で選択境界が変わるか確かめる。データを見てから有利な配分を方式ごとに選び、primary結果に上書きしない。別配分の最適化自体が研究主題だという主張もしない。

## 12. 誤差源のsigned decomposition

PRのfinite-RTE corrected signalを\(z_{\mathrm{PR},K}\)、exact-tailを用いる外側PF信号を\(z_{\mathrm{PF,exact\ tail}}\)とおくと、

\[
z_{\mathrm{PR},K}-z
=\underbrace{(z_{\mathrm{PF,exact\ tail}}-z)}_{\mathrm{outer\ PF\ contribution}}
+\underbrace{(z_{\mathrm{PR},K}-z_{\mathrm{PF,exact\ tail}})}_{\mathrm{finite\ RTE\ contribution}}.
\]

これは複素差についての**厳密な代数恒等式**。絶対値は相殺し得るので、\(|\Delta_{\mathrm{PF}}|+|\Delta_{\mathrm{RTE}}|\)を誤差の等式として扱わない。

B0について、同じtargetのdiscard truncated Hamiltonianをexactに発展した信号\(z_{\mathrm{discard,exact}}\)があれば、

\[
z_{\mathrm{B0,PF}}-z
=(z_{\mathrm{discard,exact}}-z)
+(z_{\mathrm{B0,PF}}-z_{\mathrm{discard,exact}}).
\]

これによりpure discardと残存PFの寄与を分けられる。ただし現在の主保存結果にはB0用\(z_{\mathrm{discard,exact}}\)がなく、`outer_pf_bias_abs`はpure PF誤差ではない。[AX-0 evidence][C4]

**[提案]** AX-2の小さなH4 correctness/診断に限りB0のexact truncated signalを作る価値がある。ただし計算目的は説明可能性であり、結果を見てB0/PRの誤差定義を変えない。複素量のphase・identity extraction・scalarを保持し、global phaseを無視して一致判定しない。

## 13. サイズ移送モデルとしての限界

FEWの特徴に分子サイズ・内部DF rank・basis transform長などの明示的な規模変数がないため、H4の\(\theta\)をH6/H8へ移したときの大きな誤差は、未知の構造的機構ではなくモデル表現能力の不足から生じる可能性がある。

**[決定]** 旧H4 FEWは結果前に凍結したbaselineとして残す。ただし、主たるサイズ予測方法の候補を、既存の`df_deterministic_step_rz_cost`・compiled-cost proxy・fragment metadata等と照合して設計する。解析情報・局所compileから作るproxy・full-wrapper oracleの三者を明確に分ける。

**[方法上の条件]** 予測対象のfull-wrapper costやtest biasをfeature作成へ混入しない。局所compileを使う場合は、その取得CPU/wall/RSSと対象範囲を記録し、「無償の純解析予測」と呼ばない。H6参照costの閲覧後に選んだ特徴はH6に対してexploratory。H8でのconfirmatory評価にはH8真値閲覧前の仕様・係数・predictionsのfreezeが必要。

## 14. Strong baselineと量子資源指標

### 14.1 高次決定論PF

既存B1（二次決定論）のほか、少なくとも四次の対称PFを同一signal taskで比較できるようにする。四次には負時間、scalar phase、inner deterministic blockの解釈、control実装の正しさが必要。**高次公式のエネルギー固有値誤差性能が良いことを、同じ時間Tのcomplex-signal誤差へ自動適用しない。** H4の小規模一致を経て、個々の方法のqを同じ精度制約の下で公平に評価する。[P8]

### 14.2 Controlled synthesis

Simon–Loveの結果は対称PFについての任意角rotationの削減であり、現在のfull-wrapper RZ countをそのまま半減させる許可ではない。**同じ近似ユニタリの別回路実装**ならsignalは維持したまま費用を比較できるが、PF段数や時間分割を変える場合はbiasも再評価する。B2のdeterministic backboneに適用可能な改善は公平に与え、ランダムtrajectory全体へ同じ簡約が使えるとは仮定しない。[P7]

### 14.3 追加資源指標

主要指標RZ-workを維持し、RZ depth、CX count/depth、total depth、circuit size、必要shot、補助qubitを補助として保存する。総depth×shotを、量子ハードウェアのwall timeとは呼ばない。synthesis error/T/Toffoli、magic state、physical qubits/runtime、state preparationがない場合、QPEのFT総資源優位を主張しない。

### 14.4 State preparation感度

共通の1-shot準備費用\(P\)を仮想的に加えるなら、

\[
G_x(P)=G_x(0)+N_{x,\mathrm{total}}P.
\]

これはprepared stateが方式間で共通かつ毎shot同じ会計を用いる場合の**感度診断**。実際の状態準備回路や繰り返し可能性を確証したわけではない。共通準備費用を無視することが相対順位を変える可能性だけを検査する。

## 15. DF表現と数値正しさの境界

対象\(H_{\mathrm{DF}}\)と元の分子Hamiltonian\(H_{\mathrm{mol}}\)は区別する。全方式は**同一保存DF target**を参照し、B0は同DF targetからtailを捨てた近似、B1/B2/B3も同DF targetを基準に精度を測る。DF切断誤差、B0 discard、outer PF、finite-RTE、数値誤差、samplingを必要に応じて別々に記録する。[AX-0 benchmark][C3]

H4はlegacy rank12。H6/H8ではサイズに応じた共通DF truncation政策を採用し、実際のrankを記録する。`df_tol`の戻り値が\(\|H_{\mathrm{mol}}-H_{\mathrm{DF}}\|\)のoperator normを保証するかはsourceと一次定義で確認が必要。保証されなければ、DF representation errorの保証ではないとして表示する。共通政策H4を必要に応じて接続し、rank12 legacyと新policyの差をサイズ差と混同しない。

sector/matrix-free経路について、Hamiltonianが保存するsectorと、**各中間primitive**が保存するsectorは異なり得る。H4でfull作用とsector作用を照合し、必要ならnumber sector/full vectorへ戻す。H8 full-space dense matrix生成を禁止する計画は妥当。corrected/raw関係\(z_{\mathrm{corr}}=Bz_{\mathrm{raw}}\)、phase、one-body/scalarの重複をsource意味論どおりに検証する。[AX-0 budget][C6]

---

# 第IV部　AX-2以降の研究実施方針

## 16. 段階ごとの科学的役割

| 段階 | 科学的目的 | 主な成果物 | GO/STOPの観点 |
|---|---|---|---|
| **AX-2A（次）** | AX-1bの判断を研究・比較契約へ反映し、技術的にpilot可能な状態にする | 契約改訂案、source接続、synthetic tests、baseline実現性、計算上限案、pilot task list | 研究条件・情報分離・予算を実行前に固定できること |
| **AX-2B** | 小規模correctnessとbounded profile | H4 dense対state-action、phase/sector/finite-RTE/controlled wrapperの一致、少数H6技術cellの壁時計/RSS/回路長 | 数値的正しさと費用上限、必要なbaseline構築の実現性 |
| **AX-3（H6）** | 最初の新サイズでPRの実資源・説明因子・モデル外挿を検証 | 事前登録direct集合のbias/B/C/N、費用分解、失敗・欠測、統計 | 主RQへの回答とH8で試す説明仮説があるか |
| **AX-4（H8）** | H6までで固定した説明・比較・予測の独立確認 | Frozen predictions、参照結果、coverage、候補選択の独立監査 | 事前freezeと計算予算、未知情報の非流入 |
| **AX-5** | claim/evidence mapと原稿 | 主要表・図、limitations、再現性資料、先行研究との差分 | 主張の水準が根拠に適合するか |

### 16.1 AX-2AでCodexが一括して行う作業

- 既存AX-0契約を上書きせず、**今回の研究レビューをdecision recordとして別versionで反映**する。主RQ・補助RQ・不採用の方針・変更理由を明記する。
- H4/H6/H8のtarget・geometry・DF policy・state・T・\(\epsilon\)・比較対照・出力指標の**比較可能性matrix**を整える。
- 強い決定論baseline（最低限standard fourth-order）とcontrolled synthesisのsource再利用経路・semantic correctness・費用取得法を調査する。実装可能なものから限定的に接続する。
- H4 denseとstate-action、sector/phase/normalization、B0 exact truncated signalの最小検証を設計・synthetic test準備する。
- H6の技術pilotを、**少数のmodel-independentなtask**で実施できるよう準備する。主科学結果を広く観測するgrid scanは含めない。
- H4 FEWの凍結対照と構造proxy候補、featureの取得情報・計算費用・data leakage防止を整理する。
- プロセスごとのCPU/GPU、RAM、wall、compile call、trajectory sample、disk、failure/partial outputの上限を提案する。旧AX-1bの300秒/8 GiB上限をH6/H8へ無断流用しない。
- Codexが数値許容誤差・テスト・module境界等を裁量で設計してよい。ただし、**主RQ、参照Hamiltonian、比較対照の公平性、primary metric、held-out独立性**はGPTの判断なしに変更しない。

### 16.2 AX-2B pilotで確認したい内容（技術と科学を分離）

1. H4でcorrected/rawとcomplex signalが旧dense実装に対応するか。
2. finite-RTE平均をpolynomial/state-actionへ接続してもsource-defined phaseとevent次数が変わらないか。
3. sector primitiveの保存性、spin ordering、scalar/extracted identity、正負時間、reverse/forward順序が保たれるか。
4. 強いbaselineのcontrolled circuitを同一作用として検証できるか。
5. H6のbounded例で参照計算・poly作用・full-wrapper compileのwall/CPU/RSS・回路長・出力サイズを計測できるか。
6. 数値誤差評価が最も厳しい登録\(\epsilon\)とbias marginに対して十分か。

技術pilotで得た実際のbias/Cは**developmentとして扱う**。H6のconfirmatory model評価やH8のblinded独立検証へ混ぜない。失敗時に無断で候補集合・資源上限を拡大しない。

## 17. H6本検証での公平性・情報分離

**条件をそろえる比較**：同じ\(H_{\mathrm{DF}}\)、\(\psi\)、\(T\)、precision/failure allocation、compiler、wrapper scope、primary metric、候補選択・探索上限を方式間で合わせる。方式固有に必要な\(r,K\)等は公平な役割を持つ自由度として登録する。

**探索域**：prefix fractionをサイズに応じて生成し、B0/B2は同じprefixを比較。qと\(R=qr\)を独立に扱い、same-\(R\)比較を含める。候補quotaと主direct集合は、**モデルの予測順位と科学結果を読む前に**決める。結果を見て探索上限に接した場合は`boundary_limited`とし、手法全体の不可能性・真の最適と主張しない。[AX-0 benchmark][C3]

**情報分離**：

- H4：model開発・既観測の根拠。
- H6：初めてのサイズ移送を観測する開発系。H6を見て変更したモデルはH6に対して探索的。
- H8：H6までで凍結したモデル・選択・説明仮説の独立確認。H8の参照費用/信号を見てからモデルのfeatureや候補を変更しない。

**モデル比較の条件**：モデルに必要なfeaturesの情報取得費用を記録し、full-wrapper参照costと同じ処理をすでに実行したproxyは安価な予測と呼ばない。モデルの比較に用いるdirect集合を、モデルが良いと予測したものだけから構成しない。

## 18. Uncertaintyとconfirmationの方針

- Random trajectory平均は32標本であることと、高価なrare eventの未観測を区別。
- cosine/sineのpair、同seed、共分散を保持。異候補間の共分散は未検証のまま0とみなさない。
- 明確な主要対照や僅差の最良候補について、結果を見て恣意的な追加samplingをしないよう**追加確認条件を結果前に登録**する。
- 参照biasの数値不確かさは採用\(h_a\)に対して判断し、eligibilityに未確定域を設ける。
- 信号精度の\(\epsilon=0.001\)は、既存保存値の会計点と、新しい独立な高精度科学検証を区別する。
- 結果を見て探索集合を変更した場合、versionを分け、探索的な変更とconfirmatory結果を分離する。

## 19. 実施する図・表の方針（将来の原稿候補）

| 図表 | 内容 | 必要な注意 |
|---|---|---|
| **図A** | 精度×system size別の最小RZ-work・best method map | 各方式・候補の探索境界/coverageを併記。非到達と不可能を区別 |
| **図B** | 主要比較の\(C\)、\(B^2\)、bias-margin寄与の分解 | 単軸の積分解を二軸へ誤用せず、各軸のN・Cも表示 |
| **図C** | 強い決定論PF・controlled synthesisを含む比較 | 同task・同誤差/測定規則・scopeを表示 |
| **図D** | FEW・構造proxyの誤差/coverageと選択regret | H4 fit、H6 development、H8 independentを明確に色・注記で分離 |
| **補助図** | 精度境界、error decomposition、数値headroom、rare trajectoryの影響 | 事前ルールと形式的保証の有無を記載 |

上記は図表の企画であり、未実行のH6/H8データや性能を予告するものではない。

## 20. GO/STOP／研究終了条件

| 状況 | 判断方針 |
|---|---|
| AX-2でsector/phase/target/wrapper等の意味論が一致しない | 実装または対象を修正。科学結果として採用しない |
| AX-2の計算資源が足りない | pilot範囲を限定し、支配的費用を測る。未知の本計算を起動しない |
| 強いbaselineが実装できない | 「指定二次PF実装との比較」までclaimを狭める。一般的決定論法への優位は保留 |
| H6でPRが非優位 | **それだけではSTOPしない**。非優位の条件・原因が明らかになるなら重要な結果 |
| H6で旧FEWが大きく失敗 | 失敗範囲を報告。RQ-Rの実費用比較が有効なら継続可能 |
| 新構造モデルがH6で改善する | H6は開発結果。H8の真値閲覧前にモデルとpredictionを固定 |
| H8のための独立な問いが残らず、費用も高い | H8拡張を強制せず、限定結論で閉じる |
| 文献との差分が新しい知見でなく単なる再現 | 独立論文の主要claimを縮小し、technical/resource studyとして閉じる |
| 比較条件が保てず、主要RQに答えられない | スコープ変更または研究停止をGPTで判断 |

**論文化GOの最低条件**：同一task上で公平な競争相手がいること、誤差・測定・回路費用・数値不確かさが区別されること、十分な情報分離の未使用検証または成立境界の明確な説明があること、一次文献との差分を具体的なclaimとして書けること。図が多いこと、testがPASSしたこと、PRがbestだったこと**だけでは不足**。

---

# 第V部　進行管理と引き継ぎ

## 21. 次の担当・GPTレビューの必要性

| 問い | 判定 |
|---|---|
| AX-1b後に研究方針のGPTレビューは必要だったか | **必須**：新しい科学的知見と複数方針の選択が必要だった |
| 今回そのレビューを実施したか | **実施した**：新規性、数値結果の意味、代替方向、採用方針、限界、GO/STOPを本書に記録 |
| 次の担当 | **Codex**：AX-2Aの具体設計、実装、テスト、技術pilotの準備 |
| Codexに委任する自律性 | 技術的詳細、実装方針、テスト、数値許容誤差、記録、環境、small pilot設計を一括委任 |
| Codexが勝手に変更してはいけないこと | 主RQ、科学比較対象の公平性、primary endpoint、参照target、held-out情報分離、論文主要claim |
| 次のGPTレビュー条件 | AX-2で新しい科学的結果が生じたとき、またはH6本検証の主要条件/予算を確定し新規実行を始める前 |
| AX-2Bの科学pilot・AX-3/4を本書だけで実行してよいか | **否**。限定実行の対象・資源上限・停止条件を別途明確化する |

**注意**：AX-2Aの通常のコード修正・synthetic追加・機能接続の途中で、毎回GPTの承認を要求しない。逆に、AX-2準備中に「実は別RQを先に選ぶべき」「高次baselineを削除してPR有利だけを主張する」「H8を開発用に使用する」といった科学的方針変更が必要になった場合は、その時点でGPTに戻す。

### 21.1 Codexへの引き継ぎ要約（科学的に固定する事項）

**目的**：同一DF-native finite-time signal taskでのB0/B1/B2/B3＋強いdeterministic baselineの資源競争力と、\(C/B/h\)を通じた成立条件を調べる。RQ-P1は費用予測情報・限界の補助診断。Operational shotの完成は必須条件ではない。

**既存成果の扱い**：AX-1b結果、FEWモデル・complexity gate、原科学結果・source・manifestを上書きしない。旧FEWはH4で凍結したbaselineとして保存。

**比較上の原則**：同じ\(H_{\mathrm{DF}}\)、state、T、axis error/failure allocation、compiler/wrapper、主要cost単位、モデル非依存のdirect候補集合、strong baseline、未確定eligibilityの扱いを守る。H4 rank12と新DF policyは別レイヤー。

**結果の意味**：参照shot使用のconditional-oracle regretと、未知系へのoperational選択を分離する。同期したtrajectory costのSEはengineeringでありformal winnerではない。

**実行と保護**：AX-2Aでは技術準備と非科学synthetic testsをまとめる。AX-2B以降の新規科学計算は、比較scope・実行資源・task identityを固定した別の明示的authorizationで実施する。既存dirty、他Track、過去結果を保護する。

### 21.2 次のGPTに提出してほしい資料

- AX-2で接続したstate-action/finite-RTE/controlled wrapperの**正しさ**の証拠と失敗例。
- 強いbaselineの実装意味論、どの改善をB1/B2へ公平に与えたかの対応表。
- H4 legacy↔共通DF policyの対応と、H6/H8のrank・対象Hamiltonianの一致規則。
- 小規模profileの実測wall/CPU/RSS、最悪回路長、計算上限と対象quota。
- H6主検証の直接集合・precision/time/seed・モデルpredictionのfreeze案。H8をどう未使用に保つか。
- RQ-Rのどの問いにAX-3が答えるか、PR勝敗によらない成功/終了条件。

Codexの実行報告だけで自動GOを出さず、GPTは**H6本検証に科学的価値があるかを独立評価**する。

---

# 第VI部　Claim/evidence対応と出典

## 22. 現時点で主張できること・できないこと

| Claim | 証拠水準 | 現時点の扱い |
|---|---|---|
| H4登録集合でFEWのcost MAREがSINGLEより小さい | **保存数値結果で確認** | 主張可能。学習・group区分を併記 |
| H4のq-fold gateでFEWを採用した | **登録規則の適用結果** | 主張可能。ただし科学的な汎用保証ではない |
| B3を除外した学習でFEWの外挿誤差が増える | **保存数値結果で確認** | 主張可能。B3/prefix0交絡も併記 |
| FEWのH4開発conditional regretが小さい | **保存結果で確認** | 主張可能。参照shotを使うoracle条件を明記 |
| FEWが未知H6/H8で総資源最良候補を選べる | **未検証** | 主張不可 |
| PRが強化四次PF＋改善controlより一般に優れる | **未検証** | 主張不可 |
| PRの資源利益がbias/B/Cのどれで決まるか | **分解は数学的に可能、系統的原因の実証は未検証** | 検証仮説として維持 |
| B0 biasをpure discard/pure PFへ分けられる | **分解の数式は成立、必要なexact truncated信号は欠測** | 新しい最小検証が必要 |
| AX-1bが化学精度QPEのFT総資源を評価した | **評価対象が違う** | 主張不可 |
| H4/H6/H8で一般的な漸近スケーリングが分かる | **サイズ点が不足・比較規則も未統一** | 主張不可 |

## 23. 未解決の重要論点と優先順位

| ID | 未解決事項 | 優先度 | 対応段階・担当 |
|---|---|---|---|
| U-A | strong controlled synthesisと四次PFのsemantic一致 | **高** | AX-2／Codex |
| U-B | H4 rank12とH6/H8共通DF政策の橋渡し | **高** | AX-2／Codex。対象比較の主要変更はGPT |
| U-C | sector primitive・finite-RTE polynomial・phaseのcorrectness | **高** | AX-2／Codex |
| U-D | reference biasの数値headroomとeligibility境界 | **高** | AX-2→AX-3／Codex、判定方法の主要変更はGPT |
| U-E | H6 candidate lattice・direct quota・strong baselineの予算 | **高** | AX-2のprofile後にGPTで科学的判断 |
| U-F | 構造proxy入力の取得・source対応・情報漏洩 | 中 | AX-2／Codex |
| U-G | rare-event costと候補間rankingの不確かさ | 中〜高 | AX-3主要比較／Codex |
| U-H | 測定軸配分への感度 | 中 | AX-2/3の限定診断／Codex |
| U-I | fully operational bias/shot predictor | 中〜低（本研究必須ではない） | 後続課題としてGPT判断 |
| U-J | full QPE/FT physical resources | 低（現スコープ外） | 別研究契約 |
| U-K | 先行研究・原稿v0.1・旧`paper_d6`と一次式の来歴 | 中 | AX-2文献/claim audit／Codex。新規性の確定はGPT |

## 24. 出典・検証対象リンク

### 24.1 GitHub：結果・契約・source

- **[R1] AX-1b結果レビュー**：[`result_review_report.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/execution_audit_v1/result_review_report.md)。保存集計値の主な根拠。
- **[R2] Terminal**：[`terminal_status.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/run_v1/terminal_status.json)。実行identity・終了状態。
- **[R3] Shot availability**：[`shot_availability.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/run_v1/shot_availability.json)。operational N/A・status数。
- **[R4] Model fits**：[`model_fits.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/run_v1/model_fits.json)。係数と複雑化gate。
- **[R5] Cost diagnostics**：[`cost_metrics.csv`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/run_v1/cost_metrics.csv)。fold/method/axis別。
- **[R6] Conditional selection**：[`conditional_oracle_selection.csv`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/run_v1/conditional_oracle_selection.csv)。regret/eligible/candidate集合。
- **[R7] Output manifest**：[`output_manifest.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/run_v1/output_manifest.json)。出力同一性。
- **[C1] AX-0 research contract**：[`track_a_ax0_research_contract.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/docs/research/track_a_ax0_research_contract.md)。RQ、claim水準、段階と終了条件。
- **[C2] AX-0 model correspondence**：[`track_a_ax0_model_correspondence.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/docs/research/track_a_ax0_model_correspondence.md)。原論文とのunit/task差とsource位置づけ。
- **[C3] AX-0 benchmark protocol**：[`track_a_ax0_benchmark_protocol.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/docs/research/track_a_ax0_benchmark_protocol.md)。候補、DF policy、精度、情報分離。
- **[C4] AX-0 evidence inventory**：[`track_a_ax0_evidence_inventory.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/docs/research/track_a_ax0_evidence_inventory.md)。旧結果・保存field・欠測。
- **[C5] AX-1a evaluation protocol**：[`track_a_ax1a_evaluation_protocol.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/docs/research/track_a_ax1a_evaluation_protocol.md)。主指標、四case、regret、uncertainty。
- **[C6] AX-0 compute budget**：[`track_a_ax0_compute_budget.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/docs/research/track_a_ax0_compute_budget.md)。sector/matrix-free、数値正しさ、予算。
- **[S4] Source action definitions**：[`pr2_matched_accuracy_m1_execution.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/src/trotterlib/pr2_matched_accuracy_m1_execution.py)、`_fixed_action_count` / `_deterministic_fragment_count`。

### 24.2 一次文献（確認した版、2026-10-09）

- **[P1]** Jakob Günther et al., **Phase estimation with partially randomized time evolution**, arXiv:[2503.05647v2](https://arxiv.org/abs/2503.05647v2)（2026-07-10改訂）, *PRX Quantum* 7, 020332 (2026), [DOI](https://doi.org/10.1103/ynxb-p2xq)。**元論文の詳細DF/QPE資源会計とPR benchmark**。
- **[P2]** Marek Miller et al., **phase2: full-state vector simulation of quantum time evolution at scale**, *Communications AI & Computing* 1, 5 (2026), [DOI](https://doi.org/10.1038/s44488-026-00002-2)。40 qubits実行と32 qubitsまでの確かな誤差prefactor抽出を区別。
- **[P3]** Pablo A. M. Casares et al., **Theory and practice of Trotter product formulas for quantum chemistry**, arXiv:[2606.30741v1](https://arxiv.org/abs/2606.30741v1) (2026)。SPRINT・GRADE、具体的化学系の資源評価。
- **[P4]** Shota Kanasugi et al., **Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime**, arXiv:[2603.22778](https://arxiv.org/abs/2603.22778) (2026)。partial randomizationとUWCを用いるearly-FTQC QPE資源評価。
- **[P5]** Francesco Paganelli et al., **When is randomization advantageous in quantum simulation?**, arXiv:[2604.07448v1](https://arxiv.org/abs/2604.07448v1) (2026)。別task/ensemble上の精度・構造境界。
- **[P6]** Davide Cugini et al., **Resource-Optimal Importance Sampling for Randomized Quantum Algorithms**, arXiv:[2603.13495v1](https://arxiv.org/abs/2603.13495v1) (2026)。回路費用とestimator varianceの同時最適化。
- **[P7]** William A. Simon and Peter J. Love, **Halving the Cost of Controlled Time-Evolution**, arXiv:[2511.13855v1](https://arxiv.org/abs/2511.13855v1) (2025)。対称PFのcontrolled synthesis。
- **[P8]** Hiromu Abe et al., **Evaluating higher-order product formulae for molecular ground-state energy estimation**, arXiv:[2605.30967v1](https://arxiv.org/abs/2605.30967v1) (2026)。高次決定論PFのenergy error・資源比較。

**文献調査の限界**：ここでの新規性判定は、以上の主な一次文献とリポジトリ保存の比較資料に基づく。包括的systematic review、著者実装の独立再現、全version間の数値差の完全追跡は実施していない。原論文の特定付録式と旧sourceの`paper_d6`ラベルの対応はAX-0で未確定としており、確認済み式として転記しない。

---

## 25. 最終decision record

**レビュー結果**：Track Aは継続。主軸RQ-R、補助RQ-P1、RQ-P2は現段階の必須条件から除外。強い決定論baseline、同じDF targetの精度、数値的bias margin、normalizationとcompiled one-shot資源の分離、独立サイズ移送を中心に据える。H4のFEWを再fitして改善すること自体は研究主目的にしない。

**次の段階**：CodexがAX-2Aをまとめて設計・実装・テストし、bounded technical pilotへ進める契約と環境を整える。新規科学計算については対象と資源上限が明確に認可された範囲のみ。H6 main campaignとH8 independent verificationは未認可。

**次にGPTが戻るべき節目**：AX-2で重要なcorrectness・資源profile・baseline実現性について新しい知見が得られ、H6本検証の科学的意義と実行範囲を判断する必要が生じたとき。**細かな実装修正・通常テスト・資料作成のための定期的な形式レビューは不要**。

**レビューの判断を変えてよい場合**：重要な新しい科学的結果、先行研究との差分を否定する確かな情報、重大な比較上の弱点、または今回の判断の誤りが明らかになったとき。そのときは、変更理由を研究決定履歴に残す。

*文書終わり。*

<!-- Markdown reference definitions: document-internal source cross-references -->
[R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/execution_audit_v1/result_review_report.md
[R2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/run_v1/terminal_status.json
[R3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/run_v1/shot_availability.json
[R4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/run_v1/model_fits.json
[R5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/run_v1/cost_metrics.csv
[R6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/run_v1/conditional_oracle_selection.csv
[R7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/run_v1/output_manifest.json
[C1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/docs/research/track_a_ax0_research_contract.md
[C2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/docs/research/track_a_ax0_model_correspondence.md
[C3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/docs/research/track_a_ax0_benchmark_protocol.md
[C4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/docs/research/track_a_ax0_evidence_inventory.md
[C5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/docs/research/track_a_ax1a_evaluation_protocol.md
[C6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/docs/research/track_a_ax0_compute_budget.md
[S4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/src/trotterlib/pr2_matched_accuracy_m1_execution.py
[P1]: https://arxiv.org/abs/2503.05647v2
[P2]: https://doi.org/10.1038/s44488-026-00002-2
[P3]: https://arxiv.org/abs/2606.30741v1
[P4]: https://arxiv.org/abs/2603.22778
[P5]: https://arxiv.org/abs/2604.07448v1
[P6]: https://arxiv.org/abs/2603.13495v1
[P7]: https://arxiv.org/abs/2511.13855v1
[P8]: https://arxiv.org/abs/2605.30967v1
