# 次の最小検証：V1有限precisionとrole別予算診断

2026-11-08 JST。基準 **460726569c60ea4cf8f9839f2aff16f84157babc**。[REPORT](REPORT.md)、[OPTIONS](PRECISION_AND_BUDGET_OPTIONS.md)。**本書の科学数値callは未実行・明示承認待ち。今回の静的準備許可を科学実行へ拡張しない。**

## 1. 固定するSOURCE・入力・科学条件

実行SOURCEは、本書と同じ公開commitの[diagnose_pr_reference_post_v0_precision.py](../../../scripts/diagnose_pr_reference_post_v0_precision.py)。旧SOURCEは460726569c60ea4cf8f9839f2aff16f84157babcから`git show`で読む。新しいhelperの`main`以外の科学runnerは起動しない。

| 保存入力／関数 | 用途 |
|---|---|
| `artifacts/pr_reference_n5b2_numeric/2026-11-08/INPUTS.json` | 43 basicのj／record識別子／saved_guard、保存sqrt2有理上端。旧archive検証結果を再利用 |
| 同`RESULTS.json`／`STOP.json` | 4中央positiveとb36行の引用。旧b36 index／Q／Gを再計算しない |
| `artifacts/pr_reference_post_n5b3_v0/2026-11-08/RESULTS.json`／`STOP.json` | V0 16比較・b36中央probeをそのまま継承。V0関数を再実行しない |
| `artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json#/condition` | 全roleの物理上界、保存局所cap、joint予算、N626／630 |
| `src/trottertracks/pr_reference_dyadic_dictionary.py:round_index` | 関数bodyを変更せずAST抽出。隔離global Bだけ候補bへ設定 |
| `artifacts/pr_reference_n4a_coverage/2026-11-08/ROLE_COVERAGE.json` | D／frame／product／CSWAPの保存literal superset。新しい到達keyは生成しない |
| `artifacts/pr_reference_ct4/2026-11-07/strict_records.tar.xz` | 固定C0 299の保存strict guardだけを読む。T／sequenceの評価なし |
| `artifacts/pr_reference_n4b_consumer/2026-11-08/ROLE_WORD_CERTIFICATES.json` | 固定85の保存Q＋G。Emitterは残項へ帰属 |
| `artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/INPUTS.json#/record_inventory` | 固定C0 archiveの既存hash。旧7603／268 archiveを新たに探索しない |

4 positiveはB′ `0x1.7e4088e024c5fp-12`, `0x1.fdab627b8d37ap-14`、Reference `0x1.7e4088e024c5fp-11`, `0x1.fdab627b8d37ap-13`だけ。Negativeは同じpositive productのcomplete actual adjoint。新word文字列・行列を作らない。

背景はH4 linear1Å／STO-3G／DF12、元H厳密ground供給・準備費除外、Partial S2 LD3 h=.02/K2、ordinary diag(I,V)、canonical independent finite bits、Qiskit1.3.0/compiler A、既存B′／Reference／D融合／basis共有／scalar／event／X/Yを固定。M17 q131072/r32/bits88、N626/630、alpha1/720、failure.05、phase cap.4、energy1.5936e-4 Haを変更しない。

算術は既存のPython integer／Fractionとsaved sqrt2上端。新しいsqrt／sin／guard計算は不要。保存floatのcapとjoint予算はbinary64の厳密有理比へ変換する。表示丸めで判定しない。

## 2. 承認対象を閉じた最小数値scope

### A：V1中央4角の有限表

1. 4 thetaのreduced binary64分母2^eを読む。従来規則の`b_max=max(36,max e)`を一意に決定し、候補37…b_maxを列挙する。b36は旧行参照だけ。新しいb探索規則・ad hoc ladderを追加しない。
2. 各候補で4 positive／同negativeのindexを再利用関数で求める。Ascending binary指数J、exact delta、`Q=sqrt2_saved_upper*delta/2`、選択済み43内のG_known、missing指数を保存する。Sequenceの読取連結・word生成・provider呼出しは不要。
3. X/Y保存capについて`R=cap-Q-G_known`を評価。Missingがあればlogical全上界をnullとする。R<=0は選択構成の加法十分証明の不足、R>0はmissing guardへの条件付き余裕。全wordの誤差下界やRPE可否と解釈しない。
4. 既存43で完全coverageの行はQ+G_knownを保存する。Guard和／chunk数の変化も出し、b増加による単調性は仮定しない。T価格を読んで合算しない。

新b数n=b_max-36に対し、4n component行、8n index call、8n局所cap診断。既存b36評価は0。終端で4 targetのQ=0を検算するが、新しい科学precisionとして採択しない。

中央4 literalだけの構成診断とし、非中央C0／85 wordは固定する。Provider全体のB変更や有限overlay実装は本scopeに含めない。実装する場合の同literal共通policyの資格も後続判断として残す。

### B：保存role guard envelopeとjoint残項

**Bの保存role処理はAのb表に依存しないため並行可能**。最後の中央＋他role残差だけがAに依存する。小さい有理演算にworker起動が有利でなければ、単一processでまとめてよい。CPU/RAMの人工上限は設けず、共有設定・他ユーザーは変更しない。

1. C0 299 recordの保存strict_error_upperと85 certificateの保存Q＋Gを、同positive literalに対応させる。固定C0優先は不変更。Recordの再選択／新guard／全旧inventoryの再探索をしない。
2. 両armでD／frames／products／CSWAPの**8 group envelope**を評価する。Dはunfused＋boundary差＋actual SOURCE inverse supplementのpositive unionを使い、signed差を負のnormへ使わない。各集合は保存N4-A superset、m17への移送は既存条件CとSOURCE閉包付きとする。
3. Missing又は理由不明の空集合は上界nullとする。B′ CSWAPだけはSOURCEの不存在と保存物理数0を対応させて0にできる。全arm×X/Yの**4つのknown非中央load** `sum L_role*max e_role`を保存する。
4. 新b各行の中央2 keyにはV0と同じsafe-max形式を使い、両arm×X/Yで**4n中央load／残差行**を出す。Missingがあれば中央のcomplete上界もnull。保存joint予算から既知中央項と既知非中央項を差し引き、missing・残りrole・emitter／shellへ必要な残差を表示する。
5. Event／scalar_and_wrapper／vacuum、missing role、emitter／shellを`UNQUALIFIED_NOT_ZERO`で残す。Full joint上界とm17 RPEは未資格のまま。負の残差はこの選択guard／max certificateの不足であり実誤差の下界ではない。正でも全体PASSとしない。

これは保存budgetとの有限な再比較であって、RPE式・raw disk・shot再計算でもprecision optimizerでもない。期待多重度や観測最大は使わない。

### 検算・出力・実行入口

- 固定4 literal／43 record／299 C0／85 certificate／m17条件と保存Nの照合。
- 正負index対称、ties-to-even、index再構成、delta<=2^(-b-1)、G>=0、missing分離、終端Q=0、上記件数を検算。
- Roleごと全keyが対応しない場合はnull。既知loadをactual errorの下界にしない。未評価項とdomain問題を出力へ残す。
- Fresh `artifacts/pr_reference_post_v0_v1_budget/2026-11-08/` にRESULTS／SOURCE_INPUTS／STOPを保存し、REPORTで結果と限界を説明する。旧領域は開いて書かない。

**以下は実行承認後だけの入口。今回未実行。**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/diagnose_pr_reference_post_v0_precision.py
```

Helperはstdlibだけを使う。Python binary64／Fractionの前提を維持し、既存環境を変更しない。新しい合成runtimeは不要。承認があればSOURCEを固定公開版から再利用して実施できる。別の承認証明書・grant発行は要求しない。

## 3. STOP・後続判断

有限表・8 role envelope・4 known load・4n残差と指定検算の完了、又は固定入力／SOURCE不整合でmandatory STOP。Record不足は不足として保存し、合成で自動補完しない。閾値未決・全体未資格のままでもscope完了で止める。

次の科学判断はGPT／ユーザーが行う。

| 得られる証拠 | 次に判断すること |
|---|---|
| 既存43で新bを覆い、logical予算に余地 | 他roleとdomain／terminalを閉じる最小検証を選ぶ。bを自動採択しない |
| Missing guardへ正の予算が残る | 別保存record／別構成で埋まるか、有限追加wordが必要か。Key／epsilonを先に固定する |
| 既知他roleだけで予算を使い切る | 中央precisionだけの改善ではその証明は足りない。Role max／guard／word選択のどれを次に扱うか |
| Joint残差が正でもunknownが残る | 成功未資格を維持。Unknownを0にして進めない |

Domain D1/D2/D3、signed digit、direct guard、新合成、provider／consumer変更・runtime検証、native build/compile、RPE再会計、T価格・最適化・全18round総Tは本数値scope外。別途の固定scope・明示承認が必要で、ここから自動進行しない。強B′へ同じ改善機会を与え、N626／630を保つ。

**現在は静的準備完了のmandatory STOP。次担当：GPT／ユーザーが§2を実行承認するか、候補を保留・変更するか判断する。**
