# D → C/B：S1の比較条件とV1代表セル

[REPORT](REPORT.md)／[S1](S1_DOMAINS.md)／[V1](V1_LONG_TIME.md)／[未承認の有限案](FINITE_VALIDATION_PLAN.md)。D基点01e068af、C d7c09ef2、A3ae055fa、B ce0517deを読取参照。各担当のbranch/worktreeへ変更を加えていない。

## C担当へ

lowering比較の最小要件は、RTEを含むproducerの**SOURCE付き実lowering**と、比較familyの共通policyの双方を検査すること。compiler hashだけの一致や任意のself-declared extensionは十分でない。旧Full-H9b83/7a8bは汎用RZZ.control(1)、D PR P1/P2はCX–CRZ–CX。旧recordは不変とし、必要な旧IDはderived sidecarでsourceを示す。missing/unknownはnewへ補完しない。今回Dでは比較validatorを新設していない。

Full/Truncated PFはRTE lowering非該当。元native/control/wrapper/scalar/sourceを保持し、新RTE policyと両立するかを適用区分として扱う。同じ新ZZ IDを全methodへ貼って整合したことにしない。canonical profileへ追加するか、consumer policy/sidecarで表すかはC/Bで統一し、旧canonical/receiptは書き換えない。既存profile_extension/price bindingsを使い、別価格pipelineやapproval基盤を増やさない。

C d7c09のA shortlist対応と5round/10軸receiptをそのまま使う。Dで78軸coverage/73receipt監査を再生成していない。新4round/8 wrapper案は識別・有限holdout用で、全13roundの必要十分条件ではない。最初の代表候補P-S2はC案のsubset、A優先S2/.2 m5/T3.2/q16 `987ba8031fd5b790cf59`。fit/予測/比較はCの別科学scopeでfreezeしてから取得する。Y4/M5を価格/標本数の都合で候補から削除しない。

Aが後続d48d873のexact追加幅計画を公開したため、P-S2の承認前に元.2を比較familyへ残すか確認する。後続は新科学結果ではない。新幅のexact requested/actual・q・signed timeを受け取り、旧係数/価格を幅labelだけで移送しない。Dは両menuを一括compileする準備を追加していない。

CのPR6 pairedセル案は未採択・n_i未定のまま参照する。旧DのLD2/m0接続案・旧window24 compile案と合算しない。P2を新holdoutへ付け替えない。Full-HのF-H案は現C SOURCE367dc98のm0/T.1/q1/R448/K2への新ZZ1 normal pathであり、n1でmean modelを検証したことにはならない。

## B担当へ

正式Gの前に、同energy taskの候補別N又は許容N集合・risk資格、価格law、比較family、margin/regretと価格側confidence/誤差budgetが必要。今回参照したce0517の2固定DP診断inputから、Full PF/PR/Full-H/Truncatedの最適Nや正式winnerが得られたとはしない。後続結果が公開されたら、その固定SOURCEを使う。

valid同時price誤差bなら `Σ N b`、候補差には双方の誤差、真に共有価格の項だけ係数差を使う。未知のlong-domain/model/law/rareをSE/0で埋めず、unknown価格のため比較不能であることとアルゴリズムが高価であることを区別する。元nominal×E[S]の約97%を正式Gの寄与率にしない。

## 取得recordとSOURCE

PRは既存D adapterのpath S/C/Dを区別するrecordを維持する。Full PF/Full-Hは元native `cost_trial` のprices/axis/fingerprint/compiler/代表作用検査を使い、同じquantity/unit/control/measurement/P0を明示する。Full-HをPR専用adapterへ通すためにprefix guardを解除しない。Full-H S/Dは現Accountingの適合未検証ならnull、direct Cのreceiptとする。

Full PFは決定論full-wrapper価格、Full-Hはnormal 1 pathの有限証拠。Cの`FULL_WRAPPER_MEAN_COMPILED_PRICE`へ変換してlaw平均資格を作らない。source/lowering/scalar・identity/law・cohort/axisを正しくmappingできても、統計的/外挿資格は別である。旧平均値の数値受入・fit・E[S]＋残差結合の実入力評価も科学承認対象。

新SOURCE/caller/出力先・cohort/全seed・科学call範囲の固定は科学実行前に必要。e827のsource-bound totalはPR-LD1/LD2のselectorから渡し、Cはtotalを再計算しない。Full-Hは独自one-body scalarを含む元契約、Full PFはHamiltonian定数だけ、B0/B0Iは各retained-H契約を使う。metadata PASSは実prepare/guard成功ではない。

## 次の担当とSTOP

- **C**：S1比較policy/validatorの小修正と必要な人工ケースを担当。V1の固定仮説・使用domain・誤差根拠、必要なセルをBと整理する。DのSOURCE根拠・既存receipt・セルIDを再利用する。
- **B**：risk/Nと同taskの比較条件・必要価格精度を供給する。価格資格未成立なら正式GはUNKNOWN。
- **ユーザー**：共通lowering scenario、保存値の受入/fit、P-S2又はF-H等の必要な有限科学scopeを別途明示承認するか判断する。
- **D**：必要scopeが選ばれた場合に限り最小caller差分を人工確認・新commit固定し、その完全SOURCEへの科学承認後のみ取得する。現在は新compiled証拠を取得せず、commit/push・remote照合後 **mandatory STOP**。
