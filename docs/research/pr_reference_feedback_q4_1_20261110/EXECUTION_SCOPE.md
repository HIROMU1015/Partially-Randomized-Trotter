# Q4-1 H/E/C：固定実行仕様・未実行

2026-11-10 JST。**H/E/Cの科学数値実行は未承認・未実行。** この文書は実行許可の証明ではない。明示承認までは実装・人工検証・metadata-only preflightだけでSTOPする。

正本は[`b159d721`のQ4-0 NEXT_NUMERICAL_VALIDATION_PLAN](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b159d721cc9e7cee39c246a2b55425e63796b249/docs/research/pr_reference_feedback_q4_0_20261110/NEXT_NUMERICAL_VALIDATION_PLAN.md)。本書は実装との対応と具体的callを補足し、その物理条件・比較domain・CI family・探索範囲を変更しない。

## 1. SOURCE・入力・数値規約

新SOURCEはこの公開commitの`pr_reference_feedback_q4.py`／`_certify.py`／`_check.py`と`run_pr_reference_feedback_q4.py`。各hashと固定入力12件は[metadata preflight](../../../artifacts/pr_reference_feedback_q4_1/2026-11-10/preparation/PREFLIGHT.json)。旧入力はすべて`08f9cbe77456ec29d50a992d7165c4d726df7bf7`のGit blobから読み、元科学SOURCE f2638a78を変更しない。

- Q0 INPUTS／SOURCE／RESULTS.json.xz／CHECKS／STOP：ρL・β・disk・旧340とL2 1252・理想Hoeffding強対照witness。
- Q3-B B1／B2／CHECKS：prefix受理、κ、feedback算術、energy復元丸め。587,815組の監査は再走査しない。
- Q3-C/D C0／D／COST／CHECKS：共通core/readout、保存340十分保証、native単価／prefix費用。旧価格campaignは再実行しない。

Linear H4／1.00 Å／STO-3G／DF12、Partial S2／LD3／h=.02／K2、元H厳密ground供給・準備費除外、m0–16保存prefix、m17 q131072/r32/bits88、energy精度1.5936e−4 Ha、phase cap .4、prior |u|≤4/5、hull径≤4/5を維持する。全failureはprefix34/720＋末尾1/360=.05。各armは自分の保存ρL/β/diskを使い、真のE0／ρ／gap／stateの古典記述・校正dataを追加しない。

192-bit directed interval、exact Fraction入力、registered native a、同じideal-native／canonical IID finite-bitを維持。α／中心の停止分解能2^-128は旧計画どおり。新CP rootの2^-160は端点包含の技術的bracket目標であり、物理精度や成功条件を緩めない。曖昧なrootは外側bracketを広いまま残す。測定n0は無情報CI、測定軸のαは正。

## 2. H：強X/Y、1 campaign

2 arm×ideal/nativeの4独立task。登録CIは旧two-sided Hoeffding。上限budgetは1〜1252、各中心は実際の総数1〜budgetと全整数配分から選べる。αX+αY≤Aの連続配分を扱う。Qは既存H340を再利用し、旧Q全n走査を再実行しない。

量化順序は`min budget : forall ψ exists prefix-only (nX,nY,α) forall count`。実装は各budgetについて中心区間を覆う。点中心で配分・αを探し、見つかった合法allocationをcell全体と中心数値幅のpaddingへcertifyする。失敗下界には、**同一中心でbudget以下の全実総数・全配分・α全域**のwitnessが必要。異なる中心でのexact-Nの失敗を寄せ集めてbudget下界にしない。

理想幾何のみ、旧Q0の≤340一括排除を保存証拠としてpruneする。nativeへその下界をそのまま移さず新しい幾何で評価する。実H4の到達中心へ下界を一般化せず、全ψ robust taskの結果とする。

全countは整数boxで覆い、単調なCI端点による外側帯からgeometryをcertifyする。FAILは真の軸応答がCI内側に入る共通ρ/b二点でのみ証明する。continuous αはbranch-and-boundで覆い、gridだけのPASSを認めない。lookupの実際のallocationと証拠を保存する。

出力はbudget下界／実行可能budget／gap／lookup／同一中心での排除witness／未解決領域。数学的幅のみとenergy復元丸めまでPASSを区別する。PASS monotonicityを仮定せず、下界証拠の欠けたbudgetを排除しない。最初の保証付き可行budgetで上方の追加探索を止め、未確定gapを勝手に縮め続けない。

## 3. E：matched等尾二項CI、1 campaign

2 arm×X/Y・Qの4独立task、nativeのみ。X/Y budget 1〜340（全整数配分・prefix依存連続α）、Q n1〜340（全count）。CPの同じ等尾規約を双方へ与える。Hoeffdingの下界を流用しない。

Qは旧prefix後固定quadratureで、m17中の逐次変更なし。保存feedback誤差をbiasへ一度加え、κ／ωEを保持する。新単軸inverseを使用し、旧n340 decoderやωQ0を新CI用に書き換えない。旧Q-H340は別ラベルの有効なfallbackとして保全する。

双方の下界・可行n・gap・全count証拠を保存。X/Yが340以下を排除しても、このCI familyとrobust domainの限定下界だけ。341以上へ拡張しない。Qも同じCIで強化した結果と比較し、X/Yが旧340を下回っただけでQ優位の消失としない。

Hが最適性未確定でも独立したEを実施する。共通signalモデル／包含保証が破綻した場合はそのまま継続しない。

## 4. C：条件付き保存価格接続、1会計＋検算

新しい資格済みnative可行解がある場合だけ、保存COSTとpure Fraction helperを使う。新可行解が無ければ`NOT_RUN_NO_QUALIFIED_NEW_SOLUTION`を記録する。可行解でないものを仮想価格policyにしない。

`GXY=Gprefix+nX µX+nY µY`、`GQ=Gprefix+nQ µY`。µQ=µYの保存SOURCE条件とP費用を保持する。RZ／CX／size／shot加重depthを別々に評価し、qubits、未価格の古典feedback／optimizer／校正／生成費は別欄。

µは保存lawのtrajectory期待価格。主比較はprefix履歴に関するworst-caseで、履歴分布を新しく仮定しない。共通prefixを費用差から厳密に消し、YとQの同じµYを使う。全価格端点を独立に組み替えるより鋭い、同じ価格変数の安全側box上の境界を使う。

shot下界からの費用下界と可行lookupの費用上界を分ける。Q上界が指定familyの最小費用下界より小さい場合のみQ優位、baselineの可行上界がQ family下界より小さい場合はbaseline有利、そうでなければ境界重複／最適性未確定。shot最適と資源最適を同一視せず、新しい費用目的のoptimizerへ自動拡張しない。世界最良推定器・一般のPR優越・新規性の正式判断にはしない。

## 5. 実行・検算・保存

現在実行済みは人工テスト14件とmetadata preflightだけ。science出力directoryは**存在しない**。

科学実行を明示承認された後だけ、既存venvで次を1回実行する（未実行）：

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/new-a-total-cost-options-20261109
/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-c-h4-parallel-pilots-20261106/.venv/bin/python -B scripts/run_pr_reference_feedback_q4.py --campaign artifacts/pr_reference_feedback_q4_1/2026-11-10/science_run01
```

この固定namespace以外をrunnerは拒否し、既存directoryも上書きしない。campaignはH→E→条件付きC。各taskの完了結果を順次保存する。SOURCE、H/E各4個の`.json.xz`、C、CHECKS、STOPを独立領域へ保存し、partialやUNRESOLVEDを成功に変更しない。例外ではpartialを保持したSTOPとなり、自動再試行しない。旧STOP／KEEPに触れない。

各task内でcheckerを呼ぶ。保存証拠だけの再検算も可能（将来承認された同じcampaignの検算、現在未実行）：

```bash
/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-c-h4-parallel-pilots-20261106/.venv/bin/python -B scripts/run_pr_reference_feedback_q4.py --check artifacts/pr_reference_feedback_q4_1/2026-11-10/science_run01
```

checkerはoptimizerを呼ばず、count boxの完全被覆、α全域、共通中心、真のjoint二点、CP rootのexact polynomial bracket、丸め込み幅、gap、Cの同端点会計を確認する。primitiveは共有するので、別研究者の独立レビューとは呼ばない。

独立task数と実際のCPU affinityに応じてprocess数を決める。Hの4分類、Eの4分類を並列化し、CI／同中心allocationをcacheする。過去の小さいworker上限は持ち込まない。共有package／OS／他process／他worktreeを変更しない。実際の大規模runtime・memory・最適性gapは未測定で、人工テスト時間から推測しない。

## 6. 継続・STOP

固定SOURCE/input/statusの不一致、shared latentモデル又はgeometry包含の反証、CI coverage破綻、failure超過、192-bitでの必要な包含不能、旧one-shot／STOPとの衝突、科学条件変更の必要がある場合はSTOP。未解決leaf／最適性gapはそのまま保存する。geometryの有限候補witnessが見つからないことをFAILとしない。

このscope完了又は科学的STOP時点で成果公開・remote照合後mandatory STOP。別CI、α tailの非対称化、追加時間、Bayesian particle、逐次停止、Full／Truncated、回路build／compile／sampling、合成、総T、別Hへ進まない。

**今回の状態：PREPARATION_COMPLETE_MANDATORY_STOP。** 次の担当はユーザー（このSOURCE／入力／H/E/C scopeの科学実行許可の判断）、承認後のCodex（固定campaign・検算・公開）、結果後のGPT／ユーザー（研究貢献・主claimの判断）。
