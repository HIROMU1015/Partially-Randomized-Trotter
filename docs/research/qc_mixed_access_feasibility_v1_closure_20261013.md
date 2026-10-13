# THRIFT型mixed-access：H4成果確定・現feasibility系列終了

2026-10-13 JST。基準公開commit `ac930c0e0d255a187dc21917104e5cb5b4d09bcb`、branch `qc-mixed-access-feasibility-closure-20261013`。
[完了GPT科学レビュー原文](../reviews/representation_exploration/qc_mixed_access_feasibility_v1_scientific_review_2026-10-13.md)、[受領記録](../../artifacts/qc_mixed_access_closure/2026-10-13/review_intake.json)、[判断記録](../../artifacts/qc_mixed_access_closure/2026-10-13/research_decision.json)。

**ユーザーが採用・反映を指示した完了GPT判断に従い、現bitwise可変角回転・occupation算術templateの詳細な可逆function backend実装へ進まない。現H4 feasibility検証系列を終了し、限定成果と再開条件を固定する。**
これはTHRIFT・mixed-accessの量子化学適用全体の終了判断ではない。研究目的は元量子化学Hamiltonianの基底エネルギー計算の低コスト化を維持する。
今回は原文保存と文書・metadata整理のみ。新分子/SCF/DF/eig/SVD/PF/保存値の新数値集計/証拠再認証/synthetic/compile/GPTレビューは0。原科学結果・source・認可・STOPは変更しない。

## 固定する成果と成立条件

[一次結果](qc_mixed_access_feasibility_v1_results_20261013.md)、[固定保存集計](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/descriptive_summary.json)、[claim map](../../artifacts/qc_mixed_access_closure/2026-10-13/claim_map.json)。以下の数値は完了レビューと既存資料の転記で、新しい評価ではない。

H4直線鎖1.00Å、STO-3G、実RHF canonical MO、charge0/spin0、4 spatial/8 spin modes、ν4、frozen coreなし。development一入力でblind transferではない。
DFは補正one-bodyと全16 factorsを保持。1e−12 numerical rank10は報告用のみ、切捨てなし。L_Dなし、PRなし。
T={1,4} atomic time、r={1,2,4,8,16,32,64,128,256,512,1024}、7法、154 PF evaluations。
metricは全256 output rows×ν4の70 input columnsの作用誤差、controlled phase・sector leakageを保持。εU=.001はoperator errorで、ground-energy1mHaではない。
PF fractions .4/.6/.8、係数/function/synthesisの登録配分、numeric margin、次の2rと全大きい登録rを確認するeligibilityも元契約のまま。

### F1：理想PFのstep削減

共通PF fraction.6（PF予算.0006）の最小eligible r：

| 方法の比較 | T=1 | T=4 | 固定した意味 |
|---|---|---|---|
| R4 対 E4/D4 | 1 対 2 | 4 対 8 | 同次数のideal PF、指定gridでr半減 |
| R2 対 E2 | 8 対 8 | 32 対 64 | T1は同r、T4はr半減 |

この肯定的な結果は保持する。各methodの一step費用は異なり、step低下をgate数・compiled総費用・QPE全taskの利益へ読み替えない。

### F2：登録費用モデルの予算不足

[原result](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/result.json)の`NO_POSITIVE_BUDGET_IN_REGISTERED_MODEL`を保持する。
1386 resource rows、273 eligible R frontiersで、指定template/精度配分/κ3,6,12価格仮定の正remaining budgetは0。
T_RZ(η)はUNKNOWN、κ価格はhypothesisで、実T合成・compile達成値や最適回路lower boundではない。function compute/erase、guard/sign/zero/time/cleanupはUNKNOWNを保持し、実費用0と認定していない。
unknown functionの実費用が高かったことを敗因にはしない。現会計のknown成分に既に正余白がなく、known会計を維持する限り非負の追加function費用だけを安くしても逆転しない、という条件付き帰結を保持する。

### 研究判断とclaimの限界

完了レビュー§10–12の判断：現templateへの追加backend投資を見送り、同H4でのr/価格/配分救済scanや即時の別分子追加を標準の次作業にしない。
F1/F2は一分子・有限grid・保守的template・価格仮定・作用誤差taskの結果。任意価格/回路の不利、THRIFT一般の不能、全量子化学用途の無価値、ground-energy全費用の勝敗は示していない。
現constructorの独立新算法としての非採択は前の完了判断を保持する。本一分子結果で独立論文の主要claimが完成したとも認定しない。
これは投資配分に関する受領GPT判断の反映であり、Codexの新しい科学レビュー・新規性判断ではない。

## 現系列の終了と再開条件

[closure state](../../artifacts/qc_mixed_access_closure/2026-10-13/closure_state.json)。現feasibility系列は`CLOSED_CURRENT_TEMPLATE_FEASIBILITY_SERIES`。
詳細sqrt/atan2等backend、T合成、compile、追加benchmark、保存値の費用帰属細分化、別geometry/basis/molecule/精度/gridを自動起動しない。
消費済みone-shot・retry0とMandatory STOPを維持し、別namespaceや旧認可で再実行しない。N1STOP/N2HOLD/N3CLOSED、別Trackの状態は変更しない。

再開の検討は、次のいずれかの具体的な根拠が得られた場合に限る。

1. known費用Kそのものを変える構成：操作列、coherent control、phase/parity、sign、precision、uncompute、補助資源/準備費用を含め、同機能の強い既存対照と比較できる費用差を示す。
2. 必要stepやアクセス単価を変える適用条件：少ないtransition/occupation係数、既知に安いmixed access、より大きいstep削減など、今回の障害が変わる機構を結果取得前に示す。「別分子なら勝つかも」という期待だけでは延長しない。
3. 現claimや結論を変える具体的な矛盾・新証拠が得られる。

新primitiveの発明を一律の条件にしない。既知の回転・共有法でも上記の差が具体的なら検討できる。レビューで例示されたphase-gradient/QVR、線形Z、occupation共有等は、今回採択した成功案ではない。
Kを変えず未知Fだけを安くする案と、K/step/精度/推定器を変える案を区別する。後者は新しい科学的問いと比較契約を必要とする。
再開の重要科学レビューはGPTで別途開始承認を得る。計算・compile・再認証を伴う場合は、その固定scopeと実行承認を別に得る。経過時間・文書完成・新branchは認可ではない。

## 証拠と保全

- 科学source：`102e6b31401356d6eea775169041ebdf2cfa6ba0`、科学実行承認：`214f9ab2d7d3e0b64b287f14f49795c99a1a8244`。
- 固定結果payload：`ae3975de24bff6ab6aec3494b522851db2a8f3dd`、公開確認head：`ac930c0e0d255a187dc21917104e5cb5b4d09bcb`。
- input SHA：`f345af12281836731c3122bb2a4c0c2b389d7c90ded7ba16976d3cfeeb7bd951`。
- [全旧優先証拠索引](../../artifacts/qc_mixed_access_closure/2026-10-13/existing_evidence_index.json)：既存source/科学契約/入力/raw/certificates/resource/認可/STOP/公開記録のbyteを保護する。
- [既存saved audit](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1/independent_audit.json)、[保存算術監査](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/saved_execution_integrity.json)、[既存公開receipt](../../artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/publication/receipt.json)を参照。ここで証拠再生成・再計算・proof replayを行わない。
- [今回の決定履歴](研究ノート/2026-10-13_qc_mixed_access_feasibility_v1_closure.md)。過去の研究ノートは変更しない。

受領レビュー原文はSHA256 `6ad06d510af9a428b90691387ebd19e79fe97d01438dba4360d590ceb4e05dc7`、32,157 bytesをそのまま保存する。誤字・引用・linebreakも編集しない。
旧126 manifest result setsと全non-result-set fieldsを維持し、今回の成果固定/公開entryのみ追加。既存`relevant_commits[7].description`欠落FAILを保持する。metadata/SHA照合をCIや外部科学再現へ昇格しない。
旧結果報告とGPT引継ぎは当時の本文を維持して末尾にclosure pointerを追補し、「レビュー開始待ち」という過去stageを黙って改変しない。現在状態は本closureと研究概要の最新追記から読む。

## 整理後の次の担当

現系列の成果固定が完了すれば、同じ詳細科学レビューは不要。
次に新しい研究方向を比較する場合は、研究全体の未解決課題へ戻り、以前の次点「基底固有位相向けPF」も含む候補をGPTで検討することをユーザーが推奨している。
現THRIFT構成の継続最適化を前提にしない。既存PR/PF/DF/THRIFTで解決できていない問題と構成原理を問うが、この文書で問題・算法・候補を新規採択/詳細分析していない。
新方向の比較・選定はGPTの別途開始承認、科学計算/compileはCodexの別途実行承認が必要。今回の作業は成果固定・終了記録・公開確認まで、Mandatory STOP。
