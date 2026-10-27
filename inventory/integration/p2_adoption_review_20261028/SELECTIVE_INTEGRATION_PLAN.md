# P2選択統合：次回実行できる固定計画

今回計画のみ。統合先 **`codex/pauli-jw-research-summary-20261026`**、着手時remote tip/計画base **`b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`**。
P2 donor `04f4cd9cdc8e348a9f0a182ead9cd07f1ee69696`、H6依存donor `00958ae1945c780718b2deef07d5dab1096a5cd8`。
main/P1/H6実行branch/元dirty checkoutは変更対象外。新しい統合用branchを作らない。

## 正確な対象14 path

P2からADOPT_AS_IS5＋ADOPT_WITH_CHANGES8の計13 path：

```text
src/trottertracks/pauli_jw/__init__.py
src/trottertracks/pauli_jw/contexts.py
src/trottertracks/pauli_jw/adapters.py
src/trottertracks/pauli_jw/core_stages.py
src/trottertracks/pauli_jw/pipeline.py
src/trottertracks/pauli_jw/pipeline_fixture.py
scripts/run_pauli_jw_research_pipeline.py
tests/test_pauli_jw_research_pipeline.py
PROJECT_MAP.md
docs/README.md
scripts/README.md
src/trotterlib/README.md
tests/README.md
```

既存H6依存の追加1 path：`src/trotterlib/h6_hybrid_fairness_extension.py` @ `00958ae1945c780718b2deef07d5dab1096a5cd8`、blob `0f09c36f1b191abba75e39f56746a1828f4419f8`。
その数値bodyを編集しない。56同blob依存と2 global initializerは移植/編集不要。
P2履歴文書5件は固定URL参照とし今回の統合先へ複写しない。
新統合の実装・実測testは既存docs/README等の最小追記に保存できる。追加の科学manifest/approval記録は作らない。

## 必要な実装修正

1. **pipeline.composeの表示整合**：StageResult.bindingだけでなく、存在する`method/candidate_id/prefix/q/r/K`を対応context値と照合する。
   既存`PF_order` keyは`context.pf_order`へ明示mappingして照合する（keyを黙って変えて旧recordを再解釈しない）。誤表示はそのstageでFAILEDにし次のcallbackへ進まない。
   現caseでは同bindingに`method=PF`、`K=0`、`PF_order=Y4`を置いても末尾までSYNTHETICになる。これを拒否へ修正する。
2. **synthetic providerの欠損**：Providers.coreを属性参照する前に存在/typeとkind/scope、必要callbackを確認しPipelineErrorへする。None/非互換coreでAttributeErrorを漏らさず、input呼出し0。
   本人許可の証明やcallback真正性検証システムは作らない。通常のAPI欠損検査に留める。
3. **binding診断**：core_stages.bind_existing_coreのmodule属性が無い/非callableの場合をPipelineErrorとして明示し、元例外を原因へ保持する。
   ModuleNotFoundErrorの既存変換を保ち、別名package/fake scientific fallbackを加えない。
4. **testの基準版**：initializer不変testの`git show BASE:path`を、未継承H6履歴に依存させず統合前 `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b` の同blobと比較する形へ変更する。
   P2内の固定SOURCE/STOP/result locatorのBASE定数はそのまま。この2種類のbaseを混同しない。
5. **案内**：5既存文書はsummaryの本文bytesを保持し、P2追記だけ採用先に合わせる。H6旧/新成果は別固定refへのlinkとする。
   新test実測と旧P2 109 PASS/1 SKIPを分け、固定P2文書URL/本採用計画URLへリンクする。科学synthesis/CEM/manifestや旧比較表は書き換えない。

## 実装順序

1. ユーザーが本14 path技術実装・限定test・通常commit/pushを明示許可する。
2. remote tipを再照会。進んでいれば上記14 pathと57依存の関連差分だけ再確認し、新tipを作業baseとして記録する。重要な競合は保全して戻す。
3. 本人専用の独立clean checkoutを対象tipからdetachedで用意する。既存worktreeを切り替えず、外部PYTHONPATH/object alternatesをruntime依存にしない。
4. 既存H6 code1 fileを固定donorからbytesそのまま追加し、既存56 path/initializerを保持。H6の旧script/result/input契約を追加しない。
5. P2の新8 code/test pathを選択追加して上の最小修正を施す。全branch merge/rebase/cherry-pickを使わない。
6. 5索引へ必要な追記だけ加える。旧後続H6 docsを全文移植せず、固定source/結果/handoff URLを使う。
7. 静的signature/target module/同blob依存/initializer、人工3scope/全4H6/拒否/欠損/STOP/record/CLI/no-siteを限定testする。
8. 新規修正caseの拒否とcallback未呼出しを確認。正常依存環境が無ければ実writerはSKIP/未確認として保持し、人工writerの衝突testと区別する。
9. 元dirty checkout/旧runner/科学assets不変、変更14 pathだけをstage。新統合の実測件数を記録して通常commit。
10. remoteが再び進んでいないことを確認し通常fast-forward push。進んだ場合force/reset/rebaseせず新tipで許可差分を再構築する。
11. local/remote SHAと公開対象の全文/blob一致を確認してmandatory STOP。

## 受入条件

- 代表開発tipから明示CLI/composerを同checkout内で利用可能。H4 Hybrid/H6固定4/H6純RTE人工経路が通る。
- H6既存kernelが同path/同blobで1回だけ追加され、旧56依存とinitializer/runner/SOURCE/raw/result/cohort/STOPが不変。
- SOURCE/input/state/norm/seed/cohort/compiler/metricのbinding保持に加えmethod/PF/K等の誤表示を拒否。
- current STOP/FAILED/MISSING/UNQUALIFIEDから継続せず、science/real/resumeはinput/output/worker前に拒否。N/G/SE/rare probabilityの補完なし。
- missing core/callback/optional dependencyを明示。新出力directoryは依存解決後のみ作成、既存出力を上書きしない。
- 旧P2 109 PASS/1 SKIPと今回17観測PASSを合算せず、採用後の新codeで実行した限定testだけを新集計する。
- 通常名importの不足をalias loader/global lazy化/共有venv変更で隠さない。real数値/実compileの同等性をclaimしない。
- 保存summaryの科学的結論は時点固定、後続H6成果と新codeの役割は別に辿れる。

## 除外とrollback

real loader/provider、PF/B0/H2の新接続、候補拡大、数値law/mean/normalization/shot/G/SE、sampling/compile/新探索/旧結果再評価は含めない。
P3/B3/他研究は別作業。既存の承認/one-shot/retry/STOPはこの技術移植で変わらない。
問題時は新入口の使用を停止し、旧summary/旧runner/固定SOURCEをそのまま参照できる。
公開後に戻す必要があれば通常の逆差分commitを別指示で作り、reset/force/history書換え/branch削除をrollbackに使わない。
今回この計画の研究コード移植・代表branch commitは0。
