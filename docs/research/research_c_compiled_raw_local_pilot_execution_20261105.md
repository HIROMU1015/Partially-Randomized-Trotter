# 研究C：局所support C/R pilot・登録4-call早期STOP

2026-11-05 JST。固定6-call上限の保存pair検証を一度実行した。
**4/6 logical calls COMPLETE、FAIL/retry/resume 0。正規化CX skeletonが対応せず、契約どおり早期STOPした。**
局所Arb評価は0、新しいC/R上界はnull。未使用2 unitsを別の検証へ転用しない。

## 固定SOURCE・対象

- 科学SOURCE：`6bcf68d55ddeab852b275940fcdc78bccceba04f`、tree `45d21bbaec3e29888be0801be2c6e65dc6013eab`。
- [固定契約](research_c_compiled_raw_local_pilot_unapproved_20261105.json)、[技術準備](research_c_compiled_raw_certificate_acceleration_20261105.md)。歴史的unapprovedファイル名・metadataを変更していない。
- 独立branch：`codex/research-c-compiled-raw-local-pilot-execution-20261105`。
- Linear H4・1.00 Å/STO-3G、number CDF8/LD2、partial S2/s1/r32/K0、h=δ=.1、controlled512。
- Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
- `CDF_proposed:F6:0` と `CDF_proposed:F7:6,7` を16回交互に配置した保存32-event wordのcompiled/raw recordだけを使用。
- 既存P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx、MPFR160 declared phase、Gaussian/relative fusion、own-DF oracle・状態準備除外、IID53-bit、CA/RPE政策を維持。
- 新評価器はArb192・full512 operator用の局所support方式、`commute_CX=false` を固定。今回はその数値評価前に停止した。

実行前にSOURCE187ファイル・保存input5ファイルのhash、word/record/phase identity、依存version、fresh条件を照合した。
CPU affinity32、available RAM約56GB、load約1.41、ディスク空き約20GB。構造planは依存する順序で一度生成し、並列化対象の局所Arb stageには到達していない。CPU/RAM/worker/wall capは追加していない。
既存supervisorを外側の運用wrapperから再利用し、数値SOURCE・runner・科学契約は変更しなかった。共有環境・他jobは操作していない。

## 実行結果

Fresh出力：`artifacts/research_c_compiled_raw_acceleration/2026-11-05/local-pilot01`。
無変更entry `scripts/prepare_research_c_compiled_raw_acceleration.py --execute-local-pilot` を1回だけ呼んだ。

| logical unit | 実績 |
|---|---:|
| records_load | 1 COMPLETE |
| identity_gate | 1 COMPLETE |
| structural_plan | 1 COMPLETE |
| local_certificate | 0 |
| comparison | 0 |
| summary | 1 COMPLETE |
| 合計 | **4 COMPLETE、FAIL 0** |

| 項目 | 結果 |
|---|---|
| 判定 | `NOT_APPLICABLE_NO_SHARED_CX_SKELETON` |
| 正規化後CX events：compiled / expanded raw | **2894 / 2912** |
| Candidate展開前後のgate列長：compiled / raw候補 | 11386 / 32073 |
| 登録inverse pair消去：compiled / raw候補 | 0 / 9752組（CXだけの件数ではない） |
| raw候補のunique primitive definitions | 175 |
| 対応block pairs | 0 |
| 局所Arb評価 | **0** |
| 新しいC/R誤差上界 | **null** |
| plan wall | 約0.645752秒 |
| supervisor全wall | 約1.367998秒 |
| returncode / terminating signal | 0 / null |

上表は保存された構造planと実行記録の表示であり、追加の正規化や数値評価を行っていない。
CX件数が異なるため、今回の固定方式が必要とする共通skeletonが成立しない。どの変換が18件の差や順序差を生んだかは、この実行では切り分けていない。
全runの短いwallは**適用性判定の費用**であり、旧1173秒のC/R上界計算からのspeedupとは比較できない。

記録：[result](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/local-pilot01/result.json)、[structural plan](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/local-pilot01/structural_plan.json)、[journal](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/local-pilot01/science_calls.jsonl)、[STOP](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/local-pilot01/STOP.json)、[SOURCE copy](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/local-pilot01/SOURCE.json)、[contract copy](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/local-pilot01/contract.json)、[preflight](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/local-pilot-execution-preflight.json)、[終了状態](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/local-pilot-launch01/exit_status.json)。

## 科学的な意味と限界

これは登録された適用性gateによる正常な早期STOPであり、数値包囲のcorrectness FAILではない。
Compiled/rawのoperatorが不一致であることを示す結果でもない。変換・消去後に同じCX列を対応付ける現在の方式が、この保存pairへ直接適用できないという結果である。

既存7-callのC/R上界 `1.585733058571e-11`、Stage Pの固定word包囲、inverse等の旧証拠は維持する。
新しい上界・phase/inverse/shell資格・速度改善は取得していない。上界を0や2で補完していない。
人工例の約162倍短縮を実H4へ移送せず、`commute_CX=true`、別word、dense column fallback、別compiler、追加callへ切り替えていない。

新sampling/native build/compile、inverse・R/Dの再評価、期待bias/tail/分散校正、RPE shots/energy、Stage C、actual CA、重要科学レビューは0。
Compiler全word成功、failure-aware scope、N・δ・abort許容・cache/非停止政策は未採択のまま。

## 保全・検証・次の担当

SOURCE187件・input5件、SOURCE/契約コピー、call journal/STOP、plan digestとrecord identity、旧準備・旧7-call参照の不変更を実行後にも確認した。
旧SOURCE/raw/契約/grant/ledger/marker/STOPを変更していない。準備commitで人工55件・関連290件PASSした同じ数値SOURCEを使用し、実分子の追加testや再実行は行っていない。
照合：[local-pilot-verification.json](../../artifacts/research_c_compiled_raw_acceleration/2026-11-05/local-pilot-verification.json)。

次の提案は、**Codexによる保存planと既存compiler sourceの技術的な構造差調査**。
より大きい局所blockの厳密対応や、追加の厳密rewriteが必要かを調べる候補であり、今回その調査・修正・再検証を自動開始しない。
ユーザー／研究Cが後続作業を判断する。科学的な再評価には新しい固定SOURCE・契約・明示許可が必要で、残2 unitsや旧予算を使用しない。

結果commitは保存・閲覧用であり、科学SOURCEを付け替えない。公開後remote SHA/tree・本文・主要blobを照合して終了する。
**Mandatory STOP。**
