# C1 Stage3：Qiskit compile競合修正と再実行SOURCE

2026-10-25 JST。起点 `4c4a78d48b9d77f8a15cd11ad67968e5f11195ee`。
Branch `codex/research-c-c1-stage3-cost-compat-20261025`。

ユーザーの「修正を行って再実行してよい」「意図しないところで中断された場合修正して再実行してよい」に従い、旧SOURCE・旧STOPを保全して修正版を用意した。今回の許可は同じ8候補cost-onlyの修正・再実行に適用し、別の科学実行許可を要求し直さない。科学条件・数値許容差を変更する許可には拡張しない。

[旧技術的STOP](research_c_c1_stage3_cost_execution_20261025.md)／[新実行契約](research_c_c1_stage3_cost_compat_contract_20261025.json)／[人工再現・テスト・SOURCE](../../artifacts/research_c_c1_stage3_cost_compat_technical/2026-10-25/)。

## 原因の再現と最小差分

固定Qiskit1.3 `HighLevelSynthesis._definitely_skip_node()`の`self._equiv_lib.keys()`で前回の例外が発生した。equivalence libraryは固定環境ではRust実装の`BaseEquivalenceLibrary`を用いている。旧P0 wrapperは独立した回路を複数threadで同時に`UnrollCustomDefinitions`／`transpile`へ渡していた。

6qubit・512層の人工RY/RZ/controlled-phase/XXPlusYY回路を8件同時にcompileしたところ、**5 PASS／3 `Already mutably borrowed`**を観測した。H4・保存QPYは一切読んでいない。小さい32層の人工例は8 PASSで、発生が回路・実行時条件に依存する負の対照も保存する。再現用SOURCEと例外tracebackは上の人工証拠にある。

`src/trottertracks/c1_stage3_cost_compat.py`の`CompileSerialCalls`で、**当該processのP0 compile callbackだけ**をmutexで排他実行する。旧SOURCE19ファイル・Qiskit本体・共有venv・SessionEquivalenceLibrary・全体設定は変更しない。旧cost backendをsubclass化し、準備時にcall接続を置き換えるだけで、native emitter・P0条件・seed・角度・Operator照合・費用集計は元の関数を呼ぶ。

候補workの並列・Gaussian cache・数値Operator照合・metrics・費用展開は維持する。mutexを待つcompileは取得後に旧CallsのSTOP検査を受けてから計上するため、先行FAIL後に新規compileを始めない。排他化は実際に再現した共有mutable-state競合への対処で、CPU／RAM／workerの恣意的な制限ではない。compile区間の並列性は失われるので、性能向上は主張しない。

## 人工検証と保存

- 新規6 tests、既存94 tests、合計**100 PASS**。
- 512層の同じ人工8回路は排他化後**8 PASS／例外0**。
- 旧P0の逐次結果と、8並列requestを排他化した結果を別の人工8回路で照合し、全metrics一致、full Operator差1e-12以内、rawとの相対位相を含む差1e-10以内を確認した。
- compile最大同時実行1、非compile処理の並列維持、先行FAIL後の待機compile未計上、936-call人工全scope、fresh output・provenanceを検証した。
- 新契約のproposal／候補／call上限／元input／保存inputは旧契約と比較し、変更を拒否する。byte preflight PASS、旧科学値の再解析はここではしていない。

## 新しい固定契約と再実行

科学scopeは元の`C1_STAGE3_EIGHT_CANDIDATE_COST_ONLY_V1`を保つ。保存P-H4（直鎖1Å/STO-3G、Nα=Nβ2）、CDF R8 number・元g κ／XDF R10、δ0.1、m0..12、D S2 s1/s4・PR L_D2/s4/r16/K0/K2、own-DF oracle・状態準備除外・保存楽観shots、PR校正32、P0、同じ数値許容差・936 logical unitsを変更しない。合成／実装law誤差・CA・保証付き総資源優位は未資格のまま。

旧run01はそのSOURCE・106 calls・retry/resume0・STOPを保全する。新しいSOURCEでfresh output `artifacts/research_c_c1_stage3_cost_compat_science/<date>/<fresh-run-id>/`へ実行する。新attempt内のin-place retry／resumeは0。予期しない技術的中断時はユーザーの今回の許可に従い、証拠を残して修正・人工検証・SOURCE再固定後にfresh attemptを用い、全attempt／累積callを報告する。科学的な数値FAILは閾値変更で通さずSTOPする。

新entryは`scripts/run_research_c_c1_stage3_cost_compat.py`。既定はbyte preflight、`--execute --output <fresh>`で旧runnerを新互換backendへ接続する。

```bash
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python \
 scripts/run_research_c_c1_stage3_cost_compat.py \
 --contract docs/research/research_c_c1_stage3_cost_compat_contract_20261025.json \
 --input /path/to/saved/P-H4.npz --execute --output <fresh>
```

承認証明書・承認記録・新grantは作成しない。修正版SOURCEをcommit・push・remote照合した後、今回の許可で限定再実行する。完了後は結果・証拠を公開・照合してmandatory STOPし、次段階へ進めない。
