# E/T v4 one-shot execution authorization

2026-10-12 JST。利用者の明示指示により、固定source Sのauthorization-only直接子Aから一回実行する。

- Source S: `016e02b486cd4b182b654af18743ef1ef50fe7e6`
- Branch: `track-b-df-correction-et-v4-one-shot-execution-20261012`
- Contract SHA256: `c2d2c91004e6fbe78b08a38620bc96f1d099d8cdfb514abfcdedb267cab3a77b`
- Source manifest SHA256: `94d7013e7e6a34e2cd2a2efc409b832b03176660e31a95aaee1a4b36e993a6b8`
- Source-review policy: accepted GPT layered-acceptance review + Codex focused technical validation (141 PASS / 411 audit checks); no unresolved new scientific blocker identified.
- Runs=1, retries=0, mandatory_STOP=true.

明示指示の原文はauthorization JSONの`explicit_user_instruction`に末尾空白を含め保存。内容:

> source `016e02b486cd4b182b654af18743ef1ef50fe7e6` の固定契約v4で、authorization-only直接子Aを作成・pushし、E/T比較を一回だけ実行してください。retry=0、旧結果・markerを保持し、終了後はmandatory STOPしてください。結果と監査資料をcommit・pushしてください。

E-r3/r6 saved inputsのuncontrolled/controlled、最大22 native compileと最大64 T keysのみ。CDは保存済み6行の条件付きsubstage参照で再計算しない。fixture/seed/rank/sign/u/compiler/precision/error/capsを変更しない。旧source/expected/results/auth/consumed markers/STOP保持。partial/numeric/storage/cap failureでもretry禁止。全outcome後に科学実行STOP、保存値/provenanceのread-only監査と資料公開だけを行う。科学的勝敗・次stage採択はGPTへ戻す。

HEAD=A、parent=S一つ、diffはこのreceiptと新authorization JSONだけ、remote=A、clean、manifest全369files、固定runtime、新namespace未作成をlaunch前に確認。実行HEADとしてsource公開確認commitを使用しない。
