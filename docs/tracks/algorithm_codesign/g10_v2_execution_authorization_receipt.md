# G10 v2 explicit one-shot execution authorization

Source S2: `a139b91f119d109430ae3154a045d0fdcf722233`.
Review reference only: `0e6a6f115a16691813fefbaa95d68ee9cb70c6cf`; it is not the execution source or parent.

Contract: `artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2/contract_v2.json`
SHA256: `5e464aa91ff46571e28152fc10dc39377822fba7d15baf81672af72c8ee5612e`.

The user explicitly instructed:

> source `a139b91f119d109430ae3154a045d0fdcf722233` の固定契約で、authorization-only直接子A2を作成・pushし、G10 v2を一回だけ実行してください。retry=0、旧結果・markerを保持し、終了後はmandatory STOPしてください。保存結果と外側processのstatus・provenanceを監査し、資料をcommit・pushしてください。

The authorization commit is a direct single-parent child of S2 and changes only the new authorization JSON and this receipt. Execution starts from that clean remote-matched commit. runs=1, retries=0, mandatory STOP=true. A fresh v2 exclusive marker is required; all prior results, markers and STOPs remain unchanged. No old failed G10 sequences are imported as rescue cache. Scientific conditions, caps and tool identity remain frozen.

After the one run, only saved-value/outer-process/provenance audit and publication are authorized. File completion alone and exit code zero alone are insufficient: outer terminal status, payload/token/STOP identities and provenance must all agree. Any failure consumes the one-shot; no retry, source rescue or next science stage. Research interpretation returns to GPT.
