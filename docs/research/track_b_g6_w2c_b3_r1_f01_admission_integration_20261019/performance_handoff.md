# 高速化担当へ渡すR1技術baseline

別サーバーCodexのlive branch/worktree・未公開結果へアクセスしていない。結果を評価・採択したとは扱わない。
今回scheduler/cache/buffering/保存内容/FunctionObserver/Meterの性能変更・新性能benchmarkは行っていない。

後続担当Codexは、ユーザーが指定した**固定公開commit**だけを受領し、今回baselineとの差分を次の順で照合する。

1. F01のcurrent attempt/last completed/candidate vs committed/emit正常return/partial tail/primary-secondary例外を保存する。
2. host/Boundary/portの全pin、独立approval、scope/claim/store/output、atomic spentと外側STOPを保つ。新entrypoint/workerも必要に応じてpinへ含める。
3. sampling law/bit消費順/seedprefix/X-Y識別、phase/weight/word、R/C/A、angle/requery、保存N/error/confidence、16cells/予算が変化しないか比較。
4. 24 constructor/prepare3/preflight9、X/Y共有、cache禁止、二pass、保存全文/compact/hash/stream/上限とsource/state guardを比較する。
5. schedule/worker順やobserver/保存方式に差があれば、性能効果と比較scopeへの影響を明示し、ユーザー採択後の別versionへ反映する。
6. SOURCE/host/schema/pin/正式契約の新versionを作り、必要な正常8ケース・統合275件・worker障害/correctness gate/一意性・STOP人工回帰を再実行する。
7. 公開/保全/実import path照合後、科学実行用release・正式digestを実値で別途固定する。最後にユーザーからscienceと実authority/claimの別承認を受ける。

上記は提案する後続技術作業。今回自動着手しない。高速化の速度倍率・優位・新scheduleの採否は未評価。
