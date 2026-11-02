# Fused canonicalの接続と互換性

ad3117d287c4c28c9b861b3b8dffa7bfbba407e1のexact_serializationをbytes不変で追加した。
新driverのidentity、Stream payload、event traceを一つのcanonical入口へ接続する。
P/Sはreference `audit.canonical(exact(value))`、PF/SFはfused。SOURCEは共通で、codec設定もfake承認へ束縛する。

exact int/Fractionのhex/rat、str(key)・順序・collision last-wins、tuple/list、ensure_asciiの意味を維持する。
Unicode/surrogate/DEL/control、巨大整数、container subclass、unsupported-overwritten値、拒否・例外順を検査する。
特殊値はreferenceへfallbackする。F3の独自Evidence codecや科学SOURCEは変更しない。
冪漸化式、GMP、orjson/native JSON、buffering変更は対象外。

新30 canonical/reference検査には1000人工tree、全Stream bytes/chain/file SHAのfake-clock照合、6種類のF01全文一致を含む。
追加25 F01検査は新driverで実行。旧逐次execute専用の1件は旧suiteで保持し、統合Hostの失敗/cleanup/STOPケースで別途検査した。
新helperはSOURCE pin/runtime import manifest/observerの実pathへ含める。
通常fused pathはexact/_pack二重copyから新関数へ移るため、旧同名関数のnsだけを差し引いて利益を作らない。

実時刻/実測generator CPUを含む異なるrunのfile SHAを同一とは要求しない。
各fileのcanonical/chain/file SHAは個別に再読検査し、非決定metadataを除いたevent/bit/traceの同値性を比較する。
同じpayload/fake clockでは全bytes/chain/file SHAが一致する。
結果は人工技術範囲の互換性であり、実G7全domainや正式SOURCE採択の認定ではない。
