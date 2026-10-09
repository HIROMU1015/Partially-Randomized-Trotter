# Track B：G1結果前契約・入力の準備

2026-10-09 JST。利用者の「その方針で進めてください」に従い、直前に示した①②の準備を行った。
基点`86df033cea392664cf365a80e5f068091dc8f5e8`、独立branch
`track-b-g1-result-prior-preparation-20261009`。

[準備報告](../../tracks/algorithm_codesign/g1_result_prior_preparation_20261009.md)にG1 packetのscopeを記録した。
数学は固定7 prototypeに対する6頂点・precision復元・v4への対応と限界の独立監査に限定。
backendは旧未実行6件をobject不変で使い、新しい高精度2件を可逆な対角変数変換で結果前に固定した。
この2件は最大109桁の分母を含むが、旧10^-99 gap failureの再実行・修復ではない。

全体wall 1,200秒、各backend command 30秒、RSS/address space 1,536 MiB、output 64 MiB、
solver 8 calls、echo 8 calls、runs=1、retry=0を契約化した。構造監査は60秒・256 MiB。
guardのCPUは測定scopeを固定し、未実装のCPU hard capを主張しない。

証明未取得と取得済み不正証明を分離し、全8件の独立certificateが揃う前に全体PASSとしない。
初回失敗で停止し、prefixから研究的positive/negativeを出さない。
全outcomeでGPT G1へ戻し、production・登録optimization・新science authorizationへ進まない。

静的preparation checksはPASS。旧6件identity、canonical rational形式、8件dim/bounds、
追加2件の固定変数変換、既存binaryとlibrariesのhashを照合した。
本構造監査、LP solve、proof verification、backend invocation、tests、synthesis/scienceは全て未実施。
旧source/contract/authorization/result/marker/STOPは不変。

限定監査script/controllerとoff-domain launch testsは未実装。
判定は`G1_CONTRACT_AND_INPUTS_PREPARED_IMPLEMENTATION_PENDING`であり、実行readyまたはauthorizationではない。
この準備文書と入力をcommit・pushし、ここで停止する。
