# N3保存帰属v1：一回の事後評価認可（2026-10-13 JST）

ユーザーの「次の作業に進んで大丈夫」を、直前に提示した固定N3保存データの数値事後評価＋既存3証拠の再認証一回への明示指示として受領した。
[固定plan](n3_saved_attribution_v1_execution_plan_20261013.json)と[限定契約](n3_saved_attribution_v1_posthoc_contract_20261013.md)の条件を維持する。過去のfalse template/planは履歴として保持し、新規[authorization](../../artifacts/n3_saved_attribution_v1/2026-10-13/execution_authorization_v1/authorization.json)により今回だけ認可する。

準備commit `239402c8e0825e6751e1ee1c2be313a8606d4087`、source `a7a9f42d53d5efaca8715bee26edd9f1a0bad999`、plan SHA `a5f3fd39e47372e277365d9873f234cbd882ae1a84b6ca25a4f051e339f0c61f`。
元6Hは非負density、full5/6・ν2/3、一候補一外部mode、2候補/H、元ε_proj=.01。geometry/basis/DF rank/L_D/PF delta窓は対象外。保存216 path/432 solver event/696候補を解析し、指定保存証拠3件を厳密に再認証する。

Python3.12.3、CPU2 worker、各1 thread、AS各1GiB・aggregate sampled RSS6GiB。保存JSONをcaseごと一回読み、stdlib有理数演算を使う。変更されていないsourceのsynthetic23件とspawn/serial一致が準備時に確認済み。現在の機器情報・identity-only preflightは同artifact directoryに保存する。性能倍率は主張しない。

専用execution branch、attempt1/retry0、fresh evaluation1に限定する。実行前は新しい数値事後評価・PSD再認証0。新Hamiltonian、新ε、新κ探索、新solver、新certificate生成、元科学batchの再実行は認可されていない。終了後は認可を消費してMandatory STOP、N1停止/N2保留、中心テーマnull。新結果のGPT科学判断開始も別認可とする。
