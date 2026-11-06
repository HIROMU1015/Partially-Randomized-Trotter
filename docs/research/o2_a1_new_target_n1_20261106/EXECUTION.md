# A1新finite target：N1限定実行SOURCE

2026-11-06 JST、SERVER1（GPUA100-gleap-01）。ユーザーが固定2b2e6c1の新targetを保存DFモデル限定で採択し、N1を明示許可した。
旧契約草案・旧STOPは履歴として保全する。この採択はN2/N3や、未採択の測定予算/ε/αの実行許可ではない。

対象：保存linear H4/1.00 Å/STO-3G/legacy DF12/8qubit/L_D3、m0/q1/δ=.02/r1/K2、threshold0。Ne4は後段stateの条件で、N1ではstateを生成しない。
NPZは既存`rte_connected_cluster_cost_validation`の固定snapshot、SHA256 `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。

## 実装とcorrectness

[新N1 module](../../../src/trottertracks/algorithm_codesign/a1_new_target_n1.py)、[runner](../../../scripts/tracks/algorithm_codesign/run_a1_new_target_n1.py)、[人工tests](../../../tests/tracks/algorithm_codesign/test_a1_new_target_n1.py)。
coreの追加はDFTailExtractionへの既存canonical hash入力文字列の保持だけ。係数計算・既存hash式・samplerは変更せず、pre-threshold identity/exact-zeroも同じ計算から保存する。
人工40件PASS：[TESTS](TESTS.json)。既存29件を再利用し、数値/hash AST不変、完全payload、独立な小law全列挙によるν/joint/層被覆、candidate意味不一致、禁止call、fresh STOPを確認した。これは科学N1の成功記録ではない。

## 固定と一回の取得

[PROFILE_LOCK](PROFILE_LOCK.json)にSOURCE 533ファイル、installed distribution metadata132件、native library332件、Python実体・NumPy build・プロセスthread/cache設定を固定した。
Python3.12.3、NumPy1.26.4、Qiskit1.3.0。共有venv/packageを変更せず、MPL/NUMBA/XDG cacheのみ自分専用`/tmp/a1-n1-cache-20261106/`を使う。CPU/RAM/threadの人工上限は新設していない。
このSOURCEを公開・再取得照合後、cleanな専用worktreeから以下を**一度だけ**実行する。

```bash
MPLCONFIGDIR=/tmp/a1-n1-cache-20261106/mpl \
NUMBA_CACHE_DIR=/tmp/a1-n1-cache-20261106/numba \
XDG_CACHE_HOME=/tmp/a1-n1-cache-20261106/xdg \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src <existing-opt2-python> -B \
scripts/tracks/algorithm_codesign/run_a1_new_target_n1.py \
  --profile-lock docs/research/o2_a1_new_target_n1_20261106/PROFILE_LOCK.json \
  --output-dir artifacts/o2_a1_new_target/2026-11-06/n1
```

load→partition→preparationを各API一回、同じ準備からconfig/CDF/Q/support/ν/joints/2層/5層を取得し、新policyのsupport basisを登録・対応確認する。
core内部のrank検査・policy用軌道基底再構成は既存処理のままであり、「全内部eighが一回」という意味ではない。many-body state solve、全event列挙、full interrogation build/compileを含めない。
全partition/preparation/tailのcanonical hash入力をそのまま保存して、新identityの自己整合を検査する。旧opt2 hashの一致は要求しない。
a=`df-fragment-3:Z0Z1`、b=`df-fragment-6:Z0`のfragment/support/signと新catalogの対応、positive Q、classical joint恒等式、層被覆を検査する。別labelへ変更しない。

policyは既存`register_support_restricted_bases`を再利用し、全componentに必要なkey、source-basis操作の一致、保持column残差≤既存許容値1e-10を検査する。
その残差はoperator/数値biasの厳密上界ではない。新policy宣言・新preparation・新basis定義からfingerprintを作り、旧training fingerprintは流用しない。run閾値ruleは固定し、実pathでのfull回路同値や価格はN1で検査しない。

RNG、state solver、full step/repeated/Hadamard/RTE sequence builder、compilerの禁止entry pointはN1 contextで例外にする。basis操作構成はN1に含む。
出力はfresh directory必須。不整合では取得済みpayload/STOP/呼出しcountを保存し、retryやSOURCE/条件変更で救済しない。
結果の公開とremote SHA/本文/blob/tree照合後、**mandatory STOP**。N2/N3へ進まない。
