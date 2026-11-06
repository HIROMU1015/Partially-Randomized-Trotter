# A1 Q1/Q2-L：固定された非sampling実行

2026-11-06 JST。ユーザーの「実行してよい」は直前に提示した最小scope、Q1とQ2-Lに適用する。
[固定50f4b43の実行仕様](../o2_a1_0_connections_20261106/A1_0_SCIENCE_RUN_SCOPE.md)を継承し、Q2-T/Q3/Q4/Q5へ進まない。
新しい承認証明・grantを要求せず、以下は科学SOURCE・条件・再現性の記録である。

対象：保存linear H4、1.00 Å、STO-3G、legacy DF rank12、4電子/8qubit、L_D3、m0/q1/δ=t=.02/r1/K2、threshold0、X/Y共通のlaw。
元opt2の固定NPZ・H/partition/preparation/tail hash、a=`df-fragment-3:Z0Z1`/b=`df-fragment-6:Z0`を変えない。
SOURCEは本書・[library](../../../src/trottertracks/algorithm_codesign/a1_law_validation.py)・[runner](../../../scripts/tracks/algorithm_codesign/run_a1_q1_q2l.py)・[人工test](../../../tests/tracks/algorithm_codesign/test_a1_law_validation.py)を含む公開commitに固定してから一回実行する。

## 実行とSTOP

既存opt2 venv（Python3.12.3/NumPy1.26.4/Qiskit1.3.0）を変更せず用いる。
既存 `shared_cpu.module` で同じライブラリをprivate CPU namespaceへ読み、eager package初期化/GPU preloadを回避する。
NPZ loader、LD3 split、threshold0 preparation、RTE config/distribution、A1 CDF/control接続を再利用する。
Q1に必要な小さい軌道基底の再構成を含むが、physical-sector groundの対角化やfull interrogation生成は行わない。
SOURCEと実際に読み込んだSOURCEのhash、完全catalog、CDF、格子law、既知momentを保存する。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src <existing-opt2-python> -B \
  scripts/tracks/algorithm_codesign/run_a1_q1_q2l.py \
  --output-dir artifacts/o2_a1_q1_q2l/2026-11-06/v1
```

出力先は新規専用directory。既存directoryを拒否し、hash/support/profile不一致なら途中payload/STOPを保存して終了する。別入力・別label・再sampling・科学retryで救済しない。
RNG、state/ground solve、Hadamard simulation、full回路build/compile、variance/N/費用評価、threshold差のQ2-Tは0 callのscope。
科学worker/GPUを起動せず、CPU/RAM/workerに恣意的な低い上限を課さない。共有環境・他jobを操作しない。

## Q2-Lの上界が対象とするもの

記録binary64をexact rationalとして、λ_f、τ_f、B_f、係数c_iを固定する。
SOURCEのalgebraic targetを `M_fin=P3(−i τ_f Σ c_i P_i/λ_f)/B_f` とする。
P_iは同じ記録basisにおけるsign吸収済みの理想Hermitian involutionである。

S=Σ|c_i|、s=τ_f S/λ_f、reference label law p_i=|c_i|/S。
reference order weightsは `w0=√(1+s²)`、`w2=(s²/2)√(1+(s/3)²)`、B*=w0+w2、p_n=w_n/B*。
このreference unitary mixtureは `P3(−i τ_f Σ c_i P_i/λ_f)/B*` を与える。
λ_f=S、B_f=B*、binary64 probabilityの和=1という同一視をしない。

既存 `root_interval`（160bit dyadic endpoints）でw/Bを包絡し、`interval_angle`（mpmath interval 80dps）でatanを包絡する。
Parentのproposal生成・native synthesis・providerは呼ばず、既知のinterval処理だけを再利用する。

実qのorder/label L1距離をd_order/d_label、referenceのlabel数上界をE_p[n+1]_Uとすると
`b_law ≤ d_order + E_p[n+1]_U d_label`。
さらに `Σ p_n |φ_float,n−atan(s/(n+1))|` の上界と `|B*/B_f−1|` の上界を加える。
bounded unitary応答、product-lawのtelescoping、共通外側unitaryのnorm1による標準評価で、未知state/path期待値を要しない。
X/Yとも各coordinateの偏差を同じ和で抑える。exact rational値が正本で、表示floatは外向き上側previewにする。

これは**理想involution・同じ外側unitary・selected policy等価性・IID random-wordモデルに条件付けたsampling/角度/normalizationの上界**。
DF/basisからselected回路への独立した数値誤差やnative/backend効果を厳密保証したとはしない。
未評価項を0にせず、完全な数値残余幅の保証を発行しない。旧εとの差は条件付きarithmetic residualとしてだけ記録する。
この未閉鎖を隠して測定へ進むことはなく、Q1/Q2-Lの保存・公開後mandatory STOPとする。
