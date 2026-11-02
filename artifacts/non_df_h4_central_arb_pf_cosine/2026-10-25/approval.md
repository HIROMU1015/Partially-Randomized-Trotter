# H4 Pauli central PF cosine Arb包含評価：利用者の明示承認

受領日：2026-10-25（Asia/Tokyo）。基点branch
`codex/non-df-h4-central-arb-preparation-20261024`、
commit `5c69e1339955c77273b356df30e94ec3c57458b1`。

> PF cosine 1件のArb包含評価を正式に承認します。

固定H4（1.00 Å／STO-3G）、Pauli-JW184項、S2／T=0.8のcentral-control PF。
保存compiled QPY・raw state/ray・保存C-target区間を再利用し、元phase/wire/binary64/norm規約を維持する。
既存bridge・C区間演算・人工テストを優先再利用し、production接続を実装・人工検証する。

- 精度64→96→128 bit、各精度の状態作用最大1回、初回科学計算最大3状態作用。
- 保存ρ予算約2.6490038338323596e-6との関係が確定した時点で終了。
- straddleまたはnonfiniteで判定不能の場合だけ次精度へ進む。128bitでも判定不能ならINCONCLUSIVEでSTOP。
- 幅1e-8は参考目標。判定後の不要な高精度化は禁止。
- 実行前に新科学SOURCE/runtime/入力/契約を固定し、新one-shot grantを発行する。
- 意図しない技術中断は原因修正後の限定再実行を許可。ただし失敗call・旧SOURCE/grant/STOPを保全し、
  修正後は新SOURCE・未使用grantを使う。成功済み計算の不要な再実行は禁止。
- 新compile/sampling/Hamiltonian/state生成、PR拡張、R2、N/G再解析、正式certificate受理は禁止。
- 局所残差の数値包含と正式certificate受理を区別する。
- CPU/RAM/worker数の恣意的上限なし、共有環境・他ユーザーへの影響禁止、旧証拠保全。
- 信号・局所ρ区間/予算比較/bit/幅/call/SOURCE/grant/native終了/監査/報告書を保存し、commit/pushとremote SHA確認。
- 結果保存・公開処理後mandatory STOP。
