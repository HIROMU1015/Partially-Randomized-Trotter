# PR-Pauli LD2 B_PRICE 2配分：実行前の固定と照合

2026-11-15 JST、B2（Claude Code、主agentのみ）。ユーザーが固定計画 `15c08ff29` のうち FPN-B と OWN-B を採択し、B_PRICE による具体列の生成、モデル上の費用計算、各配分最大1回・合計最大2回の新規Q3を許可した範囲。P7N-B・中間予算・OWN-A、新規path・compile・再fit・結果依存の配分変更・retryは含めない。

## 実行前に固定した2配分（Q3の結果を見る前）

価格は B2 `a7eceecb` の26軸モデル参照価格 C_ref（E[C]ではない）。重み VP/C_ref、各軸 1≤N≤VP、Bの breakpoint 規則（`06f4210e` の`generate`を正の有理数価格に広げただけ）。X/Yは別々に生成し、結果として同じ列になった。

| 案 | 予算（RZ） | N_X = N_Y（m0〜m12） | モデル費用 T(N)（RZ、表示） | 未使用予算 |
|---|---:|---|---:|---:|
| FPN-B | 615,304,502 | [48,45,42,39,36,33,29,26,23,19,7,1,1] | 600,045,823.85 | 15,258,678.15 |
| OWN-B | 3/4×T_ref（丸め前、2,533,877,262.39…） | [48,45,42,39,36,33,29,26,23,19,16,13,5] | 2,442,845,584.51 | 91,031,677.89 |

exact値・各軸のN×C_ref・元価格行への参照は [INPUTS.json](INPUTS.json) にある。T(N)は m0〜m9 と m10〜m12 に分けて保存した。

## 重複照合（共通運用指示§10.5）

- LD2の既存Q3はVP（`50be68b8`、適合）とV0（`caf1417e`）だけ。2配分はどちらとも一致しない（各成分がVP以下で、m10〜m12がVPより小さい）。V0は採択閾値の判定欄がない診断記録だが、そもそも別の列なので再利用の対象にならない。
- 2案は互いに異なる列なので、Q3は2回とも新規評価になる。
- 後続の保存結果：計画公開（`15c08ff29`）以降、remoteにLD2の新しいQ3結果は無い。BのPF6の18点（`b4296b110`）は別候補。
- 実行中のjob：作成時点で自ユーザーのQ3／Pauli関連processは無い。

## 再利用する固定条件

- 信号・canonical・bridge・target・law・数値追跡条件・採択済みの評価条件：`f54b41a2e`の`docs/research/pauli_q3_three_quarter_20261115/INPUTS.json`のLD2 caseをそのまま使い、allocationだけを差し替えた（statistical contractのidとmodel loss contractのscope文字列だけ今回の名前にした）。
- Q3経路：`scripts/run_pauli_q3_three_quarter.py`（`f54b41a2e`と同一bytes）の`worker`・`assess`。その中で `run_pauli_total_q3_accelerated.py`（Observed accelerator、8 worker・batch 8）と固定のPauli route（`A.accept_saved`→`T.create`→`bind_signal`→`T.evaluate`、closure gate・lemma gate・total DP gate）を通る。
- 新しく足したのは、workerが自分で API return・RESULT・資源記録を保存してから親へ返す部分（PF6やPR-DF/Full-Hのcallerと同じ方式）だけである。

## count-onlyと閉包

- 各列のcount-only上限は既存範囲内：FPN-B は round outcomes 2401・verification 18,097,040・transitions 75,951,909、OWN-B は 2401・18,250,468・79,630,117（範囲は 4225・55,043,173・141,559,036）。
- binary64閉包（ρ=1）は各callの中で検査する（既存のclosure gate）。VPの閉包結果は流用しない。

## 人工fixture

新7件＋再利用経路の既存3件＝10件PASS（[TESTS.txt](TESTS.txt)）。B_PRICE生成がBの関数と整数価格で一致すること、有理数価格での上下限・予算・最大性、count-only上限、実際のPauli routeの上での worker保存（API return・RESULT・資源・PERSISTED）、monitor例外時に未開始のcallを止めて開始済みworkerの記録を残すこと、通常経路を確認した。人工の信号だけを使い、実H4のriskは起動していない。

## 実行条件

- 開始前の観測：64 CPU、MemAvailable 約512 GB、load average 約0.3。2 jobを並列、各8 DP worker。CPU・RAMの恒常的な上限ではない。
- 判定：有効上界≤τで適合、下界>τでモデル不適合、跨げばINCONCLUSIVE、gate不成立はUNKNOWN。
- 2件の終了、または重大な異常でmandatory STOP。retry・列の変更・追加callはしない。
