# SP-1後：研究Bの方針再評価をGPTへ戻す

2026-10-06 JST。**one-shot complete、mandatory STOP、retry0、次stage未認可。**
Codexは固定契約の実行・保存値照合を完了した。研究方向の採否・RQ/新規性/論文着地点・
追加検証の必要性/範囲はGPT側へ戻す。この文書は新しい科学実行authorizationではない。

## 固定資料

1. [今回の結果照合・scope](sp1_one_shot_result_validation_20261006.md)。
2. [raw result / consumed marker / 保存値audit / manifest](../../../artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/)。
3. [固定source Sの結果前契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)、
   [Rのsource review依頼](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0502f22039dc0b5de227ce4b7b5a229c55f1e74/docs/tracks/algorithm_codesign/sp1_final_source_review_request_0d01ed9.md)。
4. [Aの明示指示・別authorization](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9630e06122172af238ac5219bec6dfc8b01ca837/docs/tracks/algorithm_codesign/sp1_execution_authorization_receipt.md)。
5. [SP-0.5 primitive確認](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e57c1fdd28589422e9c973e34e53f6c725b921e9/docs/tracks/algorithm_codesign/sp05_one_shot_result_validation_20261006.md)。
6. 過去closure：[B-F](bf_current_hypothesis_closure_20261005.md)、
   [BM-0.5同値性](bm05_review_packet_20261005.md)。旧STOPは解除しない。

source S=`0d01ed9a332ebc5b66ed08acf56214a9b9c0236d`、
実行A=`9630e06122172af238ac5219bec6dfc8b01ca837`、親[S]。
準備P=`c2a49f72eaffba4514743689eb8ccee02ca060a8`は保存し、実行HEADに使わなかった。
raw result SHA256 `9a874f027b2927dcfde44cfb3b3e08a577cdf4b71aa16760a9d58c46a9464c48`、
marker SHA256 `fd1064efdacde90953c73eb0092e6f3bd22ab283457bfe5cf63e862986f101f8`。
branchは`track-b-sp1-one-shot-execution-20261006`。公開result/review commitは引継ぎ時にfull SHAを提示する。
result/review HEADは実行HEADではない。

## GPT判断のための観測

2-qubit synthetic A/B/C、n={8,16,32,64}、NONE/D/R/DR、同じπ/4 catalogue、
complex ε=.05、familywise α=.05、5% materiality、common Bernstein sufficient shotsを維持した。
primaryはadditive synthesized-primitive期待T費用で、actual compiled costではない。

- 48 mask rows／96 axes完了。適格42、モデルshot-cap6。technical failureなし。
- A/BのD（DR duplicate）はn=8/16でgain、n=32でloss、n=64でshot cap。
- CのDRはn=8でNONE比約0.01856、n=16で約0.10261、n=32で約3.16062、n=64でshot cap。
- CのD-only/R-onlyのmaterial gainは0。n=8は両方5%分離なし、n≥16はlossまたはcap。
- raw gain10のうちduplicate4を除いて独立gain6。Cのgain2はいずれもDR。
- Cで「単独placementがgain、DRが不利」のwitnessは登録setにない。

前reviewの分岐へ無理に一つのlabelを付けず、この観測全体を評価してください。
短いwrapperでcheap probabilistic synthesisの利益が残り、登録nの増加でlossへ反転した。
selective-only advantageを確認したとは言わない。
CのD/Rはgate数・angle・sign・outer randomnessがconfoundし、一般法則にはならない。

## 依頼する研究判断

1. 既知PAI/Sparse PS/TE-PAIのaccumulation/crossover確認を超えるmethod deltaが、
   この観測からまだ候補として残るか。今回のcrossover自体を新規性へ数えない。
2. Cのselective-only gainがないこと、short-DR gainとlong-loss/capを踏まえ、
   synthesis-placement主線を閉じるか、mechanism/application noteへ縮小するか、
   独立の狭い研究問いを立てる価値があるか。
3. actual finite-RTE外側weight/DF populationへ接続する追加検証に情報価値があるなら、
   必要性・限定scope・強い既知法baseline・公平なmetric・結果前STOP条件を先に定義する。
4. compiled費用・oracle-free設計・独立validationを本当に必要とするか。
   現在の結果だけでそこへ自動GOしない。
5. 研究BのRQ・新規性・論文の最小着地点を、positive/negative双方を含めて再評価する。

独立validation/最良compiler/新sampler/finite-time certificate/DF-native improvementの証拠はない。
M1/M2はTrack Aの証拠で、M2 H4 1.30 Åは開封済み。Bのnew held-outには使わない。
今回もdevelopment/mechanism local evidenceで、immutable CIや外部再現ではない。

## 現在の実行境界

one-shot markerはconsumed、retry0。threshold/caps/gridを変更しない。
source・contract・authorization・元result/marker・SP-0.5 artifactを変更しない。
新target/catalogue/precision/n、分子NPZ/DF/Hamiltonian、trajectory、GPU、actual wrapper compilation、
旧16-cell/BF/BM再開は未認可。Codexは追加scienceを実行しない。
GPTが必要性とscopeを決め、別contract/source/authorization/明示指示を閉じるまでmandatory STOP。
