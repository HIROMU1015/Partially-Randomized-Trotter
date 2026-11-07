# O2候補の比較と除外理由

2026-11-08。性能順位を作る数値評価は実施せず、[主文書](O2_NEW_ALGORITHM_CANDIDATES.md)の操作・数学的成立条件・未解決の接口で選別した。

## 残した2案

| 観点 | 案M：共通basisのlabel平均を測定へ移す | 案D：局所回転差の別測定 |
|---|---|---|
| 変更する操作 | label別controlled gateを外し、system zと平均observableのancilla測定から重み付き出力 | fine targetを有料base＋有料additive correctionへ分割 |
| 新しい情報利用 | 通常捨てるsystem readout。振幅/Bornは古典側へ渡さない | 既知の角度差とunitary prefix/suffix。coarseの平均も未知で有料 |
| sampling | 元Qの(η,g)周辺分布を保持。群内labelは演算・samplingから解析的に除く | 元path Q＋新しい有限確率の補正slot取得。重みはactual確率と一致させる |
| 統計的機構 | 同じ条件付き平均でM2≤1。群内固有値が混ざる場合に厳密改善し得る | 補正のM2=Σ|γ|²/p。理想比例samplingならΓ²。ただしbase負担あり |
| 回路削減の可能性 | controlled product/rotationを省ける。枝分けやreadoutにより逆転もあり得る | 有料baseを安くし、小さい補正へ少数shotを回せる場合。full補正回路は安いとは限らない |
| SCV/A2との差 | 新たな量子readoutを使用、学習不要、heraldなし | 非unitary差分stateを準備しない。既知小係数を測定値へ掛ける |
| Returnとの関係 | 語のClosed/CTSではなくoperator平均の測定 | return lawを変えず、full unitary差を加法的に分解 |
| 先行研究との重複 | direct/sequential measurement・commuting measurementの応用 | telescoping/multifidelity・角度補間・C-S局所residualと重複大 |
| 必要なアクセス | A†/Bの条件付き枝、既知V、system全bit readout、rotation版のfeed-forward | system V_jの制御付き実行、phase readout、既知三角係数と正確なslot確率 |
| 主な相殺要因 | full-wrapper境界相殺の損失、readout/feed-forward/RZ、r(z)演算、新誤差認証 | Γ累積、midpoint合成、baseのnoise、finite-bit重み増、二段階の全費用 |
| 既存codeの再利用 | RTE/selected basis、conditional Q、Hadamard符号、facade/価格/記録 | system gate IR、Hadamard/phase、finite-bit/会計。C-Sは重複確認のみ |
| 未実装の最小差分 | eligibility、群平均decoder、枝・readout adapter、新誤差接口 | system telescoping列、slot sampler/decoder、base＋補正会計 |
| 最初の判別材料 | 実state不要の共通factorization証明と小規模の解析的instrument例 | 同一生成子の局所差分が全wrapperを含めて成り立つ証明とcoarseの入力元 |
| 分類 | KNOWN_METHOD_APPLICATION + ACTIONABLE_CANDIDATE（技術段階のみ） | KNOWN_METHOD_APPLICATION + PROMISING_BUT_INCOMPLETE |
| 推奨 | 最初の小さい技術確認へ提案 | 保留。Mの条件不成立を理由に自動実装しない |

ACTIONABLEは有限の技術課題へ分けられる意味であり、科学計算の実行許可・実H4適合・総費用利益ではない。どちらも学術的新規性は未認定。

## 7探索領域をどう扱ったか

| 探索領域 | 検討した具体的操作 | 今回の扱いと理由 |
|---|---|---|
| A：sampling law | metadata/推定価格でproposalを更新 | 独立候補には残さない。安いnormalizerと正support/rangeを得る新構造がなく、S2/Parent/J1へ還元。案M/Dのsampling変更は各測定機構に必要な範囲だけ |
| B：測定再設計 | common basisでsystemを読み、zに応じて平均observableを測る | 案M。単なるX/Y同時測定や未知真phaseへの軸整列とは異なる |
| C：推定器 | metadata CV追加、複数shot平均、局所差分を別測定 | 前二者はSCV/既知配分に還元。案Dのみ明示小係数のアクセスがある条件で残す |
| D：return/context | 可換labelのoperator平均、basis逆相殺 | operator平均は案Mへ統合。逆相殺だけなら既存Closed/CTS/D2へ還元 |
| E：部分解析 | rotationのI/Pauli成分を測定observableへ移す | 案M/Dへ具体化。productがIという理由だけで全wrapper信号を既知にしない |
| F：適応取得 | pilotで軸/配分を更新しconfidenceで停止 | KNOWN_METHOD_APPLICATION。両座標復元、pilot費、valid stoppingを超える新しい操作的利点を特定できず、独立候補にしない |
| G：sampling-return結合 | stack/小summaryからfull価格重みを生成 | NO_SUPPORTED_MECHANISM（今回調べた操作に限る）。J1が欠くconditional部分和を新しく提供しておらず、調査を繰り返さない |

追加の除外：既知π/2位相をaxis/bitへ移すだけの案は既知phase-frame/測定最適化としてbaselineに含める。A2の失敗flagも活用するだけでscale/noiseが消えるとはいえず、旧LCUの再命名は候補にしない。これらの除外は一般的な不可能性の主張ではない。

## 強いbaselineと公正な比較

案Mは元のcanonical PRだけでなく、同じQ・正supportのB1-OLや利用可能なgroup層別、既知の単独label direct measurement、共通basis grouping、scalar/phase処理、選択済みbasis/境界最適化と比べる。元方式も同じ(η,g)配分を使えるようにし、群平均による差と配分差を混ぜない。単独label direct測定にも重み付きM2の利益があり得るので、「旧±1よりM2が小さい」だけで最強baselineへの改善を主張しない。

案Dは元fineの強い層別測定、単純coherent fine/coarse差測定、既知のadditive/telescopingと角度補間、適用可能ならC-Sのper-shot residualを比較対象とする。C-Sの固定科学実験を自動再開する意味ではない。channel補間の結果をcoherent平均の保証へ流用しない。[TE-PAIの一次資料](https://arxiv.org/html/2410.16850v2)

共通の比較対象は同じZ、座標誤差・失敗率、ideal/native精度、状態準備の除外規則である。新observableのrange/重み、全attempt、学習/calibration、compiled full-wrapper、古典生成/演算/cache、追加readout/latencyを費用へ含める。期待費用と一回campaignの実現費用を分ける。今回、N・T/RZ・期待費用・時間短縮率の新数値は算出していない。

## 不足している決定材料

M：実際のselected policyに共通factorizationがあるか、その群が非自明か、新readoutが旧最適化より安いか、新しい数値誤差を保証できるか。D：安いcoarseを既知入力として与えられるか、補正係数の累積とfinite sampling負担が小さいか、別shot化がC-S/既知法との差を持つか。これらはUNKNOWNであり、技術資料の豊富さを科学的有望性の順位へ変換しない。
